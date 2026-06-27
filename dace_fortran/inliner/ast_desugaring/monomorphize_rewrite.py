# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Static-vtable rewrite (the M2 engine) for the monomorphisation feature.

Consumes the :class:`MonomorphizationPlan` produced by
:mod:`dace_fortran.inliner.ast_desugaring.monomorphize` and rewrites runtime
type-bound-procedure dispatch into a static ``if`` ladder over the closed set of
concrete arms -- the *emit-all-always* model (every arm emitted; no collapse), so
the program lowers with only direct ``fir.call``s and never trips the bridge's
``RejectPolymorphism``.

Two slot shapes are handled, both expanded into an integer type-tag plus one
concrete allocatable per arm:

  * a polymorphic ``CLASS(base)`` *local variable* (:func:`monomorphize_local_dispatch`),
    where ``ALLOCATE(concrete :: v)`` sets the tag and ``CALL v%binding(args)`` becomes
    the per-arm ladder; and
  * a polymorphic ``CLASS(base)`` *component* of a container type
    (:func:`monomorphize_component_dispatch`, e.g. ICON's ``t_ocean_solve%act``), where
    the container type def is expanded and *every* statement that references the slot --
    a dispatch, a whole-statement data-member assignment, or a slot member buried in a
    sub-expression -- is re-emitted once per arm with ``%slot`` retargeted to
    ``%slot__arm`` (a static concrete-``TYPE`` bind).

(The shared-interposer-clone and sibling-axis retype cases build on these primitives
and follow.)
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f03
from fparser.api import get_reader
from fparser.two.utils import walk

from dace_fortran.inliner import ast_utils
from dace_fortran.inliner.ast_desugaring.monomorphize import (analyze, MonomorphizationPlan, parse_program,
                                                              read_type_info, UnsupportedProgram)
from dace_fortran.inliner.ast_desugaring.utils import (append_children, find_name_of_node, prepend_children,
                                                       remove_self, replace_node)

logger = logging.getLogger(__name__)

SCOPES = (f03.Subroutine_Subprogram, f03.Function_Subprogram)


def _tag_var(var: str) -> str:
    return f"{var}__tag"


def _arm_slot(var: str, type_name: str) -> str:
    return f"{var}__{type_name}"


def _find_arm(plan: MonomorphizationPlan, type_name: str) -> Tuple[int, str]:
    """1-based tag value + canonical arm name for a concrete type (case-insensitive)."""
    for i, arm in enumerate(plan.arms, start=1):
        if arm.type_name.lower() == type_name.lower():
            return i, arm.type_name
    raise KeyError(f"type `{type_name}` is not a concrete arm of `{plan.abstract_base}`")


def _parse_decls(text: str) -> List[f03.Type_Declaration_Stmt]:
    prog = parse_program(f"subroutine zz_tmp()\n{text}\nend subroutine\n")
    spec = walk(prog, f03.Specification_Part)[0]
    return [c for c in spec.children if isinstance(c, f03.Type_Declaration_Stmt)]


def _parse_exec(text: str) -> List[f03.Base]:
    prog = parse_program(f"subroutine zz_tmp()\n{text}\nend subroutine\n")
    exe = walk(prog, f03.Execution_Part)
    return list(exe[0].children) if exe else []


def _parse_expr(text: str) -> f03.Base:
    """Parse one Fortran expression (via a throwaway assignment) and return its node."""
    _, _, rhs = _parse_exec(f"zz_x = {text}")[0].children
    return rhs


def _rewrite_allocated_queries(program: f03.Program, slot_names: Set[str]) -> None:
    """``ALLOCATED(<prefix>%<slot>)`` -> ``(<prefix>%<slot>__tag /= 0)``.

    After a component is laddered into a tag + per-arm concrete slots it has no
    allocation status of its own; the tag (set at the construction site, 0 before)
    *is* the "is it constructed?" status. These guards are the ``IF
    (ALLOCATED(this%act)) CALL finish(...)`` init checks -- an ``If_Stmt`` the slot
    ladder (Call/Assignment/Pointer-assignment only) never visits."""
    for fref in walk(program, (f03.Intrinsic_Function_Reference, f03.Function_Reference)):
        name, args = fref.children
        if str(name).upper() != "ALLOCATED" or args is None:
            continue
        arg = next(iter(args.children), None)
        if not isinstance(arg, f03.Data_Ref):
            continue
        prefix, tail = _ref_prefix_and_tail(arg)
        if tail.lower() not in slot_names:
            continue
        replace_node(fref, _parse_expr(f"{prefix}%{_tag_var(tail)} /= 0"))


def _expanded_decls(var: str, plan: MonomorphizationPlan, stack_slots: bool = False) -> List[f03.Type_Declaration_Stmt]:
    """``integer :: v__tag`` + one ``type(arm), allocatable :: v__arm`` per arm.

    With ``stack_slots`` the per-arm slot is a plain (non-allocatable) stack object:
    after monomorphisation the allocatable is no longer needed (the polymorphism is
    gone) and the dace-fortran bridge cannot lower an allocatable derived-type
    scalar, so stack slots are what make the rewritten kernel SDFG-lowerable."""
    attr = "" if stack_slots else ", allocatable"
    lines = [f"integer :: {_tag_var(var)}"]
    for arm in plan.arms:
        lines.append(f"type({arm.type_name}){attr} :: {_arm_slot(var, arm.type_name)}")
    return _parse_decls("\n".join(lines))


def _allocation_rewrite(var: str,
                        alloc_type: str,
                        plan: MonomorphizationPlan,
                        stack_slots: bool = False) -> List[f03.Base]:
    """``allocate(concrete :: v)`` -> ``v__tag = <tag>`` + ``allocate(v__<arm>)``.
    With ``stack_slots`` the slot is a stack object, so only the tag is set (no
    ``allocate``)."""
    tag, arm = _find_arm(plan, alloc_type)
    if stack_slots:
        return _parse_exec(f"{_tag_var(var)} = {tag}")
    return _parse_exec(f"{_tag_var(var)} = {tag}\nallocate({_arm_slot(var, arm)})")


def _dispatch_ladder(var: str, binding: str, argstr: str, plan: MonomorphizationPlan) -> List[f03.Base]:
    """``call v%binding(args)`` -> one ``if (v__tag == k) call <arm-proc>(v__arm, args)`` per arm."""
    lines = []
    for tag, arm in enumerate(plan.arms, start=1):
        proc = arm.bindings[binding]
        callee_args = _arm_slot(var, arm.type_name) + (f", {argstr}" if argstr else "")
        lines.append(f"{'if' if tag == 1 else 'else if'} ({_tag_var(var)} == {tag}) then")
        lines.append(f"  call {proc}({callee_args})")
    lines.append("end if")
    return _parse_exec("\n".join(lines))


def _class_locals(spec: f03.Specification_Part, base: str) -> List[Tuple[f03.Type_Declaration_Stmt, List[str]]]:
    """``(decl, [names])`` for each ``CLASS(base), ... :: ...`` declaration in ``spec``."""
    out = []
    for decl in walk(spec, f03.Type_Declaration_Stmt):
        type_spec = decl.children[0]
        if (isinstance(type_spec, f03.Declaration_Type_Spec) and type_spec.children[0] == 'CLASS'
                and str(type_spec.children[1]).lower() == base.lower()):
            out.append((decl, [str(e.children[0]) for e in walk(decl, f03.Entity_Decl)]))
    return out


def _locally_constructed(scope: f03.Base, plan: MonomorphizationPlan) -> Set[str]:
    """Names of entities constructed *in this scope* via ``ALLOCATE(arm :: v)`` with a
    plain-name target -- the source construction sites that seed a type tag.  A
    component target (``ALLOCATE(arm :: this%act)``, a ``Data_Ref``) is excluded: that
    is the component primitive's job, not local-variable dispatch."""
    arms = {a.type_name.lower() for a in plan.arms}
    out: Set[str] = set()
    for alloc in walk(scope, f03.Allocate_Stmt):
        alloc_type, alloc_list, _ = alloc.children
        if alloc_type is None or str(alloc_type).lower() not in arms:
            continue
        out |= {str(obj).lower() for obj in alloc_list.children if isinstance(obj, f03.Name)}
    return out


def monomorphize_local_dispatch(program: f03.Program, plan: MonomorphizationPlan, stack_slots: bool = False) -> int:
    """Rewrite ``CLASS(plan.abstract_base)`` *local-variable* dispatch in ``program``
    into the static emit-all-always ladder, in place.  Returns the number of
    polymorphic locals rewritten (so callers can detect a no-op).

    A target is a ``CLASS(base)`` entity *constructed in its scope* via
    ``ALLOCATE(concrete :: v)`` -- the tag construction site.  This deliberately
    excludes a ``CLASS(base)`` dummy argument (e.g. a shared interposer's
    passed-object), which carries no local construction and is the clone/retype
    primitives' responsibility, not local-variable dispatch's."""
    base = plan.abstract_base
    deferred = {d.lower() for d in plan.deferred}
    rewritten = 0

    for scope in walk(program, SCOPES):
        spec = ast_utils.atmost_one(ast_utils.children_of_type(scope, f03.Specification_Part))
        if spec is None:
            continue

        constructed = _locally_constructed(scope, plan)
        if not constructed:
            continue

        targets = [(decl, [n for n in ns if n.lower() in constructed]) for decl, ns in _class_locals(spec, base)]
        targets = [(decl, ns) for decl, ns in targets if ns]
        names = [n for _, ns in targets for n in ns]
        if not names:
            continue

        for decl, decl_names in targets:
            expanded = [d for v in decl_names for d in _expanded_decls(v, plan, stack_slots)]
            replace_node(decl, expanded)

        for var in names:
            for alloc in walk(scope, f03.Allocate_Stmt):
                alloc_type, alloc_list, _ = alloc.children
                if alloc_type is not None and var.lower() in {str(n).lower() for n in walk(alloc_list, f03.Name)}:
                    replace_node(alloc, _allocation_rewrite(var, str(alloc_type), plan, stack_slots))

            for call in walk(scope, f03.Call_Stmt):
                designator = call.children[0]
                if not isinstance(designator, f03.Procedure_Designator):
                    continue
                obj, _, binding = designator.children
                if str(obj).lower() != var.lower() or str(binding).lower() not in deferred:
                    continue
                args = call.children[1]
                argstr = ', '.join(str(a) for a in args.children) if args is not None else ''
                replace_node(call, _dispatch_ladder(var, str(binding).lower(), argstr, plan))
            rewritten += 1

    return rewritten


def _parse_component_decls(text: str) -> List[f03.Data_Component_Def_Stmt]:
    prog = parse_program(f"module zz_tmp\n  type zz_t\n{text}\n  end type\nend module\n")
    return list(walk(prog, f03.Data_Component_Def_Stmt))


def _expanded_component_decls(slot: str,
                              plan: MonomorphizationPlan,
                              stack_slots: bool = False) -> List[f03.Data_Component_Def_Stmt]:
    """``integer :: act__tag`` + one ``type(arm), allocatable :: act__arm`` component per arm.
    With ``stack_slots`` the per-arm component is a plain (non-allocatable) member -- the
    SDFG-lowerable form (see :func:`_expanded_decls`)."""
    attr = "" if stack_slots else ", allocatable"
    # The tag defaults to 0 ("not constructed") so an `ALLOCATED(this%act)` guard
    # rewritten to `this%act__tag /= 0` reads correctly before the construction
    # site sets it (a derived-type component is otherwise undefined until set).
    lines = [f"integer :: {_tag_var(slot)} = 0"]
    for arm in plan.arms:
        lines.append(f"type({arm.type_name}){attr} :: {_arm_slot(slot, arm.type_name)}")
    return _parse_component_decls("\n".join(lines))


def _component_slots(program: f03.Program, base: str) -> List[Tuple[f03.Data_Component_Def_Stmt, List[str]]]:
    """``(component_stmt, [names])`` for each ``CLASS(base), ... :: ...`` *component*."""
    out = []
    for comp in walk(program, f03.Data_Component_Def_Stmt):
        type_spec = comp.children[0]
        if (isinstance(type_spec, f03.Declaration_Type_Spec) and type_spec.children[0] == 'CLASS'
                and str(type_spec.children[1]).lower() == base.lower()):
            out.append((comp, [str(cd.children[0]) for cd in walk(comp, f03.Component_Decl)]))
    return out


def _ref_prefix_and_tail(ref: f03.Data_Ref) -> Tuple[str, str]:
    """Split a ``Data_Ref`` into (``a%b`` prefix, last selector) -- e.g. ``this%act`` -> (``this``, ``act``).
    ``Data_Ref.children`` is the flat tuple of path parts ``(Name('this'), Name('act'))``."""
    parts = ref.children
    return '%'.join(str(p) for p in parts[:-1]), str(parts[-1])


#: statement kinds that can reference a slot (dispatch, assignment, pointer-assign).
SLOT_STMT_TYPES = (f03.Call_Stmt, f03.Assignment_Stmt, f03.Pointer_Assignment_Stmt)


def _slot_statement_ladder(stmt: f03.Base, slot_names: Set[str],
                           plan: MonomorphizationPlan) -> Optional[List[f03.Base]]:
    """If ``stmt`` references a slot ``<prefix>%<slot>`` (anywhere, even buried in a
    sub-expression), return a tag ladder that re-emits ``stmt`` once per arm with
    every ``%slot`` retargeted to ``%slot__arm``; ``None`` if it touches no slot.
    Because each arm slot is a concrete ``TYPE``, every retargeted reference -- a
    type-bound dispatch or a plain data-member access -- becomes a static bind."""
    prefix = slot = None
    for ref in walk(stmt, f03.Data_Ref):
        parts = ref.children
        for i, part in enumerate(parts):
            if i > 0 and str(part).lower() in slot_names:
                prefix, slot = '%'.join(str(p) for p in parts[:i]), str(part)
                break
        if slot is not None:
            break
    if slot is None:
        return None

    text = str(stmt)
    lines = []
    for tag, arm in enumerate(plan.arms, start=1):
        retargeted = re.sub(r'(%\s*)' + re.escape(slot) + r'\b',
                            lambda m, a=arm: m.group(1) + _arm_slot(slot, a.type_name),
                            text)
        lines.append(f"{'if' if tag == 1 else 'else if'} ({prefix}%{_tag_var(slot)} == {tag}) then")
        lines.append(f"  {retargeted}")
    lines.append("end if")
    return _parse_exec("\n".join(lines))


def monomorphize_component_dispatch(program: f03.Program, plan: MonomorphizationPlan, stack_slots: bool = False) -> int:
    """Rewrite dispatch on a ``CLASS(plan.abstract_base)`` *component* (e.g. ICON's
    ``t_ocean_solve%act``): expand the container type's component into a tag + one
    concrete allocatable per arm, then route ``ALLOCATE(concrete :: obj%slot)`` and
    ``CALL obj%slot%binding(args)`` through the emit-all static ladder, in place.
    Returns the number of polymorphic components expanded."""
    base = plan.abstract_base
    slots = _component_slots(program, base)
    if not slots:
        return 0
    slot_names = {n.lower() for _, names in slots for n in names}

    # 1. expand the container component into a tag + one concrete slot per arm.
    for comp, names in slots:
        replace_node(comp, [d for n in names for d in _expanded_component_decls(n, plan, stack_slots)])

    # 2. factory ALLOCATE(concrete :: prefix%slot) -> set the tag + allocate the
    #    matching concrete slot.
    for alloc in walk(program, f03.Allocate_Stmt):
        alloc_type, alloc_list, _ = alloc.children
        if alloc_type is None:
            continue
        for obj in alloc_list.children:
            if not isinstance(obj, f03.Data_Ref):
                continue
            prefix, tail = _ref_prefix_and_tail(obj)
            if tail.lower() not in slot_names:
                continue
            tag, arm = _find_arm(plan, str(alloc_type))
            set_tag = f"{prefix}%{_tag_var(tail)} = {tag}"
            rewrite = set_tag if stack_slots else f"{set_tag}\nallocate({prefix}%{_arm_slot(tail, arm)})"
            replace_node(alloc, _parse_exec(rewrite))
            break

    # 2b. ALLOCATED(prefix%slot) init guards -> the tag check (the slot has no
    #     allocation status of its own once laddered).
    _rewrite_allocated_queries(program, slot_names)

    # 3. every other statement referencing the slot -> ladder over the tag with
    #    `%slot` retargeted to `%slot__arm`.  This subsumes dispatch (the call
    #    becomes a static concrete-TYPE bind), whole-statement data-member
    #    assignment (`this%act%res_loc_wp => ...`) and a slot member buried in a
    #    sub-expression (`call timer_start(this%act%lhs%timer)`), and -- unlike a
    #    deferred-only dispatch rewrite -- ICON's non-deferred shared `this%act%solve`.
    for stmt in walk(program, SLOT_STMT_TYPES):
        ladder = _slot_statement_ladder(stmt, slot_names, plan)
        if ladder is not None:
            replace_node(stmt, ladder)

    return len(slots)


def _base_nondeferred_bindings(program: f03.Program, base: str) -> dict:
    """``{binding_name: target_proc}`` for the base type's non-deferred (shared) TBPs.
    These are the interposers (``solve``/``construct``) every backend inherits and that
    dispatch internally on their ``CLASS(base)`` passed-object."""
    out = {}
    for dtdef in walk(program, f03.Derived_Type_Def):
        stmt = ast_utils.singular(ast_utils.children_of_type(dtdef, f03.Derived_Type_Stmt))
        if str(stmt.children[1]).lower() != base.lower():
            continue
        for binding in walk(dtdef, f03.Specific_Binding):
            _, _, _, bname, target = binding.children
            if target is not None:
                out[str(bname).lower()] = str(target)
    return out


def _arm_from_slot_tail(tail: str, plan: MonomorphizationPlan) -> Optional[str]:
    """Recover the arm type from an expanded slot reference, e.g. ``act__t_gmres`` -> ``t_gmres``."""
    for arm in plan.arms:
        if tail.lower().endswith('__' + arm.type_name.lower()):
            return arm.type_name
    return None


def _find_subprogram(program: f03.Program, name: str) -> Optional[f03.Base]:
    for sub in walk(program, SCOPES):
        if (find_name_of_node(sub) or '').lower() == name.lower():
            return sub
    return None


def _clone_interposer(sub: f03.Base, proc: str, base: str, arm_type: str, clone_name: str) -> f03.Base:
    """Clone subprogram ``proc`` as ``clone_name`` with its ``CLASS(base)`` passed-object
    dummy retyped to the concrete ``TYPE(arm_type)``.  A concrete passed-object makes every
    ``this%<deferred>`` call inside the body a static bind, so no body edits are needed."""
    text = sub.tofortran()
    text = re.sub(r'\b' + re.escape(proc) + r'\b', clone_name, text)
    text = re.sub(r'CLASS\s*\(\s*' + re.escape(base) + r'\s*\)', f'TYPE({arm_type})', text, flags=re.IGNORECASE)
    prog = parse_program(f"module zz_tmp\ncontains\n{text}\nend module\n")
    return walk(prog, SCOPES)[0]


def _enclosing_of_type(node: f03.Base, types_: tuple) -> Optional[f03.Base]:
    p = node.parent
    while p is not None and not isinstance(p, types_):
        p = p.parent
    return p


def _module_name_of(node: f03.Base) -> Optional[str]:
    """Lower-cased name of the module / main program that lexically contains ``node``."""
    mod = _enclosing_of_type(node, (f03.Module, f03.Main_Program))
    if mod is None:
        return None
    stmt = ast_utils.atmost_one(ast_utils.children_of_type(mod, (f03.Module_Stmt, f03.Program_Stmt)))
    return str(stmt.children[1]).lower() if stmt is not None else None


def _ensure_use(sub: f03.Base, module: str, name: str) -> None:
    """Add ``USE <module>, ONLY: <name>`` to subprogram ``sub`` if not already there.

    A monomorphised ladder redirects a (formerly type-bound, explicit-interface)
    dispatch to a direct call of a per-arm clone that lives in the interposer's
    module. When the call site is in a *different* module the direct call needs an
    explicit interface, or flang rejects the keyword arguments and the pruner drops
    the clone as unreferenced -- so import the clone at the call site."""
    spec = ast_utils.atmost_one(ast_utils.children_of_type(sub, f03.Specification_Part))
    if spec is not None:
        for u in ast_utils.children_of_type(spec, f03.Use_Stmt):
            if name.lower() in str(u).lower() and module.lower() in str(u).lower():
                return
        prepend_children(spec, f03.Use_Stmt(f"USE {module}, ONLY: {name}"))
    else:
        append_children(sub, f03.Specification_Part(get_reader(f"use {module}, only: {name}\n")))


def clone_shared_interposers(program: f03.Program, plan: MonomorphizationPlan) -> int:
    """Specialise the base's shared interposers per arm so their internal dispatch
    resolves statically.  After :func:`monomorphize_component_dispatch`, a dispatch on a
    shared (non-deferred) binding reads ``obj%slot__arm%binding(args)`` -- a static call
    into the *shared* interposer, whose body still dispatches on its ``CLASS(base)`` dummy.
    For each ``(interposer, arm)`` actually called, emit a clone with the dummy retyped to
    the arm's concrete ``TYPE`` and redirect the call to it.  Returns the clone count."""
    interposers = _base_nondeferred_bindings(program, plan.abstract_base)
    if not interposers:
        return 0

    redirects = []
    for call in walk(program, f03.Call_Stmt):
        designator = call.children[0]
        if not isinstance(designator, f03.Procedure_Designator):
            continue
        obj, _, binding = designator.children
        proc = interposers.get(str(binding).lower())
        if proc is None or not isinstance(obj, f03.Data_Ref):
            continue
        _, tail = _ref_prefix_and_tail(obj)
        arm_type = _arm_from_slot_tail(tail, plan)
        if arm_type is None:
            continue
        redirects.append((call, obj, proc, arm_type))

    clones = {}
    for _, _, proc, arm_type in redirects:
        if (proc, arm_type) in clones:
            continue
        sub = _find_subprogram(program, proc)
        if sub is None:
            continue
        clone_name = f"{proc}__{arm_type}"
        # The clone lands in the interposer's own module (appended to sub.parent).
        clones[(proc, arm_type)] = (clone_name, _module_name_of(sub))
        append_children(sub.parent, _clone_interposer(sub, proc, plan.abstract_base, arm_type, clone_name))

    for call, obj, proc, arm_type in redirects:
        if (proc, arm_type) not in clones:
            continue
        clone_name, clone_mod = clones[(proc, arm_type)]
        # Capture the call's scope/module BEFORE replacing it (replace detaches it).
        call_sub = _enclosing_of_type(call, SCOPES)
        call_mod = _module_name_of(call)
        args = call.children[1]
        argstr = ', '.join(str(a) for a in args.children) if args is not None else ''
        callee = str(obj) + (f", {argstr}" if argstr else "")
        replace_node(call, _parse_exec(f"call {clone_name}({callee})"))
        if call_sub is not None and clone_mod is not None and call_mod != clone_mod:
            _ensure_use(call_sub, clone_mod, clone_name)

    # the original interposers are now dead (every dispatch was redirected to a
    # concrete clone); drop each fully-redirected one + its base TBP binding so no
    # residual ``fir.dispatch`` survives in the now-unreachable shared body.
    proc_binding = {proc: bname for bname, proc in interposers.items()}
    for proc in {p for p, _ in clones}:
        binding = proc_binding.get(proc)
        if binding is not None and not _has_binding_dispatch(program, binding):
            _drop_interposer(program, proc, binding, plan.abstract_base)

    return len(clones)


def _has_binding_dispatch(program: f03.Program, binding: str) -> bool:
    """True if any ``obj%binding(...)`` type-bound dispatch still remains."""
    for call in walk(program, f03.Call_Stmt):
        designator = call.children[0]
        if isinstance(designator, f03.Procedure_Designator) and str(designator.children[2]).lower() == binding.lower():
            return True
    return False


def _drop_interposer(program: f03.Program, proc: str, binding: str, base: str) -> None:
    """Remove the dead interposer subprogram ``proc`` and the base's ``binding => proc`` TBP."""
    for dtdef in walk(program, f03.Derived_Type_Def):
        stmt = ast_utils.singular(ast_utils.children_of_type(dtdef, f03.Derived_Type_Stmt))
        if str(stmt.children[1]).lower() != base.lower():
            continue
        for spec in walk(dtdef, f03.Specific_Binding):
            _, _, _, bname, target = spec.children
            if str(bname).lower() == binding.lower() and target is not None:
                remove_self(spec)
    sub = _find_subprogram(program, proc)
    if sub is not None:
        remove_self(sub)


def _in_interface(node: f03.Base) -> bool:
    anc = node.parent
    while anc is not None:
        if isinstance(anc, f03.Interface_Block):
            return True
        anc = anc.parent
    return False


def _concrete_type_spec(concrete: str) -> f03.Declaration_Type_Spec:
    prog = parse_program(f"subroutine zz_tmp()\ntype({concrete}) :: zz_v\nend subroutine\n")
    return walk(prog, f03.Type_Declaration_Stmt)[0].children[0]


def retype_to_concrete(program: f03.Program, base: str, concrete: str) -> int:
    """Specialise an axis that is *fixed* to one concrete type at the call site
    (ICON's ``trans`` -> ``t_trivial_transfer``, ``agen`` -> ``t_primal_flip_flop_lhs``):
    rewrite every ``CLASS(base)`` declaration -- component, dummy or local -- to
    ``TYPE(concrete)``, so each ``%binding`` on it becomes a static bind and the
    pointer associations stay type-consistent.  Declarations inside an abstract
    interface are left alone (the deferred-interface signature must stay
    polymorphic).  Returns the number of declarations retyped."""
    count = 0
    for decl in walk(program, (f03.Type_Declaration_Stmt, f03.Data_Component_Def_Stmt)):
        if _in_interface(decl):
            continue
        type_spec = decl.children[0]
        if (isinstance(type_spec, f03.Declaration_Type_Spec) and type_spec.children[0] == 'CLASS'
                and str(type_spec.children[1]).lower() == base.lower()):
            replace_node(type_spec, _concrete_type_spec(concrete))
            count += 1
    return count


# ---------------------------------------------------------------------------
# Driver: compose the four primitives over a per-translation-unit spec.
#
# A translation unit can carry several orthogonal polymorphic axes at once
# (ICON's solver TU has three: the backend ``t_ocean_solve%act``, the transfer
# ``trans`` and the lhs ``agen``).  Each axis is one of two shapes:
#
#   * *ladder* -- the entity is runtime-allocated to one of a closed set of
#     concrete arms; the analyzer's :class:`MonomorphizationPlan` drives the tag
#     ladder (local and/or component) and the shared interposers are cloned; or
#   * *retype* -- the entity is pinned to a single concrete type at its
#     construction site, so a plain ``CLASS(base)`` -> ``TYPE(concrete)`` retype
#     of its declarations makes every access static, no tag needed.
#
# The spec names each axis and its strategy; for now it is hand-written (the
# locked design: a second pass will auto-generate it from the construction site).
# ---------------------------------------------------------------------------
LADDER = 'ladder'
RETYPE = 'retype'


@dataclass
class AxisSpec:
    """One polymorphic dispatch axis and how to collapse it.

    ``strategy`` is :data:`LADDER` (closed set of runtime arms -> tag ladder +
    interposer clone; ``concrete`` unused) or :data:`RETYPE` (pinned to a single
    ``concrete`` type at the construction site -> declaration retype)."""
    base: str
    strategy: str
    concrete: Optional[str] = None


@dataclass
class MonomorphizationSpec:
    """The per-translation-unit monomorphisation plan: each polymorphic axis
    paired with its collapse strategy.  Hand-written first; auto-generated later.
    Retype axes are applied before ladder axes so a cloned interposer that reads
    a retyped member already sees the concrete type."""
    axes: List[AxisSpec]


@dataclass
class MonomorphizationStats:
    """Per-strategy counts of what the driver rewrote, so a caller can detect a
    no-op axis or assert the expected amount of rewriting happened."""
    locals_rewritten: int = 0
    components_rewritten: int = 0
    interposers_cloned: int = 0
    declarations_retyped: int = 0


def monomorphize(program: f03.Program, spec: MonomorphizationSpec, stack_slots: bool = False) -> MonomorphizationStats:
    """Apply ``spec`` to ``program`` in place, collapsing every listed polymorphic
    axis into static dispatch, and return the per-strategy rewrite counts.

    Order: the analyzer plans are taken from the *pristine* AST first; then retype
    axes run (pure declaration rewrites that make member accesses concrete), then
    ladder axes (slot expansion + tag ladder + interposer clone) using those plans.

    Raises :class:`UnsupportedProgram` (from the analyzer) if a ladder axis's
    hierarchy is outside the soundly-monomorphisable class, or ``ValueError`` for a
    malformed spec (unknown strategy, retype without a concrete type, or a ladder
    axis with no matching dispatch plan in the unit)."""
    unknown = sorted({a.strategy for a in spec.axes if a.strategy not in (LADDER, RETYPE)})
    if unknown:
        raise ValueError(f"unknown monomorphisation strategy(s): {unknown}")

    retype_axes = [a for a in spec.axes if a.strategy == RETYPE]
    ladder_axes = [a for a in spec.axes if a.strategy == LADDER]

    # Plans are pure data (names), so reading them from the pristine AST keeps the
    # ladder independent of the retype rewrites that run between here and its use.
    # Scope the analysis to the spec's ladder axes: a large extraction closure
    # carries unrelated dispatch roots / generic CLASS(*) containers the spec
    # never rewrites, and validating those would reject a hierarchy the pass does
    # not touch.
    plans = ({
        p.abstract_base.lower(): p
        for p in analyze(program, only_bases=[a.base for a in ladder_axes])
    } if ladder_axes else {})

    stats = MonomorphizationStats()
    for axis in retype_axes:
        if axis.concrete is None:
            raise ValueError(f"retype axis `{axis.base}` needs a concrete type to retype to")
        stats.declarations_retyped += retype_to_concrete(program, axis.base, axis.concrete)

    for axis in ladder_axes:
        plan = plans.get(axis.base.lower())
        if plan is None:
            raise ValueError(f"ladder axis `{axis.base}` has no dispatch plan: the abstract base is "
                             f"absent, non-polymorphic, or has no concrete arm in this translation unit")
        stats.locals_rewritten += monomorphize_local_dispatch(program, plan, stack_slots)
        stats.components_rewritten += monomorphize_component_dispatch(program, plan, stack_slots)
        stats.interposers_cloned += clone_shared_interposers(program, plan)

    return stats


# ---------------------------------------------------------------------------
# Auto-discovery: the default fparser pass.
#
# The driver above takes a hand-written spec.  In the inliner pipeline we want
# monomorphisation to run *by default and always*, with no per-kernel spec --
# so single-level abstract dispatch (ICON's halo ``t_comm_pattern``) collapses
# to static calls the bridge can lower, and nobody has to remember to ask for
# it.  :func:`discover_axes` finds every soundly-monomorphisable axis in a
# program and picks its strategy; :func:`monomorphize_auto` applies them.
# ---------------------------------------------------------------------------


def _dispatch_binding_names(program: f03.Program) -> Set[str]:
    """Lower-cased binding names that appear in a live type-bound dispatch
    ``obj%binding(...)`` -- a :class:`Procedure_Designator` (a ``CALL obj%b`` or
    a function-reference ``obj%b(...)``) anywhere in the program."""
    names: Set[str] = set()
    for des in walk(program, f03.Procedure_Designator):
        names.add(str(des.children[2]).lower())
    return names


def _has_inunit_construction(program: f03.Program, plan: MonomorphizationPlan) -> bool:
    """True if some arm of ``plan`` is constructed in this unit via
    ``ALLOCATE(arm :: ...)`` -- the tag source a ladder needs.  Without it a
    ladder would leave every tag 0 (no arm runs), so a multi-arm axis with no
    in-unit construction is not laddered (see :func:`discover_axes`)."""
    arms = {a.type_name.lower() for a in plan.arms}
    for alloc in walk(program, f03.Allocate_Stmt):
        alloc_type = alloc.children[0]
        if alloc_type is not None and str(alloc_type).lower() in arms:
            return True
    return False


def discover_axes(program: f03.Program) -> List[AxisSpec]:
    """Find every single-level abstract dispatch axis in ``program`` that the
    pass can soundly collapse, and pick a strategy for each.

    An axis is considered only when it has *live dispatch* (a deferred binding
    of the abstract base is actually invoked as ``obj%binding(...)``) AND its
    concrete arm(s) are present in the unit.  The arm-present requirement is the
    key gate: a kernel that externalises its halo (or whose concrete arm is
    built by an externalised factory and so is absent from the closure) has no
    arm to retype to, :func:`analyze` rejects the base, and the pass is a
    precise no-op -- it never perturbs such a kernel.

    Strategy selection (sound by construction):

      * exactly one concrete arm -> :data:`RETYPE` to it (every ``CLASS(base)``
        becomes ``TYPE(arm)``; trivially correct -- only one runtime type
        exists).  This is ICON's standard-build halo: ``t_comm_pattern`` with
        ``t_comm_pattern_yaxt`` cpp'd out, leaving ``t_comm_pattern_orig``.
      * two or more arms with in-unit construction -> :data:`LADDER` over the
        tags set at the ``ALLOCATE(arm :: ..)`` sites.
      * two or more arms, no in-unit construction -> *skipped* (logged): the
        runtime arm cannot be inferred, so neither retype (which one?) nor a
        ladder (no tag) is sound.  An explicit :class:`MonomorphizationSpec`
        must pin it.

    A dispatch root outside the soundly-monomorphisable class (multi-level,
    ``CLASS(*)`` in its own definition, a missing override) is skipped too --
    :func:`analyze` rejects it and the rejection is swallowed per axis, leaving
    it for downstream externalisation / the bridge's polymorphism reject."""
    live = _dispatch_binding_names(program)
    if not live:
        return []

    # Candidate bases: abstract dispatch roots (a type with deferred bindings)
    # at least one of whose deferred bindings is live.
    candidates: List[str] = []
    for dtd in walk(program, f03.Derived_Type_Def):
        ti = read_type_info(dtd)
        if ti.deferred and any(d in live for d in ti.deferred):
            candidates.append(ti.name)

    axes: List[AxisSpec] = []
    for base in sorted(set(candidates)):
        try:
            plans = analyze(program, only_bases=[base])
        except UnsupportedProgram as exc:
            logger.debug("monomorphize: skipping abstract base `%s` (not soundly monomorphisable: %s)", base, exc)
            continue
        if not plans:
            continue
        plan = plans[0]
        if len(plan.arms) == 1:
            axes.append(AxisSpec(base, RETYPE, concrete=plan.arms[0].type_name))
        elif _has_inunit_construction(program, plan):
            axes.append(AxisSpec(base, LADDER))
        else:
            logger.warning(
                "monomorphize: abstract base `%s` has %d concrete arms (%s) but no in-unit construction; "
                "cannot infer the runtime arm -- leaving the dispatch polymorphic (pin it with an explicit "
                "MonomorphizationSpec if it must lower)", base, len(plan.arms),
                ", ".join(a.type_name for a in plan.arms))
    return axes


def _module_of(program: f03.Program, type_name: str) -> Optional[f03.Module]:
    """The :class:`Module` whose specification part defines ``type_name``."""
    for mod in walk(program, f03.Module):
        spec = ast_utils.atmost_one(ast_utils.children_of_type(mod, f03.Specification_Part))
        if spec is None:
            continue
        for dtd in ast_utils.children_of_type(spec, f03.Derived_Type_Def):
            if read_type_info(dtd).name == type_name.lower():
                return mod
    return None


def _module_name(mod: f03.Module) -> str:
    stmt = ast_utils.singular(ast_utils.children_of_type(mod, f03.Module_Stmt))
    return str(stmt.children[1]).lower()


def _toposort_type_defs(spec: f03.Specification_Part) -> None:
    """Reorder the derived-type definitions in ``spec`` so a type appears after
    every local type it depends on (its ``EXTENDS`` parent and any local type
    used as a component).  After consolidation a container that was retyped to a
    formerly-downstream arm would otherwise be "used before defined"."""
    dtds = list(ast_utils.children_of_type(spec, f03.Derived_Type_Def))
    if len(dtds) < 2:
        return
    local = {read_type_info(d).name: d for d in dtds}

    def deps(dtd: f03.Derived_Type_Def) -> Set[str]:
        ti = read_type_info(dtd)
        out = {ti.parent} if ti.parent in local else set()
        for ts in walk(dtd, f03.Declaration_Type_Spec):
            tname = str(ts.children[1]).lower() if len(ts.children) > 1 else None
            if tname in local and tname != ti.name:
                out.add(tname)
        return out

    # Stable DFS post-order topo-sort (a back-edge from a self/cyclic ref is
    # ignored -- it can only be a recursive pointer, which Fortran allows).
    ordered: List[f03.Derived_Type_Def] = []
    seen: Set[str] = set()
    onstack: Set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        onstack.add(name)
        for d in sorted(deps(local[name])):
            if d not in onstack:
                visit(d)
        onstack.discard(name)
        ordered.append(local[name])

    for d in dtds:
        visit(read_type_info(d).name)

    if [id(d) for d in ordered] == [id(d) for d in dtds]:
        return  # already in dependency order
    # Re-thread the spec: keep the non-type-def items in place and splice the
    # topo-ordered type defs back in right after the USE / IMPLICIT prologue
    # (derived-type defs must follow IMPLICIT, and precede the declarations that
    # use them).
    non_dtd = [c for c in spec.children if c not in dtds]
    ins = 0
    for i, c in enumerate(non_dtd):
        if isinstance(c, (f03.Use_Stmt, f03.Implicit_Part)):
            ins = i + 1
    new_content = non_dtd[:ins] + ordered + non_dtd[ins:]
    spec.content[:] = new_content
    for c in new_content:
        c.parent = spec


def _redirect_uses(program: f03.Program, old_mod: str, new_mod: str) -> None:
    """Rewrite every ``USE old_mod[...]`` to ``USE new_mod[...]`` (its symbols
    now live in ``new_mod`` after a module merge); a self-``USE`` of the merged
    module is dropped."""
    for use in walk(program, f03.Use_Stmt):
        nm = ast_utils.atmost_one(ast_utils.children_of_type(use, f03.Name))
        if nm is None or str(nm).lower() != old_mod.lower():
            continue
        enclosing = _module_name_of(use)
        if enclosing == new_mod.lower():
            remove_self(use)  # the symbols are now local to this module
        else:
            replace_node(
                use,
                f03.Use_Stmt(re.sub(r'\b' + re.escape(old_mod) + r'\b', new_mod, str(use), count=1,
                                    flags=re.IGNORECASE)))


def consolidate_arm_module(program: f03.Program, base_type: str, arm_type: str) -> bool:
    """Merge the module that defines ``arm_type`` into the module that defines
    ``base_type``, so a retyped container in the base module no longer creates a
    circular module dependency on the (formerly downstream) arm.

    Moves the arm module's type definitions + declarations into the base
    module's specification part, its procedures into the base module's
    ``CONTAINS``, redirects every ``USE`` of the arm module to the base module,
    drops the now-empty arm module, and topologically re-orders the merged type
    defs.  A no-op (returns ``False``) when the two types already share a module.
    """
    base_mod = _module_of(program, base_type)
    arm_mod = _module_of(program, arm_type)
    if base_mod is None or arm_mod is None or base_mod is arm_mod:
        return False
    base_name, arm_name = _module_name(base_mod), _module_name(arm_mod)

    base_spec = ast_utils.singular(ast_utils.children_of_type(base_mod, f03.Specification_Part))
    arm_spec = ast_utils.atmost_one(ast_utils.children_of_type(arm_mod, f03.Specification_Part))

    # 1. Move the arm spec's declarations (everything but its IMPLICIT and a
    #    self-or-base USE) into the base spec.
    if arm_spec is not None:
        for child in list(arm_spec.children):
            if isinstance(child, f03.Implicit_Part):
                continue
            if isinstance(child, f03.Use_Stmt):
                nm = ast_utils.atmost_one(ast_utils.children_of_type(child, f03.Name))
                if nm is not None and str(nm).lower() in (base_name, arm_name):
                    continue  # base symbols are already in scope post-merge
            remove_self(child)
            append_children(base_spec, child)

    # 2. Move the arm module's procedures into the base module's CONTAINS.  A
    #    new ``Module_Subprogram_Part`` must be inserted BEFORE the module's
    #    ``END MODULE`` (appending to the module would orphan it after the end
    #    statement).
    arm_sub = ast_utils.atmost_one(ast_utils.children_of_type(arm_mod, f03.Module_Subprogram_Part))
    if arm_sub is not None:
        base_sub = ast_utils.atmost_one(ast_utils.children_of_type(base_mod, f03.Module_Subprogram_Part))
        subs = [c for c in arm_sub.children if isinstance(c, SCOPES)]
        if base_sub is None:
            for s in subs:
                remove_self(s)
            new_part = f03.Module_Subprogram_Part(
                get_reader("contains\n" + "\n".join(s.tofortran() for s in subs) + "\n"))
            end_stmt = ast_utils.singular(ast_utils.children_of_type(base_mod, f03.End_Module_Stmt))
            base_mod.content.insert(base_mod.children.index(end_stmt), new_part)
            new_part.parent = base_mod
        else:
            for s in subs:
                remove_self(s)
                append_children(base_sub, s)

    # 3. Redirect USEs, drop the emptied arm module, order the merged types.
    remove_self(arm_mod)
    _redirect_uses(program, arm_name, base_name)
    _toposort_type_defs(base_spec)

    # 4. The retype may have rewritten ``CLASS(base)`` -> ``TYPE(arm)`` in scopes
    #    in OTHER modules (ICON's halo wrappers live in a third module, separate
    #    from both the base type and the arm).  Those now name the arm type but
    #    may not import it -- it lives in the base module post-merge.  Import it.
    for sub in walk(program, SCOPES):
        if _module_name_of(sub) == base_name:
            continue
        if any(
                len(ts.children) > 1 and str(ts.children[1]).lower() == arm_type.lower()
                for ts in walk(sub, f03.Declaration_Type_Spec)):
            _ensure_use(sub, base_name, arm_type)
    return True


def monomorphize_auto(program: f03.Program, stack_slots: bool = False) -> MonomorphizationStats:
    """Discover and collapse every soundly-monomorphisable single-level abstract
    dispatch axis in ``program``, in place -- the default, spec-free pass.

    Returns the per-strategy rewrite counts (all-zero when the program has no
    live abstract dispatch with a concrete arm present, the common case)."""
    axes = discover_axes(program)
    if not axes:
        return MonomorphizationStats()
    logger.debug("monomorphize: auto-collapsing %d axis(es): %s", len(axes),
                 ", ".join(f"{a.base}->{a.strategy}" for a in axes))
    stats = monomorphize(program, MonomorphizationSpec(axes), stack_slots=stack_slots)
    # Co-locate each arm's definition with its base so a retyped container in the
    # base module doesn't depend circularly on the (formerly downstream) arm.
    for axis in axes:
        if axis.strategy == RETYPE and axis.concrete is not None:
            if consolidate_arm_module(program, axis.base, axis.concrete):
                logger.debug("monomorphize: consolidated arm `%s` into base `%s`'s module", axis.concrete, axis.base)
    return stats
