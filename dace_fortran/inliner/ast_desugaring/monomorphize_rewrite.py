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
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk

from dace_fortran.inliner import ast_utils
from dace_fortran.inliner.ast_desugaring.monomorphize import analyze, MonomorphizationPlan, parse_program
from dace_fortran.inliner.ast_desugaring.utils import (append_children, find_name_of_node, remove_self, replace_node)

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
    lines = [f"integer :: {_tag_var(slot)}"]
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
        clones[(proc, arm_type)] = clone_name
        append_children(sub.parent, _clone_interposer(sub, proc, plan.abstract_base, arm_type, clone_name))

    for call, obj, proc, arm_type in redirects:
        if (proc, arm_type) not in clones:
            continue
        args = call.children[1]
        argstr = ', '.join(str(a) for a in args.children) if args is not None else ''
        callee = str(obj) + (f", {argstr}" if argstr else "")
        replace_node(call, _parse_exec(f"call {clones[(proc, arm_type)]}({callee})"))

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
