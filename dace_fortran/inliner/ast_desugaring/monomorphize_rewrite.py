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
import hashlib
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
from dace_fortran.inliner.ast_desugaring.pruning import keep_sorted_used_modules
from dace_fortran.inliner.ast_desugaring.utils import (append_children, find_name_of_node, prepend_children,
                                                       remove_self, replace_node)

logger = logging.getLogger(__name__)

SCOPES = (f03.Subroutine_Subprogram, f03.Function_Subprogram)


def _tag_var(var: str) -> str:
    return f"{var}__tag"


def _arm_slot(var: str, type_name: str) -> str:
    return f"{var}__{type_name}"


#: Fortran 2008 caps an identifier at 63 characters.
_FORTRAN_MAX_NAME = 63


def _clone_name(proc: str, arm: str) -> str:
    """Compose a per-arm interposer/constructor clone name ``proc__arm``, shortened
    to stay within Fortran's 63-char identifier limit.  A ladder that composes
    several dispatch axes (ICON's solver = backend x agen x transfer, whose shared
    ``construct``/``solve`` interposer is cloned once per axis) chains a ``__<arm>``
    suffix per axis and overruns the limit.  When it would, keep a readable prefix
    of the full name plus a stable hash of it -- the definition and every redirected
    call / ``USE`` all read the one computed name back from the clone dicts, so a
    deterministic transform keeps them in agreement.  (Component slot names built by
    :func:`_arm_slot` compose a single axis and never overrun, so they are left
    verbatim -- other passes match them by the literal ``__<arm>`` tail.)"""
    full = f"{proc}__{arm}"
    if len(full) <= _FORTRAN_MAX_NAME:
        return full
    digest = hashlib.md5(full.encode()).hexdigest()[:8]
    return f"{full[:_FORTRAN_MAX_NAME - len(digest) - 2]}__{digest}"


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
                              stack_slots: bool = False,
                              pointer: bool = False) -> List[f03.Data_Component_Def_Stmt]:
    """``integer :: act__tag`` + one ``type(arm), allocatable :: act__arm`` component per arm.
    With ``stack_slots`` the per-arm component is a plain (non-allocatable) member -- the
    SDFG-lowerable form (see :func:`_expanded_decls`).  With ``pointer`` (an original
    ``CLASS(base), POINTER`` slot bound by pointer association, ICON's ``t_lhs%trans`` /
    ``%agen``) the per-arm component is itself a ``POINTER`` so the association's aliasing
    is preserved -- and it lowers to a View, not a copy."""
    if pointer:
        attr, init = ", pointer", " => NULL()"
    else:
        attr, init = ("" if stack_slots else ", allocatable"), ""
    # The tag defaults to 0 ("not constructed") so an `ALLOCATED(this%act)` guard
    # rewritten to `this%act__tag /= 0` reads correctly before the construction
    # site sets it (a derived-type component is otherwise undefined until set).
    lines = [f"integer :: {_tag_var(slot)} = 0"]
    for arm in plan.arms:
        lines.append(f"type({arm.type_name}){attr} :: {_arm_slot(slot, arm.type_name)}{init}")
    return _parse_component_decls("\n".join(lines))


def _component_is_pointer(comp: f03.Data_Component_Def_Stmt) -> bool:
    """True if a derived-type component is declared ``POINTER`` (its arm slots must
    then stay pointers so a ``this%slot => target`` association type-checks)."""
    attrs = comp.children[1]
    return attrs is not None and 'POINTER' in str(attrs).upper()


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


def _member_prefix_and_tail(obj: f03.Base) -> Optional[Tuple[str, str]]:
    """Split a member reference into (``a%b`` prefix, last component), for either a
    ``Data_Ref`` (an rvalue path ``this%trans``) or a ``Data_Pointer_Object`` (a
    pointer-assign LHS ``this%op``, whose children are ``(base, '%', name)``).
    Returns ``None`` for a bare name (no ``%`` component selector)."""
    if isinstance(obj, f03.Data_Ref):
        return _ref_prefix_and_tail(obj)
    if isinstance(obj, f03.Data_Pointer_Object):
        parts = [c for c in obj.children if c != '%']
        if len(parts) < 2:
            return None
        return '%'.join(str(p) for p in parts[:-1]), str(parts[-1])
    return None


def _stmt_refs_slot(node: f03.Base, slot_names: Set[str]) -> bool:
    """True if ``node`` references a slot ``...%<slot>`` -- a ``<slot>`` appearing as
    a *component* (a non-first part of a ``Data_Ref`` / ``Data_Pointer_Object``),
    not merely a same-named local.  Used to find slot reads in an IF/else-if
    condition, which the statement ladder (Call/Assign/Ptr-assign only) misses."""
    for ref in walk(node, (f03.Data_Ref, f03.Data_Pointer_Object)):
        parts = [c for c in ref.children if not isinstance(c, str)]
        if any(str(p).lower() in slot_names for p in parts[1:]):
            return True
    return False


def _slot_is_data_carrying(program: f03.Program, slot: str) -> bool:
    """True if ``slot`` is read as a *data member* -- some ``...%slot%<member>`` where
    ``slot`` is not the final path part, so ``<member>`` is a stored-component access
    on it (ICON's ``this%trans%nidx``).  A whole-slot reference (``this%slot => x``) or
    a dispatch receiver (``this%slot%binding(...)``, where ``slot`` is the receiver
    ``Data_Ref``'s last part and the binding is the ``Procedure_Designator``'s own
    child, not a ``Data_Ref`` part) does *not* count.

    A data-carrying ``CLASS(base), POINTER`` slot is kept ``CLASS`` rather than
    expanded away: its data reads then lower natively -- including declaration
    dimensions (``REAL :: x_t(this%trans%nidx)``) and DO bounds a statement ladder
    cannot reach, and without routing them through a per-arm ``POINTER`` (the
    unflattened ptr-member-struct read) -- while only its dispatch is laddered."""
    sl = slot.lower()
    for ref in walk(program, f03.Data_Ref):
        parts = ref.children
        for i, part in enumerate(parts):
            if str(part).lower() == sl and i < len(parts) - 1:
                return True
    return False


def _stmt_has_slot_dispatch(node: f03.Base, slot_names: Set[str]) -> bool:
    """True if ``node`` contains a type-bound dispatch through a slot -- a
    ``Procedure_Designator`` ``...%slot%binding`` (a ``CALL`` or function-ref) whose
    receiver ``Data_Ref`` ends in one of ``slot_names``.  This distinguishes a
    dispatch through a kept-``CLASS`` hybrid slot (which must be laddered onto the
    concrete per-arm pointer) from a plain data-member read of it (which stays)."""
    for des in walk(node, f03.Procedure_Designator):
        obj = des.children[0]
        if isinstance(obj, f03.Data_Ref) and str(obj.children[-1]).lower() in slot_names:
            return True
    return False


#: statement kinds that can reference a slot (dispatch, assignment, pointer-assign).
SLOT_STMT_TYPES = (f03.Call_Stmt, f03.Assignment_Stmt, f03.Pointer_Assignment_Stmt)


def _component_slot_owner_types(program: f03.Program, base: str) -> Set[str]:
    """Lower-cased names of the derived types that DECLARE a ``CLASS(base)`` component
    -- the types that OWN a laddered slot.  A component of the SAME name on any other
    type (ICON's concrete ``t_p_comm_pattern_orig%p`` vs the abstract 18-arm
    ``t_stack_op`` slot ``p``) is unrelated and must not be retargeted by the ladder."""
    owners: Set[str] = set()
    for comp, _ in _component_slots(program, base):
        dtd = _enclosing_of_type(comp, (f03.Derived_Type_Def, ))
        if dtd is not None:
            owners.add(read_type_info(dtd).name)
    return owners


def _ref_path_type(scope: f03.Base, parts: List[f03.Base], dtds: dict) -> Optional[str]:
    """Derived-type name a component path (``Data_Ref`` parts, each possibly
    array-subscripted) resolves to, or ``None`` when unresolvable (an intrinsic, a
    function result, or a name with no visible declaration).  Used to decide whether
    a ``%slot`` reference sits on a slot-owning container."""

    def base_name(node: f03.Base) -> Optional[str]:
        if isinstance(node, f03.Name):
            return str(node)
        if isinstance(node, f03.Part_Ref) and isinstance(node.children[0], f03.Name):
            return str(node.children[0])
        return None

    b = base_name(parts[0])
    if b is None:
        return None
    t = _entity_declared_type(scope, b)
    for p in parts[1:]:
        if t is None:
            return None
        nm = base_name(p)
        t = _component_type(dtds, t, nm) if nm else None
    return t


def _slot_statement_ladder(stmt: f03.Base,
                           slot_names: Set[str],
                           plan: MonomorphizationPlan,
                           owner_types: Optional[Set[str]] = None,
                           scope: Optional[f03.Base] = None,
                           dtds: Optional[dict] = None,
                           tinfos: Optional[dict] = None) -> Optional[List[f03.Base]]:
    """If ``stmt`` references a slot ``<prefix>%<slot>`` (anywhere, even buried in a
    sub-expression), return a tag ladder that re-emits ``stmt`` once per arm with
    every ``%slot`` retargeted to ``%slot__arm``; ``None`` if it touches no slot.
    Because each arm slot is a concrete ``TYPE``, every retargeted reference -- a
    type-bound dispatch or a plain data-member access -- becomes a static bind.

    When ``owner_types``/``scope`` are given, the ladder is TYPE-AWARE: a ``%slot``
    reference whose container POSITIVELY resolves to a type that does not own the slot
    (nor extend an owner) is left alone -- ``re.sub`` would otherwise corrupt an
    unrelated same-named component (ICON's ``t_p_comm_pattern_orig%p``).  Conservative:
    an unresolvable container (``None``) still ladders, preserving prior behaviour."""
    prefix = slot = None
    container_parts: Optional[Tuple[f03.Base, ...]] = None
    for ref in walk(stmt, f03.Data_Ref):
        parts = ref.children
        for i, part in enumerate(parts):
            if i > 0 and str(part).lower() in slot_names:
                prefix, slot = '%'.join(str(p) for p in parts[:i]), str(part)
                container_parts = parts[:i]
                break
        if slot is not None:
            break
    if slot is None:
        return None
    if owner_types is not None and scope is not None and container_parts:
        ctype = _ref_path_type(scope, list(container_parts), dtds or {})
        if ctype is not None and not (owner_types & set(_arm_ancestor_rank(tinfos or {}, ctype))):
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
    """Rewrite dispatch on a ``CLASS(plan.abstract_base)`` *component*, in place, and
    return the number of polymorphic components handled.

    Two slot shapes, distinguished by whether the slot carries *data*:

      * a *full-expansion* slot -- an ``ALLOCATE``-constructed one (ICON's
        ``t_ocean_solve%act``), or a dispatch-only ``POINTER`` -- is replaced by a
        tag + one concrete slot per arm, and *every* statement/condition that
        references it is re-emitted per arm with ``%slot`` retargeted to
        ``%slot__arm`` (so a data-member access on the now-gone slot goes through the
        concrete per-arm slot); and
      * a *hybrid* slot -- a ``CLASS(base), POINTER`` slot read as a data member
        (``this%trans%nidx``, :func:`_slot_is_data_carrying`) -- KEEPS its ``CLASS``
        component (its data reads lower natively, including declaration dimensions a
        statement ladder cannot reach) and gains a tag + per-arm ``POINTER`` slots
        beside it; only its *dispatch* (``CALL this%trans%into(...)``) is laddered.
        The pointer association that sets the tag is completed by
        :func:`devirtualize_pointer_flow`, which keeps the kept slot's association."""
    base = plan.abstract_base
    slots = _component_slots(program, base)
    if not slots:
        return 0

    # Classify each slot: a data-carrying POINTER slot is hybrid (kept CLASS); every
    # other slot is fully expanded.
    hybrid_names: Set[str] = set()
    full_names: Set[str] = set()
    for comp, names in slots:
        pointer = _component_is_pointer(comp)
        for n in names:
            (hybrid_names if pointer and _slot_is_data_carrying(program, n) else full_names).add(n.lower())

    # The types that OWN a laddered slot (declare the ``CLASS(base)`` component) --
    # captured NOW, before step 1 deletes a full slot's CLASS component.  The slot
    # ladder's textual ``%slot`` retarget is restricted to these (and their
    # extensions) so a same-named component on an unrelated type (ICON's concrete
    # ``t_p_comm_pattern_orig%p`` vs the abstract 18-arm ``t_stack_op`` slot ``p``)
    # is left alone.
    owner_types = _component_slot_owner_types(program, base)

    # 1. expand the container component.  A hybrid slot keeps its CLASS component and
    #    adds the tag + per-arm POINTER slots beside it; a full slot is replaced by
    #    them.  A ``POINTER`` slot expands to ``POINTER`` arm slots so the tag
    #    source's ``this%slot => target`` type-checks and the aliasing survives.
    for comp, names in slots:
        pointer = _component_is_pointer(comp)
        additions = [d for n in names for d in _expanded_component_decls(n, plan, stack_slots, pointer)]
        if any(n.lower() in hybrid_names for n in names):
            replace_node(comp, _parse_component_decls(str(comp)) + additions)
        else:
            replace_node(comp, additions)

    # 2. factory ALLOCATE(concrete :: prefix%slot) -> set the tag + allocate the
    #    matching concrete slot.  Only a full slot is ALLOCATE-constructed.
    for alloc in walk(program, f03.Allocate_Stmt):
        alloc_type, alloc_list, _ = alloc.children
        if alloc_type is None:
            continue
        for obj in alloc_list.children:
            if not isinstance(obj, f03.Data_Ref):
                continue
            prefix, tail = _ref_prefix_and_tail(obj)
            if tail.lower() not in full_names:
                continue
            tag, arm = _find_arm(plan, str(alloc_type))
            set_tag = f"{prefix}%{_tag_var(tail)} = {tag}"
            rewrite = set_tag if stack_slots else f"{set_tag}\nallocate({prefix}%{_arm_slot(tail, arm)})"
            replace_node(alloc, _parse_exec(rewrite))
            break

    # 2b. ALLOCATED(prefix%slot) init guards -> the tag check (full slots only; a
    #     kept hybrid slot retains its own association status).
    _rewrite_allocated_queries(program, full_names)

    # Type maps for the ladder's owner-aware ``%slot`` retarget (``owner_types`` was
    # captured before step 1 deleted the CLASS component).
    dtds = {read_type_info(d).name: d for d in walk(program, f03.Derived_Type_Def)}
    tinfos = {read_type_info(d).name: read_type_info(d) for d in walk(program, f03.Derived_Type_Def)}

    # 3. Ladder executable CONSTRUCTS whose CONDITION reads a FULL slot member (an
    #    `IF (this%act%...)` / else-if / one-line-IF) -- the statement ladder below
    #    only visits Call/Assign/Ptr-assign, so a slot read in a condition would
    #    survive as a reference to the expanded-away slot.  Re-emit the whole
    #    enclosing construct once per arm; the shared `%slot`->`%slot__arm` retarget
    #    covers its body too, so those inner statements are not re-laddered by step
    #    4.  A hybrid slot's condition read (`IF (this%trans%is_solver_pe)`) is a
    #    data read that stays on the kept CLASS slot; only a hybrid *dispatch* buried
    #    in a condition (rare) needs laddering.
    cond_units: List[f03.Base] = []
    seen_units: Set[int] = set()
    for carrier in walk(program, (f03.If_Then_Stmt, f03.Else_If_Stmt, f03.If_Stmt)):
        if not (_stmt_refs_slot(carrier, full_names) or _stmt_has_slot_dispatch(carrier, hybrid_names)):
            continue
        unit = carrier if isinstance(carrier, f03.If_Stmt) else _enclosing_of_type(carrier, (f03.If_Construct, ))
        if unit is not None and id(unit) not in seen_units:
            seen_units.add(id(unit))
            cond_units.append(unit)
    for unit in cond_units:
        ladder = _slot_statement_ladder(unit, full_names | hybrid_names, plan, owner_types,
                                        _enclosing_of_type(unit, SCOPES), dtds, tinfos)
        if ladder is not None:
            replace_node(unit, ladder)

    # 4. every other statement -> ladder over the tag with `%slot` retargeted to
    #    `%slot__arm`.  A full slot is laddered on ANY reference (dispatch or data,
    #    since its CLASS slot is gone): the call becomes a static concrete-TYPE bind,
    #    a whole-statement data-member assignment (`this%act%res_loc_wp => ...`) and a
    #    slot member buried in a sub-expression (`call timer_start(this%act%lhs%timer)`)
    #    retarget too, as does ICON's non-deferred shared `this%act%solve`.  A hybrid
    #    slot is laddered ONLY when the statement DISPATCHES through it
    #    (`CALL this%trans%into(...)`); its plain data reads/writes stay on the kept
    #    CLASS slot and lower natively.
    for stmt in walk(program, SLOT_STMT_TYPES):
        if not (_stmt_refs_slot(stmt, full_names) or _stmt_has_slot_dispatch(stmt, hybrid_names)):
            continue
        ladder = _slot_statement_ladder(stmt, full_names | hybrid_names, plan, owner_types,
                                        _enclosing_of_type(stmt, SCOPES), dtds, tinfos)
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
        clone_name = _clone_name(proc, arm_type)
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
# Pointer-association tag source.
#
# The ``ALLOCATE(arm :: slot)`` tag source (component dispatch) names the concrete
# arm inline.  ICON's ``t_lhs%trans`` / ``%agen`` are instead ``CLASS(base),
# POINTER`` slots bound by *pointer association* one interprocedural hop away: a
# constructor ``ctor(this, .., dummy)`` with a ``CLASS(base), TARGET :: dummy``
# does ``this%slot => dummy``, and the concrete arm arrives as the actual argument
# at the call site (``CALL obj%ctor(.., a_concrete_local)``).  This clones the
# constructor per concrete arm -- the dummy retyped to ``TYPE(arm)`` (so every
# ``dummy%binding`` inside becomes a static bind) and the association rewritten to
# set the slot's tag + associate the concrete per-arm slot -- and redirects each
# call to the matching clone, mirroring :func:`clone_shared_interposers`.
# ---------------------------------------------------------------------------


def _arm_index(plan: MonomorphizationPlan, type_name: str) -> Optional[int]:
    """1-based tag value for a concrete arm type, or ``None`` if it is not an arm."""
    for i, arm in enumerate(plan.arms, start=1):
        if arm.type_name.lower() == type_name.lower():
            return i
    return None


def _specific_binding_targets(program: f03.Program) -> dict:
    """``{binding_name: target_proc}`` for every specific (non-deferred) type-bound
    procedure in the program -- maps an ``obj%binding`` dispatch to its procedure."""
    out = {}
    for binding in walk(program, f03.Specific_Binding):
        _, _, _, bname, target = binding.children
        if target is not None:
            out[str(bname).lower()] = str(target).lower()
    return out


def _dummy_arg_names(sub: f03.Base) -> List[str]:
    """Ordered dummy-argument names of subprogram ``sub``."""
    stmt = ast_utils.atmost_one(ast_utils.children_of_type(sub, (f03.Subroutine_Stmt, f03.Function_Stmt)))
    if stmt is None:
        return []
    dal = next(iter(walk(stmt, f03.Dummy_Arg_List)), None)
    if dal is not None:
        return [str(a) for a in dal.children]
    # a single dummy is a lone Name in the statement, not a Dummy_Arg_List.
    names = [str(n) for n in ast_utils.children_of_type(stmt, f03.Name)]
    return names[1:] if names else []  # drop the subprogram's own name


def _ptr_assoc_sites_by_dummy(scope: f03.Base, base: str) -> List[Tuple[str, str]]:
    """``(slot, dummy)`` for each ``<prefix>%<slot> => <dummy>`` in ``scope`` whose
    ``<dummy>`` is a plain-name ``CLASS(base)`` entity -- the pointer-association
    tag sources.  Detected by the (still-``CLASS``) dummy rather than the slot name,
    so it works after the slot component has been expanded away."""
    spec = ast_utils.atmost_one(ast_utils.children_of_type(scope, f03.Specification_Part))
    if spec is None:
        return []
    class_names = {n.lower() for _, ns in _class_locals(spec, base) for n in ns}
    if not class_names:
        return []
    out = []
    for pa in walk(scope, f03.Pointer_Assignment_Stmt):
        lhs, _, rhs = pa.children
        pt = _member_prefix_and_tail(lhs)
        if pt is None:
            continue
        if isinstance(rhs, f03.Name) and str(rhs).lower() in class_names:
            out.append((pt[1], str(rhs)))
    return out


def _arm_ancestor_rank(tinfos: dict, type_name: str) -> dict:
    """``{lower-cased type name: inheritance depth}`` for ``type_name`` (depth 0) and
    each of its ``EXTENDS`` ancestors (1, 2, ...).  Used to match a concrete type
    against a ``SELECT TYPE`` ``CLASS IS`` guard: an ancestor matches, the nearest
    (smallest depth) wins."""
    rank: dict = {}
    t, depth = type_name.lower(), 0
    while t and t not in rank:
        rank[t] = depth
        ti = tinfos.get(t)
        t = ti.parent if ti is not None else None
        depth += 1
    return rank


def _resolve_select_type_on_dummy(scope: f03.Base, dummy: str, arm_type: str, tinfos: dict) -> None:
    """Statically resolve every ``SELECT TYPE(dummy)`` in ``scope`` -- whose ``dummy``
    a per-arm clone retyped from ``CLASS(base)`` to the concrete ``TYPE(arm_type)`` --
    to its matching type-guard branch, in place.  A concrete selector is not
    polymorphic, so the construct cannot survive (flang rejects it).  Fortran's
    guard-matching picks ``TYPE IS (arm_type)`` (exact), else the nearest
    ``CLASS IS (ancestor)``, else ``CLASS DEFAULT`` (else the construct is dropped)."""
    rank = _arm_ancestor_rank(tinfos, arm_type)
    for cst in list(walk(scope, f03.Select_Type_Construct)):
        sel_stmt = cst.children[0]
        assoc, selector = sel_stmt.children
        if not (isinstance(selector, f03.Name) and str(selector).lower() == dummy.lower()):
            continue
        # split the flat construct into (guard, [body statements]) groups
        groups: List[Tuple[f03.Type_Guard_Stmt, List[f03.Base]]] = []
        for ch in cst.children[1:-1]:
            if isinstance(ch, f03.Type_Guard_Stmt):
                groups.append((ch, []))
            elif groups:
                groups[-1][1].append(ch)
        chosen: Optional[List[f03.Base]] = None
        default: Optional[List[f03.Base]] = None
        best: Optional[int] = None
        for guard, body in groups:
            kind = str(guard.children[0]).upper()
            gtype = str(guard.children[1]).lower() if guard.children[1] is not None else None
            if kind == 'CLASS DEFAULT':
                default = body
            elif kind == 'TYPE IS' and gtype == arm_type.lower():
                chosen, best = body, -1
                break
            elif kind == 'CLASS IS' and gtype in rank and (best is None or rank[gtype] < best):
                chosen, best = body, rank[gtype]
        body = chosen if chosen is not None else (default if default is not None else [])
        stmts = _parse_exec("\n".join(str(s) for s in body)) if body else []
        # `SELECT TYPE (s => dummy)` binds the branch body to `s`; the branch now runs
        # directly on the concrete dummy, so rename `s` -> dummy in the spliced body.
        if assoc is not None and stmts:
            aname = str(assoc).lower()
            for stmt in stmts:
                for nm in walk(stmt, f03.Name):
                    if str(nm).lower() == aname:
                        replace_node(nm, f03.Name(dummy))
        if stmts:
            replace_node(cst, stmts)
        else:
            remove_self(cst)


def _clone_ptr_constructor(sub: f03.Base, proc: str, base: str, arm_type: str, clone_name: str, dummy: str,
                           plan: MonomorphizationPlan, hybrid_slots: Set[str], tinfos: dict) -> f03.Base:
    """Clone constructor ``proc`` as ``clone_name`` with its ``CLASS(base)`` dummy
    ``dummy`` retyped to ``TYPE(arm_type)``, and each ``this%slot => dummy``
    association rewritten to set the tag + associate the concrete per-arm slot
    (``this%slot__tag = <tag>`` + ``this%slot__arm => dummy``).  A concrete dummy
    makes every ``dummy%binding`` call inside the body a static bind, so no further
    body edits are needed for the association's own use.

    A *hybrid* slot (kept ``CLASS`` for its data reads, in ``hybrid_slots``) KEEPS its
    original ``this%slot => dummy`` association alongside the tag + per-arm assoc, so
    the CLASS slot its data reads target is still bound (``TYPE(arm_type)`` dummy ->
    ``CLASS(base)`` slot is a valid pointer association)."""
    text = sub.tofortran()
    text = re.sub(r'\b' + re.escape(proc) + r'\b', clone_name, text)
    text = re.sub(r'CLASS\s*\(\s*' + re.escape(base) + r'\s*\)', f'TYPE({arm_type})', text, flags=re.IGNORECASE)
    clone = walk(parse_program(f"module zz_tmp\ncontains\n{text}\nend module\n"), SCOPES)[0]
    tag, arm = _find_arm(plan, arm_type)
    for pa in list(walk(clone, f03.Pointer_Assignment_Stmt)):
        lhs, _, rhs = pa.children
        if not (isinstance(rhs, f03.Name) and str(rhs).lower() == dummy.lower()):
            continue
        pt = _member_prefix_and_tail(lhs)
        if pt is None:
            continue
        prefix, slot = pt
        assoc = f"{prefix}%{_tag_var(slot)} = {tag}\n{prefix}%{_arm_slot(slot, arm)} => {rhs}"
        if slot.lower() in hybrid_slots:
            assoc = f"{prefix}%{slot} => {rhs}\n{assoc}"
        replace_node(pa, _parse_exec(assoc))
    # The retyped (now concrete) dummy is no longer polymorphic, so any
    # ``SELECT TYPE(dummy)`` in the body must be resolved to its matching guard
    # branch -- flang rejects a ``SELECT TYPE`` on a non-polymorphic selector.
    _resolve_select_type_on_dummy(clone, dummy, arm_type, tinfos)
    return clone


def _host_scope_specs(scope: f03.Base) -> List[f03.Specification_Part]:
    """Specification parts visible from ``scope`` by host association -- the scope's
    own, then each enclosing procedure's / module's, nearest first.  A concrete-arm
    typed entity passed as an actual, or a dispatch receiver, is often a *module*
    variable (ICON's ``free_sfc_solver`` / ``free_sfc_solver_trans_triv``), not a
    local of the constructor's caller, so a type lookup must climb into the host."""
    specs = []
    node = scope
    seen: Set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        sp = ast_utils.atmost_one(ast_utils.children_of_type(node, f03.Specification_Part))
        if sp is not None:
            specs.append(sp)
        node = _enclosing_of_type(node, SCOPES + (f03.Module, f03.Main_Program))
    return specs


def _lookup_entity_type(scope: f03.Base, name: str) -> Tuple[bool, Optional[Tuple[str, str]]]:
    """``(declared, (kind, type_name) | None)`` for ``name`` visible from ``scope``
    (own scope, then host association).  ``declared`` says whether any declaration
    was found; the pair is ``('TYPE'|'CLASS', type)`` for a derived-type
    declaration, else ``None`` (intrinsic).  The nearest scope wins -- a local
    shadows a host entity -- so the search stops at the first scope declaring it."""
    name_l = name.lower()
    for spec in _host_scope_specs(scope):
        for decl in walk(spec, f03.Type_Declaration_Stmt):
            if name_l not in {str(e.children[0]).lower() for e in walk(decl, f03.Entity_Decl)}:
                continue
            ts = decl.children[0]
            if isinstance(ts, f03.Declaration_Type_Spec) and ts.children[0] in ('TYPE', 'CLASS'):
                return True, (str(ts.children[0]), str(ts.children[1]))
            return True, None
    return False, None


def _entity_declared_type(scope: f03.Base, name: str) -> Optional[str]:
    """Declared derived-type name of ``name`` visible from ``scope`` for a
    ``TYPE(t)`` *or* ``CLASS(t)`` declaration (the type used to resolve a
    dispatch), ``None`` for an intrinsic type or a missing declaration."""
    _, info = _lookup_entity_type(scope, name)
    return info[1].lower() if info is not None else None


def _component_type(dtds: dict, type_name: str, comp: str) -> Optional[str]:
    """Derived-type name of component ``comp`` of ``type_name`` (searching its
    ``EXTENDS`` parents), ``None`` if absent or intrinsic."""
    dtd = dtds.get(type_name.lower())
    if dtd is None:
        return None
    for cdef in walk(dtd, f03.Data_Component_Def_Stmt):
        comps = {str(cd.children[0]).lower() for cd in walk(cdef, f03.Component_Decl)}
        if comp.lower() in comps:
            ts = cdef.children[0]
            if isinstance(ts, f03.Declaration_Type_Spec) and ts.children[0] in ('TYPE', 'CLASS'):
                return str(ts.children[1]).lower()
            return None
    parent = read_type_info(dtd).parent
    return _component_type(dtds, parent, comp) if parent else None


def _type_of_ref(scope: f03.Base, ref: f03.Base, dtds: dict) -> Optional[str]:
    """Derived-type name a reference resolves to: a bare ``Name`` (local/dummy) or a
    member path ``a%b%c`` (each component's type walked through ``dtds``)."""
    if isinstance(ref, f03.Name):
        return _entity_declared_type(scope, str(ref))
    if isinstance(ref, f03.Data_Ref):
        parts = list(ref.children)
        t = _entity_declared_type(scope, str(parts[0]))
        for p in parts[1:]:
            if t is None:
                return None
            t = _component_type(dtds, t, str(p))
        return t
    return None


def _binding_target(tinfos: dict, type_name: str, binding: str) -> Optional[str]:
    """Concrete procedure a static ``type_name%binding`` dispatch resolves to,
    walking ``EXTENDS`` for an overriding binding.  ``None`` if the binding is
    deferred at ``type_name`` (resolved only by the dynamic type -- not a static
    caller of any one arm) or absent."""
    t = type_name.lower()
    b = binding.lower()
    while t in tinfos:
        ti = tinfos[t]
        if b in ti.overrides:
            return ti.overrides[b]
        if b in ti.deferred:
            return None
        t = ti.parent
    return None


def _resolved_call_targets(program: f03.Program) -> Set[str]:
    """Lower-cased procedure names that some ``Call_Stmt`` resolves to -- a direct
    ``CALL proc`` or a statically-resolvable ``obj%binding`` dispatch.  A dispatch
    whose receiver type cannot be resolved is treated conservatively (every arm
    that overrides that binding is kept live) so a still-referenced constructor is
    never dropped out from under a caller."""
    tinfos = {read_type_info(d).name: read_type_info(d) for d in walk(program, f03.Derived_Type_Def)}
    dtds = {read_type_info(d).name: d for d in walk(program, f03.Derived_Type_Def)}
    targets: Set[str] = set()
    for call in walk(program, f03.Call_Stmt):
        des = call.children[0]
        if isinstance(des, f03.Name):
            targets.add(str(des).lower())
        elif isinstance(des, f03.Procedure_Designator):
            obj, _, binding = des.children
            b = str(binding).lower()
            scope = _enclosing_of_type(call, SCOPES)
            tn = _type_of_ref(scope, obj, dtds) if scope is not None else None
            tgt = _binding_target(tinfos, tn, b) if tn is not None else None
            if tgt is not None:
                targets.add(tgt)
            elif tn is None:
                # unresolved receiver: keep every overrider of this binding live.
                for ti in tinfos.values():
                    if b in ti.overrides:
                        targets.add(ti.overrides[b])
    return targets


def _drop_use_only_import(program: f03.Program, name: str) -> None:
    """Remove ``name`` from every ``USE mod, ONLY: ...`` import (dropping the whole
    statement when it was the sole imported name).  Called when a cloned constructor
    is deleted: an intermediate clone (an interposer specialised for the axis, later
    re-specialised and dropped) is imported at earlier redirect sites via
    :func:`_ensure_use`, and those imports would otherwise dangle."""
    name_l = name.lower()
    for use in list(walk(program, f03.Use_Stmt)):
        if 'ONLY' not in str(use).upper():
            continue
        names = [str(n) for n in walk(use, f03.Name)]
        imported = names[1:]  # names[0] is the module
        if name_l not in {n.lower() for n in imported}:
            continue
        keep = [n for n in imported if n.lower() != name_l]
        if keep:
            replace_node(use, f03.Use_Stmt(f"USE {names[0]}, ONLY: {', '.join(keep)}"))
        else:
            remove_self(use)


def _drop_dead_constructors(program: f03.Program, procs: Set[str]) -> None:
    """Remove each constructor in ``procs`` and its ``binding => proc`` TBPs once no
    call resolves to it, to a fixed point (a pass-through goes dead only after the
    caller that still dispatches it is dropped).  A cloned original still holds the
    pre-expansion ``this%slot => dummy`` -- referencing an expanded-away component --
    and its ``dummy%binding`` calls stay polymorphic, so leaving a dead one would
    fail to compile or reach the bridge's polymorphism reject."""
    remaining = set(procs)
    changed = True
    while changed and remaining:
        changed = False
        live = _resolved_call_targets(program)
        for proc in sorted(remaining):
            if proc in live:
                continue
            for b in [
                    b for b in walk(program, f03.Specific_Binding)
                    if b.children[4] is not None and str(b.children[4]).lower() == proc
            ]:
                remove_self(b)
            sub = _find_subprogram(program, proc)
            if sub is not None:
                remove_self(sub)
            _drop_use_only_import(program, proc)
            remaining.discard(proc)
            changed = True


def _callee_of(call: f03.Call_Stmt, scope: Optional[f03.Base], tinfos: dict,
               dtds: dict) -> Tuple[Optional[str], Optional[f03.Base], bool]:
    """``(proc_name, passed_object_or_None, has_passed_object)`` for a call: a
    plain ``CALL proc`` (no passed object), or a type-bound dispatch ``obj%binding``
    resolved *precisely* by ``obj``'s static type (:func:`_type_of_ref` +
    :func:`_binding_target`).  Precision matters: a binding name is shared across
    types (``bconstruct`` on every backend arm, ``construct`` on the solver, lhs
    and transfers), so a name-keyed map would mis-resolve a laddered dispatch on a
    concrete slot (``this%act__t_cg%bconstruct``) to the wrong arm's procedure."""
    des = call.children[0]
    if isinstance(des, f03.Name):
        return str(des).lower(), None, False
    if isinstance(des, f03.Procedure_Designator):
        obj, _, binding = des.children
        tn = _type_of_ref(scope, obj, dtds) if scope is not None else None
        proc = _binding_target(tinfos, tn, str(binding)) if tn is not None else None
        return proc, obj, True
    return None, None, False


def _dummy_is_dispatched(sub: f03.Base, dummy: str) -> bool:
    """True if ``sub`` dispatches on its dummy ``dummy`` -- a ``dummy%binding(...)``
    type-bound call (a :class:`Procedure_Designator` whose object is the bare dummy).
    A plain ``dummy%member`` data read is a :class:`Data_Ref`, not a dispatch, so a
    data-only ``CLASS(base)`` dummy (which compiles fine) is not cloned."""
    d = dummy.lower()
    for des in walk(sub, f03.Procedure_Designator):
        obj = des.children[0]
        if isinstance(obj, f03.Name) and str(obj).lower() == d:
            return True
    return False


def _class_base_dummies(sub: f03.Base, base: str) -> dict:
    """``{dummy_name: position}`` for each ``CLASS(base)`` *dummy argument* of
    ``sub`` -- the arguments a concrete arm can flow into to be devirtualised."""
    spec = ast_utils.atmost_one(ast_utils.children_of_type(sub, f03.Specification_Part))
    if spec is None:
        return {}
    class_names = {n.lower() for _, ns in _class_locals(spec, base) for n in ns}
    dummies = [d.lower() for d in _dummy_arg_names(sub)]
    return {d: i for i, d in enumerate(dummies) if d in class_names}


def devirtualize_pointer_flow(program: f03.Program, plan: MonomorphizationPlan) -> int:
    """Devirtualise a pointer-association tag source (see the section comment) by
    an interprocedural forward fixed point over concrete-arm argument flow.

    A concrete arm reaches a ``this%slot => dummy`` association several call hops
    from where its type is known: ICON seeds it with a typed local at the entry
    (``ocean_solve_construct(.., a_trivial_transfer)``), which forwards it through
    a *pass-through* constructor's ``CLASS(base)`` dummy, through the (already
    laddered) backend dispatch, into ``lhs_construct``'s dummy, and finally into
    ``this%trans => trans``.  Each round: find every call passing a concrete-arm
    actual into a ``CLASS(base)`` dummy, clone the callee with that dummy retyped
    to ``TYPE(arm)`` (so every ``dummy%binding`` inside is a static bind and any
    ``this%slot => dummy`` becomes a tag set + concrete association), and redirect
    the call.  Retyping the dummy makes the clone forward a *concrete* actual on
    its own out-edges, exposing the next hop -- so the fixed point walks the whole
    constructor chain.  Returns the number of ``(proc, arm)`` clones emitted."""
    base = plan.abstract_base
    # Only bases with a pointer-association tag source are devirtualised this way;
    # an ``ALLOCATE``-constructed axis (ICON's ``%act``) is handled entirely by the
    # component ladder, and running the flow fixed point on it would be a no-op at
    # best and clone unrelated call chains at worst.
    if not any(_ptr_assoc_sites_by_dummy(s, base) for s in walk(program, SCOPES)):
        return 0
    # Slots the component pass kept as CLASS (data-carrying hybrids) still appear as
    # CLASS(base) components; their ``this%slot => dummy`` association is preserved
    # (for the native data reads) in addition to the tag + per-arm pointer.
    hybrid_slots = _slot_component_names(program, base)
    cloned = {}  # (proc, arm) -> (clone_name, clone_mod)

    changed = True
    while changed:
        changed = False
        subs = {(find_name_of_node(s) or '').lower(): s for s in walk(program, SCOPES)}
        tinfos = {read_type_info(d).name: read_type_info(d) for d in walk(program, f03.Derived_Type_Def)}
        dtds = {read_type_info(d).name: d for d in walk(program, f03.Derived_Type_Def)}
        flows = []  # (call, obj, proc, arm, dummy, passed_object)
        for call in walk(program, f03.Call_Stmt):
            scope = _enclosing_of_type(call, SCOPES)
            proc, obj, passed_object = _callee_of(call, scope, tinfos, dtds)
            if proc is None or proc not in subs:
                continue
            cbd = _class_base_dummies(subs[proc], base)
            if not cbd:
                continue
            args = call.children[1]
            actuals = list(args.children) if args is not None else []
            for dummy, dummy_idx in cbd.items():
                # passed-object binds dummy 0 (default PASS): actual for dummy k is
                # at index k-1; a plain call has no passed object (index k).
                actual_idx = dummy_idx - 1 if passed_object else dummy_idx
                if actual_idx < 0 or actual_idx >= len(actuals):
                    continue
                actual = actuals[actual_idx]
                if not isinstance(actual, f03.Name) or scope is None:
                    continue
                arm = _entity_concrete_type(scope, str(actual))
                if arm is not None and _arm_index(plan, arm) is not None:
                    flows.append((call, obj, proc, arm, dummy, passed_object))

        for _, _, proc, arm, dummy, _ in flows:
            if (proc, arm) in cloned:
                continue
            sub = subs[proc]
            clone_name = _clone_name(proc, arm)
            clone = _clone_ptr_constructor(sub, proc, base, arm, clone_name, dummy, plan, hybrid_slots, tinfos)
            append_children(sub.parent, clone)
            cloned[(proc, arm)] = (clone_name, _module_name_of(sub))
            changed = True

        for call, obj, proc, arm, _, passed_object in flows:
            clone_name, clone_mod = cloned[(proc, arm)]
            call_sub = _enclosing_of_type(call, SCOPES)
            call_mod = _module_name_of(call)
            args = call.children[1]
            actuals = [str(a) for a in args.children] if args is not None else []
            callee = ', '.join(([str(obj)] if passed_object else []) + actuals)
            replace_node(call, _parse_exec(f"call {clone_name}({callee})"))
            if call_sub is not None and clone_mod is not None and call_mod != clone_mod:
                _ensure_use(call_sub, clone_mod, clone_name)

    # Drop the now-dead originals (their post-expansion ``this%slot => dummy`` is
    # invalid and their ``dummy%binding`` calls stay polymorphic).  Composed over
    # axes this also clears the previous axis's partial clones: after the ``agen``
    # flow redirects the entry to a ``(trans, agen)``-specialised clone, the
    # ``trans``-only clone the ``trans`` flow left is dead and dropped here.
    _drop_dead_constructors(program, {p for (p, _) in cloned})

    return len(cloned)


def devirtualize_dummy_dispatch(program: f03.Program, plan: MonomorphizationPlan) -> int:
    """Devirtualise a dispatch on a ``CLASS(base)`` DUMMY of a helper that is CALLED
    with a hybrid slot actual whose concrete arm is only known at runtime.

    The component ladder devirtualises a slot DISPATCH (``CALL this%trans%into(...)``)
    but leaves a slot PASS (``CALL restart(x, this%trans)``) on the kept CLASS slot, so
    a ``dummy%binding`` inside the helper stays polymorphic (ICON's
    ``ocean_restart_gmres(trans)%sync``).  For each call passing a hybrid slot into a
    ``CLASS(base)`` dummy the helper DISPATCHES on, clone the helper per arm with that
    dummy retyped to ``TYPE(arm)`` -- so every ``dummy%binding`` is a static bind -- and
    replace the call with a tag ladder routing each arm to its clone with the matching
    per-arm slot.  The direct concrete-arm-actual flow is handled by
    :func:`devirtualize_pointer_flow`; this is its runtime-tag sibling for the kept
    hybrid slot.  Returns the number of ``(proc, arm)`` clones emitted."""
    base = plan.abstract_base
    hybrid_slots = _slot_component_names(program, base)
    if not hybrid_slots:
        return 0
    subs = {(find_name_of_node(s) or '').lower(): s for s in walk(program, SCOPES)}
    tinfos = {read_type_info(d).name: read_type_info(d) for d in walk(program, f03.Derived_Type_Def)}
    dtds = {read_type_info(d).name: d for d in walk(program, f03.Derived_Type_Def)}

    # (call, proc, actual_idx, prefix, slot) for each hybrid-slot -> dispatched-dummy pass.
    sites = []
    for call in walk(program, f03.Call_Stmt):
        scope = _enclosing_of_type(call, SCOPES)
        proc, _, passed_object = _callee_of(call, scope, tinfos, dtds)
        if proc is None or proc not in subs:
            continue
        cbd = _class_base_dummies(subs[proc], base)
        if not cbd:
            continue
        args = call.children[1]
        actuals = list(args.children) if args is not None else []
        for dummy, dummy_idx in cbd.items():
            if not _dummy_is_dispatched(subs[proc], dummy):
                continue
            actual_idx = dummy_idx - 1 if passed_object else dummy_idx
            if actual_idx < 0 or actual_idx >= len(actuals):
                continue
            actual = actuals[actual_idx]
            if not isinstance(actual, f03.Data_Ref):
                continue  # a concrete-arm local Name is handled by devirtualize_pointer_flow
            prefix, tail = _ref_prefix_and_tail(actual)
            if tail.lower() not in hybrid_slots:
                continue
            sites.append((call, proc, actual_idx, prefix, tail))

    clones = {}  # (proc, arm) -> (clone_name, clone_mod)
    for _, proc, _, _, _ in sites:
        for arm in plan.arms:
            if (proc, arm.type_name) in clones:
                continue
            sub = subs[proc]
            clone_name = _clone_name(proc, arm.type_name)
            clones[(proc, arm.type_name)] = (clone_name, _module_name_of(sub))
            append_children(sub.parent, _clone_interposer(sub, proc, base, arm.type_name, clone_name))

    for call, proc, actual_idx, prefix, slot in sites:
        call_sub = _enclosing_of_type(call, SCOPES)
        call_mod = _module_name_of(call)
        args = call.children[1]
        actuals = [str(a) for a in args.children] if args is not None else []
        lines = []
        for tag, arm in enumerate(plan.arms, start=1):
            clone_name, clone_mod = clones[(proc, arm.type_name)]
            per = list(actuals)
            per[actual_idx] = f"{prefix}%{_arm_slot(slot, arm.type_name)}"
            lines.append(f"{'if' if tag == 1 else 'else if'} ({prefix}%{_tag_var(slot)} == {tag}) then")
            lines.append(f"  call {clone_name}({', '.join(per)})")
            if call_sub is not None and clone_mod is not None and call_mod != clone_mod:
                _ensure_use(call_sub, clone_mod, clone_name)
        lines.append("end if")
        replace_node(call, _parse_exec("\n".join(lines)))

    return len(clones)


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
    pointer_constructors_cloned: int = 0
    dummy_dispatch_cloned: int = 0


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

    ladder_plans = []
    for axis in ladder_axes:
        plan = plans.get(axis.base.lower())
        if plan is None:
            raise ValueError(f"ladder axis `{axis.base}` has no dispatch plan: the abstract base is "
                             f"absent, non-polymorphic, or has no concrete arm in this translation unit")
        ladder_plans.append(plan)
        stats.locals_rewritten += monomorphize_local_dispatch(program, plan, stack_slots)
        stats.components_rewritten += monomorphize_component_dispatch(program, plan, stack_slots)
        stats.interposers_cloned += clone_shared_interposers(program, plan)

    # Pointer-association devirtualisation runs as a second phase, after *every*
    # axis's slots are expanded and dispatches laddered.  The transfer/lhs flow
    # threads through the (now laddered) backend dispatch and through constructors
    # that also carry the other pointer axis's dummy; running it only once both are
    # in their expanded form keeps the interprocedural clones consistent.
    for plan in ladder_plans:
        stats.pointer_constructors_cloned += devirtualize_pointer_flow(program, plan)

    # A kept hybrid slot can also be PASSED to a helper that dispatches on its
    # CLASS(base) dummy (ICON's ocean_restart_gmres(trans)%sync) -- the runtime-tag
    # sibling of the pointer flow above.  Runs last, once every slot is expanded and
    # every construct-chain clone settled.
    for plan in ladder_plans:
        stats.dummy_dispatch_cloned += devirtualize_dummy_dispatch(program, plan)

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


def _entity_concrete_type(scope: f03.Base, name: str) -> Optional[str]:
    """The concrete ``TYPE(t)`` name declared for entity ``name`` visible from
    ``scope`` (a local, dummy, or host-associated *module* variable), or ``None``
    when it is polymorphic (``CLASS(...)``) or intrinsic.  Reads the arm type of a
    concrete actual passed to a pointer-assoc constructor (the ladder's tag
    source); ICON declares those actuals as module variables, so the lookup climbs
    into the host module (:func:`_lookup_entity_type`)."""
    _, info = _lookup_entity_type(scope, name)
    return str(info[1]) if (info is not None and info[0] == 'TYPE') else None


def _ptr_assoc_slot_sites(scope: f03.Base, base: str,
                          slot_names: Set[str]) -> List[Tuple[f03.Pointer_Assignment_Stmt, str, str]]:
    """``(pointer_assign_stmt, slot, dummy_name)`` for each ``<prefix>%<slot> =>
    <dummy>`` in ``scope`` where ``slot`` is one of ``slot_names`` and ``<dummy>``
    is a plain-name entity declared ``CLASS(base)`` in ``scope`` -- the interior
    of a constructor that pointer-associates an abstract slot to its (still
    abstract) dummy.  This is the pointer-association analogue of an
    ``ALLOCATE(arm :: slot)`` tag source, one interprocedural hop away: the
    concrete arm arrives as the actual argument bound to ``<dummy>``."""
    spec = ast_utils.atmost_one(ast_utils.children_of_type(scope, f03.Specification_Part))
    if spec is None:
        return []
    class_names = {n.lower() for _, ns in _class_locals(spec, base) for n in ns}
    if not class_names:
        return []
    out = []
    for pa in walk(scope, f03.Pointer_Assignment_Stmt):
        lhs, _, rhs = pa.children
        pt = _member_prefix_and_tail(lhs)
        if pt is None or pt[1].lower() not in slot_names:
            continue
        if isinstance(rhs, f03.Name) and str(rhs).lower() in class_names:
            out.append((pa, pt[1], str(rhs)))
    return out


def _slot_component_names(program: f03.Program, base: str) -> Set[str]:
    """Lower-cased names of every ``CLASS(base)`` derived-type *component* (the
    stored slots a dispatch reads as ``obj%slot%binding``)."""
    return {n.lower() for _, names in _component_slots(program, base) for n in names}


def _has_concrete_arm_actual(program: f03.Program, plan: MonomorphizationPlan) -> bool:
    """True if some call in the unit passes a concrete-arm-typed entity as an actual
    argument -- the seed that lets the pointer-association ladder resolve a non-zero
    tag (an ``ALLOCATE``-free axis whose arm never enters as a concrete actual could
    not be laddered without silently no-oping).  The seed is frequently a *module*
    variable (ICON's ``free_sfc_solver_trans_triv``), so the arm-typed names are
    collected program-wide (any scope) rather than per call-site."""
    arms = {a.type_name.lower() for a in plan.arms}
    arm_typed: Set[str] = set()
    for decl in walk(program, f03.Type_Declaration_Stmt):
        ts = decl.children[0]
        if isinstance(ts, f03.Declaration_Type_Spec) and ts.children[0] == 'TYPE' and str(
                ts.children[1]).lower() in arms:
            arm_typed |= {str(e.children[0]).lower() for e in walk(decl, f03.Entity_Decl)}
    if not arm_typed:
        return False
    for call in walk(program, f03.Call_Stmt):
        args = call.children[1]
        if args is not None and any(str(a).lower() in arm_typed for a in walk(args, f03.Name)):
            return True
    return False


def _has_pointer_assoc_construction(program: f03.Program, plan: MonomorphizationPlan) -> bool:
    """True if ``plan``'s arms are constructed by *pointer association* -- a
    ``<prefix>%<slot> => <class-dummy>`` inside a constructor, seeded by a
    concrete-arm typed actual somewhere in the unit.  This is the tag source for
    the pointer-association ladder, the interprocedural analogue of
    :func:`_has_inunit_construction`'s ``ALLOCATE(arm :: slot)``.  Requires BOTH
    an association site AND a concrete-arm actual so the ladder is guaranteed a
    resolvable, non-zero tag rather than a silent no-op."""
    base = plan.abstract_base
    slot_names = _slot_component_names(program, base)
    if not slot_names:
        return False
    if not any(_ptr_assoc_slot_sites(scope, base, slot_names) for scope in walk(program, SCOPES)):
        return False
    return _has_concrete_arm_actual(program, plan)


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
      * two or more arms with a tag source -> :data:`LADDER`.  The tag is set
        either at an ``ALLOCATE(arm :: slot)`` site (:func:`_has_inunit_construction`)
        or, one interprocedural hop away, at a ``slot => dummy`` pointer
        association whose dummy receives a concrete-arm actual
        (:func:`_has_pointer_assoc_construction` -- ICON's ``t_lhs%trans`` /
        ``%agen``, bound in ``lhs_construct`` from a typed local).
      * two or more arms, no tag source -> *skipped* (logged): the runtime arm
        cannot be inferred, so neither retype (which one?) nor a ladder (no tag)
        is sound.  An explicit :class:`MonomorphizationSpec` must pin it.

    A dispatch root outside the soundly-monomorphisable class (multi-level,
    ``CLASS(*)`` in its own definition, a missing override) is skipped too --
    :func:`analyze` rejects it and the rejection is swallowed per axis, leaving
    it for downstream externalisation / the bridge's polymorphism reject."""
    live = _dispatch_binding_names(program)
    if not live:
        return []

    # A generic dispatch (``obj%apply(...)``) invokes the generic's underlying
    # specific binding (``GENERIC :: apply => lhs_wp``), so a live generic name
    # makes its specific(s) live too.  Without this an abstract base dispatched
    # ONLY through generics -- ICON's ``t_lhs_agen``, whose deferred ``lhs_wp`` /
    # ``lhs_matrix_shortcut`` are reached solely as ``apply`` / ``matrix_shortcut``
    # -- would never be discovered (its deferred names never appear in a dispatch).
    for gb in walk(program, f03.Generic_Binding):
        _, gname, plist = gb.children
        if str(gname).lower() in live and plist is not None:
            live |= {str(s).lower() for s in plist.children}

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
        elif _has_inunit_construction(program, plan) or _has_pointer_assoc_construction(program, plan):
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


def _module_scope_declared_names(mod: f03.Module) -> Set[str]:
    """Lower-cased names DECLARED at ``mod``'s own specification-part scope (module
    variables / named constants).  Names introduced only by ``USE`` (imports) or
    inside a contained procedure are excluded -- they are not module-scope
    definitions and would not clash as redefinitions on a merge."""
    spec = ast_utils.atmost_one(ast_utils.children_of_type(mod, f03.Specification_Part))
    if spec is None:
        return set()
    return {
        str(ent.children[0]).lower()
        for decl in ast_utils.children_of_type(spec, f03.Type_Declaration_Stmt)
        for ent in walk(decl, f03.Entity_Decl)
    }


def _rename_module_scope_entity(mod: f03.Module, old: str, new: str) -> None:
    """Rename a module-scope entity ``old`` -> ``new`` throughout ``mod`` (its
    declaration and every reference), skipping any contained procedure that
    declares its own ``old`` (a local shadow that binds to a different entity)."""
    old_l = old.lower()
    shadow = set()
    for sub in walk(mod, SCOPES):
        sp = ast_utils.atmost_one(ast_utils.children_of_type(sub, f03.Specification_Part))
        if sp is not None and any(
                str(ent.children[0]).lower() == old_l
                for decl in ast_utils.children_of_type(sp, f03.Type_Declaration_Stmt)
                for ent in walk(decl, f03.Entity_Decl)):
            shadow.add(id(sub))
    for nm in list(walk(mod, f03.Name)):
        if str(nm).lower() != old_l:
            continue
        enc = _enclosing_of_type(nm, SCOPES)
        if enc is not None and id(enc) in shadow:
            continue
        replace_node(nm, f03.Name(new))


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

    # 0. Rename arm module-scope entities that collide with a base module-scope
    #    name before merging -- else the merge creates two definitions of one
    #    identifier in the base module.  A multi-arm ladder consolidates several
    #    near-identical sibling modules (ICON's Krylov backends), each carrying the
    #    same private ``this_mod_name`` PARAMETER; without this the base ends up
    #    with N ``this_mod_name`` definitions.
    for nm in sorted(_module_scope_declared_names(arm_mod) & _module_scope_declared_names(base_mod)):
        _rename_module_scope_entity(arm_mod, nm, f"{nm}__{arm_name}")

    # 1. Move the arm spec's declarations into the base spec.  A ``USE`` must
    #    precede every declaration in a specification part, so imported symbols
    #    are PREPENDED (ahead of the base's already-present type defs) while
    #    everything else is appended.  Appending a moved ``USE`` after the base's
    #    type definitions is illegal Fortran and -- worse -- makes ``alias_specs``
    #    miss the import: the arm's host-associated types (ICON's
    #    ``t_subset_range``, re-exported through ``mo_grid_subset``) then fail to
    #    resolve in the moved subprograms and get dropped from their ``USE``s, so
    #    the emitted TU references a type it never imports ("used before defined").
    if arm_spec is not None:
        for child in list(arm_spec.children):
            if isinstance(child, f03.Implicit_Part):
                continue
            if isinstance(child, f03.Use_Stmt):
                nm = ast_utils.atmost_one(ast_utils.children_of_type(child, f03.Name))
                if nm is not None and str(nm).lower() in (base_name, arm_name):
                    continue  # base symbols are already in scope post-merge
                remove_self(child)
                prepend_children(base_spec, child)
                continue
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
    # A ``USE base`` now sitting *inside* the base module -- e.g. an ``_ensure_use``
    # that imported a clone's callee from the base while the caller still lived in
    # a (now-merged) arm module -- is a self-use once both are in the base, which
    # a compiler rejects ("cannot USE a module currently being built").  Drop every
    # such self-use (at module scope and in the contained procedures).
    for use in list(walk(base_mod, f03.Use_Stmt)):
        nm = ast_utils.atmost_one(ast_utils.children_of_type(use, f03.Name))
        if nm is not None and str(nm).lower() == base_name:
            remove_self(use)
    _toposort_type_defs(base_spec)

    # 4. The retype may have rewritten ``CLASS(base)`` -> ``TYPE(arm)`` anywhere in
    #    OTHER modules: in a halo wrapper's dummy (a third module's subprogram) AND
    #    in a type COMPONENT (ICON's ``t_patch%comm_pat_*`` in ``mo_model_domain``).
    #    Those now name the arm type but may not import it -- it lives in the base
    #    module post-merge.  Import it at the MODULE level (host association then
    #    covers the module's subprograms too).
    arm_l = arm_type.lower()
    for mod in walk(program, f03.Module):
        if _module_name(mod) == base_name:
            continue
        if any(
                len(ts.children) > 1 and str(ts.children[1]).lower() == arm_l
                for ts in walk(mod, f03.Declaration_Type_Spec)):
            _ensure_use(mod, base_name, arm_type)
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
    # Capture each ladder axis's concrete arm types from the pristine hierarchy,
    # before the rewrite expands the component slot and drops the abstract
    # bindings.  A ladder emits per-arm interposer clones AND direct per-arm
    # binding calls into the base module (the dispatch site), so every arm's
    # module must be consolidated into the base's the same way a retype's single
    # arm is -- otherwise the base module names concrete arm types / procedures
    # that live in formerly-downstream modules ("used before defined" + a
    # circular USE, since each arm module already USEs the base to EXTEND it).
    ladder_arms = {}
    for axis in axes:
        if axis.strategy == LADDER:
            plans = analyze(program, only_bases=[axis.base])
            if plans:
                ladder_arms[axis.base] = [arm.type_name for arm in plans[0].arms]
    stats = monomorphize(program, MonomorphizationSpec(axes), stack_slots=stack_slots)
    # Co-locate each arm's definition with its base module so a retyped container
    # (RETYPE) or a per-arm interposer clone / direct binding call (LADDER) placed
    # in the base module doesn't depend circularly on the (formerly downstream)
    # arm module.
    merged = False
    for axis in axes:
        arms = [axis.concrete] if axis.strategy == RETYPE and axis.concrete is not None else ladder_arms.get(
            axis.base, [])
        for arm in arms:
            if consolidate_arm_module(program, axis.base, arm):
                logger.debug("monomorphize: consolidated arm `%s` into base `%s`'s module", arm, axis.base)
                merged = True
    if merged:
        # A merge folds the arm module's USE dependencies into the base module,
        # so the base may now precede a module it newly depends on (e.g. the arm's
        # ``USE mo_grid_subset`` for a host-associated type).  ``alias_specs``
        # resolves USEs in document order and assumes the modules are topologically
        # sorted; an out-of-order base leaves its inherited imports unresolved, so
        # a re-exported host type (ICON's ``t_subset_range``, defined in
        # ``mo_model_domain`` and re-exported via ``mo_grid_subset``) is dropped
        # from the moved subprograms' ``USE``s ("used before defined").  Restore the
        # topological module order the downstream resolution passes rely on.
        keep_sorted_used_modules(program)
    return stats
