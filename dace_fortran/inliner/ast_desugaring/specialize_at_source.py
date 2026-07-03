"""Specialize a *named* set of subprograms to their call sites by source-level
inlining -- so call-site-constant arguments fold the body before the bridge lowers
it.

WHY AT THE SOURCE (and not HLFIR inline-all).  The bridge already inlines the whole
call graph into the entry via ``hlfir-inline-all``; for almost everything that is
sufficient.  But a few wrappers carry a construct the bridge cannot lower while one
of their arguments is still symbolic -- notably ICON's halo ``sync_patch_array``
family, whose ``IF (typ==N) p_pat => p_patch%comm_pat_<X>`` ladder is a
runtime-selected pointer rebind.  HLFIR inlining is TOO LATE: the ladder is folded
only AFTER ``hlfir-rewrite-pointer-assigns`` has already rejected the
runtime-selected rebind.  Inlining at the source instead lets each call site's
compile-time-constant ``typ`` flow into the body, and the existing
``const_eval_nodes`` + ``prune_branches`` collapse ``IF (1==1) ... ELSE IF (1==2)
...`` to the single live arm -- leaving one ``p_pat => p_patch%comm_pat_c`` rebind,
an ordinary single-source View the bridge lowers.  This is effectively per-call-site
MONOMORPHIZATION (specialization on the constant argument) done by inlining -- hence
the name.

Scope is deliberately the named set only: a general body inliner has broad blast
radius, and everything else lowers fine through HLFIR inline-all.  Calls the splice
cannot express soundly (an OMITTED argument that is not OPTIONAL, or an absent
optional used in an unmodelled way) are left untouched for the caller's downstream
passes to handle -- never silently miscompiled.

Entry point: :func:`specialize_at_source` (runs the subprogram-call + function-ref
inliners to a fixpoint).
"""
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import logging

from fparser.two import Fortran2003 as f03
from fparser.two.utils import walk

from dace_fortran.inliner.ast_desugaring import analysis, pruning, utils
from dace_fortran.inliner.ast_desugaring.monomorphize import parse_program
from dace_fortran.inliner.ast_utils import children_of_type

logger = logging.getLogger(__name__)

#: A Fortran identifier as a whole word (so ``arr`` does not match inside
#: ``arr_g``).  Fortran identifiers are ``[A-Za-z][A-Za-z0-9_]*``; the lookarounds
#: keep us off the middle of a longer identifier.
_IDENT = r"(?<![A-Za-z0-9_])({names})(?![A-Za-z0-9_])"

#: A loop guard: how many inline rounds before we assume a cycle and stop.
_MAX_ROUNDS = 256

#: Marker substituted for an OMITTED optional dummy.  After ``PRESENT`` / ``SIZE``
#: folding and forwarded-keyword dropping, no live occurrence should remain -- if
#: one does the inline is abandoned (an absent optional used in a context we do
#: not model) rather than emitting an undeclared name.
_ABSENT = "f2dace_absent_arg"


def _subprogram_of(stmt: f03.Base) -> Optional[f03.Base]:
    """The ``Subroutine_Subprogram`` / ``Function_Subprogram`` enclosing a
    Subroutine_Stmt / Function_Stmt (its parent), or None."""
    par = stmt.parent
    if isinstance(par, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        return par
    return None


def _dummy_arg_names(sub_stmt: f03.Base) -> List[str]:
    """The dummy-argument names of a Subroutine_Stmt / Function_Stmt, in order."""
    dal = next(children_of_type(sub_stmt, (f03.Dummy_Arg_List, f03.Dummy_Arg_Name_List)), None)
    if dal is None:
        return []
    return [str(c) for c in dal.children]


def _entity_names(spec: f03.Base) -> Set[str]:
    """Every entity name DECLARED in a Specification_Part (lower-cased)."""
    names: Set[str] = set()
    for ed in walk(spec, (f03.Entity_Decl, f03.Component_Decl)):
        nm = next(children_of_type(ed, f03.Name), None)
        if nm is not None:
            names.add(str(nm).lower())
    return names


#: A keyword-argument NAME position: ``(kw =`` or ``, kw =`` (but not ``==``).  The
#: keyword of a nested call's actual-argument spec is the CALLEE's parameter name,
#: not a reference -- so it must NOT be substituted even when it collides with a
#: dummy being inlined (``CALL mixprec(typ = typ, lacc = .TRUE.)`` -- the left
#: ``typ`` / ``lacc`` are mixprec's parameter names and stay).
_KW_BEFORE = re.compile(r"[(,]\s*$")
_KW_AFTER = re.compile(r"^\s*=(?!=)")

#: A struct-COMPONENT / type-bound selector position (``p % comp`` -- including the
#: inner names of a chain ``a % b % comp``).  A name right after ``%`` is a
#: derived-type component or bound-procedure name, never a reference to a local or
#: dummy being inlined, so it must NOT be substituted: renaming ``p_pat % n_send``
#: to ``p_pat % n_send_inlN`` (because the callee has a local ``n_send``) invents a
#: non-existent component and breaks downstream component-spec resolution.
_COMP_BEFORE = re.compile(r"%\s*$")


def _subst(text: str, mapping: Dict[str, str]) -> str:
    """Replace every whole-word occurrence of a key (case-insensitive) with its
    value, in a single pass so substitutions never cascade into one another.

    Two textual positions are left alone (the name there is a callee-side selector,
    not a reference to the name being inlined): a keyword-argument NAME (``(name =``
    / ``, name =``) and a struct-component selector (``... % name``).  fparser
    serialises each statement on one line, so these checks are reliable on the body
    text we operate on.
    """
    if not mapping:
        return text
    lower = {k.lower(): v for k, v in mapping.items()}
    pat = re.compile(_IDENT.format(names="|".join(re.escape(k) for k in lower)), re.IGNORECASE)

    def repl(m: "re.Match") -> str:
        before = text[max(0, m.start() - 16):m.start()]
        if _KW_BEFORE.search(before) and _KW_AFTER.match(text[m.end():m.end() + 4]):
            return m.group(1)  # keyword-argument name: preserve verbatim
        if _COMP_BEFORE.search(before):
            return m.group(1)  # struct component / type-bound name after ``%``
        return lower[m.group(1).lower()]

    return pat.sub(repl, text)


#: ``OPTIONAL`` as a whole attribute word in a declaration's attribute list.
_OPTIONAL_ATTR = re.compile(r"(?i)(?:^|,)\s*OPTIONAL\s*(?:,|$)")


def _optional_dummy_names(callee_spec: Optional[f03.Base], dummies: Set[str]) -> Set[str]:
    """The subset of ``dummies`` declared ``OPTIONAL`` in the callee spec.

    The attribute is matched from the declaration text (the attribute list left of
    ``::``) rather than via ``isinstance(child, Attr_Spec_List)`` -- fparser's
    dynamically-built ``Attr_Spec_List`` class identity does not always match a
    statically-imported reference, so an ``isinstance`` filter silently misses it.
    """
    opt: Set[str] = set()
    if callee_spec is None:
        return opt
    for decl in children_of_type(callee_spec, f03.Type_Declaration_Stmt):
        lhs, sep, _ = str(decl).partition("::")
        if not sep or not _OPTIONAL_ATTR.search(lhs):
            continue
        for ed in walk(decl, f03.Entity_Decl):
            nm = next(iter(children_of_type(ed, f03.Name)), None)
            if nm is not None and str(nm).lower() in dummies:
                opt.add(str(nm).lower())
    return opt


def _bind_actuals_to_dummies(call: f03.Call_Stmt, dummies: List[str],
                             optionals: Set[str]) -> Optional[Tuple[Dict[str, str], Set[str]]]:
    """Map each dummy to its actual argument text at ``call``, classifying omitted
    optionals as ABSENT.

    Returns ``(present, absent)`` -- ``present`` maps a dummy to its actual source
    text, ``absent`` is the set of OMITTED optional dummies.  An actual that is
    itself the absent marker (a forwarded already-omitted optional from an earlier
    inline round) classifies the dummy ABSENT too.  Returns None when a NON-optional
    dummy is unsupplied (a call we cannot soundly inline).
    """
    _, arglist = call.children
    actuals = list(arglist.children) if arglist is not None else []
    present: Dict[str, str] = {}
    pos = 0
    for a in actuals:
        if isinstance(a, f03.Actual_Arg_Spec):
            kw, val = a.children
            present[str(kw).lower()] = str(val)
        else:
            if pos >= len(dummies):
                return None
            present[dummies[pos].lower()] = str(a)
            pos += 1
    absent: Set[str] = set()
    for d in dummies:
        dl = d.lower()
        if dl not in present:
            if dl not in optionals:
                return None  # a non-optional dummy left out: cannot inline
            absent.add(dl)
        elif present[dl].strip().lower() == _ABSENT:
            # A forwarded optional that was itself omitted upstream.
            del present[dl]
            absent.add(dl)
    return present, absent


def _fold_optionals(text: str, present: Set[str], absent: Set[str]) -> str:
    """Resolve ``PRESENT`` / ``SIZE`` queries on dummies whose presence is now
    statically known, BEFORE the dummy names are substituted away.

    ``PRESENT(x)`` -> ``.TRUE.`` / ``.FALSE.``; ``SIZE(absent, ...)`` -> ``0`` (an
    absent field contributes nothing to a dimension accumulation).  Sound: only
    fires on a name known present or known absent at this call.
    """
    for d in present:
        text = re.sub(r"(?i)\bPRESENT\s*\(\s*" + re.escape(d) + r"\s*\)", ".TRUE.", text)
    for d in absent:
        text = re.sub(r"(?i)\bPRESENT\s*\(\s*" + re.escape(d) + r"\s*\)", ".FALSE.", text)
        # SIZE(absent) / SIZE(absent, dim) -> 0 ; UBOUND/LBOUND likewise unused.
        text = re.sub(r"(?i)\bSIZE\s*\(\s*" + re.escape(d) + r"\s*(,[^()]*)?\)", "0", text)
    return text


def _drop_absent_actuals(text: str) -> str:
    """Drop ``keyword = <absent>`` actual arguments from CALL/reference arg lists,
    cleaning up the surrounding commas.  Run after absent dummies are substituted
    to the absent marker."""
    s = _ABSENT
    text = re.sub(r",\s*\w+\s*=\s*" + s + r"\b", "", text)  # ", kw=absent" (not first)
    text = re.sub(r"\b\w+\s*=\s*" + s + r"\s*,\s*", "", text)  # "kw=absent, " (first)
    text = re.sub(r"\(\s*\w+\s*=\s*" + s + r"\s*\)", "()", text)  # sole arg
    return text


def _drop_calls_passing_absent(text: str) -> str:
    """Drop whole ``CALL`` statements that still pass the absent marker after the
    keyword-actual cleanup -- i.e. a statically-OMITTED optional passed POSITIONALLY
    (which ``_drop_absent_actuals`` cannot remove without shifting the other
    positionals).

    Sound for THIS specialization: passing an absent optional positionally to a
    non-optional dummy is invalid Fortran, so such a call can only be reached when
    the field is present -- and in this inlined copy the field is statically absent,
    so the call is dead.  (ICON's debug ``check_patch_array_3d_dp(typ, p_patch,
    f3dinN_dp, ...)`` over a field count larger than the present set.)  Operates
    line-wise: fparser serialises one statement per line.
    """
    if _ABSENT not in text:
        return text
    kept = []
    for line in text.splitlines():
        if _ABSENT in line and re.search(r"\bCALL\b", line, re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept)


def _carry_uses(callee_sub: f03.Base) -> List[str]:
    """Source text of the USE statements the spliced body needs to keep resolving in
    the caller (a DIFFERENT module):

      * the callee subprogram's own USEs (its imported types / externals);
      * its enclosing module's USEs (symbols reached by host association, e.g. a
        ``t_comm_pattern`` the module imports once for every contained procedure);
      * a whole-module ``USE <callee_module>`` -- so the body's references to the
        callee's MODULE-LEVEL siblings (the debug ``check_patch_array_3d_dp`` it
        calls, the ``do_sync_checks`` module variable it reads) still resolve once
        the body lands in the caller; without it gfortran reports "no IMPLICIT
        type" / "requires explicit interface".

    Deduplicated by the caller-side merge."""
    uses: List[str] = []
    spec = next(iter(children_of_type(callee_sub, f03.Specification_Part)), None)
    if spec is not None:
        uses.extend(str(u) for u in children_of_type(spec, f03.Use_Stmt))
    mod = callee_sub.parent
    while mod is not None and not isinstance(mod, f03.Module):
        mod = mod.parent
    if mod is not None:
        mspec = next(iter(children_of_type(mod, f03.Specification_Part)), None)
        if mspec is not None:
            uses.extend(str(u) for u in children_of_type(mspec, f03.Use_Stmt))
        mstmt = next(iter(children_of_type(mod, f03.Module_Stmt)), None)
        modname = str(mstmt.children[1]) if mstmt is not None else None
        if modname:
            uses.append(f"USE {modname}")
    return uses


def _local_decls(callee_spec: Optional[f03.Base], dummies: Set[str]) -> List[f03.Base]:
    """The callee's NON-dummy local Type_Declaration_Stmt nodes (the ones the
    spliced body needs declared in the caller)."""
    if callee_spec is None:
        return []
    out: List[f03.Base] = []
    for decl in children_of_type(callee_spec, f03.Type_Declaration_Stmt):
        decl_names = {str(n).lower() for ed in walk(decl, f03.Entity_Decl) for n in children_of_type(ed, f03.Name)}
        if decl_names and decl_names.isdisjoint(dummies):
            out.append(decl)
    return out


#: A whole-line ``<bare-name> = .TRUE./.FALSE.`` assignment (fparser serialises one
#: statement per line, so an anchored line match is reliable on the body text).
_LOGICAL_LIT_ASSIGN = re.compile(r"(?im)^[ \t]*([A-Za-z]\w*)[ \t]*=[ \t]*(\.TRUE\.|\.FALSE\.)[ \t]*$")


def _fold_logical_literal_locals(exec_text: str) -> str:
    """Propagate a straight-line, SINGLE-assignment ``<local> = .TRUE./.FALSE.`` into
    the local's other occurrences and drop the assignment, so a guard written through
    the local (``lsend = PRESENT(send)`` -> ``lsend = .TRUE.`` then ``IF (lsend)``)
    folds to a literal condition the branch-pruner can resolve.

    Conservative: only a bare-Name LHS assigned a logical literal, and only when that
    name is assigned EXACTLY ONCE in the fragment (so the value is unambiguous on every
    use -- a reassigned local is left alone).  The local's declaration stays
    (harmlessly unused).  Alias-map-free, so it runs on a fragment whose ``USE``d
    modules are absent.
    """
    counts: Dict[str, int] = {}
    lit: Dict[str, str] = {}
    for m in _LOGICAL_LIT_ASSIGN.finditer(exec_text):
        nm = m.group(1).lower()
        counts[nm] = counts.get(nm, 0) + 1
        lit[nm] = m.group(2)
    const = {nm: lit[nm] for nm, c in counts.items() if c == 1}
    if not const:
        return exec_text
    kept = [
        line for line in exec_text.splitlines()
        if not ((m := _LOGICAL_LIT_ASSIGN.match(line)) and m.group(1).lower() in const)
    ]
    return _subst("\n".join(kept), const)


def _reparse_fragment(uses: List[str], decls: List[str], exec_text: str) -> Tuple[List[f03.Base], List[f03.Base]]:
    """Reparse a substituted body fragment (carried USEs + renamed local decls +
    substituted executable text) and return ``(spec_children, exec_children)``."""
    # An omitted optional makes ``PRESENT(it)`` fold to ``.FALSE.`` and ``PRESENT`` of
    # a supplied one to ``.TRUE.`` (done in ``_fold_optionals`` before substitution).
    # Propagate any local assigned such a folded literal (``lsend = PRESENT(send)`` ->
    # ``lsend = .TRUE.``) into its later uses so a guard written THROUGH the local
    # (``IF (lsend) ... ELSE <recv-only, uses absent send> END IF``) becomes a literal
    # condition the branch-pruner below can resolve -- otherwise the dead arm's
    # ``_ABSENT`` marker would (over-conservatively) abandon the inline.  Done on the
    # exec text (the decls keep the local declared, harmlessly unused) and
    # alias-map-free, since the callee's ``USE``d modules are absent in this fragment
    # so the full const-propagation pass cannot run; only straight-line
    # single-assignment logical locals fold.  Runtime guards
    # (``my_process_is_mpi_parallel()``) and the not-yet-folded ``typ`` ladder are
    # left for the downstream const-eval/prune loop.
    exec_text = _fold_logical_literal_locals(exec_text)
    # Fortran statement order: USE, then IMPLICIT, then declarations, then body.
    head = "\n".join(uses + ["implicit none"] + decls)
    text = f"module zz_inl_m\ncontains\nsubroutine zz_inl_s\n{head}\n{exec_text}\nend subroutine\nend module\n"
    prog = parse_program(text)
    pruning.prune_branches(prog, alias_map={})
    sub = next(iter(walk(prog, (f03.Subroutine_Subprogram, ))))
    spec = next(iter(children_of_type(sub, f03.Specification_Part)), None)
    expart = next(iter(children_of_type(sub, f03.Execution_Part)), None)
    spec_children = list(spec.children) if spec is not None else []
    exec_children = list(expart.children) if expart is not None else []
    return spec_children, exec_children


def _fragment_has_absent(frag_spec: List[f03.Base], frag_exec: List[f03.Base]) -> bool:
    """True when the ``_ABSENT`` marker survives the (already dead-branch-pruned)
    fragment -- a genuine use of an omitted optional in LIVE code that this
    specialization cannot express, so the inline must be abandoned rather than emit an
    undeclared name."""
    return any(_ABSENT in str(c) for c in frag_exec) or any(_ABSENT in str(c) for c in frag_spec)


def _enclosing_module_name(node: f03.Base) -> Optional[str]:
    """Lower-cased name of the MODULE lexically containing ``node`` (None if it is
    a free program unit)."""
    mod = node.parent
    while mod is not None and not isinstance(mod, f03.Module):
        mod = mod.parent
    if mod is None:
        return None
    mstmt = next(iter(children_of_type(mod, f03.Module_Stmt)), None)
    return str(mstmt.children[1]).lower() if mstmt is not None else None


def _merge_into_caller_spec(caller_spec: f03.Base, frag_spec: List[f03.Base]) -> None:
    """Merge the fragment's USE + declaration statements into the caller's
    Specification_Part: USEs prepended (deduplicated by text), decls appended.

    A ``USE <m>`` is skipped when the caller already LIVES in module ``<m>`` -- the
    carried whole-module USE (for the callee's host-associated siblings) is a
    self-USE there and the symbols are already host-associated, so it is both
    invalid ("module cannot USE itself") and unnecessary."""
    own_module = _enclosing_module_name(caller_spec)
    existing = {str(c).strip().lower() for c in caller_spec.children}
    for node in frag_spec:
        if str(node).strip().lower() in existing:
            continue
        if isinstance(node, f03.Use_Stmt):
            used = next(iter(children_of_type(node, f03.Name)), None)
            if own_module is not None and used is not None and str(used).lower() == own_module:
                continue  # self-USE: the caller is already inside this module
            utils.prepend_children(caller_spec, node)
        elif isinstance(node, f03.Type_Declaration_Stmt):
            utils.append_children(caller_spec, node)
        # The ``implicit none`` we injected into the parse harness (and anything
        # else) is dropped: the caller already has its own implicit policy.
        existing.add(str(node).strip().lower())


def _inline_one_call(call: f03.Call_Stmt, callee_sub: f03.Base, counter: int) -> bool:
    """Splice ``callee_sub``'s body into ``call``'s site with the actuals bound.
    Returns True on success, False when the call shape is left untouched."""
    callee_stmt = next(iter(children_of_type(callee_sub, (f03.Subroutine_Stmt, ))), None)
    if callee_stmt is None:
        return False
    dummies = _dummy_arg_names(callee_stmt)
    callee_spec0 = next(iter(children_of_type(callee_sub, f03.Specification_Part)), None)
    optionals = _optional_dummy_names(callee_spec0, {d.lower() for d in dummies})
    binding = _bind_actuals_to_dummies(call, dummies, optionals)
    if binding is None:
        return False
    bound, absent = binding

    caller_sub = call
    while caller_sub is not None and not isinstance(
            caller_sub, (f03.Subroutine_Subprogram, f03.Function_Subprogram, f03.Main_Program)):
        caller_sub = caller_sub.parent
    if caller_sub is None:
        return False
    caller_spec = next(iter(children_of_type(caller_sub, f03.Specification_Part)), None)
    if caller_spec is None:
        return False

    callee_spec = next(iter(children_of_type(callee_sub, f03.Specification_Part)), None)
    callee_exec = next(iter(children_of_type(callee_sub, f03.Execution_Part)), None)
    if callee_exec is None:
        # Nothing to inline (a stub body); just drop the call.
        utils.remove_self(call)
        return True

    dummy_set = {d.lower() for d in dummies}
    local_decls = _local_decls(callee_spec, dummy_set)
    # Rename every callee local uniquely so it cannot clash with a caller name.
    rename: Dict[str, str] = {}
    for decl in local_decls:
        for ed in walk(decl, f03.Entity_Decl):
            nm = next(iter(children_of_type(ed, f03.Name)), None)
            if nm is not None:
                rename[str(nm)] = f"{str(nm)}_inl{counter}"

    # Resolve PRESENT / SIZE queries on now-known optional dummies BEFORE the
    # names are substituted away.
    exec_text = _fold_optionals(str(callee_exec), set(bound), absent)

    # One combined substitution: present dummies -> actual text, omitted optionals
    # -> the absent marker, locals -> unique names.
    mapping: Dict[str, str] = dict(bound)
    mapping.update({d: _ABSENT for d in absent})
    mapping.update(rename)
    exec_text = _subst(exec_text, mapping)

    # Drop forwarded ``keyword = <absent>`` actuals and any whole CALL that passes the
    # absent marker positionally (a statically-omitted optional -> dead call here).
    exec_text = _drop_calls_passing_absent(_drop_absent_actuals(exec_text))

    decl_texts = [_subst(str(d), rename) for d in local_decls]
    uses = _carry_uses(callee_sub)

    # Reparse (which also prunes the now-dead PRESENT(absent) branches).  A marker that
    # STILL survives is a use of an omitted optional in live code we cannot express:
    # abandon the inline rather than emit an undeclared name.
    frag_spec, frag_exec = _reparse_fragment(uses, decl_texts, exec_text)
    if _fragment_has_absent(frag_spec, frag_exec):
        return False
    if not frag_exec:
        utils.remove_self(call)
        return True

    _merge_into_caller_spec(caller_spec, frag_spec)
    utils.replace_node(call, frag_exec)
    return True


def _function_result_name(func_stmt: f03.Function_Stmt) -> str:
    """The result variable name of a Function_Stmt: the ``RESULT(name)`` suffix
    when present, else the function's own name."""
    suffix = next(iter(children_of_type(func_stmt, f03.Suffix)), None)
    if suffix is not None:
        nm = next(iter(children_of_type(suffix, f03.Name)), None)
        if nm is not None:
            return str(nm)
    return utils.find_name_of_stmt(func_stmt) or ""


def _enclosing_exec_stmt(node: f03.Base) -> Optional[f03.Base]:
    """The ancestor of ``node`` that is a direct statement of an Execution_Part
    (so the inlined function body can be spliced in just before it)."""
    cur = node
    while cur is not None and not isinstance(cur.parent, f03.Execution_Part):
        cur = cur.parent
    return cur


def _inline_one_funcref(ref: f03.Base, callee_sub: f03.Base, counter: int) -> bool:
    """Inline a FUNCTION reference used as an expression: splice the body (which
    assigns the result variable) in just before the enclosing statement, and
    replace the reference with the (renamed, caller-declared) result variable.

    Used for the ICON ``comm_pat_of_type(p_patch, typ)`` shape -- a pointer-result
    function whose body is the ``typ`` ladder; once the actual ``typ`` (a constant)
    is bound and the ladder folds, the reference becomes a single-source rebind in
    the caller.
    """
    func_stmt = next(iter(children_of_type(callee_sub, (f03.Function_Stmt, ))), None)
    if func_stmt is None:
        return False
    dummies = _dummy_arg_names(func_stmt)
    callee_spec = next(iter(children_of_type(callee_sub, f03.Specification_Part)), None)
    optionals = _optional_dummy_names(callee_spec, {d.lower() for d in dummies})
    binding = _bind_actuals_to_dummies(ref, dummies, optionals)
    if binding is None:
        return False
    bound, absent = binding

    callee_exec = next(iter(children_of_type(callee_sub, f03.Execution_Part)), None)
    if callee_exec is None:
        return False
    enclosing = _enclosing_exec_stmt(ref)
    if enclosing is None:
        return False
    exec_part = enclosing.parent
    caller_sub = enclosing
    while caller_sub is not None and not isinstance(
            caller_sub, (f03.Subroutine_Subprogram, f03.Function_Subprogram, f03.Main_Program)):
        caller_sub = caller_sub.parent
    caller_spec = next(iter(children_of_type(caller_sub, f03.Specification_Part)), None) if caller_sub else None
    if caller_spec is None:
        return False

    result_name = _function_result_name(func_stmt)
    dummy_set = {d.lower() for d in dummies}
    local_decls = _local_decls(callee_spec, dummy_set)
    rename: Dict[str, str] = {}
    for decl in local_decls:
        for ed in walk(decl, f03.Entity_Decl):
            nm = next(iter(children_of_type(ed, f03.Name)), None)
            if nm is not None:
                rename[str(nm)] = f"{str(nm)}_fn{counter}"
    result_repl = rename.get(result_name)
    if result_repl is None:
        # The result variable must be among the callee's declarations to rename +
        # re-declare in the caller; if it is not declared (implicit-typed result)
        # we cannot soundly hoist it -- skip.
        return False

    exec_text = _fold_optionals(str(callee_exec), set(bound), absent)
    mapping: Dict[str, str] = dict(bound)
    mapping.update({d: _ABSENT for d in absent})
    mapping.update(rename)
    exec_text = _drop_calls_passing_absent(_drop_absent_actuals(_subst(exec_text, mapping)))

    decl_texts = [_subst(str(d), rename) for d in local_decls]
    frag_spec, frag_exec = _reparse_fragment(_carry_uses(callee_sub), decl_texts, exec_text)
    if _fragment_has_absent(frag_spec, frag_exec):
        return False

    _merge_into_caller_spec(caller_spec, frag_spec)
    # Splice the function body in just before the enclosing statement.
    new_children = []
    for c in exec_part.children:
        if c is enclosing:
            new_children.extend(frag_exec)
        new_children.append(c)
    utils.set_children(exec_part, new_children)
    # Replace the reference with the result variable.
    utils.replace_node(ref, f03.Name(result_repl))
    return True


def _target_defs(ast: f03.Program, want: Set[str], stmt_type) -> Dict[str, f03.Base]:
    """Map each target NAME to its (unique) subprogram definition.  A fallback for
    resolving a target call whose CALLER-LOCAL alias scope does not reach it -- once
    an outer wrapper is inlined, its forwarded ``CALL mixprec`` lands in the caller's
    module (e.g. ``solve_nh`` in ``mo_solve_nonhydro``) where ``mixprec`` (defined in
    ``mo_sync``) is not in a local USE-ONLY rename, so ``search_real_local_alias_spec``
    cannot resolve it.  Matching the call's own name against the target definitions
    closes that gap; targets are specific named procedures, so a name match is the
    intended callee."""
    out: Dict[str, f03.Base] = {}
    for sub in walk(ast, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        stmt = next(iter(children_of_type(sub, stmt_type)), None)
        if stmt is None:
            continue
        nm = (utils.find_name_of_stmt(stmt) or "").lower()
        if nm in want and nm not in out:
            out[nm] = sub
    return out


def _resolve_target_callee(procname: f03.Name, alias_map, want: Set[str], target_defs: Dict[str, f03.Base],
                           stmt_type) -> Optional[f03.Base]:
    """Resolve a call/reference name to a TARGET subprogram definition: first via the
    caller's local alias scope (handles USE-renamed ``deconiface`` specifics), then
    by direct target-name match (handles a forwarded call that landed cross-module)."""
    spec = analysis.search_real_local_alias_spec(procname, alias_map)
    if spec is not None and spec in alias_map:
        callee_stmt = alias_map[spec]
        if isinstance(callee_stmt, stmt_type):
            callee_name = (utils.find_name_of_stmt(callee_stmt) or "").lower()
            if callee_name in want or procname.string.lower() in want:
                sub = _subprogram_of(callee_stmt)
                if sub is not None:
                    return sub
    return target_defs.get(procname.string.lower())


def inline_named_functions(ast: f03.Program, targets: Iterable[str]) -> int:
    """Inline references to FUNCTIONs in ``targets`` (used as expressions) into
    their enclosing statement.  Returns the number of references inlined."""
    want = {t.lower() for t in targets}
    if not want:
        return 0
    inlined = 0
    for _ in range(_MAX_ROUNDS):
        alias_map = analysis.alias_specs(ast)
        target_defs = _target_defs(ast, want, f03.Function_Stmt)
        progressed = False
        # A function used as an actual argument (``ex(cot(p, t), a)``) parses as a
        # ``Part_Ref`` (fparser cannot tell array-index from call); a top-level
        # call parses as a ``Function_Reference``.  Walk both; ``_resolve_target_callee``
        # admits only references that resolve to a TARGET function, so an array index
        # (name resolves to a variable, not a target) is never touched.
        for ref in list(walk(ast, (f03.Function_Reference, f03.Part_Ref))):
            if ref.parent is None:
                continue
            name = ref.children[0]
            if not isinstance(name, f03.Name):
                continue
            callee_sub = _resolve_target_callee(name, alias_map, want, target_defs, f03.Function_Stmt)
            if callee_sub is None:
                continue
            if _inline_one_funcref(ref, callee_sub, inlined):
                inlined += 1
                progressed = True
                break
        if not progressed:
            break
    return inlined


def inline_named_subprograms(ast: f03.Program, targets: Iterable[str]) -> int:
    """Inline every call to a procedure in ``targets`` (by name, case-insensitive)
    into its caller.  Generic-interface specifics are matched by their own name
    (resolve the generic to its specific BEFORE calling this).  Returns the number
    of call sites inlined.

    Iterates to a fixpoint so a target that calls another target (the
    ``sync_patch_array_mult_f3din_dp`` -> ``sync_patch_array_mult_mixprec`` chain)
    is fully flattened.
    """
    want = {t.lower() for t in targets}
    if not want:
        return 0
    inlined = 0
    for _ in range(_MAX_ROUNDS):
        alias_map = analysis.alias_specs(ast)
        target_defs = _target_defs(ast, want, f03.Subroutine_Stmt)
        progressed = False
        for call in list(walk(ast, f03.Call_Stmt)):
            if call.parent is None:
                continue  # already spliced away this round
            procname = call.children[0]
            if not isinstance(procname, f03.Name):
                continue  # type-bound / indirect -- not our target
            callee_sub = _resolve_target_callee(procname, alias_map, want, target_defs, f03.Subroutine_Stmt)
            if callee_sub is None:
                continue
            if _inline_one_call(call, callee_sub, inlined):
                inlined += 1
                progressed = True
                break  # re-walk: the AST mutated
        if not progressed:
            break
    return inlined


def specialize_at_source(ast: f03.Program, targets: Iterable[str]) -> Tuple[int, int]:
    """Specialize every call to a ``targets`` procedure to its call site by
    inlining its body (SUBROUTINE calls and FUNCTION references both), so the
    call-site-constant arguments fold the body.  Returns ``(n_subprograms,
    n_functions)`` inlined.  A no-op when ``targets`` is empty."""
    n_sub = inline_named_subprograms(ast, targets)
    n_fun = inline_named_functions(ast, targets)
    # Diagnostic: a TARGET call that survives (by its own name) was NOT specialized
    # -- e.g. an absent-optional shape the splice could not express -- and will
    # carry its runtime-selected construct into the bridge, where it fails to lower.
    want = {t.lower() for t in targets}
    survivors = sorted({
        c.children[0].string.lower()
        for c in walk(ast, f03.Call_Stmt)
        if isinstance(c.children[0], f03.Name) and c.children[0].string.lower() in want
    })
    if survivors:
        logger.warning("specialize_at_source: %d TARGET call(s) NOT inlined (will block lowering): %s", len(survivors),
                       survivors)
    return n_sub, n_fun
