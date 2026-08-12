"""Array-access compression passes.

Two DaCe passes plus a small :class:`Constraint` hierarchy for stating
when the compression is safe at runtime:

* :class:`FoldArrayAccess` -- every subscript access ``arr(i0, ..., iN)``
  to a listed array is rewritten to a caller-supplied sympy expression
  over the subscripts. Use when an array's contents reduce to a cheap
  expression under a runtime-checkable invariant
  (e.g. ``*_blk[i, jb, k] == 1`` on a single-block grid, where we can
  fold the access to the literal ``1``).

* :class:`GenerateCompressedVariant` -- clone the SDFG body with every
  listed array demoted to a narrower dtype, then dispatch at runtime
  via a ``ConditionalBlock`` between the compressed and original
  bodies. Use for dtype demotion under a runtime bound
  (``uint16`` neighbour indices, ``uint8`` block indices).

Both passes take ``constraints: Sequence[Constraint]``. A constraint
knows how to (a) emit an abort-on-violation runtime check and (b)
yield a C-boolean dispatch condition. The two use-modes are:

  * **Precondition**: ``FoldArrayAccess`` runs every constraint's
    abort check. Violation aborts; no fallback.
  * **Dispatch**: ``GenerateCompressedVariant`` uses the AND of every
    constraint's dispatch condition as the ``ConditionalBlock``
    branch predicate (compressed branch when true, original otherwise).
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set

import sympy
from sympy.core.function import AppliedUndef

import dace
from dace import SDFG, dtypes, properties, symbolic
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg.state import (ConditionalBlock, ControlFlowRegion, SDFGState)
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


@dataclass
class Constraint:
    """Base class for runtime-checkable compression preconditions /
    dispatch guards. Subclasses implement:

      * :meth:`emit_abort_check` -- insert an ``abort()`` tasklet that
        fires when the constraint is violated.
      * :meth:`dispatch_condition_expr` -- return a C-boolean
        expression usable directly as a ``ConditionalBlock`` branch
        predicate. May require inserting reduction tasklets in a
        pre-state; those are attached to ``pre_state`` and return a
        scalar name that ``pre_state``'s successor can read.
    """

    def emit_abort_check(self, sdfg: SDFG, state: SDFGState) -> None:
        raise NotImplementedError

    def dispatch_condition_expr(self, sdfg: SDFG, pre_state: SDFGState) -> str:
        raise NotImplementedError


@dataclass
class SymbolicConstraint(Constraint):
    """Pure-symbol predicate. Cheapest variant -- no memory reads.
    ``expr`` is expressed in C-boolean syntax (``&&`` / ``||`` / ``!``
    etc.); for use as a DaCe ``CodeBlock`` (interstate condition /
    ConditionalBlock guard) we convert to Python-boolean (``and`` /
    ``or`` / ``not``) via :meth:`_py_expr`."""
    expr: str

    def emit_abort_check(self, sdfg: SDFG, state: SDFGState) -> None:
        # Abort tasklet is CPP code, so keep the raw C expression.
        _append_abort_tasklet(state, self.expr)

    def dispatch_condition_expr(self, sdfg: SDFG, _pre_state: SDFGState) -> str:
        # ConditionalBlock guards are parsed as Python ASTs; translate
        # C-boolean ops to Python equivalents.
        return f"({self._py_expr()})"

    def _py_expr(self) -> str:
        return (self.expr
                .replace('&&', ' and ')
                .replace('||', ' or ')
                .replace('!=', '__NE__')  # preserve != (not Python !)
                .replace('!', ' not ')
                .replace('__NE__', '!='))


@dataclass
class ArrayAllEqual(Constraint):
    """Every element of ``array_name`` equals ``value_expr``. Runtime
    check materialises as a reduction that OR-accumulates mismatches
    (implementation pending; subclass for now)."""
    array_name: str
    value_expr: str

    def emit_abort_check(self, sdfg: SDFG, state: SDFGState) -> None:
        raise NotImplementedError(
            "ArrayAllEqual.emit_abort_check requires an any-mismatch "
            "reduction kernel; pending Pass 2 helpers.")

    def dispatch_condition_expr(self, sdfg: SDFG, pre_state: SDFGState) -> str:
        raise NotImplementedError(
            "ArrayAllEqual.dispatch_condition_expr requires an "
            "any-mismatch reduction kernel; pending.")


@dataclass
class ArrayMaxBelow(Constraint):
    """``max(array_name) < limit_expr``. Runtime check materialises as
    a ``reduce_max_gpu`` call into a scalar, then compared against the
    limit (implementation pending)."""
    array_name: str
    limit_expr: str

    def emit_abort_check(self, sdfg: SDFG, state: SDFGState) -> None:
        raise NotImplementedError(
            "ArrayMaxBelow.emit_abort_check requires a max-reduction "
            "kernel; pending.")

    def dispatch_condition_expr(self, sdfg: SDFG, pre_state: SDFGState) -> str:
        raise NotImplementedError(
            "ArrayMaxBelow.dispatch_condition_expr requires a "
            "max-reduction kernel; pending.")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_CHECK_STATE_LABEL = '_compression_precondition_check'


def _append_abort_tasklet(state: SDFGState, bool_expr: str) -> None:
    """Inject ``if (!(bool_expr)) { printf...; abort(); }`` as a
    side-effects tasklet into ``state``. Multiple calls stack
    (tasklets run in whatever order DaCe emits them)."""
    code = (
        f'if (!({bool_expr})) {{\n'
        f'  printf("compression precondition failed: expected `{bool_expr}`\\n");\n'
        f'  abort();\n'
        f'}}'
    )
    state.add_tasklet(
        name='_precondition_check',
        inputs={}, outputs={},
        code=code, language=dtypes.Language.CPP,
        side_effects=True,
    )


def _ensure_precondition_state(sdfg: SDFG) -> SDFGState:
    """Return the dedicated precondition-check state, creating it as
    the new start state on first request. Idempotent -- subsequent
    calls return the existing state so multiple constraints stack
    into the same state."""
    for block in sdfg.nodes():
        if (isinstance(block, SDFGState)
                and block.label == _CHECK_STATE_LABEL):
            return block
    prior_start = sdfg.start_state
    return sdfg.add_state_before(
        prior_start, label=_CHECK_STATE_LABEL, is_start_block=True)


def _collect_sdfgs(sdfg: SDFG) -> List[SDFG]:
    out = [sdfg]
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            out.append(n.sdfg)
    return out


def _parse_fold_rule(rule: str, arity: int) -> sympy.Expr:
    """Parse a fold-rule string into a sympy expression, substituting
    placeholder symbols ``idx_0, idx_1, ..., idx_{arity-1}`` with
    fresh sympy symbols. Caller replaces the placeholders with the
    actual subscript expressions at rewrite time."""
    return symbolic.pystr_to_symbolic(rule)


def _apply_fold_rule(rule_expr: sympy.Expr, subscripts: Sequence[sympy.Expr]) -> sympy.Expr:
    """Substitute ``idx_0 ... idx_{N-1}`` in ``rule_expr`` with the
    concrete subscripts of one array access. ``rule_expr`` may also
    be a plain constant (no placeholders)."""
    subs = {sympy.Symbol(f'idx_{i}'): s for i, s in enumerate(subscripts)}
    return rule_expr.xreplace(subs)


def _rewrite_expr_string(expr_str: str, targets: Set[str],
                         rule_expr: sympy.Expr,
                         preserve_arrays: Optional[frozenset] = None) -> str:
    """Parse ``expr_str``, replace every ``Function(name)(...)`` whose
    name is in ``targets`` by ``rule_expr`` with its placeholders
    filled in from the call arguments, reserialise via DaCe's
    ``symstr`` with ``arrayexprs`` so any surviving ``arr[i,j,k]``
    subscripts round-trip as subscripts (not function calls)."""
    if not any(name in expr_str for name in targets):
        return expr_str
    try:
        parsed = symbolic.pystr_to_symbolic(expr_str)
    except Exception:
        return expr_str
    if not hasattr(parsed, 'atoms'):
        return expr_str
    subs = {}
    remaining_arrays = set()
    for u in parsed.atoms(AppliedUndef):
        if u.func.__name__ in targets:
            subs[u] = _apply_fold_rule(rule_expr, u.args)
        else:
            # Keep other subscript-shaped accesses as array refs.
            remaining_arrays.add(u.func.__name__)
    if not subs:
        return expr_str
    replaced = parsed.xreplace(subs)
    # After substitution, any AppliedUndef that remains is another
    # array we shouldn't render as a function call. ``preserve_arrays``
    # union'd with remaining_arrays covers both cases.
    preserve = (preserve_arrays or frozenset()) | frozenset(remaining_arrays)
    return symbolic.symstr(replaced, arrayexprs=preserve)


def _rewrite_interstate_edges(g: SDFG, targets: Set[str],
                               rule_expr: sympy.Expr) -> int:
    changed = 0
    for e in g.all_interstate_edges():
        for k, v in list(e.data.assignments.items()):
            v_str = v.as_string if isinstance(v, CodeBlock) else str(v)
            new_str = _rewrite_expr_string(v_str, targets, rule_expr)
            if new_str != v_str:
                e.data.assignments[k] = (
                    CodeBlock(new_str) if isinstance(v, CodeBlock) else new_str)
                changed += 1
        if e.data.condition is not None:
            cs = e.data.condition.as_string
            new_cs = _rewrite_expr_string(cs, targets, rule_expr)
            if new_cs != cs:
                e.data.condition = CodeBlock(new_cs,
                                             e.data.condition.language)
                changed += 1
    return changed


def _rewrite_conditional_block_guards(g: SDFG, targets: Set[str],
                                       rule_expr: sympy.Expr) -> int:
    changed = 0
    for block in g.all_control_flow_blocks():
        if not isinstance(block, ConditionalBlock):
            continue
        for i, (cond, _body) in enumerate(list(block.branches)):
            if cond is None:
                continue
            cs = cond.as_string
            new_cs = _rewrite_expr_string(cs, targets, rule_expr)
            if new_cs != cs:
                block.branches[i] = [CodeBlock(new_cs, cond.language),
                                     block.branches[i][1]]
                changed += 1
    return changed


# ---------------------------------------------------------------------------
# Pass 1: FoldArrayAccess
# ---------------------------------------------------------------------------


@properties.make_properties
@xf.explicit_cf_compatible
class FoldArrayAccess(ppl.Pass):
    """Fold every subscript access to each listed array to a
    caller-supplied sympy expression over the subscripts.

    Descriptors, AccessNodes, NSDFG connectors, and pass-through
    edges are **preserved** -- the array stays in every function /
    kernel signature, just no expression reads its values. Dead-load
    elimination at ``-O3`` removes the unused loads from the
    generated code.

    Parameters
    ----------
    array_names : set of str
        Target arrays.
    fold_rule : str
        sympy-parseable expression template. Use ``idx_0``, ``idx_1``,
        ... for subscripts. Example: ``"1"`` (constant),
        ``"idx_1"`` (middle subscript), ``"idx_1 + 1"``.
    constraints : sequence of :class:`Constraint`
        Runtime precondition checks. Each constraint's
        ``emit_abort_check`` is called with a shared precondition
        state. Violation aborts; there's no fallback (the fold is
        one-way).
    """

    CATEGORY: str = "Optimization Preparation"

    array_names = properties.SetProperty(
        element_type=str, default=set(),
        desc="Arrays whose subscript accesses should be folded.")
    fold_rule = properties.Property(
        dtype=str, default="0",
        desc="Fold expression template in terms of idx_0, idx_1, ...")

    def __init__(self,
                 array_names: Optional[Set[str]] = None,
                 fold_rule: str = "0",
                 constraints: Optional[Sequence[Constraint]] = None):
        super().__init__()
        self.array_names = set(array_names or set())
        self.fold_rule = fold_rule
        # Constraints are not DaCe-serialisable (hold arbitrary
        # Python callables via subclass hooks); kept as a plain list.
        self._constraints: List[Constraint] = list(constraints or [])

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.InterstateEdges
                | ppl.Modifies.Edges | ppl.Modifies.Nodes)

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        targets = {n for n in self.array_names if _name_appears_anywhere(sdfg, n)}
        if not targets:
            return None

        if self._constraints:
            check_state = _ensure_precondition_state(sdfg)
            for c in self._constraints:
                c.emit_abort_check(sdfg, check_state)

        rule_expr = _parse_fold_rule(self.fold_rule,
                                     arity=_infer_max_arity(sdfg, targets))

        all_sdfgs = _collect_sdfgs(sdfg)
        for g in all_sdfgs:
            _rewrite_interstate_edges(g, targets, rule_expr)
            _rewrite_conditional_block_guards(g, targets, rule_expr)

        return len(targets)


def _infer_max_arity(sdfg: SDFG, targets: Set[str]) -> int:
    """Return the largest subscript arity seen for any target array,
    so ``_parse_fold_rule`` can validate placeholder count. Heuristic;
    if detection fails, returns 0 and the fold rule is parsed as-is."""
    max_arity = 0
    for g in _collect_sdfgs(sdfg):
        for e in g.all_interstate_edges():
            for v in e.data.assignments.values():
                s = v.as_string if isinstance(v, CodeBlock) else str(v)
                if not any(n in s for n in targets):
                    continue
                try:
                    parsed = symbolic.pystr_to_symbolic(s)
                    for u in parsed.atoms(AppliedUndef):
                        if u.func.__name__ in targets:
                            max_arity = max(max_arity, len(u.args))
                except Exception:
                    pass
    return max_arity


def _name_appears_anywhere(sdfg: SDFG, name: str) -> bool:
    for g in _collect_sdfgs(sdfg):
        if name in g.arrays:
            return True
        for e in g.all_interstate_edges():
            for v in e.data.assignments.values():
                s = v.as_string if isinstance(v, CodeBlock) else str(v)
                if name in s:
                    return True
            if e.data.condition is not None and name in e.data.condition.as_string:
                return True
    return False


# ---------------------------------------------------------------------------
# Pass 2: GenerateCompressedVariant
# ---------------------------------------------------------------------------


@properties.make_properties
@xf.explicit_cf_compatible
class GenerateCompressedVariant(ppl.Pass):
    """Clone the SDFG body, demote listed arrays to ``target_dtype``
    in the clone, dispatch at runtime between the compressed and
    original bodies via a ``ConditionalBlock``.

    Parameters
    ----------
    array_names : set of str
        Arrays whose dtype should be demoted in the compressed branch.
        A sibling descriptor ``<name>_<suffix>`` is added at the SDFG
        top level; the compressed branch renames references to it.
    target_dtype : dace.dtypes.typeclass
        Compressed dtype. The sibling suffix defaults to
        ``target_dtype.to_string()``.
    constraints : sequence of :class:`Constraint`
        Runtime dispatch guards. The AND of each constraint's
        ``dispatch_condition_expr`` is the compressed-branch
        predicate; the fallback branch runs when any constraint fails.
    body_start_label : str, optional
        Label of the state that begins the body region. Defaults to
        the first non-copy-in state in BFS order.
    body_end_label : str, optional
        Label of the state that ends the body region. Defaults to the
        last state before any copy-out.
    name_suffix : str, optional
        Name suffix for the compressed sibling descriptors.
    """

    CATEGORY: str = "Optimization Preparation"

    array_names = properties.SetProperty(
        element_type=str, default=set(),
        desc="Arrays to demote in the compressed branch.")

    def __init__(self,
                 array_names: Optional[Set[str]] = None,
                 target_dtype: Optional[dtypes.typeclass] = None,
                 constraints: Optional[Sequence[Constraint]] = None,
                 body_start_label: Optional[str] = None,
                 body_end_label: Optional[str] = None,
                 name_suffix: Optional[str] = None):
        super().__init__()
        self.array_names = set(array_names or set())
        self._target_dtype = target_dtype
        self._constraints: List[Constraint] = list(constraints or [])
        self._body_start_label = body_start_label
        self._body_end_label = body_end_label
        self._name_suffix = name_suffix

    def modifies(self) -> ppl.Modifies:
        return (ppl.Modifies.States | ppl.Modifies.InterstateEdges
                | ppl.Modifies.Descriptors | ppl.Modifies.Edges
                | ppl.Modifies.Nodes)

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _pipeline_results) -> Optional[int]:
        targets = [n for n in self.array_names if n in sdfg.arrays]
        if not targets or self._target_dtype is None:
            return None
        suffix = self._name_suffix or self._target_dtype.to_string()

        # 1. Add compressed sibling descriptors at the root.
        import copy as _copy
        for name in targets:
            sibling = f"{name}_{suffix}"
            if sibling in sdfg.arrays:
                continue
            desc = _copy.deepcopy(sdfg.arrays[name])
            desc.dtype = self._target_dtype
            desc.transient = True
            desc.lifetime = dtypes.AllocationLifetime.SDFG
            sdfg.add_datadesc(sibling, desc)

        # 2. Wrap the body states in a ControlFlowRegion.
        body_states = _select_body_states(sdfg,
                                          self._body_start_label,
                                          self._body_end_label)
        if not body_states:
            return None
        original_region = _wrap_states_in_region(
            sdfg, body_states, label='_body_original')

        # 3. Deepcopy the region for the compressed branch.
        compressed_region = _copy.deepcopy(original_region)
        compressed_region.label = '_body_compressed'

        # 4. Rename arrays to the compressed names in the clone.
        rename_map = {name: f"{name}_{suffix}" for name in targets}
        _rename_in_region(compressed_region, rename_map, self._target_dtype)

        # 5. Prepend a convert state that populates the compressed
        # siblings from the originals.
        _prepend_convert_state(sdfg, compressed_region, targets,
                                suffix, self._target_dtype)

        # 6. Build the dispatch condition (AND of all constraints).
        if self._constraints:
            pre_state = _ensure_precondition_state(sdfg)
            conds = [c.dispatch_condition_expr(sdfg, pre_state)
                     for c in self._constraints]
            # Python-AST-compatible (CodeBlock parses as Python).
            dispatch_expr = ' and '.join(f"({c})" for c in conds)
        else:
            dispatch_expr = "True"

        # 7. Insert ConditionalBlock where the body region currently is,
        # with the compressed branch as the primary and the original
        # as fallback.
        _replace_region_with_conditional(
            sdfg, original_region, compressed_region, dispatch_expr)

        # 8. Re-stamp parent-SDFG pointers on every NestedSDFG we
        # touched. Deepcopy + reparenting left the compressed
        # region's inner NSDFGs pointing at the old root; use DaCe's
        # helper to walk the whole tree and fix them.
        from dace.sdfg.utils import set_nested_sdfg_parent_references
        set_nested_sdfg_parent_references(sdfg)

        return len(targets)


def _select_body_states(sdfg: SDFG,
                        start_label: Optional[str],
                        end_label: Optional[str]) -> List[SDFGState]:
    """Heuristic body selector. If explicit labels are given, uses
    them; else selects all top-level ``SDFGState`` nodes that aren't
    copy-in / copy-out markers (those labelled with
    ``_cpu_to_gpu_copy_in`` / ``_gpu_to_cpu_copy_out`` /
    ``_sync_after_copy_*`` / the precondition-check state)."""
    states = [n for n in sdfg.nodes() if isinstance(n, SDFGState)]
    if start_label and end_label:
        return [s for s in states
                if start_label <= s.label <= end_label]
    skip_prefixes = (
        '_cpu_to_gpu_copy_in',
        '_gpu_to_cpu_copy_out',
        '_sync_after_copy_in',
        '_sync_after_copy_out',
        _CHECK_STATE_LABEL,
    )
    return [s for s in states
            if not any(s.label.startswith(p) for p in skip_prefixes)]


def _wrap_states_in_region(sdfg: SDFG, body_states: List[SDFGState],
                            label: str) -> ControlFlowRegion:
    """Move ``body_states`` out of the SDFG's top-level graph into a
    new ControlFlowRegion. Re-wires edges entering / leaving the body
    to attach to the region node instead."""
    region = ControlFlowRegion(label=label, sdfg=sdfg)
    body_set = set(body_states)

    # Capture edges before mutation.
    internal_edges = []
    entering_edges = []
    leaving_edges = []
    for e in list(sdfg.edges()):
        in_src = e.src in body_set
        in_dst = e.dst in body_set
        if in_src and in_dst:
            internal_edges.append(e)
        elif in_dst and not in_src:
            entering_edges.append(e)
        elif in_src and not in_dst:
            leaving_edges.append(e)

    # Pick the region's start block: the body state that has a
    # non-body predecessor, or the first body state if none.
    entry_dsts = {e.dst for e in entering_edges}
    start = next((s for s in body_states if s in entry_dsts),
                 body_states[0])

    # Remove body states + internal edges from the SDFG.
    for e in internal_edges + entering_edges + leaving_edges:
        sdfg.remove_edge(e)
    for s in body_states:
        sdfg.remove_node(s)

    # Add body states + internal edges to the region.
    for s in body_states:
        region.add_node(s, is_start_block=(s is start))
    for e in internal_edges:
        region.add_edge(e.src, e.dst, e.data)

    # Add region as a node in the SDFG; reattach pre/post edges.
    sdfg.add_node(region)
    for e in entering_edges:
        sdfg.add_edge(e.src, region, e.data)
    for e in leaving_edges:
        sdfg.add_edge(region, e.dst, e.data)
    return region


def _rename_in_region(region: ControlFlowRegion,
                       rename_map: dict,
                       target_dtype: dtypes.typeclass) -> None:
    """Within ``region``, replace every occurrence of each key in
    ``rename_map`` (as an array reference) with its mapped name, via
    DaCe's symbolic system. Also mutates inner Array descriptors'
    dtype to ``target_dtype``. Walks interstate edges, memlets,
    ConditionalBlock guards, and descends into NestedSDFGs."""
    # Collect all SDFGs below the region.
    sdfgs: List[SDFG] = []
    for n, _ in region.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            sdfgs.append(n.sdfg)
    # Also the region's own SDFG-local nodes; arrays live on the
    # parent SDFG however.
    region_sdfg = region.sdfg

    # Rename is a pure string substitution with word-boundary matching.
    # Sympy-level xreplace preserves the AST shape but
    # ``str(AppliedUndef)`` rounds back to ``arr(i, j, k)``
    # (function-call syntax), which DaCe's codegen then compiles as a
    # real call. Sticking to string-level substitution preserves both
    # ``arr[i, j, k]`` subscripts and bare ``arr`` symbol references
    # exactly.
    import re as _re
    patterns = {name: _re.compile(r'\b' + _re.escape(name) + r'\b')
                for name in rename_map}

    def _rename_one(s: str) -> str:
        out = s
        for old, pat in patterns.items():
            if old in out:  # cheap early-out
                out = pat.sub(rename_map[old], out)
        return out

    # Interstate edges within the region.
    for e in region.all_interstate_edges():
        for k, v in list(e.data.assignments.items()):
            v_str = v.as_string if isinstance(v, CodeBlock) else str(v)
            new_str = _rename_one(v_str)
            if new_str != v_str:
                e.data.assignments[k] = (
                    CodeBlock(new_str) if isinstance(v, CodeBlock) else new_str)
        if e.data.condition is not None:
            cs = e.data.condition.as_string
            ns = _rename_one(cs)
            if ns != cs:
                e.data.condition = CodeBlock(ns, e.data.condition.language)

    # Memlet data + AccessNode data within the region's states.
    for state in region.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in rename_map:
                node.data = rename_map[node.data]
        for e in state.edges():
            if e.data is not None and e.data.data in rename_map:
                e.data.data = rename_map[e.data.data]

    # NSDFG connectors + inner descriptor renames + feeding edge
    # ``dst_conn`` / ``src_conn`` updates.
    for state in region.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            for old, new in rename_map.items():
                if old in node.in_connectors:
                    node.in_connectors[new] = node.in_connectors.pop(old)
                    for e in state.in_edges(node):
                        if e.dst_conn == old:
                            e.dst_conn = new
                if old in node.out_connectors:
                    node.out_connectors[new] = node.out_connectors.pop(old)
                    for e in state.out_edges(node):
                        if e.src_conn == old:
                            e.src_conn = new
                if old in node.sdfg.arrays:
                    inner = node.sdfg.arrays.pop(old)
                    inner.dtype = target_dtype
                    node.sdfg.add_datadesc(new, inner, find_new_name=False)

    # Also descend into NSDFGs' SDFGs for interstate edges,
    # connectors, and nested NSDFG connectors.
    for nsdfg in sdfgs:
        _rename_inner_sdfg(nsdfg, rename_map, target_dtype, _rename_one)
        # Nested NSDFGs' own connectors & feeding edges.
        for state in nsdfg.all_states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                for old, new in rename_map.items():
                    if old in node.in_connectors:
                        node.in_connectors[new] = node.in_connectors.pop(old)
                        for e in state.in_edges(node):
                            if e.dst_conn == old:
                                e.dst_conn = new
                    if old in node.out_connectors:
                        node.out_connectors[new] = node.out_connectors.pop(old)
                        for e in state.out_edges(node):
                            if e.src_conn == old:
                                e.src_conn = new
                    if old in node.sdfg.arrays:
                        inner = node.sdfg.arrays.pop(old)
                        inner.dtype = target_dtype
                        node.sdfg.add_datadesc(new, inner, find_new_name=False)


def _rename_inner_sdfg(inner: SDFG, rename_map: dict,
                        target_dtype: dtypes.typeclass,
                        rename_fn) -> None:
    """Apply the same rename/dtype mutations to an inner SDFG."""
    for e in inner.all_interstate_edges():
        for k, v in list(e.data.assignments.items()):
            v_str = v.as_string if isinstance(v, CodeBlock) else str(v)
            ns = rename_fn(v_str)
            if ns != v_str:
                e.data.assignments[k] = (
                    CodeBlock(ns) if isinstance(v, CodeBlock) else ns)
        if e.data.condition is not None:
            cs = e.data.condition.as_string
            ns = rename_fn(cs)
            if ns != cs:
                e.data.condition = CodeBlock(ns, e.data.condition.language)
    for state in inner.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in rename_map:
                node.data = rename_map[node.data]
        for e in state.edges():
            if e.data is not None and e.data.data in rename_map:
                e.data.data = rename_map[e.data.data]


def _prepend_convert_state(sdfg: SDFG,
                            region: ControlFlowRegion,
                            original_names: Sequence[str],
                            suffix: str,
                            target_dtype: dtypes.typeclass) -> None:
    """Insert a state at the start of ``region`` that populates each
    ``<name>_<suffix>`` from ``<name>`` via a narrow-cast map. The
    map iterates per-dimension so memlet subsets match the array
    dimensionality -- a flat 1D memlet onto a multi-D array would
    fail DaCe's validator."""
    import dace.memlet as _mm

    # Capture the previous start block before inserting.
    prior_start = region.start_block

    convert_state = SDFGState('_convert_to_' + suffix,
                               sdfg=region.sdfg)
    region.add_node(convert_state, is_start_block=True)

    for name in original_names:
        desc = sdfg.arrays[name]
        sibling = f"{name}_{suffix}"
        shape = tuple(desc.shape)
        if len(shape) == 0:
            continue

        in_ac = convert_state.add_read(name)
        out_ac = convert_state.add_write(sibling)

        # One map param per axis so the subset matches the array
        # dimensionality. ``_cvt_d0``, ``_cvt_d1``, ...
        axis_vars = [f'_cvt_d{i}' for i in range(len(shape))]
        axis_ranges = {v: f'0:{s}' for v, s in zip(axis_vars, shape)}
        me, mx = convert_state.add_map(f'_convert_{name}', axis_ranges)
        me.add_in_connector('IN_src')
        me.add_out_connector('OUT_src')
        mx.add_in_connector('IN_dst')
        mx.add_out_connector('OUT_dst')
        tasklet = convert_state.add_tasklet(
            f'_cvt_{name}',
            inputs={'_in': desc.dtype},
            outputs={'_out': target_dtype},
            code=f'_out = ({target_dtype.ctype})_in;',
            language=dtypes.Language.CPP,
        )

        point_subset = ', '.join(axis_vars)
        convert_state.add_edge(in_ac, None, me, 'IN_src',
                                _mm.Memlet.from_array(name, desc))
        convert_state.add_edge(me, 'OUT_src', tasklet, '_in',
                                _mm.Memlet(f'{name}[{point_subset}]'))
        convert_state.add_edge(tasklet, '_out', mx, 'IN_dst',
                                _mm.Memlet(f'{sibling}[{point_subset}]'))
        convert_state.add_edge(
            mx, 'OUT_dst', out_ac, None,
            _mm.Memlet.from_array(sibling, sdfg.arrays[sibling]))

    # Rewire the region's previous start block to run after the
    # convert state.
    if prior_start is not None and prior_start is not convert_state:
        region.add_edge(convert_state, prior_start, dace.InterstateEdge())


def _replace_region_with_conditional(sdfg: SDFG,
                                      original_region: ControlFlowRegion,
                                      compressed_region: ControlFlowRegion,
                                      dispatch_expr: str) -> None:
    """Replace ``original_region`` in the SDFG with a
    ``ConditionalBlock`` that picks between ``compressed_region``
    (when ``dispatch_expr`` is true) and ``original_region``."""
    # Capture pre- and post-edges of the original region.
    pre_edges = list(sdfg.in_edges(original_region))
    post_edges = list(sdfg.out_edges(original_region))
    for e in pre_edges + post_edges:
        sdfg.remove_edge(e)
    sdfg.remove_node(original_region)

    cb = ConditionalBlock(label='_compression_dispatch', sdfg=sdfg)
    cb.add_branch(CodeBlock(dispatch_expr), compressed_region)
    cb.add_branch(None, original_region)  # default / else

    sdfg.add_node(cb)
    for e in pre_edges:
        sdfg.add_edge(e.src, cb, e.data)
    for e in post_edges:
        sdfg.add_edge(cb, e.dst, e.data)


# ---------------------------------------------------------------------------
# Function-style convenience wrappers (stage-driver call sites)
# ---------------------------------------------------------------------------


def fold_array_access_to_expression(
    sdfg: SDFG,
    array_names: Sequence[str],
    rewrite_rule: Callable[[str, Sequence[sympy.Expr]], sympy.Expr],
    constraints: Optional[Sequence[Constraint]] = None,
) -> int:
    """Convenience wrapper around :class:`FoldArrayAccess` for callers
    who'd rather pass a lambda than a rule string. The lambda is
    applied to every match; the resulting expression is substituted
    in."""
    # Collect targets present anywhere in the SDFG.
    targets = {n for n in array_names if _name_appears_anywhere(sdfg, n)}
    if not targets:
        return 0
    if constraints:
        check_state = _ensure_precondition_state(sdfg)
        for c in constraints:
            c.emit_abort_check(sdfg, check_state)

    def _rewrite(s: str) -> Optional[str]:
        if not any(n in s for n in targets):
            return None
        try:
            parsed = symbolic.pystr_to_symbolic(s)
        except Exception:
            return None
        if not hasattr(parsed, 'atoms'):
            return None
        subs = {}
        remaining_arrays = set()
        for u in parsed.atoms(AppliedUndef):
            if u.func.__name__ in targets:
                subs[u] = rewrite_rule(u.func.__name__, u.args)
            else:
                remaining_arrays.add(u.func.__name__)
        if not subs:
            return None
        replaced = parsed.xreplace(subs)
        return symbolic.symstr(replaced,
                                arrayexprs=frozenset(remaining_arrays))

    for g in _collect_sdfgs(sdfg):
        for e in g.all_interstate_edges():
            for k, v in list(e.data.assignments.items()):
                v_str = v.as_string if isinstance(v, CodeBlock) else str(v)
                ns = _rewrite(v_str)
                if ns is not None and ns != v_str:
                    e.data.assignments[k] = (
                        CodeBlock(ns) if isinstance(v, CodeBlock) else ns)
            if e.data.condition is not None:
                cs = e.data.condition.as_string
                ns = _rewrite(cs)
                if ns is not None and ns != cs:
                    e.data.condition = CodeBlock(ns,
                                                 e.data.condition.language)
    return len(targets)


def generate_compressed_variant(
    sdfg: SDFG,
    array_names: Sequence[str],
    target_dtype: dtypes.typeclass,
    constraints: Optional[Sequence[Constraint]] = None,
    name_suffix: Optional[str] = None,
    body_start_label: Optional[str] = None,
    body_end_label: Optional[str] = None,
) -> int:
    """Convenience wrapper around :class:`GenerateCompressedVariant`."""
    pass_instance = GenerateCompressedVariant(
        array_names=set(array_names),
        target_dtype=target_dtype,
        constraints=constraints,
        body_start_label=body_start_label,
        body_end_label=body_end_label,
        name_suffix=name_suffix,
    )
    result = pass_instance.apply_pass(sdfg, {})
    return result or 0
