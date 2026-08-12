"""Remove the ``clip_count`` accumulator, its feeding predicate, and the
now-dead guard scaffolding.

The velocity-tendencies SDFG keeps a ``clip_count`` integer symbol that
counts CFL-violated columns per block, then a later block guards the
clipping work with ``if clip_count == 0: continue``. We specialise the
pipeline to always trigger the check (assume ``clip_count > 0``), so all
computation feeding ``clip_count`` is dead.

Steps:

1. Before anything else, walk every ``ConditionalBlock`` and remember the
   *guard-predicate names* (free symbols of any guard whose branch edge
   assigns ``clip_count``). No name-matching -- we parse the predicate
   out of the actual guard CodeBlock.
2. Drop every interstate-edge assignment whose LHS is ``clip_count``.
3. ``sdfg.replace_dict({"clip_count": "1"})`` rewrites remaining
   references (guards like ``clip_count == 0`` become ``1 == 0`` -- const
   False).
4. Remove ``clip_count`` from ``sdfg.symbols``.
5. Prune ``ConditionalBlock`` branches whose guard is now constant False
   OR whose body is a no-op (empty states + no interstate assignments).
   Empty branches get dropped; ``ConditionalBlock``s with no live branches
   disappear too.
6. For each guard-predicate name collected in step 1: if it's a ``Scalar``
   descriptor in ``sdfg.arrays`` AND nothing still reads it, drop the
   writer tasklet, its orphaned inputs, and the descriptor.
"""

import dace
import sympy
from typing import Iterable, Set

from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


_SYMBOL = "clip_count"


def remove_clip_count(sdfg: dace.SDFG) -> int:
    """Apply the full elimination; returns the number of interstate-edge
    assignments dropped in step 2."""
    guard_names = _collect_guard_predicate_names(sdfg)

    dropped = 0
    for edge in sdfg.all_interstate_edges():
        if _SYMBOL in edge.data.assignments:
            del edge.data.assignments[_SYMBOL]
            dropped += 1

    sdfg.replace_dict({_SYMBOL: "1"})

    if _SYMBOL in sdfg.symbols:
        sdfg.remove_symbol(_SYMBOL)

    _prune_dead_conditionals(sdfg)
    _drop_unused_scalars(sdfg, guard_names)
    return dropped


def _collect_guard_predicate_names(sdfg: dace.SDFG) -> Set[str]:
    """Walk every ``ConditionalBlock``; if any branch contains an interstate
    edge that assigns ``clip_count``, collect the branch's guard free
    symbols (the predicate variables)."""
    out: Set[str] = set()
    for node, _ in sdfg.all_nodes_recursive():
        if not isinstance(node, ConditionalBlock):
            continue
        for cond, branch in node.branches:
            if cond is None:
                continue
            writes_clip = any(
                _SYMBOL in (e.data.assignments or {}) for e in branch.edges()
            )
            if writes_clip:
                out |= {str(s) for s in cond.get_free_symbols()}
    return out


def _prune_dead_conditionals(sdfg: dace.SDFG):
    """Remove ``ConditionalBlock`` branches whose guard is constant-False
    or whose body is a no-op. Drop the whole block if nothing lives."""
    # Collect work first -- we can't mutate ``branches`` while iterating.
    work = []
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, ConditionalBlock):
            work.append(node)

    for cb in work:
        live_branches = []
        for cond, branch in cb.branches:
            if cond is None:
                live_branches.append((cond, branch))
                continue
            if _cond_is_const_false(cond.as_string):
                continue
            if _branch_is_noop(branch):
                continue
            live_branches.append((cond, branch))

        if not live_branches:
            parent = cb.parent_graph
            if parent is not None:
                # Replace the ConditionalBlock with an empty state so
                # predecessors / successors stay connected.
                replacement = parent.add_state(cb.label + "_elided")
                in_edges = list(parent.in_edges(cb))
                out_edges = list(parent.out_edges(cb))
                for e in in_edges:
                    parent.add_edge(e.src, replacement, e.data)
                for e in out_edges:
                    parent.add_edge(replacement, e.dst, e.data)
                parent.remove_node(cb)
            continue

        # Replace the branches list in place.
        cb._branches = live_branches


def _cond_is_const_false(cond_str: str) -> bool:
    try:
        expr = dace.symbolic.pystr_to_symbolic(cond_str)
    except Exception:
        return False
    try:
        simp = sympy.simplify(expr)
    except Exception:
        return False
    return simp is sympy.false or simp == 0


def _branch_is_noop(branch: ControlFlowRegion) -> bool:
    for node in branch.nodes():
        if not isinstance(node, dace.SDFGState):
            return False
        if node.nodes():  # any dataflow node
            return False
    for edge in branch.edges():
        if edge.data.assignments:
            return False
        cond = edge.data.condition
        if cond is not None and cond.as_string.strip() not in ("1", "True"):
            return False
    return True


def _drop_unused_scalars(sdfg: dace.SDFG, candidate_names: Iterable[str]):
    """For each name: if it's a ``Scalar`` in ``sdfg.arrays`` and nothing
    still reads it, delete the producing tasklet subgraph + the descriptor.
    """
    for name in candidate_names:
        if name not in sdfg.arrays:
            continue
        desc = sdfg.arrays[name]
        if not isinstance(desc, dace.data.Scalar):
            continue
        if _has_readers(sdfg, name):
            continue
        _delete_producer_and_scalar(sdfg, name)


def _has_readers(sdfg: dace.SDFG, name: str) -> bool:
    # State-level readers: any AccessNode(name) whose out-degree is > 0.
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, dace.nodes.AccessNode) and n.data == name:
                if state.out_degree(n) > 0:
                    return True
    # Guard / interstate readers.
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, ConditionalBlock):
            for cond, _ in node.branches:
                if cond is not None and name in {str(s) for s in cond.get_free_symbols()}:
                    return True
    for e in sdfg.all_interstate_edges():
        if e.data.condition is not None:
            try:
                if name in {str(s) for s in e.data.condition.get_free_symbols()}:
                    return True
            except Exception:
                pass
        for v in (e.data.assignments or {}).values():
            if name in str(v):
                return True
    return False


def _delete_producer_and_scalar(sdfg: dace.SDFG, name: str):
    for state in list(sdfg.all_states()):
        writers = [n for n in state.nodes()
                   if isinstance(n, dace.nodes.AccessNode) and n.data == name]
        for w in writers:
            # Walk backwards: drop every edge feeding ``w`` and the source
            # node if it becomes orphaned.
            for e in list(state.in_edges(w)):
                src = e.src
                state.remove_edge(e)
                if isinstance(src, dace.nodes.Tasklet):
                    # If the tasklet has no other consumers, collapse it.
                    if state.out_degree(src) == 0:
                        for ie in list(state.in_edges(src)):
                            src_of_ie = ie.src
                            state.remove_edge(ie)
                            if (isinstance(src_of_ie, dace.nodes.AccessNode)
                                    and state.degree(src_of_ie) == 0):
                                state.remove_node(src_of_ie)
                        state.remove_node(src)
            if state.degree(w) == 0:
                state.remove_node(w)
    # Drop the descriptor last.
    try:
        sdfg.remove_data(name, validate=False)
    except Exception:
        del sdfg.arrays[name]
