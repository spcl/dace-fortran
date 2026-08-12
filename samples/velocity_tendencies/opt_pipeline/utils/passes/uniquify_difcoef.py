"""Uniquify per-occurrence references to a thread-local scratch scalar.

``difcoef`` is thread-local in the velocity kernel: every read-modify-write
chain is confined to a single state's dataflow, and separate chains never
alias at run time. The baseline nevertheless exposes one shared ``Scalar``
in ``sdfg.arrays``, which blocks ``LoopToMap`` with ``Read-write conflict
detected for difcoef``.

This pass walks every SDFG (top-level + nested) in traversal order and
bumps a counter for each ``AccessNode(difcoef)`` encountered, creating a
fresh transient ``difcoef<N>`` (``difcoef0``, ``difcoef1``, ...) per
AccessNode. The AccessNode's ``data`` field and the memlets of every edge
touching it are updated together so the dataflow stays consistent.

Nested SDFG connector names are local; the walker uniquifies inner-scope
occurrences independently when it descends into a nested SDFG.
"""
import dace
from dace import nodes


def uniquify_difcoef(sdfg: dace.SDFG, name: str = "difcoef") -> int:
    counter = [0]
    _walk(sdfg, name, counter)
    return counter[0]


def _walk(g: dace.SDFG, name: str, counter: list):
    for state in list(g.all_states()):
        _rename_in_state(g, state, name, counter)
        for n in state.nodes():
            if isinstance(n, nodes.NestedSDFG):
                _walk(n.sdfg, name, counter)
    if name in g.arrays and not _any_reference(g, name):
        try:
            g.remove_data(name, validate=False)
        except Exception:
            g.arrays.pop(name, None)


def _rename_in_state(g: dace.SDFG, state, name: str, counter: list):
    if name not in g.arrays:
        return
    desc = g.arrays[name]
    for an in list(state.nodes()):
        if not (isinstance(an, nodes.AccessNode) and an.data == name):
            continue
        new_name = _next_name(g, name, counter)
        g.add_scalar(new_name, dtype=desc.dtype, transient=desc.transient)
        an.data = new_name
        for e in list(state.in_edges(an)) + list(state.out_edges(an)):
            if e.data is not None and e.data.data == name:
                e.data.data = new_name


def _next_name(g: dace.SDFG, base: str, counter: list) -> str:
    while True:
        candidate = f"{base}{counter[0]}"
        counter[0] += 1
        if candidate not in g.arrays:
            return candidate


def _any_reference(g: dace.SDFG, name: str) -> bool:
    for state in g.all_states():
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and n.data == name:
                return True
        for e in state.edges():
            if e.data is not None and e.data.data == name:
                return True
    return False
