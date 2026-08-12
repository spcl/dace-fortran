"""Demote every ``int64`` / ``uint64`` to its 32-bit equivalent.

Walks the SDFG tree and rewrites:
  - ``sdfg.symbols`` — any symbol typed as int64/uint64.
  - Array descriptors' ``dtype``.
  - ``NestedSDFG`` in/out connector types.
  - Interstate-edge-assigned symbols that are otherwise undeclared.
    DaCe's codegen infers their type via ``InterstateEdge.new_symbols``,
    and a literal like ``lhs = 1`` lands on ``int64`` (Python ``int``
    default). Left unfixed, they surface in the emitted C++ as
    ``int64_t lhs;`` local declarations and ``int64_t`` parameter types
    in generated ``__dace_runkernel_*`` prototypes. We register them in
    ``sdfg.symbols`` as ``int32`` so the framecode's interstate-symbol
    pass picks up the narrower type.

Fixes the ``conversion from 'long int' to 'int' may change value`` warnings
emitted by the C++ toolchain when DaCe's default 64-bit loop / map params
are passed to a callee signature declared as ``int``.
"""
import dace
from dace.sdfg import nodes


_I64_TO_I32 = {dace.int64: dace.int32, dace.uint64: dace.uint32}


def int64_to_int32(sdfg: dace.SDFG) -> int:
    count = 0
    for g in _all_sdfgs(sdfg):
        for arr in g.arrays.values():
            new = _I64_TO_I32.get(arr.dtype)
            if new is not None:
                arr.dtype = new
                count += 1
        for sym_name, sym_type in list(g.symbols.items()):
            new = _I64_TO_I32.get(sym_type)
            if new is not None:
                g.symbols[sym_name] = new
                count += 1
        for state in g.all_states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                for conn, t in list(node.in_connectors.items()):
                    new = _I64_TO_I32.get(t)
                    if new is not None:
                        node.in_connectors[conn] = new
                        count += 1
                for conn, t in list(node.out_connectors.items()):
                    new = _I64_TO_I32.get(t)
                    if new is not None:
                        node.out_connectors[conn] = new
                        count += 1
        count += _register_undeclared_interstate_symbols(g)
    return count


def _register_undeclared_interstate_symbols(g: dace.SDFG) -> int:
    """Declare any interstate-assigned symbol not in ``g.symbols``/``g.arrays``.

    Only int64/uint64 inferences are registered (as int32/uint32); non-int
    inferences (e.g. a ``tmp_call`` carrying a double) are left to the
    normal codegen path.
    """
    count = 0
    declared_seed = dict(g.symbols)
    declared_seed.update({k: v.dtype for k, v in g.arrays.items()})
    for e in g.all_interstate_edges():
        try:
            inferred = e.data.new_symbols(g, declared_seed)
        except Exception:
            continue
        for lhs, t in inferred.items():
            if lhs in g.symbols or lhs in g.arrays:
                continue
            narrowed = _I64_TO_I32.get(t)
            if narrowed is None:
                continue
            g.symbols[lhs] = narrowed
            declared_seed[lhs] = narrowed
            count += 1
    return count


def _all_sdfgs(sdfg: dace.SDFG):
    yield sdfg
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            yield n.sdfg
