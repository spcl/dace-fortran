"""Set every transient array's allocation lifetime to ``SDFG``.

Walks the whole SDFG tree (top-level + every ``NestedSDFG.sdfg``) and, for
each descriptor with ``transient=True``, assigns
``dace.dtypes.AllocationLifetime.SDFG``. Non-transient descriptors are
left untouched.

Lifetime ``SDFG`` = allocated once when the innermost enclosing SDFG is
entered, freed when it exits. This is the right default for per-invocation
scratch; promoting to ``Persistent`` (carrying storage across SDFG calls)
is deferred until GPU-side init/exit bookends are decided by the
replacement codegen.
"""
import dace
from dace import dtypes
from dace.sdfg import nodes


def set_transient_sdfg_lifetime(sdfg: dace.SDFG) -> int:
    count = 0
    for g in _all_sdfgs(sdfg):
        for name, desc in g.arrays.items():
            if desc.transient and desc.lifetime != dtypes.AllocationLifetime.SDFG:
                desc.lifetime = dtypes.AllocationLifetime.SDFG
                count += 1
    return count


def _all_sdfgs(sdfg: dace.SDFG):
    yield sdfg
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            yield n.sdfg
