# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared base and helpers for the Fortran I/O library nodes."""
from typing import List, Tuple

from dace import data, dtypes
from dace.sdfg import nodes

#: DaCe base type -> (``dace_fio_*`` entry suffix, C scalar type) for the
#: shipped wrappers.  The suffix selects the typed ``read``/``write`` entry; the
#: C type is the pointer cast at the call site (so int64 vs ``long long`` and
#: similar width spellings never trip ``-Werror``).
_FIO_TYPES = {
    dtypes.float64: ("f64", "double"),
    dtypes.float32: ("f32", "float"),
    dtypes.int32: ("i32", "int"),
    dtypes.int64: ("i64", "long long"),
}


def fio_type(dtype) -> Tuple[str, str]:
    """Resolve the ``dace_fio_*`` wrapper suffix and C cast type for ``dtype``."""
    base = dtype.base_type
    if base not in _FIO_TYPES:
        raise NotImplementedError(f"fortran_io has no wrapper for dtype {dtype}")
    return _FIO_TYPES[base]


class FortranIONode(nodes.LibraryNode):
    """Abstract base for a Fortran external-file I/O library node.

    Fortran I/O has observable side effects (it touches the file system), so
    these nodes report :py:meth:`has_side_effects` and must never be removed as
    dead code even when, like ``WRITE``, they have no output connectors.
    """

    def has_side_effects(self, sdfg) -> bool:
        return True

    def _ordered_items(self, sdfg, state, prefix: str, edges_in: bool) -> List[Tuple[str, object, str, bool]]:
        """Resolve the connected I/O items in connector order, as ``(connector,
        descriptor, count, is_value)``.  ``is_value`` marks a scalar/single-element
        connector (emitted by value, so the call site takes its address)."""
        if edges_in:
            edges = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn}
        else:
            edges = {e.src_conn: e for e in state.out_edges(self) if e.src_conn}
        items = []
        for i in range(self.num_items):
            conn = f"{prefix}{i}"
            edge = edges.get(conn)
            if edge is None:
                raise ValueError(f"{type(self).__name__} '{self.name}': item connector '{conn}' is not connected")
            desc = sdfg.arrays[edge.data.data]
            num_elements = edge.data.subset.num_elements()
            is_value = isinstance(desc, data.Scalar) or num_elements == 1
            count = "*".join(str(s) for s in edge.data.subset.size_exact()) or "1"
            items.append((conn, desc, count, is_value))
        return items
