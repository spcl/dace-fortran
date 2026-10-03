# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared base and helpers for the Fortran I/O library nodes."""

from typing import List, NamedTuple

from dace import SDFG, SDFGState, data, dtypes
from dace.sdfg import nodes
from dace.subsets import Range


class FioType(NamedTuple):
    suffix: str
    ctype: str


class IoItem(NamedTuple):
    connector: str
    descriptor: data.Data
    elements: str
    is_value: bool


#: DaCe base type -> (``dace_fio_*`` entry suffix, C scalar type) for the
#: shipped wrappers.  The suffix selects the typed ``read``/``write`` entry; the
#: C type is the pointer cast at the call site (so int64 vs ``long long`` and
#: similar width spellings never trip ``-Werror``).
_FIO_TYPES = {
    dtypes.float64: FioType("f64", "double"),
    dtypes.float32: FioType("f32", "float"),
    dtypes.int32: FioType("i32", "int"),
    dtypes.int64: FioType("i64", "long long"),
}


def fio_type(dtype: dtypes.typeclass) -> FioType:
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

    def has_side_effects(self, sdfg: SDFG) -> bool:
        return True

    def ordered_items(self, sdfg: SDFG, state: SDFGState, prefix: str, edges_in: bool, num_items: int) -> List[IoItem]:
        """Resolve the ``num_items`` connected I/O items in connector order, as ``(connector,
        descriptor, count, is_value)``.  ``is_value`` marks a scalar/single-element
        connector (emitted by value, so the call site takes its address)."""
        if edges_in:
            edges = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn}
        else:
            edges = {e.src_conn: e for e in state.out_edges(self) if e.src_conn}
        items: List[IoItem] = []
        for i in range(num_items):
            conn = f"{prefix}{i}"
            edge = edges.get(conn)
            if edge is None:
                raise ValueError(f"{type(self).__name__} '{self.name}': item connector '{conn}' is not connected")
            subset = edge.data.subset
            if edge.data.data is None or not isinstance(subset, Range):
                raise ValueError(f"{type(self).__name__} '{self.name}': connector '{conn}' carries no range memlet")
            desc = sdfg.arrays[edge.data.data]
            is_value = isinstance(desc, data.Scalar) or subset.num_elements() == 1
            count = "*".join(str(s) for s in subset.size_exact()) or "1"
            items.append(IoItem(conn, desc, count, is_value))
        return items
