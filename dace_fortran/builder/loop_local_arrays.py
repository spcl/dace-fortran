# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Place the arrays a loop owns on the stack.

A Fortran ``ALLOCATE`` / automatic array inside a ``DO`` body becomes a scope-lifetime transient whose
size is often a runtime value; on the heap it costs an allocation per iteration.  An array that is
only ever touched inside one outermost loop is private to that loop, so it is marked
``StorageType.Register(dynamic=True)``, which places it on the stack whatever its size; a symbolic size
becomes a variable-length array.
"""

from __future__ import annotations

from dace import data, dtypes
from dace.sdfg import SDFG
from dace.sdfg.state import ControlFlowBlock, LoopRegion, SDFGState


def _outermost_loop(block: ControlFlowBlock) -> LoopRegion | None:
    """The outermost ``LoopRegion`` of ``block``'s own SDFG that contains it (the block itself included)."""
    outermost = None
    region: ControlFlowBlock | None = block
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion):
            outermost = region
        region = region.parent_graph
    return outermost


def place_loop_local_arrays_on_stack(sdfg: SDFG) -> None:
    """Mark every scope-lifetime transient array accessed within a single outermost loop (and nowhere
    else) of its SDFG as ``StorageType.Register(dynamic=True)``."""
    for sd in sdfg.all_sdfgs_recursive():
        owners: dict[str, set[LoopRegion | None]] = {}

        def own(names: set[str], loop: LoopRegion | None) -> None:
            for name in names & sd.arrays.keys():
                owners.setdefault(name, set()).add(loop)

        for block in sd.all_control_flow_blocks():
            if isinstance(block, SDFGState):
                own({node.data for node in block.data_nodes()}, _outermost_loop(block))
            else:
                own(block.used_symbols(all_symbols=True, with_contents=False), _outermost_loop(block))
        for edge in sd.all_interstate_edges():
            own(edge.data.read_symbols(), _outermost_loop(edge.src))

        for name, loops in owners.items():
            desc = sd.arrays[name]
            if (
                type(desc) is data.Array
                and desc.transient
                and desc.lifetime == dtypes.AllocationLifetime.Scope
                and desc.storage in (dtypes.StorageType.Default, dtypes.StorageType.Register)
                and len(loops) == 1
                and None not in loops
            ):
                desc.storage = dtypes.StorageType.Register(dynamic=True)
