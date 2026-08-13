# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual GPU offload for **cloudsc**.

``apply_gpu_offload(sdfg)`` puts every array on the device, schedules the outermost map of every
nest on ``GPU_Device`` (everything below it Sequential), and leaves exactly two synchronization
points -- one before the first kernel, one after the last.

A map whose range mentions a block symbol (``*blk*``/``*block*``) is the outer blocking loop
rather than parallelism; ``demote_block_maps`` turns those into sequential loops so the device
schedule lands on the maps inside them. cloudsc's block loop is already a loop region, so that
is a no-op guard there.

**Velocity does not come here.** It is offloaded by ``OffloadVelocityToGPU`` alone -- see
``velocity_offload``. This module's ``GPUTransformSDFG`` call is reachable from cloudsc only.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import dace
from dace import data as dt
from dace import dtypes
from dace import memlet as mm
from dace import symbolic
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.transformation import transformation
from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate import GPUTransformSDFG

from gpu_timers import add_timers

BLOCK_TOKENS = ("blk", "block")


def _is_block_symbol(name: str) -> bool:
    low = str(name).lower()
    return any(tok in low for tok in BLOCK_TOKENS)


def _is_block_map(entry: nd.MapEntry) -> bool:
    """True when this map iterates the blocking dimension (``nblks``/``i_startblk:i_endblk``)."""
    if any(_is_block_symbol(p) for p in entry.map.params):
        return True
    for rng in entry.map.range:
        for expr in rng:
            if any(_is_block_symbol(s) for s in symbolic.symlist(expr)):
                return True
    return False


@transformation.explicit_cf_compatible
class BlockMapToForLoop(MapToForLoop):
    """MapToForLoop restricted to the outermost blocking maps (and CF-region aware, as it builds one)."""

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return (graph.entry_node(self.map_entry) is None and _is_block_map(self.map_entry)
                and super().can_be_applied(graph, expr_index, sdfg, permissive))


def _block_map_labels(sdfg: SDFG) -> List[str]:
    return [
        f"{state.sdfg.name}/{state.label}:{node.map.label}{list(node.map.range)}" for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, nd.MapEntry) and state.entry_node(node) is None and _is_block_map(node)
    ]


def _demote_block_maps(sdfg: SDFG) -> List[str]:
    """Turn every top-level block map into a sequential loop, so the device schedule lands inside it."""
    labels = _block_map_labels(sdfg)
    if labels:
        sdfg.apply_transformations_repeated(BlockMapToForLoop, validate=False)
        left = _block_map_labels(sdfg)
        if left:
            raise RuntimeError(f"block maps that could not be demoted to loops: {left}")
    return labels


def _schedule_scope(state: SDFGState, entry: nd.MapEntry, under_gpu: bool, host_blocks: bool,
                    report: List[Tuple[str, str, str]]) -> None:
    if under_gpu:
        entry.map.schedule = dtypes.ScheduleType.Sequential
        kind = "inner -> Sequential"
        child_gpu = True
    elif host_blocks and _is_block_map(entry):
        entry.map.schedule = dtypes.ScheduleType.Sequential
        kind = "BLOCK map -> host Sequential"
        child_gpu = False
    else:
        entry.map.schedule = dtypes.ScheduleType.GPU_Device
        kind = "top -> GPU_Device"
        child_gpu = True
    report.append((f"{state.sdfg.name}/{state.label}", entry.map.label, kind))
    for child in state.scope_children()[entry]:
        _schedule_node(state, child, child_gpu, host_blocks, report)


def _schedule_node(state: SDFGState, node: nd.Node, under_gpu: bool, host_blocks: bool,
                   report: List[Tuple[str, str, str]]) -> None:
    if isinstance(node, nd.MapEntry):
        _schedule_scope(state, node, under_gpu, host_blocks, report)
    elif isinstance(node, nd.NestedSDFG):
        _schedule_sdfg(node.sdfg, under_gpu, host_blocks, report)


def _schedule_sdfg(sdfg: SDFG, under_gpu: bool, host_blocks: bool, report: List[Tuple[str, str, str]]) -> None:
    for state in sdfg.states():
        for node in state.scope_children()[None]:
            _schedule_node(state, node, under_gpu, host_blocks, report)


def schedule_report(sdfg: SDFG) -> List[Tuple[str, str, str, str]]:
    """Every map in ``sdfg`` (nested SDFGs included) as (location, label, range, schedule)."""
    rows: List[Tuple[str, str, str, str]] = []

    def walk(cur: SDFG, depth: int) -> None:
        for state in cur.states():
            for node in state.nodes():
                if isinstance(node, nd.MapEntry):
                    lvl = "  " * (depth + (0 if state.entry_node(node) is None else 1))
                    rows.append((f"{lvl}{cur.name}/{state.label}", node.map.label, str(node.map.range),
                                 str(node.map.schedule).replace("ScheduleType.", "")))
                elif isinstance(node, nd.NestedSDFG):
                    walk(node.sdfg, depth + 1)

    walk(sdfg, 0)
    return rows


def _code_to_code_components(state: SDFGState) -> List[List[nd.CodeNode]]:
    """Groups of top-level tasklets wired to each other by tasklet->tasklet memlets."""
    parent: Dict[nd.Node, nd.Node] = {}

    def find(x):
        while parent.get(x, x) is not x:
            x = parent[x]
        return x

    seen = set()
    for edge in state.edges():
        if not (isinstance(edge.src, nd.CodeNode) and isinstance(edge.dst, nd.CodeNode)):
            continue
        if state.entry_node(edge.src) is not None or state.entry_node(edge.dst) is not None:
            continue
        a, b = find(edge.src), find(edge.dst)
        parent[edge.src], parent[edge.dst] = a, a
        seen.update((edge.src, edge.dst))
        if a is not b:
            parent[b] = a
    groups: Dict[nd.Node, List[nd.CodeNode]] = {}
    for node in seen:
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def _wrap_in_gpu_map(state: SDFGState, nodes_in: List[nd.CodeNode], label: str) -> None:
    """Put ``nodes_in`` inside one size-1 GPU map, rerouting only the edges crossing the group."""
    inside = set(nodes_in)
    entry, exit_ = state.add_map(label, {f"{label}_i": "0:1"}, schedule=dtypes.ScheduleType.GPU_Device)
    n_in = n_out = 0
    for node in nodes_in:
        for edge in list(state.in_edges(node)):
            if edge.src in inside:
                continue
            state.remove_edge(edge)
            if edge.data.is_empty():
                state.add_nedge(edge.src, entry, mm.Memlet())
                state.add_nedge(entry, node, mm.Memlet())
                continue
            conn = f"IN_{n_in}"
            entry.add_in_connector(conn)
            entry.add_out_connector(f"OUT_{n_in}")
            state.add_edge(edge.src, edge.src_conn, entry, conn, edge.data)
            state.add_edge(entry, f"OUT_{n_in}", node, edge.dst_conn, dace.Memlet.from_memlet(edge.data))
            n_in += 1
        for edge in list(state.out_edges(node)):
            if edge.dst in inside:
                continue
            state.remove_edge(edge)
            if edge.data.is_empty():
                state.add_nedge(node, exit_, mm.Memlet())
                state.add_nedge(exit_, edge.dst, mm.Memlet())
                continue
            conn = f"IN_{n_out}"
            exit_.add_in_connector(conn)
            exit_.add_out_connector(f"OUT_{n_out}")
            state.add_edge(node, edge.src_conn, exit_, conn, edge.data)
            state.add_edge(exit_, f"OUT_{n_out}", edge.dst, edge.dst_conn, dace.Memlet.from_memlet(edge.data))
            n_out += 1
        if not state.in_edges(node):
            state.add_nedge(entry, node, mm.Memlet())
        if not state.out_edges(node):
            state.add_nedge(node, exit_, mm.Memlet())


def _break_code_to_code(sdfg: SDFG) -> List[str]:
    """Put every top-level tasklet->tasklet group inside its own size-1 GPU map.

    ``GPUTransformSDFG.can_be_applied`` refuses an SDFG that has any such pair at the top level of
    a state, and ``apply_transformations`` reports the refusal by doing nothing at all. Its own
    wrapping step (step 7) cannot do this itself: it assumes every edge of a wrapped tasklet
    carries a connector, and these groups are held together by empty ordering edges.
    """
    wrapped = []
    for state in sdfg.states():
        for i, comp in enumerate(_code_to_code_components(state)):
            _wrap_in_gpu_map(state, comp, f"{state.label}_c2c{i}_gmap")
            wrapped.append(f"{state.label}:{'+'.join(n.label for n in comp)}")
    return wrapped


def _release_host_transients(sdfg: SDFG, host_data: Sequence[str]) -> List[str]:
    """Un-pin the CPU-heap transients the pipeline's persistent-transient pass created.

    ``MakeTransientsPersistent`` gives them an explicit CPU_Heap storage, which inside a kernel is
    an illegal copy (host storage under a device schedule). Back to Default lets the device
    schedule place them: GPU_Global at the top level, registers/shared inside a kernel.
    """
    released = []
    host = set(host_data)
    for cur in sdfg.all_sdfgs_recursive():
        for name, desc in cur.arrays.items():
            if (desc.transient and name not in host
                    and desc.storage in (dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal)):
                desc.storage = dtypes.StorageType.Default
                if desc.lifetime == dtypes.AllocationLifetime.Persistent:
                    desc.lifetime = dtypes.AllocationLifetime.Scope
                released.append(f"{cur.name}:{name}")
    return released


def _host_level_sdfgs(sdfg: SDFG) -> List[SDFG]:
    """``sdfg`` and every nested SDFG reachable without crossing a map scope: the host-code side."""
    out = [sdfg]
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nd.NestedSDFG) and state.entry_node(node) is None:
                out.extend(_host_level_sdfgs(node.sdfg))
    return out


def _premark_device_transients(sdfg: SDFG, host_data: Sequence[str]) -> List[str]:
    """Transients that host control flow reads must be GPU_Global BEFORE GPUTransformSDFG runs.

    The transform only builds host copies for interstate-edge reads of data it knows is on the
    device (cloned arguments, GPU scalars, already-device data). A transient it promotes itself in
    its step 6 misses that list, and the SDFG fails validation with "read an inaccessible data
    container in host code interstate edge". Marking them up front puts them on the list.
    """
    marked = []
    for cur in _host_level_sdfgs(sdfg):
        names = set()
        for edge in cur.all_interstate_edges():
            names.update(m.data for m in edge.data.get_read_memlets(cur.arrays))
        for blk in cur.all_control_flow_blocks():
            if hasattr(blk, "get_meta_read_memlets"):
                names.update(m.data for m in blk.get_meta_read_memlets())
        for name in sorted(names):
            desc = cur.arrays[name]
            if (desc.transient and isinstance(desc, dt.Array) and not isinstance(desc, dt.Scalar)
                    and name not in host_data and desc.storage != dtypes.StorageType.GPU_Global):
                desc.storage = dtypes.StorageType.GPU_Global
                marked.append(f"{cur.name}:{name}")
    return marked


def apply_gpu_offload(sdfg: SDFG, kind: str = "cloudsc", validate: bool = True) -> Dict[str, object]:
    """Offload cloudsc to the GPU in place; returns what was decided, for the runner to print."""
    if kind != "cloudsc":
        raise ValueError(f"gpu_offload handles cloudsc only, not {kind!r} -- velocity is offloaded by "
                         "OffloadVelocityToGPU, through velocity_offload.offload")

    demoted = _demote_block_maps(sdfg)
    released = _release_host_transients(sdfg, ())
    premarked = _premark_device_transients(sdfg, ())
    buffered = _break_code_to_code(sdfg)
    applied = sdfg.apply_transformations(GPUTransformSDFG,
                                         options=dict(sequential_innermaps=True,
                                                      register_trans=True,
                                                      simplify=False,
                                                      host_data=[]),
                                         validate=False)
    if applied != 1:
        raise RuntimeError("GPUTransformSDFG did not apply -- nothing was offloaded "
                           "(can_be_applied refused; check for consume scopes or code->code memlets)")

    sched: List[Tuple[str, str, str]] = []
    _schedule_sdfg(sdfg, False, True, sched)

    # A copy-out state ends in a device->host copy by construction; the end-of-run sync covers it.
    force_silent = {b for b in sdfg.nodes() if isinstance(b, SDFGState) and b.label.endswith("_copyout")}
    codegen_syncs = add_timers(sdfg, force_silent, validate=validate)

    gpu_maps = [r for r in sched if r[2].endswith("GPU_Device")]
    return {
        "gpu_maps": len(gpu_maps),
        "block_maps": demoted,
        "demoted": True,
        "host_data": [],
        "released": released,
        "premarked": premarked,
        "buffered": buffered,
        "mid_syncs": [],
        "codegen_syncs": codegen_syncs,
        "schedules": sched,
    }
