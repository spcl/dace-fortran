# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual GPU offload for the sample SDFGs, plus the in-SDFG timers the GPU lanes are measured with.

``apply_gpu_offload(sdfg, kind)`` puts every array on the device, schedules the outermost map of
every nest on ``GPU_Device`` (everything below it Sequential), and leaves exactly two
synchronization points -- one before the first kernel, one after the last -- with a
``CLOCK_MONOTONIC`` tasklet behind each. ``kind='velocity'`` adds the one mid-SDFG sync the
host-side block reduction needs.

A map whose range mentions a block symbol (``*blk*``/``*block*``) is the outer blocking loop
rather than parallelism. ``demote_block_maps`` turns those into sequential loops so the device
schedule lands on the maps inside them; see ``apply_gpu_offload`` for why that is the default for
cloudsc but not for velocity.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import dace
from dace import data as dt
from dace import dtypes
from dace import memlet as mm
from dace import symbolic
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.state import ControlFlowRegion, LoopRegion
from dace.transformation import transformation
from dace.transformation.dataflow import MapToForLoop
from dace.transformation.interstate import GPUTransformSDFG

TIMER_BEGIN = "__gpu_t_begin"
TIMER_END = "__gpu_t_end"
SYNC_TICK = "__gpu_sync_tick"
BLOCK_TOKENS = ("blk", "block")

_TIMER_CODE = ("struct timespec __ts;\n"
               "clock_gettime(CLOCK_MONOTONIC, &__ts);\n"
               "__out = 1.0e3 * (double)__ts.tv_sec + 1.0e-6 * (double)__ts.tv_nsec;")
_SYNC_CODE = ("cudaError_t __e = cudaDeviceSynchronize();\n"
              "if (__e != cudaSuccess) { fprintf(stderr, \"gpu sync: %s\\n\", cudaGetErrorString(__e)); }\n"
              "__out = 0.0;")
_GLOBAL_CODE = "#include <time.h>\n#include <stdio.h>\n#include <cuda_runtime.h>\n"


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


def find_host_reductions(sdfg: SDFG) -> List[Tuple[SDFGState, nd.Tasklet, List[str], List[str]]]:
    """Loop-carried folds of an array into a scalar, outside any map: the host-side reductions."""
    found = []
    for state in sdfg.states():
        if not isinstance(state.parent_graph, LoopRegion):
            continue
        for node in state.nodes():
            if not isinstance(node, nd.Tasklet) or state.entry_node(node) is not None:
                continue
            arrays = [
                e.src.data for e in state.in_edges(node)
                if isinstance(e.src, nd.AccessNode) and isinstance(sdfg.arrays[e.src.data], dt.Array)
            ]
            scalars = [
                e.dst.data for e in state.out_edges(node)
                if isinstance(e.dst, nd.AccessNode) and isinstance(sdfg.arrays[e.dst.data], dt.Scalar)
            ]
            if arrays and scalars:
                found.append((state, node, sorted(set(arrays)), sorted(set(scalars))))
    return found


def _stage_to_host(sdfg: SDFG, state: SDFGState, arrays: Sequence[str]) -> Tuple[List[str], SDFGState]:
    """Give the host reduction its own host copy of ``arrays``, filled right before the loop."""
    loop = state.parent_graph
    parent: ControlFlowRegion = loop.parent_graph
    stage = parent.add_state_before(loop, "__gpu_stage_host_reduction", is_start_block=parent.start_block is loop)
    host_names = []
    for name in arrays:
        desc = sdfg.arrays[name].clone()
        desc.transient = True
        desc.storage = dtypes.StorageType.CPU_Heap
        desc.lifetime = dtypes.AllocationLifetime.SDFG
        hostname = sdfg.add_datadesc(f"{name}_host", desc, find_new_name=True)
        host_names.append(hostname)
        stage.add_nedge(stage.add_access(name), stage.add_access(hostname),
                        mm.Memlet.from_array(name, sdfg.arrays[name]))
        for node in state.nodes():
            if isinstance(node, nd.AccessNode) and node.data == name:
                node.data = hostname
        for edge in state.edges():
            if edge.data.data == name:
                edge.data.data = hostname
    return host_names, stage


def _add_host_array(sdfg: SDFG, name: str, transient: bool) -> str:
    desc = dt.Array(dace.float64, (1, ),
                    transient=transient,
                    storage=dtypes.StorageType.CPU_Heap,
                    lifetime=dtypes.AllocationLifetime.SDFG)
    return sdfg.add_datadesc(name, desc)


def _code_state(cfg: ControlFlowRegion, label: str, code: str, out_array: str) -> SDFGState:
    state = cfg.add_state(label)
    tasklet = state.add_tasklet(label, {}, {"__out"}, code, language=dtypes.Language.CPP)
    state.add_edge(tasklet, "__out", state.add_access(out_array), None, mm.Memlet(f"{out_array}[0]"))
    return state


def _prepend(sdfg: SDFG, label: str, code: str, out_array: str) -> SDFGState:
    first = sdfg.start_block
    state = _code_state(sdfg, label, code, out_array)
    for edge in list(sdfg.in_edges(first)):
        sdfg.remove_edge(edge)
        sdfg.add_edge(edge.src, state, edge.data)
    sdfg.add_edge(state, first, dace.InterstateEdge())
    sdfg.start_block = sdfg.node_id(state)
    return state


def _append(sdfg: SDFG, label: str, code: str, out_array: str) -> SDFGState:
    sinks = [b for b in sdfg.sink_nodes()]
    state = _code_state(sdfg, label, code, out_array)
    for sink in sinks:
        sdfg.add_edge(sink, state, dace.InterstateEdge())
    return state


def _insert_after(cfg: ControlFlowRegion, block, label: str, code: str, out_array: str) -> SDFGState:
    state = _code_state(cfg, label, code, out_array)
    for edge in list(cfg.out_edges(block)):
        cfg.remove_edge(edge)
        cfg.add_edge(state, edge.dst, edge.data)
    cfg.add_edge(block, state, dace.InterstateEdge())
    return state


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


def _is_len1(desc) -> bool:
    return isinstance(desc, dt.Scalar) or (isinstance(desc, dt.Array) and str(desc.total_size) == "1")


def host_scalar_tail(sdfg: SDFG, seeds: Sequence[str]) -> List[str]:
    """The len-1 containers the host reduction's tail tasklets pass its result through.

    Everything the tail touches has to stay on the host together: one len-1 argument left to
    GPUTransformSDFG's cloning drags the tail's tasklets into a one-thread kernel, and the
    accumulator they share then has a device writer and a host writer.
    """
    frontier = set(seeds)
    found: set = set()
    changed = True
    while changed:
        changed = False
        for state in sdfg.states():
            for node in state.nodes():
                if not isinstance(node, nd.Tasklet) or state.entry_node(node) is not None:
                    continue
                touched = ({e.src.data for e in state.in_edges(node) if isinstance(e.src, nd.AccessNode)}
                           | {e.dst.data for e in state.out_edges(node) if isinstance(e.dst, nd.AccessNode)})
                if not touched & frontier:
                    continue
                for name in touched - frontier:
                    if _is_len1(sdfg.arrays[name]):
                        frontier.add(name)
                        found.add(name)
                        changed = True
    return sorted(found | set(seeds))


def _as_scalars(sdfg: SDFG, names: Sequence[str]) -> Dict[str, object]:
    """Present len-1 array ARGUMENTS as scalars, which GPUTransformSDFG leaves on the host."""
    saved = {}
    for name in names:
        desc = sdfg.arrays[name]
        if isinstance(desc, dt.Array) and not desc.transient and _is_len1(desc):
            saved[name] = desc
            sdfg.arrays[name] = dt.Scalar(desc.dtype, transient=False, storage=desc.storage)
    return saved


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


def _has_device_to_host_copy(state: SDFGState) -> bool:
    sdfg = state.sdfg
    for edge in state.edges():
        if not (isinstance(edge.src, nd.AccessNode) and isinstance(edge.dst, nd.AccessNode)):
            continue
        if (sdfg.arrays[edge.src.data].storage == dtypes.StorageType.GPU_Global
                and sdfg.arrays[edge.dst.data].storage != dtypes.StorageType.GPU_Global):
            return True
    return False


def _silence_codegen_syncs(sdfg: SDFG, force_silent) -> List[str]:
    """No end-of-state sync anywhere the host does not read what the device just wrote."""
    kept = []
    for cur in sdfg.all_sdfgs_recursive():
        for state in cur.all_states():
            if state not in force_silent and _has_device_to_host_copy(state):
                kept.append(f"{cur.name}/{state.label}")
            else:
                state.nosync = True
    return kept


def apply_gpu_offload(sdfg: SDFG,
                      kind: str,
                      validate: bool = True,
                      demote_block_maps: Optional[bool] = None) -> Dict[str, object]:
    """Offload ``sdfg`` to the GPU in place; returns what was decided, for the runner to print.

    :param kind: ``cloudsc`` (device-resident whole computation) or ``velocity`` (same, plus the
                 host block reduction kept on the CPU behind one extra sync).
    :param demote_block_maps: turn top-level block maps into sequential loops so the device
                              schedule lands on the maps inside them. Defaults to True for
                              cloudsc (whose block loop is already a loop region, so it is a
                              no-op guard) and False for velocity, where the block map is the
                              scope that holds ALL of the kernel's data-dependent control flow:
                              demoting it makes that control flow host code reading device
                              arrays, which costs a whole-array copy back per iteration.
    """
    if kind not in ("cloudsc", "velocity"):
        raise ValueError(f"unknown offload kind {kind!r}")
    if demote_block_maps is None:
        demote_block_maps = kind == "cloudsc"

    host_data: List[str] = []
    staged: List[Tuple[SDFGState, List[str]]] = []
    reductions = find_host_reductions(sdfg) if kind == "velocity" else []
    for state, tasklet, arrays, scalars in reductions:
        host_names, _ = _stage_to_host(sdfg, state, arrays)
        host_data.extend(host_names)
        host_data.extend(host_scalar_tail(sdfg, scalars))
        staged.append((state, host_names))
    if kind == "velocity" and not staged:
        raise RuntimeError("velocity offload: no host reduction found -- the mid sync has no anchor")

    demoted = _demote_block_maps(sdfg) if demote_block_maps else _block_map_labels(sdfg)
    released = _release_host_transients(sdfg, host_data)
    premarked = _premark_device_transients(sdfg, host_data)
    buffered = _break_code_to_code(sdfg)
    saved = _as_scalars(sdfg, host_data)
    applied = sdfg.apply_transformations(GPUTransformSDFG,
                                         options=dict(sequential_innermaps=True,
                                                      register_trans=True,
                                                      simplify=False,
                                                      host_data=host_data),
                                         validate=False)
    sdfg.arrays.update(saved)
    if applied != 1:
        raise RuntimeError("GPUTransformSDFG did not apply -- nothing was offloaded "
                           "(can_be_applied refused; check for consume scopes or code->code memlets)")

    sched: List[Tuple[str, str, str]] = []
    _schedule_sdfg(sdfg, False, demote_block_maps, sched)

    sdfg.append_global_code(_GLOBAL_CODE)
    _add_host_array(sdfg, SYNC_TICK, transient=True)
    _add_host_array(sdfg, TIMER_BEGIN, transient=False)
    _add_host_array(sdfg, TIMER_END, transient=False)

    force_silent = set()
    mid_syncs = []
    for state, _ in staged:
        loop = state.parent_graph
        stage_state = next(iter(loop.parent_graph.in_edges(loop))).src
        force_silent.add(stage_state)
        mid_syncs.append(
            _insert_after(loop.parent_graph, stage_state, "__gpu_sync_host_reduction", _SYNC_CODE, SYNC_TICK).label)
    force_silent.update(b for b in sdfg.nodes() if isinstance(b, SDFGState) and b.label.endswith("_copyout"))

    _prepend(sdfg, "__gpu_timer_begin", _TIMER_CODE, TIMER_BEGIN)
    _prepend(sdfg, "__gpu_sync_begin", _SYNC_CODE, SYNC_TICK)
    _append(sdfg, "__gpu_sync_end", _SYNC_CODE, SYNC_TICK)
    _append(sdfg, "__gpu_timer_end", _TIMER_CODE, TIMER_END)

    codegen_syncs = _silence_codegen_syncs(sdfg, force_silent)
    if validate:
        sdfg.validate()

    gpu_maps = [r for r in sched if r[2].endswith("GPU_Device")]
    return {
        "gpu_maps": len(gpu_maps),
        "block_maps": demoted,
        "demoted": demote_block_maps,
        "host_data": host_data,
        "released": released,
        "premarked": premarked,
        "buffered": buffered,
        "mid_syncs": mid_syncs,
        "codegen_syncs": codegen_syncs,
        "schedules": sched,
    }


def timer_arrays() -> Dict[str, object]:
    """The two host arrays the SDFG writes its own begin/end timestamps into."""
    import numpy as np
    return {TIMER_BEGIN: np.zeros(1, dtype=np.float64), TIMER_END: np.zeros(1, dtype=np.float64)}


def elapsed_ms(call: Dict[str, object]) -> float:
    return float(call[TIMER_END][0]) - float(call[TIMER_BEGIN][0])


def count_syncs(sdfg: SDFG, build_folder: Optional[str] = None) -> Dict[str, int]:
    """Synchronization calls in the generated code, by kind -- the check behind the 2-sync rule."""
    import copy
    import re
    from dace.codegen import codegen
    counts: Dict[str, int] = {}
    for obj in codegen.generate_code(copy.deepcopy(sdfg)):
        for kind in ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize"):
            n = len(re.findall(kind, obj.clean_code))
            if n:
                counts[f"{obj.name}.{obj.language}:{kind}"] = n
    return counts
