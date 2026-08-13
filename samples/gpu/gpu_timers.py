# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-SDFG timers and sync control for the GPU lanes.

Instrumentation only -- no scheduling, no storage decisions. Every GPU lane bolts the same
begin/end pair on, so the ``ms`` column means the same thing whichever offload produced the
SDFG: a ``CLOCK_MONOTONIC`` read behind a device sync, before the first kernel and after the
last.

``silence_codegen_syncs`` then clears DaCe's per-state sync everywhere the host does not read
what the device just wrote, leaving the two brackets (plus whatever the offload itself asked to
keep) as the only synchronization in the run.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import dace
from dace import data as dt
from dace import dtypes
from dace import memlet as mm
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes as nd
from dace.sdfg.state import ControlFlowRegion

TIMER_BEGIN = "__gpu_t_begin"
TIMER_END = "__gpu_t_end"
SYNC_TICK = "__gpu_sync_tick"

TIMER_CODE = ("struct timespec __ts;\n"
              "clock_gettime(CLOCK_MONOTONIC, &__ts);\n"
              "__out = 1.0e3 * (double)__ts.tv_sec + 1.0e-6 * (double)__ts.tv_nsec;")
SYNC_CODE = ("cudaError_t __e = cudaDeviceSynchronize();\n"
             "if (__e != cudaSuccess) { fprintf(stderr, \"gpu sync: %s\\n\", cudaGetErrorString(__e)); }\n"
             "__out = 0.0;")
GLOBAL_CODE = "#include <time.h>\n#include <stdio.h>\n#include <cuda_runtime.h>\n"


def add_host_array(sdfg: SDFG, name: str, transient: bool) -> str:
    desc = dt.Array(dace.float64, (1, ),
                    transient=transient,
                    storage=dtypes.StorageType.CPU_Heap,
                    lifetime=dtypes.AllocationLifetime.SDFG)
    return sdfg.add_datadesc(name, desc)


def code_state(cfg: ControlFlowRegion, label: str, code: str, out_array: str) -> SDFGState:
    state = cfg.add_state(label)
    tasklet = state.add_tasklet(label, {}, {"__out"}, code, language=dtypes.Language.CPP)
    state.add_edge(tasklet, "__out", state.add_access(out_array), None, mm.Memlet(f"{out_array}[0]"))
    return state


def prepend(sdfg: SDFG, label: str, code: str, out_array: str) -> SDFGState:
    first = sdfg.start_block
    state = code_state(sdfg, label, code, out_array)
    for edge in list(sdfg.in_edges(first)):
        sdfg.remove_edge(edge)
        sdfg.add_edge(edge.src, state, edge.data)
    sdfg.add_edge(state, first, dace.InterstateEdge())
    sdfg.start_block = sdfg.node_id(state)
    return state


def append(sdfg: SDFG, label: str, code: str, out_array: str) -> SDFGState:
    sinks = list(sdfg.sink_nodes())
    state = code_state(sdfg, label, code, out_array)
    for sink in sinks:
        sdfg.add_edge(sink, state, dace.InterstateEdge())
    return state


def insert_after(cfg: ControlFlowRegion, block, label: str, code: str, out_array: str) -> SDFGState:
    state = code_state(cfg, label, code, out_array)
    for edge in list(cfg.out_edges(block)):
        cfg.remove_edge(edge)
        cfg.add_edge(state, edge.dst, edge.data)
    cfg.add_edge(block, state, dace.InterstateEdge())
    return state


def has_device_to_host_copy(state: SDFGState) -> bool:
    sdfg = state.sdfg
    for edge in state.edges():
        if not (isinstance(edge.src, nd.AccessNode) and isinstance(edge.dst, nd.AccessNode)):
            continue
        if (sdfg.arrays[edge.src.data].storage == dtypes.StorageType.GPU_Global
                and sdfg.arrays[edge.dst.data].storage != dtypes.StorageType.GPU_Global):
            return True
    return False


def silence_codegen_syncs(sdfg: SDFG, force_silent) -> List[str]:
    """No end-of-state sync anywhere the host does not read what the device just wrote."""
    kept = []
    for cur in sdfg.all_sdfgs_recursive():
        for state in cur.all_states():
            if state not in force_silent and has_device_to_host_copy(state):
                kept.append(f"{cur.name}/{state.label}")
            else:
                state.nosync = True
    return kept


def add_timers(sdfg: SDFG, force_silent: Iterable[SDFGState] = (), validate: bool = True) -> List[str]:
    """Bracket the whole SDFG in sync + timestamp, then silence the rest of the syncs.

    ``force_silent`` are states the caller already synchronizes by hand (a copy-out it brackets
    itself, a mid-SDFG host reduction); they get ``nosync`` regardless of what they copy.
    Returns the states whose codegen sync survived.
    """
    sdfg.append_global_code(GLOBAL_CODE)
    add_host_array(sdfg, SYNC_TICK, transient=True)
    add_host_array(sdfg, TIMER_BEGIN, transient=False)
    add_host_array(sdfg, TIMER_END, transient=False)

    prepend(sdfg, "__gpu_timer_begin", TIMER_CODE, TIMER_BEGIN)
    prepend(sdfg, "__gpu_sync_begin", SYNC_CODE, SYNC_TICK)
    append(sdfg, "__gpu_sync_end", SYNC_CODE, SYNC_TICK)
    append(sdfg, "__gpu_timer_end", TIMER_CODE, TIMER_END)

    kept = silence_codegen_syncs(sdfg, set(force_silent))
    if validate:
        sdfg.validate()
    return kept


def timer_arrays() -> Dict[str, object]:
    """The two host arrays the SDFG writes its own begin/end timestamps into."""
    import numpy as np
    return {TIMER_BEGIN: np.zeros(1, dtype=np.float64), TIMER_END: np.zeros(1, dtype=np.float64)}


def elapsed_ms(call: Dict[str, object]) -> float:
    return float(call[TIMER_END][0]) - float(call[TIMER_BEGIN][0])


def count_syncs(sdfg: SDFG, build_folder: str | None = None) -> Dict[str, int]:
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
