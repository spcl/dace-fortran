# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The velocity GPU offload: ``OffloadVelocityToGPU``, and nothing else.

Both GPU velocity lanes go through this module, so they differ only in the SDFG they hand it:

    dace-gpu-pipeline   frontend SDFG -> velocity_pipeline.optimize_velocity -> here
    dace-gpu-manual     VTP stage-3 artifact -> stage4.optimization_action (which calls
                        ``OffloadVelocityToGPU`` itself) -> ``post_fix`` here

``OffloadVelocityToGPU`` lives in the VelocityTendenciesPipeline checkout (``VTP_DIR``, in-tree
at ``samples/velocity_tendencies/opt_pipeline``) and is the ONLY thing that decides velocity's
device schedule and storage. ``GPUTransformSDFG`` is never called from the velocity lanes.

``post_fix`` then reconciles the pass's output with this DaCe branch: the pass targets a branch
with coherent memory and ``gpu*`` aliases, this one has neither. Five adjustments, none of which
touches a schedule or a storage decision.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]
VTP_INTREE = REPO / "samples" / "velocity_tendencies" / "opt_pipeline"
HEADERS = ("reductions_cpu.h", "reductions_kernel.cuh", "reductions_device.cuh")


def vtp_root() -> Path:
    """The VelocityTendenciesPipeline checkout: ``VTP_DIR``, else the in-tree copy."""
    return Path(os.environ.get("VTP_DIR", str(VTP_INTREE)))


def _require_root() -> Path:
    root = vtp_root()
    if not root.is_dir():
        raise SystemExit(f"no VTP checkout at {root} (set VTP_DIR)")
    if not (root / "utils" / "passes" / "offload_velocity_to_gpu.py").is_file():
        raise SystemExit(f"{root} has no utils/passes/offload_velocity_to_gpu.py (VTP_DIR wrong?)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def offload(sdfg) -> None:
    """Put velocity on the GPU. ``OffloadVelocityToGPU`` decides everything; we only adapt after."""
    _require_root()
    os.environ.setdefault("__DACE_NO_SYNC", "1")
    from utils.passes.offload_velocity_to_gpu import OffloadVelocityToGPU
    OffloadVelocityToGPU(sdfg)
    post_fix(sdfg)


def post_fix(sdfg) -> None:
    """Reconcile ``OffloadVelocityToGPU`` output with this DaCe branch."""
    _demote_persistent_scalars(sdfg)
    _demote_host_write_gpu_maps(sdfg)
    _absolutize_includes(sdfg)
    _shim_gpu_aliases(sdfg)
    _inline_reduction_kernels(sdfg)


def _shim_gpu_aliases(sdfg) -> None:
    """The pass's sync tasklets say gpuStreamSynchronize; this DaCe branch aliases gpu* to cuda*
    nowhere -- shim it into the frame."""
    import dace
    from dace.properties import CodeBlock
    shim = ("#ifndef gpuStreamSynchronize\n"
            "#define gpuStreamSynchronize cudaStreamSynchronize\n"
            "#endif\n")
    existing = sdfg.global_code.get("frame")
    prior = existing.code if existing is not None and isinstance(existing.code, str) else ""
    sdfg.global_code["frame"] = CodeBlock(shim + prior, dace.dtypes.Language.CPP)


def _demote_persistent_scalars(sdfg) -> None:
    """The pass promotes Default-storage Scalars to Persistent; this DaCe's CUDA preprocess infers
    them to Register, and Register+Persistent fails validation. Demote just those -- the GPU/heap
    buffers keep their Persistent promotion."""
    import dace
    from dace import data
    demotable = (dace.StorageType.Default, dace.StorageType.Register)
    count = 0
    for g in sdfg.all_sdfgs_recursive():
        for desc in g.arrays.values():
            if (desc.transient and isinstance(desc, data.Scalar) and desc.lifetime == dace.AllocationLifetime.Persistent
                    and desc.storage in demotable):
                desc.lifetime = dace.AllocationLifetime.SDFG
                count += 1
    if count:
        print(f"velocity_offload: demoted {count} Default/Register scalar(s) from Persistent to SDFG lifetime",
              flush=True)


def _demote_host_write_gpu_maps(sdfg) -> None:
    """The reduction pass leaves seed tasklets (constant writes to host-side async scratch) inside
    1-iteration GPU maps; on a coherent-memory branch that is fine, on this one it is an
    IllegalCopy. A GPU map that reads nothing and writes only host-storage data runs on the host."""
    import dace
    from dace.sdfg import nodes as nd
    count = 0
    for g in sdfg.all_sdfgs_recursive():
        for state in g.states():
            for node in state.nodes():
                if not (isinstance(node, nd.MapEntry) and node.map.schedule == dace.ScheduleType.GPU_Device):
                    continue
                reads = [e for e in state.in_edges(node) if e.data is not None and e.data.data]
                writes = [
                    e.data.data for e in state.out_edges(state.exit_node(node)) if e.data is not None and e.data.data
                ]
                if reads or not writes:
                    continue
                if all(g.arrays[w].storage != dace.StorageType.GPU_Global for w in writes):
                    node.map.schedule = dace.ScheduleType.Sequential
                    count += 1
    if count:
        print(f"velocity_offload: demoted {count} host-write GPU map(s) to Sequential", flush=True)


def _absolutize_includes(sdfg) -> None:
    """The pass's global code quotes headers relative to its include/; the harness build runs from
    a dacecache dir, so point them at the checkout."""
    from dace.properties import CodeBlock
    inc = vtp_root() / "include"
    for key, block in list(sdfg.global_code.items()):
        code = block.code
        if not isinstance(code, str):
            continue
        for header in HEADERS:
            code = code.replace(f'"{header}"', f'"{inc / header}"')
        sdfg.global_code[key] = CodeBlock(code, block.language)


def _inline_reduction_kernels(sdfg) -> None:
    """VTP compiles src/reductions{_kernel.cu,.cpp} as extra sources; sdfg.compile() has no hook
    for that, so their text goes into the global code instead."""
    root = vtp_root()
    inc = root / "include"
    for name, location in (("reductions_kernel.cu", "cuda"), ("reductions.cpp", "frame")):
        src = (root / "src" / name).read_text()
        for header in HEADERS:
            src = src.replace(f'"{header}"', f'"{inc / header}"')
        sdfg.append_global_code("\n" + src, location)


def schedule_report(sdfg) -> List[str]:
    """Every map and its schedule, outermost first -- what the lane prints with --show-schedules."""
    from dace.sdfg import nodes as nd
    rows = []
    for g in sdfg.all_sdfgs_recursive():
        for state in g.states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nd.MapEntry):
                    depth = 0
                    cur = sdict[node]
                    while cur is not None:
                        depth += 1
                        cur = sdict[cur]
                    rows.append(f"{'  ' * depth}{g.name}/{state.label}:{node.map.label} {node.map.schedule.name}")
    return rows
