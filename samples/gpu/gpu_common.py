# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared plumbing for the GPU sample runners: CUDA/DaCe env, artifact cache, timed loop, CSV."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List

REPO = Path(__file__).resolve().parents[2]
GH200_ARCH = "90"


def configure_gpu(arch: str = GH200_ARCH) -> str:
    """Pin the DaCe GPU knobs and make nvcc findable; returns the nvcc that will be used.

    Env wins over Config in DaCe, and ``setdefault`` keeps an explicit sbatch export ahead of us.
    ``max_concurrent_streams=-1`` puts every kernel and every copy on the legacy default stream,
    which is what the two-sync rule assumes. ``CUDAHOSTCXX`` follows ``CXX`` (samples/env.sh pins
    it): CMake otherwise hands nvcc the system g++, which cannot compile DaCe's C++20 runtime.
    """
    os.environ.setdefault("DACE_compiler_cuda_max_concurrent_streams", "-1")
    os.environ.setdefault("DACE_compiler_cuda_cuda_arch", arch)
    os.environ.setdefault("DACE_compiler_cuda_backend", "cuda")
    os.environ.setdefault("DACE_compiler_allow_view_arguments", "false")
    cxx = os.environ.get("CXX")
    if cxx:
        os.environ.setdefault("CUDAHOSTCXX", cxx)
        # nvcc 13.3 caps its host compiler at gcc 15 and this tree is pinned to gcc 16.1. The DaCe
        # flag REPLACES its '-lineinfo' default, so restate it; CUDAFLAGS seeds CMAKE_CUDA_FLAGS
        # early enough to also cover CMake's nvcc compiler-id probe. env.sh sets both when sourced.
        os.environ.setdefault("DACE_compiler_cuda_args",
                              "-lineinfo --allow-unsupported-compiler --expt-relaxed-constexpr")
        os.environ.setdefault("CUDAFLAGS", "--allow-unsupported-compiler --expt-relaxed-constexpr")
        extra = os.environ.get("DACE_compiler_extra_cmake_args", "")
        if "CMAKE_CUDA_HOST_COMPILER" not in extra:
            os.environ["DACE_compiler_extra_cmake_args"] = f"{extra} -DCMAKE_CUDA_HOST_COMPILER={cxx}".strip()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        home = os.environ.get("CUDA_HOME") or _spack_cuda()
        if home and Path(home, "bin", "nvcc").is_file():
            os.environ["CUDA_HOME"] = home
            os.environ["CUDA_PATH"] = home
            os.environ["PATH"] = f"{home}/bin:{os.environ['PATH']}"
            nvcc = f"{home}/bin/nvcc"
    if nvcc is None:
        raise SystemExit("no nvcc on PATH and no CUDA_HOME (source samples/env.sh, or spack load /ickio72)")
    os.environ.setdefault("CUDA_HOME", str(Path(nvcc).resolve().parents[1]))
    return nvcc


def _spack_cuda() -> str:
    root = os.environ.get("SPACK_ROOT", "")
    if not root:
        return ""
    hits = sorted(glob(f"{root}/*/cuda-*-ickio72*"))
    return hits[-1] if hits else ""


def git_describe() -> str:
    out = subprocess.check_output(["git", "-C", str(REPO), "describe", "--always", "--dirty"], text=True)
    return out.strip().replace("/", "-")


def work_root() -> Path:
    """The single output root every lane writes under: env override, else <repo>/samples/_work."""
    return Path(os.environ.get("WORK_ROOT") or REPO / "samples" / "_work")


def cache_root() -> Path:
    """Beside the sbatch dacecache when set, else under the work root; never /tmp (tmpfs)."""
    dace_build = os.environ.get("DACE_default_build_folder")  # noqa: SIM112 -- DaCe's env-var casing
    root = Path(dace_build).parent if dace_build else work_root() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def timed_loop(compiled, call: Dict, pristine: Dict, reps: int, warmup: int, row: Callable[[int, float, float],
                                                                                           str]) -> List[str]:
    """Warmup + reps calls; ``ms`` comes from the SDFG's own timer tasklets, ``host_ms`` from here."""
    import numpy as np

    from gpu_offload import elapsed_ms
    rows = []
    for rep in range(-warmup, reps):
        for key, value in pristine.items():
            np.copyto(call[key], value)
        t0 = time.perf_counter_ns()
        compiled(**call)
        host_ms = (time.perf_counter_ns() - t0) / 1e6
        if rep >= 0:
            rows.append(row(rep, elapsed_ms(call), host_ms))
            print(rows[-1], flush=True)
    return rows


def append_csv(path: Path, header: str, rows: List[str]) -> None:
    fresh = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        if fresh:
            f.write(header + "\n")
        for line in rows:
            f.write(line + "\n")


def report_offload(info: Dict, verbose: bool) -> None:
    print(
        f"offload: {info['gpu_maps']} GPU_Device maps, "
        f"block maps ({'demoted to loops' if info['demoted'] else 'left as maps'}): {info['block_maps'] or 'none'}, "
        f"host data: {info['host_data'] or 'none'}, mid syncs: {info['mid_syncs'] or 'none'}",
        flush=True)
    print(
        f"offload: {len(info['premarked'])} transients pre-marked device for host-read interstate edges, "
        f"{len(info['codegen_syncs'])} states left with a codegen sync (host reads device data there)",
        flush=True)
    if verbose:
        for loc, label, kind in info["schedules"]:
            print(f"  sched {loc} {label}: {kind}", flush=True)


def sync_report(sdfg) -> None:
    from gpu_offload import count_syncs
    counts = count_syncs(sdfg)
    print(f"syncs in generated code: {counts or 'none'}", flush=True)
