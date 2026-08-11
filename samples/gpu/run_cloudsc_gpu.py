#!/usr/bin/env python
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""CloudSC GPU driver: one timed sweep point per invocation, klon variant, whole kernel on device.

    run_cloudsc_gpu.py --lane dace-gpu --reps N --csv PATH [--threads-col 0]

Phase A is the CPU lane's: ``samples/cloudsc/run_cloudsc_perf.py`` builds and pipeline-optimizes
the SDFG (and its .sdfgz cache is reused when the warm job already made one), then
``gpu_offload.apply_gpu_offload`` puts it on the device. Reported ``ms`` is read out of the
SDFG's own timer tasklets; ``host_ms`` is the python wall time of the same call, for cross-check.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "samples" / "cloudsc"), str(REPO / "tests")]

KLON = 65536
NBLOCKS = 1
MODE = "klon"
LANES = ("dace-gpu", )
CSV_HEADER = "kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane,host_ms"


def build_call(sdfg, klon: int, nblocks: int, inputs_kind: str, h5_path: Path, seed: int):
    """Same input binding as the CPU driver's run_timed, plus the SDFG's timer arrays."""
    from cloudsc.full._harness import _SCALAR_TYPES, accepted_names, lower_keys, sdfg_call_args
    from cloudsc.full._registries import get_inputs_physical, get_outputs

    from gpu_offload import timer_arrays

    rng = np.random.default_rng(seed)
    if inputs_kind == "h5":
        from dwarf_inputs import load_dwarf_inputs
        inputs = load_dwarf_inputs(h5_path, klon, nblocks)
    else:
        inputs = get_inputs_physical(rng)
    outputs = lower_keys(get_outputs(rng))
    scalars = {k.lower(): v for k, v in inputs.items() if isinstance(v, _SCALAR_TYPES)}
    kwargs = {k.lower(): np.array(v, order="F") for k, v in inputs.items() if not isinstance(v, _SCALAR_TYPES)}
    kwargs.update({k: np.array(v, order="F") for k, v in outputs.items()})
    kwargs.update(sdfg_call_args(sdfg, scalars))
    accepted = accepted_names(sdfg)
    call = {k: v for k, v in kwargs.items() if k in accepted}
    call.update(timer_arrays())
    pristine = {k: call[k].copy(order="F") for k in outputs if k in call}
    return call, pristine


def load_gpu_sdfg(lane: str, klon: int, nblocks: int, verbose: bool):
    import run_cloudsc_perf as cpu
    from dace import SDFG

    from gpu_common import cache_root, git_describe, report_offload
    from gpu_offload import apply_gpu_offload

    root = cache_root()
    tag = f"cloudsc_klon{klon}_nb{nblocks}_{git_describe()}"
    gpu_cache = root / f"{tag}_{lane}.sdfgz"
    if gpu_cache.is_file():
        print(f"phase A: GPU cache hit {gpu_cache}", flush=True)
        return SDFG.from_file(str(gpu_cache))
    sdfg = cpu.load_or_build(root / f"cloudsc_klon{klon}_nb{nblocks}_gcc_{git_describe()}.sdfgz", root / tag)
    info = apply_gpu_offload(sdfg, "cloudsc")
    report_offload(info, verbose)
    sdfg.save(str(gpu_cache), compress=True)
    return sdfg


def main() -> int:
    ap = argparse.ArgumentParser(description="CloudSC GPU: one timed sweep point per invocation.")
    ap.add_argument("--lane", choices=LANES, default="dace-gpu")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--threads-col", type=int, default=0, help="value for the CSV threads column (GPU rows: 0)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--klon", type=int, default=KLON, help="smoke-test override; the lane size is fixed at 65536")
    ap.add_argument("--h5", type=Path, default=REPO / "samples" / "cloudsc" / "data" / "input.h5")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--show-schedules", action="store_true", help="print every map and its schedule")
    args = ap.parse_args()

    os.environ["CLOUDSC_KLON"] = str(args.klon)
    os.environ["CLOUDSC_NBLOCKS"] = str(NBLOCKS)
    from gpu_common import append_csv, configure_gpu, sync_report, timed_loop
    print(f"nvcc: {configure_gpu()}", flush=True)

    sdfg = load_gpu_sdfg(args.lane, args.klon, NBLOCKS, args.show_schedules)
    sync_report(sdfg)
    compiled = sdfg.compile()
    if args.build_only:
        print(f"phase A done: cloudsc {args.lane} klon={args.klon}", flush=True)
        return 0

    inputs_kind = "h5" if args.h5.is_file() else "synthetic"
    call, pristine = build_call(sdfg, args.klon, NBLOCKS, inputs_kind, args.h5, args.seed)

    print(CSV_HEADER, flush=True)
    rows = timed_loop(
        compiled, call, pristine, args.reps, args.warmup, lambda rep, ms, host_ms:
        (f"cloudsc,{MODE},{args.klon},{NBLOCKS},{args.threads_col},{rep},{ms:.3f},{inputs_kind},{args.lane},"
         f"{host_ms:.3f}"))
    if args.csv is not None:
        append_csv(args.csv, CSV_HEADER, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
