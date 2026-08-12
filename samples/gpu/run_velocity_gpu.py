#!/usr/bin/env python
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""velocity_tendencies GPU driver: one timed sweep point per invocation, flat nproma=491520.

    run_velocity_gpu.py --lane dace-gpu-pipeline|dace-gpu-manual --reps N --csv PATH [--threads-col 0]

Two DaCe GPU lanes:

    dace-gpu-pipeline   the automated dace-fortran pipeline off the frontend SDFG:
                        velocity_pipeline.optimize_velocity + gpu_offload.apply_gpu_offload
    dace-gpu-manual     the human-written VelocityTendenciesPipeline flow (vtp_manual.py):
                        VTP's stage-3 artifact through its stage-4 GPU entry point, imported
                        from the checkout at VTP_DIR, whose git sha is logged at lane start

Reported ``ms`` comes from timer tasklets inside the SDFG; ``host_ms`` is the python wall time.
The manual lane gets the same timer tasklets (and nothing else) so both lanes report the same
quantity.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "samples" / "velocity_tendencies"), str(REPO / "tests"), str(REPO / "samples")]

VARIANT = "noloopexch"
ENTRY = "mo_velocity_advection::velocity_tendencies"
TU = "velocity_advection_inlined_no_loop_exchange_single_tu.f90"
NPROMA = 491520
LANES = ("dace-gpu-pipeline", "dace-gpu-manual")
CSV_HEADER = "kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane,host_ms"
DEFAULT_NPZ = f"velocity_r02b06_nproma{NPROMA}.npz"


def frontend_sdfg(build_dir: Path):
    """Unoptimized frontend SDFG: the one input both lanes start from."""
    from _util import build_sdfg, have_flang
    if not have_flang():
        raise SystemExit("no LLVM flang on PATH to build the velocity SDFG (source samples/env.sh)")
    tu = REPO / "tests" / "icon" / "atmosphere" / TU
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(tu.read_text(), build_dir, name=f"velocity_{VARIANT}", entry=ENTRY).build()


def optimize_lane(sdfg, lane: str, verbose: bool, demote: bool = False):
    from gpu_common import report_offload
    from dace_fortran.bindings.frozen_signature import refreeze

    from gpu_offload import apply_gpu_offload
    from velocity_pipeline import num_maps, optimize_velocity
    optimize_velocity(sdfg)
    if num_maps(sdfg) == 0:
        raise AssertionError("pipeline produced no maps -- nothing was parallelized")
    refreeze(sdfg)
    report_offload(apply_gpu_offload(sdfg, "velocity", demote_block_maps=demote or None), verbose)
    return sdfg


def add_timers(sdfg):
    """The manual lane's SDFG comes from VTP, so it needs the begin/end sync + timer states bolted on."""
    from gpu_offload import (SYNC_TICK, TIMER_BEGIN, TIMER_END, _GLOBAL_CODE, _SYNC_CODE, _TIMER_CODE, _add_host_array,
                             _append, _prepend, _silence_codegen_syncs)
    sdfg.append_global_code(_GLOBAL_CODE)
    _add_host_array(sdfg, SYNC_TICK, transient=True)
    _add_host_array(sdfg, TIMER_BEGIN, transient=False)
    _add_host_array(sdfg, TIMER_END, transient=False)
    _prepend(sdfg, "__gpu_timer_begin", _TIMER_CODE, TIMER_BEGIN)
    _prepend(sdfg, "__gpu_sync_begin", _SYNC_CODE, SYNC_TICK)
    _append(sdfg, "__gpu_sync_end", _SYNC_CODE, SYNC_TICK)
    _append(sdfg, "__gpu_timer_end", _TIMER_CODE, TIMER_END)
    _silence_codegen_syncs(sdfg, set())
    sdfg.validate()
    return sdfg


def load_gpu_sdfg(lane: str, verbose: bool, demote: bool = False):
    from sdfg_cache import load_sdfg_cached, save_sdfg_atomic

    from gpu_common import cache_root, git_describe
    root = cache_root()
    if lane == "dace-gpu-manual":
        import vtp_manual
        tag = f"velocity_{vtp_manual.vtp_variant()}_{vtp_manual.vtp_describe().replace('/', '-')}"
    else:
        tag = f"velocity_{VARIANT}_{git_describe()}"
    gpu_cache = root / f"{tag}_{lane}{'_demoted' if demote else ''}.sdfgz"
    cached = load_sdfg_cached(gpu_cache, label="phase A: GPU")
    if cached is not None:
        return cached
    if lane == "dace-gpu-manual":
        sdfg = vtp_manual.build_manual_sdfg()
        add_timers(sdfg)
    else:
        sdfg = frontend_sdfg(root / f"{tag}_{lane}")
        optimize_lane(sdfg, lane, verbose, demote)
    save_sdfg_atomic(sdfg, gpu_cache)
    return sdfg


def build_call(sdfg, npz: Path, lane: str):
    import run_velocity_perf as cpu

    from gpu_offload import timer_arrays
    arrays, meta = cpu.load_npz(npz)
    if lane == "dace-gpu-manual":
        import vtp_manual
        call = vtp_manual.bind_manual_call(sdfg, arrays, meta)
    else:
        call = cpu.bind_call(sdfg, arrays, meta)
    call.update(timer_arrays())
    return call, cpu.snapshot_outputs(call), meta


def main() -> int:
    ap = argparse.ArgumentParser(description="velocity_tendencies GPU: one timed sweep point per invocation.")
    ap.add_argument("--lane", choices=LANES, required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--threads-col", type=int, default=0, help="value for the CSV threads column (GPU rows: 0)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--npz", type=Path, default=None, help="smoke-test override; the lane deck is the flat one")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--show-schedules", action="store_true", help="print every map and its schedule")
    ap.add_argument("--demote-block-maps",
                    action="store_true",
                    help="schedule the maps INSIDE the block map instead of the block map itself "
                    "(pipeline lane only; see gpu_offload.apply_gpu_offload)")
    args = ap.parse_args()

    from gpu_common import append_csv, configure_gpu, sync_report, timed_loop
    print(f"nvcc: {configure_gpu()}", flush=True)

    sdfg = load_gpu_sdfg(args.lane, args.show_schedules, args.demote_block_maps)
    sync_report(sdfg)
    compiled = sdfg.compile()
    if args.build_only:
        print(f"phase A done: velocity {args.lane}", flush=True)
        return 0

    npz = args.npz or _find_npz()
    call, pristine, meta = build_call(sdfg, npz, args.lane)
    import run_velocity_perf as cpu
    inputs = cpu.inputs_kind(npz)

    mode = VARIANT
    if args.lane == "dace-gpu-manual":
        import vtp_manual
        mode = vtp_manual.vtp_variant().removeprefix("velocity_no_nproma_")

    print(CSV_HEADER, flush=True)
    rows = timed_loop(
        compiled, call, pristine, args.reps, args.warmup, lambda rep, ms, host_ms:
        (f"velocity_tendencies,{mode},{meta['nproma']},{meta['nblks_e']},{args.threads_col},{rep},{ms:.3f},"
         f"{inputs},{args.lane},{host_ms:.3f}"))
    if args.csv is not None:
        append_csv(args.csv, CSV_HEADER, rows)
    return 0


def _find_npz() -> Path:
    """The flat deck, wherever the warm job put it: VELOCITY_NPZ_FLAT, the cache root, or here."""
    import os
    explicit = os.environ.get("VELOCITY_NPZ_FLAT")
    if explicit:
        return Path(explicit)
    from gpu_common import cache_root
    for cand in (cache_root() / DEFAULT_NPZ, cache_root().parent / DEFAULT_NPZ,
                 REPO / "samples" / "velocity_tendencies" / DEFAULT_NPZ):
        if cand.is_file():
            return cand
    raise SystemExit(f"no {DEFAULT_NPZ}; set VELOCITY_NPZ_FLAT or pass --npz "
                     "(samples/velocity_tendencies/convert_data.py --nproma 491520 makes it)")


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if not isinstance(exc.code, (int, type(None))):
            print(exc.code, file=sys.stderr)
    # os._exit skips interpreter teardown, which aborts (glibc heap corruption, exit 134)
    # under the current dace AFTER the run has already printed its terminal state -- the
    # sbatch wrappers judge lanes by exit code, so the teardown must not overwrite it.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(rc)
