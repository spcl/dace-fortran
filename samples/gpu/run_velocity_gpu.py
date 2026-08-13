#!/usr/bin/env python
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""velocity_tendencies GPU driver: one timed sweep point per invocation, flat nproma=491520.

    run_velocity_gpu.py --lane dace-gpu-pipeline|dace-gpu-manual --reps N --csv PATH [--threads-col 0]

Two DaCe GPU lanes, differing only in the SDFG they hand to the offload:

    dace-gpu-pipeline   the automated dace-fortran pipeline off the frontend SDFG:
                        velocity_pipeline.optimize_velocity, then velocity_offload.offload
    dace-gpu-manual     the human-written VelocityTendenciesPipeline flow (vtp_manual.py):
                        VTP's stage-3 artifact through its stage-4 GPU entry point, imported
                        from the checkout at VTP_DIR, whose git sha is logged at lane start

``OffloadVelocityToGPU`` is the only offload either lane runs -- see velocity_offload. Both then
get the same timer tasklets, so ``ms`` means the same thing in both rows; ``host_ms`` is the
python wall time.

The Fortran-facing signature is frozen AFTER the offload (``refreeze``), which is what records
each argument's device storage for the binding generator; freezing before it would regenerate
bindings against pre-offload storage.
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


def _signature(sdfg):
    """The caller-visible argument list, as the binding generator sees it."""
    return tuple((name, type(desc).__name__, str(desc.dtype), tuple(str(d) for d in desc.shape))
                 for name, desc in sorted(sdfg.arrays.items()) if not desc.transient)


def optimize_lane(sdfg, lane: str, verbose: bool):
    """Frontend SDFG -> parallel maps -> OffloadVelocityToGPU -> re-frozen Fortran signature."""
    from dace_fortran.bindings.frozen_signature import refreeze

    import velocity_offload
    from velocity_pipeline import num_maps, optimize_velocity
    optimize_velocity(sdfg)
    if num_maps(sdfg) == 0:
        raise AssertionError("pipeline produced no maps -- nothing was parallelized")

    before = _signature(sdfg)
    velocity_offload.offload(sdfg)
    if _signature(sdfg) != before:
        raise AssertionError("OffloadVelocityToGPU changed the caller-visible signature; the Fortran "
                             "bindings would no longer match the deck")
    # After the offload, so each arg's device storage reaches the binding generator.
    refreeze(sdfg)
    if verbose:
        print("\n".join(velocity_offload.schedule_report(sdfg)), flush=True)
    return sdfg


def add_timers(sdfg):
    """Both lanes get the same begin/end sync + timer states bolted on."""
    from gpu_timers import add_timers as _bolt_on
    _bolt_on(sdfg)
    return sdfg


def load_gpu_sdfg(lane: str, verbose: bool):
    from sdfg_cache import load_sdfg_cached, save_sdfg_atomic

    from gpu_common import cache_root, git_describe
    root = cache_root()
    if lane == "dace-gpu-manual":
        import vtp_manual
        tag = f"velocity_{vtp_manual.vtp_variant()}_{vtp_manual.vtp_describe().replace('/', '-')}"
    else:
        tag = f"velocity_{VARIANT}_{git_describe()}"
    gpu_cache = root / f"{tag}_{lane}.sdfgz"
    cached = load_sdfg_cached(gpu_cache, label="phase A: GPU")
    if cached is not None:
        return cached
    if lane == "dace-gpu-manual":
        sdfg = vtp_manual.build_manual_sdfg()
    else:
        sdfg = frontend_sdfg(root / f"{tag}_{lane}")
        optimize_lane(sdfg, lane, verbose)
    # Instrumentation last: the timer arrays are harness state, not part of the Fortran ABI.
    add_timers(sdfg)
    save_sdfg_atomic(sdfg, gpu_cache)
    return sdfg


def build_call(sdfg, npz: Path, lane: str):
    import run_velocity_perf as cpu

    from gpu_timers import timer_arrays
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
    args = ap.parse_args()

    from gpu_common import append_csv, configure_gpu, sync_report, timed_loop
    print(f"nvcc: {configure_gpu()}", flush=True)

    sdfg = load_gpu_sdfg(args.lane, args.show_schedules)
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
