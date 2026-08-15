#!/usr/bin/env python3
"""DaCe CPU lane for both QE us_exx kernels: compile the optimized SDFG with the lane's
C++ compiler, verify every timed call, append the shared 11-column measurement CSV.

Shape follows ``samples/cloudsc/run_cloudsc_perf.py``, this repo's other DaCe CPU lane:
``--lane dace-llvm`` points ``DACE_compiler_cpu_executable`` at clang++, ``--build-only``
warms the dacecache once before the thread sweep, and the CSV is exactly the one
``baseline/cpu/run_lane_worker.sh`` writes for the Fortran lanes -- same columns, same
``inputs``/``deck_rep`` conventions -- so the figures merge the four lanes untouched.

Deck loading, the ABI kwargs assembly and the replication helpers are the GPU bench's
(``bench_<k>_gpu.py``, ``qe_deck_replicate.py``): the CPU SDFG keeps the same host array
ABI, so those load straight in.  Two SDFG-level fixes the Fortran binding flow applies to
the generated .cpp instead are applied here in memory: the gate copy-ins (inside
``optimize_sdfg_<k>.py::run_pipeline``, already in the ``_opt.sdfg`` input) and the eigts
lbound rebase (``offload_<k>.py::_rebase_eigts_lbounds``).

Timing brackets ``fast_call`` with the argument vectors built ONCE per set, so the ~60-array
ctypes marshalling is outside the bracket -- the same "kernel call only" bracket the Fortran
drivers' ``omp_get_wtime`` pair gives.

    PYTHONHASHSEED=0 python3 samples/qe_cpu_perf.py --kernel addusxx_g --lane dace-gcc \
        --deck samples/addusxx_g/data/BaO_nat002 --rep 64 --csv out.csv
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: kernel -> its replication helper in qe_deck_replicate
KERNELS = {"addusxx_g": "replicate_addusxx", "newdxx_g": "replicate_newdxx"}

CSV_HEADER = "kernel,mode,nnr,ngmt,threads,rep,ms,inputs,lane,alloc,deck_rep"


def set_backend(lane: str) -> str:
    """Point DaCe at the lane's C++ compiler (env outranks Config); returns the backend name."""
    backend = lane[len("dace-"):] if lane.startswith("dace-") else ""
    if backend == "llvm":
        exe = os.environ.get("CLANGXX") or shutil.which("clang++-22") or shutil.which("clang++")
        if not exe:
            raise SystemExit("--lane dace-llvm: no clang++ found (common.sh probe_compilers exports CLANGXX)")
        os.environ["DACE_compiler_cpu_executable"] = exe
    elif backend != "gcc":
        raise SystemExit(f"unknown lane {lane!r} (dace-gcc | dace-llvm)")
    return backend


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", choices=sorted(KERNELS), required=True)
    ap.add_argument("--lane", required=True, help="dace-gcc | dace-llvm")
    ap.add_argument("--sdfg", type=Path, default=None, help="default: samples/<kernel>/outputs/<kernel>_opt.sdfg")
    ap.add_argument("--deck", type=Path, default=None, help="default: samples/<kernel>/data/BaO_nat002")
    ap.add_argument("--sets", type=int, nargs="*", default=None, help="default: every set the deck holds")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--rep", type=int, default=1, help="deck replication factor (1 = the raw deck)")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--build-only", action="store_true", help="compile the SDFG and exit (warms the dacecache)")
    args = ap.parse_args()

    backend = set_backend(args.lane)
    kdir = HERE / args.kernel
    short = args.kernel[:-len("_g")]
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(kdir))

    import dace
    import numpy as np
    bench = importlib.import_module(f"bench_{short}_gpu")
    offload = importlib.import_module(f"offload_{short}")
    replicate = getattr(importlib.import_module("qe_deck_replicate"), KERNELS[args.kernel])

    sdfg_path = args.sdfg or kdir / "outputs" / f"{args.kernel}_opt.sdfg"
    deck = args.deck or kdir / "data" / "BaO_nat002"
    if not sdfg_path.is_file():
        raise SystemExit(f"SDFG not found: {sdfg_path} (run optimize_sdfg_{args.kernel}.py)")

    t0 = time.time()
    sdfg = dace.SDFG.from_file(str(sdfg_path))
    offload._rebase_eigts_lbounds(sdfg)
    # Per-backend name so the two lanes can never reuse each other's dacecache: the C++
    # compiler is not part of the SDFG hash, so a shared folder would hand the second lane
    # the first lane's binary.
    sdfg.name = f"{args.kernel}_cpu_{backend}"
    csdfg = sdfg.compile()
    print(f"{args.lane}: compiled {sdfg.name} in {time.time() - t0:.0f}s", flush=True)
    if args.build_only:
        return 0

    if not (deck / "adxndx_static_meta.txt").is_file():
        raise SystemExit(f"deck not found: {deck} (run download_data.sh)")
    static = bench.read_meta(deck / "adxndx_static_meta.txt")
    sets = args.sets if args.sets is not None else sorted(
        int(p.name.split("_")[1]) for p in deck.glob("*_[0-9]_meta.txt") if p.name.split("_")[1].isdigit())
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    alloc = os.environ.get("MEAS_ALLOC_ACTIVE", "system")

    rows = []
    for iset in sets:
        arrays, symbols, x_in, ref = bench.load_deck(deck, iset)
        arrays, symbols, x_in, ref = replicate(arrays, symbols, x_in, ref, args.rep, True)
        x = np.array(x_in, copy=True)
        kwargs, _ = bench.build_kwargs(sdfg, arrays, symbols, x)
        callargs, initargs = csdfg.construct_arguments(**kwargs)
        maxref = max(1.0, float(np.abs(ref).max()))

        x[:] = x_in
        csdfg.fast_call(callargs, initargs)
        rel = float(np.abs(x - ref).max()) / maxref
        verdict = "PASS" if rel <= args.tol else "FAIL"
        print(f"set {iset} rep={args.rep} threads={threads}: rel = {rel:.4e}  VERIFY: {verdict}", flush=True)
        if verdict == "FAIL":
            return 1

        inputs = f"set{iset}" if args.rep == 1 else f"set{iset}_rep{args.rep}"
        for i in range(args.warmup + args.reps):
            x[:] = x_in
            t = time.perf_counter()
            csdfg.fast_call(callargs, initargs)
            ms = (time.perf_counter() - t) * 1e3
            if float(np.abs(x - ref).max()) / maxref > args.tol:
                raise SystemExit(f"set {iset}: timed call FAILED the correctness gate")
            if i >= args.warmup:
                rows.append(f"{args.kernel},{deck.name},{static['nnr']},{static['ngmt']},{threads},"
                            f"{i - args.warmup},{ms:.6f},{inputs},{args.lane},{alloc},{args.rep}")

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("a") as fh:
            if args.csv.stat().st_size == 0:
                fh.write(CSV_HEADER + "\n")
            fh.write("\n".join(rows) + "\n")
        print(f"wrote {len(rows)} rows to {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
