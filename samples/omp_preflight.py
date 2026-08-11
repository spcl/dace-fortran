#!/usr/bin/env python3
"""OpenMP-runtime purity preflight for one cached dacecache kernel.

A process that maps BOTH libgomp and libomp runs two OpenMP runtimes at once; each one
sees only its own threads, they oversubscribe the cpuset, and the measured kernel lands
~34x slow. That pathology is silent -- the run completes and writes plausible CSV rows --
so a lane must abort on it rather than measure it.

Also asserts the positive half: a dace-llvm lane that quietly fell back to libgomp is not
the LLVM lane, and a dace-gcc lane that picked up libomp is not the gcc lane.

dlopen happens in THIS process, in the environment the lane already set up, so the mapping
this checks is the one the timed run will get -- including any LD_PRELOAD.

    omp_preflight.py --so <path> --expect {llvm,gcc} [--label <lane>]

Exit 0 = pure. Exit 1 = violation, with LLVM_OMP_PREFLIGHT_EXIT=1 on stderr.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

# (mapped-name fragment, human label). libgomp is GCC's runtime, libomp LLVM's.
GOMP = "libgomp.so"
IOMP = "libomp.so"


def mapped_runtimes() -> tuple[str, str]:
    """(libgomp path, libomp path) mapped in this process; empty string when absent.

    Paths, not booleans: nvhpc ships its own libomp.so, so "some libomp is mapped" is not
    enough to prove the LLVM lane got the LLVM runtime it was built against.
    """
    gomp = omp = ""
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[-1].startswith("/"):
            continue
        path = parts[-1]
        base = path.rsplit("/", 1)[-1]
        if not gomp and base.startswith(GOMP):
            gomp = path
        if not omp and base.startswith(IOMP):
            omp = path
    return gomp, omp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--so", type=Path, required=True, help="cached kernel .so to dlopen")
    ap.add_argument("--expect", choices=("llvm", "gcc"), required=True, help="which OpenMP runtime this lane must use")
    ap.add_argument("--label", default="", help="lane name for the verdict lines")
    args = ap.parse_args()

    tag = f"[{args.label}] " if args.label else ""

    if not args.so.is_file():
        print(f"{tag}OMP_PREFLIGHT: no such .so: {args.so}", file=sys.stderr, flush=True)
        print("LLVM_OMP_PREFLIGHT_EXIT=1", file=sys.stderr, flush=True)
        return 1

    pre_gomp, pre_omp = mapped_runtimes()
    try:
        ctypes.CDLL(str(args.so), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        print(f"{tag}OMP_PREFLIGHT: dlopen failed: {exc}", file=sys.stderr, flush=True)
        print("LLVM_OMP_PREFLIGHT_EXIT=1", file=sys.stderr, flush=True)
        return 1

    gomp, omp = mapped_runtimes()
    want, forbid = (IOMP, GOMP) if args.expect == "llvm" else (GOMP, IOMP)
    have_want, have_forbid = (omp, gomp) if args.expect == "llvm" else (gomp, omp)

    print(f"{tag}OMP_PREFLIGHT so={args.so}")
    print(f"{tag}OMP_PREFLIGHT expect={args.expect} libgomp={int(bool(gomp))} libomp={int(bool(omp))} "
          f"(before dlopen: libgomp={int(bool(pre_gomp))} libomp={int(bool(pre_omp))})")
    if gomp:
        print(f"{tag}OMP_PREFLIGHT libgomp_path={gomp}")
    if omp:
        print(f"{tag}OMP_PREFLIGHT libomp_path={omp}")

    # nvhpc ships a libomp.so of its own. If it shadowed the LLVM one, the lane is measuring a
    # runtime the kernel was not built against -- the boolean check alone would call that OK.
    if args.expect == "llvm" and omp and "nvhpc" in omp:
        print(f"{tag}OMP_PREFLIGHT=VIOLATION libomp came from nvhpc ({omp}), not the LLVM install -- "
              "prepend the llvm libdir to LD_LIBRARY_PATH ahead of nvhpc",
              file=sys.stderr,
              flush=True)
        print("LLVM_OMP_PREFLIGHT_EXIT=1", file=sys.stderr, flush=True)
        return 1

    if have_want and not have_forbid:
        print(f"{tag}OMP_PREFLIGHT=OK pure {want}")
        return 0

    if have_want and have_forbid:
        why = f"BOTH runtimes mapped ({want} + {forbid}) -- two OpenMP runtimes in one process, the ~34x pathology"
    elif have_forbid:
        why = f"wrong runtime: {forbid} mapped, {want} absent -- this is not the {args.expect} lane"
    else:
        why = f"no OpenMP runtime mapped: {want} absent (kernel built without -fopenmp?)"

    print(f"{tag}OMP_PREFLIGHT=VIOLATION {why}", file=sys.stderr, flush=True)
    print("LLVM_OMP_PREFLIGHT_EXIT=1", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
