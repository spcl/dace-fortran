"""CloudSC CPU scaling driver: one process = one (KLON, NBLOCKS, OMP_NUM_THREADS) sweep point; see README.md."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tests" / "cloudsc" / "full" / "cloudsc.F90"
# tests/ is not an installed package; same sys.path pattern as tests/conftest.py.
sys.path.insert(0, str(REPO / "tests"))

# BITEXACT_CPU_ARGS (tests/_util.py) without the --param caps sized for a 16 GB CI runner.
PERF_CPU_ARGS = "-fPIC -O3 -march=native -fno-fast-math -ffp-contract=off -fno-math-errno -fno-trapping-math"

MODE_DEFAULTS = {"klon": (65536, 1), "nblks": (32, 2048)}  # see README.md table

CSV_HEADER = "kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane"
H5_DEFAULT = Path(__file__).resolve().parent / "data" / "input.h5"


def set_backend(backend: str) -> str:
    """Point DaCe at the lane's C++ compiler (env outranks Config); returns the lane name."""
    if backend == "llvm":
        exe = os.environ.get("CLANGXX") or shutil.which("clang++-21") or shutil.which("clang++")
        if not exe:
            raise SystemExit("--backend llvm: no clang++-21/clang++ found (common.sh probe_compilers exports CLANGXX)")
        os.environ["DACE_compiler_cpu_executable"] = exe
    return f"dace-{backend}"


def git_describe() -> str:
    out = subprocess.check_output(["git", "-C", str(REPO), "describe", "--always", "--dirty"], text=True)
    return out.strip().replace("/", "-")


def cache_root() -> Path:
    """Next to the sbatch dacecache when set, else a stable per-user dir; never /tmp (tmpfs)."""
    dace_build = os.environ.get("DACE_default_build_folder")
    root = Path(dace_build).parent if dace_build else Path.home() / ".cache" / "dace-fortran-samples"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_or_build(cache: Path, build_dir: Path):
    """Return the optimized SDFG: from ``cache`` when present, else build + optimize + save."""
    from dace import SDFG
    if cache.is_file():
        print(f"phase A: cache hit {cache}", flush=True)
        return SDFG.from_file(str(cache))

    # Species-count specialize split reused from the e2e lane (tests/e2e/test_cloudsc.py).
    from _util import build_sdfg, have_flang
    from e2e.test_cloudsc import _split_specialize

    from dace_fortran.pipelines import num_maps, optimize

    if not have_flang():
        raise SystemExit(f"no cached SDFG at {cache} and flang-new-21 not on PATH to build one")
    build_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(SRC.read_text(), build_dir, name="cloudsc", entry="cloudscouter").build()
    syms, scalar_consts = _split_specialize(sdfg)
    optimize(sdfg, symbols=syms, scalars=scalar_consts)
    print(f"phase A: built + optimized ({num_maps(sdfg)} maps), caching {cache}", flush=True)
    sdfg.save(str(cache), compress=True)
    return sdfg


def run_timed(sdfg, compiled, mode: str, klon: int, nblocks: int, reps: int, warmup: int, seed: int, inputs_kind: str,
              h5_path: Path, lane: str) -> list:
    """Phase B: dwarf-h5 or seeded synthetic inputs, warmup + reps timed calls, return the CSV rows."""
    from cloudsc.full._harness import _SCALAR_TYPES, accepted_names, lower_keys, sdfg_call_args
    from cloudsc.full._registries import get_inputs_physical, get_outputs

    rng = np.random.default_rng(seed)
    if inputs_kind == "h5":
        from dwarf_inputs import load_dwarf_inputs  # sibling module, importable when run as a script
        inputs: dict = load_dwarf_inputs(h5_path, klon, nblocks)
    else:
        inputs = get_inputs_physical(rng)
    outputs = lower_keys(get_outputs(rng))
    scalars = {k.lower(): v for k, v in inputs.items() if isinstance(v, _SCALAR_TYPES)}
    kwargs = {k.lower(): v for k, v in inputs.items() if not isinstance(v, _SCALAR_TYPES)}
    kwargs.update({k: v.copy(order="F") for k, v in outputs.items()})
    kwargs.update(sdfg_call_args(sdfg, scalars))
    # Keep what the (possibly cache-loaded) SDFG accepts; specialization baked the rest out.
    accepted = accepted_names(sdfg)
    call = {k: v for k, v in kwargs.items() if k in accepted}
    # INTENT(OUT)/INOUT arrays are restored before every rep (untimed) so all reps see identical state.
    pristine = {k: call[k].copy(order="F") for k in outputs if k in call}

    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    rows = []
    for rep in range(-warmup, reps):
        for k, v in pristine.items():
            np.copyto(call[k], v)
        t0 = time.perf_counter_ns()
        compiled(**call)
        ms = (time.perf_counter_ns() - t0) / 1e6
        if rep >= 0:
            rows.append(f"cloudsc,{mode},{klon},{nblocks},{threads},{rep},{ms:.3f},{inputs_kind},{lane}")
            print(rows[-1], flush=True)
    return rows


def append_csv(path: Path, rows: list) -> None:
    fresh = not path.exists()
    with path.open("a") as f:
        if fresh:
            f.write(CSV_HEADER + "\n")
        for row in rows:
            f.write(row + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="CloudSC CPU scaling: one timed sweep point per invocation.")
    ap.add_argument("--mode", choices=sorted(MODE_DEFAULTS), required=True, help="which loop carries the parallelism")
    ap.add_argument("--backend", choices=("gcc", "llvm"), default="gcc", help="DaCe CPU compiler (CSV lane column)")
    ap.add_argument("--inputs", choices=("h5", "synthetic"), default="h5", help="input source (default dwarf h5 deck)")
    ap.add_argument("--h5", type=Path, default=H5_DEFAULT, help="dwarf input.h5 (download_data.sh fetches it)")
    ap.add_argument("--reps", type=int, default=50, help="timed repetitions (default 50)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warmup calls (default 2)")
    ap.add_argument("--seed", type=int, default=42, help="input rng seed (default 42)")
    ap.add_argument("--csv", type=Path, default=None, help="append CSV rows here (stdout always gets them)")
    ap.add_argument("--build-only", action="store_true", help="phase A only: build/optimize/compile the cache")
    args = ap.parse_args()

    klon_default, nblocks_default = MODE_DEFAULTS[args.mode]
    # Must land BEFORE the registries import (evaluates its shape dict at import time); setdefault
    # so an explicit sbatch export still wins.
    os.environ.setdefault("CLOUDSC_KLON", str(klon_default))
    os.environ.setdefault("CLOUDSC_NBLOCKS", str(nblocks_default))
    # A DACE_* env var outranks Config.set, so the perf flags survive whatever the scaffolding touches.
    os.environ.setdefault("DACE_compiler_cpu_args", PERF_CPU_ARGS)
    klon = int(os.environ["CLOUDSC_KLON"])
    nblocks = int(os.environ["CLOUDSC_NBLOCKS"])

    inputs_kind = args.inputs
    if inputs_kind == "h5" and not args.h5.is_file():
        print(f"WARNING: {args.h5} missing (run samples/cloudsc/download_data.sh); "
              "falling back to synthetic inputs",
              file=sys.stderr,
              flush=True)
        inputs_kind = "synthetic"

    lane = set_backend(args.backend)
    root = cache_root()
    tag = f"cloudsc_klon{klon}_nb{nblocks}_{args.backend}_{git_describe()}"
    sdfg = load_or_build(root / f"{tag}.sdfgz", root / tag)
    # Compile is part of phase A: DACE_cache=name + PYTHONHASHSEED=0 make later rebuilds a no-op.
    compiled = sdfg.compile()
    if args.build_only:
        print(f"phase A done: {tag}", flush=True)
        return 0

    print(CSV_HEADER, flush=True)
    rows = run_timed(sdfg, compiled, args.mode, klon, nblocks, args.reps, args.warmup, args.seed, inputs_kind, args.h5,
                     lane)
    if args.csv is not None:
        append_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
