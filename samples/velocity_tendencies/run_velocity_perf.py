"""ICON velocity_tendencies CPU scaling driver: one (TU variant, threads) sweep point; cluster-only, see README.md."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))  # reuse the test scaffolding, like tests/conftest.py does

# BITEXACT_CPU_ARGS (tests/_util.py) without the CI-sized --param ggc-*/gcse caps: perf nodes have the RAM.
PERF_CPU_ARGS = "-fPIC -O3 -march=native -fno-fast-math -ffp-contract=off -fno-math-errno -fno-trapping-math"

ENTRY = "mo_velocity_advection::velocity_tendencies"
# ids match test_velocity_numerical_e2e.py: loopexch = __LOOP_EXCHANGE layout TU.
VARIANT_TUS = {
    "loopexch": "velocity_advection_inlined_single_tu.f90",
    "noloopexch": "velocity_advection_inlined_no_loop_exchange_single_tu.f90",
}
# Kernel write set (VelocityTendenciesPipeline/main.cpp got/want list); restored before every rep.
OUTPUTS = ("p_diag_ddt_vn_apc_pc", "p_diag_ddt_vn_cor_pc", "p_diag_ddt_w_adv_pc", "p_diag_vt", "p_diag_vn_ie",
           "p_diag_vn_ie_ubc", "p_diag_w_concorr_c", "p_diag_max_vcfl_dyn", "z_kin_hor_e", "z_vt_ie", "z_w_concorr_me")
FIXED_SCALARS = {"ntnd": 1, "istep": 1, "lvn_only": 0, "l_vert_nested": 0, "ddt_vn_cor_associated": 0}

CSV_HEADER = "kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane"
INPUTS_KIND = "r02b05"  # real R02B05 grid data (README); column kept position-compatible with cloudsc/vexx


def set_backend(backend: str) -> str:
    """Point DaCe at the lane's C++ compiler; same contract as the cloudsc sibling driver."""
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
    # Beside the dacecache when the sbatch set one (common.sh setup_build_root); never /tmp (tmpfs).
    dace_build = os.environ.get("DACE_default_build_folder")  # noqa: SIM112 -- DaCe's real env-var casing
    root = Path(dace_build).parent if dace_build else Path.home() / ".cache" / "dace-fortran-samples"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_or_build(variant: str, cache: Path, build_dir: Path):
    from dace import SDFG
    if cache.is_file():
        print(f"phase A: cache hit {cache}", flush=True)
        return SDFG.from_file(str(cache))

    from _util import build_sdfg, have_flang
    from dace_fortran.bindings.frozen_signature import refreeze
    from dace_fortran.pipelines import num_maps, optimize

    if not have_flang():
        raise SystemExit(f"no cached SDFG at {cache} and flang-new-21 not on PATH to build one")
    tu = REPO / "tests" / "icon" / "atmosphere" / VARIANT_TUS[variant]
    build_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(tu.read_text(), build_dir, name=f"velocity_{variant}", entry=ENTRY).build()
    # Same pipeline + bars as tests/e2e/_hooks.py velocity_optimize: no specialize, refreeze after.
    optimize(sdfg)
    if num_maps(sdfg) == 0:
        raise AssertionError("pipeline produced no maps -- nothing was parallelized")
    refreeze(sdfg)
    print(f"phase A: built + optimized ({num_maps(sdfg)} maps), caching {cache}", flush=True)
    sdfg.save(str(cache), compress=True)
    return sdfg


def flat_name(sdfg_arg: str) -> str:
    # __CG_p_patch__CG_cells__m_edge_idx -> p_patch_cells_edge_idx (marshalled-struct scheme).
    return sdfg_arg.replace("__CG_", "_").replace("__m_", "_").lstrip("_")


def load_npz(path: Path) -> tuple[dict, dict]:
    with np.load(path) as zf:
        meta = json.loads(str(zf["meta"]))
        arrays = {k: zf[k] for k in zf.files if k != "meta"}
    return {k: np.asfortranarray(v) for k, v in arrays.items()}, meta


def bind_call(sdfg, arrays: dict, meta: dict) -> dict:
    """Map npz arrays + meta scalars onto the SDFG's arglist and free symbols; loud on any gap."""
    import dace.data as dt  # deferred: dace import is heavy, keep --help/module import cheap
    from dace import symbolic

    lbounds = meta["lbounds"]
    scalars = dict(meta)
    scalars.pop("lbounds")
    scalars.update(FIXED_SCALARS)
    scalars["nflatlev"] = meta["nflatlev_jg"]
    scalars["nrdmax"] = meta["nrdmax_jg"]

    call: dict = {}
    sym_vals: dict = {}
    missing: list[str] = []
    for name, desc in sdfg.arglist().items():
        if isinstance(desc, dt.Array):
            flat = flat_name(name)
            arr = arrays.get(flat)
            if arr is None and flat in scalars:  # module array read as rank-1 (nflatlev/nrdmax)
                arr = np.full(tuple(int(s) for s in desc.shape), scalars[flat], dtype=np.int32, order="F")
            if arr is None:
                missing.append(f"array {name} (flat {flat})")
                continue
            call[name] = arr
            lb = lbounds.get(flat, [1] * arr.ndim)
            for dim, expr in enumerate(desc.shape):
                if str(expr).isidentifier():
                    sym_vals.setdefault(str(expr), int(arr.shape[dim]))
            for dim, expr in enumerate(desc.offset):
                for s in symbolic.symlist(expr):
                    if s.startswith("__f2dace_SOA"):
                        sym_vals.setdefault(s, int(lb[dim]))
        else:
            flat = flat_name(name)
            if flat in scalars:
                call[name] = scalars[flat]
            elif flat in arrays and arrays[flat].size == 1:  # e.g. max_vcfl_dyn, stored as a (1,) buffer
                call[name] = arrays[flat].dtype.type(arrays[flat][0])
            else:
                missing.append(f"scalar {name}")
    for sym in sorted(str(s) for s in sdfg.free_symbols):
        if sym in call:
            continue
        if sym in sym_vals:
            call[sym] = sym_vals[sym]
        elif sym in scalars:
            call[sym] = int(scalars[sym])
        else:
            missing.append(f"symbol {sym}")
    if missing:
        raise RuntimeError("cannot bind SDFG inputs from the npz:\n  " + "\n  ".join(missing))
    return call


def run_timed(compiled, call: dict, variant: str, meta: dict, reps: int, warmup: int, lane: str) -> list[str]:
    reverse = {flat_name(k): k for k in call}
    pristine = {reverse[f]: call[reverse[f]].copy(order="F") for f in OUTPUTS if f in reverse}
    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    rows = []
    for rep in range(-warmup, reps):
        for k, v in pristine.items():
            np.copyto(call[k], v)
        t0 = time.perf_counter_ns()
        compiled(**call)
        ms = (time.perf_counter_ns() - t0) / 1e6
        if rep >= 0:
            rows.append(f"velocity_tendencies,{variant},{meta['nproma']},{meta['nblks_e']},{threads},{rep},{ms:.3f},"
                        f"{INPUTS_KIND},{lane}")
            print(rows[-1], flush=True)
    return rows


def append_csv(path: Path, rows: list[str]) -> None:
    fresh = not path.exists()
    with path.open("a") as f:
        if fresh:
            f.write(CSV_HEADER + "\n")
        for row in rows:
            f.write(row + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="velocity_tendencies CPU scaling: one timed sweep point per invocation.")
    ap.add_argument("--variant", choices=sorted(VARIANT_TUS), required=True)
    ap.add_argument("--backend", choices=("gcc", "llvm"), default="gcc", help="DaCe CPU compiler (CSV lane column)")
    ap.add_argument("--npz", type=Path, default=Path(__file__).resolve().parent / "velocity_r02b05_nproma32.npz")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--build-only", action="store_true", help="phase A only: build/optimize/compile the cache")
    args = ap.parse_args()

    # DACE_* env OUTRANKS Config.set, so the perf flags survive whatever the scaffolding touches.
    os.environ.setdefault("DACE_compiler_cpu_args", PERF_CPU_ARGS)

    lane = set_backend(args.backend)
    root = cache_root()
    tag = f"velocity_{args.variant}_{args.backend}_{git_describe()}"
    sdfg = load_or_build(args.variant, root / f"{tag}.sdfgz", root / tag)
    compiled = sdfg.compile()
    if args.build_only:
        print(f"phase A done: {tag}", flush=True)
        return 0

    arrays, meta = load_npz(args.npz)
    call = bind_call(sdfg, arrays, meta)
    print(CSV_HEADER, flush=True)
    rows = run_timed(compiled, call, args.variant, meta, args.reps, args.warmup, lane)
    if args.csv is not None:
        append_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
