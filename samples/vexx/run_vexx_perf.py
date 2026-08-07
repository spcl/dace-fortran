"""vexx CPU scaling driver: one process = one (mode, OMP_NUM_THREADS) sweep point -- see README.md."""
import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# tests/ is not an installed package; same sys.path trick as tests/conftest.py.
sys.path.insert(0, str(REPO / "tests"))

# BITEXACT_CPU_ARGS (tests/_util.py) minus the --param CI-runner memory caps, exactly like the
# cloudsc sibling driver: perf nodes have the RAM, the caps only cost compile time.
PERF_CPU_ARGS = "-fPIC -O3 -march=native -fno-fast-math -ffp-contract=off -fno-math-errno -fno-trapping-math"

# (m, nnr) per mode; m * nnr within ~2x of each other, all synthetic (README).
MODE_DEFAULTS = {"bands": (64, 48**3), "grid": (4, 128**3)}

CSV_HEADER = "kernel,mode,m,nnr,threads,rep,ms,inputs,lane"
INPUTS_KIND = "synthetic"  # all vexx dims are synthetic (README); column kept position-compatible with cloudsc

LIB_NAME = "vexx_perf_lib"


def set_backend(backend: str) -> str:
    """Point DaCe at the lane's C++ compiler; same contract as the cloudsc sibling driver."""
    if backend == "llvm":
        exe = os.environ.get("CLANGXX") or shutil.which("clang++-21") or shutil.which("clang++")
        if not exe:
            raise SystemExit("--backend llvm: no clang++-21/clang++ found (common.sh probe_compilers exports CLANGXX)")
        os.environ["DACE_compiler_cpu_executable"] = exe
    return f"dace-{backend}"


# Mirrors _SDFG_DRIVER (tests/qe/exx_bp/test_vexx_bp_k_gpu_parse.py), split so the state init
# stays out of the timed window.  binding_entry's contract (tests/e2e/test_vexx.py): the
# "vexx_bp_k_gpu_dace" stem below is substituted with "<sdfg.name>_dace" before compiling.
_PERF_DRIVER = r"""
subroutine vexx_perf_init_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in) &
    bind(c, name="vexx_perf_init_c")
  ! ``, intrinsic`` dodges the QE checkpoint's stub iso_c_binding module (no c_int).
  use, intrinsic :: iso_c_binding
  use kinds, only: dp
  use us_exx, only: becxx
  implicit none
  integer(c_int), value :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
  interface
    subroutine init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, &
                                               nnr, nqs_in, jblock_in) &
        bind(c, name="init_vexx_bp_k_gpu_perf_state_c")
      use, intrinsic :: iso_c_binding
      integer(c_int), value :: lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in
    end subroutine
  end interface
  call init_vexx_bp_k_gpu_perf_state_c(lda, n, m, npol_in, max_ibands_in, nnr, nqs_in, jblock_in)
  ! becxx: degenerate AoS marshal target for the binding, never read (okvan=.false.).
  if (allocated(becxx)) deallocate(becxx)
  allocate(becxx(1))
  allocate(becxx(1)%k(1,1));  becxx(1)%k = (0.0_dp, 0.0_dp)
end subroutine vexx_perf_init_c

subroutine vexx_perf_run_c(lda, n, m, psi, hpsi) bind(c, name="vexx_perf_run_c")
  use, intrinsic :: iso_c_binding
  use kinds, only: dp
  use becmod, only: bec_type
  use noncollin_module, only: npol
  use mp_exx, only: max_ibands
  use vexx_bp_k_gpu_dace_bindings, only: vexx_bp_k_gpu_dace
  implicit none
  integer(c_int), value :: lda, n, m
  complex(dp), intent(inout) :: psi(lda*npol, max_ibands)
  complex(dp), intent(inout) :: hpsi(lda*npol, max_ibands)
  ! becpsi components are c_loc'd by the wrapper; SAVEd so timed reps pay no allocs.
  type(bec_type), save :: becpsi
  logical, save :: becpsi_ready = .false.
  if (.not. becpsi_ready) then
    allocate(becpsi%r(1,1));    becpsi%r = 0.0_dp
    allocate(becpsi%k(1,1));    becpsi%k = (0.0_dp, 0.0_dp)
    allocate(becpsi%nc(1,1,1)); becpsi%nc = (0.0_dp, 0.0_dp)
    becpsi%nbnd = 1
    becpsi_ready = .true.
  end if
  ! No <name>_dace_finalize: the binding handle is ref-counted, process exit reclaims it.
  call vexx_bp_k_gpu_dace(lda, n, m, psi, hpsi, becpsi)
end subroutine vexx_perf_run_c
"""


def git_describe() -> str:
    out = subprocess.check_output(["git", "-C", str(REPO), "describe", "--always", "--dirty"], text=True)
    return out.strip().replace("/", "-")


def cache_root() -> Path:
    """Cache dir next to the dacecache when sbatch set one, else a per-user fallback (never tmpfs)."""
    dace_build = os.environ.get("DACE_default_build_folder")
    root = Path(dace_build).parent if dace_build else Path.home() / ".cache" / "dace-fortran-samples"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dims_from_env(mode: str) -> dict:
    """Resolve the synthetic problem dims: mode defaults, each env-overridable."""
    m_default, nnr_default = MODE_DEFAULTS[mode]
    n = int(os.environ.get("VEXX_N", "4096"))
    return {
        "m": int(os.environ.get("VEXX_M", str(m_default))),
        "nnr": int(os.environ.get("VEXX_NNR", str(nnr_default))),
        "n": n,
        "lda": int(os.environ.get("VEXX_LDA", str(n))),
        "nqs": int(os.environ.get("VEXX_NQS", "1")),
        "jblock": int(os.environ.get("VEXX_JBLOCK", "1")),
    }


def set_fft_expansion(sdfg, impl: str) -> int:
    """Pick the expansion of every FFT/IFFT lib node (QE fwfft/invfft sites); returns the count."""
    from dace.libraries.fft.nodes import FFT, IFFT
    count = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, (FFT, IFFT)):
            node.implementation = "FFTW3" if impl == "fftw3" else "pure"
            if impl == "fftw3":
                # emit_fft stamps factor=1/N on IFFT (QE invfft norm); the FFTW3 expansion
                # rejects non-1 factors, so fftw3 mode drops it -- perf-only lane (README).
                node.factor = 1
            count += 1
    return count


def load_or_build(so_path: Path, build_dir: Path, mode: str, fft: str) -> ctypes.CDLL:
    """Phase A: return the binding library from cache, else build it like the e2e lane does."""
    if so_path.is_file():
        print(f"phase A: cache hit {so_path}", flush=True)
        return ctypes.CDLL(str(so_path))

    from _util import build_sdfg, have_flang
    from dace_fortran.bindings.build_fortran_library import build_fortran_library
    from dace_fortran.bindings.flatten_plan import FlattenPlan
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    from dace_fortran.bindings.frozen_signature import refreeze
    from dace_fortran.pipelines import num_maps, optimize
    from qe.exx_bp import test_vexx_bp_k_gpu_parse as vexx

    if not have_flang():
        raise SystemExit(f"no cached library at {so_path} and flang-new-21 not on PATH to build one")
    build_dir.mkdir(parents=True, exist_ok=True)

    src = vexx._make_paw_flag_public(vexx._restore_fft_interfaces(vexx._SRC.read_text()))
    src_path = build_dir / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, build_dir / "sdfg", name=f"vexx_{mode}", entry=vexx._ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = f"vexx_{mode}"

    # No specialize, no scalar-fission (e2e lane note: this kernel maps without either);
    # refreeze re-snapshots the binding signature against the optimized SDFG.
    optimize(sdfg)
    refreeze(sdfg)
    n_fft = set_fft_expansion(sdfg, fft)
    print(f"phase A: optimized ({num_maps(sdfg)} maps, {n_fft} FFT nodes -> {fft})", flush=True)

    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    driver_path = build_dir / "driver.f90"
    driver_path.write_text(_PERF_DRIVER.replace("vexx_bp_k_gpu_dace", f"{sdfg.name}_dace"))
    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(so_path.parent),
        name=LIB_NAME,
        prelude_sources=[src_path],
        extra_sources=[vexx._CALLER, driver_path],
        # Same flag as the e2e lane: pruned qvan2 arg-kind mismatch behind IF(okvan), never run.
        extra_flags=["-fallow-argument-mismatch"])
    print(f"phase A: built + linked {lib.so_path}", flush=True)
    return lib.load()


def run_timed(cdll: ctypes.CDLL, mode: str, dims: dict, reps: int, warmup: int, seed: int, lane: str) -> list:
    """Phase B: perf-state init once, seeded psi/hpsi, warmup + reps timed calls, CSV rows."""
    from qe.exx_bp.test_vexx_bp_k_gpu_parse import _make_random_inputs

    init_fn = cdll.vexx_perf_init_c
    init_fn.restype = None
    init_fn.argtypes = [ctypes.c_int] * 8
    run_fn = cdll.vexx_perf_run_c
    run_fn.restype = None
    run_fn.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2

    m, nnr, n, lda = dims["m"], dims["nnr"], dims["n"], dims["lda"]
    npol = 1
    init_fn(lda, n, m, npol, m, nnr, dims["nqs"], dims["jblock"])
    psi, hpsi = _make_random_inputs(lda, npol, m, seed=seed)
    # hpsi is accumulated into (and psi marshalled INOUT); restore both before every rep,
    # outside the timed window -- same pristine discipline as the cloudsc sibling.
    pristine_psi = psi.copy(order="F")
    pristine_hpsi = hpsi.copy(order="F")

    threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    rows = []
    for rep in range(-warmup, reps):
        np.copyto(psi, pristine_psi)
        np.copyto(hpsi, pristine_hpsi)
        t0 = time.perf_counter_ns()
        run_fn(lda, n, m, psi.ctypes.data, hpsi.ctypes.data)
        ms = (time.perf_counter_ns() - t0) / 1e6
        if rep == -warmup:
            # No-op-trap guard: the identity path would return hpsi unchanged.
            if np.array_equal(hpsi, pristine_hpsi):
                raise SystemExit("first call left hpsi unchanged -- kernel ran the no-op path")
            if not np.all(np.isfinite(hpsi)):
                raise SystemExit("first call produced non-finite hpsi -- bad synthetic state")
        if rep >= 0:
            rows.append(f"vexx,{mode},{m},{nnr},{threads},{rep},{ms:.3f},{INPUTS_KIND},{lane}")
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
    ap = argparse.ArgumentParser(description="vexx CPU scaling: one timed sweep point per invocation.")
    ap.add_argument("--mode", choices=sorted(MODE_DEFAULTS), required=True, help="which loop carries the work")
    ap.add_argument("--backend", choices=("gcc", "llvm"), default="gcc", help="DaCe CPU compiler (CSV lane column)")
    ap.add_argument("--fft",
                    choices=("fftw3", "pure"),
                    default="fftw3",
                    help="FFT lib-node expansion (default fftw3; pure is O(nnr^2), tiny grids only)")
    ap.add_argument("--reps", type=int, default=50, help="timed repetitions (default 50)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warmup calls (default 2)")
    ap.add_argument("--seed", type=int, default=42, help="input rng seed (default 42)")
    ap.add_argument("--csv", type=Path, default=None, help="append CSV rows here (stdout always gets them)")
    ap.add_argument("--build-only", action="store_true", help="phase A only: build/optimize/link the cache")
    args = ap.parse_args()

    # DACE_* env outranks Config.set; setdefault keeps a deliberate external override.
    os.environ.setdefault("DACE_compiler_cpu_args", PERF_CPU_ARGS)
    dims = dims_from_env(args.mode)

    # Dims stay symbolic in the SDFG (baked per process at the first binding call), so the
    # cached .so is dim-agnostic: the tag keys mode, fft expansion, compiler backend, source version.
    lane = set_backend(args.backend)
    tag = f"vexx_{args.mode}_{args.fft}_{args.backend}_{git_describe()}"
    root = cache_root() / tag
    cdll = load_or_build(root / "lib" / f"lib{LIB_NAME}.so", root, args.mode, args.fft)
    if args.build_only:
        print(f"phase A done: {tag}", flush=True)
        return 0

    print(CSV_HEADER, flush=True)
    rows = run_timed(cdll, args.mode, dims, args.reps, args.warmup, args.seed, lane)
    if args.csv is not None:
        append_csv(args.csv, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
