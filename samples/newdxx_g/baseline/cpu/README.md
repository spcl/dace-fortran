# newdxx_g CPU/OpenMP baseline (standalone)

Self-contained Fortran baseline for Quantum ESPRESSO's EXX non-local
projection kernel `us_exx::newdxx_g` (the `flag='c'` k-point arm invoked
from `exx_bp::vexx_bp_k`; CPU, OpenMP) — compiles and runs from this
directory alone, no QE build required.  The kernel is byte-faithful QE 7.6:
`PW/src/us_exx.f90` in the ingested export is verified identical to the
pristine `/workspace/qe-omp2/q-e` tree (whose only local edits are
start/stop_clock instrumentation in OTHER files), with every `!$omp`
sentinel intact.

## Quick start

### 1. Fetch the data decks — `../../download_data.sh`

    ../../download_data.sh                  # all decks -> ../../data/<MAT>/

| deck (`MAT`) | system | sets |
|---|---|---|
| `BaO_nat002` | BaO rocksalt HSE06, PAW Ba+O, 12 bands, 40^3 EXX grid | 2 |
| `BaTiO3_nat005` | BaTiO3 HSE06, PAW(Ba,O)+US(Ti), 24 bands, 40^3 EXX grid | 2 |

Per deck: set 0 = the very first vexx_bp_k-site call (q = Gamma), set 1 =
the first call at a different q.  Same decks as the SDFG lane
(`../../outputs/lib`); the static tables (`adxndx_static_*`) are shared with
the addusxx_g deck.

### 2. Toolchain + build — `. ./setup_env.sh`, then `./build.sh`

    . ./setup_env.sh        # SOURCE it (cluster only); loads the all-GCC stack
    ./build.sh              # -> ./verify_newdxx

On a generic Linux box / the dev container skip `setup_env.sh` — `build.sh`
falls back to netlib `-llapack -lblas` automatically.  Options (env):
`MPIFC=` (MPI Fortran wrapper, default `mpif90`), `CC_STUB=`, `FFLAGS=`
(default `-O3`), `BUILD_DIR=` (intermediates, default `lib/`; `rm -rf lib` =
clean build), `BLAS_DIR= LIBS=`.  Unlike the vexx_bp_k baseline there is NO
FFTW dependency — newdxx_g is a pure G-space kernel.

### 3. Correctness gate — `./run_verify.sh`

    ./run_verify.sh                         # default deck: MAT=BaO_nat002
    MAT=BaTiO3_nat005 ./run_verify.sh       # the heavier deck
    ./run_verify.sh /path/to/deck "1 8 32"  # explicit dumpdir + thread list

Verifies the accumulated `deexx` of every present set against the dumped
pw.x ground truth (bound 1e-11 rel).  Observed: BITWISE-IDENTICAL
(`max|diff| = 0.0`) on both decks at 1/8/32 threads — the kernel's
accumulation order is thread-count independent (static-schedule atom
ownership; serial block loop).  Expected tail:
`ALL <n> VERIFICATIONS PASSED`.

### 4. Benchmark — driver bench mode

    OMP_NUM_THREADS=32 ./verify_newdxx ../../data/BaTiO3_nat005 0 10 2

Args: `<dumpdir> <set> [reps] [warmup]`; `reps=0` (default) = one verified
call.  With reps>0 every timed call is verified (deexx reset to the dumped
input first) and per-call `kernel_time_s` plus a min/mean summary is
printed.  Reference points (dev container, 2x EPYC 7713): BaTiO3_nat005
set 0 ~0.20 s at 1T but only ~85 ms at 32T.  SCALING CAVEAT: newdxx_g's
`!$omp do` is over ATOMS inside a serial G-block loop, so parallelism is
capped at nat (5 here, 2 on BaO) — unlike addusxx_g, thread counts beyond
nat buy nothing.

### 5. Performance sweep — `./measure_sweep.sh`

    ./measure_sweep.sh --list               # preview lanes + count, run nothing
    TOTAL=128 ./measure_sweep.sh            # (MPI ranks x OMP threads) sweep
    MAT=BaO_nat002 TOTAL=128 NTS="1 2 4 8" ./measure_sweep.sh
    LAUNCHER=srun TOTAL=128 ./measure_sweep.sh          # Cray / SLURM

Mirror of the vexx_bp_k baseline's sweep: one lane per (np, nt) with
`MIN_CORES <= np*nt <= TOTAL`; each rank is an INDEPENDENT verified kernel
instance on its own pinned core set, so the sweep measures per-kernel
latency under co-location.  Per lane: `WARMUP` discarded + `REPS` recorded
invocations, every invocation PASS-gated, each rep a fresh process
(cold-start semantics — for steady-state numbers use the driver's own
reps/warmup args).  Options (env): `TOTAL=128 MIN_CORES=32 NPS= NTS=
WARMUP=1 REPS=10 MAT=/DUMP= SET=0 CSV=measure_sweep.csv LAUNCHER=mpirun|srun
MPIRUN_EXTRA=`.  Output: one CSV row per (lane, rep, rank):
`np,threads,cores,rep,rank,kernel_s,verified`, plus min/max/mean per lane.
Because of the atom-parallelism cap, expect the interesting axis to be np
(rank packing), not nt: past nt=nat every lane hits the same per-kernel
latency floor.

## Contents

| file | role |
|---|---|
| `newdxx_g_baseline_cpu_omp.f90` | single TU: the OMP-preserving merged `us_exx` closure (QE 7.6 export, `!$omp` intact) + QE utility stubs (no-op clocks, loud `errore`).  Regenerate: `f2dace-qe-source/run_regex_omp_newdxx_g.py` |
| `verify_newdxx.f90` | driver: loads a dump set into the TU's module state, calls the REAL `newdxx_g`, verifies + times; usage above |
| `setup_env.sh` | cluster toolchain loader (source it) |
| `build.sh` | build with auto-generated stub closure for unreachable externals |
| `run_verify.sh` | correctness gate (step 3) |
| `measure_sweep.sh` | (np x threads) partition sweep (step 5) |
| `../../data/<MAT>/` | per-material decks: `adxndx_static_*` (shared with addusxx_g) + `ndx_<set>_*` + `MANIFEST.md5` + `PROVENANCE.txt` |

## Requirements

MPI Fortran wrapper (the TU is the `__MPI`-resolved export, so the merged
`mp`/`parallel_include` modules `USE mpi`; the binary runs as an MPI
*singleton* and never executes an MPI call), LAPACK/BLAS.  `run_verify.sh`
sets `ulimit -s unlimited` + `OMP_STACKSIZE=512M` (vexx-baseline
convention); a bare `./verify_newdxx` normally works without them at these
problem sizes.

## Semantics

* Scope: `flag='c'` complex k-point arm, `okvan=T`, `gamma_only=F`, single
  rank — exactly the vexx_bp_k call-site configuration the decks were
  dumped from (`qgm`/`nij_type` are per-set INPUTS; the kernel does not
  call `qvan_init` internally, and neither does the driver).
* Ground truth `ndx_<set>_deexx_out.bin` = deexx after the in-QE call
  (deexx is INTENT(INOUT): the driver seeds it from `ndx_<set>_deexx_in.bin`
  before every call), with the kernels themselves untouched (see each
  deck's `PROVENANCE.txt`).
* `nij_type` holds CUMULATIVE species offsets: qgm column =
  `nij_type(nt) + ijtoh(ih,jh,nt)`.
* The TU also carries `addusxx_g` (same module) plus the qvan_init closure
  (`ylmr2`/`qvan2`, dead at runtime; qvan2 is 7.6-verbatim including the
  7.6-only `qmod(1)` typo fixed later in develop — irrelevant here).

Provenance/tooling (not needed to run): TU generated by
`f2dace-qe-source/run_regex_omp_newdxx_g.py` over
`src_f90_vexx_preprocessed` (QE 7.6 export); decks dumped by the
instrumented qe-omp pw.x run described in each deck's `PROVENANCE.txt`.
