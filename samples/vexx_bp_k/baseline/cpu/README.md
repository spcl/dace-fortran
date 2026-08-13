# vexx_bp_k CPU/OpenMP baseline (standalone)

Self-contained Fortran baseline for Quantum ESPRESSO's band-parallel exact-
exchange kernel `exx_bp::vexx_bp_k` (CPU, OpenMP) — compiles and runs from
this directory alone, no QE build required.

## Quick start

### 1. Fetch the data decks — `../../download_data.sh`

    ../../download_data.sh                  # all decks -> ../../data/<MAT>/
    ../../download_data.sh BaO_nat002       # just one deck

| deck (`MAT`) | system | slots | weight |
|---|---|---|---|
| `BaTiO3_nat005` | BaTiO3 HSE06, PAW(Ba,O)+US(Ti), 24 bands, 40^3 EXX grid | 4 (k {3,6} x exx iters {1,2}) | full call ~132 s @32T |
| `BaO_nat002` | BaO rocksalt HSE06, PAW only, 12 bands, 40^3 EXX grid | 2 (k {3,6}, iter 1) | full call ~21 s @32T |

Options (env): `DATA_ROOT=<dir>` overrides the destination;
`VEXX_BP_K_DATA_DIR=<dir>` adds a local directory to probe before any network
fetch.  Integrity is enforced in every mode: pinned tarball md5 for fetches,
per-file `MANIFEST.md5` for local/extracted trees; the script refuses to
clobber a directory that fails its manifest.

### 2. Toolchain + build — `. ./setup_env.sh`, then `./build.sh`

    . ./setup_env.sh        # SOURCE it (cluster only); loads the all-GCC stack
    ./build.sh              # -> ./verify_vexx

`setup_env.sh` is for the cluster: it makes `module` available in
non-interactive (PBS) shells, unloads conflicting modules (notably nvhpc,
whose `FC=nvfortran` export otherwise hijacks the build), loads
`openmpi/4.1.7-gcc11  fftw/3.3.10-gcc11  openblas/0.3.23`, and pins the
nested-library thread counts.  On a generic Linux box / the dev container,
skip it — `build.sh` falls back to system FFTW (`/usr/include/fftw3.f03`) and
netlib `-llapack -lblas` automatically.

`build.sh` options (env): `MPIFC=` (MPI Fortran wrapper, default `mpif90`;
deliberately not `$FC` — see the toolchain block in the script), `CC_STUB=`
(C compiler for generated stubs, default `gcc`), `FFLAGS=` (default `-O3`),
`BUILD_DIR=` (intermediates dir, default `lib/`; `rm -rf lib` = clean build —
do this after switching compiler flavour), `FFTW_DIR= BLAS_DIR= FFTW_INC=
LIBS=` (library locations).  An nvfortran flag block is kept inactive in the
script for the eventual GPU port.

### 3. Correctness gate — `./run_verify.sh`

    ./run_verify.sh                         # default deck: MAT=BaTiO3_nat005
    MAT=BaO_nat002 ./run_verify.sh          # the lighter BaO deck
    ./run_verify.sh /path/to/deck "1 32"    # explicit dumpdir + thread list

Verifies every present slot of the deck in BOTH operator flavours
(`full` = US+PAW operator, `nc` = norm-conserving core) against the dumped
pw.x ground truth (bound 1e-11 rel; observed ~1e-15).  Args: `[dumpdir]`
(overrides `MAT`), `["thread list"]` (default "32").  Fails loudly on an
unknown deck and refuses to pass if zero verifications ran.  Expected tail:
`ALL <n> VERIFICATIONS PASSED`.

### 4. Performance sweep — `./measure_sweep.sh`

    ./measure_sweep.sh --list               # preview lanes + count, run nothing
    TOTAL=128 ./measure_sweep.sh            # (MPI ranks x OMP threads) sweep
    MAT=BaO_nat002 TOTAL=128 NTS="8 16 32 64 128" ./measure_sweep.sh
    LAUNCHER=srun TOTAL=128 ./measure_sweep.sh          # Cray / SLURM

Sweeps node partitions: one lane per (np, nt) with
`MIN_CORES <= np*nt <= TOTAL`.  Each rank is an INDEPENDENT verified kernel
instance (negrp==1, no inter-rank communication) on its own pinned core set —
the sweep measures per-kernel latency under co-location.  Per lane it runs
`WARMUP` discarded + `REPS` recorded invocations; every invocation is
PASS-gated, and a failed invocation discards the whole lane.  Each rep is a
fresh process, so FFT planning + the coulomb-cache build are inside every
timed call ("first application" semantics, uniform across lanes).

Options (env): `TOTAL=128` (node cores), `MIN_CORES=32` (skip smaller lanes),
`NPS="1 2 4 8 16 32"`, `NTS="1 2 4 8 16 32 64 128"`, `WARMUP=1`, `REPS=10`,
`MAT=` / `DUMP=` (deck), `SLOT=0`, `MODE=full|nc`, `CSV=measure_sweep.csv`,
`LAUNCHER=mpirun|srun`, `MPIRUN_EXTRA=`.  Output: one CSV row per
(lane, rep, rank): `np,threads,cores,rep,rank,kernel_s,verified`, plus
min/max/mean per lane on stdout.

Budget guidance: per-kernel time is set by `nt`, not by total cores — with
`MODE=full`, lanes at `nt<8` can exceed 8 min per call (the kernel's serial
fraction), so start with `NTS="8 16 32 64 128"` (12 lanes on 128 cores) and
widen later.  `--list` first.

## Contents

| file | role |
|---|---|
| `vexx_bp_k_baseline_cpu_omp.f90` | single TU: the OMP-preserving merged `vexx_bp_k` closure (byte-faithful, from QE develop/qe-omp, `!$omp`/`!$acc` intact) + the FFTW3 `fwfft_y`/`invfft_y` shim (QE conventions, cached threaded `FFTW_MEASURE` plans) + QE utility stubs (no-op clocks, loud `errore`).  Standalone pieces archived in `f2dace-qe-source/out_regex_omp/` |
| `verify_vexx.f90` | driver: loads a dump slot into the TU's module state, ONE verified kernel call, prints `kernel_time_s`; usage `./verify_vexx <dumpdir> <slot> [full|nc|aug]` |
| `setup_env.sh` | cluster toolchain loader (source it) |
| `build.sh` | build with auto-generated stub closure for unreachable externals |
| `run_verify.sh` | correctness gate (step 3) |
| `measure_sweep.sh` | partition sweep (step 4) |
| `../../data/<MAT>/` | per-material decks: flat `vexx_{slot,static,itN}_<var>.{bin,txt}` + `MANIFEST.md5` + `PROVENANCE.txt` |

## Requirements

MPI Fortran wrapper (binary runs as an MPI *singleton* — no `mpirun` needed
for a single instance, but `libmpi` must be present), FFTW3
(+ `libfftw3_omp`), LAPACK/BLAS (or OpenBLAS).  Always run with
`ulimit -s unlimited` and `OMP_STACKSIZE>=512M` — the scripts set both; a
direct `./verify_vexx` without them segfaults in the kernel's stack
temporaries.  On Cray/cray-mpich compute nodes, launch via `srun -n 1` if a
direct invocation aborts in `MPI_Init`.

## Semantics

* `full` = US+PAW operator (okvan/okpaw as dumped); ground truth
  `vexx_<slot>_big_result_full.bin` = post-call hpsi from pw.x.
* `nc` = same inputs, augmentation gates off (norm-conserving Fock core);
  ground truth from the instrumented run's probe call.
* The TU recomputes `qgm`/`ylm`/`vkb`/`coulomb_fac` internally from the
  dumped static tables, exactly as inside pw.x; `coulomb_fac` is
  cross-checked against the dump (matches exactly).
* Scope: `negrp == 1` (single band group), `npol == 1`, little-endian
  doubles.  The same decks serve GPU/OpenACC and SDFG lanes (verification
  bound absorbs FFT-backend / reduction-order differences).

Provenance/tooling (not needed to run): TU generated by
`f2dace-qe-source/run_regex_omp_vexx_bp_k.py`; decks dumped by the
boundary-dump instrumentation in `qe-omp/q-e/PW/src/exx_bp.f90`
(`QE_DUMP_DIR` / `QE_DUMP_EXIT`); per-deck details in each deck's
`PROVENANCE.txt`.
