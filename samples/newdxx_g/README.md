# newdxx_g — Quantum ESPRESSO `us_exx::newdxx_g`

Standalone CPU/OpenMP baseline plus the SDFG lowering artifacts for QE 7.6's
EXX non-local projection kernel `us_exx::newdxx_g` (the `flag='c'` k-point arm invoked from
`exx_bp::vexx_bp_k`).  Ported from spcl/dace-fortran PR #3 (branch `vexx_bp_k`,
head `70bd047`); the kernel sources, drivers, decks and SDFG pipeline are that PR's,
the daint measurement harness (`run_newdxx_g_cpu.sbatch`, `baseline/cpu/run_lane_worker.sh`,
the rewritten `baseline/cpu/build.sh`, `plot_newdxx_g_cpu.py`) is new here.

`newdxx_g` is a pure G-space kernel: no FFTs, no MPI calls at one rank, parallelism from the
OpenMP loop over G-blocks.  Sibling sample: `../addusxx_g` (same TU, same decks' static tables).

## Layout

| path | role |
|---|---|
| `baseline/cpu/newdxx_g_baseline_cpu_omp.f90` | single TU: OMP-preserving merged `us_exx` closure (QE 7.6 export, `!$omp` intact) + QE utility stubs |
| `baseline/cpu/verify_newdxx.f90` | driver: loads a dump set, calls the REAL `newdxx_g`, verifies + times |
| `baseline/cpu/build.sh` | build one Fortran lane (`LANE=original-openmp|flang-openmp`) |
| `baseline/cpu/run_verify.sh` | correctness-only gate, no timing (PR verbatim) |
| `baseline/cpu/run_lane_worker.sh` | one measurement lane, one socket, sweeps `DECK_REPS`, writes the 11-column CSV |
| `../qe_cpu_perf.py` | the DaCe CPU lanes (`dace-gcc`/`dace-llvm`): compiles `outputs/newdxx_g_opt.sdfg` with the lane's C++ compiler, verifies every timed call, appends the same CSV |
| `../qe_deck_replicate.py`, `../check_qe_deck_replicate.py` | deck replication for the Python/GPU path + its numpy gate and negative control |
| `run_newdxx_g_cpu.sbatch` | the CPU experiment: 4 lanes x 2 decks x sets x threads |
| `download_data.sh` | fetch the pinned dump decks into `data/<MAT>/` (PR verbatim) |
| `outputs/` | SDFG lowering artifacts (`.F90` TU, original/optimized/GPU `.sdfg`) |
| `run_regex_default_newdxx_g.py`, `run_optimize_original.py`, `optimize_sdfg_newdxx_g.py`, `offload_newdxx.py`, `apply_manual_fixes.sh`, `build_verify.sh`, `verify_newdxx.f90` | PR's SDFG/binding chain and its own verify driver (the Fortran-bindings route; the CPU job calls the SDFG from Python instead) |
| `plot_newdxx_g_cpu.py`, `figures/`, `output_data/` | figures and measurement CSVs |

## Decks

`./download_data.sh` fetches both, md5-pinned, into `data/<MAT>/`:

| deck (`MAT`) | system | sets |
|---|---|---|
| `BaO_nat002` | BaO rocksalt HSE06, PAW Ba+O, 12 bands, 40^3 EXX grid | 2 |
| `BaTiO3_nat005` | BaTiO3 HSE06, PAW(Ba,O)+US(Ti), 24 bands, 40^3 EXX grid | 2 |

Set 0 = the first `vexx_bp_k`-site call (q = Gamma), set 1 = the first call at a different q.
Ground truth is `ndx_<set>_deexx_out.bin`, the accumulated deexx after the in-QE call; the driver's gate is
`max|out - ref| <= 1e-11 * max(1, max|ref|)`.  `nij_type` holds CUMULATIVE species offsets.
The static tables (`adxndx_static_*`) are shared with the `addusxx_g` deck.

## Deck replication (`DECK_REP`)

`DECK_REP=R` grows one dump set R-fold **at load time** (no giant files on disk;
`../qe_deck_replicate.py` does the same for the Python/GPU path).  For `newdxx_g` the axis is the
**atom** axis — *not* the G-vector axis `../addusxx_g` uses — because the output is a reduction
`deexx(ofsbeta(na)+ih) += ...`, so the output indirection is `ofsbeta`, not `dfftt%nl`:

| class | arrays | treatment |
|---|---|---|
| replicable payload | `deexx`, `becphi_c`, `eigts1-3`, `tau`, `ityp`, `vkb` | tiled R times (`nat -> R*nat`, `nkb -> R*nkb`) |
| output indirection | `ofsbeta` | tiled **and shifted `+k*nkb`** for replica k |
| shared, must not replicate | `qgm`, `mill`, `dfftt%nl`, `vc`, `nij_type`, `nh`, `ijtoh` | species- or G-indexed; untouched |

The shift keeps each replica's beta manifold — hence its `deexx` target range — disjoint, so every
replica reproduces the pw.x answer on its own `deexx` slice and the ordinary gate over the tiled
reference is a per-replica check.  Observed error is unchanged from R=1 (~1e-19).

**Why the atom axis and not G.** `newdxx_g`'s `!$omp do` is the **atom** loop; the G-block loop is
the sequential outer one.  With `nat` = 2 (BaO) / 5 (BaTiO3), 67+ of 72 threads are idle no matter
how many G-vectors are added, which is the real cause of this kernel's flat thread scaling —
measured in job 4478781, BaO gfortran runs 7.09 ms at 1 thread and 5.89 ms at 72, i.e. 1.21x and
then dead flat from 2 threads on.  It is **not** intrinsic serial structure.  Replicating along G
would have added work without adding parallelism; replicating along atoms gives `nat*R` units.

`DECK_REP_NOOFFSET=1` is the **negative control**: it drops the shift, all replicas accumulate into
replica 0's `deexx` range (an aliasing bug *and* a genuine race, since the `omp do` is over `na`),
and verification must FAIL — it does, at rel 1.25 for BaO set 0 at R=8.  It is rejected outright by
`run_lane_worker.sh`, so a by-design-failing run can never reach a timing CSV.

> **The two kernels behave OPPOSITELY on GPU copy cost, and this is easy to misread.**
> `newdxx_g` replicates along atoms and keeps the big `qgm` array **shared**, so the H2D copy stays
> roughly constant while execution grows R-fold — this axis *does* improve the copy:execution ratio.
> `../addusxx_g` replicates along G, which grows `qgm`, so its copy and execution scale **together**
> and its ratio is *unchanged*.  Same feature, opposite behaviour; do not generalise one to the other.

    ../check_qe_deck_replicate.py --deck data/BaO_nat002     # numpy gate + negative control

## Build and verify by hand (login node)

    . samples/env.sh
    ./download_data.sh
    cd baseline/cpu
    LANE=original-openmp ./build.sh          # or LANE=flang-openmp
    ./run_verify.sh                          # MAT=BaTiO3_nat005 ./run_verify.sh

`build.sh` needs no MPI installation: `USE mpi` is satisfied by dace_fortran's
OpenMPI-header stub module, compiled with the lane's own compiler so the `.mod` format matches,
and every MPI entry point becomes an abort-if-called link stub (one rank reaches none of them).
`OMPI_INCLUDE=` overrides the header probe, `MPIFC=` short-circuits the stub with a real wrapper,
`BLAS_DIR=`/`OPENBLAS_DIR=` override the BLAS/LAPACK prefix.

## The CPU experiment

    sbatch -p debug -A <account> samples/newdxx_g/run_newdxx_g_cpu.sbatch

One node, 4 lanes running concurrently on one exclusive 72-core socket each -- two Fortran
(`original-openmp` = gfortran, `flang-openmp` = LLVM flang) and two DaCe (`dace-gcc` = g++,
`dace-llvm` = clang++, both on `outputs/newdxx_g_opt.sdfg`) -- sweeping `$THREADS` over both
decks and both sets.  Timing is steady state: one process per (deck, set, deck_rep, threads) runs
`WARMUP=3` discarded + `REPS=20` timed in-process calls, and the driver verifies EVERY timed call,
so a lane that drifts numerically contributes no rows.  Neither SDFG holds a top-level BLAS
library node, so the DaCe lanes link no BLAS at all and their only OpenMP runtime is the one
`-fopenmp` pulls in; `../omp_preflight.py` asserts that per lane before any row is written.  Knobs: `LANES`, `MATS`, `REPS`, `WARMUP`,
`THREADS`, `DECK_REPS`, `CSV`, `MEAS_ALLOC`.

`DECK_REPS` (default `1 64`) is the deck-replication sweep; `1` is always kept so the original
small-deck case stays on record rather than being replaced.

Output: `output_data/newdxx_g_cpu_<jobid>.csv`, the repo's schema plus one appended column

    kernel,mode,nnr,ngmt,threads,rep,ms,inputs,lane,alloc,deck_rep

with `mode` = material deck, `nnr`/`ngmt` = **the deck's static** FFT-grid and G-vector counts (the
two size dimensions `f2dace_style.load_runs` reads *positionally* as `cols[2]`/`cols[3]` — which is
why `deck_rep` is appended at the end and nothing may be inserted before `threads`).

> **`rep` and `deck_rep` are different things — do not conflate them.**
> `rep` (pre-existing) is the timing loop's repetition **index**, 0..`REPS`-1.
> `deck_rep` (new) is the **replication factor** R passed to the driver as `DECK_REP`.

`inputs` is `set<n>` at R=1 and `set<n>_rep<R>` for R>1.  That tag is load-bearing: `nnr`/`ngmt`
stay at the deck's static values, so `problem_size = nnr*ngmt` is identical for every R, and
without the distinct `inputs`/`grid` label the figures would silently average R=1 and R=64 into one
bar.

    python plot_newdxx_g_cpu.py     # -> figures/newdxx_g_cpu.{pdf,png} (+ _violin)

## The GPU numbers

**There is no reference implementation for these kernels on GPU.** No CUDA, no OpenACC, no
Fortran GPU port exists in this tree or upstream, so there is nothing to plot the DaCe lane
against — a one-series bar chart beside the four-lane CPU panels would read as three lanes that
failed. The numbers therefore live here as a table and are deliberately **not** drawn.

Absolute, **compute only**. The SDFG owns its copy states, so `../qe_gpu_timing.py` gates them and
the rep loop calls the phases separately: copy-in (untimed) → COMPUTE (timed) → copy-out (untimed)
→ verify. **H2D/D2H are listed only for context and are outside the timing bracket** — never fold
them into the compute figure.

| deck | set | compute (ms) | H2D (ms) | D2H (ms) |
|---|---|---|---|---|
| `BaO_nat002` | 0 | **35.48** | 1.84 | 1.04 |
| `BaO_nat002` | 1 | **35.47** | 1.84 | 1.04 |
| `BaTiO3_nat005` | 0 | **118.48** | 3.08 | 1.08 |
| `BaTiO3_nat005` | 1 | **118.46** | 3.08 | 1.06 |

Source: `output_data/newdxx_g_gpu_4479825.csv`, `--rep 1`, `--reps 20` (median), every call verified
at 1e-11. Regenerate — and refresh this table from the new CSV — with

    PYTHONHASHSEED=0 python3 bench_newdxx_gpu.py --deck data/<MAT> --reps 20 --csv output_data/newdxx_g_gpu_<jobid>.csv

> Do **not** read `output_data/newdxx_g_gpu_4479163.csv`: that is the pre-unblocking SDFG and its
> compute column is stale. These figures move whenever the offload pipeline changes, so always
> re-derive them from the newest CSV rather than editing a number in place.

## Not ported from PR #3

`baseline/cpu/setup_env.sh` (loads NSCC modulefiles `openmpi/4.1.7-gcc11`, `fftw/3.3.10-gcc11`,
`openblas/0.3.23` — none exist here; `samples/env.sh` is this tree's equivalent) and
`baseline/cpu/measure_sweep.sh` (an `mpirun` (ranks x threads) node-partition sweep; there is no
`mpirun` here, and `run_newdxx_g_cpu.sbatch` is the repo-shaped replacement).  Both remain
retrievable from the PR ref, and the PR's own `baseline/cpu/README.md` was folded into this file.
