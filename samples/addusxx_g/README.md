# addusxx_g — Quantum ESPRESSO `us_exx::addusxx_g`

Standalone CPU/OpenMP baseline plus the SDFG lowering artifacts for QE 7.6's
US-augmentation kernel `us_exx::addusxx_g` (the `flag='c'` k-point arm invoked from
`exx_bp::vexx_bp_k`).  Ported from spcl/dace-fortran PR #3 (branch `vexx_bp_k`,
head `70bd047`); the kernel sources, drivers, decks and SDFG pipeline are that PR's,
the daint measurement harness (`run_addusxx_g_cpu.sbatch`, `baseline/cpu/run_lane_worker.sh`,
the rewritten `baseline/cpu/build.sh`, `plot_addusxx_g_cpu.py`) is new here.

`addusxx_g` is a pure G-space kernel: no FFTs, no MPI calls at one rank, parallelism from the
OpenMP loop over G-blocks.  Sibling sample: `../newdxx_g` (same TU, same decks' static tables).

## Layout

| path | role |
|---|---|
| `baseline/cpu/addusxx_g_baseline_cpu_omp.f90` | single TU: OMP-preserving merged `us_exx` closure (QE 7.6 export, `!$omp` intact) + QE utility stubs |
| `baseline/cpu/verify_addusxx.f90` | driver: loads a dump set, calls the REAL `addusxx_g`, verifies + times |
| `baseline/cpu/build.sh` | build one Fortran lane (`LANE=original-openmp|flang-openmp`) |
| `baseline/cpu/run_verify.sh` | correctness-only gate, no timing (PR verbatim) |
| `baseline/cpu/run_lane_worker.sh` | one measurement lane, one socket, sweeps `DECK_REPS`, writes the 11-column CSV |
| `../qe_cpu_perf.py` | the DaCe CPU lanes (`dace-gcc`/`dace-llvm`): compiles `outputs/addusxx_g_opt.sdfg` with the lane's C++ compiler, verifies every timed call, appends the same CSV |
| `../qe_deck_replicate.py`, `../check_qe_deck_replicate.py` | deck replication for the Python/GPU path + its numpy gate and negative control |
| `run_addusxx_g_cpu.sbatch` | the CPU experiment: 4 lanes x 2 decks x sets x threads |
| `download_data.sh` | fetch the pinned dump decks into `data/<MAT>/` (PR verbatim) |
| `outputs/` | SDFG lowering artifacts (`.F90` TU, original/optimized/GPU `.sdfg`) |
| `run_regex_default_addusxx_g.py`, `run_optimize_original.py`, `optimize_sdfg_addusxx_g.py`, `offload_addusxx.py`, `apply_manual_fixes.sh`, `build_verify.sh`, `verify_addusxx.f90` | PR's SDFG/binding chain and its own verify driver (the Fortran-bindings route; the CPU job calls the SDFG from Python instead) |
| `plot_addusxx_g_cpu.py`, `figures/`, `output_data/` | figures and measurement CSVs |

## Decks

`./download_data.sh` fetches both, md5-pinned, into `data/<MAT>/`:

| deck (`MAT`) | system | sets |
|---|---|---|
| `BaO_nat002` | BaO rocksalt HSE06, PAW Ba+O, 12 bands, 40^3 EXX grid | 2 |
| `BaTiO3_nat005` | BaTiO3 HSE06, PAW(Ba,O)+US(Ti), 24 bands, 40^3 EXX grid | 2 |

Set 0 = the first `vexx_bp_k`-site call (q = Gamma), set 1 = the first call at a different q.
Ground truth is `adx_<set>_rhoc_out.bin`, rhoc after the in-QE call; the driver's gate is
`max|out - ref| <= 1e-11 * max(1, max|ref|)`.  `nij_type` holds CUMULATIVE species offsets.
The static tables (`adxndx_static_*`) are shared with the `newdxx_g` deck.

## Deck replication (`DECK_REP`)

The shipped decks are too small to measure: BaO is 0.30 ms at 72 threads.  `DECK_REP=R` grows one
dump set R-fold **at load time** (no giant files on disk; `../qe_deck_replicate.py` does the same
for the Python/GPU path).  For `addusxx_g` the axis is the **G-vector** axis, because the output is
a scatter `rhoc(dfftt%nl(ig)) += ...`:

| class | arrays | treatment |
|---|---|---|
| replicable payload | `qgm`, `rhoc` | tiled R times |
| output indirection | `dfftt%nl` | tiled **and shifted `+k*nnr`** for replica k |
| shared, must not shift | `mill` (indexes the shared `eigts` tables) | tiled, values unchanged |
| shared static | `eigts1-3`, `ityp`, `tau`, `ofsbeta`, `ijtoh`, `nh` | untouched |

`dfftt%nl` is an injective G -> FFT-grid map, and that injectivity is exactly why the GPU scatter
needs no atomics/WCR.  The `+k*nnr` shift is what keeps the **union** of the R replicas injective,
so the no-atomics argument survives replication; `dfftt%ngm -> R*ngmt` and `dfftt%nnr -> R*nnr`.

Verification is per-replica by construction: the reference is tiled too, so the ordinary gate over
the whole array demands that **every** replica reproduce the pw.x answer on its own slice.
Observed error is unchanged from R=1 (~1e-18).

`DECK_REP_NOOFFSET=1` is the **negative control**: it drops the shift, replicas alias replica 0, and
verification must FAIL — it does, at rel 2.8e-2 for BaO set 0 at R=4, with the predicted signature
(replica 0 over-accumulated R-fold, replicas 1..R-1 left at their input).  It is rejected outright
by `run_lane_worker.sh`, so a by-design-failing run can never reach a timing CSV.

> **The two kernels behave OPPOSITELY on GPU copy cost, and this is easy to misread.**
> `addusxx_g` replicates along G, which grows `qgm` — so the H2D copy and the execution scale
> **together** and the copy:execution ratio is *unchanged*.  Replication makes the measurement
> trustworthy; it does **not** relieve copy dominance, which is intrinsic to this kernel.
> `../newdxx_g` replicates along atoms, keeps `qgm` shared, and therefore *does* improve its ratio.

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

    sbatch -p debug -A <account> samples/addusxx_g/run_addusxx_g_cpu.sbatch

One node, 4 lanes running concurrently on one exclusive 72-core socket each -- two Fortran
(`original-openmp` = gfortran, `flang-openmp` = LLVM flang) and two DaCe (`dace-gcc` = g++,
`dace-llvm` = clang++, both on `outputs/addusxx_g_opt.sdfg`) -- sweeping `$THREADS` over both
decks and both sets.  Timing is steady state: one process per (deck, set, deck_rep, threads) runs
`WARMUP=3` discarded + `REPS=20` timed in-process calls, and the driver verifies EVERY timed call,
so a lane that drifts numerically contributes no rows.  Neither SDFG holds a top-level BLAS
library node, so the DaCe lanes link no BLAS at all and their only OpenMP runtime is the one
`-fopenmp` pulls in; `../omp_preflight.py` asserts that per lane before any row is written.  Knobs: `LANES`, `MATS`, `REPS`, `WARMUP`,
`THREADS`, `DECK_REPS`, `CSV`, `MEAS_ALLOC`.

`DECK_REPS` (default `1 64`) is the deck-replication sweep; `1` is always kept so the original
small-deck case stays on record rather than being replaced.

Output: `output_data/addusxx_g_cpu_<jobid>.csv`, the repo's schema plus one appended column

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

    python plot_addusxx_g_cpu.py     # -> figures/addusxx_g_cpu.{pdf,png} (+ _violin)

## Not ported from PR #3

`baseline/cpu/setup_env.sh` (loads NSCC modulefiles `openmpi/4.1.7-gcc11`, `fftw/3.3.10-gcc11`,
`openblas/0.3.23` — none exist here; `samples/env.sh` is this tree's equivalent) and
`baseline/cpu/measure_sweep.sh` (an `mpirun` (ranks x threads) node-partition sweep; there is no
`mpirun` here, and `run_addusxx_g_cpu.sbatch` is the repo-shaped replacement).  Both remain
retrievable from the PR ref, and the PR's own `baseline/cpu/README.md` was folded into this file.
