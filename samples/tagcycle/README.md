# Tag cycle: measuring the sample kernels at a new commit on daint.alps

These scripts take a freshly pushed dace-fortran commit ("the tag") through bridge rebuild,
cache pre-warm, and a 4-variant CPU measurement on one Grace node. They are the scripts that
produced the `<variant>_<tag>_<jobid>.csv` files in the measurement runs directory.

## Layout

**No script here contains an absolute path.** Everything is derived from the repo root, which each
launcher resolves from its own location — and, under `sbatch`, from `SLURM_SUBMIT_DIR`, because
Slurm runs a *spool copy* of the script and `BASH_SOURCE` no longer points into the repo.

> **Submit every tagcycle job from the repo root**, as `sbatch … samples/tagcycle/<job>.sbatch`.
> Slurm also resolves the `#SBATCH -o/-e` paths against the submit directory, so submitting from
> anywhere else sends the job stdout somewhere else (or fails to open it at all).

| What | Path | Override |
|---|---|---|
| Repo / tree under test | resolved per launcher | `REPO=` |
| Work root (gitignored) | `<repo>/samples/_work` | `WORK_ROOT=` |
| Measurement artifacts, caches, CSVs | `$WORK_ROOT/meas` | `BR2=` |
| Run CSVs and per-rank logs | `$WORK_ROOT/meas/runs` | `RUNS=` |
| Job stdout (`#SBATCH -o/-e`) | `samples/_work/logs`, relative to the submit dir | — |
| Dev / login-node outputs | `$WORK_ROOT/dev` | — |
| GPU build root (per-lane `dacecache_<lane>`, shared warm→meas) | `$WORK_ROOT/meas/gpu` | `GPUROOT=` |
| Default DaCe build cache | `$WORK_ROOT/cache` | `BUILD_ROOT_BASE=` |
| ICON source | `$WORK_ROOT/icon-model` | — |
| Python interpreter | `$PYTHON` from `samples/env.sh` | `PYTHON=`, `PY=` |
| Spack | `$SPACK_ROOT` from `samples/env.sh` | `SPACK_ROOT=`, `SPACK_SETUP_ENV=` |

`samples/env.sh` is the gitignored site hook (template: `samples/env.spack.example`) and is the
**only** place machine-specific absolute paths belong.

There is no pinned mirror clone any more: jobs run out of the working repo, so nothing is
`chmod -R a-w`-frozen and nothing is copied to a sibling `probe/` directory. Two guards replace
what the freeze bought — every job re-asserts `git describe --always --dirty` == `$TAG` and
refuses to run against a moved or dirty tree, and every measurement job sets
`DACE_FORTRAN_NO_REBUILD=1` so a phase A cache miss dies instead of rebuilding on a compute node.
**Do not edit the tree between submitting a cycle and its last job finishing.**

The Python environment must resolve `dace` from the `FaCe` branch: it carries
`MakeTransientsPersistent` and the launcher build safeguards in `dace/codegen/compiler.py`
(SIGCHLD unblock plus PMI environment stripping around every build subprocess). Without those
safeguards any `sdfg.compile()` inside an `srun` step hangs in cmake configure until walltime.

`samples/common.sh` sets `DACE_cache_distaware=0`, and every lane depends on it. dace #2484
appends `_rank<n>` to the build folder whenever a launcher advertises a rank, and `srun` always
sets `SLURM_PROCID` — so with it left on, lanes build into `dacecache_*_rank0/` while the paths
here (and `omp_preflight.py`) look at the unsuffixed name. That surfaced twice, and the second
way is the dangerous one: velocity and `cloudsc-sweep` aborted loudly on "no such .so", but
`cloudsc-klon`/`cloudsc-nblks` found a surviving pre-#2484 folder and reported
`OMP_PREFLIGHT=OK` against a **stale** `.so` the timed run never loads — a green preflight over
numbers that measure the wrong binary. If a lane ever reports a cache miss it cannot explain,
check for a `_rank*` build folder first.

## The four variants

One Grace node has 4 sockets of 72 cores; the measurement runs one variant per socket,
all four concurrently:

- `cloudsc-klon` — `run_cloudsc_perf.py --mode klon`, both gcc and llvm backends (KLON=65536, NBLOCKS=1)
- `cloudsc-nblks` — `run_cloudsc_perf.py --mode nblks`, both backends (KLON=32, NBLOCKS=2048)
- `velocity-loopexch` — the `__LOOP_EXCHANGE` TU, both backends × both data layouts
- `velocity-noloopexch` — the no-loop-exchange TU, same 2×2

cloudsc splits by mode; velocity splits by **TU variant**, with the backend and the data layout as
sweep dimensions inside each rank (4 combinations per rank). The layout is deliberately not tied to
the TU: only when both TUs see both layouts can a row be read as "which TU wins at this layout".
The CSV keeps the axes separate — `mode` is the TU, `nproma`+`nblks_e` are the layout,
`lane` is the backend.

Both velocity TUs build an SDFG named `velocity_tendencies`, and `DACE_cache=name` keys the build
folder on that name, so two concurrent velocity ranks would otherwise compile into the same
directory. `setup_lane_root` takes a partition suffix: the velocity lanes share the per-backend
root `velocity_tendencies_dace-{gcc,llvm}` (where the `.sdfgz` caches, the `.arglist.txt` dumps and
the baseline arglists already live under TU-unique names) but build under
`dacecache_{loopexch,noloopexch}` with `tmp_{loopexch,noloopexch}` beside it.

`velocity-openacc` is a fifth, separately composable variant — the nvfortran `-acc=multicore`
Fortran twin, built and timed by `driver_velocity.f90` against `dump_data.py`'s stream-binary
dumps rather than an `.npz`. It is not part of the default four; add it with
`VARIANTS="... velocity-openacc"` on a rank that has a free socket.

`velocity-openmp` is a sixth, likewise composable variant — the same `velocity_advection_acc.f90`
source built `-fopenmp`/`-mp` instead of `-acc=multicore`, sweeping all three compilers
(`original-openmp-gcc`, `original-openmp-flang`, `original-openmp-nvhpc`) inside one rank, one CSV.
The file's 23 `!$OMP` directives (block-loop level) compile clean under gcc, flang-22, and nvhpc
alike; flang's own `!$ACC` lowering is broken (see `probe/toolchain_matrix_audit.md`), so this
variant never passes `-fopenacc` to it. Add it the same way: `VARIANTS="... velocity-openmp"`.

`velocity-icon-integ` is a seventh, likewise composable variant — `samples/velocity_tendencies/run_icon_velo_timing.sh`,
gcc backend only, driving the ICON-integration binding e2e
(`tests/icon/full/test_velocity_full_bindings_e2e.py`) once per `(istep, lvn)` configuration across
both call paths, stock Fortran reference vs the DaCe binding. It sweeps the full 2×2 `(istep, lvn)`
matrix times the 2 call paths, 4 combinations, and reports single-threaded per-invocation medians —
this is not a thread sweep, unlike the other six. Each pytest invocation builds under its own
`tmp_path`-scoped `dacecache`, so the variant is free to run concurrently with the other variants on
its own socket, or serially after they're warm; nothing about it collides on a shared cache
directory. Add it the same way: `VARIANTS="... velocity-icon-integ"`.

## The velocity dataset (R02B06)

Since `prepare_r02b06.sh` the velocity lanes read the real R02B06 grid:
**327680 cells / 491520 edges / 163842 verts, nlev=90** — 16× the cell count of the deck used
through job 4391146. (The old deck is `data_r02b05` by name only: what it holds is 20480 cells /
30720 edges, i.e. R02B04 scale. Any "81920 cells" claim about it is wrong.)

Two layouts, both swept by both velocity ranks:

| Layout | Deck | nproma | nblks_c/e/v |
|---|---|---|---|
| many-blocks | `velocity_r02b06_nproma32.npz` | 32 | 10240 / 15360 / 5121 |
| flat | `velocity_r02b06_nproma491520.npz` | 491520 | 1 / 1 / 1 |

One iteration is now ~1 s sequential and ~90 ms on 72 threads, and a deck is 6.9 GB in memory
(0.74 GB compressed), so a velocity rank spends more wall time starting 32 driver processes than
running the timed calls. Both decks are produced by
`samples/velocity_tendencies/prepare_r02b06.sh`; the warm stage will convert a missing one itself
(under an `flock`, so the two velocity ranks do not both do it).

## Running a cycle

```bash
# 0. check out the tag yourself; prepare no longer moves the tree
git checkout <tag>

# 1. login node, FROM THE REPO ROOT: assert the tree is clean at <tag>, create the gitignored
#    work root (samples/_work/{logs,meas/runs,meas/logs,dev}), print the whole chain below.
#    Submits nothing. With no argument it uses whatever HEAD describes as.
bash samples/tagcycle/tagcycle_prepare.sh <tag>
```

`tagcycle_prepare.sh` prints the rest ready to paste: the bridge rebuild, then all five warm jobs
in parallel behind it (`cpu_warm` small+large, `cpu_velocity_warm`, `gpu_warm`, `tagcycle_warm`),
then the seven measurement jobs — `meas_4rank`, the two `cpu_sweep_cloudsc` arms, the efficiency
ladder, `cpu_velocity_r02b06`, and the GPU split pair — each `afterok` on the warm that feeds it
and `afterany` on the previous meas job, so exactly one measurement runs at a time. Every line it
prints is `sbatch … samples/tagcycle/<job>.sbatch` run from the repo root; the jobs re-enter the
repo by path (`cpu_job_common.sh` and `meas_4rank` exec `samples/tagcycle/tagcycle_lane.sh`), so
the code that runs is always the code in the tree the tag assertion checked.
Submit `gpu_meas_cloudsc.sbatch` + `gpu_meas_velocity.sbatch`, not `gpu_meas.sbatch` (see below).

Every stage prints grep-able verdicts to its stdout file under `samples/_work/logs/`:
`BRIDGE_BUILD_EXIT=0`;
`BRIDGE_FRESH=1`, `WARM_<variant>_EXIT=0`, `ARGLIST_DIFF_<variant>=OK`, `TAG_STABLE=1`;
`MEAS_<variant>_EXIT=0 rows=<n>` and `MEAS_ALL_EXIT=0`. `velocity-icon-integ` prints its own set
instead of the common `MEAS_<variant>_EXIT` line: `VELO_TOOLCHAIN fc=...` (compiler provenance),
`VELO_PYTEST_EXIT istep=<i> lvn=<l> pass=<p> rc=<n>` (one per pytest invocation), `VELO_SAMPLES=<n>`,
`VELO_LABELS=present|absent`, `VELO_PARSE_EXIT=0 rows=<n>`, and finally `ICON_VELO_LOG=`,
`ICON_VELO_CSV=`, `ICON_VELO_EXIT=0`.

The GPU stages follow the same discipline with a `_GPU` infix: `WARM_GPU_<lane>_EXIT=0` per lane,
`WARM_GPU_ARTIFACTS ...`, `WARM_GPU_ALL_EXIT=0`; `MEAS_GPU_<lane>_EXIT=0 rows=<n> csv=<path>` per
lane and `MEAS_GPU_ALL_EXIT=0`. Both stages also print `MEMSPLIT=` — megabytes per concurrent rank
in the measurement jobs, `all-single-rank` in the sequential warm job — plus one
`RANK lane=… gpu=… socket=…` line per measurement rank.

`velocity-icon-integ`'s CSV is not the 9-column `kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane`
schema the other six variants write. `parse_icon_timers.py` emits its own 5-column schema —
`config,count,median_ms,p25,p75` — with one row per `(path,istep,lvn)` instead of one row per
timed rep, since it reports medians already reduced across the pytest passes rather than raw
per-invocation samples.

A `DIVERGED` arglist means the SDFG
signature moved at this commit — read the `.names.diff` under the artifact tree's `logs/` (a
full-line `.diff` sits beside it for debugging) before trusting any numbers. Phase A itself is
judged by the `phase A done:` marker in its log rather than by exit code, since the HLFIR bridge
can finish all its work and still die rc=134 in CPython teardown. `TAG_STABLE=0` means the tree
moved or was edited *during* phase A, so the caches it just wrote are not keyed to the code in
them — that stage is red regardless of the per-variant verdicts. `dump_arglists.py` runs under
`$DUMP_PYTHON` (default `$PYTHON` from `samples/env.sh`) so the baseline and every warm dump read
SDFG signatures through the same dace build.

The measurement job may also be submitted without the dependency chain when the caches are already
warm at the target tag; if the caches are cold it aborts loudly within seconds instead of
rebuilding on a compute node (`DACE_FORTRAN_NO_REBUILD=1`).

## Experiment matrix (CPU)

Legend: ✅ measured, ⚠️ supported but not yet run, ❌ not available. CPU experiments (this
harness, `tagcycle_warm.sbatch` + `meas_4rank.sbatch`) run BOTH shape variants per kernel; the GPU
experiments are the separate stage pair below and run only the big flat variant.

**cloudsc (klon + nblks):**

| config | gcc | llvm | nvhpc |
|---|---|---|---|
| dace-optimize | ✅ measured | ✅ measured | ❌ no dace-nvhpc lane exists in the harness |
| autopar | ⚠️ supported (`-ftree-parallelize-loops`), never run | ❌ flang has no autopar flag — genuinely unsupported | ❌ nvfortran has `-Mconcur` but no harness arm written |
| original OpenMP / OpenACC-CPU | ⚠️ `original-openmp` arm exists, never run | ❌ no arm (only the gcc one is wired) | ⚠️ `openacc-cpu` arm exists and is **unblocked**, never run |

The cloudsc `openacc-cpu` blocker is gone: spack carries `hdf5@1.14.6+fortran%nvhpc@26.3` and
`samples/env.sh` exports `HDF5_NVFORTRAN_ROOT` from it, which is the only thing
`baselines.sh`'s `openacc-cpu` arm was missing. It still has no tagcycle rank of its own — it runs
through `samples/cloudsc/baselines.sh`.

**velocity (loopexch + noloopexch):**

| config | gcc | llvm | nvhpc |
|---|---|---|---|
| dace-optimize | ✅ measured* | ✅ measured* | ❌ no lane |
| autopar | ⚠️ supported, never run | ❌ impossible (flang) | ❌ no arm |
| original | ⚠️ wired for gcc — `original-openmp-gcc` (`baselines.sh`) + `velocity-openmp` tagcycle sub-lane, never run | ⚠️ wired for llvm — `original-openmp-flang` (`baselines.sh`) + `velocity-openmp` tagcycle sub-lane, never run | ⚠️ wired for nvhpc — `original-openmp-nvhpc` (`baselines.sh`) + `velocity-openmp` tagcycle sub-lane, never run; `velocity-openacc` (OpenACC-CPU, `-acc=multicore`, 0d125c8) also wired, never run |

\* the ✅ rows are job 4391146 on the old R02B04-scale deck, both TUs at 32×960 — the wrong shape
for noloopexch and the wrong dataset for everything. Every velocity number needs re-measuring at
R02B06 under the TU-per-rank layout above.

## Experiment matrix (GPU)

One GH200 node is 4 modules — 4 sockets × 72 cores, each socket paired with its own H100 — so the
measurement runs the lanes **concurrently, one lane per GPU**, the same shape as the 4-rank CPU job.
Lane *i* takes GPU *i* mod `NGPU` and lanes go out in **waves** of `NGPU`, so no two running lanes
ever share a device. Each rank is its own `srun --exact --ntasks=1 --cpus-per-task=72
--cpu-bind=cores --mem-bind=local` step with an explicit `CUDA_VISIBLE_DEVICES` and `1/N` of the
node's memory. Only the big flat shapes: cloudsc `NGPTOT=65536 NPROMA=128`, velocity `nproma=491520`
(`nblks_e=1`) on the R02B06 deck. `threads` is `0` in every GPU row — no CPU thread sweep applies.

| job | partition | walltime | ranks × GPUs | what it does |
|---|---|---|---|---|
| `gpu_warm.sbatch` | `normal` | 1 h | 1 × 1, sequential | builds the cloudsc CUDA binary (`nvcc`) and the velocity `-acc=gpu` driver against the noloopexch ACC twin (`nvfortran`), generates the flat R02B06 dump, warms all three DaCe GPU caches at `--reps 1` |
| `gpu_meas_cloudsc.sbatch` | `debug` | 30 min | 2 × 1, one wave | **submit this**: `cuda-ref` on GPU 0, `dace-gpu` on GPU 1, 50 reps. ~6 min |
| `gpu_meas_velocity.sbatch` | `debug` | 30 min | 3 × 1, one wave | **submit this**: `openacc-ref` GPU 0, `dace-gpu-pipeline` GPU 1, `dace-gpu-manual` GPU 2, 50 reps. ~8–10 min |
| `gpu_meas.sbatch` | `debug` | 30 min | 4 × 1, two waves | all five lanes — the shared body the two halves above `exec` into. ~15 min; useful as a one-slot smoke run |

Wall time is now the **slowest lane in a wave**, not the sum: the velocity half dropped from ~20 min
sequential to ~8–10 min, and all three forms sit comfortably inside the 30-minute debug cap. The
split pair is still what to submit — two short jobs beat one that has to fit two waves.

Velocity lane semantics: `dace-gpu-pipeline` is the **automated** dace-fortran pipeline
(`velocity_pipeline.optimize_velocity` + `gpu_offload.apply_gpu_offload`, unchanged);
`dace-gpu-manual` is the **human-written** VelocityTendenciesPipeline flow (`samples/gpu/
vtp_manual.py`): the VTP stage-3 artifact run through VTP's stage-4 GPU entry point, imported from
the checkout at `VTP_DIR` (a sibling of the repo by default; variant via `VTP_VARIANT`), with that
checkout's `git describe` sha logged at lane start and baked
into the phase-A cache name so every measurement records pipeline provenance. The former
`dace-gpu-autoopt` lane (stock `auto_optimize`) is removed everywhere by owner decision.
The manual lane binds the harness deck (single source of truth) to VTP's SoA signature via
`vtp_manual.bind_manual_call`; the deck holds one timestep (`istep=1`, `lvn_only=0`), so the lane
is pinned to the matching variant `if_prop_lvn_only_0_istep_1` and refuses a mismatched
`VTP_VARIANT`. The ICON-integration measurement (separate, later) will measure all 4 VTP variants
as separate rows, since ICON exercises every timestep combination.

## ICON source, patches and build lanes

ICON is **not vendored** here. The reproducible tree is upstream at a pinned sha plus our patches:

    login node:  bash scripts/fetch_icon_source.sh          # clone @ pin + submodules
    then:        the lane configure script applies scripts/icon_patches/*.patch idempotently
                 -> configure -> make -> run

| item | value |
|---|---|
| upstream | `https://gitlab.dkrz.de/icon/icon-model.git` |
| pin | `8597da45ef4b86323f3fb844caedc4ae5e1ffc01` (tag `icon-2026.04-public`) |
| clone | `$WORK_ROOT/icon-model` i.e. `samples/_work/icon-model` (login-node fetch only; jobs assert the pin, like the grids) |

Patches (unified diffs against the pin, applied by `scripts/icon_daint_common.sh`):

| patch | what it fixes | lanes |
|---|---|---|
| `icon_patches/icon_acc_device_management_nompi.patch` | `USE mpi` + `mpi_comm_size`/`mpi_allgather` guarded `#ifndef NOMPI`; serial path takes device 0. Without it `--enable-gpu=openacc` + `--disable-mpi` cannot compile (`Unable to open MODULE file mpi.mod`) | both |
| `icon_patches/icon_nh_supervise_acc_directives.patch` | invalid OpenACC in `mo_nh_supervise.f90`: `REDUCTION(+, ...)` → `REDUCTION(+: ...)` (×2) and `COLLAPSE(2)` over a single loop | both |
| `icon_patches/icon_pinit_seed_guard.patch` | soil-temp perturbation guarded on `pinit_seed /= 0` (segfault with `inwp_surface=0`) | both |
| `icon_patches/icon_velocity_dace_dispatch.patch` | velocity_tendencies dispatches into `libvelocity_inner_wrap.so`; generated from `tests/icon/full/apply_velocity_dace_patch.py`, regenerate when either changes | DaCe-linked build only (opt-in) |

Compiler → script → build dir, **one build tree per compiler**, never shared:

There are **exactly two** build trees and their names are derived, never spelled: the build dir is
`{cpu,gpu}_nvhpc`, a pure function of (compiler lane, `GPU` flag), computed by
`icon_daint_build_dir_name` in `scripts/icon_daint_common.sh`. A third name cannot be produced —
the helper rejects anything outside the two. `BUILD_DIR=` still overrides for one-off trees, and
says so loudly when it does.

| lane | configure invocation | build dir | status |
|---|---|---|---|
| cpu_nvhpc | `scripts/configure_icon_nvhpc_daint.sh` | `samples/_work/icon-model/build/cpu_nvhpc` | builds, runs |
| gpu_nvhpc | `GPU=1 scripts/configure_icon_nvhpc_daint.sh` | `samples/_work/icon-model/build/gpu_nvhpc` | builds; OpenACC `-gpu=cc90` |

**ICON is nvhpc-only. gcc is not a supported ICON lane** — it keeps its lanes in the standalone
velocity and cloudsc samples, whose extracted kernels carry none of ICON's derived-type surface.

The blocker was never a build flag. gcc 16.1.0 here really does offload
(`--enable-offload-targets=nvptx-none`, ICON's configure detects `-fopenacc` and reports
`_OPENACC=201711`, and a standalone `!$acc parallel loop` binary JITs its sm_80 PTX onto the
GH200's cc90 correctly). What blocks it is that ICON's GPU port was written against nvfortran and
Cray, and leans on OpenACC's derived-type manual deep copy: it names a derived-type variable and
one of its components in a single directive (`copyin(this, this%concs)`). nvfortran implements
that; gfortran has no such path and rejects every site with `Symbol '<x>' has mixed component and
non-component accesses` — **29 files / 172 sites** in the configuration these lanes enable, first
hit in `externals/rte-rrtmgp`. No gfortran flag relaxes it, and each site needs its own directive
rather than just a split clause, so making gcc work means porting ICON's GPU data movement.

That port was attempted and abandoned: it reached 19 files across advection, dynamics, AES and
NWP physics while still expanding, and the pressure to make the build go green produced edits
that *deleted* clauses (`copyin(fluxes) copyout(fluxes%flux_net)` → `copyout(fluxes%flux_net)`).
Those compile and are silently wrong at runtime. If this is ever revisited, the rule is
**split, never delete**: the multiset of `(symbol, clause-kind)` pairs must be identical before
and after, verified per region.

Stock and DaCe-linked binaries live side by side in the lane tree as `bin/icon.stock` and
`bin/icon.dace` (only the icon.mk link rule and the velocity patch differ). Each configure run
writes `$BUILD/.configure.stamp`, a sha256 fingerprint of the lane script + the shared helper +
every patch; a job reuses an existing build only when the stamp still matches, so a debug-slot
rerun resumes instead of rebuilding. Each ICON job prints `ICON_SHA=` and one `ICON_PATCH=` line
per patch for provenance.

## ICON integration jobs (CPU, no MPI)

Full-ICON Held-Suarez R02B05 with the DaCe velocity lib linked in (`docs/ICON_INTEGRATION.md`),
one compiler lane per submission (`COMPILER=gcc|nvhpc`; nvhpc lane is stock-only until the DaCe
lib builder grows an nvfortran arm). ICON is configured `--disable-mpi` — serial binary, OpenMP
only. Builds run plain on the node (never under `srun`); only the timing step is `srun`-pinned.

All three ICON jobs **must be submitted from the repo root**: their `#SBATCH -o/-e` paths are
repo-relative and slurm resolves them through the submit dir, and each job takes the repo from
`$SLURM_SUBMIT_DIR`. Create the log dir once — it lives under the gitignored work root:

    mkdir -p samples/_work/logs
    sbatch --export=ALL,TAG=<tag>,COMPILER=gcc samples/tagcycle/icon_cpu_build.sbatch

| job | partition | walltime | tasks | what it does |
|---|---|---|---|---|
| `icon_cpu_build.sbatch` | `normal` | 4 h | 1, plain make | stock ICON (`configure_icon_<lane>_daint.sh`, FMA on) → `bin/icon.stock` → DaCe velocity lib (`build_icon_dace_libs.py --release`, `DACE_FORTRAN_FP_CONTRACT=fast`) → velocity-patched relink in the same tree → `bin/icon.dace`. Verdicts `ICON_STOCK_BUILD_EXIT[lane]` / `DACE_LIBS_EXIT[lane]` / `ICON_DACE_BUILD_EXIT[lane]` / `ICON_BUILD_EXIT[lane]` |
| `icon_cpu_timing.sbatch` | `debug` | 30 min | 1 × 72 cores, one socket | runs the patched ICON on Held-Suarez R02B05 (seeds pinned 0, PT10S steps) via `srun --ntasks=1`, greps `VELO_TIMER`, `parse_icon_timers.py` → per-config median CSV in `runs/`. Verdict `ICON_TIMING_EXIT[lane]` |
| `icon_gpu_smoke.sbatch` | `debug` | 30 min | 1 × 72 cores + GPU 0 | **the integration TEST job**: OpenACC GPU ICON (nvhpc, `-acc=gpu -gpu=cc90 -Minfo=accel`, `--enable-gpu=openacc+cuda`, no MPI) — configure+`make -j72` (resumable: build dir kept, resubmit continues) then a 3-step Held-Suarez on `GRID=R02B04..R02B09` (grids pre-staged by `scripts/fetch_icon_grids.sh`, ids 0013/0019/0021/0023/0025/0015). Sweep intent: B04–B07 fit one GH200; B08 is attempted anyway and allowed to OOM-crash; B09 does not fit a single module (grid staged for multi-module futures). Builds into `build/gpu_nvhpc`; the model run is capped at `RUN_CAP=180`s — nothing long may sit on a compute node, so this is a smoke test judged by stdout step progress, never a measurement. Verdicts `ICON_GPU_CONFIGURE_EXIT` / `ICON_GPU_BUILD_EXIT` / `ICON_GPU_RUN_EXIT` |

`gpu_warm.sbatch` stays deliberately sequential on a single GPU. Nothing in it is timed, so the
lanes gain nothing from running together, and the mechanism that would parallelise them — `srun` —
is exactly what must never wrap a build (blocked `SIGCHLD` deadlocks `cmake` in `select()`).
`gpu_meas.sbatch` may use `srun` precisely because it compiles nothing; its DaCe lanes still run the
SIGCHLD-shim preflight inside their own step before touching a cache.

**Deck-load contention.** Three velocity ranks each pull the 6.9 GB deck off capstor at once and the
drivers have no start barrier, so the launcher staggers rank starts by `RANK_STAGGER_S` (15 s
default). That spreads the read burst only — every rank times its own loop after its own load
finishes, so the measured regions are never serialized. Raise it if the logs show loads still
colliding.

**Affinity is checked, not assumed.** The lane→GPU mapping relies on Slurm handing consecutive
`--exact` steps consecutive socket-sized slices, so each worker prints
`RANK lane=<l> gpu=<g> socket=<s> …` and warns when its socket does not match its GPU. The launcher
dumps `nvidia-smi topo -m` once for provenance. Steps request no GRES — the job holds all four GPUs
and the worker picks one with `CUDA_VISIBLE_DEVICES`, which keeps device ids meaning what the script
says; if a step ever reports no visible GPU, switch to `--gpus-per-task=1` and drop the manual
variable.

| kernel | reference lane | DaCe lanes |
|---|---|---|
| cloudsc | `cuda-ref` — vendored `cloudsc_cuda` driver, `nvcc` | `dace-gpu` |
| velocity | `openacc-ref` — `driver_velocity.f90` + `velocity_advection_noloopexch_acc.f90`, `nvfortran -acc=gpu` | `dace-gpu-pipeline` (automated), `dace-gpu-manual` (VTP) |

`openacc-ref-loopexch` is a sixth, composable lane: the same build against the `loopexch` twin
(`velocity_advection_acc.f90`). It is available in both stages but deliberately **out of the default
`LANES`** — it is a different kernel from the one the DaCe GPU lanes compile, so it is not the
like-for-like reference. Add it with `LANES="... openacc-ref-loopexch"`.

CSVs land in the artifact tree's `runs/` as `<kernel>_<lane>_gpu_<tag>_<jobid>.csv`, all in the same
**10-column** schema — the 9 columns the CPU harness writes plus a trailing `alloc` — with the `lane`
column set to the lane name above so the files concatenate. The two reference drivers are converted
into that schema by `gpu_meas.sbatch` itself.

### The two ACC twins

`scripts/annotate_velocity_acc.py` now generates either twin; it picks the variant from the source
filename and prints it into the generated header:

```bash
# loopexch (unchanged default)
python scripts/annotate_velocity_acc.py

# noloopexch — the GPU reference twin
python scripts/annotate_velocity_acc.py \
  tests/icon/atmosphere/velocity_advection_inlined_no_loop_exchange_single_tu.f90 \
  samples/velocity_tendencies/velocity_advection_noloopexch_acc.f90
```

The two TUs are the same 701 lines with ten `(horizontal, jk)` nests transposed, so every anchor
keeps its line number and every directive lands on the same line in both twins — 27 `!$ACC PARALLEL`
regions, all `ASYNC(1)`, in each. The only directive *text* difference is eight `TILE` factors: the
port maps directives **by loop variable**, so a factor that sat on the horizontal loop stays on the
horizontal loop, which is the same tiling written in the other order (`TILE(128, 1)` → `TILE(1, 128)`,
`TILE(32, 4)` → `TILE(4, 32)`).

`driver_velocity.f90` cannot see which twin it was linked against, so the build states it:
`VELOCITY_TU=<variant>` sets the CSV `mode` column, defaulting to `loopexch` so every existing caller
is unchanged. Both stages assert the binary self-labels with the variant they asked for, which is
what stops a stale binary being reported under the wrong TU.

### Allocator

Every lane — references included — runs under the allocator `samples/alloc_pool.sh` selects,
mimalloc by default; both GPU stages source it before any runner or reference binary starts, so the
Python that `dlopen`s a dacecache `.so` is under it too. One comparison, one allocator.
`MEAS_ALLOC=system` is the escape hatch. What actually loaded is echoed as `alloc: <name>` and
stamped into every row's `alloc` column, so a mimalloc run can never be silently compared against an
older system-malloc one.

**Every CSV in this harness is now 10 columns**, CPU and GPU alike — the 9 above plus a trailing
`alloc`. `samples/cloudsc/baselines.sh`, `samples/cloudsc/gpu_baselines.sh` and both GPU stages
stamp it themselves; a 9-column file is from before this change and is not comparable.

An `LD_PRELOAD`ed allocator sits underneath whatever OpenMP runtime the lane loads, so the two have
to be kept honest together: `tagcycle_lane.sh` now runs an OpenMP purity preflight per lane — llvm
lanes must resolve `libomp` **by path**, gcc lanes `libgomp` — and a lane that pulls in the other
one aborts with `LLVM_OMP_PREFLIGHT_EXIT=1` rather than producing rows. Grep for that alongside the
`MEAS_*_EXIT` verdicts: a lane can fail there before it ever reaches a timer.

### Host-timer discipline on the reference lanes

Both reference drivers time `sync → t0 → body → sync → t1`, per rep, never across an async gap:

- **cloudsc CUDA** (`cloudsc_driver.cu`): upstream launched the kernel exactly once and the harness
  re-ran the whole process 50 times, parsing the `TOTAL` line — which spans the H2D and D2H copies
  and takes `t1` with no sync in front of it. It now runs its own rep loop (`CLOUDSC_REPS` /
  `CLOUDSC_WARMUP`), restages `plude` (the kernel's only in-out field) untimed before each rep,
  brackets the launch with `cudaDeviceSynchronize()` on both sides, and prints one
  `REP <rep> <ms>` line per timed rep. One process, 50 rows, kernel only.
- **velocity OpenACC** (`driver_velocity.f90`): all 27 compute regions in the TU are `ASYNC(1)`. An
  `!$ACC WAIT` now sits immediately before each of the two clock reads. Both twins happen to end on
  an `!$ACC WAIT` of their own, so today these two drain nothing — they are there to make the
  bracket's guarantee **local**, instead of resting on the callee's last line, where a regenerated
  TU would silently put the timer back around the launches. Comment for gfortran/flang without
  `-fopenacc`, no-op under `-acc=multicore`: the CPU lanes are unaffected.

## Experiment matrix (CPU problem-size sweep — figure A)

Total columns 4096 / 8192 / 16384 / 32768 / 65536 / 131072 / 262144, NPROMA=32
(NGPBLKS = NGPTOT/32). Sizes mirror f2dace-artifact's `cloudsc/original/job_cpu_*.sh`;
NPROMA does not — the artifact's 128 puts DaCe's per-block transients at 690 KiB, inside
mimalloc's 512 KiB–1 MiB cliff. `samples/cloudsc/sweep_sizes.sh` prints the table and exits
nonzero on any cliff-flagged NPROMA.

| lane | 1 thread | 72 threads | full grid @65536 |
|---|---|---|---|
| `dace-gcc`   | ✅ `cpu_sweep_cloudsc` | ✅ `cpu_sweep_cloudsc` | ✅ `meas_4rank` (cloudsc-nblks) |
| `dace-llvm`  | ✅ `cpu_sweep_cloudsc` | ✅ `cpu_sweep_cloudsc` | ✅ `meas_4rank` (cloudsc-nblks) |
| `original-openmp` (gfortran dwarf) | ✅ REPS=10 | ✅ REPS=50 | ✅ `cpu_cloudsc_efficiency` |
| `c-openmp` (C rewrite, g++) | ✅ REPS=10 | ✅ REPS=50 | ✅ `cpu_cloudsc_efficiency` |

`c-openmp` is the dwarf's `cloudsc_cuda/cloudsc/cloudsc_c.cu` compiled for the host:
`samples/cloudsc/c_openmp/cuda_shim/cuda.h` neutralises `__global__` and supplies the
thread-local block/column indices, and `cloudsc_driver_omp.cpp` replaces the CUDA driver with
an OpenMP block loop. The vendored kernel is compiled **verbatim** — no edit, no fork. It reps
inside one process (`CLOUDSC_REPS`), unlike the Fortran dwarf.

## Experiment matrix (CPU velocity — both grids)

| lane | r02b05 (nproma=32, nblks_e=960) | r02b06 (nproma=32, nblks_e=15360) |
|---|---|---|
| `dace-gcc` / `dace-llvm` (loopexch + noloopexch) | ✅ `cpu_velocity_r02b06` @16/32/72 | ✅ same job |
| `original-openmp-{gcc,flang,nvhpc}` | ✅ `velocity-openmp` sub-lane | ✅ same |

`VELOCITY_LAYOUTS` now defaults to `r02b05 r02b06` (was `nproma32 flat`); `flat`
(nproma=491520) is a GPU decomposition and is opt-in. `nproma32` stays an alias of `r02b06`.

## CPU job list

| job | partition | walltime | ranks | what it does |
|---|---|---|---|---|
| `cpu_warm.sbatch` CHUNK=small | `normal` | 1 h | 4 | phase A for 4096–32768 × {gcc,llvm} + both baseline binaries |
| `cpu_warm.sbatch` CHUNK=large | `normal` | 1 h | 4 | phase A for 65536–262144 × {gcc,llvm} |
| `cpu_velocity_warm.sbatch` | `normal` | 1 h | 3 | npz + raw dumps for both grids, Fortran driver builds |
| `cpu_sweep_cloudsc.sbatch` THREADS=72 REPS=50 | `debug` | 30 min | 4 | the 72-thread figure row |
| `cpu_sweep_cloudsc.sbatch` THREADS=1 REPS=10 | `debug` | 30 min | 4 | the 1-thread figure row (reduced reps — see below) |
| `cpu_cloudsc_efficiency.sbatch` | `debug` | 30 min | 2 | thread grid 1..72 @65536 for the two baseline lanes |
| `cpu_velocity_r02b06.sbatch` | `debug` | 30 min | 3 | velocity on both grids at 16/32/72 threads |

**Rep budgets are not uniform.** The vendored Fortran dwarf runs one process per rep and
re-expands the deck each time (~60 s at NGPTOT=262144), so it is the critical path everywhere.
The 1-thread arm and the efficiency job use REPS=10; the 72-thread arm uses REPS=50. Never
compare a median across two different rep budgets without saying so.

Every CPU job is `--exclusive --mem=0` and splits RealMemory evenly across its ranks
(`srun --mem`, `--mem-bind=local`), printing `MEMSPLIT=<MB per rank>`. Verdicts:
`grep -E 'MEMSPLIT=|_EXIT=' <job>-<id>.out`.

## Figure generation

The figure scripts live in `samples/figures/` (copied from the working set in
`probe/f2dace_plots/`). All are data-driven: `--runs-dir` points at a directory of 10-column
CSVs (`…,lane,alloc`); missing series render as gaps and are listed in the output, never
fabricated. `MANIFEST.md` there records the lane → color/legend mapping (red DaCe ▶ G++, blue
Original ▶ GFortran OpenMP, green C Rewrite ▶ G++) and which series each figure still awaits.

## Rules the scripts encode

- Never `numactl` inside an `srun` cpuset: the node exposes 36 NUMA nodes and
  `--cpunodebind` returns `sched_setaffinity EINVAL`. Pinning is done with
  `srun --exact --ntasks=1 --cpus-per-task=72 --cpu-bind=cores --mem-bind=local`.
- Never wrap the bridge build in `srun`: cmake configure hangs on the blocked SIGCHLD mask.
  The build runs in the sbatch batch body.
- `OMP_PLACES` is derived from `/proc/self/status` `Cpus_allowed_list`, never from `lscpu`,
  which reports all 288 CPUs of the node regardless of the step's cpuset.
- `DACE_compiler_allow_view_arguments` stays `false` everywhere; every argument passed to a
  compiled program must own its storage.
- A missing cloudsc `input.h5` is a hard failure, not a silent fall-back to synthetic inputs.
- CSVs from the 4-rank layout are comparable across cycles (same layout every cycle) but not
  against the older one-job-per-variant whole-node runs.
- A measurement job builds nothing. `gpu_meas.sbatch` aborts loudly on a missing reference binary
  or a missing dump rather than compiling on the `debug` partition; `gpu_warm.sbatch` owns every
  build, on `normal`.
- No job in this directory submits or resubmits another. The `sbatch` lines above are run by hand.
