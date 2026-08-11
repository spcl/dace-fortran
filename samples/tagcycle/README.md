# Tag cycle: measuring the sample kernels at a new commit on daint.alps

These scripts take a freshly pushed dace-fortran commit ("the tag") through bridge rebuild,
cache pre-warm, and a 4-variant CPU measurement on one Grace node. They are the scripts that
produced the `<variant>_<tag>_<jobid>.csv` files in the measurement runs directory; paths are
site-specific to the daint.alps setup described in the top-level reproduction notes.

## Layout on the machine

| What | Path |
|---|---|
| Frozen measurement clone | `/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas` |
| Artifacts, caches, CSVs | `/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas` |
| Python environment | `/capstor/scratch/cscs/ybudanaz/aarch64/venv-meas` (dace resolved from the `FaCe` worktree) |
| Job scripts and stdout | `/capstor/scratch/cscs/ybudanaz/aarch64/probe` |

The measurement clone is kept read-only (`chmod -R a-w`) between cycles so an accidental edit can
never flip `git describe` to `-dirty` mid-measurement. The Python environment must resolve `dace`
from the `FaCe` branch: it carries `MakeTransientsPersistent` and the launcher build safeguards in
`dace/codegen/compiler.py` (SIGCHLD unblock plus PMI environment stripping around every build
subprocess). Without those safeguards any `sdfg.compile()` inside an `srun` step hangs in cmake
configure until walltime.

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
# 1. login node: unfreeze the clone and move it to the new commit (submits nothing)
bash tagcycle_prepare.sh <tag>

# 2. rebuild the native bridge at the new commit
B=$(sbatch --parsable --export=ALL,TAG=$TAG tagcycle_bridge.sbatch)

# 3. pre-warm all four variants, diff argument lists against the baseline, refreeze on green
W=$(sbatch --parsable --dependency=afterok:$B --export=ALL,TAG=$TAG tagcycle_warm.sbatch)

# 4. the measurement: 1 node, 4 ranks x 72 cores, 2 hours
M=$(sbatch --parsable --dependency=afterok:$W --export=ALL,TAG=$TAG meas_4rank.sbatch)
```

Every stage prints grep-able verdicts to its stdout file: `BRIDGE_BUILD_EXIT=0`;
`BRIDGE_FRESH=1`, `WARM_<variant>_EXIT=0`, `ARGLIST_DIFF_<variant>=OK`, `REFROZEN=1`;
`MEAS_<variant>_EXIT=0 rows=<n>` and `MEAS_ALL_EXIT=0`. A `DIVERGED` arglist means the SDFG
signature moved at this commit — read the `.names.diff` under the artifact tree's `logs/` (a
full-line `.diff` sits beside it for debugging) before trusting any numbers. On a red warm stage
the clone is deliberately left unfrozen; phase A itself is judged by the `phase A done:` marker in
its log rather than by exit code, since the HLFIR bridge can finish all its work and still die
rc=134 in CPython teardown. `dump_arglists.py` always runs under the pinned `venv-meas` interpreter
so the baseline and every warm dump read SDFG signatures through the same dace build.

The measurement job may also be submitted without the dependency chain when the caches are already
warm at the target tag; if the caches are cold it aborts loudly within seconds instead of
rebuilding on a compute node (`DACE_FORTRAN_NO_REBUILD=1` plus the frozen clone).

## Experiment matrix (CPU)

Legend: ✅ measured, ⚠️ supported but not yet run, ❌ not available. CPU experiments (this
harness) run BOTH shape variants per kernel; GPU experiments live in a separate harness under
`probe/gpu` (`gpu_lane.sh` plus its own warm/meas sbatch pair per kernel) and run only the big
flat variant — cloudsc klon huge with nblks=1, velocity at the single-block layout.

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
| original | ❌ no OpenMP source exists — the original twin is OpenACC-only | ❌ | ⚠️ `velocity-openacc` variant wired (0d125c8), never run |

\* the ✅ rows are job 4391146 on the old R02B04-scale deck, both TUs at 32×960 — the wrong shape
for noloopexch and the wrong dataset for everything. Every velocity number needs re-measuring at
R02B06 under the TU-per-rank layout above.

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
