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
- `velocity-gcc` — `run_velocity_perf.py --backend gcc`, both loop-exchange TU variants
- `velocity-llvm` — same with the llvm backend

cloudsc splits by mode and velocity splits by backend because both velocity TU variants produce an
SDFG named `velocity_tendencies` and therefore share one dacecache directory; splitting by backend
keeps every concurrent rank in a private cache root.

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
signature moved at this commit — read the diff under the artifact tree's `logs/` before trusting
any numbers. On a red warm stage the clone is deliberately left unfrozen.

The measurement job may also be submitted without the dependency chain when the caches are already
warm at the target tag; if the caches are cold it aborts loudly within seconds instead of
rebuilding on a compute node (`DACE_FORTRAN_NO_REBUILD=1` plus the frozen clone).

## Experiment matrix (CPU)

Legend: ✅ measured, ⚠️ supported but not yet run, ❌ not available. CPU experiments (this
harness) run BOTH shape variants per kernel; GPU experiments (separate harness, `probe/gpu`) run
only the big flat variant (cloudsc klon huge, nblks=1; velocity nproma=30720, nblks_e=1).

**cloudsc (klon + nblks):**

| config | gcc | llvm | nvhpc |
|---|---|---|---|
| dace-optimize | ✅ measured | ✅ measured | ❌ no dace-nvhpc lane exists in the harness |
| autopar | ⚠️ supported (`-ftree-parallelize-loops`), never run | ❌ flang has no autopar flag — genuinely unsupported | ❌ nvfortran has `-Mconcur` but no harness arm written |
| original OpenMP / OpenACC-CPU | ⚠️ `original-openmp` arm exists, never run | ❌ no arm (only the gcc one is wired) | ⚠️ `openacc-cpu` arm exists, blocked on an nvfortran-built HDF5 |

**velocity (loopexch + noloopexch):**

| config | gcc | llvm | nvhpc |
|---|---|---|---|
| dace-optimize | ✅ measured* | ✅ measured* | ❌ no lane |
| autopar | ⚠️ supported, never run | ❌ impossible (flang) | ❌ no arm |
| original | ❌ no OpenMP source exists — the original twin is OpenACC-only | ❌ | ⚠️ `openacc-cpu` arm exists, being wired now |

\* noloopexch was measured at the wrong shape (32x960) in job 4391146; fixed in 2f1f38b to
nproma=30720, nblks_e=1; velocity dace lanes need re-measurement.

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
