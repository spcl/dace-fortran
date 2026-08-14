#!/bin/bash
# ICON velocity_tendencies on real R02B05 data, re-blocked to nproma=32: CPU thread sweep, default
# lane dace-gcc x the loopexch TU variant (VARIANTS opts noloopexch back in, see below).
# Site-neutral header (no partition/account in-file, pass them on the command line):
#     sbatch -p <partition> -A <account> samples/velocity_tendencies/run_velocity.sbatch
#     Submit from the repo root (or from this script's own directory); Slurm spools the
#     script, so the repo is found via $SLURM_SUBMIT_DIR. DACE_FORTRAN_REPO=<repo> overrides,
#     and is REQUIRED alongside -D/--chdir, which repoints $SLURM_SUBMIT_DIR away from the repo.
# Header shape: dace_perf_jobs .../cloudsc_variants/slurm_cloudsc_variants.sh; srun
# binding: cpu_codegen_perf_jobs/submit_daint_readable.sbatch.

#SBATCH --job-name=velocity-tendencies-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH -p debug
#SBATCH --time=00:30:00
#SBATCH -o samples/_work/logs/velocity_tendencies-%j.out
#SBATCH -e samples/_work/logs/velocity_tendencies-%j.err
# The -o/-e paths are repo-relative on purpose: slurm resolves them against the submit dir, and
# this job is submitted from the repo root.

set -euo pipefail

# Slurm copies the batch script into its spool dir before running it, so ${BASH_SOURCE[0]} is
# /var/spool/slurmd/... in a job and cannot find the repo. Order: an explicit DACE_FORTRAN_REPO
# (which must resolve), then the submitting directory -- repo root, or samples/<kernel>/ two levels
# down -- then the script's own path, which only resolves outside Slurm.
resolve_repo() {
    local c
    if [ -n "${DACE_FORTRAN_REPO:-}" ]; then
        if [ ! -f "$DACE_FORTRAN_REPO/samples/common.sh" ]; then
            echo "FATAL: DACE_FORTRAN_REPO=$DACE_FORTRAN_REPO holds no samples/common.sh" >&2
            return 1
        fi
        (cd "$DACE_FORTRAN_REPO" && pwd)
        return 0
    fi
    for c in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../..}" \
        "$(dirname "${BASH_SOURCE[0]}")/../.."; do
        [ -n "$c" ] && [ -f "$c/samples/common.sh" ] || continue
        (cd "$c" && pwd)
        return 0
    done
    echo "FATAL: cannot locate the dace-fortran repo. Slurm spools the batch script, so the script's" >&2
    echo "  own path is under /var/spool in a job, and \$SLURM_SUBMIT_DIR did not resolve either." >&2
    echo "  Submit from the repo root or from samples/<kernel>/, or pass DACE_FORTRAN_REPO=<repo>." >&2
    return 1
}
REPO="$(resolve_repo)"
source "$REPO/samples/common.sh"
# LANES: space list from {dace-gcc,dace-llvm,baselines,acc-multicore} (cloudsc sbatch convention;
# baselines runs the Fortran lanes of baselines.sh, which picks its own subset via BASELINE_LANES;
# acc-multicore is baselines.sh's openacc-cpu lane alone -- nvfortran -acc=multicore, the CPU-side
# twin of the GPU ACC-vs-DaCe story in opt_pipeline/run_acc_vs_dace.sbatch). Any other token is
# skipped, so a global LANES carrying cloudsc-only lanes still runs this job.
LANES="${LANES:-dace-gcc}"
BUILD_ROOT_BASE="${BUILD_ROOT:-$WORK_ROOT/cache}"
# exported so the baselines lane (which unsets BUILD_ROOT to escape the dace lanes' dacecache)
# still lands under this root instead of falling back to $HOME.
export BUILD_ROOT_BASE
mkdir -p "$BUILD_ROOT_BASE"

HERE="$REPO/samples/velocity_tendencies"
DATA_DIR="${VELOCITY_DATA_DIR:-$HERE/data_r02b05}"
NPZ="${VELOCITY_NPZ:-$BUILD_ROOT_BASE/velocity_r02b05_nproma32.npz}"
NPZ_NOLOOPEXCH="${VELOCITY_NPZ_NOLOOPEXCH:-$BUILD_ROOT_BASE/velocity_r02b05_nproma30720.npz}"

lscpu | head -30
probe_topology
probe_compilers
echo "topology: sockets=$TOPO_NSOCKETS numa_nodes=$TOPO_NNUMA physical_cores=$TOPO_NCORES"
echo "physical cpu ids (numa-major): $TOPO_PHYS_CPUS"

# Idempotent data prep: download+patch skips when populated; convert skips when the npz exists.
OUTPUT_DIR="$DATA_DIR" bash "$HERE/download_data.sh"
if [ ! -f "$NPZ" ]; then
    "$PY" "$HERE/convert_data.py" --data-dir "$DATA_DIR" --nproma 32 --out "$NPZ"
fi
if [ ! -f "$NPZ_NOLOOPEXCH" ]; then
    "$PY" "$HERE/convert_data.py" --data-dir "$DATA_DIR" --nproma 30720 --out "$NPZ_NOLOOPEXCH"
fi

# Data shape is its OWN axis, not a function of the TU variant.  They used to share the
# loopexch/noloopexch names, so switching the TU silently switched nproma 32 -> 30720 and
# left the CPU block loop with one block to distribute.
npz_for() {
    case "${DATA_SHAPE:-nproma32}" in
        bignproma) echo "$NPZ_NOLOOPEXCH" ;;
        *) echo "$NPZ" ;;
    esac
}

DRIVER="$HERE/run_velocity_perf.py"
CSV="${CSV:-velocity_tendencies_${SLURM_JOB_ID:-local}.csv}"
NCPUS="${SLURM_CPUS_ON_NODE:-$TOPO_NCORES}"
# TU variant only.  Everything is no-loop-exchange -- the SDFG baselines, the ACC reference and
# both ICON lanes -- so the DaCe side must be lowered from the same TU.  Data shape is DATA_SHAPE
# (nproma32 default; bignproma = nproma 30720 / nblks_e 1, the GPU shape, meaningless on CPU
# because a block-parallel schedule then has exactly one block to distribute).
VARIANTS="${VARIANTS:-noloopexch}"
# OpenMP schedule sweep for the DaCe block-loop maps: space-separated "kind[,chunk]" specs, each
# its own build (the clause is baked into the pragma, so a point is a recompile). Default is the
# single point; the three-point sweep is
#   OMP_SCHEDULES="static,1 guided static,8"
# A spec contains a comma, which is also `sbatch --export`'s separator, so export from the
# submitting shell instead of spelling the list inside --export:
#   OMP_SCHEDULES="static,1 guided static,8" THREADS="16 32 72" LANES=dace-gcc \
#       sbatch --export=ALL -p debug -A g34 --time=00:30:00 samples/velocity_tendencies/run_velocity.sbatch
OMP_SCHEDULES="${OMP_SCHEDULES:-default}"
sched_slug() { echo "${1//,/_}"; }

for lane in $LANES; do
    case "$lane" in
        dace-gcc | dace-llvm) ;;
        baselines)
            srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                bash "$HERE/baselines.sh"
            continue
            ;;
        acc-multicore)
            if [ "${HAVE_NVFORTRAN:-0}" != 1 ]; then
                echo "SKIP $lane: no nvfortran on PATH" >&2
                continue
            fi
            # openacc-cpu alone: nvfortran -acc=multicore, one build, ACC_NUM_CORES sweeps THREADS
            # (baselines.sh), same CSV so this lane lands beside dace-gcc/dace-llvm for comparison.
            BASELINE_LANES=openacc-cpu CSV="$CSV" \
                srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                bash "$HERE/baselines.sh"
            continue
            ;;
        *)
            echo "SKIP unknown lane: $lane" >&2
            continue
            ;;
    esac
    backend="${lane#dace-}"
    # Per-lane build root: a gcc-compiled dacecache must never be reused by the llvm lane.
    BUILD_ROOT="$BUILD_ROOT_BASE/velocity_tendencies_${lane}"
    setup_build_root "velocity_tendencies_${lane}"

    # Phase A once per variant, at max threads: build + optimize + compile into the cache (untimed).
    set_omp_env "$TOPO_NCORES"
    for v in $VARIANTS; do
        # Both variants build an SDFG named velocity_tendencies, so under DACE_cache=name one
        # dacecache would make each variant a full rebuild of the other (tagcycle_lane.sh idiom).
        # Same reason for the schedule slug: each spec is a different pragma, hence a different .so.
        for s in $OMP_SCHEDULES; do
            export DACE_default_build_folder="$BUILD_ROOT/dacecache_${v}_$(sched_slug "$s")"
            srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                "$PY" "$DRIVER" --variant "$v" --backend "$backend" --npz "$(npz_for "$v")" \
                --omp-schedule "$s" --build-only
        done
    done

    for t in $THREADS; do
        if [ "$t" -gt "$TOPO_NCORES" ]; then
            echo "skip threads=$t (> $TOPO_NCORES physical cores)"
            continue
        fi
        set_omp_env "$t"
        for v in $VARIANTS; do
            for s in $OMP_SCHEDULES; do
                export DACE_default_build_folder="$BUILD_ROOT/dacecache_${v}_$(sched_slug "$s")"
                srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                    "$PY" "$DRIVER" --variant "$v" --backend "$backend" --npz "$(npz_for "$v")" \
                    --omp-schedule "$s" --csv "$CSV"
            done
        done
    done
done
echo "done: $CSV"
