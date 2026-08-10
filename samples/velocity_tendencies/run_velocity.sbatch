#!/bin/bash
# ICON velocity_tendencies on real R02B05 data, re-blocked to nproma=32: CPU thread sweep
# over both __LOOP_EXCHANGE TU variants (different transient DATA LAYOUTS, run both).
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
#SBATCH --time=02:00:00
#SBATCH --output=velocity_tendencies_%j.out

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
# LANES: space list from {dace-gcc,dace-llvm,baselines} (cloudsc sbatch convention; baselines runs
# the Fortran lanes of baselines.sh, which picks its own subset via BASELINE_LANES).  Any other
# token is skipped, so a global LANES carrying cloudsc-only lanes still runs this job.
LANES="${LANES:-dace-gcc}"
BUILD_ROOT_BASE="${BUILD_ROOT:-$HOME/.cache/dace-fortran-samples}"
# exported so the baselines lane (which unsets BUILD_ROOT to escape the dace lanes' dacecache)
# still lands under this root instead of falling back to $HOME.
export BUILD_ROOT_BASE
mkdir -p "$BUILD_ROOT_BASE"

HERE="$REPO/samples/velocity_tendencies"
DATA_DIR="${VELOCITY_DATA_DIR:-$HERE/data_r02b05}"
NPZ="${VELOCITY_NPZ:-$BUILD_ROOT_BASE/velocity_r02b05_nproma32.npz}"

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

DRIVER="$HERE/run_velocity_perf.py"
CSV="${CSV:-velocity_tendencies_${SLURM_JOB_ID:-local}.csv}"
NCPUS="${SLURM_CPUS_ON_NODE:-$TOPO_NCORES}"
VARIANTS="${VARIANTS:-loopexch noloopexch}"

for lane in $LANES; do
    case "$lane" in
        dace-gcc | dace-llvm) ;;
        baselines)
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
        srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
            "$PY" "$DRIVER" --variant "$v" --backend "$backend" --npz "$NPZ" --build-only
    done

    for t in $THREADS; do
        if [ "$t" -gt "$TOPO_NCORES" ]; then
            echo "skip threads=$t (> $TOPO_NCORES physical cores)"
            continue
        fi
        set_omp_env "$t"
        for v in $VARIANTS; do
            srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                "$PY" "$DRIVER" --variant "$v" --backend "$backend" --npz "$NPZ" --csv "$CSV"
        done
    done
done
echo "done: $CSV"
