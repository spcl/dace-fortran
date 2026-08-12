#!/bin/bash
# CloudSC CPU scaling, mode 2 (nblks): KLON=32, NBLOCKS=2048 -- parallelism from the IBL block loop;
# same NGPTOT=65536 as the klon mode, so the two sweeps are directly comparable.
# Site-neutral header (optarena submit_deterministic.sbatch convention), partition/account on the
# command line:  sbatch -p <partition> -A <account> samples/cloudsc/run_cloudsc_nblks.sbatch
#     Submit from the repo root (or from this script's own directory); Slurm spools the
#     script, so the repo is found via $SLURM_SUBMIT_DIR. DACE_FORTRAN_REPO=<repo> overrides,
#     and is REQUIRED alongside -D/--chdir, which repoints $SLURM_SUBMIT_DIR away from the repo.
# Header shape: dace_perf_jobs cloudsc_variants; srun binding: cpu_codegen submit_daint_readable.

#SBATCH --job-name=cloudsc-nblks-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH -o samples/_work/logs/cloudsc_nblks-%j.out
#SBATCH -e samples/_work/logs/cloudsc_nblks-%j.err
# The -o/-e paths are repo-relative on purpose: slurm resolves them against the submit dir, and
# this job is submitted from the repo root.

set -euo pipefail

MODE=nblks
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
# LANES: space list from {dace-gcc,dace-llvm,baselines,gpu} (baselines/gpu scripts run both
# klon+nblks themselves -- request them from only one job of the pair).
LANES="${LANES:-dace-gcc}"
BUILD_ROOT_BASE="${BUILD_ROOT:-$WORK_ROOT/cache}"
# exported so the baselines lane (which unsets BUILD_ROOT to escape the dace lanes' dacecache)
# still lands under this root instead of falling back to $HOME.
export BUILD_ROOT_BASE

# Idempotent input-deck fetch; on failure the driver falls back to synthetic inputs (with a warning).
if ! bash "$REPO/samples/cloudsc/download_data.sh"; then
    echo "WARN: download_data.sh failed; driver will fall back to --inputs synthetic" >&2
fi

lscpu | head -30
probe_topology
probe_compilers
echo "topology: sockets=$TOPO_NSOCKETS numa_nodes=$TOPO_NNUMA physical_cores=$TOPO_NCORES"
echo "physical cpu ids (numa-major): $TOPO_PHYS_CPUS"

DRIVER="$REPO/samples/cloudsc/run_cloudsc_perf.py"
CSV="${CSV:-cloudsc_${MODE}_${SLURM_JOB_ID:-local}.csv}"
NCPUS="${SLURM_CPUS_ON_NODE:-$TOPO_NCORES}"

# Phase A once at max threads (build + optimize + compile into the cache, untimed), then the sweep.
run_dace_lane() {
    local backend="$1" t
    set_omp_env "$TOPO_NCORES"
    srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
        "$PY" "$DRIVER" --mode "$MODE" --backend "$backend" --build-only
    for t in $THREADS; do
        if [ "$t" -gt "$TOPO_NCORES" ]; then
            echo "skip threads=$t (> $TOPO_NCORES physical cores)"
            continue
        fi
        set_omp_env "$t"
        srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
            "$PY" "$DRIVER" --mode "$MODE" --backend "$backend" --csv "$CSV"
    done
}

for lane in $LANES; do
    case "$lane" in
        dace-gcc | dace-llvm)
            # Per-lane dacecache: a gcc-built .so must never be reused by the llvm lane.
            BUILD_ROOT="$BUILD_ROOT_BASE/cloudsc_${MODE}_${lane}"
            setup_build_root "cloudsc_${MODE}_${lane}"
            run_dace_lane "${lane#dace-}"
            ;;
        baselines)
            srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                bash "$REPO/samples/cloudsc/baselines.sh"
            ;;
        gpu)
            srun --ntasks=1 --cpus-per-task="$NCPUS" --cpu-bind=verbose,cores \
                bash "$REPO/samples/cloudsc/gpu_baselines.sh"
            ;;
        *)
            echo "SKIP unknown lane: $lane" >&2
            continue
            ;;
    esac
done
echo "done: $CSV"
