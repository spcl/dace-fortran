#!/bin/bash
# CloudSC CPU scaling, mode 1 (klon): NBLOCKS=1, KLON=65536 -- parallelism from the inner JL loops.
# Site-neutral header (optarena submit_deterministic.sbatch convention), partition/account on the
# command line:  sbatch -p <partition> -A <account> samples/cloudsc/run_cloudsc_klon.sbatch
# Header shape: dace_perf_jobs cloudsc_variants; srun binding: cpu_codegen submit_daint_readable.

#SBATCH --job-name=cloudsc-klon-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=cloudsc_klon_%j.out

set -euo pipefail

MODE=klon
REPO="${DACE_FORTRAN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/samples/common.sh"
# LANES: space list from {dace-gcc,dace-llvm,baselines,gpu} (baselines/gpu scripts run both
# klon+nblks themselves -- request them from only one job of the pair).
LANES="${LANES:-dace-gcc}"
BUILD_ROOT_BASE="${BUILD_ROOT:-$HOME/.cache/dace-fortran-samples}"

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
