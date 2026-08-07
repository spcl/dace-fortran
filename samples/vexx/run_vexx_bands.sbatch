#!/bin/bash
# vexx CPU scaling, mode 1 (bands): many bands, moderate grid (m=64, nnr=48^3).
# Site-neutral header (cloudsc sibling convention): pass partition/account on the command line:
#     sbatch -p <partition> -A <account> samples/vexx/run_vexx_bands.sbatch

#SBATCH --job-name=vexx-bands-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=vexx_bands_%j.out

set -euo pipefail

MODE=bands
REPO="${DACE_FORTRAN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO/samples/common.sh"
# LANES: space list from {dace-gcc,dace-llvm} (cloudsc sbatch convention; no Fortran baseline
# lanes here -- no standalone driver exists for this kernel outside the harness).
LANES="${LANES:-dace-gcc}"
BUILD_ROOT_BASE="${BUILD_ROOT:-$HOME/.cache/dace-fortran-samples}"

# Topology into the log for the record; probed at runtime, never assumed.
lscpu | head -30
probe_topology
probe_compilers
echo "topology: sockets=$TOPO_NSOCKETS numa_nodes=$TOPO_NNUMA physical_cores=$TOPO_NCORES"
echo "physical cpu ids (numa-major): $TOPO_PHYS_CPUS"

DRIVER="$REPO/samples/vexx/run_vexx_perf.py"
CSV="${CSV:-vexx_${MODE}_${SLURM_JOB_ID:-local}.csv}"
NCPUS="${SLURM_CPUS_ON_NODE:-$TOPO_NCORES}"

for lane in $LANES; do
    case "$lane" in
        dace-gcc | dace-llvm) ;;
        *)
            echo "unknown lane: $lane" >&2
            exit 1
            ;;
    esac
    backend="${lane#dace-}"
    # Per-lane build root: a gcc-linked cached .so must never be reused by the llvm lane.
    BUILD_ROOT="$BUILD_ROOT_BASE/vexx_${MODE}_${lane}"
    setup_build_root "vexx_${MODE}_${lane}"

    # Phase A once, at max threads: build + optimize + link into the cache (untimed).
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
done
echo "done: $CSV"
