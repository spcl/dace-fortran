#!/bin/bash
# vexx CPU scaling, mode 2 (grid): few bands, huge grid (m=4, nnr=128^3).
# Site-neutral header (cloudsc sibling convention): pass partition/account on the command line:
#     sbatch -p <partition> -A <account> samples/vexx/run_vexx_grid.sbatch
#     Submit from the repo root (or from this script's own directory); Slurm spools the
#     script, so the repo is found via $SLURM_SUBMIT_DIR. DACE_FORTRAN_REPO=<repo> overrides,
#     and is REQUIRED alongside -D/--chdir, which repoints $SLURM_SUBMIT_DIR away from the repo.

#SBATCH --job-name=vexx-grid-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=vexx_grid_%j.out

set -euo pipefail

MODE=grid
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
# LANES: space list from {dace-gcc,dace-llvm} (cloudsc sbatch convention; no Fortran baseline
# lanes here -- no standalone driver exists for this kernel outside the harness).  Any other
# token is skipped, so a global LANES carrying cloudsc-only lanes still runs this job.
LANES="${LANES:-dace-gcc}"
BUILD_ROOT_BASE="${BUILD_ROOT:-$HOME/.cache/dace-fortran-samples}"
# exported so the baselines lane (which unsets BUILD_ROOT to escape the dace lanes' dacecache)
# still lands under this root instead of falling back to $HOME.
export BUILD_ROOT_BASE

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
            echo "SKIP unknown lane: $lane" >&2
            continue
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
