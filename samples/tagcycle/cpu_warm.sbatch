#!/bin/bash
#SBATCH -J cpu-warm
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 01:00:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/cpu_warm-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/cpu_warm-%j.out
# Phase A for everything the new CPU jobs measure.  Needs the clone WRITABLE.
#
# CHUNKING: a cloudsc SDFG build is ~13 min and KLON/NBLOCKS are specialized into the graph, so
# the 7-size sweep is 7 builds per backend = 14 builds.  One hour of 4 ranks holds ~4 builds per
# rank, so the sweep is split into two submissions by CHUNK:
#   CHUNK=small  4096 8192 16384 32768   -> 4 sizes x 2 backends, 2 ranks x 4 builds  (~55 min)
#   CHUNK=large  65536 131072 262144     -> 3 sizes x 2 backends, 2 ranks x 3 builds  (~40 min)
# The other two ranks of each chunk build the non-DaCe binaries and stage the velocity decks,
# which are minutes, not tens of minutes -- they finish early and idle, which is fine.
#   sbatch --export=ALL,TAG=<tag>,CHUNK=small samples/tagcycle/cpu_warm.sbatch
#   sbatch --export=ALL,TAG=<tag>,CHUNK=large samples/tagcycle/cpu_warm.sbatch
set -uo pipefail

MEAS=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas
CHUNK="${CHUNK:-small}"
case "$CHUNK" in
    small) export CLOUDSC_SWEEP_SIZES="${CLOUDSC_SWEEP_SIZES:-4096 8192 16384 32768}" ;;
    large) export CLOUDSC_SWEEP_SIZES="${CLOUDSC_SWEEP_SIZES:-65536 131072 262144}" ;;
    *)
        echo "ABORT: CHUNK must be small|large (got '$CHUNK')" >&2
        echo "WARM_ALL_EXIT=1"
        exit 1
        ;;
esac

[ -w "$MEAS/dace_fortran" ] || {
    echo "ABORT: $MEAS is frozen -- phase A needs it writable (build_bridge flock)" >&2
    echo "WARM_ALL_EXIT=1"
    exit 1
}
# Phase A BUILDS; NO_REBUILD belongs to the measurement jobs only.
unset DACE_FORTRAN_NO_REBUILD

# Ranks 0/1 shard the sweep by backend (one SDFG family each, which is what makes the chunk fit
# the hour); ranks 2/3 build the two non-DaCe binaries and smoke them at 4096 columns.
export STAGE=WARM LANE_MODE=warm
export VARIANTS="${VARIANTS:-cloudsc-sweep%CLOUDSC_SWEEP_BACKENDS=gcc cloudsc-sweep%CLOUDSC_SWEEP_BACKENDS=llvm cloudsc-sweep-fortran cloudsc-sweep-c}"
export THREADS="${THREADS:-72}"

echo "CPU_WARM chunk=$CHUNK sizes=$CLOUDSC_SWEEP_SIZES"

# shellcheck disable=SC1091
. "$MEAS/samples/tagcycle/cpu_job_common.sh"
exit "$cpu_overall"
