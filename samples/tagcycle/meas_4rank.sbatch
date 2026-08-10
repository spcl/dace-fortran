#!/bin/bash
#SBATCH -J meas-4rank
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/meas_4rank-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/meas_4rank-%j.out
# Step 4 of the tag cycle: THE measurement. One Grace node = 4 sockets x 72 cores = 4 ranks, one
# variant per rank, all four concurrent, each pinned to its own socket. Every variant gets its own
# log and its own CSV, and a red rank never takes its siblings down.
#
# WALLTIME: 02:00:00 here; the single-variant jobs got 06:00:00 from _harness/submit.sh and a
# cloudsc-nblks sweep once died at 2h36. Raise -t (up to the 24h partition cap) before submitting
# if the previous cycle came close.
#
# ABSOLUTE RULE: no numactl anywhere below. This node exposes 36 NUMA nodes; numactl --cpunodebind
# inside an srun cpuset returns sched_setaffinity EINVAL and kills the step outright (4390808,
# 4390809). srun --cpu-bind=cores --mem-bind=local does the same job correctly.
#   sbatch --export=ALL,TAG=<tag> meas_4rank.sbatch
set -uo pipefail

PROBE=/capstor/scratch/cscs/ybudanaz/aarch64/probe
MEAS=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas
BR2=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas
RUNS="$BR2/runs"
PROV="$RUNS/PROVENANCE.txt"
VARIANTS="${VARIANTS:-cloudsc-klon cloudsc-nblks velocity-gcc velocity-llvm}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

mkdir -p "$RUNS"
cd "$MEAS" || exit 1
ulimit -c 0
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1

# The clone is frozen at this point: a phase A cache miss must die, not quietly rebuild the bridge
# from a compute node into the shared tree.
export DACE_FORTRAN_NO_REBUILD=1
export EXPECTED_TAG="$TAG" MEAS BR2
export DACE_compiler_allow_view_arguments=false
export THREADS="${THREADS:-1 2 4 8 16 32 64 72}"

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "MEAS host=$(hostname) tag=$TAG_NOW expected=$TAG job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
echo "THREADS=$THREADS"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $TAG" >&2
    exit 1
}
if [ -w "$MEAS/dace_fortran" ]; then
    echo "WARN: $MEAS is WRITABLE -- it should have been refrozen by tagcycle_warm.sbatch" >&2
fi

{
    [ -f "$PROV" ] || echo "# utc | job id | job | clone tag | notes"
    printf '%s | %-8s | %-11s | %s | %s\n' "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "meas_4rank" \
        "$TAG_NOW" "4 ranks share one node, socket-pinned"
} >> "$PROV"

declare -A PID RC
for v in $VARIANTS; do
    csv="$RUNS/${v}_${TAG}_${SLURM_JOB_ID}.csv"
    log="$RUNS/${v}_${TAG}_${SLURM_JOB_ID}.log"
    echo "rank -> $v  csv=$csv"
    CSV="$csv" srun --exact --ntasks=1 --cpus-per-task=72 --cpu-bind=cores --mem-bind=local \
        bash "$PROBE/tagcycle_lane.sh" meas "$v" > "$log" 2>&1 &
    PID[$v]=$!
done

# No `set -e` and no `|| true` on the wait itself: `wait x || true` would overwrite $? with 0.
for v in $VARIANTS; do
    wait "${PID[$v]}"
    RC[$v]=$?
done

overall=0
for v in $VARIANTS; do
    log="$RUNS/${v}_${TAG}_${SLURM_JOB_ID}.log"
    csv="$RUNS/${v}_${TAG}_${SLURM_JOB_ID}.csv"
    echo "===== tail $log ====="
    tail -20 "$log"
    rows=0
    [ -f "$csv" ] && rows="$(($(wc -l < "$csv") - 1))"
    echo "MEAS_${v}_EXIT=${RC[$v]} rows=$rows csv=$csv"
    [ "${RC[$v]}" -eq 0 ] || overall=1
done

echo "MEAS_ALL_EXIT=$overall"
echo "MEAS_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
