#!/bin/bash
#SBATCH -J meas-4rank
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 02:00:00
#SBATCH -o samples/_work/logs/meas_4rank-%j.out
#SBATCH -e samples/_work/logs/meas_4rank-%j.out
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
# Step 4 of the tag cycle: THE measurement. One Grace node = 4 sockets x 72 cores; one rank per
# VARIANT, all concurrent, each pinned to its own socket. Every variant gets its own
# log and its own CSV, and a red rank never takes its siblings down.
#
# cloudsc-klon IS NOT MEASURED ON CPU -- it is a GPU-only shape. KLON=65536/NBLOCKS=1 leaves the
# OpenMP block loop with exactly one iteration, so the only parallelism available is the inner JL
# loops, which the DaCe CPU lanes do not exploit: job 4411783 measured 15020 ms at 1 thread against
# 14779 ms at 72, flat across the whole grid, at ~1.66 h of timed work per backend. It is the right
# shape on GPU, where KLON becomes device-wide parallelism, and samples/gpu/run_cloudsc_gpu.py
# pins MODE=klon/NBLOCKS=1 for exactly that. The GPU lane builds its own SDFG under the GPU cache
# root, so dropping klon here costs the GPU nothing.
#
# WALLTIME: 02:00:00 for the three remaining ranks, which finish in 24-28 min. Raise -t (24h
# partition cap) if a cloudsc-nblks sweep ever comes close -- one died at 2h36 historically.
#
# ABSOLUTE RULE: no numactl anywhere below. This node exposes 36 NUMA nodes; numactl --cpunodebind
# inside an srun cpuset returns sched_setaffinity EINVAL and kills the step outright (4390808,
# 4390809). srun --cpu-bind=cores --mem-bind=local does the same job correctly.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/meas_4rank.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
TAGDIR="$REPO/samples/tagcycle"
# MEAS/BR2 keep their pinned-mirror names: the tree under test (now the repo) and the output root.
MEAS="${MEAS:-$REPO}"
BR2="${BR2:-$WORK_ROOT/meas}"
RUNS="$BR2/runs"
PROV="$RUNS/PROVENANCE.txt"
# cloudsc-klon is deliberately absent: it is a GPU-only shape (see the walltime note).
VARIANTS="${VARIANTS:-cloudsc-nblks velocity-loopexch velocity-noloopexch}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

mkdir -p "$RUNS"
cd "$MEAS" || exit 1
ulimit -c 0
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1

# A phase A cache miss must die here, not quietly rebuild the bridge from a compute node.
export DACE_FORTRAN_NO_REBUILD=1
export EXPECTED_TAG="$TAG" REPO WORK_ROOT MEAS BR2
export DACE_compiler_allow_view_arguments=false
export THREADS="${THREADS:-1 2 4 8 16 32 64 72}"

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "MEAS host=$(hostname) tag=$TAG_NOW expected=$TAG job=$SLURM_JOB_ID start=$(date -u +%FT%TZ)"
echo "THREADS=$THREADS"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: repo describes as $TAG_NOW, expected $TAG (check out the tag, clean the tree)" >&2
    exit 1
}

{
    [ -f "$PROV" ] || echo "# utc | job id | job | repo tag | notes"
    printf '%s | %-8s | %-11s | %s | %s\n' "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "meas_4rank" \
        "$TAG_NOW" "$(echo "$VARIANTS" | wc -w) ranks share one node, socket-pinned"
} >> "$PROV"

# --mem=0 above hands the whole node to the batch step; split it evenly so one rank cannot
# starve its siblings (a cloudsc rank at a large NGPTOT holds tens of GB of expanded deck).
NRANKS="$(echo "$VARIANTS" | wc -w)"
NODE_MB="$(scontrol show node "$SLURMD_NODENAME" 2> /dev/null | tr ' ' '\n' \
    | sed -n 's/^RealMemory=//p' | head -1)"
[ -n "$NODE_MB" ] || NODE_MB="$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)"
MEM_PER_RANK="$(((NODE_MB - 4096) / NRANKS))"
[ "$MEM_PER_RANK" -gt 0 ] || MEM_PER_RANK="$((NODE_MB / NRANKS))"
echo "MEMSPLIT=$MEM_PER_RANK  (node RealMemory=$NODE_MB MB over $NRANKS ranks)"

declare -A PID RC
for v in $VARIANTS; do
    csv="$RUNS/${v}_${TAG}_${SLURM_JOB_ID}.csv"
    log="$RUNS/${v}_${TAG}_${SLURM_JOB_ID}.log"
    echo "rank -> $v  csv=$csv"
    CSV="$csv" srun --exact --ntasks=1 --cpus-per-task=72 --mem="${MEM_PER_RANK}M" \
        --cpu-bind=cores --mem-bind=local \
        bash "$TAGDIR/tagcycle_lane.sh" meas "$v" > "$log" 2>&1 &
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
