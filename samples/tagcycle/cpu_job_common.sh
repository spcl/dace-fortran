#!/usr/bin/env bash
# Shared body for the CPU measurement/warm jobs (cpu_*.sbatch).  SOURCE this from a batch
# script after setting VARIANTS and whatever sweep envs the lanes read.
#
# What it owns:
#   * the EXPECTED_TAG guard (a tree at the wrong commit, or dirty, must never produce rows)
#   * the memory split: the node is --exclusive with --mem=0, so the batch step owns all of
#     RealMemory; each of the N concurrent ranks then gets RealMemory/N via `srun --mem`, and
#     MEMSPLIT= is printed so the log says what each rank was allowed.  Without the split one
#     rank at NGPTOT=262144 (~28 GB of expanded deck) can starve its three siblings.
#   * one srun step per variant, all concurrent, socket-pinned, each with its own CSV and log
#   * <STAGE>_<variant>_EXIT= / <STAGE>_ALL_EXIT= verdict lines for the grep-able post-mortem
#
# ABSOLUTE RULE inherited from meas_4rank.sbatch: no numactl anywhere.  This node exposes 36
# NUMA nodes and numactl --cpunodebind inside an srun cpuset returns sched_setaffinity EINVAL
# and kills the step (4390808, 4390809).  srun --cpu-bind=cores --mem-bind=local is the way.

# Sourced by path out of the repo, so BASH_SOURCE is the real file here (not a slurm spool copy);
# the sourcing sbatch has normally already resolved REPO and this just re-uses it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
TAGDIR="$REPO/samples/tagcycle"
# MEAS/BR2 keep their names from the pinned-mirror era: MEAS is the tree under test (now the repo
# itself, there is no second checkout) and BR2 is the measurement output root.
MEAS="${MEAS:-$REPO}"
BR2="${BR2:-$WORK_ROOT/meas}"
RUNS="${RUNS:-$BR2/runs}"
PROV="$RUNS/PROVENANCE.txt"
STAGE="${STAGE:-MEAS}"
LANE_MODE="${LANE_MODE:-meas}"
CPUS_PER_RANK="${CPUS_PER_RANK:-72}"

: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"
: "${VARIANTS:?VARIANTS must be set before sourcing cpu_job_common.sh}"

mkdir -p "$RUNS"
cd "$MEAS" || exit 1
ulimit -c 0
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1

export EXPECTED_TAG="$TAG" REPO WORK_ROOT MEAS BR2
export DACE_compiler_allow_view_arguments=false

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "$STAGE host=$(hostname) tag=$TAG_NOW expected=$TAG job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
echo "THREADS=${THREADS:-unset} REPS=${REPS:-unset} VARIANTS=$VARIANTS"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: repo describes as $TAG_NOW, expected $TAG (check out the tag, clean the tree)" >&2
    echo "${STAGE}_ALL_EXIT=1"
    exit 1
}

# --- memory split -------------------------------------------------------------
# RealMemory is what slurm believes the node has; fall back to /proc when scontrol is unhappy
# (a wrong guess must not silently hand a rank the whole node).
NRANKS="$(echo "$VARIANTS" | wc -w)"
node_real_mb() {
    local mb=""
    if command -v scontrol > /dev/null 2>&1 && [ -n "${SLURMD_NODENAME:-}" ]; then
        mb="$(scontrol show node "$SLURMD_NODENAME" 2> /dev/null | tr ' ' '\n' \
            | sed -n 's/^RealMemory=//p' | head -1)"
    fi
    if [ -z "$mb" ]; then
        mb="$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)"
        echo "WARN: RealMemory unavailable from scontrol, using MemTotal $mb MB" >&2
    fi
    printf '%s' "$mb"
}
NODE_MB="$(node_real_mb)"
# 4 GB head-room for the batch shell, the srun daemons and page cache slack.
MEM_PER_RANK="$(((NODE_MB - 4096) / NRANKS))"
[ "$MEM_PER_RANK" -gt 0 ] || MEM_PER_RANK="$((NODE_MB / NRANKS))"
echo "MEMSPLIT=$MEM_PER_RANK  (node RealMemory=$NODE_MB MB over $NRANKS ranks)"

{
    [ -f "$PROV" ] || echo "# utc | job id | job | repo tag | notes"
    printf '%s | %-8s | %-11s | %s | %s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-none}" \
        "${SLURM_JOB_NAME:-cpu}" "$TAG_NOW" "$NRANKS ranks, ${MEM_PER_RANK}MB each, $STAGE"
} >> "$PROV"

# --- the ranks ----------------------------------------------------------------
# A VARIANTS entry is `name` or `name%KEY=VAL%KEY=VAL`: the %-suffixes are per-rank env, which
# is how one job shards a sweep dimension (backend, size list) across its four ranks without
# needing a variant per shard.  The rank's log/CSV/verdict key is the whole entry with the
# %/= punctuation flattened, so two shards of one variant never collide on a filename.
rank_key() { printf '%s' "$1" | tr '%=,' '___'; }
rank_variant() { printf '%s' "${1%%\%*}"; }
rank_env() { # newline-separated KEY=VAL for entry $1
    local rest="${1#*%}"
    [ "$rest" = "$1" ] && return 0
    printf '%s' "$rest" | tr '%' '\n'
}

declare -A CPU_PID CPU_RC
for entry in $VARIANTS; do
    key="$(rank_key "$entry")"
    v="$(rank_variant "$entry")"
    csv="$RUNS/${SLURM_JOB_NAME:-cpu}_${key}_${TAG}_${SLURM_JOB_ID:-local}.csv"
    log="$RUNS/${SLURM_JOB_NAME:-cpu}_${key}_${TAG}_${SLURM_JOB_ID:-local}.log"
    mapfile -t rank_overrides < <(rank_env "$entry")
    echo "rank -> $v  ${rank_overrides[*]:-}  csv=$csv"
    CSV="$csv" env "${rank_overrides[@]}" \
        srun --exact --ntasks=1 --cpus-per-task="$CPUS_PER_RANK" --mem="${MEM_PER_RANK}M" \
        --cpu-bind=cores --mem-bind=local \
        bash "$TAGDIR/tagcycle_lane.sh" "$LANE_MODE" "$v" > "$log" 2>&1 &
    CPU_PID[$key]=$!
done

# No `set -e` and no `|| true` on the wait: `wait x || true` would overwrite $? with 0.
for entry in $VARIANTS; do
    key="$(rank_key "$entry")"
    wait "${CPU_PID[$key]}"
    CPU_RC[$key]=$?
done

cpu_overall=0
for entry in $VARIANTS; do
    key="$(rank_key "$entry")"
    log="$RUNS/${SLURM_JOB_NAME:-cpu}_${key}_${TAG}_${SLURM_JOB_ID:-local}.log"
    csv="$RUNS/${SLURM_JOB_NAME:-cpu}_${key}_${TAG}_${SLURM_JOB_ID:-local}.csv"
    echo "===== tail $log ====="
    tail -20 "$log"
    rows=0
    [ -f "$csv" ] && rows="$(($(wc -l < "$csv") - 1))"
    echo "${STAGE}_${key}_EXIT=${CPU_RC[$key]} rows=$rows csv=$csv"
    [ "${CPU_RC[$key]}" -eq 0 ] || cpu_overall=1
done

echo "${STAGE}_ALL_EXIT=$cpu_overall"
echo "${STAGE}_DONE end=$(date -u +%FT%TZ)"
