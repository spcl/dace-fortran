#!/bin/bash
# QE us_exx CPU scaling for one kernel sample: the standalone OpenMP baseline built by both
# Fortran compilers plus the DaCe SDFG built by both C++ compilers, swept over OMP threads on
# both dumped material decks.
# Site-neutral header, partition/account on the command line:
#     sbatch -p debug -A <account> samples/newdxx_g/run_newdxx_g_cpu.sbatch
#     Submit from the repo root (or from this script's own directory); Slurm spools the script,
#     so the repo is found via $SLURM_SUBMIT_DIR. DACE_FORTRAN_REPO=<repo> overrides, and is
#     REQUIRED alongside -D/--chdir, which repoints $SLURM_SUBMIT_DIR away from the repo.
# Header shape + lane layout: samples/cloudsc/run_cloudsc_nblks.sbatch.
#
# LAYOUT: these nodes are 4 sockets x 72 cores, so the 4 lanes run fully concurrently, one per
# socket, each claiming a whole socket via `srun --exact --cpus-per-task=72 --cpu-bind=cores
# --mem-bind=local` (never numactl). No lane shares cores with another, so the full THREADS
# list stays valid for every lane.

#SBATCH --job-name=newdxx_g-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o samples/_work/logs/%x-%j.out
#SBATCH -e samples/_work/logs/%x-%j.err
# The -o/-e paths are repo-relative on purpose: slurm resolves them against the submit dir, and
# this job is submitted from the repo root.

set -uo pipefail
# No `set -e`: exit codes are collected explicitly below so one lane's failure cannot abort its
# concurrent sibling (`wait lane || true` would instead overwrite $? with 0).

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
REPO="$(resolve_repo)" || exit 1
export REPO

# The job name carries the sample directory (see the #SBATCH line above); QE_KERNEL overrides it
# for a submission that renames the job.
KERNEL="${QE_KERNEL:-${SLURM_JOB_NAME:-newdxx_g-cpu}}"
KERNEL="${KERNEL%-cpu}"
KDIR="$REPO/samples/$KERNEL"
WORKER="$KDIR/baseline/cpu/run_lane_worker.sh"
if [ ! -x "$WORKER" ]; then
    echo "FATAL: no lane worker at $WORKER (KERNEL=$KERNEL; set QE_KERNEL=<sample dir>)" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$REPO/samples/common.sh"
set +e

LANES="${LANES:-original-openmp flang-openmp dace-gcc dace-llvm}"
read -ra LANES_ARR <<< "$LANES"

JOB_BUILD_ROOT_BASE="${BUILD_ROOT:-$WORK_ROOT/cache}"
# Per-lane build root: a gfortran .mod tree must never be reused by the flang lane.
lane_build_root_base() { printf '%s/%s_cpu_lane_%s' "$JOB_BUILD_ROOT_BASE" "$KERNEL" "$1"; }

# Idempotent deck fetch, once, before any lane forks: the lanes share $KDIR/data read-only.
if ! bash "$KDIR/download_data.sh"; then
    echo "FATAL: download_data.sh failed; no deck means no verified row" >&2
    exit 1
fi

lscpu | head -30
probe_topology
probe_compilers
echo "node topology (unconfined view): sockets=$TOPO_NSOCKETS numa_nodes=$TOPO_NNUMA physical_cores=$TOPO_NCORES"

# --mem=0 hands this step the whole node; split over the 4 physical sockets (site convention).
node_real_mb() {
    local mb=""
    mb="$(scontrol show node "$(hostname)" 2> /dev/null | tr ' ' '\n' | sed -n 's/^RealMemory=//p' | head -1)"
    if [ -z "$mb" ]; then
        mb="$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)"
        echo "WARN: RealMemory unavailable from scontrol, using local MemTotal $mb MB" >&2
    fi
    printf '%s' "$mb"
}
NODE_MB="$(node_real_mb)"
MEM_PER_RANK="$(((NODE_MB - 4096) / 4))"
[ "$MEM_PER_RANK" -gt 0 ] || MEM_PER_RANK="$((NODE_MB / 4))"
echo "MEMSPLIT=${MEM_PER_RANK}MB per rank (node RealMemory=${NODE_MB}MB over 4 sockets)"

CSV="${CSV:-$KDIR/output_data/${KERNEL}_cpu_${SLURM_JOB_ID:-local}.csv}"
RANKDIR="$WORK_ROOT/logs/${KERNEL}_cpu_${SLURM_JOB_ID:-local}_ranks"
mkdir -p "$RANKDIR" "$(dirname "$CSV")"

declare -A PID RC
for lane in "${LANES_ARR[@]}"; do
    csv="$RANKDIR/$lane.csv"
    log="$RANKDIR/$lane.log"
    echo "rank -> $lane  csv=$csv"
    REPO="$REPO" CSV="$csv" BUILD_ROOT_BASE="$(lane_build_root_base "$lane")" \
        srun --exact --ntasks=1 --cpus-per-task=72 --mem="${MEM_PER_RANK}M" \
        --cpu-bind=cores --mem-bind=local \
        bash "$WORKER" "$lane" > "$log" 2>&1 &
    PID[$lane]=$!
done

for lane in "${LANES_ARR[@]}"; do
    wait "${PID[$lane]}"
    RC[$lane]=$?
done

# Merge: each lane wrote its own CSV (no concurrent writers on one file), header taken once.
overall=0
first=1
for lane in "${LANES_ARR[@]}"; do
    log="$RANKDIR/$lane.log"
    csv="$RANKDIR/$lane.csv"
    echo "===== tail $log ====="
    tail -20 "$log"
    rows=0
    if [ -f "$csv" ]; then
        rows="$(($(wc -l < "$csv") - 1))"
        if [ "$first" -eq 1 ]; then
            cp "$csv" "$CSV"
            first=0
        else
            tail -n +2 "$csv" >> "$CSV"
        fi
    fi
    echo "LANE_${lane}_EXIT=${RC[$lane]} rows=$rows csv=$csv"
    [ "${RC[$lane]}" -eq 0 ] || overall=1
done

echo "done: $CSV"
echo "QE_CPU_ALL_EXIT=$overall"
exit "$overall"
