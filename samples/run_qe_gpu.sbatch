#!/bin/bash
# QE GPU offloads, both kernels in ONE job: 1 node, 2 ranks -- rank 0 = addusxx_g on GPU 0,
# rank 1 = newdxx_g on GPU 1. Each rank walks both decks (BaO_nat002, BaTiO3_nat005) and both
# dump sets, writing its own CSV; the CSVs are merged per kernel at the end.
#     sbatch samples/run_qe_gpu.sbatch
# Submit from the repo root; DACE_FORTRAN_REPO=<repo> overrides when -D/--chdir is used.
#
# It BUILDS NOTHING. The dacecache is warmed on the LOGIN node (srun starts tasks with SIGCHLD
# blocked, which deadlocks cmake), so a missing cache is a loud abort and DACE_compiler_use_cache
# makes sdfg.compile() load the prebuilt .so instead of reconfiguring.
# No numactl anywhere: this node exposes 36 NUMA nodes and numactl --cpunodebind inside an srun
# cpuset returns EINVAL. srun --cpu-bind=cores --mem-bind=local does the job.

#SBATCH --job-name=qe-gpu
#SBATCH --partition=debug
#SBATCH --account=g34
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH -o samples/_work/logs/%x-%j.out
#SBATCH -e samples/_work/logs/%x-%j.err

set -uo pipefail
# No `set -e`: each rank's exit code is collected explicitly so one kernel's failure cannot
# abort its concurrent sibling.

resolve_repo() {
    local c
    if [ -n "${DACE_FORTRAN_REPO:-}" ]; then
        [ -f "$DACE_FORTRAN_REPO/samples/common.sh" ] || {
            echo "FATAL: DACE_FORTRAN_REPO=$DACE_FORTRAN_REPO holds no samples/common.sh" >&2
            return 1
        }
        (cd "$DACE_FORTRAN_REPO" && pwd)
        return 0
    fi
    for c in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../..}" \
        "$(dirname "${BASH_SOURCE[0]}")/.."; do
        [ -n "$c" ] && [ -f "$c/samples/common.sh" ] || continue
        (cd "$c" && pwd)
        return 0
    done
    echo "FATAL: cannot locate the dace-fortran repo; submit from the repo root or set DACE_FORTRAN_REPO" >&2
    return 1
}
REPO="$(resolve_repo)" || exit 1
export REPO

KERNELS="${QE_KERNELS:-addusxx_g newdxx_g}"
read -ra KERNELS_ARR <<< "$KERNELS"
DECKS="${QE_DECKS:-BaO_nat002 BaTiO3_nat005}"
SETS="${QE_SETS:-0 1}"
REPS="${QE_REPS:-20}"
WARMUP="${QE_WARMUP:-3}"
REP="${QE_REP:-1}"

# The venv that owns dace 2.0.0a5; handed to common.sh as PYTHON so its interpreter probe and
# every lane below agree on one interpreter. common.sh also sources samples/env.sh and exports
# DACE_cache_distaware=0 -- without that srun's SLURM_PROCID would send dace looking for a
# <cache>_rank0 folder the login node never built.
PY="${QE_GPU_PY:-$(cd "$REPO/.." && pwd)/venv-df313/bin/python}"
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY (set QE_GPU_PY)" >&2; exit 1; }
export PYTHON="$PY"

# shellcheck disable=SC1091
source "$REPO/samples/common.sh"
set +e

export DACE_compiler_cuda_cuda_arch=90
export DACE_cache=name
export DACE_compiler_use_cache=1
export DACE_default_build_folder="${QE_DACECACHE:-$WORK_ROOT/gpu_qe/dacecache}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for k in "${KERNELS_ARR[@]}"; do
    bash "$REPO/samples/$k/download_data.sh" || { echo "FATAL: download_data.sh failed for $k" >&2; exit 1; }
    cache="$DACE_default_build_folder/${k}_gpu"
    [ -d "$cache/build" ] || {
        echo "FATAL: no warmed cache at $cache -- warm it on the LOGIN node, this job never builds" >&2
        exit 1
    }
done

echo "QE_GPU host=$(hostname) job=${SLURM_JOB_ID:-local} start=$(date -u +%FT%TZ)"
echo "config: decks='$DECKS' sets='$SETS' reps=$REPS warmup=$WARMUP rep=$REP cache=$DACE_default_build_folder"
nvidia-smi -L
nvidia-smi topo -m 2> /dev/null | head -8

NODE_MB="$(scontrol show node "$(hostname)" 2> /dev/null | tr ' ' '\n' | sed -n 's/^RealMemory=//p' | head -1)"
[ -n "$NODE_MB" ] || NODE_MB="$(awk '/^MemTotal:/ { printf "%d", $2 / 1024 }' /proc/meminfo)"
MEM_PER_RANK="$(((NODE_MB - 4096) / 2))"
echo "MEMSPLIT=${MEM_PER_RANK}MB per rank (node RealMemory=${NODE_MB}MB)"

RANKDIR="$WORK_ROOT/logs/qe_gpu_${SLURM_JOB_ID:-local}_ranks"
mkdir -p "$RANKDIR"

# One rank per kernel, one GPU per rank. GH200 module affinity is positional (GPU i is local to
# socket i) and slurm hands consecutive --exact steps consecutive socket-sized slices, so the
# mapping is checked rather than trusted.
rank_body() {
    local kernel="$1" gpu="$2" deck csv rc=0 first_cpu sock
    export CUDA_VISIBLE_DEVICES="$gpu"
    first_cpu="$(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status | tr ',' '\n' | head -1 | cut -d- -f1)"
    sock="$(lscpu -p=CPU,SOCKET 2> /dev/null | awk -F, -v c="$first_cpu" '!/^#/ && $1 == c { print $2; exit }')"
    echo "RANK kernel=$kernel gpu=$gpu socket=${sock:-?} first_cpu=$first_cpu"
    [ -z "$sock" ] || [ "$sock" = "$gpu" ] || echo "WARN: $kernel on socket $sock drives GPU $gpu -- non-local" >&2
    for deck in $DECKS; do
        csv="$RANKDIR/${kernel}.${deck}.csv"
        # shellcheck disable=SC2086
        (cd "$REPO/samples/$kernel" && "$PY" "bench_${kernel%_g}_gpu.py" \
            --deck "$REPO/samples/$kernel/data/$deck" --sets $SETS \
            --reps "$REPS" --warmup "$WARMUP" --rep "$REP" --csv "$csv")
        [ $? -eq 0 ] || rc=1
    done
    return "$rc"
}
export -f rank_body
export RANKDIR DECKS SETS REPS WARMUP REP PY

declare -A PID RC
gpu=0
for k in "${KERNELS_ARR[@]}"; do
    log="$RANKDIR/$k.log"
    echo "rank -> $k gpu=$gpu log=$log"
    srun --exact --ntasks=1 --cpus-per-task=72 --mem="${MEM_PER_RANK}M" \
        --cpu-bind=cores --mem-bind=local \
        bash -c "rank_body $k $gpu" > "$log" 2>&1 &
    PID[$k]=$!
    gpu=$((gpu + 1))
done

for k in "${KERNELS_ARR[@]}"; do
    wait "${PID[$k]}"
    RC[$k]=$?
done

overall=0
for k in "${KERNELS_ARR[@]}"; do
    echo "===== $RANKDIR/$k.log ====="
    grep -E "^RANK |^WARN|VERIFY|COMPUTE|RESULT|FATAL|Error" "$RANKDIR/$k.log"
    out="$REPO/samples/$k/output_data/${k}_gpu_${SLURM_JOB_ID:-local}.csv"
    mkdir -p "$(dirname "$out")"
    first=1
    rows=0
    for csv in "$RANKDIR/$k".*.csv; do
        [ -f "$csv" ] || continue
        if [ "$first" -eq 1 ]; then cp "$csv" "$out"; first=0; else tail -n +2 "$csv" >> "$out"; fi
    done
    [ -f "$out" ] && rows="$(($(wc -l < "$out") - 1))"
    echo "KERNEL_${k}_EXIT=${RC[$k]} rows=$rows csv=$out"
    [ "${RC[$k]}" -eq 0 ] && [ "$rows" -gt 0 ] || overall=1
done

echo "QE_GPU_ALL_EXIT=$overall"
echo "QE_GPU_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
