#!/bin/bash
#SBATCH -J gpu-meas
#SBATCH -p debug
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o samples/_work/logs/gpu_meas-%j.out
#SBATCH -e samples/_work/logs/gpu_meas-%j.out
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
# Stage 2 of the GPU cycle: THE MEASUREMENT. One GH200 node is 4 modules = 4 sockets x 72 cores,
# each socket paired with its own H100, so the lanes run CONCURRENTLY, one lane per GPU -- the same
# shape as the 4-rank CPU job (meas_4rank.sbatch), not a sequential sweep of one device.
#
#   cloudsc   cuda-ref (the vendored dwarf CUDA driver)   dace-gpu
#   velocity  openacc-ref (driver_velocity -acc=gpu       dace-gpu-pipeline   dace-gpu-manual
#             against the NOLOOPEXCH ACC twin)
#
# LANE -> GPU is positional: lane i takes GPU (i mod NGPU) and lanes are launched in WAVES of NGPU,
# so no two concurrently-running lanes ever share a device. The split pair is one wave each
# (cloudsc 2 ranks, velocity 3 ranks); the whole-matrix smoke form is 5 lanes = two waves.
# Each rank gets its own srun step, socket-pinned with --cpu-bind=cores --mem-bind=local, an
# explicit CUDA_VISIBLE_DEVICES, and 1/N of the node's memory.
#
# openacc-ref is the noloopexch twin because that is the TU the DaCe GPU lanes compile; the
# loopexch twin is lane openacc-ref-loopexch, buildable and runnable but out of the default LANES.
# Every lane runs under the same allocator (mimalloc by default) and stamps it into an alloc column.
#
# It BUILDS NOTHING: both reference binaries and all three DaCe caches come from gpu_warm.sbatch,
# and a missing artifact is a loud abort, never a rebuild on the debug partition. That is also what
# makes srun safe here -- srun starts tasks with SIGCHLD blocked, which deadlocks cmake, and the
# only thing that would call cmake is a phase A rebuild that this job refuses to do. The dace lanes
# still run the SIGCHLD-shim preflight inside their step before touching a cache.
#
# ABSOLUTE RULE, inherited from meas_4rank.sbatch: no numactl anywhere below. This node exposes 36
# NUMA nodes; numactl --cpunodebind inside an srun cpuset returns sched_setaffinity EINVAL and kills
# the step outright. srun --cpu-bind=cores --mem-bind=local does the same job correctly.
#
# WALLTIME: with the lanes concurrent a wave costs about its SLOWEST lane, not their sum --
# cloudsc ~6 min, velocity ~8-10 min (the 6.9 GB deck load dominates and three of them contend for
# capstor; RANK_STAGGER_S spreads the load phase, never the timed region). The whole-matrix form is
# two waves, ~15 min. All three fit debug, but the split pair is still the one to submit.
#
# This job submits nothing and resubmits nothing.
#   sbatch --export=ALL,TAG=<tag> gpu_meas.sbatch
set -uo pipefail

# Re-entrant: the launcher runs one srun step per lane, and each step re-enters this same file as a
# single-lane worker. $0 is the spool copy on a compute node, so self-reference goes through $REPO.
WORKER_MODE=0
if [ "${1:-}" = "--lane-worker" ]; then
    WORKER_MODE=1
    WORKER_LANE="$2"
    WORKER_CSV="$3"
    WORKER_LOG="$4"
    WORKER_GPU="$5"
fi

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
# A worker step re-enters this file BY REPO PATH, so BASH_SOURCE is the real file in that case.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
# MEAS/BR2 keep their pinned-mirror names: the tree under test (now the repo) and the output root.
MEAS="${MEAS:-$REPO}"
BR2="${BR2:-$WORK_ROOT/meas}"
GPUROOT="${GPUROOT:-$BR2/gpu}"
RUNS="$BR2/runs"
PROV="$RUNS/PROVENANCE.txt"
SELF="$REPO/samples/tagcycle/gpu_meas.sbatch"
LANES="${LANES:-cuda-ref dace-gpu openacc-ref dace-gpu-pipeline dace-gpu-manual}"
REPS="${REPS:-50}"
WARMUP="${WARMUP:-2}"
# Delay between backgrounded rank launches. Spreads the deck-load I/O only: each rank starts its own
# timed loop after its own load, so this never serializes a measured region.
RANK_STAGGER_S="${RANK_STAGGER_S:-15}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

CLOUDSC_NGPTOT="${CLOUDSC_NGPTOT:-65536}"
CLOUDSC_GPU_NPROMA="${CLOUDSC_GPU_NPROMA:-128}"
CLOUDSC_NBLOCKS=$(((CLOUDSC_NGPTOT + CLOUDSC_GPU_NPROMA - 1) / CLOUDSC_GPU_NPROMA))
VELOCITY_FLAT_NPROMA="${VELOCITY_FLAT_NPROMA:-491520}"
VELOCITY_INPUTS="${VELOCITY_INPUTS:-r02b06}"
FLATDUMP="$GPUROOT/dump_r02b06_nproma${VELOCITY_FLAT_NPROMA}"

CUDA_EXE="$GPUROOT/cloudsc_gpu_baselines/dwarf-cloudsc-cuda"
CUDA_RUNDIR="$GPUROOT/cloudsc_gpu_baselines/run"
ACC_ROOT="$GPUROOT/velocity_acc_gpu"
CLOUDSC_RUNNER="$MEAS/samples/gpu/run_cloudsc_gpu.py"
VELOCITY_RUNNER="$MEAS/samples/gpu/run_velocity_gpu.py"
CLOUDSC_HEADER="kernel,mode,klon,nblocks,threads,rep,ms,inputs,lane,alloc"
VELOCITY_HEADER="kernel,mode,nproma,nblks_e,threads,rep,ms,inputs,lane,alloc"

mkdir -p "$RUNS"
cd "$MEAS" || exit 1
ulimit -c 0
ulimit -s unlimited 2> /dev/null || echo "warning: could not raise the stack limit" >&2
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1
# python3.11 before python3: the system python3 on these nodes is 3.6, which no sample runs on.
PY="${PY:-${PYTHON:-$(command -v python3.11 || command -v python3 || true)}}"
[ -x "$PY" ] || {
    echo "ABORT: no usable interpreter at $PY (set PYTHON in samples/env.sh or export PY)" >&2
    exit 1
}
export PY PYTHON="$PY"
# mimalloc by default for EVERY lane, references included -- one comparison, one allocator, and the
# python that dlopens the dacecache .so must be under it too, so this is sourced before any runner.
# MEAS_ALLOC=system is the escape hatch; MEAS_ALLOC_ACTIVE is stamped into the CSV alloc column.
# Idempotent via MEAS_ALLOC_LOADED, which the srun steps inherit, so a worker does not re-preload.
if [ -f samples/alloc_pool.sh ]; then
    # shellcheck disable=SC1091
    . samples/alloc_pool.sh
else
    echo "WARN: no samples/alloc_pool.sh in this clone -- lanes run on system malloc" >&2
    MEAS_ALLOC_ACTIVE=system
fi
export MEAS_ALLOC="${MEAS_ALLOC:-system}" MEAS_ALLOC_ACTIVE

# A phase A cache miss must die here, not quietly rebuild the bridge from a compute node.
export DACE_FORTRAN_NO_REBUILD=1
export EXPECTED_TAG="$TAG" REPO WORK_ROOT MEAS BR2
export DACE_compiler_allow_view_arguments=false
unset BUILD_ROOT
export BUILD_ROOT_BASE="$GPUROOT"

# CPU ids THIS process may actually touch. NOT lscpu: that reports all 288 node cpus regardless of
# the cpuset. In a worker this is the step's own 72-core slice, which is exactly what we want to
# pin against. GPU lanes carry threads=0 in the CSV -- no CPU thread sweep applies -- so the host
# side is held to one core and one BLAS thread.
job_cpus() {
    local part lo hi
    for part in $(awk '/^Cpus_allowed_list:/ { print $2 }' /proc/self/status | tr ',' ' '); do
        case "$part" in
            *-*)
                lo="${part%-*}" hi="${part#*-}"
                seq "$lo" "$hi"
                ;;
            *) echo "$part" ;;
        esac
    done
}
JOB_CPUS="$(job_cpus | tr '\n' ' ')"
FIRST_CPU="${JOB_CPUS%% *}"
export OMP_PLACES="{$FIRST_CPU}" OMP_NUM_THREADS=1 OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1

socket_of_cpu() {
    lscpu -p=CPU,SOCKET 2> /dev/null | awk -F, -v c="$1" '!/^#/ && $1 == c { print $2; exit }'
}

kernel_of() {
    case "$1" in
        cuda-ref | dace-gpu) echo cloudsc ;;
        openacc-ref | openacc-ref-loopexch | dace-gpu-pipeline | dace-gpu-manual) echo velocity_tendencies ;;
        *) return 1 ;;
    esac
}

need() {
    [ -e "$1" ] && return 0
    echo "FATAL: missing $1 -- gpu_warm.sbatch builds it; this job never does" >&2
    return 1
}

# The two reference drivers print their own timing lines; both are converted here to the sample's
# 10-column schema so every CSV concatenates. threads is 0 and the lane column is the lane name
# above, not the driver's internal one, so the CSV rows match the verdict lines.
meas_cuda_ref() {
    local csv="$1" log="$2" rows
    need "$CUDA_EXE" || return 1
    (cd "$CUDA_RUNDIR" && CLOUDSC_REPS="$REPS" CLOUDSC_WARMUP="$WARMUP" \
        "$CUDA_EXE" 1 "$CLOUDSC_NGPTOT" "$CLOUDSC_GPU_NPROMA" > "$log" 2>&1) || {
        echo "FATAL: $CUDA_EXE failed; tail of $log:" >&2
        tail -10 "$log" >&2
        return 1
    }
    rows="$(awk -v k="cloudsc,nblks,$CLOUDSC_GPU_NPROMA,$CLOUDSC_NBLOCKS,0" -v a="$MEAS_ALLOC_ACTIVE" \
        '$1 == "REP" { print k "," $2 "," $3 ",h5,cuda-ref," a }' "$log")"
    [ -n "$rows" ] || {
        echo "FATAL: no REP lines from $CUDA_EXE -- driver predates the per-rep host timers" >&2
        return 1
    }
    echo "$CLOUDSC_HEADER" > "$csv"
    printf '%s\n' "$rows" >> "$csv"
}

# $3 is the TU variant the binary was linked against; VELOCITY_TU makes the driver self-label its
# mode column with it, so the row says which kernel actually ran rather than a build-time guess.
meas_openacc_ref() {
    local csv="$1" log="$2" tu="$3" lane="$4" exe="$ACC_ROOT/$3/driver_velocity" rows
    need "$exe" || return 1
    need "$FLATDUMP/manifest.txt" || return 1
    VELOCITY_TU="$tu" "$exe" "$FLATDUMP" openacc "0" "$REPS" "$WARMUP" > "$log" 2>&1 || {
        echo "FATAL: $exe failed; tail of $log:" >&2
        tail -10 "$log" >&2
        return 1
    }
    rows="$(awk -v i="$VELOCITY_INPUTS" -v l="$lane" -v a="$MEAS_ALLOC_ACTIVE" \
        'BEGIN { FS = ","; OFS = "," }
        /^velocity_tendencies,/ { $8 = i; $9 = l; print $0, a }' "$log")"
    [ -n "$rows" ] || {
        echo "FATAL: no CSV rows from $exe" >&2
        return 1
    }
    grep -q "^velocity_tendencies,$tu," "$log" || {
        echo "FATAL: $exe emitted mode != $tu -- wrong twin linked into $exe" >&2
        return 1
    }
    echo "$VELOCITY_HEADER" > "$csv"
    printf '%s\n' "$rows" >> "$csv"
}

# The DaCe runners own their own CSV. Until they emit the alloc column themselves, add it here so
# every file shares one 10-column schema; a runner that already emits it is left alone.
ensure_alloc_column() {
    local csv="$1"
    [ -f "$csv" ] || return 0
    head -1 "$csv" | grep -q ',alloc$' && return 0
    awk -v a="$MEAS_ALLOC_ACTIVE" 'NR == 1 { print $0 ",alloc"; next } { print $0 "," a }' \
        "$csv" > "$csv.alloc" && mv "$csv.alloc" "$csv"
}

# srun starts tasks with SIGCHLD blocked; dace's sdfg.compile() shells out to cmake, which reaps its
# configure helpers through SIGCHLD and otherwise spins in select() until the walltime. Only the
# combination is fatal, so the guard tests exactly that combination -- and only for dace lanes,
# which are the only ones that go near a cache.
dace_sigchld_preflight() {
    "$PY" -c "
import inspect, signal, sys
import dace, dace.codegen.compiler as c
blocked = signal.SIGCHLD in signal.pthread_sigmask(signal.SIG_BLOCK, [])
shim = 'SIGCHLD' in inspect.getsource(c)
print('DACE_PATH=' + dace.__file__)
print('SIGCHLD_BLOCKED=%d DACE_SIGCHLD_SHIM=%d' % (blocked, shim))
sys.exit(1 if (blocked and not shim) else 0)
" || {
        echo "FATAL: this dace has no SIGCHLD unblock shim and SIGCHLD is blocked inside the step" >&2
        return 1
    }
}

meas_lane() {
    local lane="$1" csv="$2" log="$3" r
    case "$lane" in
        cuda-ref) meas_cuda_ref "$csv" "$log" ;;
        openacc-ref) meas_openacc_ref "$csv" "$log" noloopexch openacc-ref ;;
        openacc-ref-loopexch) meas_openacc_ref "$csv" "$log" loopexch openacc-ref-loopexch ;;
        dace-gpu)
            need "$CLOUDSC_RUNNER" || return 1
            dace_sigchld_preflight || return 1
            "$PY" "$CLOUDSC_RUNNER" --lane dace-gpu --reps "$REPS" --csv "$csv" > "$log" 2>&1
            r=$?
            ensure_alloc_column "$csv"
            return "$r"
            ;;
        dace-gpu-pipeline | dace-gpu-manual)
            need "$VELOCITY_RUNNER" || return 1
            dace_sigchld_preflight || return 1
            "$PY" "$VELOCITY_RUNNER" --lane "$lane" --reps "$REPS" --csv "$csv" > "$log" 2>&1
            r=$?
            ensure_alloc_column "$csv"
            return "$r"
            ;;
        *)
            echo "FATAL: unknown GPU lane: $lane" >&2
            return 1
            ;;
    esac
}

# --- worker: one lane, one GPU, inside its own srun step ----------------------
if [ "$WORKER_MODE" = 1 ]; then
    export CUDA_VISIBLE_DEVICES="$WORKER_GPU"
    sock="$(socket_of_cpu "$FIRST_CPU")"
    echo "RANK lane=$WORKER_LANE gpu=$WORKER_GPU socket=${sock:-?} ncpus=$(echo "$JOB_CPUS" | wc -w) first_cpu=$FIRST_CPU"
    # Module affinity on Grace-Hopper: GPU i is local to socket i. Slurm hands consecutive --exact
    # steps consecutive socket-sized slices, which is what makes the positional mapping hold -- but
    # it is an ordering assumption, so it gets checked rather than trusted.
    if [ -n "$sock" ] && [ "$sock" != "$WORKER_GPU" ]; then
        echo "WARN: lane $WORKER_LANE runs on socket $sock but drives GPU $WORKER_GPU -- non-local" >&2
    fi
    meas_lane "$WORKER_LANE" "$WORKER_CSV" "$WORKER_LOG"
    exit $?
fi

# --- launcher -----------------------------------------------------------------
NGPU="${NGPU:-$(nvidia-smi -L 2> /dev/null | grep -c '^GPU')}"
[ "${NGPU:-0}" -gt 0 ] || NGPU=1
NLANES="$(echo "$LANES" | wc -w)"
WAVE="$NLANES"
[ "$WAVE" -gt "$NGPU" ] && WAVE="$NGPU"

# Node memory split per concurrent rank. --mem=0 gives the job everything; each step then asks for
# its share so one rank cannot starve its siblings.
node_mem_mb() {
    local m
    m="$(scontrol show node "${SLURMD_NODENAME:-$(hostname -s)}" 2> /dev/null \
        | tr ' ' '\n' | sed -n 's/^RealMemory=//p' | head -1)"
    if [ -n "$m" ] && [ "$m" -gt 0 ] 2> /dev/null; then
        echo "$m"
        return
    fi
    awk '/^MemTotal:/ { printf "%d\n", $2 / 1024 }' /proc/meminfo
}
NODE_MEM_MB="$(node_mem_mb)"
RANK_MEM_MB=$((NODE_MEM_MB / WAVE))

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "MEAS_GPU host=$(hostname) tag=$TAG_NOW expected=$TAG job=${SLURM_JOB_ID:-local} start=$(date -u +%FT%TZ)"
echo "lanes: $LANES reps=$REPS warmup=$WARMUP"
echo "layout: $NLANES lanes, $NGPU gpus, $WAVE concurrent ranks per wave, stagger=${RANK_STAGGER_S}s"
echo "shape: cloudsc ngptot=$CLOUDSC_NGPTOT nproma=$CLOUDSC_GPU_NPROMA nblocks=$CLOUDSC_NBLOCKS"
echo "shape: velocity nproma=$VELOCITY_FLAT_NPROMA nblks_e=1 inputs=$VELOCITY_INPUTS"
echo "alloc: $MEAS_ALLOC_ACTIVE (MEAS_ALLOC=$MEAS_ALLOC, so=${MEAS_ALLOC_SO:-none})"
# --exclusive --mem=0 at job level, RealMemory/nranks per step: the ranks are concurrent, so the
# node's memory is split rather than handed whole to one of them.
echo "MEMSPLIT=$RANK_MEM_MB"
echo "node memory: ${NODE_MEM_MB}MB total -> ${RANK_MEM_MB}MB per rank"
echo "cpuset: $JOB_CPUS"
nvidia-smi -L
echo "--- GPU/CPU affinity (provenance for the lane->gpu mapping) ---"
nvidia-smi topo -m 2> /dev/null | head -12
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: repo describes as $TAG_NOW, expected $TAG (check out the tag, clean the tree)" >&2
    exit 1
}
if ! nvidia-smi -L 2> /dev/null | grep -q '^GPU'; then
    echo "ABORT: no GPU visible -- this job needs --gpus-per-node=4" >&2
    exit 1
fi
[ -f "$SELF" ] || {
    echo "ABORT: cannot re-enter myself at $SELF (the srun steps run this file)" >&2
    exit 1
}

{
    [ -f "$PROV" ] || echo "# utc | job id | job | repo tag | notes"
    printf '%s | %-8s | %-11s | %s | %s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-local}" "gpu_meas" \
        "$TAG_NOW" "$WAVE ranks/wave, 1 gpu each, ${RANK_MEM_MB}MB/rank, reps=$REPS, alloc=$MEAS_ALLOC_ACTIVE, lanes: $LANES"
} >> "$PROV"

declare -A PID RC CSVOF LOGOF GPUOF
overall=0
idx=0
wave_lanes=()

# Collect one wave: never `wait x || true`, that would overwrite $? with 0.
drain_wave() {
    local lane
    for lane in "${wave_lanes[@]}"; do
        wait "${PID[$lane]}"
        RC[$lane]=$?
    done
    wave_lanes=()
}

for l in $LANES; do
    k="$(kernel_of "$l")" || {
        echo "MEAS_GPU_${l}_EXIT=1 rows=0 csv=- (unknown lane)"
        overall=1
        continue
    }
    gpu=$((idx % NGPU))
    csv="$RUNS/${k}_${l}_gpu_${TAG}_${SLURM_JOB_ID:-local}.csv"
    log="$RUNS/${k}_${l}_gpu_${TAG}_${SLURM_JOB_ID:-local}.log"
    CSVOF[$l]="$csv"
    LOGOF[$l]="$log"
    GPUOF[$l]="$gpu"
    echo "rank -> lane $l ($k) gpu=$gpu mem=${RANK_MEM_MB}MB csv=$csv"
    # The step asks for no GRES: the job holds all four GPUs and the worker selects one with
    # CUDA_VISIBLE_DEVICES, which keeps the device ids meaning what this script says they mean.
    # If a step ever reports no visible GPU, add --gpus-per-task=1 here and drop the manual
    # CUDA_VISIBLE_DEVICES in the worker -- Slurm then renumbers each task's device to 0.
    srun --exact --ntasks=1 --cpus-per-task=72 --mem="${RANK_MEM_MB}M" \
        --cpu-bind=cores --mem-bind=local \
        bash "$SELF" --lane-worker "$l" "$csv" "$log" "$gpu" \
        > "$log.rank" 2>&1 &
    PID[$l]=$!
    wave_lanes+=("$l")
    idx=$((idx + 1))
    if [ "${#wave_lanes[@]}" -ge "$NGPU" ]; then
        drain_wave
    elif [ "$RANK_STAGGER_S" -gt 0 ]; then
        sleep "$RANK_STAGGER_S"
    fi
done
[ "${#wave_lanes[@]}" -eq 0 ] || drain_wave

for l in $LANES; do
    [ -n "${RC[$l]:-}" ] || continue
    log="${LOGOF[$l]}"
    csv="${CSVOF[$l]}"
    r="${RC[$l]}"
    echo "===== lane $l (gpu ${GPUOF[$l]}) ====="
    grep -h '^RANK \|^WARN' "$log.rank" 2> /dev/null | head -4
    tail -20 "$log" 2> /dev/null
    rows=0
    [ -f "$csv" ] && rows="$(($(wc -l < "$csv") - 1))"
    [ "$rows" -eq "$REPS" ] || echo "WARN: lane $l wrote $rows data rows, expected $REPS" >&2
    if [ "$rows" -eq 0 ] && [ "$r" -eq 0 ]; then
        echo "FATAL: lane $l exited clean but wrote no rows to $csv" >&2
        r=1
    fi
    echo "MEAS_GPU_${l}_EXIT=$r rows=$rows csv=$csv"
    [ "$r" -eq 0 ] || overall=1
done

echo "MEAS_GPU_ALL_EXIT=$overall"
echo "MEAS_GPU_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
