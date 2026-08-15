#!/bin/bash
#SBATCH -J gpu-warm
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 01:00:00
#SBATCH -o samples/_work/logs/gpu_warm-%j.out
#SBATCH -e samples/_work/logs/gpu_warm-%j.out
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
# Stage 1 of the GPU cycle: BUILD AND WARM ONLY, nothing here is a measurement. It produces every
# artifact gpu_meas.sbatch refuses to build for itself:
#   * the reference binaries -- cloudsc CUDA (nvcc) and the velocity OpenACC-GPU driver (nvfortran
#     -acc=gpu) built against the noloopexch ACC twin, both under $GPUROOT so the measurement job
#     finds them by path. The loopexch twin builds the same way under lane openacc-ref-loopexch,
#     which is available but deliberately out of the default LANES: it is a different TU from the
#     one the DaCe GPU lanes compile, so it is not the like-for-like reference;
#   * the flat R02B06 velocity dump at nproma=491520 (nblks_e = 1);
#   * DaCe phase A for the three GPU lanes, by calling their runners with --reps 1.
#
# `normal` and one GPU: this stage compiles an -rdc=true CUDA link, a 27-region OpenACC TU, and
# three DaCe GPU codegens. If phase A is cold at this tag the hour below is not enough -- raise -t
# (24h partition cap) rather than letting a half-warmed cache fall into the debug measurement job.
#
# DELIBERATELY SEQUENTIAL, one GPU, unlike the concurrent per-GPU ranks of gpu_meas.sbatch. Warm
# wall time is not precious and nothing here is timed, so the lanes gain nothing from parallelism --
# while the thing that would make them parallel, srun, is exactly what must not wrap a build.
#
# NO srun ANYWHERE BELOW. srun starts tasks with SIGCHLD blocked; nvcc, nvfortran and dace's
# sdfg.compile() all fork, and cmake then spins in select() until the walltime. (gpu_meas.sbatch
# may use srun precisely because it compiles nothing.)
#
# This job submits nothing and resubmits nothing.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/gpu_warm.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
# MEAS/BR2 keep their pinned-mirror names: the tree under test (now the repo) and the output root.
MEAS="${MEAS:-$REPO}"
BR2="${BR2:-$WORK_ROOT/meas}"
GPUROOT="${GPUROOT:-$BR2/gpu}"
LOG="$BR2/logs"
LANES="${LANES:-cuda-ref openacc-ref dace-gpu dace-gpu-pipeline dace-gpu-manual}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

# The GPU matrix is the big flat shapes only; the small-block shapes stay CPU-only by owner rule.
CLOUDSC_NGPTOT="${CLOUDSC_NGPTOT:-65536}"
CLOUDSC_GPU_NPROMA="${CLOUDSC_GPU_NPROMA:-128}"
VELOCITY_FLAT_NPROMA="${VELOCITY_FLAT_NPROMA:-491520}"
# The grid was pinned by three r02b06 literals, so the GPU lane could only ever run 40 km. It is a
# knob now; the default keeps existing behaviour. 160 km is VELOCITY_FLAT_GRID=r02b05 with
# VELOCITY_FLAT_NPROMA=30720 (that deck's edge count, so nblks_e comes out 1 -- the flat shape).
VELOCITY_FLAT_GRID="${VELOCITY_FLAT_GRID:-r02b06}"
# VELOCITY_R06_DATA_DIR kept as a fallback so an existing caller that sets it still works.
VELOCITY_FLAT_DATA_DIR="${VELOCITY_FLAT_DATA_DIR:-${VELOCITY_R06_DATA_DIR:-$BR2/data_$VELOCITY_FLAT_GRID}}"
VELOCITY_TIMESTEP="${VELOCITY_TIMESTEP:-1}"
FLATDUMP="$GPUROOT/dump_${VELOCITY_FLAT_GRID}_nproma${VELOCITY_FLAT_NPROMA}"
# The DaCe velocity runner probes only its cache root; the flat deck lives beside the runs, so the
# path is handed over explicitly rather than guessed.
export VELOCITY_NPZ_FLAT="${VELOCITY_NPZ_FLAT:-$BR2/velocity_${VELOCITY_FLAT_GRID}_nproma${VELOCITY_FLAT_NPROMA}.npz}"

CUDA_EXE="$GPUROOT/cloudsc_gpu_baselines/dwarf-cloudsc-cuda"
ACC_ROOT="$GPUROOT/velocity_acc_gpu"
VELOCITY_SRC="$MEAS/samples/velocity_tendencies"
CLOUDSC_RUNNER="$MEAS/samples/gpu/run_cloudsc_gpu.py"
VELOCITY_RUNNER="$MEAS/samples/gpu/run_velocity_gpu.py"

# TU variant -> the ACC twin it is generated from. Both twins are the same 701-line kernel with ten
# (horizontal, jk) nests transposed, so both are legitimate GPU references -- but only noloopexch
# matches the DaCe GPU lanes' TU, which is why it is the one in the default LANES.
acc_src_for() {
    case "$1" in
        noloopexch) echo "$VELOCITY_SRC/velocity_advection_noloopexch_acc.f90" ;;
        loopexch) echo "$VELOCITY_SRC/velocity_advection_acc.f90" ;;
        *) return 1 ;;
    esac
}

mkdir -p "$LOG" "$GPUROOT"
cd "$MEAS" || exit 1
ulimit -c 0
# ICON transients are per-block; at nproma=491520 the driver's automatic arrays are GBs.
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
# mimalloc is the default allocator for every native lane, reference lanes included -- one
# comparison, one allocator. MEAS_ALLOC=system is the escape hatch; MEAS_ALLOC_ACTIVE records
# what actually loaded and is stamped into the CSV's alloc column by gpu_meas.sbatch.
if [ -f samples/alloc_pool.sh ]; then
    # shellcheck disable=SC1091
    . samples/alloc_pool.sh
else
    echo "WARN: no samples/alloc_pool.sh in this clone -- lanes run on system malloc" >&2
    MEAS_ALLOC_ACTIVE=system
fi
export MEAS_ALLOC="${MEAS_ALLOC:-system}" MEAS_ALLOC_ACTIVE

# Phase A BUILDS -- the whole point. NO_REBUILD belongs to the measurement job only.
unset DACE_FORTRAN_NO_REBUILD
unset BUILD_ROOT
# Read by gpu_baselines.sh (samples/common.sh setup_build_root) for the cuda-ref lane.
export BUILD_ROOT_BASE="$GPUROOT"
export EXPECTED_TAG="$TAG" REPO WORK_ROOT MEAS BR2
export DACE_compiler_allow_view_arguments=false
export CLOUDSC_NGPTOT CLOUDSC_GPU_NPROMA
export CUDA_VISIBLE_DEVICES=0

# CPU ids this job may actually touch. NOT lscpu: that reports all 288 node cpus regardless of the
# cpuset, and an OMP_PLACES id outside the cpuset is the EINVAL class of bug numactl hits here.
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

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "WARM_GPU host=$(hostname) tag=$TAG_NOW expected=$TAG job=${SLURM_JOB_ID:-local} start=$(date -u +%FT%TZ)"
echo "shape: cloudsc ngptot=$CLOUDSC_NGPTOT nproma=$CLOUDSC_GPU_NPROMA | velocity nproma=$VELOCITY_FLAT_NPROMA nblks_e=1"
echo "cpuset: $JOB_CPUS"
# --exclusive --mem=0: one rank owns the whole node's memory. No per-rank split exists here, unlike
# the 4-rank CPU job -- the lanes are sequential, so each in turn gets everything.
echo "MEMSPLIT=all-single-rank"
nvidia-smi -L
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: repo describes as $TAG_NOW, expected $TAG (check out the tag, clean the tree)" >&2
    exit 1
}
if ! nvidia-smi -L 2> /dev/null | grep -q '^GPU'; then
    echo "ABORT: no GPU visible -- this job needs --gpus-per-node=1" >&2
    exit 1
fi

need_runner() {
    [ -f "$1" ] && return 0
    echo "FATAL: no GPU runner at $1 -- samples/gpu/ is not in this clone at $TAG" >&2
    return 1
}

tu_of() {
    case "$1" in
        openacc-ref) echo noloopexch ;;
        openacc-ref-loopexch) echo loopexch ;;
        *) return 1 ;;
    esac
}

ensure_flat_dump() {
    [ -f "$FLATDUMP/manifest.txt" ] && return 0
    if [ ! -d "$VELOCITY_FLAT_DATA_DIR" ]; then
        echo "FATAL: VELOCITY_FLAT_DATA_DIR=$VELOCITY_FLAT_DATA_DIR is not a directory;" >&2
        echo "  cannot build the $VELOCITY_FLAT_GRID flat dump. Refusing to fall back to another grid." >&2
        return 1
    fi
    "$PY" "$VELOCITY_SRC/dump_data.py" --data-dir "$VELOCITY_FLAT_DATA_DIR" \
        --timestep "$VELOCITY_TIMESTEP" --nproma "$VELOCITY_FLAT_NPROMA" --out "$FLATDUMP"
}

# Flags mirror baselines.sh's openacc lane (-O3 -acc=gpu, no mem:managed, no -gpu=deepcopy):
# driver_velocity.f90 stages every array with explicit ENTER DATA, so the kernel's PRESENT clauses
# must hit real device memory. One build dir per TU: both twins define the same module names.
build_acc_gpu() {
    local tu="$1" src dir
    src="$(acc_src_for "$tu")" || return 1
    dir="$ACC_ROOT/$tu"
    [ -f "$src" ] || {
        echo "FATAL: no ACC twin at $src (regenerate with scripts/annotate_velocity_acc.py)" >&2
        return 1
    }
    [ -n "${NVFORTRAN:-}" ] || NVFORTRAN="$(command -v nvfortran)"
    [ -x "$NVFORTRAN" ] || {
        echo "FATAL: no nvfortran on PATH -- samples/env.sh must spack load nvhpc" >&2
        return 1
    }
    mkdir -p "$dir" || return 1
    "$NVFORTRAN" -O3 -acc=gpu -c "$src" -o "$dir/tu.o" -module "$dir" || return 1
    "$NVFORTRAN" -O3 -acc=gpu -c "$VELOCITY_SRC/driver_velocity.f90" -o "$dir/driver.o" \
        -module "$dir" || return 1
    "$NVFORTRAN" -O3 -acc=gpu "$dir/tu.o" "$dir/driver.o" -o "$dir/driver_velocity" || return 1
    VELOCITY_TU="$tu" "$dir/driver_velocity" "$FLATDUMP" openacc 0 1 0 > "$dir/warm.log" 2>&1 || {
        echo "FATAL: warm run of $dir/driver_velocity failed; tail:" >&2
        tail -10 "$dir/warm.log" >&2
        return 1
    }
    grep -q "^velocity_tendencies,$tu," "$dir/warm.log" || {
        echo "FATAL: $dir/driver_velocity did not self-label mode=$tu" >&2
        return 1
    }
}

# One lane, all output to its own log; the caller judges by exit code alone.
warm_lane() {
    case "$1" in
        cuda-ref)
            CSV="$GPUROOT/warm_cuda_ref.csv" GPU_LANES=cuda REPS=1 WARMUP=0 \
                bash "$MEAS/samples/cloudsc/gpu_baselines.sh" || return 1
            [ -x "$CUDA_EXE" ] || {
                echo "FATAL: gpu_baselines.sh left no binary at $CUDA_EXE" >&2
                return 1
            }
            ;;
        openacc-ref | openacc-ref-loopexch)
            ensure_flat_dump || return 1
            build_acc_gpu "$(tu_of "$1")" || return 1
            ;;
        dace-gpu)
            need_runner "$CLOUDSC_RUNNER" || return 1
            "$PY" "$CLOUDSC_RUNNER" --lane dace-gpu --reps 1 --csv "$GPUROOT/warm_dace-gpu.csv" || return 1
            ;;
        dace-gpu-pipeline | dace-gpu-manual)
            need_runner "$VELOCITY_RUNNER" || return 1
            "$PY" "$VELOCITY_RUNNER" --lane "$1" --reps 1 --csv "$GPUROOT/warm_$1.csv" || return 1
            ;;
        *)
            echo "FATAL: unknown GPU lane: $1" >&2
            return 1
            ;;
    esac
}

overall=0
for l in $LANES; do
    log="$LOG/warm_gpu_${l}_${TAG}_${SLURM_JOB_ID:-local}.log"
    # Same per-lane build folder gpu_meas.sbatch's worker uses, so the measurement job finds the
    # cache this job warms; DACE_cache=name alone would collide on the shared SDFG names.
    export DACE_default_build_folder="$GPUROOT/dacecache_$l" DACE_cache=name
    mkdir -p "$DACE_default_build_folder" || exit 1
    echo "=== warm $l -> $log"
    warm_lane "$l" > "$log" 2>&1
    r=$?
    tail -25 "$log"
    echo "WARM_GPU_${l}_EXIT=$r"
    [ "$r" -eq 0 ] || overall=1
done

echo "WARM_GPU_ARTIFACTS cuda=$CUDA_EXE acc_root=$ACC_ROOT dump=$FLATDUMP alloc=$MEAS_ALLOC_ACTIVE"
echo "WARM_GPU_ALL_EXIT=$overall"
echo "WARM_GPU_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
