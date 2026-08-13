#!/bin/bash
# Standalone velocity-tendencies GPU verification: run the 4 specialised stage-4 SDFGs through
# main.cpp's got/want dispatcher and diff the dumps.  This is a CORRECTNESS job, not a perf job --
# reps default to 1 and no timing is recorded.
# Site-neutral header (partition/account on the command line, repo-root submit):
#     sbatch -p debug -A g34 samples/velocity_tendencies/opt_pipeline/run_standalone_gpu_verify.sbatch
# Knobs: VT_STAGE (4), VT_TIMESTEPS (1,2,7,9), VT_REPS (1), VT_DATA (data_r02b05).

#SBATCH --job-name=velocity-standalone-gpu-verify
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH -o samples/_work/logs/velocity_standalone_gpu_verify-%j.out
#SBATCH -e samples/_work/logs/velocity_standalone_gpu_verify-%j.err

set -uo pipefail

# Slurm spools the script, so ${BASH_SOURCE[0]} is under /var/spool in a job: resolve the repo
# from an explicit override, then the submit dir (repo root, or this script's own directory
# three levels down), same ladder as run_velocity.sbatch.
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
    for c in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../../..}" \
        "$(dirname "${BASH_SOURCE[0]}")/../../.."; do
        [ -n "$c" ] && [ -f "$c/samples/common.sh" ] || continue
        (cd "$c" && pwd)
        return 0
    done
    echo "FATAL: cannot locate the dace-fortran repo; pass DACE_FORTRAN_REPO=<repo>." >&2
    return 1
}
REPO="$(resolve_repo)" || exit 1
# common.sh sources env.sh (CUDA prefix, gcc/llvm loads) and alloc_pool.sh, so this job runs on
# the same mimalloc preload as every measurement lane rather than on glibc malloc.
source "$REPO/samples/common.sh"

HERE="$REPO/samples/velocity_tendencies/opt_pipeline"
STAGE="${VT_STAGE:-4}"
EXE="$HERE/velocity_stage${STAGE}"
DATA="${VT_DATA:-data_r02b05}"
TIMESTEPS="${VT_TIMESTEPS:-1,2,7,9}"
REPS="${VT_REPS:-1}"
OUTDIR="$WORK_ROOT/meas/velocity_standalone_gpu/${SLURM_JOB_ID:-local}"

[ -x "$EXE" ] || {
    echo "FATAL: no binary at $EXE -- build it first with:" >&2
    echo "  GENCODE_NUMBER=90a \$PYTHON -m utils.stages.stage${STAGE} --compile" >&2
    exit 1
}

mkdir -p "$OUTDIR"
nvidia-smi -L
echo "exe=$EXE data=$DATA timesteps=$TIMESTEPS reps=$REPS alloc=${MEAS_ALLOC_ACTIVE:-unknown}"
echo "outdir=$OUTDIR"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-72}" --gpus-per-task=1 \
    --cpu-bind=verbose,cores \
    bash -c "cd '$HERE' && '$EXE' --data '$DATA' --reps '$REPS' --timesteps '$TIMESTEPS' --output-dir '$OUTDIR'"
run_rc=$?
echo "run rc=$run_rc; dumps: $(ls -1 "$OUTDIR" 2>/dev/null | wc -l) files"
[ "$run_rc" -eq 0 ] || exit "$run_rc"

# The differ is pure python + polars over the .got/.want text dumps -- no GPU, no srun.
"$PY" "$HERE/utils/compare_got_and_want.py" --root "$OUTDIR" --timesteps "$TIMESTEPS"
echo "compare rc=$?"
