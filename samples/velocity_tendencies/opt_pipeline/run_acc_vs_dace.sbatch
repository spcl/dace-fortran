#!/bin/bash
# Task 1.5: the original single-TU OpenACC Fortran velocity_tendencies against our split-4 DaCe
# variants, same timesteps, same serde-deserialised inputs, same binary, same compiler family
# (nvfortran / nvc++ / nvcc from one nvhpc).
#
# Two passes, because they need different rep counts:
#   correctness -- reps=1, so each engine's got/want dump reflects exactly one call; compared with
#                  utils/compare_got_and_want.py (polars) per engine against the same want files.
#   performance -- reps=50, engine timings only; the dumps after 50 reps are meaningless by design
#                  (neither engine restores state between reps, matching the pre-existing runner).
#
#     sbatch -p debug -A g34 samples/velocity_tendencies/opt_pipeline/run_acc_vs_dace.sbatch
# Knobs: VT_STAGE (5), VT_TIMESTEPS (1,2,7,9), VT_PERF_REPS (50), VT_DATA (data_r02b05).

#SBATCH --job-name=velocity-acc-vs-dace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH -o samples/_work/logs/velocity_acc_vs_dace-%j.out
#SBATCH -e samples/_work/logs/velocity_acc_vs_dace-%j.err

set -uo pipefail

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
source "$REPO/samples/common.sh"

HERE="$REPO/samples/velocity_tendencies/opt_pipeline"
# The binary links nvhpc's libnvf / libacchost / libaccdevice, so the same toolchain that built it
# has to be on LD_LIBRARY_PATH at run time.
source "$HERE/env_nvhpc.sh"

STAGE="${VT_STAGE:-5}"
EXE="$HERE/velocity_stage${STAGE}"
DATA="${VT_DATA:-data_r02b05}"
TIMESTEPS="${VT_TIMESTEPS:-1,2,7,9}"
PERF_REPS="${VT_PERF_REPS:-50}"
OUTDIR="$WORK_ROOT/meas/velocity_acc_vs_dace/${SLURM_JOB_ID:-local}"

[ -x "$EXE" ] || {
    echo "FATAL: no binary at $EXE -- build it first with:" >&2
    echo "  . ./env_nvhpc.sh && VT_WITH_ACC=1 \$PYTHON -m utils.stages.stage${STAGE} --compile" >&2
    exit 1
}

mkdir -p "$OUTDIR"
nvidia-smi -L
echo "exe=$EXE data=$DATA timesteps=$TIMESTEPS perf_reps=$PERF_REPS alloc=${MEAS_ALLOC_ACTIVE:-unknown}"
echo "outdir=$OUTDIR"

run_engine_pass() {
    local reps="$1" outdir="$2"
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-72}" --gpus-per-task=1 \
        --cpu-bind=verbose,cores \
        bash -c "cd '$HERE' && '$EXE' --data '$DATA' --reps '$reps' --engine both \
                 --timesteps '$TIMESTEPS' --output-dir '$outdir'"
}

echo "=== pass 1/2: correctness (reps=1) ==="
run_engine_pass 1 "$OUTDIR/verify" | tee "$OUTDIR/verify.log"
rc=${PIPESTATUS[0]}
echo "verify run rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

fail=0
for eng in dace acc; do
    echo "--- compare $eng ---"
    "$PY" "$HERE/utils/compare_got_and_want.py" --root "$OUTDIR/verify/$eng" --timesteps "$TIMESTEPS"
    erc=$?
    echo "compare[$eng] rc=$erc"
    [ "$erc" -eq 0 ] || fail=1
done

echo "=== pass 2/2: performance (reps=$PERF_REPS) ==="
run_engine_pass "$PERF_REPS" "$OUTDIR/perf" | tee "$OUTDIR/perf.log"
rc=${PIPESTATUS[0]}
echo "perf run rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

# One tidy CSV of the head-to-head, straight out of the runner's TIME rows.
grep -h '^TIME,' "$OUTDIR/perf.log" | sed 's/^TIME,//' > "$OUTDIR/acc_vs_dace_timings.csv"
echo "timings: $OUTDIR/acc_vs_dace_timings.csv"
cat "$OUTDIR/acc_vs_dace_timings.csv"

exit "$fail"
