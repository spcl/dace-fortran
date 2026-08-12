#!/usr/bin/env bash
# "Velocity integrated into ICON" timing experiment.
#
# Drives the ICON-integration binding e2e (tests/icon/full/test_velocity_full_bindings_e2e.py --
# the DaCe-generated velocity_tendencies_dace driven through the same flat caller ICON's patched
# mo_velocity_advection reaches) once per (istep, lvn) configuration, VELO_REPS timing-only
# invocations per pass on top of the correctness call, captures stdout, and reduces the
# VELO_TIMER lines to a median per configuration with parse_icon_timers.py.
#
# Runs unchanged on a login node, on an interactive compute node, and inside a future sbatch:
# no srun, no salloc, no MPI.  A batch script's payload is just `bash run_icon_velo_timing.sh`.
#
# Knobs (all env):
#   PASSES      pytest passes per configuration                     (default 3)
#   VELO_REPS   timing-only invocations per pass, per side          (default 15)
#   CONFIGS     space-separated istep:lvn pairs                     (default the full 2x2)
#   TEST        the ICON-integration test to drive
#   KEXPR       -k selector inside TEST                             (default inline)
#   OUT_DIR     log + CSV destination                               (default $PWD/velo_timing)
#   VELO_FC     THE Fortran compiler, single-sourced                (default gfortran)
#   PYTEST_PY   interpreter that owns dace / dace_fortran / flang
#   PARSE_PY    interpreter for the parser (numpy only)
#
# VELO_FC is the ONE place a Fortran compiler is named.  ICON and the SDFG binding it
# dispatches into must be built by the same compiler or the derived-type ABI and the module
# layout do not line up, so the test reads VELO_FC too and nothing hard-codes a compiler name.
# The resolved absolute path and version of both VELO_FC and flang are recorded into the log as
# a VELO_TOOLCHAIN line, so a captured log always carries the provenance of its own numbers.
# KEXPR defaults to `inline` for the same reason: that arm is the one whose link line VELO_FC
# controls end to end (`build_fortran_library` picks its own compiler internally).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# Same env-hook contract as samples/common.sh: an explicit SAMPLES_ENV must resolve, otherwise
# samples/env.sh when present.  This is what puts flang / gfortran / the spack python on PATH.
if [ -n "${SAMPLES_ENV:-}" ]; then
    if [ ! -f "$SAMPLES_ENV" ]; then
        echo "FATAL: SAMPLES_ENV=$SAMPLES_ENV does not exist" >&2
        echo "ICON_VELO_EXIT=2"
        exit 2
    fi
elif [ -f "$REPO/samples/env.sh" ]; then
    SAMPLES_ENV="$REPO/samples/env.sh"
fi
if [ -n "${SAMPLES_ENV:-}" ] && [ -z "${SAMPLES_ENV_LOADED:-}" ]; then
    echo "velo-timing: sourcing env hook $SAMPLES_ENV"
    # shellcheck disable=SC1090
    . "$SAMPLES_ENV"
    SAMPLES_ENV_LOADED=1
    export SAMPLES_ENV SAMPLES_ENV_LOADED
fi

PASSES="${PASSES:-3}"
VELO_REPS="${VELO_REPS:-15}"
CONFIGS="${CONFIGS:-1:0 1:1 2:0 2:1}"
TEST="${TEST:-tests/icon/full/test_velocity_full_bindings_e2e.py}"
KEXPR="${KEXPR:-inline}"
OUT_DIR="${OUT_DIR:-$PWD/velo_timing}"
mkdir -p "$OUT_DIR" || { echo "ICON_VELO_EXIT=2"; exit 2; }
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
TAG="${SLURM_JOB_ID:-local}"
LOG="${OUT_DIR}/icon_velo_${TAG}.log"
CSV="${OUT_DIR}/icon_velo_${TAG}.csv"
PARSER="$HERE/parse_icon_timers.py"

pick_python() {
    for cand in "$@"; do
        [ -n "$cand" ] && [ -x "$cand" ] && { echo "$cand"; return 0; }
    done
    command -v python3
}
PYTEST_PY="$(pick_python "${PYTEST_PY:-}" "${PYTHON:-}")"
PARSE_PY="$(pick_python "${PARSE_PY:-}" "$PYTEST_PY")"

VELO_FC="${VELO_FC:-gfortran}"
FC_PATH="$(command -v "$VELO_FC" || true)"
if [ -z "$FC_PATH" ]; then
    echo "FATAL: VELO_FC=$VELO_FC is not on PATH" >&2
    echo "ICON_VELO_EXIT=2"
    exit 2
fi
FLANG_PATH="$(command -v flang-22 flang-new-21 flang-new flang 2>/dev/null | head -1)"
if [ -z "$FLANG_PATH" ]; then
    echo "FATAL: no flang on PATH -- the binding e2e would SKIP and measure nothing" >&2
    echo "ICON_VELO_EXIT=2"
    exit 2
fi
FC_VERSION="$("$FC_PATH" --version 2>&1 | head -1)"
FLANG_VERSION="$("$FLANG_PATH" --version 2>&1 | head -1)"

: > "$LOG"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export MPI4PY_RC_INITIALIZE=0
export VELO_REPS VELO_FC

echo "VELO_TOOLCHAIN fc=$FC_PATH fc_version='$FC_VERSION' flang=$FLANG_PATH flang_version='$FLANG_VERSION'" \
    | tee -a "$LOG"
echo "velo-timing: repo=$REPO"
echo "velo-timing: test=$TEST -k $KEXPR passes=$PASSES reps=$VELO_REPS configs='$CONFIGS'"
echo "velo-timing: pytest_py=$PYTEST_PY parse_py=$PARSE_PY"
echo "velo-timing: log=$LOG csv=$CSV"

cd "$REPO" || { echo "ICON_VELO_EXIT=2"; exit 2; }

fail=0
for cfg in $CONFIGS; do
    istep="${cfg%%:*}"
    lvn="${cfg##*:}"
    for pass in $(seq 1 "$PASSES"); do
        echo "velo-timing: istep=$istep lvn=$lvn pass=$pass/$PASSES"
        VELO_ISTEP="$istep" VELO_LVN="$lvn" \
            "$PYTEST_PY" -m pytest -s -q --no-header -p no:cacheprovider \
            "$TEST" -k "$KEXPR" 2>&1 | tee -a "$LOG"
        rc="${PIPESTATUS[0]}"
        echo "VELO_PYTEST_EXIT istep=$istep lvn=$lvn pass=$pass rc=$rc" | tee -a "$LOG"
        [ "$rc" -eq 0 ] || fail=1
    done
done

echo
echo "=== VELO_TIMER medians ==="
"$PARSE_PY" "$PARSER" "$LOG" --csv "$CSV" --min-samples 1
prc=$?

[ "$prc" -eq 0 ] || fail=1
echo "ICON_VELO_LOG=$LOG"
echo "ICON_VELO_CSV=$CSV"
echo "ICON_VELO_EXIT=$fail"
exit "$fail"
