#!/bin/bash
#SBATCH -J cpu-velocity-warm
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 01:00:00
#SBATCH -o samples/_work/logs/cpu_velocity_warm-%j.out
#SBATCH -e samples/_work/logs/cpu_velocity_warm-%j.out
# Phase A + deck staging for cpu_velocity_r02b06.sbatch.  SUBMIT FROM THE REPO ROOT (slurm
# resolves the -o/-e paths above against the submit dir; tagcycle_prepare.sh creates that dir).
#
# What it produces, per layout in VELOCITY_LAYOUTS:
#   velocity_r02b0{5,6}_nproma32.npz        (convert_data.py, for the DaCe lanes)
#   velocity_dump_r02b0{5,6}_nproma32/      (dump_data.py, for the Fortran drivers)
#   the gcc/flang/nvhpc driver_velocity binaries (velocity-openmp warm mode)
# ensure_npz takes an flock, so the two DaCe ranks never both convert the same deck.
#
# PREREQUISITE: the raw R02B06 ICON dump must already be at $BR2/data_r02b06 -- stage it with
# samples/velocity_tendencies/prepare_r02b06.sh before submitting.  This job does NOT generate
# it: that is a multi-hour, tens-of-GB job of its own and does not belong inside a warm step.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/cpu_velocity_warm.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$REPO/samples/_work}"
BR2="${BR2:-$WORK_ROOT/meas}"
export REPO WORK_ROOT BR2

[ -d "${VELOCITY_DATA_DIR:-$BR2/data_r02b06}" ] || {
    echo "ABORT: no R02B06 deck at ${VELOCITY_DATA_DIR:-$BR2/data_r02b06}" >&2
    echo "  stage it first: samples/velocity_tendencies/prepare_r02b06.sh" >&2
    echo "WARM_ALL_EXIT=1"
    exit 1
}
unset DACE_FORTRAN_NO_REBUILD

export STAGE=WARM LANE_MODE=warm
export THREADS="${THREADS:-72}"
export VELOCITY_LAYOUTS="${VELOCITY_LAYOUTS:-r02b05 r02b06}"
export VARIANTS="${VARIANTS:-velocity-loopexch velocity-noloopexch velocity-openmp}"

echo "CPU_VELOCITY_WARM layouts=$VELOCITY_LAYOUTS"
# shellcheck disable=SC1091
. "$REPO/samples/tagcycle/cpu_job_common.sh"
exit "$cpu_overall"
