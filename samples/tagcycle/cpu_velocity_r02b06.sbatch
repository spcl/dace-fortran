#!/bin/bash
#SBATCH -J cpu-velocity-r02b06
#SBATCH -p debug
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o samples/_work/logs/cpu_velocity_r02b06-%j.out
#SBATCH -e samples/_work/logs/cpu_velocity_r02b06-%j.out
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
# Velocity CPU panels on BOTH blocked decks: r02b05 (nproma=32, nblks_e=960) and r02b06
# (nproma=32, nblks_e=15360).  VELOCITY_LAYOUTS drives that; the `flat` nproma=491520 layout is
# a GPU decomposition and is deliberately absent, so every row here is the blocked CPU layout.
#
# THREADS=16 32 72 are the figure panels.  The full grid is affordable for the DaCe ranks but
# not for velocity-openmp, which rebuilds nothing but does pay a 6.9 GB deck load per driver
# start (2 layouts x 3 compilers x N thread points); 3 thread points keeps that at 18 starts.
# Widen with THREADS="1 2 4 8 16 32 64 72" only for a dedicated scaling run.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/cpu_velocity_r02b06.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export REPO

# NO_REBUILD is what keeps a phase A cache miss from rebuilding on a measurement node; the
# EXPECTED_TAG guard in cpu_job_common.sh is what keeps a moved-or-dirty tree from producing rows.
export DACE_FORTRAN_NO_REBUILD=1

export STAGE=MEAS LANE_MODE=meas
export THREADS="${THREADS:-16 32 72}"
export REPS="${REPS:-50}"
export WARMUP="${WARMUP:-2}"
export VELOCITY_LAYOUTS="${VELOCITY_LAYOUTS:-r02b05 r02b06}"
# Two DaCe TU variants + the original Fortran OpenMP lane (gcc/flang/nvhpc inside the variant).
export VARIANTS="${VARIANTS:-velocity-loopexch velocity-noloopexch velocity-openmp}"

echo "CPU_VELOCITY threads=$THREADS reps=$REPS layouts=$VELOCITY_LAYOUTS"
# shellcheck disable=SC1091
. "$REPO/samples/tagcycle/cpu_job_common.sh"
exit "$cpu_overall"
