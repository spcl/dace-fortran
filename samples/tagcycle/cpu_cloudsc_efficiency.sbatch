#!/bin/bash
#SBATCH -J cpu-cloudsc-eff
#SBATCH -p debug
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o samples/_work/logs/cpu_cloudsc_eff-%j.out
#SBATCH -e samples/_work/logs/cpu_cloudsc_eff-%j.out
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
# Efficiency panel: the FULL thread grid at the canonical 65536 columns, for the two non-DaCe
# lanes only.  The DaCe half of this panel is already produced by meas_4rank.sbatch, whose
# cloudsc-nblks rank is exactly (klon=32, nblocks=2048, THREADS=1..72) -- running it again here
# would duplicate rows under a second job id, so it is deliberately not in VARIANTS.
#
# REPS=10, not 50: original-openmp is one dwarf process per rep, so 8 thread points x 50 reps
# would be ~400 process starts on top of the deck expansion.  10 reps keeps the rank near 10 min
# and is still enough for a median at these runtimes.  c-openmp reps inside one process and
# could afford 50; it is held at the same count so the two baselines stay comparable.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/cpu_cloudsc_efficiency.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export REPO

export DACE_FORTRAN_NO_REBUILD=1
export STAGE=MEAS LANE_MODE=meas
export THREADS="${THREADS:-1 2 4 8 16 32 64 72}"
export REPS="${REPS:-10}"
export WARMUP="${WARMUP:-2}"
export CLOUDSC_SWEEP_SIZES="${CLOUDSC_SWEEP_SIZES:-65536}"
export VARIANTS="${VARIANTS:-cloudsc-sweep-fortran cloudsc-sweep-c}"

echo "CPU_CLOUDSC_EFF threads=$THREADS reps=$REPS size=$CLOUDSC_SWEEP_SIZES"
# shellcheck disable=SC1091
. "$REPO/samples/tagcycle/cpu_job_common.sh"
exit "$cpu_overall"
