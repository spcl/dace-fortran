#!/bin/bash
#SBATCH -J cpu-sweep-cloudsc
#SBATCH -p debug
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/cpu_sweep_cloudsc-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/cpu_sweep_cloudsc-%j.out
# Figure-A row: cloudsc across the 7 problem sizes at ONE thread count, four lanes in parallel
# (dace-gcc, dace-llvm, original-openmp, c-openmp), one rank each, socket-pinned.
#
# ARM is chosen by THREADS, and the two arms are separate submissions because their rep budgets
# differ by 5x:
#   THREADS=72 REPS=50   the "CPU (72 Threads)" row      ~20 min, critical path = original-openmp
#   THREADS=1  REPS=10   the "CPU (1 Thread)" row        ~25 min, same critical path
# original-openmp is the critical path in both arms because the vendored Fortran dwarf has no
# internal rep loop: it is one process per rep, and each process re-expands the deck (~60 s at
# NGPTOT=262144).  c-openmp and the DaCe lanes all rep inside one process and finish far earlier.
# If the 1-thread arm overruns the debug cap, drop REPS to 5 or move that arm to `normal`
# (-p normal -t 02:00:00) -- and say so in the run notes, because the rep count is then not
# comparable with the 72-thread arm's.
#   sbatch --export=ALL,TAG=<tag>,THREADS=72,REPS=50 samples/tagcycle/cpu_sweep_cloudsc.sbatch
#   sbatch --export=ALL,TAG=<tag>,THREADS=1,REPS=10  samples/tagcycle/cpu_sweep_cloudsc.sbatch
set -uo pipefail

MEAS=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas

# The clone is frozen here: a phase A cache miss must die, not rebuild from a compute node.
export DACE_FORTRAN_NO_REBUILD=1
[ -w "$MEAS/dace_fortran" ] && echo "WARN: $MEAS is WRITABLE -- cpu_warm.sbatch should have refrozen it" >&2

export STAGE=MEAS LANE_MODE=meas
export THREADS="${THREADS:-72}"
export REPS="${REPS:-50}"
export WARMUP="${WARMUP:-2}"
export CLOUDSC_SWEEP_SIZES="${CLOUDSC_SWEEP_SIZES:-4096 8192 16384 32768 65536 131072 262144}"
# One lane per rank: the DaCe ranks are split by backend so neither waits on the other.
export VARIANTS="${VARIANTS:-cloudsc-sweep%CLOUDSC_SWEEP_BACKENDS=gcc cloudsc-sweep%CLOUDSC_SWEEP_BACKENDS=llvm cloudsc-sweep-fortran cloudsc-sweep-c}"

echo "CPU_SWEEP_CLOUDSC threads=$THREADS reps=$REPS sizes=$CLOUDSC_SWEEP_SIZES"
# shellcheck disable=SC1091
. "$MEAS/samples/tagcycle/cpu_job_common.sh"
exit "$cpu_overall"
