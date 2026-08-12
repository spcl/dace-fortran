#!/bin/bash
#SBATCH -J gpu-meas-cloudsc
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
#SBATCH -o samples/_work/logs/gpu_meas_cloudsc-%j.out
#SBATCH -e samples/_work/logs/gpu_meas_cloudsc-%j.out
# SUBMIT FROM THE REPO ROOT: slurm resolves the -o/-e paths above against the submit dir.
# The cloudsc half of gpu_meas.sbatch: TWO CONCURRENT RANKS in one wave --
#   rank 0  cuda-ref   GPU 0, socket 0
#   rank 1  dace-gpu   GPU 1, socket 1
# 50 reps each, each rank socket-pinned with --cpu-bind=cores --mem-bind=local and holding half the
# node's memory. Wall time is the slower of the two lanes (~6 min), not their sum.
#
# This half plus gpu_meas_velocity.sbatch is the RECOMMENDED way to take the measurement: the two
# jobs together are two debug slots, and neither comes near the 30-minute cap.
#
# Everything below the header lives in gpu_meas.sbatch; running that file with bash is exactly the
# same job body (the #SBATCH lines in it are comments), so the two halves cannot drift apart. The
# node-level flags must be repeated here because a wrapper's own header is the one Slurm reads.
#
# This job submits nothing and resubmits nothing.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/gpu_meas_cloudsc.sbatch
set -uo pipefail

# Under sbatch, BASH_SOURCE is the slurm spool copy -- fall back to the submit dir (the repo root).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[ -d "$REPO/samples/tagcycle" ] || REPO="${SLURM_SUBMIT_DIR:-$PWD}"
export REPO LANES="${LANES:-cuda-ref dace-gpu}"
exec bash "$REPO/samples/tagcycle/gpu_meas.sbatch"
