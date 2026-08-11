#!/bin/bash
#SBATCH -J gpu-meas-velocity
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
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/gpu_meas_velocity-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/gpu_meas_velocity-%j.out
# The velocity half of gpu_meas.sbatch: THREE CONCURRENT RANKS in one wave, all at nproma=491520
# (nblks_e = 1) on the R02B06 deck --
#   rank 0  openacc-ref          GPU 0, socket 0
#   rank 1  dace-gpu-pipeline    GPU 1, socket 1
#   rank 2  dace-gpu-autoopt     GPU 2, socket 2
# 50 reps each, each rank socket-pinned with --cpu-bind=cores --mem-bind=local and holding a third
# of the node's memory. Wall time is the slowest single lane (~8-10 min), not their sum.
#
# THE ONE CONTENTION RISK: three processes each load the 6.9 GB deck at once, off capstor. The
# drivers have no start barrier, so the launcher staggers the rank starts by RANK_STAGGER_S (15 s
# default) to spread the read burst. That delays only the load phase -- every rank times its own
# loop after its own load completes, so the measured regions are never serialized. Raise
# RANK_STAGGER_S if the logs show the loads still colliding.
#
# Everything below the header lives in gpu_meas.sbatch; running that file with bash is exactly the
# same job body (the #SBATCH lines in it are comments), so the two halves cannot drift apart. The
# node-level flags must be repeated here because a wrapper's own header is the one Slurm reads.
#
# This job submits nothing and resubmits nothing.
#   sbatch --export=ALL,TAG=<tag> gpu_meas_velocity.sbatch
set -uo pipefail

MEAS="${MEAS:-/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas}"
export LANES="${LANES:-openacc-ref dace-gpu-pipeline dace-gpu-autoopt}"
exec bash "$MEAS/samples/tagcycle/gpu_meas.sbatch"
