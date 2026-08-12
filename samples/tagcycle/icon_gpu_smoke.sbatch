#!/bin/bash
#SBATCH -J icon-gpu-smoke
#SBATCH -p debug
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=1
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/icon_gpu_smoke-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/icon_gpu_smoke-%j.out
# Smoke: configure + make + short-run the OpenACC GPU ICON (nvhpc, cc90,
# --disable-mpi) on a GH200 node.  RESUMABLE: the build dir is kept, each
# phase has its own verdict, and a rerun continues where make stopped -- a
# cold `make -j72` may not fit one 30-min debug slot.  FORCE_CLEAN=1 wipes.
# GRID=R02B04..R02B09 picks the Held-Suarez resolution (default R02B04);
# grids must be pre-staged by scripts/fetch_icon_grids.sh (no internet here).
# configure/make run PLAIN (srun-wrapped builds deadlock on SIGCHLD); only
# the model run goes through srun.  No numactl.
#   sbatch --export=ALL,TAG=<tag> samples/tagcycle/icon_gpu_smoke.sbatch
#   sbatch --export=ALL,TAG=<tag>,GRID=R02B06 samples/tagcycle/icon_gpu_smoke.sbatch
set -uo pipefail

WORK=/capstor/scratch/cscs/ybudanaz/aarch64
DF="${DF:-$WORK/dace-fortran}"
ICON_SRC="${ICON_SRC:-$WORK/icon-model}"
GRID_DIR="${GRID_DIR:-$WORK/icon-grids}"
BUILD="${BUILD:-$ICON_SRC/build/nvhpc}"
GRID="${GRID:-R02B04}"
MAKE_J="${MAKE_J:-72}"
ICON_DACE_OPT="${ICON_DACE_OPT:-O3}"
RUNS="${RUNS:-$WORK/dace-fortran-samples-meas/runs}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

TAG_NOW="$(git -C "$DF" describe --always --dirty)"
echo "ICON_GPU_SMOKE host=$(hostname) grid=$GRID tag=$TAG_NOW expected=$TAG job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $TAG" >&2
    echo "ICON_GPU_RUN_EXIT=1"
    exit 1
}

case "$GRID" in
    R02B04) GRID_ID=0013 ;;
    R02B05) GRID_ID=0019 ;;
    R02B06) GRID_ID=0021 ;;
    R02B07) GRID_ID=0023 ;;
    R02B08) GRID_ID=0025 ;;
    R02B09) GRID_ID=0015 ;;
    *) echo "ABORT: unknown GRID=$GRID" >&2; echo "ICON_GPU_RUN_EXIT=1"; exit 1 ;;
esac
GRID_FILE="$GRID_DIR/$GRID_ID/icon_grid_${GRID_ID}_${GRID}_G.nc"
[ -f "$GRID_FILE" ] || {
    echo "ABORT: grid not staged: $GRID_FILE (run scripts/fetch_icon_grids.sh on a login node)" >&2
    echo "ICON_GPU_RUN_EXIT=1"
    exit 1
}

# shellcheck disable=SC1091
source "$WORK/spack/share/spack/setup-env.sh"
spack load "gcc@16.1.0 arch=linux-sles15-neoverse_v2"

# --- phase 1: configure (reused only when the stamp the configure script wrote
# still matches: an edited lane script, helper or ICON patch invalidates
# icon.mk, whose sub-configure recipes bake in the compiler paths and flags)
# shellcheck disable=SC1091
source "$DF/scripts/icon_daint_common.sh"
( icon_dir="$ICON_SRC" ICON_DAINT_LANE=icon_gpu_smoke icon_daint_print_provenance )
CONF_ID="$(icon_daint_stamp_id "GPU=1 OPT=$ICON_DACE_OPT DACE=" configure_icon_nvhpc_daint.sh)"
[ "${FORCE_CLEAN:-0}" = 1 ] && rm -rf "$BUILD"
mkdir -p "$BUILD"
if [ -f "$BUILD/icon.mk" ] && [ "$(cat "$BUILD/.configure.stamp" 2>/dev/null)" = "$CONF_ID" ]; then
    echo "(configure reused from previous run)"
    echo "ICON_GPU_CONFIGURE_EXIT=0"
else
    [ -f "$BUILD/icon.mk" ] && echo "(configure inputs changed -- reconfiguring in place)"
    ( GPU=1 ICON_DACE_OPT="$ICON_DACE_OPT" DACE_LIBS_DIR= BUILD_DIR="$BUILD" ICON_SRC="$ICON_SRC" \
        bash "$DF/scripts/configure_icon_nvhpc_daint.sh" ) > "$BUILD/configure.log" 2>&1
    rc=$?
    echo "ICON_GPU_CONFIGURE_EXIT=$rc"
    [ $rc -eq 0 ] || { tail -30 "$BUILD/configure.log"; exit 1; }
fi

# --- phase 2: build (make resumes; -Minfo=accel lands in make.log)
# The delayed sub-configures and cmake externals inherit THIS shell's env;
# build-env.sh is what the configure phase actually used (nvhpc bin on PATH,
# CUDA redirects scrubbed).
# shellcheck disable=SC1091
[ -f "$BUILD/build-env.sh" ] && source "$BUILD/build-env.sh"
make -C "$BUILD" -j"$MAKE_J" >> "$BUILD/make.log" 2>&1
rc=$?
echo "ICON_GPU_BUILD_EXIT=$rc"
if [ $rc -ne 0 ] || [ ! -x "$BUILD/bin/icon" ]; then
    tail -40 "$BUILD/make.log"
    echo "(build incomplete -- resubmit to continue from here)"
    exit 1
fi
echo "MINFO_ACCEL_LINES=$(grep -c 'Generating' "$BUILD/make.log" || true)"

# --- phase 3: short Held-Suarez run, 1 task + GPU 0
EXP="atm_heldsuarez_gpu_${GRID}"
RUN="$ICON_SRC/run"
cp "$RUN/exp.atm_heldsuarez" "$RUN/exp.$EXP"
sed -i \
    -e "s|^grid_id=.*|grid_id=$GRID_ID|" \
    -e "s|^grid_refinement=.*|grid_refinement=$GRID|" \
    -e "s|^icon_data_poolFolder=.*|icon_data_poolFolder=\"$GRID_DIR\"|" \
    -e 's|0011-01-01T00:00:00Z|0000-01-01T00:00:30Z|' \
    -e 's|modelTimeStep *= *"PT10M"|modelTimeStep    = "PT10S"|' \
    -e 's|inwp_radiation *= *[0-9]\{1,\}|inwp_radiation = 0|g' \
    -e 's|\([A-Za-z_]*seed\) *= *-\{0,1\}[0-9]\{1,\}|\1 = 0|g' \
    -e 's|output_interval="P1D"|output_interval="PT10S"|g' \
    -e 's|include_last *= *\.FALSE\.|include_last = .TRUE.|g' \
    -e 's|num_io_procs *= *1|num_io_procs = 0|g' \
    "$RUN/exp.$EXP"

ln -sfn "$BUILD/run/set-up.info" "$RUN/set-up.info"
for entry in "$RUN"/*; do
    base=$(basename "$entry")
    [ -e "$BUILD/run/$base" ] || ln -sn "$entry" "$BUILD/run/$base"
done
( cd "$ICON_SRC" && ./make_runscripts "$EXP" ) || { echo "ICON_GPU_RUN_EXIT=1"; exit 1; }
ln -sfn "$RUN/exp.$EXP.run" "$BUILD/run/exp.$EXP.run"
rm -rf "$BUILD/experiments/$EXP"

mkdir -p "$RUNS"
LOG="$RUNS/icon_gpu_smoke_${GRID}_${TAG}_${SLURM_JOB_ID:-local}.log"
export CUDA_VISIBLE_DEVICES=0
# 20-min cap on the model run (data init is slow, that much wall is expected);
# success is judged from stdout step progress, a timeout with steps is a pass.
RUN_CAP="${RUN_CAP:-1200}"
timeout -k 60 "$RUN_CAP" srun --exact --ntasks=1 --cpus-per-task=72 --cpu-bind=cores --mem-bind=local \
    bash -c "cd '$BUILD/run' && bash 'exp.$EXP.run'" > "$LOG" 2>&1
rc=$?
[ $rc -eq 124 ] && echo "ICON_GPU_RUN_TIMEOUT=1 cap=${RUN_CAP}s"
[ $rc -eq 0 ] || tail -40 "$LOG"
nsteps=$(grep -cE 'Time step:|TIME STEP' "$LOG" || true)
if [ $rc -eq 124 ] && [ "$nsteps" -gt 0 ]; then
    rc=0
fi
echo "ICON_GPU_RUN_STEPS=$nsteps log=$LOG"
echo "ICON_GPU_RUN_EXIT=$rc"
echo "ICON_GPU_SMOKE_DONE grid=$GRID end=$(date -u +%FT%TZ)"
exit "$rc"
