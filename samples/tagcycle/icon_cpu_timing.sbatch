#!/bin/bash
#SBATCH -J icon-cpu-timing
#SBATCH -p debug
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 00:30:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/icon_cpu_timing-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/icon_cpu_timing-%j.out
# Held-Suarez R02B05 timing run of the DaCe-patched ICON built by
# icon_cpu_build.sbatch (same COMPILER lane).  Serial ICON (--disable-mpi),
# one srun task pinned to one socket; VELO_TIMER lines from the wrapper are
# parsed by parse_icon_timers.py into a per-config median CSV.
# The experiment mirrors run_icon_e2e.sh step 5: 3 x PT10S steps, all
# perturbation seeds pinned to 0 (pairs with the pinit_seed guard patch the
# build job applied), radiation off, output PT10S + include_last.
# No numactl anywhere -- srun --cpu-bind=cores is the only pinning allowed.
#   sbatch --export=ALL,TAG=<tag>,COMPILER=gcc samples/tagcycle/icon_cpu_timing.sbatch
set -uo pipefail

WORK=/capstor/scratch/cscs/ybudanaz/aarch64
DF="${DF:-$WORK/dace-fortran}"
ICON_SRC="${ICON_SRC:-$WORK/icon-model}"
GRID_DIR="${GRID_DIR:-$WORK/icon-grids}"
COMPILER="${COMPILER:-gcc}"
LANE_BUILD="${LANE_BUILD:-$ICON_SRC/build/$COMPILER}"
PY="${PY:-$WORK/venv-meas/bin/python}"
EXP="${EXP:-atm_heldsuarez_dace_r02b05}"
THREADS="${THREADS:-72}"
RUNS="${RUNS:-$WORK/dace-fortran-samples-meas/runs}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

TAG_NOW="$(git -C "$DF" describe --always --dirty)"
echo "ICON_TIMING host=$(hostname) lane=$COMPILER tag=$TAG_NOW expected=$TAG job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $TAG" >&2
    echo "ICON_TIMING_EXIT[$COMPILER]=1"
    exit 1
}
# The lane tree holds both binaries; time the DaCe-linked one, and make it the
# bin/icon the generated runscript execs.
[ -x "$LANE_BUILD/bin/icon.dace" ] || {
    echo "ABORT: no DaCe icon binary at $LANE_BUILD/bin/icon.dace -- run icon_cpu_build.sbatch first" >&2
    echo "ICON_TIMING_EXIT[$COMPILER]=1"
    exit 1
}
cp "$LANE_BUILD/bin/icon.dace" "$LANE_BUILD/bin/icon"

# shellcheck disable=SC1091
source "$WORK/spack/share/spack/setup-env.sh"
spack load "gcc@16.1.0 arch=linux-sles15-neoverse_v2"
mkdir -p "$RUNS"

RUN="$ICON_SRC/run"
EXP_FILE="$RUN/exp.$EXP"
cp "$RUN/exp.atm_heldsuarez" "$EXP_FILE"
sed -i \
    -e "s|^grid_id=.*|grid_id=0014|" \
    -e 's|^grid_refinement=.*|grid_refinement=R02B05|' \
    -e "s|^icon_data_poolFolder=.*|icon_data_poolFolder=\"$GRID_DIR\"|" \
    -e 's|0011-01-01T00:00:00Z|0000-01-01T00:00:30Z|' \
    -e 's|modelTimeStep *= *"PT10M"|modelTimeStep    = "PT10S"|' \
    -e 's|inwp_radiation *= *[0-9]\{1,\}|inwp_radiation = 0|g' \
    -e 's|\([A-Za-z_]*seed\) *= *-\{0,1\}[0-9]\{1,\}|\1 = 0|g' \
    -e 's|output_interval="P1D"|output_interval="PT10S"|g' \
    -e 's|include_last *= *\.FALSE\.|include_last = .TRUE.|g' \
    -e 's|num_io_procs *= *1|num_io_procs = 0|g' \
    "$EXP_FILE"

ln -sfn "$LANE_BUILD/run/set-up.info" "$RUN/set-up.info"
for entry in "$RUN"/*; do
    base=$(basename "$entry")
    [ -e "$LANE_BUILD/run/$base" ] || ln -sn "$entry" "$LANE_BUILD/run/$base"
done
( cd "$ICON_SRC" && ./make_runscripts "$EXP" ) || {
    echo "ICON_TIMING_EXIT[$COMPILER]=1"
    exit 1
}
ln -sfn "$RUN/exp.$EXP.run" "$LANE_BUILD/run/exp.$EXP.run"
rm -rf "$LANE_BUILD/experiments/$EXP"

LOG="$RUNS/icon_timing_${COMPILER}_${TAG}_${SLURM_JOB_ID:-local}.log"
CSV="$RUNS/icon_timing_${COMPILER}_${TAG}_${SLURM_JOB_ID:-local}.csv"

# Same allocator as every other measurement lane (mimalloc by default,
# MEAS_ALLOC=system escape); the srun step inherits LD_PRELOAD, so ICON
# itself runs under it.
if [ -f "$DF/samples/alloc_pool.sh" ]; then
    # shellcheck disable=SC1091
    . "$DF/samples/alloc_pool.sh"
else
    echo "WARN: no samples/alloc_pool.sh -- ICON runs on system malloc" >&2
    MEAS_ALLOC_ACTIVE=system
fi
export MEAS_ALLOC="${MEAS_ALLOC:-system}" MEAS_ALLOC_ACTIVE
echo "alloc: $MEAS_ALLOC_ACTIVE (MEAS_ALLOC=$MEAS_ALLOC, so=${MEAS_ALLOC_SO:-none})"

# One serial task on one socket; the runscript execs bin/icon directly (no
# mpirun exists in a --disable-mpi build).  srun only places/pins the step.
export OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores
srun --exact --ntasks=1 --cpus-per-task="$THREADS" --cpu-bind=cores --mem-bind=local \
    bash -c "cd '$LANE_BUILD/run' && bash 'exp.$EXP.run'" > "$LOG" 2>&1
run_rc=$?
echo "ICON_RUN_EXIT[$COMPILER]=$run_rc log=$LOG"
[ $run_rc -eq 0 ] || tail -40 "$LOG"

ntimer=$(grep -c 'VELO_TIMER' "$LOG" || true)
echo "VELO_TIMER_LINES=$ntimer"
parse_rc=1
if [ "$ntimer" -gt 0 ]; then
    grep 'VELO_TIMER' "$LOG" | "$PY" "$DF/samples/velocity_tendencies/parse_icon_timers.py" - --csv "$CSV"
    parse_rc=$?
    if [ $parse_rc -eq 0 ]; then
        awk -v a="$MEAS_ALLOC_ACTIVE" 'NR == 1 { print $0 ",alloc"; next } { print $0 "," a }' \
            "$CSV" > "$CSV.tmp" && mv "$CSV.tmp" "$CSV"
    fi
    echo "TIMER_CSV=$CSV"
else
    echo "ERROR: no VELO_TIMER lines -- was the DaCe wrapper actually linked/called?" >&2
fi

overall=0
{ [ $run_rc -eq 0 ] && [ $parse_rc -eq 0 ]; } || overall=1
echo "ICON_TIMING_EXIT[$COMPILER]=$overall"
echo "ICON_TIMING_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
