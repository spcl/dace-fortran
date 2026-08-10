#!/bin/bash
#SBATCH -J tagcycle-bridge
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH -t 01:00:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/aarch64/probe/tagcycle_bridge-%j.out
#SBATCH -e /capstor/scratch/cscs/ybudanaz/aarch64/probe/tagcycle_bridge-%j.out
# Step 2 of the tag cycle: rebuild the HLFIR bridge (and the velocity_data nanobind module) in the
# meas clone at $TAG.  Needs a clone left WRITABLE by tagcycle_prepare.sh.
#
# NOTHING HERE MAY BE WRAPPED IN srun.  srun starts its tasks with SIGCHLD blocked, cmake reaps its
# configure helpers through SIGCHLD, and both of these cmake calls run straight out of subprocess
# with no unblock shim (dace/codegen/compiler.py has one; build_bridge.py does not) -- so a step
# would hang in select() until the walltime.
#   sbatch --export=ALL,TAG=<tag> tagcycle_bridge.sbatch
set -uo pipefail

MEAS=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-meas
BR2=/capstor/scratch/cscs/ybudanaz/aarch64/dace-fortran-samples-meas
LOG="$BR2/logs"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

mkdir -p "$LOG"
cd "$MEAS" || exit 1
ulimit -c 0
# shellcheck disable=SC1091
. samples/env.sh > /dev/null 2>&1

# Building IS the point of this job.
unset DACE_FORTRAN_NO_REBUILD

TAG_NOW="$(git -C "$MEAS" describe --always --dirty)"
echo "BRIDGE host=$(hostname) tag=$TAG_NOW expected=$TAG python=$PYTHON start=$(date -u +%FT%TZ)"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $TAG" >&2
    echo "BRIDGE_BUILD_EXIT=1"
    exit 1
}
[ -w "$MEAS/dace_fortran" ] || {
    echo "ABORT: $MEAS is still frozen -- run tagcycle_prepare.sh $TAG first" >&2
    echo "BRIDGE_BUILD_EXIT=1"
    exit 1
}

# Judged by MARKER, not by rc: the bridge and anything importing it can finish its work and still
# abort in teardown with rc=134 (heap corruption), which an rc-only check misreads as a failure.
step() { # step <name> <marker> <cmd...>
    local name="$1" marker="$2" rc
    shift 2
    "$@" > "$LOG/$name.log" 2>&1
    rc=$?
    tail -40 "$LOG/$name.log"
    if grep -Eq "$marker" "$LOG/$name.log"; then
        echo "OK $name (rc=$rc)"
        return 0
    fi
    echo "ERR $name rc=$rc -> $LOG/$name.log"
    return 1
}

step "bridge_build_$TAG" "BRIDGE BUILD MARKER OK" \
    "$PYTHON" -c "import dace_fortran.build_bridge as b; b.build(); print('BRIDGE BUILD MARKER OK', b._local_so())"
bridge_rc=$?

step "bridge_import_$TAG" "BRIDGE IMPORT MARKER OK" \
    "$PYTHON" -c "import dace_fortran; dace_fortran.SDFGBuilder; print('BRIDGE IMPORT MARKER OK')"
import_rc=$?

# Same cmake-under-srun hazard, same job: the velocity lane's nanobind reader.
step "velocity_data_$TAG" "selftest OK" \
    "$PYTHON" samples/velocity_tendencies/convert_data.py --selftest
vd_rc=$?

echo "BRIDGE_BUILD_STEP_EXIT=$bridge_rc"
echo "BRIDGE_IMPORT_STEP_EXIT=$import_rc"
echo "VELOCITY_DATA_BUILD_EXIT=$vd_rc"
overall=0
[ "$bridge_rc" -eq 0 ] && [ "$import_rc" -eq 0 ] && [ "$vd_rc" -eq 0 ] || overall=1
echo "BRIDGE_BUILD_EXIT=$overall"
echo "BRIDGE_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
