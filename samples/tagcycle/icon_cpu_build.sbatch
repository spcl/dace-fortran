#!/bin/bash
#SBATCH -J icon-cpu-build
#SBATCH -p normal
#SBATCH -A g34
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH -t 04:00:00
#SBATCH -o samples/_work/logs/icon_cpu_build-%j.out
#SBATCH -e samples/_work/logs/icon_cpu_build-%j.out
# ICON CPU integration build, one compiler lane per submission: stock ICON,
# then the DaCe velocity lib (--release, FMA on), then the velocity-patched
# ICON linking that lib.  No MPI anywhere -- ICON is configured --disable-mpi.
# Builds run PLAIN on the allocated node: srun must never wrap configure/make
# (blocked SIGCHLD deadlocks the conftest select() loop), and nothing here is
# timed so plain -j is the right shape.
# SUBMIT FROM THE REPO ROOT: the -o/-e paths above and $DF below both resolve
# through the submit dir (slurm copies this script to a spool dir, so its own
# location tells us nothing).  Create samples/_work/logs/ first.
#   mkdir -p samples/_work/logs
#   sbatch --export=ALL,TAG=<tag>,COMPILER=nvhpc samples/tagcycle/icon_cpu_build.sbatch
set -uo pipefail

DF="${DF:-${SLURM_SUBMIT_DIR:-$PWD}}"
WORK_ROOT="${WORK_ROOT:-$DF/samples/_work}"
SITE_ROOT="$(cd "$DF/.." && pwd)"
ICON_SRC="${ICON_SRC:-$WORK_ROOT/icon-model}"
GRID_DIR="${GRID_DIR:-$WORK_ROOT/icon-grids}"
COMPILER="${COMPILER:-nvhpc}"
DACE_LIBS="${DACE_LIBS:-$WORK_ROOT/dace-icon-libs-$COMPILER}"
# ONE build tree per (compiler, GPU) lane -- this is the CPU job, so
# build/cpu_$COMPILER; the stock and the DaCe-linked binaries are kept side by
# side inside it as bin/icon.stock and bin/icon.dace, since only the icon.mk
# link rule differs between them.
LANE_BUILD="${LANE_BUILD:-$ICON_SRC/build/cpu_$COMPILER}"
CONF_SH="$DF/scripts/configure_icon_${COMPILER}_daint.sh"
PY="${PY:-$SITE_ROOT/venv-meas/bin/python}"
MAKE_J="${MAKE_J:-64}"
ICON_DACE_OPT="${ICON_DACE_OPT:-O3}"
: "${TAG:?TAG must be exported into this job (--export=ALL,TAG=<tag>)}"

TAG_NOW="$(git -C "$DF" describe --always --dirty)"
echo "ICON_BUILD host=$(hostname) lane=$COMPILER tag=$TAG_NOW expected=$TAG job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ)"
[ "$TAG_NOW" = "$TAG" ] || {
    echo "ABORT: clone tag is $TAG_NOW, expected $TAG" >&2
    echo "ICON_BUILD_EXIT[$COMPILER]=1"
    exit 1
}

# shellcheck disable=SC1091
source "${SPACK_ROOT:-$SITE_ROOT/spack}/share/spack/setup-env.sh"
spack load "gcc@16.1.0 arch=linux-sles15-neoverse_v2"
# The DaCe lib build needs LLVM flang on PATH next to gfortran (gcc lane only).
[ "$COMPILER" = gcc ] && spack load llvm@22.1.5

# shellcheck disable=SC1091
source "$DF/scripts/icon_daint_common.sh"
( icon_dir="$ICON_SRC" ICON_DAINT_LANE=icon_cpu_build icon_daint_print_provenance )

overall=0
fail() { echo "$1"; overall=1; }

# Grid presence only -- compute nodes have no internet; fetch on a login node.
GRID_FILE="$GRID_DIR/0014/icon_grid_0014_R02B05_G.nc"
[ -f "$GRID_FILE" ] || { fail "ICON_BUILD_EXIT[$COMPILER]=1  # grid missing: $GRID_FILE"; exit 1; }

# The ICON source patches (pinit_seed guard, NOMPI acc device management, acc
# reduction syntax) are applied by the lane configure script itself.
VELOCITY_F90="$ICON_SRC/src/atm_dyn_iconam/mo_velocity_advection.f90"
[ -f "$VELOCITY_F90.bak" ] || cp "$VELOCITY_F90" "$VELOCITY_F90.bak"
cp "$VELOCITY_F90.bak" "$VELOCITY_F90"

build_icon() { # $1=DACE_LIBS_DIR ('' for stock)  $2=name to keep the binary under
    ( ICON_DACE_OPT="$ICON_DACE_OPT" DACE_LIBS_DIR="$1" \
        BUILD_DIR="$LANE_BUILD" ICON_SRC="$ICON_SRC" bash "$CONF_SH" ) > "$LANE_BUILD.configure.log" 2>&1 &&
        # shellcheck disable=SC1091
        { source "$LANE_BUILD/build-env.sh" &&
          make -C "$LANE_BUILD" -j"$MAKE_J" > "$LANE_BUILD/make.log" 2>&1; } &&
        cp "$LANE_BUILD/bin/icon" "$LANE_BUILD/bin/$2"
}

echo "=== stock ICON ($COMPILER, $ICON_DACE_OPT) -> $LANE_BUILD (bin/icon.stock)"
rm -rf "$LANE_BUILD"
build_icon "" icon.stock
rc=$?
[ $rc -eq 0 ] || { tail -40 "$LANE_BUILD/make.log" 2>/dev/null; overall=1; }
echo "ICON_STOCK_BUILD_EXIT[$COMPILER]=$rc"

if [ "$COMPILER" = gcc ] && [ $rc -eq 0 ]; then
    echo "=== DaCe velocity lib (--release, fp-contract=fast) -> $DACE_LIBS"
    rel=--release
    [ "$ICON_DACE_OPT" = O0 ] && rel=
    DACE_FORTRAN_FP_CONTRACT=fast "$PY" "$DF/scripts/build_icon_dace_libs.py" \
        $rel --icon-src "$ICON_SRC" --icon-build "$LANE_BUILD" --out-dir "$DACE_LIBS"
    lib_rc=$?
    echo "DACE_LIBS_EXIT[$COMPILER]=$lib_rc"
    [ $lib_rc -eq 0 ] || overall=1

    if [ $lib_rc -eq 0 ]; then
        # Same tree, DaCe link rule + velocity dispatch patch: make relinks and
        # recompiles only what the patch touched.
        echo "=== patched ICON ($COMPILER) -> $LANE_BUILD (bin/icon.dace)"
        icon_dir="$ICON_SRC" ICON_DAINT_LANE=icon_cpu_build \
            icon_daint_apply_one_patch "$ICON_DAINT_DACE_PATCH"
        build_icon "$DACE_LIBS" icon.dace
        drc=$?
        [ $drc -eq 0 ] || tail -40 "$LANE_BUILD/make.log" 2>/dev/null
        echo "ICON_DACE_BUILD_EXIT[$COMPILER]=$drc"
        [ $drc -eq 0 ] || overall=1
        cp "$VELOCITY_F90.bak" "$VELOCITY_F90"
    fi
else
    # The ONLY path now: ICON is nvhpc-only, so the gcc arm above is
    # unreachable and kept solely for when a DaCe lib exists for nvfortran.
    # build_icon_dace_libs.py hard-requires gfortran+flang, so it does not yet.
    echo "DACE_LIBS_EXIT[$COMPILER]=skip  # blocked: no nvfortran-compatible DaCe lib"
fi

echo "ICON_BUILD_EXIT[$COMPILER]=$overall"
echo "ICON_BUILD_DONE end=$(date -u +%FT%TZ)"
exit "$overall"
