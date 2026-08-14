#!/bin/bash
# Build the standalone CPU (OpenMP) baseline for the QE us_exx kernel named by this
# sample directory.  Adapted from spcl/dace-fortran PR #3 (branch vexx_bp_k,
# samples/<kernel>/baseline/cpu/build.sh) for daint: that site had an OpenMPI
# modulefile stack behind mpif90, this one has no MPI at all, so `USE mpi` is
# satisfied by dace_fortran's own OpenMPI-header stub module compiled with the lane's
# Fortran compiler.  Everything else -- one TU, the verify/bench driver, and the
# stub-generating link loop -- is the PR's.
#
#   LANE=original-openmp ./build.sh     gfortran (default)
#   LANE=flang-openmp    ./build.sh     LLVM flang
#
# env: LANE BUILD_DIR BIN_PATH FC_OVERRIDE MPIFC FFLAGS OMPI_INCLUDE BLAS_DIR LIBS PYTHON
set -e
cd "$(dirname "$0")"

KERNEL=$(cd ../.. && basename "$PWD")
TU="${KERNEL}_baseline_cpu_omp.f90"
DRIVER=$(basename "$(ls verify_*.f90 | head -1)")
BIN="${DRIVER%.f90}"
[ -f "$TU" ] || { echo "ERROR: no $TU beside this script" >&2; exit 1; }

LANE=${LANE:-original-openmp}
BUILD_DIR=${BUILD_DIR:-lib}
BIN_PATH=${BIN_PATH:-$PWD/$BIN}
mkdir -p "$BUILD_DIR"

# The three gfortran relaxations are gfortran-only spellings (line length, QE's mp_base
# MPI idiom, F77-era constructs); flang accepts the TU without any of them.
case "$LANE" in
    original-openmp)
        FC=${FC_OVERRIDE:-gfortran}
        FF="${FFLAGS:--O3 -march=native} -fopenmp -ffree-line-length-none -fallow-argument-mismatch -std=legacy"
        MODFLAG="-J$BUILD_DIR"
        BLAS_SPEC="openblas %gcc@16.1.0"
        ;;
    flang-openmp)
        FC=${FC_OVERRIDE:-$(command -v flang-22 || command -v flang || echo flang)}
        FF="${FFLAGS:--O3 -mcpu=native} -fopenmp"
        MODFLAG="-module-dir $BUILD_DIR"
        # OpenBLAS is threads=openmp, so it drags in ITS builder's OpenMP runtime.  A
        # gcc-built one puts libgomp beside this lane's libomp; both runtimes then spin
        # against each other and the lane gets SLOWER with every added thread (measured
        # job 4478503: 10 ms at 1 thread -> 598 ms at 72).  Match BLAS to the lane.
        BLAS_SPEC="openblas %clang@22.1.5"
        ;;
    *)
        echo "ERROR: unknown LANE '$LANE' (original-openmp | flang-openmp)" >&2
        exit 1
        ;;
esac
MODINC="-I$BUILD_DIR"
CC_STUB=${CC_STUB:-gcc}
command -v "$FC" > /dev/null 2>&1 || { echo "ERROR: '$FC' not on PATH (source samples/env.sh)" >&2; exit 1; }

# MPIFC (an MPI wrapper whose own mpi.mod is readable by this lane's compiler) short-circuits
# the stub; without it the module comes from dace_fortran's OpenMPI-header wrapper, compiled
# here so the .mod format matches $FC.  Only the module is needed: at one rank the kernel
# reaches no MPI call, and the link loop below stubs every MPI symbol as abort-if-called.
if [ -n "${MPIFC:-}" ]; then
    FC="$MPIFC"
    MPI_OBJ=""
else
    # Highest hpcx under the newest CUDA subtree wins; only the Fortran headers are read.
    if [ -z "${OMPI_INCLUDE:-}" ] && command -v nvfortran > /dev/null 2>&1; then
        NV_COMPILERS=$(dirname "$(dirname "$(command -v nvfortran)")")
        OMPI_INCLUDE=$(ls -1d "$NV_COMPILERS"/../comm_libs/*/hpcx/hpcx-*/ompi/include 2> /dev/null | tail -1)
    fi
    if [ -z "${OMPI_INCLUDE:-}" ] && command -v spack > /dev/null 2>&1; then
        OMPI_INCLUDE=$(ls -1d "$(spack location -i nvhpc 2> /dev/null)"/Linux_*/*/comm_libs/*/hpcx/hpcx-*/ompi/include 2> /dev/null | tail -1)
    fi
    [ -n "$OMPI_INCLUDE" ] && [ -f "$OMPI_INCLUDE/mpif-config.h" ] || {
        echo "ERROR: no OpenMPI mpif-*.h headers found; set OMPI_INCLUDE=<dir with mpif-config.h>" >&2
        exit 1
    }
    "${PYTHON:-python3}" -c \
        "from dace_fortran.flang_codebase import mpi_stub_source; open('$BUILD_DIR/mpi_stub.f90','w').write(mpi_stub_source())"
    echo "== mpi.mod stub ($OMPI_INCLUDE) =="
    $FC $FF $MODFLAG -I"$OMPI_INCLUDE" -c "$BUILD_DIR/mpi_stub.f90" -o "$BUILD_DIR/mpi_stub.o"
    MPI_OBJ="$BUILD_DIR/mpi_stub.o"
fi

BLAS_DIR=${BLAS_DIR:-${OPENBLAS_DIR:-}}
if [ -z "$BLAS_DIR" ] && command -v spack > /dev/null 2>&1; then
    BLAS_DIR=$(spack location -i $BLAS_SPEC 2> /dev/null || true)
fi
if [ -n "$BLAS_DIR" ] && [ -d "$BLAS_DIR/lib" ]; then
    # DT_RPATH, not DT_RUNPATH: samples/env.sh `spack load`s the gcc OpenBLAS, so its lib dir
    # is on LD_LIBRARY_PATH -- which OVERRIDES RUNPATH and would drag libgomp back into the
    # flang lane no matter what this link says.  RPATH is the one search path it cannot beat.
    BLAS_L="-L$BLAS_DIR/lib -Wl,--disable-new-dtags,-rpath,$BLAS_DIR/lib -lopenblas"
else
    BLAS_L="-llapack -lblas"
fi
LIBS="${LIBS:-$BLAS_L}"

echo "== compile TU ($LANE: $FC $FF) -> $BUILD_DIR/ =="
$FC $FF $MODFLAG -c "$TU" -o "$BUILD_DIR/kernel_tu.o"
echo "== compile driver =="
$FC $FF $MODFLAG $MODINC -c "$DRIVER" -o "$BUILD_DIR/driver.o"

# Correct as no-ops at one rank (in-place allreduce/bcast over COMM_SELF); everything else
# aborts if entered, so a silently-wrong path can never pass verification.
NOOP_OK="reduce_base_real reduce_base_integer reduce_base_integer8 bcast_real bcast_integer bcast_integer8 bcast_logical mp_synchronize"

gen_stubs() {
    {
        echo "! auto-generated link stubs (see build.sh)"
        while read -r s; do
            [ -z "$s" ] && continue
            if echo " $NOOP_OK " | grep -q " $s "; then
                echo "subroutine $s()"
                echo "end subroutine $s"
            else
                echo "subroutine $s()"
                echo "  write(*,*) 'FATAL: unreachable stub called: $s'"
                echo "  error stop 9"
                echo "end subroutine $s"
            fi
        done < "$1"
    } > "$BUILD_DIR/stubs_auto.f90"
    {
        echo "/* auto-generated C link stubs (unreachable clib wrappers) */"
        echo "#include <stdio.h>"
        echo "#include <stdlib.h>"
        while read -r s; do
            [ -z "$s" ] && continue
            echo "void $s() { fprintf(stderr, \"FATAL: unreachable C stub called: $s\\n\"); abort(); }"
        done < "$2"
    } > "$BUILD_DIR/stubs_auto_c.c"
}

echo "== link (auto-stub loop) =="
echo "! empty" > "$BUILD_DIR/stubs_auto.f90"
echo "/* empty */" > "$BUILD_DIR/stubs_auto_c.c"
$FC $FF $MODFLAG -c "$BUILD_DIR/stubs_auto.f90" -o "$BUILD_DIR/stubs_auto.o"
$CC_STUB -c "$BUILD_DIR/stubs_auto_c.c" -o "$BUILD_DIR/stubs_auto_c.o"

OBJS="$BUILD_DIR/kernel_tu.o $MPI_OBJ $BUILD_DIR/stubs_auto.o $BUILD_DIR/stubs_auto_c.o $BUILD_DIR/driver.o"
LINK_ERR=$BUILD_DIR/link.err
for attempt in 1 2 3 4 5; do
    if $FC $FF -o "$BIN_PATH" $OBJS $LIBS 2> "$LINK_ERR"; then
        echo "link OK (attempt $attempt)"
        break
    fi
    grep -o "undefined reference to \`[A-Za-z0-9_]*'" "$LINK_ERR" | sed "s/.*\`//; s/'$//" | sort -u > "$BUILD_DIR/undef_all.txt"
    grep '_$' "$BUILD_DIR/undef_all.txt" | sed 's/_$//' > "$BUILD_DIR/undef_f.txt" || true
    grep -v '_$' "$BUILD_DIR/undef_all.txt" > "$BUILD_DIR/undef_c.txt" || true
    n=$(wc -l < "$BUILD_DIR/undef_all.txt")
    if [ "$n" -eq 0 ]; then
        echo "link failed, no undefined symbols parsed:"
        head "$LINK_ERR"
        exit 1
    fi
    echo "attempt $attempt: $n undefined ($(wc -l < "$BUILD_DIR/undef_f.txt") fortran, $(wc -l < "$BUILD_DIR/undef_c.txt") C) -> stubs"
    gen_stubs "$BUILD_DIR/undef_f.txt" "$BUILD_DIR/undef_c.txt"
    $FC $FF $MODFLAG -c "$BUILD_DIR/stubs_auto.f90" -o "$BUILD_DIR/stubs_auto.o"
    $CC_STUB -c "$BUILD_DIR/stubs_auto_c.c" -o "$BUILD_DIR/stubs_auto_c.o"
done
[ -x "$BIN_PATH" ] || { echo BUILD FAILED; tail -20 "$LINK_ERR"; exit 1; }
echo "build complete: $BIN_PATH <dumpdir> <set> [reps] [warmup]   (verifies + prints kernel_time_s)"
