#!/bin/bash
# Build the standalone CPU (OpenMP) baseline for QE's us_exx::newdxx_g.
#
# Everything is local to this directory: ONE translation unit
# (newdxx_g_baseline_cpu_omp.f90 = the OMP-preserving merged us_exx closure
# from the QE 7.6 export -- PW/src/us_exx.f90 verified identical to the
# pristine qe-omp2 tree -- + QE utility stubs; regenerate with
# f2dace-qe-source/run_regex_omp_newdxx_g.py) and the verify/bench driver.
#
# Requirements: an MPI Fortran wrapper (mpif90/OpenMPI -- the TU is the
# __MPI-resolved export; the binary runs as an MPI *singleton*, no mpirun
# needed and no MPI call is ever executed), LAPACK/BLAS.  No FFTW: unlike
# vexx_bp_k, newdxx_g is a pure G-space kernel with no FFT calls.
#
# Link-time closure: BLAS/LAPACK/MPI come from libraries; the tail of
# unreachable QE utilities gets generated stubs -- no-op for the mp_base
# reduction/bcast workers (correct at 1 rank), abort-if-called for the rest,
# so a silently-wrong path can never pass verification.
set -e
cd "$(dirname "$0")"

TU=newdxx_g_baseline_cpu_omp.f90
DRIVER=verify_newdxx.f90
BIN=verify_newdxx

# All build intermediates land here: .mod files, .o files, the generated stub
# sources, and the link-loop scratch (link.err, undef_*.txt).  Only the final
# binary is left in the source directory.  `rm -rf lib` is a clean build --
# worth doing after switching compiler flavour, since a stale .mod from the
# other compiler is unreadable and the error is opaque.
BUILD_DIR=${BUILD_DIR:-lib}
mkdir -p "$BUILD_DIR"

# ---------------------------------------------------------------- toolchain --
# Same all-GCC stack as the vexx_bp_k baseline (see its build.sh for the full
# rationale): on the cluster `. ./setup_env.sh` first; on a generic Linux box
# or the dev container the netlib fallback below suffices.
# Deliberately NOT ${FC:-mpif90}: the nvhpc modulefiles export FC=nvfortran,
# which would silently win and then fail on `USE mpi`.  Override with MPIFC=.
FC=${MPIFC:-mpif90}
CC_STUB=${CC_STUB:-gcc}

# gfortran flavour (ACTIVE).  The three relaxations are gfortran-only
# spellings (line length, QE's mp_base MPI idiom, F77-era constructs).
FF="${FFLAGS:--O3} -fopenmp -ffree-line-length-none -fallow-argument-mismatch -std=legacy"
MODFLAG="-J$BUILD_DIR"
#
# nvfortran flavour (INACTIVE -- swap the two blocks to use it).
#FF="${FFLAGS:--O3} -mp"
#MODFLAG="-module $BUILD_DIR"

MODINC="-I$BUILD_DIR"

BLAS_DIR=${BLAS_DIR:-${NSCC_OPENBLAS_DIR:-/app/libs/openblas/0.3.23}}

command -v "$FC" >/dev/null 2>&1 || {
  echo "ERROR: '$FC' not on PATH -- module load openmpi/4.1.7-gcc11" >&2; exit 1; }

# -lopenblas supplies BLAS *and* LAPACK; generic Linux falls back to netlib
# from the default library path.
if [ -d "$BLAS_DIR/lib" ]; then
  BLAS_L="-L$BLAS_DIR/lib -lopenblas"
else
  BLAS_L="-llapack -lblas"
fi
LIBS="${LIBS:-$BLAS_L}"

echo "== compile TU ($FC $FF) -> $BUILD_DIR/ =="
$FC $FF $MODFLAG -c "$TU" -o "$BUILD_DIR/kernel_tu.o"
echo "== compile driver =="
$FC $FF $MODINC -c "$DRIVER" -o "$BUILD_DIR/driver.o"

# no-op-correct-at-1-rank symbols (in-place allreduce/bcast over COMM_SELF)
NOOP_OK="reduce_base_real reduce_base_integer reduce_base_integer8 bcast_real bcast_integer bcast_integer8 bcast_logical mp_synchronize"

gen_stubs () {
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
$FC $FF $MODINC -c "$BUILD_DIR/stubs_auto.f90" -o "$BUILD_DIR/stubs_auto.o"
$CC_STUB -c "$BUILD_DIR/stubs_auto_c.c" -o "$BUILD_DIR/stubs_auto_c.o"

OBJS="$BUILD_DIR/kernel_tu.o $BUILD_DIR/stubs_auto.o $BUILD_DIR/stubs_auto_c.o $BUILD_DIR/driver.o"
LINK_ERR=$BUILD_DIR/link.err
for attempt in 1 2 3 4 5; do
  if $FC $FF -o "$BIN" $OBJS $LIBS 2> "$LINK_ERR"; then
    echo "link OK (attempt $attempt)"
    break
  fi
  grep -o "undefined reference to \`[A-Za-z0-9_]*'" "$LINK_ERR" | sed "s/.*\`//; s/'$//" | sort -u > "$BUILD_DIR/undef_all.txt"
  grep '_$'  "$BUILD_DIR/undef_all.txt" | sed 's/_$//'  > "$BUILD_DIR/undef_f.txt" || true
  grep -v '_$' "$BUILD_DIR/undef_all.txt"               > "$BUILD_DIR/undef_c.txt" || true
  n=$(wc -l < "$BUILD_DIR/undef_all.txt")
  if [ "$n" -eq 0 ]; then echo "link failed, no undefined symbols parsed:"; head "$LINK_ERR"; exit 1; fi
  echo "attempt $attempt: $n undefined ($(wc -l < "$BUILD_DIR/undef_f.txt") fortran, $(wc -l < "$BUILD_DIR/undef_c.txt") C) -> stubs"
  gen_stubs "$BUILD_DIR/undef_f.txt" "$BUILD_DIR/undef_c.txt"
  $FC $FF $MODINC -c "$BUILD_DIR/stubs_auto.f90" -o "$BUILD_DIR/stubs_auto.o"
  $CC_STUB -c "$BUILD_DIR/stubs_auto_c.c" -o "$BUILD_DIR/stubs_auto_c.o"
done
[ -x "$BIN" ] || { echo BUILD FAILED; tail -20 "$LINK_ERR"; exit 1; }
echo "build complete: ./$BIN <dumpdir> <set> [reps] [warmup]   (verifies + prints kernel_time_s)"
echo "  intermediates (.mod/.o/stubs/link scratch) are in $BUILD_DIR/ -- 'rm -rf $BUILD_DIR' to clean"
