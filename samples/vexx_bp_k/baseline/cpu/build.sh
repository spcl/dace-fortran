#!/bin/bash
# Build the standalone CPU (OpenMP) baseline for QE's exx_bp::vexx_bp_k.
#
# Everything is local to this directory: ONE translation unit
# (vexx_bp_k_baseline_cpu_omp.f90 = the byte-faithful OMP-preserving kernel
# closure from QE develop/qe-omp + the FFTW3 cached-plan shim + QE utility
# stubs; the standalone pieces are archived in
# f2dace-qe-source/out_regex_omp/) and the verify/bench driver.
#
# Requirements: an MPI Fortran wrapper (mpif90/OpenMPI -- the TU is the
# __MPI-resolved export; the binary runs as an MPI *singleton*, no mpirun
# needed), libfftw3 + libfftw3_omp, LAPACK/BLAS.  See the toolchain block
# below for the module set this system needs.
#
# Link-time closure: BLAS/LAPACK/FFTW/MPI come from libraries; the tail of
# unreachable QE utilities gets generated stubs -- no-op for the mp_base
# reduction/bcast workers (correct at 1 rank), abort-if-called for the rest,
# so a silently-wrong path can never pass verification.
set -e
cd "$(dirname "$0")"

TU=vexx_bp_k_baseline_cpu_omp.f90

# All build intermediates land here: .mod files, .o files, the generated stub
# sources, and the link-loop scratch (link.err, undef_*.txt).  Only the final
# ./verify_vexx binary is left in the source directory.  `rm -rf lib` is a
# clean build -- worth doing after switching compiler flavour, since a stale
# .mod from the other compiler is unreadable and the error is opaque.
BUILD_DIR=${BUILD_DIR:-lib}
mkdir -p "$BUILD_DIR"

# ---------------------------------------------------------------- toolchain --
# All-GCC stack for this system (PBS cluster with a Cray PE underneath).
# Source setup_env.sh first -- it loads the module set below and pins the
# nested-library thread counts:
#
#   . ./setup_env.sh
#   ./build.sh
#
# Equivalently, by hand:
#
#   module load openmpi/4.1.7-gcc11   # mpif90 -> gfortran 11.2; sets NSCC_OPENMPI_DIR
#   module load fftw/3.3.10-gcc11     # fftw3.f03 + libfftw3{,_omp}; sets NSCC_FFTW_DIR
#   module load openblas/0.3.23       # BLAS *and* LAPACK in one .so; sets NSCC_OPENBLAS_DIR
#
# Why all-GCC rather than nvhpc/nvfortran: every component above is gcc-11
# built, so the binary carries exactly one OpenMP runtime (libgomp) and one
# ABI.  nvfortran's -mp would load NVIDIA's OpenMP runtime *alongside* the
# libgomp already inside libfftw3_omp and libopenblas; two runtimes in one
# process each assume they own the cores, which oversubscribes them and
# destabilises precisely the timings this baseline exists to measure.
# (The nvfortran flags are kept below for the eventual GPU port.)
#
# Deliberately NOT ${FC:-mpif90}: the nvhpc modulefiles export
# FC=<prefix>/bin/nvfortran, which would silently win over the mpif90 default
# and then fail on `USE mpi` -- mpi.mod is compiler-private, and a plain
# compiler has no MPI include path.  Override with MPIFC=... if needed.
FC=${MPIFC:-mpif90}
# Plain gcc for the C stubs, not Cray PE's `cc` -- that wrapper is on PATH once
# PrgEnv-gnu loads (openmpi/4.1.7-gcc11 pulls it in) and injects cray-mpich.
CC_STUB=${CC_STUB:-gcc}

# gfortran flavour (ACTIVE).  All three relaxations are gfortran-only spellings:
# -ffree-line-length-none lifts the 132-column free-form truncation,
# -fallow-argument-mismatch demotes gfortran 10+'s mismatched-dummy-arg error
# (QE's mp_base/MPI idiom), -std=legacy re-permits the F77-era constructs.
FF="${FFLAGS:--O3} -fopenmp -ffree-line-length-none -fallow-argument-mismatch -std=legacy"
MODFLAG="-J$BUILD_DIR"
#
# nvfortran flavour (INACTIVE -- swap the two blocks to use it).  OpenMP is -mp,
# the module-output flag is "-module <dir>" and none of the three relaxations
# exist as switches: nvfortran has no free-form line limit and only warns on
# argument mismatch and legacy constructs, so they are simply dropped.
#FF="${FFLAGS:--O3} -mp"
#MODFLAG="-module $BUILD_DIR"

# Both -J (gfortran) and -module (nvfortran) add their directory to the module
# *search* path as well as setting the output path, but state it explicitly:
# every Fortran compile below needs to find the .mod files the TU wrote, and
# they are no longer in the current directory.
MODINC="-I$BUILD_DIR"

# Module-set prefixes, with install paths as fallback so the script still works
# if the environment was set up by hand rather than via `module load`.
FFTW_DIR=${FFTW_DIR:-${NSCC_FFTW_DIR:-/app/libs/fftw/3.3.10-gcc11}}
BLAS_DIR=${BLAS_DIR:-${NSCC_OPENBLAS_DIR:-/app/libs/openblas/0.3.23}}
# fftw3.f03 is a plain ISO_C_BINDING INCLUDE file, not a .mod, so it is both
# compiler-agnostic (a gcc-built FFTW is fine) and resolved via -I only --
# the module's CPATH is not searched for Fortran INCLUDE statements.
FFTW_INC=${FFTW_INC:--I$FFTW_DIR/include}
# -lopenblas supplies BLAS *and* LAPACK (dgetrf_/dsyev_/zheev_ all live in the
# .so), so the netlib-style "-llapack -lblas" pair is not needed -- and there
# is no netlib BLAS/LAPACK in the default library path on this system anyway.
LIBS="${LIBS:--L$FFTW_DIR/lib -lfftw3_omp -lfftw3 -L$BLAS_DIR/lib -lopenblas}"

# Fail fast with an actionable message rather than deep inside the compile.
command -v "$FC" >/dev/null 2>&1 || {
  echo "ERROR: '$FC' not on PATH -- module load openmpi/4.1.7-gcc11" >&2; exit 1; }
[ -f "$FFTW_DIR/include/fftw3.f03" ] || {
  echo "ERROR: no fftw3.f03 under $FFTW_DIR/include -- module load fftw/3.3.10-gcc11" >&2; exit 1; }
[ -d "$BLAS_DIR/lib" ] || {
  echo "ERROR: no BLAS at $BLAS_DIR/lib -- module load openblas/0.3.23" >&2; exit 1; }

echo "== compile TU ($FC $FF) -> $BUILD_DIR/ =="
$FC $FF $FFTW_INC $MODFLAG -c "$TU" -o "$BUILD_DIR/vexx_tu.o"
echo "== compile driver =="
$FC $FF $MODINC -c verify_vexx.f90 -o "$BUILD_DIR/verify_vexx.o"

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

OBJS="$BUILD_DIR/vexx_tu.o $BUILD_DIR/stubs_auto.o $BUILD_DIR/stubs_auto_c.o $BUILD_DIR/verify_vexx.o"
LINK_ERR=$BUILD_DIR/link.err
for attempt in 1 2 3 4 5; do
  if $FC $FF -o verify_vexx $OBJS $LIBS 2> "$LINK_ERR"; then
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
[ -x verify_vexx ] || { echo BUILD FAILED; tail -20 "$LINK_ERR"; exit 1; }
echo "build complete: ./verify_vexx <dumpdir> <slot> [full|nc] [reps] [warmup]"
echo "  intermediates (.mod/.o/stubs/link scratch) are in $BUILD_DIR/ -- 'rm -rf $BUILD_DIR' to clean"
echo
echo "Run it on a compute node (qsub), not the login node, and pin the nested"
echo "libraries to 1 thread so only the kernel's own OMP region is parallel:"
echo "  export OMP_NUM_THREADS=<ncpus>  OPENBLAS_NUM_THREADS=1"
