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
# needed), libfftw3 + libfftw3_omp, LAPACK/BLAS.
#
# Link-time closure: BLAS/LAPACK/FFTW/MPI come from libraries; the tail of
# unreachable QE utilities gets generated stubs -- no-op for the mp_base
# reduction/bcast workers (correct at 1 rank), abort-if-called for the rest,
# so a silently-wrong path can never pass verification.
set -e
cd "$(dirname "$0")"

TU=vexx_bp_k_baseline_cpu_omp.f90
FC=${FC:-mpif90}
FF="${FFLAGS:--O3} -fopenmp -ffree-line-length-none -fallow-argument-mismatch -std=legacy"
FFTW_INC=${FFTW_INC:--I/usr/include}   # fftw3.f03, included by the in-TU shim
LIBS="${LIBS:--lfftw3_omp -lfftw3 -llapack -lblas}"

echo "== compile TU ($FC $FF) =="
$FC $FF $FFTW_INC -c "$TU" -J. -o vexx_tu.o
echo "== compile driver =="
$FC $FF -c verify_vexx.f90 -o verify_vexx.o

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
  } > stubs_auto.f90
  {
    echo "/* auto-generated C link stubs (unreachable clib wrappers) */"
    echo "#include <stdio.h>"
    echo "#include <stdlib.h>"
    while read -r s; do
      [ -z "$s" ] && continue
      echo "void $s() { fprintf(stderr, \"FATAL: unreachable C stub called: $s\\n\"); abort(); }"
    done < "$2"
  } > stubs_auto_c.c
}

echo "== link (auto-stub loop) =="
echo "! empty" > stubs_auto.f90
echo "/* empty */" > stubs_auto_c.c
$FC $FF -c stubs_auto.f90 -o stubs_auto.o
cc -c stubs_auto_c.c -o stubs_auto_c.o

for attempt in 1 2 3 4 5; do
  if $FC $FF -o verify_vexx vexx_tu.o stubs_auto.o stubs_auto_c.o verify_vexx.o $LIBS 2> link.err; then
    echo "link OK (attempt $attempt)"
    break
  fi
  grep -o "undefined reference to \`[A-Za-z0-9_]*'" link.err | sed "s/.*\`//; s/'$//" | sort -u > undef_all.txt
  grep '_$'  undef_all.txt | sed 's/_$//'  > undef_f.txt || true
  grep -v '_$' undef_all.txt               > undef_c.txt || true
  n=$(wc -l < undef_all.txt)
  if [ "$n" -eq 0 ]; then echo "link failed, no undefined symbols parsed:"; head link.err; exit 1; fi
  echo "attempt $attempt: $n undefined ($(wc -l < undef_f.txt) fortran, $(wc -l < undef_c.txt) C) -> stubs"
  gen_stubs undef_f.txt undef_c.txt
  $FC $FF -c stubs_auto.f90 -o stubs_auto.o
  cc -c stubs_auto_c.c -o stubs_auto_c.o
done
[ -x verify_vexx ] || { echo BUILD FAILED; tail -20 link.err; exit 1; }
echo "build complete: ./verify_vexx <dumpdir> <slot> [full|nc] [reps] [warmup]"
