#!/bin/bash
# Build the verify driver against outputs/lib/lib<KERNEL>.so.
#
# The binding .so carries undefined symbols from DEAD prelude module
# procedures (BLAS/LAPACK, MPI, QE utilities, fwfft_y/invfft_y) -- legal for
# a shared library, but the executable must satisfy them.  Resolution:
# mpif90 (MPI), -llapack -lblas (BLAS/LAPACK), no-op clock stubs +
# loud-abort stubs for everything else (incl. fwfft_y/invfft_y: the SDFG's
# FFTs are self-contained; entering these means a dead path was reached).
set -e
cd "$(dirname "$0")"
KERNEL=${KERNEL:-addusxx_g}
DRIVER=${DRIVER:-verify_addusxx}
LIB=${LIB:-$PWD/outputs/lib}       # override to link against outputs/lib_opt etc.
EXE=${EXE:-$DRIVER}                # override to keep several driver builds side by side
FC=${MPIFC:-mpif90}
FF="-O2 -fopenmp -ffree-line-length-none"

$FC $FF -I"$LIB" -c "$DRIVER.f90" -o "$DRIVER.o"

# known-live stubs: clocks no-op, errore loud (the only utilities the kernels
# actually call); everything else aborts if entered.
cat > verify_stubs.f90 <<'EOF'
subroutine start_clock(label)
  character(len=*), intent(in) :: label
end subroutine start_clock
subroutine stop_clock(label)
  character(len=*), intent(in) :: label
end subroutine stop_clock
subroutine errore(calling_routine, message, ierr)
  character(len=*), intent(in) :: calling_routine, message
  integer, intent(in) :: ierr
  if (ierr == 0) return
  write(*,'(5A,I0)') 'ERRORE from ', trim(calling_routine), ': ', trim(message), ' ierr=', ierr
  error stop 8
end subroutine errore
subroutine infomsg(calling_routine, message)
  character(len=*), intent(in) :: calling_routine, message
end subroutine infomsg
EOF
$FC $FF -c verify_stubs.f90 -o verify_stubs.o

echo "! empty" > stubs_auto.f90
echo "/* empty */" > stubs_auto_c.c
$FC $FF -c stubs_auto.f90 -o stubs_auto.o
gcc -c stubs_auto_c.c -o stubs_auto_c.o

for attempt in 1 2 3 4 5; do
  if $FC $FF -o "$EXE" "$DRIVER.o" verify_stubs.o stubs_auto.o stubs_auto_c.o \
       -L"$LIB" -l"$KERNEL" -Wl,-rpath,"$LIB" -llapack -lblas 2> link.err; then
    echo "link OK (attempt $attempt)"
    break
  fi
  grep -o "undefined reference to \`[A-Za-z0-9_]*'" link.err | sed "s/.*\`//; s/'$//" | sort -u > undef_all.txt
  grep '_$'  undef_all.txt | sed 's/_$//'  > undef_f.txt || true
  grep -v '_$' undef_all.txt               > undef_c.txt || true
  n=$(wc -l < undef_all.txt)
  [ "$n" -eq 0 ] && { echo "link failed, nothing parsed:"; head link.err; exit 1; }
  echo "attempt $attempt: $n undefined -> abort stubs"
  {
    echo "! auto-generated abort stubs (dead prelude paths)"
    while read -r s; do
      [ -z "$s" ] && continue
      echo "subroutine $s()"
      echo "  write(*,*) 'FATAL: dead-path stub entered: $s'"
      echo "  error stop 9"
      echo "end subroutine $s"
    done < undef_f.txt
  } > stubs_auto.f90
  {
    echo "#include <stdio.h>"
    echo "#include <stdlib.h>"
    while read -r s; do
      [ -z "$s" ] && continue
      echo "void $s() { fprintf(stderr, \"FATAL: dead-path C stub: $s\\n\"); abort(); }"
    done < undef_c.txt
  } > stubs_auto_c.c
  $FC $FF -c stubs_auto.f90 -o stubs_auto.o
  gcc -c stubs_auto_c.c -o stubs_auto_c.o
done
[ -x "$EXE" ] || { echo BUILD FAILED; tail -10 link.err; exit 1; }
echo "built ./$EXE"
