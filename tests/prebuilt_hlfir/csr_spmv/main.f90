! Tiny driver: builds a 3x3 identity-like CSR matrix and prints
! ``y = A * x`` for ``x = [1, 2, 3]``.  Lives in a second file so
! the project has more than one TU -- exercises the helper's
! per-file emit on a multi-TU project that has no external deps.
program csr_demo
  use iso_fortran_env, only: real64, int32
  use mod_csr, only: csr_spmv
  implicit none
  integer(int32), parameter :: n = 3, nnz = 5
  integer(int32) :: rowptr(n + 1), colind(nnz)
  real(real64) :: values(nnz), x(n), y(n)

  rowptr = [1, 2, 4, 6]
  colind = [1, 1, 2, 2, 3]
  values = [1.0_real64, 0.5_real64, 1.0_real64, 0.5_real64, 1.0_real64]
  x = [1.0_real64, 2.0_real64, 3.0_real64]

  call csr_spmv(rowptr, colind, values, x, y, n)
  print *, "y =", y
end program csr_demo
