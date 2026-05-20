! CSR sparse matrix-vector multiplication.  Sister project to
! ``../jacobi/`` -- structurally different to exercise the
! generic prebuilt-HLFIR pipeline (no MPI, no netCDF, no stubs,
! a single Fortran module split across two files).  The entry
! ``csr_spmv`` calls the inlinable helper ``dot_row``; the bridge's
! ``hlfir-inline-all`` pass should fuse them so the SDFG has only
! the outer row loop.
module mod_csr
  use iso_fortran_env, only: real64, int32
  implicit none
  private
  public :: csr_spmv, dot_row

contains

  !> Single CSR row's dot-product with ``x``.  Pure + small -- the
  !! ideal inlining candidate; the test asserts no separate
  !! ``dot_row`` artefact survives in the SDFG.
  pure real(real64) function dot_row(values, colind, row_start, row_end, x) result(s)
    real(real64), intent(in) :: values(:)
    integer(int32), intent(in) :: colind(:)
    integer(int32), intent(in) :: row_start
    integer(int32), intent(in) :: row_end
    real(real64), intent(in) :: x(:)
    integer(int32) :: k
    s = 0.0_real64
    do k = row_start, row_end - 1
      s = s + values(k) * x(colind(k))
    end do
  end function dot_row

  !> CSR sparse matrix-vector multiply: ``y = A * x``.
  subroutine csr_spmv(rowptr, colind, values, x, y, nrows)
    integer(int32), intent(in) :: nrows
    integer(int32), intent(in) :: rowptr(nrows + 1)
    integer(int32), intent(in) :: colind(:)
    real(real64), intent(in)   :: values(:)
    real(real64), intent(in)   :: x(:)
    real(real64), intent(out)  :: y(nrows)
    integer(int32) :: i

    do i = 1, nrows
      y(i) = dot_row(values, colind, rowptr(i), rowptr(i + 1), x)
    end do
  end subroutine csr_spmv

end module mod_csr
