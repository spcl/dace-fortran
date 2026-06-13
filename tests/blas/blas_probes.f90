! BLAS frontend-recognition probes.  One subroutine per recognised routine,
! each driving the standard cBLAS / Fortran-BLAS ABI through a direct call.
! The bridge pattern-matches the callee name in dispatch.cpp and emits the
! matching ``dace.libraries.blas.*`` library node.
MODULE blas_probes
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
CONTAINS
  !
  ! AXPY: ``y := alpha*x + y``
  !
  SUBROUTINE run_daxpy(n, alpha, x, y)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: alpha
    REAL(dp), INTENT(IN) :: x(n)
    REAL(dp), INTENT(INOUT) :: y(n)
    CALL daxpy(n, alpha, x, 1, y, 1)
  END SUBROUTINE run_daxpy
  !
  ! SCAL: ``x := alpha*x``
  !
  SUBROUTINE run_dscal(n, alpha, x)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: alpha
    REAL(dp), INTENT(INOUT) :: x(n)
    CALL dscal(n, alpha, x, 1)
  END SUBROUTINE run_dscal
  !
  ! GEMV: ``y := alpha*A*x + beta*y``
  !
  SUBROUTINE run_dgemv(m, n, alpha, A, x, beta, y)
    INTEGER, INTENT(IN) :: m, n
    REAL(dp), INTENT(IN) :: alpha, beta
    REAL(dp), INTENT(IN) :: A(m, n), x(n)
    REAL(dp), INTENT(INOUT) :: y(m)
    CALL dgemv('N', m, n, alpha, A, m, x, 1, beta, y, 1)
  END SUBROUTINE run_dgemv
  !
  ! GEMM: ``C := alpha*A*B + beta*C``
  !
  SUBROUTINE run_dgemm(m, n, k, alpha, A, B, beta, C)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(dp), INTENT(IN) :: alpha, beta
    REAL(dp), INTENT(IN) :: A(m, k), B(k, n)
    REAL(dp), INTENT(INOUT) :: C(m, n)
    CALL dgemm('N', 'N', m, n, k, alpha, A, m, B, k, beta, C, m)
  END SUBROUTINE run_dgemm
END MODULE blas_probes
