! LAPACK frontend-recognition probes.
MODULE lapack_probes
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
CONTAINS
  !
  ! GETRF: LU factorisation with partial pivoting.
  !
  SUBROUTINE run_dgetrf(m, n, A, ipiv, info)
    INTEGER, INTENT(IN) :: m, n
    REAL(dp), INTENT(INOUT) :: A(m, n)
    INTEGER, INTENT(OUT) :: ipiv(min(m, n))
    INTEGER, INTENT(OUT) :: info
    CALL dgetrf(m, n, A, m, ipiv, info)
  END SUBROUTINE run_dgetrf
  !
  ! POTRF: Cholesky factorisation of a symmetric positive-definite matrix
  ! (upper triangle).
  !
  SUBROUTINE run_dpotrf(n, A, info)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(INOUT) :: A(n, n)
    INTEGER, INTENT(OUT) :: info
    CALL dpotrf('U', n, A, n, info)
  END SUBROUTINE run_dpotrf
END MODULE lapack_probes
