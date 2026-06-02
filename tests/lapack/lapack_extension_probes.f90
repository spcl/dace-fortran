! E2E frontend-recognition probes for the LAPACK extension lib nodes
! (Potrs, Geqrf, Orgqr) added in this session.
MODULE lapack_extension_probes
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
CONTAINS
  ! Cholesky-backed solve: A X = B given the Cholesky factor of A.
  SUBROUTINE run_dpotrs(n, nrhs, A, B, info)
    INTEGER, INTENT(IN) :: n, nrhs
    REAL(dp), INTENT(IN) :: A(n, n)
    REAL(dp), INTENT(INOUT) :: B(n, nrhs)
    INTEGER, INTENT(OUT) :: info
    CALL dpotrs('U', n, nrhs, A, n, B, n, info)
  END SUBROUTINE run_dpotrs
  ! QR factorisation: A := QR.
  SUBROUTINE run_dgeqrf(m, n, A, tau, work, lwork, info)
    INTEGER, INTENT(IN) :: m, n, lwork
    REAL(dp), INTENT(INOUT) :: A(m, n)
    REAL(dp), INTENT(OUT) :: tau(min(m, n))
    REAL(dp) :: work(lwork)
    INTEGER, INTENT(OUT) :: info
    CALL dgeqrf(m, n, A, m, tau, work, lwork, info)
  END SUBROUTINE run_dgeqrf
  ! Generate the explicit Q matrix from a packed GEQRF result.
  SUBROUTINE run_dorgqr(m, n, k, A, tau, work, lwork, info)
    INTEGER, INTENT(IN) :: m, n, k, lwork
    REAL(dp), INTENT(INOUT) :: A(m, n)
    REAL(dp), INTENT(IN) :: tau(k)
    REAL(dp) :: work(lwork)
    INTEGER, INTENT(OUT) :: info
    CALL dorgqr(m, n, k, A, m, tau, work, lwork, info)
  END SUBROUTINE run_dorgqr
END MODULE lapack_extension_probes
