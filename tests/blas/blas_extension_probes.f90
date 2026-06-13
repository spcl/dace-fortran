! E2E frontend-recognition probes for the BLAS extension lib nodes added
! in this session.  One ``run_<routine>`` subroutine per recognised
! Fortran callee.  The bridge pattern-matches the callee name and emits
! the matching ``dace.libraries.blas.*`` library node.
MODULE blas_extension_probes
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
CONTAINS
  ! L1: COPY -- y := x
  SUBROUTINE run_dcopy(n, x, y)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: x(n)
    REAL(dp), INTENT(OUT) :: y(n)
    CALL dcopy(n, x, 1, y, 1)
  END SUBROUTINE run_dcopy
  ! L1: SWAP -- x, y := y, x
  SUBROUTINE run_dswap(n, x, y)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(INOUT) :: x(n), y(n)
    CALL dswap(n, x, 1, y, 1)
  END SUBROUTINE run_dswap
  ! L2: GER -- A := alpha*x*y' + A
  SUBROUTINE run_dger(m, n, alpha, x, y, A)
    INTEGER, INTENT(IN) :: m, n
    REAL(dp), INTENT(IN) :: alpha, x(m), y(n)
    REAL(dp), INTENT(INOUT) :: A(m, n)
    CALL dger(m, n, alpha, x, 1, y, 1, A, m)
  END SUBROUTINE run_dger
  ! L2: TRSV -- solve op(A) x = b
  SUBROUTINE run_dtrsv(n, A, x)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: A(n, n)
    REAL(dp), INTENT(INOUT) :: x(n)
    CALL dtrsv('L', 'N', 'N', n, A, n, x, 1)
  END SUBROUTINE run_dtrsv
  ! L2: TRMV -- x := op(A) x
  SUBROUTINE run_dtrmv(n, A, x)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: A(n, n)
    REAL(dp), INTENT(INOUT) :: x(n)
    CALL dtrmv('L', 'N', 'N', n, A, n, x, 1)
  END SUBROUTINE run_dtrmv
  ! L2: SYMV -- y := alpha*A*x + beta*y (symmetric A)
  SUBROUTINE run_dsymv(n, alpha, A, x, beta, y)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN) :: alpha, beta, A(n, n), x(n)
    REAL(dp), INTENT(INOUT) :: y(n)
    CALL dsymv('L', n, alpha, A, n, x, 1, beta, y, 1)
  END SUBROUTINE run_dsymv
  ! L3: TRSM -- solve op(A) X = alpha B
  SUBROUTINE run_dtrsm(m, n, alpha, A, B)
    INTEGER, INTENT(IN) :: m, n
    REAL(dp), INTENT(IN) :: alpha, A(m, m)
    REAL(dp), INTENT(INOUT) :: B(m, n)
    CALL dtrsm('L', 'L', 'N', 'N', m, n, alpha, A, m, B, m)
  END SUBROUTINE run_dtrsm
  ! L3: TRMM -- B := alpha*op(A)*B (triangular A)
  SUBROUTINE run_dtrmm(m, n, alpha, A, B)
    INTEGER, INTENT(IN) :: m, n
    REAL(dp), INTENT(IN) :: alpha, A(m, m)
    REAL(dp), INTENT(INOUT) :: B(m, n)
    CALL dtrmm('L', 'L', 'N', 'N', m, n, alpha, A, m, B, m)
  END SUBROUTINE run_dtrmm
  ! L3: SYMM -- C := alpha*A*B + beta*C (symmetric A)
  SUBROUTINE run_dsymm(m, n, alpha, A, B, beta, C)
    INTEGER, INTENT(IN) :: m, n
    REAL(dp), INTENT(IN) :: alpha, beta, A(m, m), B(m, n)
    REAL(dp), INTENT(INOUT) :: C(m, n)
    CALL dsymm('L', 'L', m, n, alpha, A, m, B, m, beta, C, m)
  END SUBROUTINE run_dsymm
  ! L3: SYRK -- C := alpha*A*A' + beta*C
  SUBROUTINE run_dsyrk(n, k, alpha, A, beta, C)
    INTEGER, INTENT(IN) :: n, k
    REAL(dp), INTENT(IN) :: alpha, beta, A(n, k)
    REAL(dp), INTENT(INOUT) :: C(n, n)
    CALL dsyrk('L', 'N', n, k, alpha, A, n, beta, C, n)
  END SUBROUTINE run_dsyrk
END MODULE blas_extension_probes
