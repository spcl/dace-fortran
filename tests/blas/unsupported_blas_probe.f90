! Unsupported-call near-miss probe: ``DROT`` (Givens rotation) is in
! ``knownBlasNames()`` -- so the bridge detects it AS a BLAS call -- but
! is intentionally NOT in ``blasCalleeTag``'s emitter set (a rarely-used
! L1 routine).  The bridge should raise a clear ``NotImplementedError``
! with the canonical routine name + extension hint rather than degrading
! to a generic ``call`` lowering that mints ``_out = ?`` placeholders.
MODULE unsupported_blas_probe
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
CONTAINS
  SUBROUTINE run_drot(n, x, y, c, s)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(INOUT) :: x(n), y(n)
    REAL(dp), INTENT(IN) :: c, s
    CALL drot(n, x, 1, y, 1, c, s)
  END SUBROUTINE run_drot
END MODULE unsupported_blas_probe
