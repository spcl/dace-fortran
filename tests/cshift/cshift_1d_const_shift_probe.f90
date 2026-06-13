! CSHIFT 1-D with constant shift.
SUBROUTINE cshift_1d_const(n, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  REAL(8), INTENT(OUT) :: out(n)
  out = CSHIFT(arr, 1)
END SUBROUTINE cshift_1d_const
