! EOSHIFT 1-D with constant shift and explicit boundary.
SUBROUTINE eoshift_1d(n, v, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: v(n)
  REAL(8), INTENT(OUT) :: out(n)
  out = EOSHIFT(v, SHIFT=1, BOUNDARY=0.0_8)
END SUBROUTINE eoshift_1d
