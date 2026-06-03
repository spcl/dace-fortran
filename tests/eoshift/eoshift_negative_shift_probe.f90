! EOSHIFT 1-D with negative shift.
SUBROUTINE eoshift_negative(n, v, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: v(n)
  REAL(8), INTENT(OUT) :: out(n)
  out = EOSHIFT(v, SHIFT=-2, BOUNDARY=99.0_8)
END SUBROUTINE eoshift_negative
