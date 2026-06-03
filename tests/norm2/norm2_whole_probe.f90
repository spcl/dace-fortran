! NORM2 over a whole 1-D array -- scalar result.
SUBROUTINE norm2_whole(n, v, r)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: v(n)
  REAL(8), INTENT(OUT) :: r
  r = NORM2(v)
END SUBROUTINE norm2_whole
