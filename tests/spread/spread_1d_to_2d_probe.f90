! SPREAD 1-D source -> 2-D destination, inserting a new axis at DIM=1.
SUBROUTINE spread_1d_to_2d(n, v, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: v(n)
  REAL(8), INTENT(OUT) :: out(3, n)
  out = SPREAD(v, DIM=1, NCOPIES=3)
END SUBROUTINE spread_1d_to_2d
