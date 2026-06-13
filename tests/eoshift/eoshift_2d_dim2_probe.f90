! EOSHIFT 2-D along Fortran dim=2 with explicit boundary.
SUBROUTINE eoshift_2d_dim2(n, m, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  REAL(8), INTENT(OUT) :: out(n, m)
  out = EOSHIFT(arr, 1, BOUNDARY=-1.0_8, dim=2)
END SUBROUTINE eoshift_2d_dim2
