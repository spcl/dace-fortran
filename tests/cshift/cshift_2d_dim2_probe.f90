! CSHIFT 2-D along Fortran dim=2.
SUBROUTINE cshift_2d_dim2(n, m, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  REAL(8), INTENT(OUT) :: out(n, m)
  out = CSHIFT(arr, 1, dim=2)
END SUBROUTINE cshift_2d_dim2
