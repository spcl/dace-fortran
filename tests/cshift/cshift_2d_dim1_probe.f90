! CSHIFT 2-D along Fortran dim=1 -- shifts down each column independently.
SUBROUTINE cshift_2d_dim1(n, m, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  REAL(8), INTENT(OUT) :: out(n, m)
  out = CSHIFT(arr, 2, dim=1)
END SUBROUTINE cshift_2d_dim1
