! CSHIFT 3-D along Fortran dim=2.
SUBROUTINE cshift_3d_dim2(n, m, k, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: arr(n, m, k)
  REAL(8), INTENT(OUT) :: out(n, m, k)
  out = CSHIFT(arr, -1, dim=2)
END SUBROUTINE cshift_3d_dim2
