! Fortran RESHAPE: 1-D source -> 2-D destination, column-major flatten.
SUBROUTINE reshape_1d_to_2d(n, m, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n*m)
  REAL(8), INTENT(OUT) :: out(n, m)
  out = RESHAPE(arr, SHAPE=[n, m])
END SUBROUTINE reshape_1d_to_2d
