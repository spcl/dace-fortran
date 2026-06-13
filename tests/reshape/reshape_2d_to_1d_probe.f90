! Fortran RESHAPE: 2-D source -> 1-D destination.  Total element count
! preserved; the destination's column-major flattening matches Fortran's
! storage order so the lowered SDFG should reduce to a flat copy.
SUBROUTINE reshape_2d_to_1d(n, m, arr, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  REAL(8), INTENT(OUT) :: out(n*m)
  out = RESHAPE(arr, SHAPE=[n*m])
END SUBROUTINE reshape_2d_to_1d
