! MINLOC over a whole 2-D array -- result is rank-1 length-2
! (the multi-dim subscript of the smallest element).
SUBROUTINE minloc_2d_whole(n, m, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  INTEGER, INTENT(OUT) :: idx(2)
  idx = MINLOC(arr)
END SUBROUTINE minloc_2d_whole
