! MINLOC over a whole 1-D array -- scalar location result.
SUBROUTINE minloc_whole(n, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  INTEGER, INTENT(OUT) :: idx(1)
  idx = MINLOC(arr)
END SUBROUTINE minloc_whole
