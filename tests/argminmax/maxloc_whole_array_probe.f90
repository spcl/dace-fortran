! MAXLOC over a whole 1-D array -- scalar location result.
SUBROUTINE maxloc_whole(n, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  INTEGER, INTENT(OUT) :: idx(1)
  idx = MAXLOC(arr)
END SUBROUTINE maxloc_whole
