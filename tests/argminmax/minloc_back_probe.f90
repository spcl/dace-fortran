! MINLOC with BACK=.TRUE. -- last-occurrence tie-break.
SUBROUTINE minloc_back(n, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  INTEGER, INTENT(OUT) :: idx(1)
  idx = MINLOC(arr, back=.true.)
END SUBROUTINE minloc_back
