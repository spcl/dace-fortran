! MAXLOC with DIM= reduces along one axis.
SUBROUTINE maxloc_dim2(n, m, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  INTEGER, INTENT(OUT) :: idx(n)
  idx = MAXLOC(arr, dim=2)
END SUBROUTINE maxloc_dim2
