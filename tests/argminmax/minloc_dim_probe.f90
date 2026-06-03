! MINLOC with DIM= reduces along one axis, returns rank-1.
SUBROUTINE minloc_dim1(n, m, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: arr(n, m)
  INTEGER, INTENT(OUT) :: idx(m)
  idx = MINLOC(arr, dim=1)
END SUBROUTINE minloc_dim1
