SUBROUTINE run(src, dst, n, m, k)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  COMPLEX(8), INTENT(IN) :: src(n, m)
  COMPLEX(8), INTENT(OUT) :: dst(n, k)
  dst = src(:, 1:k)
END SUBROUTINE
