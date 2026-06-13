SUBROUTINE run(rhoc, prhoc, n, m, k)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  COMPLEX(8), INTENT(IN) :: rhoc(n, m)
  COMPLEX(8), INTENT(OUT) :: prhoc(n*k)
  prhoc = RESHAPE(rhoc(:, 1:k), [n*k])
END SUBROUTINE
