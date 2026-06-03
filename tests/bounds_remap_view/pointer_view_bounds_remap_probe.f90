SUBROUTINE run(rhoc, prhoc, n, m, k)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  COMPLEX(8), TARGET, INTENT(INOUT) :: rhoc(n, m)
  COMPLEX(8), POINTER :: prhoc(:)
  prhoc(1 : n*k) => rhoc(:, 1:k)
  prhoc(1) = (1.0d0, 0.0d0)
END SUBROUTINE
