SUBROUTINE run(rhoc, prhoc, n, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  COMPLEX(8), TARGET, INTENT(INOUT) :: rhoc(n, m)
  COMPLEX(8), POINTER :: prhoc(:, :)
  prhoc => rhoc
  prhoc(1, 1) = (1.0d0, 0.0d0)
END SUBROUTINE
