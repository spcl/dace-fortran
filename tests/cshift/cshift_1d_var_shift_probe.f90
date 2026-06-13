! CSHIFT 1-D with runtime-variable shift.
SUBROUTINE cshift_1d_var(n, arr, shift, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, shift
  REAL(8), INTENT(IN) :: arr(n)
  REAL(8), INTENT(OUT) :: out(n)
  out = CSHIFT(arr, shift)
END SUBROUTINE cshift_1d_var
