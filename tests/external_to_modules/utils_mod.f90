! Sidecar module whose procedures are referenced via EXTERNAL in the
! example kernels below.  The bridge needs the module to see the
! procedure interfaces; the ``replace_external_with_modules`` pass
! converts each ``EXTERNAL`` declaration into a ``USE utils_mod,
! ONLY: <name>`` import so flang resolves the type / signature.
MODULE utils_mod
  IMPLICIT NONE
CONTAINS
  REAL(8) FUNCTION dscale(x, f)
    REAL(8), INTENT(IN) :: x, f
    dscale = x * f
  END FUNCTION dscale

  SUBROUTINE dadd(a, b, c, n)
    INTEGER, INTENT(IN) :: n
    REAL(8), INTENT(IN) :: a(n), b(n)
    REAL(8), INTENT(OUT) :: c(n)
    INTEGER :: i
    DO i = 1, n
      c(i) = a(i) + b(i)
    END DO
  END SUBROUTINE dadd

  REAL(8) FUNCTION dsum(a, n)
    INTEGER, INTENT(IN) :: n
    REAL(8), INTENT(IN) :: a(n)
    INTEGER :: i
    dsum = 0.0d0
    DO i = 1, n
      dsum = dsum + a(i)
    END DO
  END FUNCTION dsum
END MODULE utils_mod
