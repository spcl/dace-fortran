! Pattern 1, multiple-name form.  ``EXTERNAL`` can list several
! procedures on one line, each defined in possibly different
! sidecar modules.  The pass resolves every name independently:
!
!   EXTERNAL :: dscale, dadd, dsum
!
! becomes
!
!   USE utils_mod, ONLY: dscale, dadd, dsum
!
! (collapsed to one USE per source module to keep the resulting
! source compact).  A name that the search-dirs can't resolve is
! left in place with a comment so the user notices the gap.
SUBROUTINE run(out_val, x, f, a, b, c, s, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: x, f, a(n), b(n)
  REAL(8), INTENT(OUT) :: c(n), s, out_val
  REAL(8) :: dscale, dsum
  EXTERNAL :: dscale, dadd, dsum
  CALL dadd(a, b, c, n)
  s = dsum(c, n)
  out_val = dscale(x, f)
END SUBROUTINE run
