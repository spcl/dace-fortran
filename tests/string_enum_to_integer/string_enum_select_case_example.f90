! Pattern 2 in ``SELECT CASE`` form -- the other common shape:
!
!   SELECT CASE (TRIM(mode))
!   CASE ('forward')
!     ...
!   CASE ('backward')
!     ...
!   CASE DEFAULT
!     ...
!   END SELECT
!
! Rewrites to:
!
!   SELECT CASE (mode)
!   CASE (0)              ! 'forward' -> 0
!     ...
!   CASE (1)              ! 'backward' -> 1
!     ...
!   CASE DEFAULT
!     ...
!   END SELECT
!
! Longer-than-1 strings are equally valid enum values; the
! rewriter sizes the integer mapping by enumerating all literals
! it sees compared against the same variable, not by string length.
SUBROUTINE run(out_val, mode)
  IMPLICIT NONE
  CHARACTER(LEN=*), INTENT(IN) :: mode
  REAL(8), INTENT(OUT) :: out_val
  SELECT CASE (mode)
  CASE ('forward')
    out_val = 1.0d0
  CASE ('backward')
    out_val = -1.0d0
  CASE ('zero')
    out_val = 0.0d0
  CASE DEFAULT
    out_val = 999.0d0
  END SELECT
END SUBROUTINE run
