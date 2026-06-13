! Basic Pattern 1: a kernel that declares one procedure as
! ``EXTERNAL`` even though its definition lives in a sibling
! ``utils_mod`` that the bridge could merge.  After the pass:
!
!   EXTERNAL :: dscale         ->   USE utils_mod, ONLY: dscale
!
! flang then sees the proper interface and inlines / lowers
! ``dscale`` like any other module procedure.
SUBROUTINE run(out_val, x, f)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: x, f
  REAL(8), INTENT(OUT) :: out_val
  REAL(8) :: dscale
  EXTERNAL :: dscale
  out_val = dscale(x, f) + 1.0d0
END SUBROUTINE run
