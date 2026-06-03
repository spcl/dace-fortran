! Pattern 1, defensive form.  The kernel ``USE``s ``utils_mod``
! AND has an ``EXTERNAL`` for one of its procedures (a legacy
! belt-and-braces declaration that flang accepts but the bridge
! gets confused by  --  the ``EXTERNAL`` shadows the module
! import's specific interface, falling back to the implicit-
! interface path).  After the pass:
!
!   USE utils_mod
!   ...
!   EXTERNAL :: dadd        ->   (deleted, dadd is already in scope)
!
! The pass must not duplicate the ``USE``, just drop the redundant
! ``EXTERNAL`` line.
SUBROUTINE run(a, b, c, n)
  USE utils_mod
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n), b(n)
  REAL(8), INTENT(OUT) :: c(n)
  EXTERNAL :: dadd
  CALL dadd(a, b, c, n)
END SUBROUTINE run
