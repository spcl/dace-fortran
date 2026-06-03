! Two pointer members of different rank inside one record -- mirrors
! Graupel's t_qx_ptr more closely (p:(:) and x:(:,:)).
MODULE m
  IMPLICIT NONE
  INTEGER, PARAMETER :: wp = 8
  TYPE t_qx_ptr
    REAL(wp), POINTER :: p(:), x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, va, vb, ma, mb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(wp), TARGET, INTENT(INOUT) :: va(n), vb(n)
    REAL(wp), TARGET, INTENT(INOUT) :: ma(n, k), mb(n, k)
    REAL(wp), INTENT(OUT) :: out(n, k)
    TYPE(t_qx_ptr) :: q(2)
    INTEGER :: i, j
    q(1)%p => va
    q(1)%x => ma
    q(2)%p => vb
    q(2)%x => mb
    DO i = 1, n
      DO j = 1, k
        out(i, j) = q(1)%x(i, j) + q(2)%x(i, j) + q(1)%p(i) * q(2)%p(i)
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
