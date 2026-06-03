! Writes through the pointer alias.  The lift pass has to insert a
! copy-out at function exit so the underlying TARGET sees the updates.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qa, qb)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(n, k), qb(n, k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO i = 1, n
      DO j = 1, k
        q(1)%x(i, j) = q(1)%x(i, j) * 2.0_8
        q(2)%x(i, j) = q(2)%x(i, j) + 1.0_8
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
