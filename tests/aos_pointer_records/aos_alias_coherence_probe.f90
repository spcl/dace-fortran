! The target of a rebound pointer component is read and written directly while the
! component points at it, so the target and the alias must show the same values.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qv, qc, t, u)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qv(n, k), qc(n, k)
    REAL(8), INTENT(INOUT) :: t(n, k), u(n, k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qv
    q(2)%x => qc
    ! Write through the alias, read the target.
    DO j = 1, k
      DO i = 1, n
        q(1)%x(i, j) = q(1)%x(i, j) + 1.0_8
        t(i, j) = qv(i, j) * 2.0_8
      END DO
    END DO
    ! Write the target, read through the alias.
    DO j = 1, k
      DO i = 1, n
        qc(i, j) = qc(i, j) + 5.0_8
        u(i, j) = q(2)%x(i, j)
      END DO
    END DO
    ! A whole-array write to the target, read back through the alias.
    qv = qv * 3.0_8
    DO j = 1, k
      DO i = 1, n
        u(i, j) = u(i, j) + q(1)%x(i, j)
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
