! Smallest-possible AoS-of-pointer-records: one record, one pointer
! member, two slots.  Mirrors the inner shape of Graupel's t_qx_ptr%x.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qa, qb, qsum)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(n, k), qb(n, k)
    REAL(8), INTENT(OUT) :: qsum(n, k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO i = 1, n
      DO j = 1, k
        qsum(i, j) = q(1)%x(i, j) + q(2)%x(i, j)
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
