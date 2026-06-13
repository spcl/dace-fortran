! Runtime index dispatch: ``q(iqx)%x(i, j)`` where ``iqx`` is a
! lookup-table-driven scalar.  This is the exact shape Graupel's
! ``ix = 1..np ; iqx = qp_ind(ix)`` loop body produces.
MODULE m
  IMPLICIT NONE
  INTEGER, PARAMETER :: wp = 8
  TYPE t_ptr_2d
    REAL(wp), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qa, qb, qc, qd, ix, out)
    INTEGER, INTENT(IN) :: n, k, ix
    REAL(wp), TARGET, INTENT(INOUT) :: qa(n, k), qb(n, k), qc(n, k), qd(n, k)
    REAL(wp), INTENT(OUT) :: out(n, k)
    TYPE(t_ptr_2d) :: q(4)
    INTEGER, PARAMETER :: lookup(4) = (/ 1, 2, 3, 4 /)
    INTEGER :: i, j, iqx
    q(1)%x => qa
    q(2)%x => qb
    q(3)%x => qc
    q(4)%x => qd
    iqx = lookup(ix)
    DO i = 1, n
      DO j = 1, k
        out(i, j) = q(iqx)%x(i, j)
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
