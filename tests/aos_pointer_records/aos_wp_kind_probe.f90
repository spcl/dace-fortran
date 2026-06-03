! Single-file AoS-of-pointer-records that uses the ``wp`` precision
! alias without a locally bound integer parameter -- the alias is
! defined via SELECTED_REAL_KIND, which the bridge can't evaluate.
! Exercises the normalize_kind_parameters preprocess so the rewrite
! makes the source self-contained before flang sees it.
MODULE m
  IMPLICIT NONE
  INTEGER, PARAMETER :: wp = SELECTED_REAL_KIND(15, 300)
  TYPE t_ptr_2d
    REAL(KIND=wp), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qa, qb, qsum)
    INTEGER, INTENT(IN) :: n, k
    REAL(KIND=wp), TARGET, INTENT(INOUT) :: qa(n, k), qb(n, k)
    REAL(KIND=wp), INTENT(OUT) :: qsum(n, k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO i = 1, n
      DO j = 1, k
        qsum(i, j) = q(1)%x(i, j) + q(2)%x(i, j) * 2.0_wp
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
