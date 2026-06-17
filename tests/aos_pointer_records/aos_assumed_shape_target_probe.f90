! Assumed-shape pointer targets: ``q(iqx)%x => target(:,:)`` where every
! rebind target is an ASSUMED-SHAPE dummy (``REAL :: t(:,:)``), so the
! AoS-of-pointer-records gather temp's inner extents are not a static
! ``(n, k)`` sequence type but are recovered at runtime via
! ``fir.box_dims`` on each target's box.  This is exactly Graupel's
! ``t_qx_ptr%x`` shape.
!
! Regression for the ``fir.box_dims`` extent -> ``<name>_d<dim>``
! resolution in ``traceExtentExpr``: without it the gather temp's inner
! dims fall through to ``"?"`` and the builder mints fresh extent symbols
! (``q_x_1_d1`` ...) that no passed array backs, so the call-time
! auto-fill defaults them to ``1`` -> the temp is under-allocated -> a
! heap-buffer-overflow corrupts the heap (observed as ``double free`` /
! ``corrupted double-linked list`` -> SIGABRT).
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(qa, qb, qc, qd, ix, out)
    REAL(8), TARGET, INTENT(INOUT) :: qa(:,:), qb(:,:), qc(:,:), qd(:,:)
    INTEGER, INTENT(IN) :: ix
    REAL(8), INTENT(OUT) :: out(:,:)
    TYPE(t_ptr_2d) :: q(4)
    INTEGER, PARAMETER :: lookup(4) = (/ 1, 2, 3, 4 /)
    INTEGER :: i, j, iqx, n, k
    n = SIZE(qa, 1)
    k = SIZE(qa, 2)
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
