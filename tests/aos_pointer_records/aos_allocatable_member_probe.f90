! An AoS whose member is ALLOCATABLE, not POINTER.  This is NOT the
! AoS-of-pointer-records pattern -- the member is filled with
! ``allocate(a(i)%w(...))`` (no ``=>`` rebind), so it belongs to
! flatten-structs' Phase 5c-A / hlfir-lift-alloc-array-of-records, not
! to hlfir-lift-aos-pointer-records.  The matcher must skip it: matching
! here would mint a bogus size-1 placeholder companion and silently
! steal the live member reads.
MODULE m
  IMPLICIT NONE
  TYPE t_alloc
    REAL(8), ALLOCATABLE :: w(:)
  END TYPE
CONTAINS
  SUBROUTINE run(out)
    REAL(8), INTENT(OUT) :: out
    TYPE(t_alloc) :: a(2)
    INTEGER :: i, j
    DO i = 1, 2
      ALLOCATE(a(i)%w(3))
      DO j = 1, 3
        a(i)%w(j) = REAL(i * j, 8)
      END DO
    END DO
    out = a(1)%w(1) + a(2)%w(2)
    DO i = 1, 2
      DEALLOCATE(a(i)%w)
    END DO
  END SUBROUTINE run
END MODULE m
