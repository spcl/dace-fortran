! Pattern 2 with case-insensitive comparisons -- the canonical QE
! ``addusxx_g`` shape:
!
!   IF (flag == 'c' .OR. flag == 'C') THEN ...
!
! Both ``'c'`` and ``'C'`` must map to the SAME integer value so
! the rewritten condition collapses to ``IF (flag == 0)`` without
! splitting case-distinct branches.  The bindings layer then
! accepts ``flag='c'`` and ``flag='C'`` (lower / upper) at the
! Python boundary and normalises both to ``0``.
SUBROUTINE run(out_val, flag)
  IMPLICIT NONE
  CHARACTER(LEN=1), INTENT(IN) :: flag
  REAL(8), INTENT(OUT) :: out_val
  LOGICAL :: add_complex, add_real, add_imaginary
  add_complex   = (flag == 'c' .OR. flag == 'C')
  add_real      = (flag == 'r' .OR. flag == 'R')
  add_imaginary = (flag == 'i' .OR. flag == 'I')
  IF (add_complex) THEN
    out_val = 1.0d0
  ELSE IF (add_real) THEN
    out_val = 2.0d0
  ELSE IF (add_imaginary) THEN
    out_val = 3.0d0
  ELSE
    out_val = 0.0d0
  END IF
END SUBROUTINE run
