! Pattern 2: a single-character string parameter used purely as
! an enum-style switch (the QE ``addusxx_g`` ``flag`` shape).
!
!   CHARACTER(LEN=1), INTENT(IN) :: action
!   ...
!   IF (action == 'c') THEN
!     ...
!   ELSE IF (action == 'r') THEN
!     ...
!   END IF
!
! After the ``rewrite_string_enum_to_integer`` pass:
!
!   INTEGER, INTENT(IN) :: action
!   ...
!   IF (action == 0) THEN     ! c -> 0
!     ...
!   ELSE IF (action == 1) THEN ! r -> 1
!     ...
!   END IF
!
! The pass also emits a sidecar ``<routine>_<argname>_enum`` map
! (``{'c': 0, 'r': 1, 'i': 2}``) that the bindings layer reads at
! generation time to expose a string-typed wrapper to the Python
! caller: ``run(action='c', ...)`` is normalised to ``action=0``
! before reaching the SDFG.
SUBROUTINE run(out_val, action)
  IMPLICIT NONE
  CHARACTER(LEN=1), INTENT(IN) :: action
  REAL(8), INTENT(OUT) :: out_val
  IF (action == 'c') THEN
    out_val = 1.0d0
  ELSE IF (action == 'r') THEN
    out_val = 2.0d0
  ELSE IF (action == 'i') THEN
    out_val = 3.0d0
  ELSE
    out_val = 0.0d0
  END IF
END SUBROUTINE run
