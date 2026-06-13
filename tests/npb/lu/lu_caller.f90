! Fortran caller wrappers for the NPB LU e2e numerical test.
!
! The bridge entry ``useapplu::call_dolu`` is parameterless; LU's
! result lives in module-level state (``rsdnm(5)`` residual norms,
! ``itmax`` solver count, the solution field ``u``).  We expose
! BIND(C) wrappers so the Python test can:
!
!   run_dolu_c     -- CALL call_dolu()
!   get_rsdnm_c    -- copy ``lu::rsdnm(5)`` to a caller-allocated buffer
!   get_itmax_c    -- copy ``lu::itmax`` to a caller-allocated buffer
!
! The SDFG side reads ``rsdnm`` directly from its kwargs dict (the
! bridge surfaces every module-level variable as an SDFG argument),
! so no SDFG-side accessor is needed.

! NPB LU configuration inputs (``nx0`` / ``ny0`` / ``nz0`` / ``itmax`` /
! ``dt``) are module-level state the user is expected to set BEFORE the
! first ``dolu`` call.  ``domain()`` reads them; ``setcoeff`` then
! computes derived grid metrics from them.  init_lu_c picks NPB Class S
! (``nx0=ny0=nz0=12``, ``itmax=50``, ``dt=0.5``) so the test finishes in
! sub-second wall time on both sides.  ``tolrsd`` is set to the in-file
! defaults; ``inorm`` is set so ``ssor`` doesn't divide by zero in its
! print-frequency check.
SUBROUTINE init_lu_c(nx0_in, ny0_in, nz0_in, itmax_in, dt_in) &
    BIND(C, NAME="init_lu_c")
  USE lu, ONLY: nx0, ny0, nz0, itmax, dt, omega, tolrsd, inorm
  IMPLICIT NONE
  INTEGER, VALUE :: nx0_in, ny0_in, nz0_in, itmax_in
  DOUBLE PRECISION, VALUE :: dt_in
  nx0 = nx0_in
  ny0 = ny0_in
  nz0 = nz0_in
  itmax = itmax_in
  dt = dt_in
  ! SSOR relaxation + convergence tolerances + print frequency: NPB
  ! Class S in-file defaults (lu.F90 declares the ``omega_default`` and
  ! ``tolrsd_def`` parameters but the module-level ``omega`` / ``tolrsd``
  ! storage is left uninitialised, so the caller must set them).
  omega = 1.2D0
  tolrsd = 1.0D-08
  inorm = itmax
END SUBROUTINE init_lu_c

SUBROUTINE run_dolu_c() BIND(C, NAME="run_dolu_c")
  USE useapplu, ONLY: call_dolu
  IMPLICIT NONE
  CALL call_dolu()
END SUBROUTINE run_dolu_c

SUBROUTINE get_rsdnm_c(rsdnm_out) BIND(C, NAME="get_rsdnm_c")
  USE lu, ONLY: rsdnm
  IMPLICIT NONE
  DOUBLE PRECISION, INTENT(OUT) :: rsdnm_out(5)
  rsdnm_out = rsdnm
END SUBROUTINE get_rsdnm_c

SUBROUTINE get_itmax_c(itmax_out) BIND(C, NAME="get_itmax_c")
  USE lu, ONLY: itmax
  IMPLICIT NONE
  INTEGER, INTENT(OUT) :: itmax_out
  itmax_out = itmax
END SUBROUTINE get_itmax_c
