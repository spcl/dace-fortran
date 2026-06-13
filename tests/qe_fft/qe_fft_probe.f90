! QE-style fwfft / invfft frontend-recognition probe.
!
! Quantum ESPRESSO's ``fft_interfaces`` module exposes:
!
!   CALL  fwfft(fft_kind, f, dfft [, howmany])  ! G-space -> R-space
!   CALL invfft(fft_kind, f, dfft [, howmany])  ! R-space -> G-space
!
! The generic resolves through an ``INTERFACE`` block to specific
! subroutines (``fwfft_y`` / ``invfft_y`` for the standard grid,
! ``fwfft_b`` / ``invfft_b`` for the box grid).  This probe calls the
! specific subroutines directly via plain ``INTERFACE`` declarations so
! flang can lower the calls without the QE ``fft_type_descriptor`` USE
! closure -- the bridge keys recognition on the callee name only.
MODULE qe_fft_probe
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
  INTERFACE
     SUBROUTINE fwfft_y(fft_kind, f, dfft_size)
       IMPORT :: dp
       CHARACTER(LEN=*), INTENT(IN)    :: fft_kind
       COMPLEX(dp),      INTENT(INOUT) :: f(*)
       INTEGER,          INTENT(IN)    :: dfft_size
     END SUBROUTINE fwfft_y
     !
     SUBROUTINE invfft_y(fft_kind, f, dfft_size)
       IMPORT :: dp
       CHARACTER(LEN=*), INTENT(IN)    :: fft_kind
       COMPLEX(dp),      INTENT(INOUT) :: f(*)
       INTEGER,          INTENT(IN)    :: dfft_size
     END SUBROUTINE invfft_y
  END INTERFACE
CONTAINS
  !
  ! G-space -> R-space (forward transform from coefficients to grid).
  !
  SUBROUTINE run_fwfft(n, f)
    INTEGER, INTENT(IN)         :: n
    COMPLEX(dp), INTENT(INOUT)  :: f(n)
    CALL fwfft_y('Wave', f, n)
  END SUBROUTINE run_fwfft
  !
  ! R-space -> G-space (inverse transform from grid to coefficients).
  !
  SUBROUTINE run_invfft(n, f)
    INTEGER, INTENT(IN)         :: n
    COMPLEX(dp), INTENT(INOUT)  :: f(n)
    CALL invfft_y('Rho', f, n)
  END SUBROUTINE run_invfft
END MODULE qe_fft_probe
