! Minimal Fortran probe: 2-D and 3-D FFTs through FFTW3.
!
! FFTW3 declarations are inlined so the probe doesn't need ``fftw3.f03``
! on the flang include path. This is exactly the shape QE's serial FFT
! backend (``FFTXlib/src/fft_scalar.FFTW3.f90``) generates internally
! after the parallel scatter pipeline reduces to a local 2-D or 3-D
! transform.
MODULE fft_probe
  USE, INTRINSIC :: iso_c_binding
  IMPLICIT NONE
  INTEGER(C_INT), PARAMETER :: FFTW_FORWARD  = -1
  INTEGER(C_INT), PARAMETER :: FFTW_BACKWARD =  1
  INTEGER(C_INT), PARAMETER :: FFTW_ESTIMATE = 64
  !
  INTERFACE
     TYPE(C_PTR) FUNCTION fftw_plan_dft_2d(n0, n1, in, out, sign, flags) &
          BIND(C, NAME='fftw_plan_dft_2d')
       IMPORT :: C_INT, C_PTR, C_DOUBLE_COMPLEX
       INTEGER(C_INT), VALUE :: n0, n1
       COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: in
       COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: out
       INTEGER(C_INT), VALUE :: sign
       INTEGER(C_INT), VALUE :: flags
     END FUNCTION fftw_plan_dft_2d
     !
     TYPE(C_PTR) FUNCTION fftw_plan_dft_3d(n0, n1, n2, in, out, sign, flags) &
          BIND(C, NAME='fftw_plan_dft_3d')
       IMPORT :: C_INT, C_PTR, C_DOUBLE_COMPLEX
       INTEGER(C_INT), VALUE :: n0, n1, n2
       COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: in
       COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: out
       INTEGER(C_INT), VALUE :: sign
       INTEGER(C_INT), VALUE :: flags
     END FUNCTION fftw_plan_dft_3d
     !
     SUBROUTINE fftw_execute_dft(plan, in, out) &
          BIND(C, NAME='fftw_execute_dft')
       IMPORT :: C_PTR, C_DOUBLE_COMPLEX
       TYPE(C_PTR), VALUE :: plan
       COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: in
       COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(*) :: out
     END SUBROUTINE fftw_execute_dft
     !
     SUBROUTINE fftw_destroy_plan(plan) BIND(C, NAME='fftw_destroy_plan')
       IMPORT :: C_PTR
       TYPE(C_PTR), VALUE :: plan
     END SUBROUTINE fftw_destroy_plan
  END INTERFACE
CONTAINS
  !
  ! 2-D forward FFT, in-place over a column-major complex(8) buffer.
  !
  SUBROUTINE run_fft_2d(M, N, x)
    INTEGER, INTENT(IN) :: M, N
    COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(M, N), INTENT(INOUT) :: x
    TYPE(C_PTR) :: plan
    plan = fftw_plan_dft_2d(N, M, x, x, FFTW_FORWARD, FFTW_ESTIMATE)
    CALL fftw_execute_dft(plan, x, x)
    CALL fftw_destroy_plan(plan)
  END SUBROUTINE run_fft_2d
  !
  ! 3-D forward FFT, in-place over a column-major complex(8) buffer.
  !
  SUBROUTINE run_fft_3d(L, M, N, x)
    INTEGER, INTENT(IN) :: L, M, N
    COMPLEX(C_DOUBLE_COMPLEX), DIMENSION(L, M, N), INTENT(INOUT) :: x
    TYPE(C_PTR) :: plan
    plan = fftw_plan_dft_3d(N, M, L, x, x, FFTW_FORWARD, FFTW_ESTIMATE)
    CALL fftw_execute_dft(plan, x, x)
    CALL fftw_destroy_plan(plan)
  END SUBROUTINE run_fft_3d
END MODULE fft_probe
