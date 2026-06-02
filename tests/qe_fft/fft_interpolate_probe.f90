! E2E probe for QE's ``fft_interpolate`` (#9 in the queue).
!
! Mirrors :file:`FFTXlib/src/fft_interfaces.f90`'s real / complex
! variants -- the bridge pattern-matches the callee name and emits a
! :class:`dace.libraries.fft.nodes.FFTInterpolate` lib node.
MODULE fft_interpolate_probe
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
  INTERFACE
     SUBROUTINE fft_interpolate_complex(dfft_in, v_in, dfft_out, v_out)
       IMPORT :: dp
       INTEGER, INTENT(IN) :: dfft_in, dfft_out
       COMPLEX(dp), INTENT(IN)  :: v_in(*)
       COMPLEX(dp), INTENT(OUT) :: v_out(*)
     END SUBROUTINE fft_interpolate_complex
     SUBROUTINE fft_interpolate_real(dfft_in, v_in, dfft_out, v_out)
       IMPORT :: dp
       INTEGER, INTENT(IN) :: dfft_in, dfft_out
       REAL(dp), INTENT(IN)  :: v_in(*)
       REAL(dp), INTENT(OUT) :: v_out(*)
     END SUBROUTINE fft_interpolate_real
  END INTERFACE
CONTAINS
  SUBROUTINE run_fft_interpolate_complex(nin, nout, v_in, v_out)
    INTEGER, INTENT(IN) :: nin, nout
    COMPLEX(dp), INTENT(IN)  :: v_in(nin)
    COMPLEX(dp), INTENT(OUT) :: v_out(nout)
    CALL fft_interpolate_complex(nin, v_in, nout, v_out)
  END SUBROUTINE run_fft_interpolate_complex
  SUBROUTINE run_fft_interpolate_real(nin, nout, v_in, v_out)
    INTEGER, INTENT(IN) :: nin, nout
    REAL(dp), INTENT(IN)  :: v_in(nin)
    REAL(dp), INTENT(OUT) :: v_out(nout)
    CALL fft_interpolate_real(nin, v_in, nout, v_out)
  END SUBROUTINE run_fft_interpolate_real
END MODULE fft_interpolate_probe
