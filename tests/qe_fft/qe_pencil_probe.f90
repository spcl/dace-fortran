! E2E probe for QE's parallel pencil-pipeline routines (#8 in the queue).
!
! Mirrors the per-axis 1-D FFT helpers (``cft_1z`` / ``cft_1y`` / ``cft_1x``,
! from ``FFTXlib/src/fft_scalar.FFTW.f90``) and the MPI alltoall transposes
! (``fft_scatter_xy`` / ``fft_scatter_yz`` from ``FFTXlib/src/fft_scatter.f90``)
! that compose into the parallel 3-D FFT in ``fft_parallel.f90``.
!
! The bridge pattern-matches each callee name and emits the matching DaCe
! lib node (``FFT`` for the per-axis FFTs, ``Alltoall`` for the scatters).
MODULE qe_pencil_probe
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
  INTERFACE
     SUBROUTINE cft_1z(c, nsl, nz, ldz, isign, cout)
       IMPORT :: dp
       COMPLEX(dp), INTENT(IN)  :: c(*)
       COMPLEX(dp), INTENT(OUT) :: cout(*)
       INTEGER,     INTENT(IN)  :: nsl, nz, ldz, isign
     END SUBROUTINE cft_1z
     SUBROUTINE cft_1y(c, nsl, nz, ldz, isign, cout)
       IMPORT :: dp
       COMPLEX(dp), INTENT(IN)  :: c(*)
       COMPLEX(dp), INTENT(OUT) :: cout(*)
       INTEGER,     INTENT(IN)  :: nsl, nz, ldz, isign
     END SUBROUTINE cft_1y
     SUBROUTINE cft_1x(c, nsl, nz, ldz, isign, cout)
       IMPORT :: dp
       COMPLEX(dp), INTENT(IN)  :: c(*)
       COMPLEX(dp), INTENT(OUT) :: cout(*)
       INTEGER,     INTENT(IN)  :: nsl, nz, ldz, isign
     END SUBROUTINE cft_1x
     SUBROUTINE fft_scatter_xy(desc, f_in, f_aux, nxx_, isgn, comm)
       IMPORT :: dp
       INTEGER,     INTENT(IN)    :: desc, nxx_, isgn, comm
       COMPLEX(dp), INTENT(IN)    :: f_in(*)
       COMPLEX(dp), INTENT(OUT)   :: f_aux(*)
     END SUBROUTINE fft_scatter_xy
     SUBROUTINE fft_scatter_yz(desc, f_in, f_aux, nxx_, isgn)
       IMPORT :: dp
       INTEGER,     INTENT(IN)    :: desc, nxx_, isgn
       COMPLEX(dp), INTENT(IN)    :: f_in(*)
       COMPLEX(dp), INTENT(OUT)   :: f_aux(*)
     END SUBROUTINE fft_scatter_yz
  END INTERFACE
CONTAINS
  SUBROUTINE run_cft_1z(nsl, nz, ldz, c, cout)
    INTEGER, INTENT(IN) :: nsl, nz, ldz
    COMPLEX(dp), INTENT(IN)  :: c(ldz * nsl)
    COMPLEX(dp), INTENT(OUT) :: cout(ldz * nsl)
    CALL cft_1z(c, nsl, nz, ldz, -1, cout)
  END SUBROUTINE run_cft_1z
  SUBROUTINE run_cft_1y(nsl, nz, ldz, c, cout)
    INTEGER, INTENT(IN) :: nsl, nz, ldz
    COMPLEX(dp), INTENT(IN)  :: c(ldz * nsl)
    COMPLEX(dp), INTENT(OUT) :: cout(ldz * nsl)
    CALL cft_1y(c, nsl, nz, ldz, -1, cout)
  END SUBROUTINE run_cft_1y
  SUBROUTINE run_cft_1x(nsl, nz, ldz, c, cout)
    INTEGER, INTENT(IN) :: nsl, nz, ldz
    COMPLEX(dp), INTENT(IN)  :: c(ldz * nsl)
    COMPLEX(dp), INTENT(OUT) :: cout(ldz * nsl)
    CALL cft_1x(c, nsl, nz, ldz, -1, cout)
  END SUBROUTINE run_cft_1x
  SUBROUTINE run_fft_scatter_xy(nxx_, f_in, f_aux)
    INTEGER, INTENT(IN) :: nxx_
    COMPLEX(dp), INTENT(IN)  :: f_in(nxx_)
    COMPLEX(dp), INTENT(OUT) :: f_aux(nxx_)
    CALL fft_scatter_xy(0, f_in, f_aux, nxx_, 1, 0)
  END SUBROUTINE run_fft_scatter_xy
  SUBROUTINE run_fft_scatter_yz(nxx_, f_in, f_aux)
    INTEGER, INTENT(IN) :: nxx_
    COMPLEX(dp), INTENT(IN)  :: f_in(nxx_)
    COMPLEX(dp), INTENT(OUT) :: f_aux(nxx_)
    CALL fft_scatter_yz(0, f_in, f_aux, nxx_, 1)
  END SUBROUTINE run_fft_scatter_yz
END MODULE qe_pencil_probe
