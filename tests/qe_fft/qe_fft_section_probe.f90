! QE fwfft/invfft on array sections, kept apart from qe_fft_probe.f90: one file holding both an f(*) and an f(:)
! interface of the same external fwfft_y/invfft_y makes flang call it through a converted address, not by name.
! Section buffers, as QE's vexx passes them (``invfft('Wave', temppsic_d(:, ii), dfftt)``): each call transforms
! one column (contiguous) or one row (strided) of a 2-D array, not the whole array.
MODULE qe_fft_section_probe
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
  INTERFACE
    SUBROUTINE fwfft_y(fft_kind, f, dfft_size)
      IMPORT :: dp
      CHARACTER(LEN=*), INTENT(IN)    :: fft_kind
      COMPLEX(dp), INTENT(INOUT) :: f(:)
      INTEGER, INTENT(IN)    :: dfft_size
    END SUBROUTINE fwfft_y
    !
    SUBROUTINE invfft_y(fft_kind, f, dfft_size)
      IMPORT :: dp
      CHARACTER(LEN=*), INTENT(IN)    :: fft_kind
      COMPLEX(dp), INTENT(INOUT) :: f(:)
      INTEGER, INTENT(IN)    :: dfft_size
    END SUBROUTINE invfft_y
  END INTERFACE
CONTAINS
  SUBROUTINE run_invfft_columns(n, m, f)
    INTEGER, INTENT(IN)         :: n, m
    COMPLEX(dp), INTENT(INOUT)  :: f(n, m)
    INTEGER :: j
    DO j = 1, m
      CALL invfft_y('Wave', f(:, j), n)
    END DO
  END SUBROUTINE run_invfft_columns
  !
  SUBROUTINE run_fwfft_rows(n, m, f)
    INTEGER, INTENT(IN)         :: n, m
    COMPLEX(dp), INTENT(INOUT)  :: f(n, m)
    INTEGER :: i
    DO i = 1, n
      CALL fwfft_y('Wave', f(i, :), m)
    END DO
  END SUBROUTINE run_fwfft_rows
  !
  ! A section of an ALLOCATABLE work array, which the bridge renders with a bare ``:`` dim.
  SUBROUTINE run_fwfft_allocatable_column(n, m, f)
    INTEGER, INTENT(IN)         :: n, m
    COMPLEX(dp), INTENT(INOUT)  :: f(n, m)
    COMPLEX(dp), ALLOCATABLE    :: work(:, :, :)
    ALLOCATE (work(n, 2, m))
    work(:, 1, :) = f
    work(:, 2, :) = f
    CALL fwfft_y('Wave', work(:, 2, m), n)
    f = work(:, 2, :)
    DEALLOCATE (work)
  END SUBROUTINE run_fwfft_allocatable_column
END MODULE qe_fft_section_probe
