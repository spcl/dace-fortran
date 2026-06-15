SUBROUTINE run(n, m, k)
  ! Bounds-remap pointer view whose TARGET is an ALLOCATABLE array.
  !
  ! This is the QE ``vexx_bp_k_gpu`` ``prhoc_d`` shape (Gate H): a 1-D
  ! POINTER rebound to a 2-D contiguous column-section of an ALLOCATABLE
  ! target.  flang lowers the section designate over a LOADED descriptor
  ! box (``fir.load %rhoc#0``), so the bridge's bounds-remap-view source
  ! trace must walk through the ``fir.load`` to reach the ``rhoc`` declare.
  ! Without the ``fir.LoadOp`` hop the trace stops at the load,
  ! ``bounds_remap_source`` stays empty, ``bounds_remap_view`` is left
  ! false, and the rebind is mis-lowered as a scalar copy.
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  COMPLEX(8), ALLOCATABLE, TARGET :: rhoc(:, :)
  COMPLEX(8), POINTER :: prhoc(:)
  ALLOCATE(rhoc(n, m))
  prhoc(1 : n*k) => rhoc(:, 1:k)
  prhoc(1) = (1.0d0, 0.0d0)
  DEALLOCATE(rhoc)
END SUBROUTINE
