! Real-MPI vertex halo exchange for the ocean nonlinear_coriolis_3d_fast_scalar
! 2-rank e2e.  The ocean kernel's real sync is
! ``mo_sync::sync_patch_array_3d_dp(typ, p_patch, arr, lacc)`` -- a no-op stub
! taking a t_patch STRUCT.  t_patch is not marshalable across the keep_external
! C ABI (nested records + pointer members), and the halo does not actually need
! the geometry-only stub struct: it needs the MPI communicator.  So (mirroring
! the dummy 2-rank halo test) the kernel's sync CALL is rewritten to this
! patch-free, comm-carrying routine, registered as a ``keep_external``.
!
! The field is passed EXPLICIT-shape (``arr(n1, n2, n3)`` with the extents as
! leading scalar args) rather than assumed-shape: the kernel's ``vort_v`` is an
! explicit-shape dummy with a derived-type-component bound
! (``patch_3d % p_patch_2d(1) % nblks_v``), and passing that to an assumed-shape
! dummy makes gfortran build a copy-in/copy-out temporary whose writeback of the
! halo block is only partially reflected back into the caller's buffer -- so the
! halo update silently vanishes.  Explicit-shape passes the contiguous array by
! reference, so ``arr(:, :, 2) = recv`` writes the caller's memory directly.
!
! ``comm`` is a Fortran ``MPI_Fint`` integer handle (OpenMPI = C ``int``);
! ``mpi4py.MPI.Comm.py2f()`` hands it out, and the C ABI carries it byte-for-byte
! (no ``MPI_Comm_f2c`` on the Fortran side).  MPI constants are hard-coded to
! OpenMPI values to avoid the flang-vs-OpenMPI ``.mod`` dependency at SDFG-bridge
! build time.
module mo_vort_sync
  use iso_c_binding
  implicit none
  ! OpenMPI values; matches mpi4py.MPI.Comm.py2f().
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_SIZE = 6
contains
  ! Original Fortran vertex halo sync -- NO bind(c).  The ocean kernel CALLs
  ! this in place of sync_patch_array_3d_dp.  ``typ`` is the ICON sync selector
  ! (=3 = SYNC_V vertices), kept for signature faithfulness.  ``arr`` is a
  ! vertex field ``(n1, n2, n3)`` = ``(nproma, n_zlev, nblks_v)`` with block 1 =
  ! owned, block 2 = halo; the swap lands the neighbour's owned block 1 into this
  ! rank's halo block 2.  Single-rank (``size <= 1``) is a real no-op, matching
  ! ICON.
  subroutine vort_v_halo_sync(typ, n1, n2, n3, arr, comm)
    integer(c_int), intent(in) :: typ, n1, n2, n3
    real(c_double), intent(inout) :: arr(n1, n2, n3)
    integer, intent(in) :: comm
    integer :: rank, size_, neigh, ierr, count
    integer :: status_arr(MPI_STATUS_SIZE)
    real(c_double), allocatable :: send_buf(:, :), recv_buf(:, :)
    external :: MPI_Comm_rank, MPI_Comm_size, MPI_Sendrecv
    call MPI_Comm_rank(comm, rank, ierr)
    call MPI_Comm_size(comm, size_, ierr)
    if (size_ <= 1) return
    neigh = 1 - rank
    count = n1 * n2
    allocate(send_buf(n1, n2))
    allocate(recv_buf(n1, n2))
    send_buf = arr(:, :, 1)
    call MPI_Sendrecv(send_buf, count, MPI_DOUBLE_PRECISION, &
                      neigh, typ, &
                      recv_buf, count, MPI_DOUBLE_PRECISION, &
                      neigh, typ, &
                      comm, status_arr, ierr)
    arr(:, :, 2) = recv_buf
    deallocate(send_buf, recv_buf)
  end subroutine vort_v_halo_sync

  ! ``bind(c)`` wrapper the SDFG lib-node invokes.  Receives the flat vort_v
  ! pointer + the three extents (leading scalar args) + comm as a Fortran
  ! MPI_Fint integer; rebuilds a contiguous pointer descriptor and forwards it to
  ! the explicit-shape routine.
  subroutine vort_v_halo_sync_c(typ, n1, n2, n3, arr_p, comm) &
    bind(c, name='vort_v_halo_sync_c')
    integer(c_int), value :: typ, n1, n2, n3, comm
    type(c_ptr), value :: arr_p
    real(c_double), pointer :: a(:, :, :)
    call c_f_pointer(arr_p, a, [n1, n2, n3])
    call vort_v_halo_sync(typ, n1, n2, n3, a, comm)
  end subroutine vort_v_halo_sync_c
end module mo_vort_sync
