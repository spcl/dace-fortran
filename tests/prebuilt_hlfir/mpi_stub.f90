! flang-compatible MPI stub.  flang-new-21 doesn't ship a built-in
! ``mpi.mod`` (only gfortran does, under
! ``/usr/lib/.../fortran/gfortran-mod-*/openmpi/mpi.mod``, in a
! binary format flang cannot consume).  We provide just enough of the
! ``mpi`` module's surface for flang to type-check the project --
! every entity the project ``use mpi``-imports is declared, but the
! procedure bodies are empty.  This file is consumed by
! ``CMakeLists.txt`` 's ``emit_hlfir`` target only; the real
! executable is linked against the system MPI as usual.
module mpi
  use iso_c_binding, only: c_int
  implicit none
  integer(c_int), parameter :: MPI_COMM_WORLD = 0
  integer(c_int), parameter :: MPI_DOUBLE_PRECISION = 17
  integer(c_int), parameter :: MPI_PROC_NULL = -1
  integer(c_int), parameter :: MPI_STATUS_SIZE = 6
contains
  subroutine MPI_Init(ierr)
    integer, intent(out) :: ierr
    ierr = 0
  end subroutine MPI_Init

  subroutine MPI_Finalize(ierr)
    integer, intent(out) :: ierr
    ierr = 0
  end subroutine MPI_Finalize

  subroutine MPI_Comm_rank(comm, rank, ierr)
    integer, intent(in) :: comm
    integer, intent(out) :: rank, ierr
    rank = 0
    ierr = 0
  end subroutine MPI_Comm_rank

  subroutine MPI_Comm_size(comm, sz, ierr)
    integer, intent(in) :: comm
    integer, intent(out) :: sz, ierr
    sz = 1
    ierr = 0
  end subroutine MPI_Comm_size

  subroutine MPI_Sendrecv(sbuf, scount, sdtype, dest, stag, &
                          rbuf, rcount, rdtype, src,  rtag, &
                          comm, status, ierr)
    real(8), intent(in)  :: sbuf(*)
    real(8), intent(out) :: rbuf(*)
    integer, intent(in)  :: scount, sdtype, dest, stag
    integer, intent(in)  :: rcount, rdtype, src,  rtag, comm
    integer, intent(out) :: status(*), ierr
    ierr = 0
  end subroutine MPI_Sendrecv
end module mpi
