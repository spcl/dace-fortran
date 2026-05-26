! flang-compatible ``mpi`` module stub for emitting ICON HLFIR.
!
! flang-new-21 ships no ``mpi.mod``; the system one (OpenMPI, under
! ``/usr/lib/.../fortran/gfortran-mod-*/openmpi/mpi.mod``) is a gfortran
! binary module flang cannot parse (``bad character (0x00)``).  ICON's
! ``mo_mpi`` / ``mo_packed_message`` / ``mo_real_timer`` /
! ``mo_reorder_info`` all ``USE mpi`` (we keep MPI on so the dynamical
! core's halo-exchange / collective structure survives into the SDFG for
! the MPI library nodes), so the bridge needs a flang-readable ``mpi.mod``.
!
! This declares the surface those TUs import: the named constants
! (datatypes, reduction ops, communicators, sentinels) and the procedures
! they call.  Choice-buffer arguments use ``type(*), dimension(..)`` so a
! call with any buffer type and rank type-checks -- mirroring how the real
! ``mpi`` (mpif.h) module leaves choice buffers without an explicit
! interface.  The bodies are empty: only the interface is needed to emit
! HLFIR; the real executable still links the system MPI.
module mpi
  use iso_c_binding, only: c_intptr_t
  implicit none
  public

  ! Address/offset kinds (used as ``integer(MPI_ADDRESS_KIND) :: x``).
  integer, parameter :: MPI_ADDRESS_KIND = c_intptr_t
  integer, parameter :: MPI_OFFSET_KIND = c_intptr_t
  integer, parameter :: MPI_COUNT_KIND = c_intptr_t

  ! Communicators / groups / sentinels.
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_COMM_SELF = 1
  integer, parameter :: MPI_COMM_NULL = 2
  integer, parameter :: MPI_GROUP_NULL = 3
  integer, parameter :: MPI_GROUP_EMPTY = 4
  integer, parameter :: MPI_REQUEST_NULL = 5
  integer, parameter :: MPI_OP_NULL = 6
  integer, parameter :: MPI_INFO_NULL = 7
  integer, parameter :: MPI_DATATYPE_NULL = 8
  integer, parameter :: MPI_ERRHANDLER_NULL = 9

  integer, parameter :: MPI_SUCCESS = 0
  integer, parameter :: MPI_UNDEFINED = -32766
  integer, parameter :: MPI_ANY_SOURCE = -1
  integer, parameter :: MPI_ANY_TAG = -1
  integer, parameter :: MPI_PROC_NULL = -2
  integer, parameter :: MPI_ROOT = -3

  ! Status indexing.
  integer, parameter :: MPI_STATUS_SIZE = 6
  integer, parameter :: MPI_SOURCE = 1
  integer, parameter :: MPI_TAG = 2
  integer, parameter :: MPI_ERROR = 3
  integer, parameter :: MPI_MAX_ERROR_STRING = 256
  integer, parameter :: MPI_MAX_PROCESSOR_NAME = 256
  integer, parameter :: MPI_MAX_LIBRARY_VERSION_STRING = 256

  ! Datatypes.
  integer, parameter :: MPI_BYTE = 1, MPI_PACKED = 2, MPI_CHARACTER = 3
  integer, parameter :: MPI_CHAR = 4, MPI_INTEGER = 5, MPI_INTEGER8 = 6
  integer, parameter :: MPI_REAL = 7, MPI_REAL4 = 8, MPI_REAL8 = 9
  integer, parameter :: MPI_DOUBLE_PRECISION = 10, MPI_LOGICAL = 11
  integer, parameter :: MPI_COMPLEX = 12, MPI_DOUBLE_COMPLEX = 13
  integer, parameter :: MPI_2INTEGER = 14, MPI_2REAL = 15
  integer, parameter :: MPI_2DOUBLE_PRECISION = 16

  ! Reduction operations.
  integer, parameter :: MPI_SUM = 1, MPI_MAX = 2, MPI_MIN = 3
  integer, parameter :: MPI_PROD = 4, MPI_LAND = 5, MPI_LOR = 6
  integer, parameter :: MPI_BAND = 7, MPI_BOR = 8, MPI_MAXLOC = 9
  integer, parameter :: MPI_MINLOC = 10

  ! Datatype typeclasses (for MPI_Type_match_size).
  integer, parameter :: MPI_TYPECLASS_INTEGER = 1
  integer, parameter :: MPI_TYPECLASS_REAL = 2
  integer, parameter :: MPI_TYPECLASS_COMPLEX = 3

  integer, parameter :: MPI_COMM_TYPE_SHARED = 1
  double precision, parameter :: MPI_WTICK = 1.0d-9

  ! Sentinel objects passed where a buffer / status array is expected.
  integer :: MPI_IN_PLACE = -1
  integer :: MPI_BOTTOM = 0
  integer :: MPI_STATUS_IGNORE(MPI_STATUS_SIZE) = 0
  integer :: MPI_STATUSES_IGNORE(MPI_STATUS_SIZE, 1) = 0

contains

  subroutine MPI_Init(ierr)
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Init

  subroutine MPI_Initialized(flag, ierr)
    logical, intent(out) :: flag
    integer, intent(out) :: ierr
    flag = .true.; ierr = MPI_SUCCESS
  end subroutine MPI_Initialized

  subroutine MPI_Finalize(ierr)
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Finalize

  subroutine MPI_Abort(comm, errorcode, ierr)
    integer, intent(in)  :: comm, errorcode
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Abort

  subroutine MPI_Barrier(comm, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Barrier

  double precision function MPI_Wtime()
    MPI_Wtime = 0.0d0
  end function MPI_Wtime

  subroutine MPI_Get_version(version, subversion, ierr)
    integer, intent(out) :: version, subversion, ierr
    version = 3; subversion = 1; ierr = MPI_SUCCESS
  end subroutine MPI_Get_version

  subroutine MPI_Get_library_version(version, resultlen, ierr)
    character(len=*), intent(out) :: version
    integer, intent(out)          :: resultlen, ierr
    version = ''; resultlen = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Get_library_version

  subroutine MPI_Error_string(errorcode, string, resultlen, ierr)
    integer, intent(in)           :: errorcode
    character(len=*), intent(out) :: string
    integer, intent(out)          :: resultlen, ierr
    string = ''; resultlen = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Error_string

  ! ---- communicator / group queries ----------------------------------
  subroutine MPI_Comm_rank(comm, rank, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: rank, ierr
    rank = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_rank

  subroutine MPI_Comm_size(comm, sz, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: sz, ierr
    sz = 1; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_size

  subroutine MPI_Comm_dup(comm, newcomm, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: newcomm, ierr
    newcomm = comm; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_dup

  subroutine MPI_Comm_create(comm, group, newcomm, ierr)
    integer, intent(in)  :: comm, group
    integer, intent(out) :: newcomm, ierr
    newcomm = comm; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_create

  subroutine MPI_Comm_split(comm, color, key, newcomm, ierr)
    integer, intent(in)  :: comm, color, key
    integer, intent(out) :: newcomm, ierr
    newcomm = comm; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_split

  subroutine MPI_Comm_free(comm, ierr)
    integer, intent(inout) :: comm
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Comm_free

  subroutine MPI_Comm_group(comm, group, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: group, ierr
    group = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_group

  subroutine MPI_Comm_remote_size(comm, sz, ierr)
    integer, intent(in)  :: comm
    integer, intent(out) :: sz, ierr
    sz = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_remote_size

  subroutine MPI_Comm_test_inter(comm, flag, ierr)
    integer, intent(in)  :: comm
    logical, intent(out) :: flag
    integer, intent(out) :: ierr
    flag = .false.; ierr = MPI_SUCCESS
  end subroutine MPI_Comm_test_inter

  subroutine MPI_Intercomm_create(local_comm, local_leader, peer_comm, &
                                  remote_leader, tag, newintercomm, ierr)
    integer, intent(in)  :: local_comm, local_leader, peer_comm
    integer, intent(in)  :: remote_leader, tag
    integer, intent(out) :: newintercomm, ierr
    newintercomm = local_comm; ierr = MPI_SUCCESS
  end subroutine MPI_Intercomm_create

  subroutine MPI_Group_free(group, ierr)
    integer, intent(inout) :: group
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Group_free

  subroutine MPI_Group_incl(group, n, ranks, newgroup, ierr)
    integer, intent(in)  :: group, n, ranks(*)
    integer, intent(out) :: newgroup, ierr
    newgroup = group; ierr = MPI_SUCCESS
  end subroutine MPI_Group_incl

  subroutine MPI_Group_translate_ranks(group1, n, ranks1, group2, ranks2, ierr)
    integer, intent(in)  :: group1, n, ranks1(*), group2
    integer, intent(out) :: ranks2(*), ierr
    ranks2(1:n) = ranks1(1:n); ierr = MPI_SUCCESS
  end subroutine MPI_Group_translate_ranks

  ! ---- point-to-point -------------------------------------------------
  subroutine MPI_Send(buf, count, datatype, dest, tag, comm, ierr)
    type(*), dimension(..), intent(in) :: buf
    integer, intent(in)  :: count, datatype, dest, tag, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Send

  subroutine MPI_Recv(buf, count, datatype, source, tag, comm, status, ierr)
    type(*), dimension(..) :: buf
    integer, intent(in)  :: count, datatype, source, tag, comm
    integer              :: status(*)
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Recv

  subroutine MPI_Isend(buf, count, datatype, dest, tag, comm, request, ierr)
    type(*), dimension(..), intent(in) :: buf
    integer, intent(in)  :: count, datatype, dest, tag, comm
    integer, intent(out) :: request, ierr
    request = MPI_REQUEST_NULL; ierr = MPI_SUCCESS
  end subroutine MPI_Isend

  subroutine MPI_Irecv(buf, count, datatype, source, tag, comm, request, ierr)
    type(*), dimension(..) :: buf
    integer, intent(in)  :: count, datatype, source, tag, comm
    integer, intent(out) :: request, ierr
    request = MPI_REQUEST_NULL; ierr = MPI_SUCCESS
  end subroutine MPI_Irecv

  subroutine MPI_Sendrecv(sbuf, scount, sdtype, dest, stag, &
                          rbuf, rcount, rdtype, src,  rtag, &
                          comm, status, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, dest, stag
    integer, intent(in)  :: rcount, rdtype, src,  rtag, comm
    integer              :: status(*)
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Sendrecv

  subroutine MPI_Iprobe(source, tag, comm, flag, status, ierr)
    integer, intent(in)  :: source, tag, comm
    logical, intent(out) :: flag
    integer              :: status(*)
    integer, intent(out) :: ierr
    flag = .false.; ierr = MPI_SUCCESS
  end subroutine MPI_Iprobe

  subroutine MPI_Get_count(status, datatype, count, ierr)
    integer, intent(in)  :: status(*), datatype
    integer, intent(out) :: count, ierr
    count = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Get_count

  subroutine MPI_Wait(request, status, ierr)
    integer, intent(inout) :: request
    integer                :: status(*)
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Wait

  subroutine MPI_Waitall(count, requests, statuses, ierr)
    integer, intent(in)    :: count
    integer, intent(inout) :: requests(*)
    integer                :: statuses(*)
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Waitall

  subroutine MPI_Waitany(count, requests, index, status, ierr)
    integer, intent(in)    :: count
    integer, intent(inout) :: requests(*)
    integer, intent(out)   :: index
    integer                :: status(*)
    integer, intent(out)   :: ierr
    index = MPI_UNDEFINED; ierr = MPI_SUCCESS
  end subroutine MPI_Waitany

  subroutine MPI_Testall(count, requests, flag, statuses, ierr)
    integer, intent(in)    :: count
    integer, intent(inout) :: requests(*)
    logical, intent(out)   :: flag
    integer                :: statuses(*)
    integer, intent(out)   :: ierr
    flag = .true.; ierr = MPI_SUCCESS
  end subroutine MPI_Testall

  ! ---- collectives ----------------------------------------------------
  subroutine MPI_Bcast(buf, count, datatype, root, comm, ierr)
    type(*), dimension(..) :: buf
    integer, intent(in)  :: count, datatype, root, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Bcast

  subroutine MPI_Reduce(sbuf, rbuf, count, datatype, op, root, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: count, datatype, op, root, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Reduce

  subroutine MPI_Allreduce(sbuf, rbuf, count, datatype, op, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: count, datatype, op, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Allreduce

  subroutine MPI_Allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, rcount, rdtype, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Allgather

  subroutine MPI_Allgatherv(sbuf, scount, sdtype, rbuf, rcounts, displs, &
                            rdtype, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, rcounts(*), displs(*), rdtype, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Allgatherv

  subroutine MPI_Gather(sbuf, scount, sdtype, rbuf, rcount, rdtype, root, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, rcount, rdtype, root, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Gather

  subroutine MPI_Gatherv(sbuf, scount, sdtype, rbuf, rcounts, displs, &
                         rdtype, root, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, rcounts(*), displs(*), rdtype, root, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Gatherv

  subroutine MPI_Scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, rcount, rdtype, root, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Scatter

  subroutine MPI_Scatterv(sbuf, scounts, displs, sdtype, rbuf, rcount, &
                          rdtype, root, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scounts(*), displs(*), sdtype, rcount, rdtype, root, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Scatterv

  subroutine MPI_Alltoall(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scount, sdtype, rcount, rdtype, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Alltoall

  subroutine MPI_Alltoallv(sbuf, scounts, sdispls, sdtype, rbuf, rcounts, &
                           rdispls, rdtype, comm, ierr)
    type(*), dimension(..), intent(in) :: sbuf
    type(*), dimension(..)             :: rbuf
    integer, intent(in)  :: scounts(*), sdispls(*), sdtype
    integer, intent(in)  :: rcounts(*), rdispls(*), rdtype, comm
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Alltoallv

  ! ---- pack / datatypes -----------------------------------------------
  subroutine MPI_Pack(inbuf, incount, datatype, outbuf, outsize, position, comm, ierr)
    type(*), dimension(..), intent(in) :: inbuf
    type(*), dimension(..)             :: outbuf
    integer, intent(in)    :: incount, datatype, outsize, comm
    integer, intent(inout) :: position
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Pack

  subroutine MPI_Unpack(inbuf, insize, position, outbuf, outcount, datatype, comm, ierr)
    type(*), dimension(..), intent(in) :: inbuf
    type(*), dimension(..)             :: outbuf
    integer, intent(in)    :: insize, outcount, datatype, comm
    integer, intent(inout) :: position
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Unpack

  subroutine MPI_Pack_size(incount, datatype, comm, size, ierr)
    integer, intent(in)  :: incount, datatype, comm
    integer, intent(out) :: size, ierr
    size = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Pack_size

  subroutine MPI_Buffer_attach(buffer, size, ierr)
    type(*), dimension(..) :: buffer
    integer, intent(in)  :: size
    integer, intent(out) :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Buffer_attach

  subroutine MPI_Type_commit(datatype, ierr)
    integer, intent(inout) :: datatype
    integer, intent(out)   :: ierr
    ierr = MPI_SUCCESS
  end subroutine MPI_Type_commit

  subroutine MPI_Type_create_struct(count, blocklengths, displacements, &
                                    types, newtype, ierr)
    integer, intent(in)  :: count, blocklengths(*), types(*)
    integer(MPI_ADDRESS_KIND), intent(in) :: displacements(*)
    integer, intent(out) :: newtype, ierr
    newtype = MPI_DATATYPE_NULL; ierr = MPI_SUCCESS
  end subroutine MPI_Type_create_struct

  subroutine MPI_Type_get_extent(datatype, lb, extent, ierr)
    integer, intent(in)  :: datatype
    integer(MPI_ADDRESS_KIND), intent(out) :: lb, extent
    integer, intent(out) :: ierr
    lb = 0; extent = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Type_get_extent

  subroutine MPI_Type_match_size(typeclass, size, datatype, ierr)
    integer, intent(in)  :: typeclass, size
    integer, intent(out) :: datatype, ierr
    datatype = MPI_DATATYPE_NULL; ierr = MPI_SUCCESS
  end subroutine MPI_Type_match_size

  subroutine MPI_Sizeof(x, size, ierr)
    type(*), dimension(..), intent(in) :: x
    integer, intent(out) :: size, ierr
    size = 0; ierr = MPI_SUCCESS
  end subroutine MPI_Sizeof

end module mpi
