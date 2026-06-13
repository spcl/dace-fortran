! MPI_Alltoall frontend-recognition probe.
!
! Standard 8-arg MPI Fortran ABI; the bridge maps this to
! :class:`dace.libraries.mpi.nodes.alltoall.Alltoall`.
!
! Uses ``external``-declared MPI symbols + ``parameter`` constants so
! the file lowers via flang-new-21 without ``mpi.mod`` -- same
! convention as ``mpi_sendrecv_test.py`` (the bridge sees an opaque
! ``fir.call @_QPmpi_alltoall`` either way).
SUBROUTINE run_alltoall(n, sendbuf, recvbuf)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN)  :: sendbuf(n)
  REAL(8), INTENT(OUT) :: recvbuf(n)
  INTEGER :: ierr, sendcount, recvcount
  INTEGER, PARAMETER :: MPI_COMM_WORLD = 0
  INTEGER, PARAMETER :: MPI_DOUBLE_PRECISION = 17
  EXTERNAL :: MPI_Alltoall
  sendcount = n
  recvcount = n
  CALL MPI_Alltoall(sendbuf, sendcount, MPI_DOUBLE_PRECISION, &
                    recvbuf, recvcount, MPI_DOUBLE_PRECISION, &
                    MPI_COMM_WORLD, ierr)
END SUBROUTINE run_alltoall
