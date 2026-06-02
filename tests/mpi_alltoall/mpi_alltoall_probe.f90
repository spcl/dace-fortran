! MPI_Alltoall frontend-recognition probe.
! Standard 8-arg MPI Fortran ABI; the bridge maps this to
! :class:`dace.libraries.mpi.nodes.alltoall.Alltoall`.
MODULE mpi_alltoall_probe
  USE mpi
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = KIND(1.0D0)
CONTAINS
  SUBROUTINE run_alltoall(n, sendbuf, recvbuf)
    INTEGER, INTENT(IN) :: n
    REAL(dp), INTENT(IN)  :: sendbuf(n)
    REAL(dp), INTENT(OUT) :: recvbuf(n)
    INTEGER :: ierr, sendcount, recvcount
    sendcount = n
    recvcount = n
    CALL MPI_Alltoall(sendbuf, sendcount, MPI_DOUBLE_PRECISION, &
                      recvbuf, recvcount, MPI_DOUBLE_PRECISION, &
                      MPI_COMM_WORLD, ierr)
  END SUBROUTINE run_alltoall
END MODULE mpi_alltoall_probe
