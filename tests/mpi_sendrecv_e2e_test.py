"""E2E: Fortran MPI_Send/Recv and Isend/Irecv/Wait lower to dace.libraries.mpi.
Ring exchange must round-trip: rank r sends r, must receive (r-1) % size. mpirun -n >=2.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_on_root, build_sdfg, have_flang

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
]

_RING = """
module ring_mod
contains
subroutine ring(buf, rbuf, n, dst, src, tag, rank)
  implicit none
  integer, intent(in) :: n, dst, src, tag, rank
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_IGNORE = -1
  external :: MPI_Send, MPI_Recv
  if (mod(rank, 2) == 0) then
    call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, ierr)
    call MPI_Recv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, MPI_COMM_WORLD, &
                  MPI_STATUS_IGNORE, ierr)
  else
    call MPI_Recv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, MPI_COMM_WORLD, &
                  MPI_STATUS_IGNORE, ierr)
    call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, ierr)
  end if
end subroutine ring
end module ring_mod
"""


@pytest.mark.mpi
def test_ring_send_recv_numeric(tmp_path: Path):
    from mpi4py import MPI
    from dace.sdfg import utils

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("MPI Send/Recv e2e needs >= 2 ranks (mpirun --oversubscribe -n 2 ...)")

    # Build on rank 0 only; distributed_compile shares the artifact (avoids per-rank build races/hangs).
    def _build_ring():
        s = build_sdfg(_RING, tmp_path / "sdfg", name="ring", entry="ring_mod::ring").build()
        s.name = "mpi_ring"
        return s

    sdfg = build_on_root(comm, _build_ring, broadcast=False)
    func = utils.distributed_compile(sdfg, comm)

    n = 8
    buf = np.full(n, float(rank), dtype=np.float64, order="F")
    rbuf = np.zeros(n, dtype=np.float64, order="F")
    dst = (rank + 1) % size
    src = (rank - 1 + size) % size

    func(buf=buf, rbuf=rbuf, n=n, dst=dst, src=src, tag=7, rank=rank)

    expected = float((rank - 1 + size) % size)
    np.testing.assert_allclose(rbuf, np.full(n, expected, dtype=np.float64))


_NB_RING = """
module nbring_mod
contains
subroutine nbring(buf, rbuf, n, dst, src, tag)
  implicit none
  integer, intent(in) :: n, dst, src, tag
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr, sreq, rreq
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_IGNORE = -1
  external :: MPI_Isend, MPI_Irecv, MPI_Wait
  call MPI_Irecv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, MPI_COMM_WORLD, rreq, ierr)
  call MPI_Isend(buf, n, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, sreq, ierr)
  call MPI_Wait(rreq, MPI_STATUS_IGNORE, ierr)
  call MPI_Wait(sreq, MPI_STATUS_IGNORE, ierr)
end subroutine nbring
end module nbring_mod
"""


@pytest.mark.mpi
def test_nonblocking_ring_numeric(tmp_path: Path):
    """Nonblocking ring (Irecv+Isend+Wait+Wait): rank r must receive (r-1)%size from its predecessor."""
    from mpi4py import MPI
    from dace.sdfg import utils

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        pytest.skip("MPI Isend/Irecv e2e needs >= 2 ranks (mpirun --oversubscribe -n 2 ...)")

    def _build_nbring():
        s = build_sdfg(_NB_RING, tmp_path / "sdfg", name="nbring", entry="nbring_mod::nbring").build()
        s.name = "mpi_nbring"
        return s

    sdfg = build_on_root(comm, _build_nbring, broadcast=False)
    func = utils.distributed_compile(sdfg, comm)

    n = 8
    buf = np.full(n, float(rank), dtype=np.float64, order="F")
    rbuf = np.zeros(n, dtype=np.float64, order="F")
    func(buf=buf, rbuf=rbuf, n=n, dst=(rank + 1) % size, src=(rank - 1 + size) % size, tag=7)

    np.testing.assert_allclose(rbuf, np.full(n, float((rank - 1 + size) % size), dtype=np.float64))


if __name__ == "__main__":
    test_ring_send_recv_numeric(Path("/tmp/mpi_ring_e2e"))
    test_nonblocking_ring_numeric(Path("/tmp/mpi_nbring_e2e"))
