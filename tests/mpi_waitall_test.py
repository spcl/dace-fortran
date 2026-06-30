"""``MPI_Waitall`` over a request array lowers to a ``dace.libraries.mpi``
``Waitall`` library node.

The bridge recognises ``MPI_Waitall(count, array_of_requests, statuses, ierr)``
and emits a ``Waitall`` node (waiting on the whole request array; the count is
derived from the request memlet), threaded after the matching Isend/Irecv
producers through the shared per-request opaque transient -- the array analogue
of ``mpi_wait`` -> ``Wait``.  Mirrors ``tests/sync_devirt_mpi_libnode_test.py``'s
libnode-presence assertion (a lowering test; no ranks required).
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

#: Nonblocking exchange whose two requests live in an array, completed by a
#: single ``MPI_Waitall`` -- the ICON halo ``mpi_waitall(p_irequest, p_request,
#: ...)`` shape in miniature.
_WAITALL = """
module waitall_mod
contains
subroutine waitall_ring(buf, rbuf, n, dst, src, tag)
  implicit none
  integer, intent(in) :: n, dst, src, tag
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr, requests(2)
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUSES_IGNORE = -1
  external :: MPI_Isend, MPI_Irecv, MPI_Waitall
  call MPI_Irecv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, MPI_COMM_WORLD, requests(1), ierr)
  call MPI_Isend(buf, n, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, requests(2), ierr)
  call MPI_Waitall(2, requests, MPI_STATUSES_IGNORE, ierr)
end subroutine waitall_ring
end module waitall_mod
"""


def test_mpi_waitall_lowers_to_waitall_libnode(tmp_path: Path):
    """``MPI_Waitall`` lowers to a ``dace.libraries.mpi`` ``Waitall`` node, beside
    the Isend/Irecv producers -- the only MPI nodes, none left external."""
    from dace.libraries.mpi.nodes.node import MPINode

    sdfg = build_sdfg(_WAITALL, tmp_path / "sdfg", name="waitall", entry="waitall_mod::waitall_ring").build()
    mpi = sorted({type(n).__name__ for n, _ in sdfg.all_nodes_recursive() if isinstance(n, MPINode)})
    assert mpi == ["Irecv", "Isend", "Waitall"], f"expected Irecv/Isend/Waitall MPI libnodes, got {mpi}"
