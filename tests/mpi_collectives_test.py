"""MPI collectives (``MPI_Barrier`` / ``MPI_Allreduce`` / ``MPI_Bcast``) ->
DaCe ``dace.libraries.mpi`` library nodes -- no MPI op should be left external.

The bridge recognises the opaque ``fir.call @_QPmpi_*`` and the positional MPI
ABI; the builder lowers it to the matching DaCe node (``Barrier`` is added here;
``Allreduce`` / ``Bcast`` already exist).  Structural tests: build + validate,
assert the nodes are wired -- no ``mpirun`` needed.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_COLLECTIVES = """
subroutine collectives(buf, rbuf, n, root)
  implicit none
  integer, intent(in) :: n, root
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_SUM = 1
  external :: MPI_Barrier, MPI_Allreduce, MPI_Bcast
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  call MPI_Allreduce(buf, rbuf, n, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  call MPI_Bcast(buf, n, MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD, ierr)
end subroutine collectives
"""


def test_collectives_lower_to_mpi_libnodes(tmp_path: Path):
    """``MPI_Barrier`` / ``MPI_Allreduce`` / ``MPI_Bcast`` become DaCe ``Barrier``
    / ``Allreduce`` / ``Bcast`` nodes (no opaque ``call`` left), and validate."""
    from dace.libraries.mpi.nodes import Allreduce, Barrier, Bcast

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_COLLECTIVES, sdfg_dir, name="collectives", entry="collectives").build()

    nodes = {
        cls: [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls)]
        for cls in (Barrier, Allreduce, Bcast)
    }
    assert len(nodes[Barrier]) == 1 and len(nodes[Allreduce]) == 1 and len(nodes[Bcast]) == 1, \
        f"expected one of each collective, got { {c.__name__: len(v) for c, v in nodes.items()} }"

    assert nodes[Allreduce][0].op == "MPI_SUM"
    # The Fortran communicator is threaded into every collective via a CommF2c
    # dataflow node feeding a ``_comm`` connector (bridge feature 93cc5f2).
    assert set(nodes[Allreduce][0].in_connectors) == {"_inbuffer", "_comm"}
    assert set(nodes[Bcast][0].in_connectors) == {"_inbuffer", "_root", "_comm"}
    assert set(nodes[Barrier][0].in_connectors) == {"_comm"}  # only the communicator

    assert "call" not in [getattr(n, "kind", None) for n, _ in sdfg.all_nodes_recursive()]
    sdfg.validate()
