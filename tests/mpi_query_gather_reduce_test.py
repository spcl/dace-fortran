"""P2b MPI reachability: the point-to-point/collective recognizer now also
lowers the *query* and *rooted-collective* family -- ``MPI_Comm_rank`` /
``MPI_Comm_size`` / ``MPI_Comm_split`` / ``MPI_Abort`` / ``MPI_Gather`` /
``MPI_Gatherv`` / ``MPI_Reduce`` -- to their DaCe ``dace.libraries.mpi`` nodes,
plus renders an array-ELEMENT ``dest`` / ``src`` (``neighbors(k)``) as a single
-element subset instead of collapsing onto the whole array.

Structural tests: build + validate, assert the node is wired and no opaque
``call`` survives.  No ``mpirun`` needed.  The communicator is a local
``MPI_COMM_WORLD`` ``parameter`` (a synthetic runtime scalar the bridge threads
as ``_comm``); reduction ops ride ``use mpi``-style module handles so the op
name survives to the builder.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_MPI_OP_MODULE = """
module mpiops
  implicit none
  integer :: mpi_sum = 1
  integer :: mpi_max = 2
end module
"""


def _first(sdfg, cls):
    return next(n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls))


def _all(sdfg, cls):
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, cls)]


def _build(src, tmp_path, name, entry):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_comm_rank_size_query(tmp_path: Path):
    """``MPI_Comm_rank`` / ``MPI_Comm_size`` write their result into a Fortran
    integer scalar via the ``_rank`` / ``_size`` output connectors -- this is the
    exact pattern the inlined ocean ``solve_free_sfc`` needs (``mpi_comm_size``
    into ``p_pe`` / ``num_work_procs``)."""
    from dace.libraries.mpi.nodes.comm_rank import CommRank
    from dace.libraries.mpi.nodes.comm_size import CommSize

    src = """
subroutine query(myrank, nprocs)
  implicit none
  integer, intent(out) :: myrank, nprocs
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  external :: MPI_Comm_rank, MPI_Comm_size
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
end subroutine query
"""
    sdfg = _build(src, tmp_path, "query", "query")
    rank, size = _first(sdfg, CommRank), _first(sdfg, CommSize)
    assert "_rank" in rank.out_connectors
    assert "_size" in size.out_connectors
    # The MPI_COMM_WORLD parameter threads in as a runtime communicator.
    assert "_comm" in rank.in_connectors and "_comm" in size.in_connectors
    assert "call" not in [vars(n).get("kind") for n, _ in sdfg.all_nodes_recursive()]
    sdfg.validate()


def test_reduce_to_root(tmp_path: Path):
    """``MPI_Reduce`` -> DaCe ``Reduce`` with the resolved op and a ``_root``
    connector (rooted, unlike ``Allreduce``)."""
    from dace.libraries.mpi.nodes.reduce import Reduce

    src = _MPI_OP_MODULE + """
subroutine red(buf, rbuf, n, root)
  use mpiops
  implicit none
  integer, intent(in) :: n, root
  real(8), intent(in) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Reduce
  call MPI_Reduce(buf, rbuf, n, MPI_DOUBLE_PRECISION, mpi_sum, root, MPI_COMM_WORLD, ierr)
end subroutine red
"""
    sdfg = _build(src, tmp_path, "red", "red")
    node = _first(sdfg, Reduce)
    assert node.op == "MPI_SUM"
    assert set(node.in_connectors) >= {"_inbuffer", "_root", "_comm"}
    assert "_outbuffer" in node.out_connectors
    sdfg.validate()


def test_gather_to_root(tmp_path: Path):
    """``MPI_Gather`` -> DaCe ``Gather`` (fixed per-rank count, rooted)."""
    from dace.libraries.mpi.nodes.gather import Gather

    src = """
subroutine gath(sbuf, rbuf, n, root)
  implicit none
  integer, intent(in) :: n, root
  real(8), intent(in) :: sbuf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Gather
  call MPI_Gather(sbuf, n, MPI_DOUBLE_PRECISION, rbuf, n, MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD, ierr)
end subroutine gath
"""
    sdfg = _build(src, tmp_path, "gath", "gath")
    node = _first(sdfg, Gather)
    assert set(node.in_connectors) >= {"_inbuffer", "_root", "_comm"}
    assert "_outbuffer" in node.out_connectors
    sdfg.validate()


def test_send_to_array_element_dest(tmp_path: Path):
    """``MPI_Send(buf, n, dt, neighbors(k), tag, comm)`` reads the destination
    rank from element ``neighbors(k-1)`` (DaCe 0-based), NOT the whole
    ``neighbors`` array -- pins the subscript-drop miscompile (audit crit#4)."""
    from dace.libraries.mpi.nodes.isend import Isend
    from dace.libraries.mpi.nodes.send import Send

    src = """
subroutine ring(buf, n, neighbors, k)
  implicit none
  integer, intent(in) :: n, k
  integer, intent(in) :: neighbors(4)
  real(8), intent(in) :: buf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Send
  call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, neighbors(k), 0, MPI_COMM_WORLD, ierr)
end subroutine ring
"""
    sdfg = _build(src, tmp_path, "ring", "ring")
    sends = [(n, p) for n, p in sdfg.all_nodes_recursive() if isinstance(n, (Send, Isend))]
    assert len(sends) == 1, f"expected one Send, got {len(sends)}"
    send, parent = sends[0]
    # Find the memlet feeding the destination connector; its subset must be a
    # single element (k-1), not the full 0:4 neighbors array.
    dest_subsets = [
        e.data.subset for e in parent.in_edges(send)
        if e.dst_conn in ("_dest", "_src") and e.data is not None and "neighbors" in str(e.data.data)
    ]
    assert dest_subsets, "no neighbors-fed dest/src memlet found on the Send node"
    for ss in dest_subsets:
        assert str(ss) != "0:4", f"dest collapsed onto whole neighbors array: {ss}"
    sdfg.validate()
