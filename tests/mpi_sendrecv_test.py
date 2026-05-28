"""MPI ``MPI_Send`` / ``MPI_Recv`` (default communicator) -> DaCe
``dace.libraries.mpi`` ``Send`` / ``Recv`` library nodes.

Flang emits no MLIR ``mpi`` dialect; ``call MPI_Send(...)`` lowers to
an opaque ``fir.call @_QPmpi_send(...)``.  The C++ bridge recognises
the callee and the positional MPI ABI
(``buf, count, datatype, dest|src, tag, comm, [status,] ierr``) and
the Python builder lowers it to the DaCe Send/Recv node (``_buffer`` /
``_dest``|``_src`` / ``_tag`` connectors; count from the buffer
memlet; MPI datatype from the buffer descriptor; communicator
``MPI_COMM_WORLD``).

These are *structural* tests (build + validate the SDFG, assert the
right library nodes are wired) so they run in the normal sweep with
no MPI runtime.  A numeric multi-rank ``mpirun`` end-to-end check is a
separate ``@pytest.mark.mpi`` concern.
"""

from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ``external`` MPI decls + ``parameter`` constants so the program needs
# no ``mpi.mod`` / MPI install to lower (the bridge sees the same
# opaque ``fir.call @_QPmpi_*`` either way).
_SENDRECV = """
subroutine sendrecv(buf, rbuf, n, dst, src, tag)
  implicit none
  integer, intent(in) :: n, dst, src, tag
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_COMM_WORLD = 0
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_IGNORE = -1
  external :: MPI_Send, MPI_Recv
  call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, ierr)
  call MPI_Recv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, MPI_COMM_WORLD, &
                MPI_STATUS_IGNORE, ierr)
end subroutine sendrecv
"""

# A runtime/user communicator (a dummy ``comm`` argument) is threaded
# into the libnode as an ``opaque(MPI_Comm)`` ``_comm`` connector; the
# Fortran integer handle is retyped on the SDFG signature so the
# generated binding wrapper can ``MPI_Comm_f2c`` it.
_USER_COMM = """
subroutine sr_usercomm(buf, n, dst, tag, comm)
  implicit none
  integer, intent(in) :: n, dst, tag, comm
  real(8), intent(inout) :: buf(n)
  integer :: ierr
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  external :: MPI_Send
  call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, dst, tag, comm, ierr)
end subroutine sr_usercomm
"""


def _build(src: str, tmp: Path, name: str, entry: str):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name=name, entry=entry).build()


def test_send_recv_lower_to_mpi_libnodes(tmp_path: Path):
    """``MPI_Send`` / ``MPI_Recv`` on MPI_COMM_WORLD become DaCe
    ``Send`` / ``Recv`` nodes with the canonical connectors, and the
    SDFG validates."""
    from dace.libraries.mpi.nodes.recv import Recv
    from dace.libraries.mpi.nodes.send import Send

    sdfg = _build(_SENDRECV, tmp_path, "sendrecv", "_QPsendrecv")

    sends = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Send)]
    recvs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Recv)]
    assert len(sends) == 1, f"expected 1 Send node, got {len(sends)}"
    assert len(recvs) == 1, f"expected 1 Recv node, got {len(recvs)}"

    assert set(sends[0].in_connectors) == {"_buffer", "_dest", "_tag"}
    assert not sends[0].out_connectors
    assert set(recvs[0].in_connectors) == {"_src", "_tag"}
    assert set(recvs[0].out_connectors) == {"_buffer"}

    # No leftover opaque ``call`` node for the recognised MPI calls.
    kinds = [getattr(n, "kind", None) for n, _ in sdfg.all_nodes_recursive()]
    assert "call" not in kinds

    sdfg.validate()


_NONBLOCKING = """
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
"""


def test_isend_irecv_wait_lower_to_mpi_libnodes(tmp_path: Path):
    """``MPI_Isend`` / ``MPI_Irecv`` / ``MPI_Wait`` become DaCe
    ``Isend`` / ``Irecv`` / ``Wait`` nodes; the non-blocking request is
    threaded producer->Wait through a synthesised
    ``opaque(MPI_Request)`` transient; each MPI call gets its own state
    so program order is enforced by interstate edges."""
    import dace
    from dace.libraries.mpi.nodes.irecv import Irecv
    from dace.libraries.mpi.nodes.isend import Isend
    from dace.libraries.mpi.nodes.wait import Wait

    sdfg = _build(_NONBLOCKING, tmp_path, "nbring", "_QPnbring")

    counts = {Isend: 0, Irecv: 0, Wait: 0}
    for nd, _ in sdfg.all_nodes_recursive():
        for cls in counts:
            if isinstance(nd, cls):
                counts[cls] += 1
    assert counts[Isend] == 1 and counts[Irecv] == 1 and counts[Wait] == 2

    reqs = {a for a in sdfg.arrays if a.startswith("_mpireq_")}
    assert reqs == {"_mpireq_sreq", "_mpireq_rreq"}
    for r in reqs:
        d = sdfg.arrays[r]
        assert d.transient and d.dtype == dace.dtypes.opaque("MPI_Request")

    # One state per MPI call (4) + the entry state -> interstate-edge
    # ordering of the side-effecting MPI nodes.
    assert len(list(sdfg.all_states())) >= 5
    sdfg.validate()


def test_runtime_communicator_lowers_to_comm_connector(tmp_path: Path):
    """A non-default (runtime dummy) communicator is supported via a
    :class:`FortranProcessGrid` descriptor: the ``Send`` node gains an
    ``opaque(MPI_Comm)`` ``_comm`` in-connector wired to
    ``dace_user_pgrid``; the pgrid's parent comm is the
    ``dace_user_comm`` symbol the bindings wrapper populates from
    ``MPI_Comm_f2c`` at ``__dace_init`` time.  The original Fortran
    integer ``comm`` dummy is dropped from ``sdfg.arrays`` -- the C
    MPI_Comm value flows through the pgrid's parent-comm symbol, not
    through a kernel call arg."""
    import dace
    from dace.libraries.mpi.nodes.send import Send
    from dace_fortran.data import FortranProcessGrid

    sdfg = _build(_USER_COMM, tmp_path, "sr_usercomm", "_QPsr_usercomm")

    sends = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Send)]
    assert len(sends) == 1, f"expected 1 Send node, got {len(sends)}"
    assert '_comm' in sends[0].in_connectors

    # The orphan Fortran ``integer`` comm dummy was removed -- a
    # FortranProcessGrid descriptor took its place.
    assert 'comm' not in sdfg.arrays, (
        "the Fortran integer ``comm`` dummy should be dropped from "
        "sdfg.arrays once the pgrid takes over wiring")
    assert 'dace_user_pgrid' in sdfg.arrays
    pgrid = sdfg.arrays['dace_user_pgrid']
    assert isinstance(pgrid, FortranProcessGrid)
    assert pgrid.parent_comm_symbol == 'dace_user_comm'

    # Init/exit code references our pgrid (via state).
    assert 'dace_user_pgrid' in sdfg.init_code['frame'].code
    assert 'dace_user_pgrid' in sdfg.exit_code['frame'].code

    # ``dace_user_comm`` is now an opaque(MPI_Comm) symbol the bindings
    # wrapper must populate from ``MPI_Comm_f2c``.
    assert 'dace_user_comm' in sdfg.symbols
    assert isinstance(sdfg.symbols['dace_user_comm'], dace.dtypes.opaque)
    assert sdfg.symbols['dace_user_comm'].ctype == 'MPI_Comm'

    sdfg.validate()
