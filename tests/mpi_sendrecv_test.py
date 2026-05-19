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

# A runtime/user communicator (a dummy ``comm`` argument) is not the
# default; Phase-1 must reject it loudly rather than silently treat it
# as MPI_COMM_WORLD.
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


def test_runtime_communicator_is_rejected(tmp_path: Path):
    """A non-default (runtime dummy) communicator must fail loudly --
    Phase 1 supports only MPI_COMM_WORLD; silently treating a user
    communicator as WORLD would miscompute."""
    with pytest.raises(Exception, match="(?i)communicator|MPI_COMM_WORLD"):
        _build(_USER_COMM, tmp_path, "sr_usercomm", "_QPsr_usercomm")
