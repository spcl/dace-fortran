"""MPI_Send/MPI_Recv (default communicator) -> DaCe dace.libraries.mpi Send/Recv library nodes.

Flang lowers ``call MPI_Send(...)`` to an opaque fir.call; the bridge recognises the callee + positional MPI ABI (buf, count, datatype, dest|src, tag, comm, [status,] ierr) and lowers it to Send/Recv nodes.  Structural tests only (build + validate, assert wiring) -- no MPI runtime needed; numeric multi-rank mpirun checks are a separate @pytest.mark.mpi concern.
"""

from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# external MPI decls + parameter constants so the program needs no mpi.mod / MPI install to lower (bridge sees the same opaque fir.call either way).
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

# runtime/user communicator threads into the libnode as an opaque(MPI_Comm) _comm connector; the Fortran integer handle is retyped on the SDFG signature so the binding wrapper can MPI_Comm_f2c it.
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


def _expanded_mpi_call_codes(sdfg):
    """Expand the SDFG's MPI library nodes, return the CPP tasklet code strings issuing MPI_Send/MPI_Recv.

    The communicator is baked into the expansion's C code, not a node property, so verifying it without mpirun means inspecting the tasklet source.
    """
    import dace
    sdfg.expand_library_nodes()
    codes = []
    for nd, _ in sdfg.all_nodes_recursive():
        if isinstance(nd, dace.sdfg.nodes.Tasklet):
            code = nd.code.as_string or ""
            if "MPI_Send(" in code or "MPI_Recv(" in code:
                codes.append(code)
    return codes


def test_send_recv_lower_to_mpi_libnodes(tmp_path: Path):
    """MPI_Send/MPI_Recv on MPI_COMM_WORLD become DaCe Send/Recv nodes with canonical connectors; SDFG validates."""
    from dace.libraries.mpi.nodes.recv import Recv
    from dace.libraries.mpi.nodes.send import Send

    sdfg = _build(_SENDRECV, tmp_path, "sendrecv", "sendrecv")

    sends = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Send)]
    recvs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Recv)]
    assert len(sends) == 1, f"expected 1 Send node, got {len(sends)}"
    assert len(recvs) == 1, f"expected 1 Recv node, got {len(recvs)}"

    assert set(sends[0].in_connectors) == {"_buffer", "_dest", "_tag", "_comm"}
    assert not sends[0].out_connectors
    assert set(recvs[0].in_connectors) == {"_src", "_tag", "_comm"}
    assert set(recvs[0].out_connectors) == {"_buffer"}

    # No leftover opaque ``call`` node for the recognised MPI calls.
    kinds = [getattr(n, "kind", None) for n, _ in sdfg.all_nodes_recursive()]
    assert "call" not in kinds

    sdfg.validate()

    # communicator dataflow: default MPI_COMM_WORLD threads into Send/Recv via the _comm connector fed by a CommF2c node (93cc5f2), not hardcoded; no _grid without a user comm.
    codes = _expanded_mpi_call_codes(sdfg)
    assert len(codes) == 2, f"expected 1 Send + 1 Recv tasklet, got {len(codes)}"
    for code in codes:
        assert "_comm" in code, f"Send/Recv must use the threaded _comm connector: {code!r}"
        assert "_grid" not in code, f"no ``_grid`` should appear without a user comm: {code!r}"


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
    """MPI_Isend/MPI_Irecv/MPI_Wait become Isend/Irecv/Wait nodes; the request threads producer->Wait via an opaque(MPI_Request) transient, and each MPI call gets its own state so interstate edges enforce program order."""
    import dace
    from dace.libraries.mpi.nodes.irecv import Irecv
    from dace.libraries.mpi.nodes.isend import Isend
    from dace.libraries.mpi.nodes.wait import Wait

    sdfg = _build(_NONBLOCKING, tmp_path, "nbring", "nbring")

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

    # one state per MPI call (4) + entry state -> interstate-edge ordering of the side-effecting MPI nodes.
    assert len(list(sdfg.all_states())) >= 5
    sdfg.validate()


def test_runtime_communicator_lowers_to_comm_connector(tmp_path: Path):
    """A non-default (runtime dummy) communicator flows as opaque dataflow: a CommF2c node (MPI_Comm_f2c) converts the Fortran integer handle into an opaque(MPI_Comm) feeding Send's _comm connector.  The Fortran handle stays in sdfg.arrays (CommF2c reads it) -- supersedes the legacy process-grid path, which dropped the handle and wired a _grid cartesian sub-comm connector instead."""
    import dace
    from dace.libraries.mpi.nodes.send import Send
    from dace.libraries.mpi.nodes.comm_f2c import CommF2c

    sdfg = _build(_USER_COMM, tmp_path, "sr_usercomm", "sr_usercomm")

    sends = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Send)]
    assert len(sends) == 1, f"expected 1 Send node, got {len(sends)}"
    # communicator threads in via an opaque(MPI_Comm) _comm connector, not the legacy _grid process-grid connector.
    assert '_comm' in sends[0].in_connectors
    assert '_grid' not in sends[0].in_connectors
    assert isinstance(sends[0].in_connectors['_comm'], dace.dtypes.opaque)
    assert sends[0].in_connectors['_comm'].ctype == 'MPI_Comm'

    # exactly one CommF2c node, reading the Fortran integer comm handle on _fcomm and producing the opaque(MPI_Comm) Send consumes.
    f2cs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CommF2c)]
    assert len(f2cs) == 1, f"expected 1 CommF2c node, got {len(f2cs)}"
    fcomm_srcs = [
        e.data.data for st in sdfg.states() for n in st.nodes() if isinstance(n, CommF2c) for e in st.in_edges(n)
        if e.dst_conn == '_fcomm'
    ]
    assert fcomm_srcs == ['comm'], f"CommF2c must read the Fortran comm handle, got {fcomm_srcs}"

    assert 'comm' in sdfg.arrays, "the Fortran integer comm handle is read by CommF2c, so it is kept"
    assert 'dace_user_pgrid' not in sdfg.arrays, "the process-grid path is superseded by CommF2c/_comm"

    sdfg.validate()

    # regression guard: send.py/recv.py once materialised _comm but still hardcoded MPI_COMM_WORLD in the call, mis-routing user-comm Send/Recv onto the world communicator -> deadlock under mpirun.
    codes = _expanded_mpi_call_codes(sdfg)
    assert len(codes) == 1, f"expected 1 Send tasklet, got {len(codes)}"
    assert "_comm" in codes[0], f"user-comm Send must use ``_comm``: {codes[0]!r}"
    assert "MPI_COMM_WORLD" not in codes[0], (f"user-comm Send must NOT fall back to MPI_COMM_WORLD: {codes[0]!r}")
