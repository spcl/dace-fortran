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


def _expanded_mpi_call_codes(sdfg):
    """Expand the SDFG's MPI library nodes and return the CPP tasklet code
    strings that actually issue an ``MPI_Send`` / ``MPI_Recv`` (etc.).

    The communicator each point-to-point op runs on is baked into the
    expansion's C code (``MPI_Send(..., <comm>)``), not into a node
    property, so verifying the comm without ``mpirun`` means inspecting the
    expanded tasklet source.  Guards the ``Send``/``Recv`` ``_grid``
    contract: a wired user communicator must emit ``_grid`` (the cartesian
    sub-comm), and the default path must fall back to ``MPI_COMM_WORLD``."""
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
    """``MPI_Send`` / ``MPI_Recv`` on MPI_COMM_WORLD become DaCe
    ``Send`` / ``Recv`` nodes with the canonical connectors, and the
    SDFG validates."""
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

    # Communicator dataflow: the Fortran comm (here the default MPI_COMM_WORLD)
    # is threaded into the expanded Send/Recv through the ``_comm`` connector fed
    # by a CommF2c node (feature 93cc5f2) -- it is NOT hardcoded into the tasklet,
    # and no ``_grid`` (process-grid) is referenced without a user comm.
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
    """``MPI_Isend`` / ``MPI_Irecv`` / ``MPI_Wait`` become DaCe
    ``Isend`` / ``Irecv`` / ``Wait`` nodes; the non-blocking request is
    threaded producer->Wait through a synthesised
    ``opaque(MPI_Request)`` transient; each MPI call gets its own state
    so program order is enforced by interstate edges."""
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

    # One state per MPI call (4) + the entry state -> interstate-edge
    # ordering of the side-effecting MPI nodes.
    assert len(list(sdfg.all_states())) >= 5
    sdfg.validate()


def test_runtime_communicator_lowers_to_comm_connector(tmp_path: Path):
    """A non-default (runtime dummy) communicator flows as opaque dataflow.

    The Fortran ``integer`` comm handle is converted by a ``CommF2c`` node
    (``MPI_Comm_f2c``) into an ``opaque(MPI_Comm)`` value that feeds the
    ``Send`` node's ``_comm`` in-connector.  The communicator is a first-class
    value the SDFG reads (from the Fortran handle) and writes (to ``_comm``),
    so multiple distinct communicators are supported and the host can pass /
    receive one across the boundary.  The Fortran handle is KEPT in
    ``sdfg.arrays`` (``CommF2c`` reads it) -- this supersedes the legacy
    process-grid path, which dropped the handle and wired a ``_grid``
    cartesian sub-comm connector instead."""
    import dace
    from dace.libraries.mpi.nodes.send import Send
    from dace.libraries.mpi.nodes.comm_f2c import CommF2c

    sdfg = _build(_USER_COMM, tmp_path, "sr_usercomm", "sr_usercomm")

    sends = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, Send)]
    assert len(sends) == 1, f"expected 1 Send node, got {len(sends)}"
    # Communicator threads in via an opaque(MPI_Comm) ``_comm`` connector, not
    # the legacy ``_grid`` process-grid connector.
    assert '_comm' in sends[0].in_connectors
    assert '_grid' not in sends[0].in_connectors
    assert isinstance(sends[0].in_connectors['_comm'], dace.dtypes.opaque)
    assert sends[0].in_connectors['_comm'].ctype == 'MPI_Comm'

    # Exactly one CommF2c node, reading the Fortran integer ``comm`` handle on
    # ``_fcomm`` and producing the opaque(MPI_Comm) the Send consumes.
    f2cs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CommF2c)]
    assert len(f2cs) == 1, f"expected 1 CommF2c node, got {len(f2cs)}"
    fcomm_srcs = [
        e.data.data for st in sdfg.states() for n in st.nodes() if isinstance(n, CommF2c) for e in st.in_edges(n)
        if e.dst_conn == '_fcomm'
    ]
    assert fcomm_srcs == ['comm'], f"CommF2c must read the Fortran comm handle, got {fcomm_srcs}"

    # The Fortran integer ``comm`` handle is KEPT (read by CommF2c), and no
    # process grid is created -- the opaque-dataflow path replaces it.
    assert 'comm' in sdfg.arrays, "the Fortran integer comm handle is read by CommF2c, so it is kept"
    assert 'dace_user_pgrid' not in sdfg.arrays, "the process-grid path is superseded by CommF2c/_comm"

    sdfg.validate()

    # The wired ``_comm`` connector must actually drive the emitted MPI call:
    # the expanded Send must issue ``MPI_Send(..., _comm)`` on the user
    # communicator, NOT ``MPI_COMM_WORLD``.  (Regression guard: send.py/recv.py
    # once materialised the connector but still hardcoded ``MPI_COMM_WORLD`` in
    # the call, mis-routing every user-comm Send/Recv onto the world
    # communicator -> deadlock under ``mpirun``.)
    codes = _expanded_mpi_call_codes(sdfg)
    assert len(codes) == 1, f"expected 1 Send tasklet, got {len(codes)}"
    assert "_comm" in codes[0], f"user-comm Send must use ``_comm``: {codes[0]!r}"
    assert "MPI_COMM_WORLD" not in codes[0], (f"user-comm Send must NOT fall back to MPI_COMM_WORLD: {codes[0]!r}")
