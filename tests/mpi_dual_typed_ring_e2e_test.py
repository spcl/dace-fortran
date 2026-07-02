"""Dual-typed (real*8 + real*4) MPI halo-exchange e2e -- the solve_nh pattern.

ICON's ``solve_nh`` halo exchange issues ``mpi_isend`` / ``mpi_irecv`` with BOTH
real*8 (``p_isend_dp``) and real*4 (``p_isend_sp``) buffers against the same MPI
entry points, then completes them with ``mpi_waitall``.  No existing MPI e2e
covers that dual-typed shape.  This module locks in what works and pins what
does not:

  * ``test_dual_typed_nonblocking_ring`` -- the bridge lowers a real*8 + real*4
    nonblocking ring (per-request ``MPI_Wait``) to ``dace.libraries.mpi`` nodes,
    picking ``MPI_DOUBLE_PRECISION`` vs ``MPI_REAL`` per buffer.  Both received
    buffers arrive, per rank, under real MPI.  PASSES -- dual-typing is sound.

  * ``test_dual_typed_ref_typestar_is_sound`` -- the reference (original Fortran)
    compiles the dual-typed calls with a ``TYPE(*)`` assumed-type interface and
    NO ``-fallow-argument-mismatch``: the interface is proven necessary
    (``EXTERNAL`` decls still error) AND sufficient (``TYPE(*)`` compiles clean).

  * ``test_mpi_waitall_covers_all_requests`` (structural) +
    ``test_mpi_waitall_ring_delivers`` (numeric) -- ``MPI_Waitall`` over a request
    ARRAY now waits on EVERY posted request.  The bridge preserves each ``reqs(k)``
    slot (dispatch.cpp ``mpiRequestSlotExtent``) and sizes the shared
    ``_mpireq_reqs`` transient to the array extent, so the Waitall's request memlet
    covers all requests and the receives complete.  (Previously all ``reqs(k)``
    collapsed onto one slot, Waitall waited on 1, and the receives were silently
    dropped -- the solve_nh halo-exchange bug.  See
    memory/project_mpi_waitall_request_array_collapse_bug.md.)

Run the MPI tests under mpirun with >= 2 ranks (NO ``MPI4PY_RC_INITIALIZE=0`` --
that breaks mpi4py's auto-init under mpirun)::

    OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader UCX_VFS_ENABLE=n \\
        mpirun --oversubscribe -n 2 python -m pytest -m mpi -p no:cacheprovider \\
        tests/mpi_dual_typed_ring_e2e_test.py
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_on_root, build_sdfg, have_flang

pytestmark = [pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")]

# Dual-typed nonblocking ring, per-request MPI_Wait (distinct request scalars
# r1..r4 -> no request-array collapse).  MPI entry points are EXTERNAL with
# OpenMPI constants hard-coded (SDFG-bridge convention: no ``use mpi`` .mod
# dependency at flang-bridge time; the bridge reads the datatype constant to
# pick MPI_DOUBLE_PRECISION / MPI_REAL per call).
_DUAL_RING_WAIT = """
module dualring_mod
contains
subroutine dualring(buf_dp, rbuf_dp, buf_sp, rbuf_sp, n, dst, src, tag)
  implicit none
  integer, intent(in) :: n, dst, src, tag
  real(8), intent(inout) :: buf_dp(n)
  real(8), intent(out)   :: rbuf_dp(n)
  real(4), intent(inout) :: buf_sp(n)
  real(4), intent(out)   :: rbuf_sp(n)
  integer, parameter :: MPI_COMM_WORLD=0, MPI_DOUBLE_PRECISION=17, MPI_REAL=13, MPI_STATUS_IGNORE=-1
  integer :: ierr, r1, r2, r3, r4
  external :: MPI_Isend, MPI_Irecv, MPI_Wait
  call mpi_irecv(rbuf_dp, n, MPI_DOUBLE_PRECISION, src, tag,   MPI_COMM_WORLD, r1, ierr)
  call mpi_isend(buf_dp,  n, MPI_DOUBLE_PRECISION, dst, tag,   MPI_COMM_WORLD, r2, ierr)
  call mpi_irecv(rbuf_sp, n, MPI_REAL,             src, tag+1, MPI_COMM_WORLD, r3, ierr)
  call mpi_isend(buf_sp,  n, MPI_REAL,             dst, tag+1, MPI_COMM_WORLD, r4, ierr)
  call mpi_wait(r1, MPI_STATUS_IGNORE, ierr)
  call mpi_wait(r2, MPI_STATUS_IGNORE, ierr)
  call mpi_wait(r3, MPI_STATUS_IGNORE, ierr)
  call mpi_wait(r4, MPI_STATUS_IGNORE, ierr)
end subroutine
end module
"""

# Single-typed nonblocking ring completed with MPI_Waitall over a request ARRAY
# reqs(1:2) -- the shape that currently mis-lowers (all reqs collapse to one
# _mpireq slot -> Waitall waits on 1 request -> the irecv never completes).
_WAITALL_RING = """
module waitall_mod
contains
subroutine waitall_ring(buf, rbuf, n, dst, src, tag)
  implicit none
  integer, intent(in) :: n, dst, src, tag
  real(8), intent(inout) :: buf(n)
  real(8), intent(out)   :: rbuf(n)
  integer, parameter :: MPI_COMM_WORLD=0, MPI_DOUBLE_PRECISION=17, MPI_STATUS_SIZE=6
  integer :: ierr, reqs(2), stats(MPI_STATUS_SIZE*2)
  external :: MPI_Isend, MPI_Irecv, MPI_Waitall
  call mpi_irecv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, MPI_COMM_WORLD, reqs(1), ierr)
  call mpi_isend(buf,  n, MPI_DOUBLE_PRECISION, dst, tag, MPI_COMM_WORLD, reqs(2), ierr)
  call mpi_waitall(2, reqs, stats, ierr)
end subroutine
end module
"""

# Send value = rank + _OFFSET so an undelivered receive (zero-init) never
# accidentally equals the expected neighbor value -- a bug then fails on EVERY
# rank, not just the one whose neighbor happens to hold 0.
_OFFSET = 10.0


def _drive_ring(comm, sdfg, *, dual: bool):
    """Run a compiled ring SDFG on this rank; return (rbuf_dp, rbuf_sp_or_None,
    expected_neighbor_value)."""
    from dace.sdfg import utils
    rank, size = comm.Get_rank(), comm.Get_size()
    func = utils.distributed_compile(sdfg, comm)
    n, dst, src = 8, (rank + 1) % size, (rank - 1 + size) % size
    expected = float((rank - 1 + size) % size) + _OFFSET
    buf_dp = np.full(n, rank + _OFFSET, dtype=np.float64, order="F")
    rbuf_dp = np.zeros(n, dtype=np.float64, order="F")
    if dual:
        buf_sp = np.full(n, rank + _OFFSET, dtype=np.float32, order="F")
        rbuf_sp = np.zeros(n, dtype=np.float32, order="F")
        func(buf_dp=buf_dp, rbuf_dp=rbuf_dp, buf_sp=buf_sp, rbuf_sp=rbuf_sp, n=n, dst=dst, src=src, tag=7)
        return rbuf_dp, rbuf_sp, expected
    func(buf=buf_dp, rbuf=rbuf_dp, n=n, dst=dst, src=src, tag=7)
    return rbuf_dp, None, expected


def _build_ring(comm, tmp_path, src, entry, name):

    def _b():
        s = build_sdfg(src, tmp_path / name, name=name, entry=entry).build()
        s.name = "mpi_" + name
        return s

    return build_on_root(comm, _b, broadcast=False)


@pytest.mark.mpi
def test_dual_typed_nonblocking_ring(tmp_path: Path):
    """Bridge lowers a real*8 + real*4 nonblocking ring (per-request Wait) to MPI
    libnodes; both received buffers equal the neighbor's value, per rank."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("dual-typed MPI ring needs >= 2 ranks (mpirun --oversubscribe -n 2 ...)")

    sdfg = _build_ring(comm, tmp_path, _DUAL_RING_WAIT, "dualring_mod::dualring", "dualring")
    rbuf_dp, rbuf_sp, expected = _drive_ring(comm, sdfg, dual=True)
    n = rbuf_dp.shape[0]
    # A dropped datatype mapping (e.g. real*4 lowered as MPI_DOUBLE_PRECISION)
    # would corrupt exactly one of these.
    np.testing.assert_array_equal(rbuf_dp, np.full(n, expected, dtype=np.float64))
    np.testing.assert_array_equal(rbuf_sp, np.full(n, np.float32(expected), dtype=np.float32))


def test_mpi_waitall_covers_all_requests(tmp_path: Path):
    """MPI_Waitall over ``reqs(1:2)`` waits on BOTH posted requests (structural).

    Regression for the request-array collapse bug.  The bridge now preserves each
    ``reqs(k)`` slot (dispatch.cpp ``mpiRequestSlotExtent``) and sizes the shared
    ``_mpireq_reqs`` transient to the array extent, so each isend/irecv writes a
    distinct slot and the ``Waitall`` node's ``_request`` memlet covers every
    posted request -- emitting ``MPI_Waitall(count, ...)``.  Before the fix the
    memlet covered 1 element (last writer wins) and the receives were silently
    dropped."""
    sdfg = build_sdfg(_WAITALL_RING, tmp_path / "waitall", name="waitall", entry="waitall_mod::waitall_ring").build()
    covered = [
        int(e.data.subset.num_elements()) for state in sdfg.states() for node in state.nodes()
        if type(node).__name__ == "Waitall" for e in state.in_edges(node) if e.dst_conn == "_request"
    ]
    assert covered, "no Waitall node found in the lowered SDFG"
    # The ring posts 2 nonblocking ops (irecv reqs(1), isend reqs(2)); Waitall
    # must cover both, else the receive is never completed.
    assert covered[0] == 2, (f"MPI_Waitall covers {covered[0]} request(s), expected 2 -- the request array "
                             f"collapsed to a single _mpireq slot")


@pytest.mark.mpi
def test_mpi_waitall_ring_delivers(tmp_path: Path):
    """MPI_Waitall over a request array completes the receive, per rank (numeric).

    Single-typed nonblocking ring completed with ``mpi_waitall(2, reqs, ...)``.
    Send values are offset (rank + 10) so an undelivered receive (zero-init) can
    never match the expected neighbor value on any rank -- the regression that a
    collapsed request array silently drops."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("MPI_Waitall ring needs >= 2 ranks (mpirun --oversubscribe -n 2 ...)")

    sdfg = _build_ring(comm, tmp_path, _WAITALL_RING, "waitall_mod::waitall_ring", "waitall")
    rbuf, _, expected = _drive_ring(comm, sdfg, dual=False)
    np.testing.assert_array_equal(rbuf, np.full(rbuf.shape[0], expected, dtype=np.float64))


# --- reference-side compile: TYPE(*) is the sound fix for dual-typed MPI calls,
# so -fallow-argument-mismatch is never needed. ------------------------------

_DUAL_CALLS = """
subroutine dualcalls(buf_dp, buf_sp, n, dest, comm)
  {use_line}
  implicit none
  integer, intent(in) :: n, dest, comm
  real(8), intent(inout) :: buf_dp(n)
  real(4), intent(inout) :: buf_sp(n)
  integer :: ierr, req1, req2
  integer, parameter :: MPI_DOUBLE_PRECISION = 17, MPI_REAL = 13
  {extern_line}
  call mpi_isend(buf_dp, n, MPI_DOUBLE_PRECISION, dest, 1, comm, req1, ierr)
  call mpi_isend(buf_sp, n, MPI_REAL,             dest, 2, comm, req2, ierr)
end subroutine
"""

_MPI_TYPESTAR_IFACE = """
module mpi_typestar_iface
  implicit none
  interface
    subroutine mpi_isend(buf, cnt, dtype, dest, tag, comm, req, ierr)
      type(*), dimension(..) :: buf
      integer :: cnt, dtype, dest, tag, comm, req, ierr
    end subroutine
  end interface
end module
"""


def _gfortran_compiles(src: str, tmp: Path, name: str, *, prelude: str = "") -> tuple[bool, str]:
    """gfortran-compile ``src`` (optionally after ``prelude``) with NO
    -fallow-argument-mismatch.  Return (ok, stderr)."""
    f90 = tmp / f"{name}.f90"
    f90.write_text(prelude + "\n" + src)
    r = subprocess.run(
        ["gfortran", "-ffree-line-length-none", f"-J{tmp}", "-c",
         str(f90), "-o",
         str(tmp / f"{name}.o")],
        capture_output=True,
        text=True,
        cwd=str(tmp))
    return r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_dual_typed_ref_typestar_is_sound(tmp_path: Path):
    """A ``TYPE(*)`` interface is the sound fix for solve_nh's dual-typed MPI
    calls -- necessary AND sufficient, so -fallow-argument-mismatch is never
    needed.  EXTERNAL decls (no interface) -> gfortran rejects the real*4 call
    against the real*8-inferred mpi_isend; TYPE(*) -> compiles clean."""
    extern_ok, extern_err = _gfortran_compiles(_DUAL_CALLS.format(use_line="", extern_line="external :: mpi_isend"),
                                               tmp_path, "extern")
    assert not extern_ok, ("dual-typed EXTERNAL mpi_isend unexpectedly compiled without -fallow; "
                           "the mismatch this test guards no longer reproduces")
    assert "mismatch" in extern_err.lower() or "type" in extern_err.lower(), \
        f"expected a real*8/real*4 argument mismatch, got:\n{extern_err[-800:]}"

    typestar_ok, typestar_err = _gfortran_compiles(_DUAL_CALLS.format(use_line="use mpi_typestar_iface",
                                                                      extern_line=""),
                                                   tmp_path,
                                                   "typestar",
                                                   prelude=_MPI_TYPESTAR_IFACE)
    assert typestar_ok, (f"TYPE(*) assumed-type interface should compile the dual-typed calls cleanly "
                         f"without -fallow-argument-mismatch, but gfortran errored:\n{typestar_err[-800:]}")
