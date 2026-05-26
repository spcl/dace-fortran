"""Numeric end-to-end check for a Fortran ``MPI_Send`` / ``MPI_Recv``
on a **runtime/user communicator**, lowered through the dace-fortran
**binding** path (the wrapper does ``MPI_Comm_f2c`` on the Fortran
integer handle).

Run under mpirun with 4 ranks::

    mpirun --oversubscribe -n 4 python -m pytest -p no:cacheprovider \\
        tests/mpi_comm_e2e_test.py

The caller (mpi4py) splits ``MPI_COMM_WORLD`` by ``color = rank % 2``
so the even ranks ``{0, 2}`` form one sub-communicator and the odd
ranks ``{1, 3}`` another (each of size 2).  The kernel takes the
communicator as an ``integer`` dummy; the generated binding wrapper
converts it with ``MPI_Comm_f2c`` and threads it into the DaCe
``Send`` / ``Recv`` ``_comm`` connector.

Inside each 2-rank split comm the two members exchange their buffers
(deadlock-free: even split-rank sends-then-recvs, odd recvs-then-sends).
World ranks pair up by parity ``{0<->2, 1<->3}``, so a rank that
ships its world-rank value must receive its partner's: the expected
result is ``world_rank XOR 2``.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import (
    FlattenPlan,
    OriginalArg,
    OriginalInterface,
    build_fortran_library,
)

pytestmark = [
    pytest.mark.mpi,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("mpif90") is None, reason="mpif90 not on PATH"),
]

_KERNEL = """
subroutine sr_usercomm(buf, rbuf, n, dst, src, tag, comm, crank)
  implicit none
  integer, intent(in) :: n, dst, src, tag, comm, crank
  real(8), intent(inout) :: buf(n)
  real(8), intent(out) :: rbuf(n)
  integer :: ierr
  integer, parameter :: MPI_DOUBLE_PRECISION = 17
  integer, parameter :: MPI_STATUS_IGNORE = -1
  external :: MPI_Send, MPI_Recv
  if (mod(crank, 2) == 0) then
    call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, dst, tag, comm, ierr)
    call MPI_Recv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, comm, &
                  MPI_STATUS_IGNORE, ierr)
  else
    call MPI_Recv(rbuf, n, MPI_DOUBLE_PRECISION, src, tag, comm, &
                  MPI_STATUS_IGNORE, ierr)
    call MPI_Send(buf, n, MPI_DOUBLE_PRECISION, dst, tag, comm, ierr)
  end if
end subroutine sr_usercomm
"""

# Stable C entry: the bindings wrapper ``sr_usercomm_dace`` is a module
# procedure (mangled symbol); this ``bind(c)`` shim gives ctypes a
# fixed name and converts the raw pointers + by-value scalars.
_DRIVER = """
subroutine run_sr(buf_p, rbuf_p, n, dst, src, tag, fcomm, crank) &
    bind(c, name='run_sr')
  use iso_c_binding
  use sr_usercomm_dace_bindings
  implicit none
  type(c_ptr), value :: buf_p, rbuf_p
  integer(c_int), value :: n, dst, src, tag, fcomm, crank
  real(c_double), pointer :: buf(:), rbuf(:)
  call c_f_pointer(buf_p, buf, [n])
  call c_f_pointer(rbuf_p, rbuf, [n])
  call sr_usercomm_dace(buf, rbuf, n, dst, src, tag, fcomm, crank)
  call sr_usercomm_dace_finalize()
end subroutine run_sr
"""

_IFACE = OriginalInterface(
    entry="sr_usercomm",
    args=(
        OriginalArg(name="buf", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="inout"),
        OriginalArg(name="rbuf", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="out"),
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="dst", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="src", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="tag", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="comm", fortran_type="integer(c_int)", rank=0, intent="in"),
        OriginalArg(name="crank", fortran_type="integer(c_int)", rank=0, intent="in"),
    ),
)


@pytest.mark.mpi
def test_user_comm_split_send_recv(tmp_path: Path):
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    wrank = world.Get_rank()
    wsize = world.Get_size()
    if wsize < 4:
        pytest.skip("user-comm split e2e needs 4 ranks (mpirun --oversubscribe -n 4 ...)")

    # Even ranks {0,2} -> one comm, odd {1,3} -> another (size 2 each).
    split = world.Split(color=wrank % 2, key=wrank)
    crank = split.Get_rank()
    partner = 1 - crank  # the other member of the 2-rank split comm

    # The library (kernel + wrapper) is rank-independent -- only the
    # runtime communicator differs.  Build it once on rank 0 and
    # broadcast the ``.so`` path; all ranks load the same artifact.
    # (``sdfg.compile`` uses a shared ``.dacecache`` build dir, so
    # concurrent multi-rank builds would race the cmake cache.)
    so_path = None
    if wrank == 0:
        sdfg_dir = tmp_path / "sdfg"
        sdfg_dir.mkdir(parents=True, exist_ok=True)
        builder = build_sdfg(_KERNEL, sdfg_dir, name="sr_usercomm", entry="sr_usercomm")
        plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
        sdfg = builder.build()
        sdfg.name = "sr_usercomm"
        sdfg.compile()
        driver_path = tmp_path / "driver.f90"
        driver_path.write_text(_DRIVER)
        lib = build_fortran_library(sdfg, _IFACE, plan, str(tmp_path / "lib"),
                                    name="sr_usercomm", extra_sources=[driver_path])
        so_path = str(lib.so_path)
    so_path = world.bcast(so_path, root=0)
    world.Barrier()

    dll = ctypes.CDLL(so_path)
    dll.run_sr.restype = None
    dll.run_sr.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

    n = 8
    buf = np.full(n, float(wrank), dtype=np.float64, order="F")
    rbuf = np.zeros(n, dtype=np.float64, order="F")
    dll.run_sr(
        buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rbuf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n, partner, partner, 7, split.py2f(), crank,
    )

    # Partner is the other world rank of the same parity: 0<->2, 1<->3.
    np.testing.assert_allclose(rbuf, np.full(n, float(wrank ^ 2), dtype=np.float64))


if __name__ == "__main__":
    test_user_comm_split_send_recv(Path("/tmp/mpi_comm_e2e"))
