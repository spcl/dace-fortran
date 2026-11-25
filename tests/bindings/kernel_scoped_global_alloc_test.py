"""A module global the kernel allocates and frees within one call is a kernel-owned transient, never an argument.

QE's ``qgm(ngms, nij)`` is allocated by ``qvan_init`` with an ``nij`` it sums over the species, and freed by
``qvan_clean``, both under ``IF (okvan)``. As an SDFG argument the binding had to allocate it before the call, by an
``nij`` only the kernel computes: it allocated it at an unset size and the kernel wrote past it. Assigned on a
conditional path only, ``nij`` also stayed a free symbol of the SDFG.
"""

import ctypes
from pathlib import Path

import numpy as np

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from tests._util import build_sdfg

_NAME = "kernel_work"
_ENTRY = "kernel_work_mod::kernel_work"
_SRC = """
module mo_work
  implicit none
  real(8), allocatable :: work(:, :)
contains
  subroutine work_init(n, nt, sizes)
    integer, intent(in) :: n, nt
    integer, intent(in) :: sizes(nt)
    integer :: nij, t, i, j
    nij = 0
    do t = 1, nt
      nij = nij + sizes(t)
    end do
    allocate(work(n, nij))
    do j = 1, nij
      do i = 1, n
        work(i, j) = dble(i + 10 * j)
      end do
    end do
  end subroutine work_init

  subroutine work_clean
    deallocate(work)
  end subroutine work_clean
end module mo_work

module kernel_work_mod
contains
  subroutine kernel_work(n, nt, sizes, out)
    use mo_work, only: work, work_init, work_clean
    implicit none
    integer, intent(in) :: n, nt
    integer, intent(in) :: sizes(nt)
    real(8), intent(out) :: out(n)
    integer :: i
    out = 0.0d0
    if (nt > 0) call work_init(n, nt, sizes)
    if (nt > 0) then
      do i = 1, n
        out(i) = sum(work(i, :))
      end do
    end if
    if (nt > 0) call work_clean
  end subroutine kernel_work
end module kernel_work_mod
"""
_DRIVERS = """
subroutine run_work_binding(n, nt, sizes, out) bind(c, name="run_work_binding")
  use kernel_work_dace_bindings, only: kernel_work_dace, kernel_work_dace_finalize
  implicit none
  integer, value :: n, nt
  integer, intent(in) :: sizes(nt)
  real(8), intent(out) :: out(n)
  call kernel_work_dace(n, nt, sizes, out)
  call kernel_work_dace_finalize()
end subroutine run_work_binding

subroutine run_work_ref(n, nt, sizes, out) bind(c, name="run_work_ref")
  use kernel_work_mod, only: kernel_work
  implicit none
  integer, value :: n, nt
  integer, intent(in) :: sizes(nt)
  real(8), intent(out) :: out(n)
  call kernel_work(n, nt, sizes, out)
end subroutine run_work_ref
"""


def build(tmp_path: Path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name=_NAME, entry=_ENTRY).build()
    sdfg.name = _NAME
    return sdfg


def test_a_global_allocated_and_freed_in_the_call_is_no_argument(tmp_path: Path):
    sdfg = build(tmp_path)

    assert sdfg.arrays["work"].transient
    assert "work" not in sdfg.arglist() and "work_init_nij" not in sdfg.arglist() and "nij" not in sdfg.arglist()


def test_a_global_allocated_and_freed_in_the_call_matches_gfortran(tmp_path: Path):
    sdfg = build(tmp_path)
    src, drivers = tmp_path / "work.f90", tmp_path / "drivers.f90"
    src.write_text(_SRC)
    drivers.write_text(_DRIVERS)
    lib = build_fortran_library(
        sdfg, None, None, str(tmp_path / "lib"), name=_NAME, prelude_sources=[src], extra_sources=[drivers]
    )
    dl = lib.load()
    n, sizes = 5, np.array([3, 1, 4], dtype=np.int32)

    def run(symbol: str) -> np.ndarray:
        out = np.zeros(n)
        fn = getattr(dl, symbol)
        fn.restype = None
        fn(
            ctypes.c_int(n),
            ctypes.c_int(len(sizes)),
            sizes.ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p),
        )
        return out

    np.testing.assert_allclose(run("run_work_binding"), run("run_work_ref"), rtol=1e-14)


if __name__ == "__main__":
    import tempfile

    for test in (
        test_a_global_allocated_and_freed_in_the_call_is_no_argument,
        test_a_global_allocated_and_freed_in_the_call_matches_gfortran,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
