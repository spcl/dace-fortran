"""An inlined callee's optional dummy that names an entry variable stays a variable of its own.

QE's ``addusxx_g`` declares ``becpsi_r(nkb)``, an optional dummy ``vexx_bp_k_gpu`` never passes, and the entry's record
dummy ``becpsi`` flattens its member ``r`` to ``becpsi_r`` as well. The two merged into one SDFG container of shape
``(nkb,)``, and the binding bound the caller's ``becpsi % r`` (allocated ``(1, 1)``) to it.
"""

import ctypes
from pathlib import Path

import numpy as np

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from tests._util import build_sdfg

_NAME = "kernel_bec"
_ENTRY = "kernel_bec_mod::kernel_bec"
_SRC = """
module mo_bec
  implicit none
  integer :: nkb = 0
  type :: t_bec
    real(8), allocatable :: r(:, :)
    real(8), allocatable :: k(:, :)
  end type t_bec
end module mo_bec

module kernel_bec_mod
contains
  subroutine add_rows(w, x, becp_r)
    use mo_bec, only: nkb
    implicit none
    real(8), intent(inout) :: w(nkb)
    real(8), intent(in) :: x(nkb)
    real(8), intent(in), optional :: becp_r(nkb)
    w = w + x
    if (present(becp_r)) w = w + becp_r
  end subroutine add_rows

  subroutine kernel_bec(becp, out)
    use mo_bec, only: nkb, t_bec
    implicit none
    type(t_bec), intent(in) :: becp
    real(8), intent(inout) :: out(:)
    real(8) :: w(nkb)
    integer :: i
    w = 1.0d0
    call add_rows(w, becp%k(:, 1))
    w(1) = w(1) + becp%r(1, 1)
    do i = 1, nkb
      out(i) = w(i)
    end do
  end subroutine kernel_bec
end module kernel_bec_mod
"""
_DRIVERS = """
subroutine fill(becp, n)
  use mo_bec, only: nkb, t_bec
  implicit none
  type(t_bec), intent(inout) :: becp
  integer, intent(in) :: n
  integer :: i
  nkb = n
  allocate(becp%r(1, 1), becp%k(nkb, 1))
  becp%r = 0.5d0
  do i = 1, nkb
    becp%k(i, 1) = dble(i) / 4
  end do
end subroutine fill

subroutine run_bec_binding(n, out) bind(c, name="run_bec_binding")
  use mo_bec, only: t_bec
  use kernel_bec_dace_bindings, only: kernel_bec_dace, kernel_bec_dace_finalize
  implicit none
  integer, value :: n
  real(8), intent(inout) :: out(n)
  type(t_bec) :: becp
  call fill(becp, n)
  call kernel_bec_dace(becp, out)
  call kernel_bec_dace_finalize()
end subroutine run_bec_binding

subroutine run_bec_ref(n, out) bind(c, name="run_bec_ref")
  use mo_bec, only: t_bec
  use kernel_bec_mod, only: kernel_bec
  implicit none
  integer, value :: n
  real(8), intent(inout) :: out(n)
  type(t_bec) :: becp
  call fill(becp, n)
  call kernel_bec(becp, out)
end subroutine run_bec_ref
"""


def build(tmp_path: Path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name=_NAME, entry=_ENTRY).build()
    sdfg.name = _NAME
    return sdfg


def test_an_optional_callee_dummy_does_not_merge_with_a_flattened_entry_member(tmp_path: Path):
    sdfg = build(tmp_path)

    assert len(sdfg.arrays["becp_r"].shape) == 2
    assert [name for name in sdfg.arrays if name.endswith("_becp_r")] == ["add_rows_becp_r"]


def test_an_optional_callee_dummy_named_like_an_entry_member_matches_gfortran(tmp_path: Path):
    sdfg = build(tmp_path)
    src, drivers = tmp_path / "bec.f90", tmp_path / "drivers.f90"
    src.write_text(_SRC)
    drivers.write_text(_DRIVERS)
    lib = build_fortran_library(
        sdfg, None, None, str(tmp_path / "lib"), name=_NAME, prelude_sources=[src], extra_sources=[drivers]
    )
    dl = lib.load()
    n = 6

    def run(symbol: str) -> np.ndarray:
        out = np.zeros(n)
        fn = getattr(dl, symbol)
        fn.restype = None
        fn(ctypes.c_int(n), out.ctypes.data_as(ctypes.c_void_p))
        return out

    np.testing.assert_allclose(run("run_bec_binding"), run("run_bec_ref"), rtol=1e-14)


if __name__ == "__main__":
    import tempfile

    for test in (
        test_an_optional_callee_dummy_does_not_merge_with_a_flattened_entry_member,
        test_an_optional_callee_dummy_named_like_an_entry_member_matches_gfortran,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
