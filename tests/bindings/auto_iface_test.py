"""Auto-derived ``OriginalInterface`` correctness.

``SDFGBuilder.build`` snapshots the entry's pre-flatten dummy list onto
``sdfg._fortran_interface_raw``; ``build_auto_interface`` turns that into an
``OriginalInterface`` so ``build_fortran_library`` can emit a binding with no
hand-written interface.  These tests check the derived interface matches what
a human would have written (arg order, iso-c types, rank/shape, intent, and
the ``use`` list for derived-type dummies), and that it drives a compilable
binding end-to-end.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import build_fortran_library
from dace_fortran.bindings.fortran_interface import (
    OriginalArg,
    OriginalInterface,
    build_auto_interface,
)

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _auto(src: str, tmp_path: Path, name: str, entry: str) -> OriginalInterface:
    sdfg = build_sdfg(src, tmp_path / "sdfg", name=name, entry=entry).build()
    return build_auto_interface(sdfg._fortran_interface_raw, name)


def test_auto_iface_flat_matches_handwritten(tmp_path):
    """The QE complex-AXPY kernel: derived iface == the README's hand-written
    one, arg order (n, a, x, y) and all."""
    src = (Path(__file__).parents[1] / "qe_loopnests" / "qe_e4_zaxpy.f90").read_text()
    auto = _auto(src, tmp_path, "kernel", "_QPkernel")
    assert auto.entry == "kernel"
    assert auto.used_modules == {}
    assert auto.args == (
        OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, shape=(), intent="in"),
        OriginalArg(name="a", fortran_type="complex(c_double)", rank=1, shape=("1",), intent="in"),
        OriginalArg(name="x", fortran_type="complex(c_double)", rank=1, shape=("n",), intent="in"),
        OriginalArg(name="y", fortran_type="complex(c_double)", rank=1, shape=("n",), intent="inout"),
    )


def test_auto_iface_scalar_and_array_dtypes(tmp_path):
    """real(4)/real(8)/int kinds + intent in/out/inout map to the right
    iso-c types in declaration order."""
    src = """
subroutine kern(ni, xr4, xr8, yi8, out8)
  use iso_c_binding
  implicit none
  integer(c_int),     intent(in)    :: ni
  real(c_float),      intent(in)    :: xr4(ni)
  real(c_double),     intent(inout) :: xr8(ni)
  integer(c_int64_t), intent(in)    :: yi8(ni)
  real(c_double),     intent(out)   :: out8(ni)
  integer :: i
  do i = 1, ni
     out8(i) = real(xr4(i), c_double) + xr8(i) + real(yi8(i), c_double)
  end do
end subroutine kern
"""
    auto = _auto(src, tmp_path, "kern", "_QPkern")
    by = {a.name: a for a in auto.args}
    assert [a.name for a in auto.args] == ["ni", "xr4", "xr8", "yi8", "out8"]
    assert by["ni"].fortran_type == "integer(c_int)" and by["ni"].rank == 0
    assert by["xr4"].fortran_type == "real(c_float)" and by["xr4"].intent == "in"
    assert by["xr8"].fortran_type == "real(c_double)" and by["xr8"].intent == "inout"
    assert by["yi8"].fortran_type == "integer(c_int64_t)" and by["yi8"].intent == "in"
    assert by["out8"].fortran_type == "real(c_double)" and by["out8"].intent == "out"


def test_auto_iface_struct_dummy_resolves_type_and_module(tmp_path):
    """An AoS dummy is recovered as ``type(point)`` with the defining module
    in the ``use`` list -- everything ``build_fortran_library`` needs for the
    wrapper signature (members come from the FlattenPlan, not the iface)."""
    src = """
module mo_pt
  use iso_c_binding
  implicit none
  integer, parameter :: N = 6
  type :: point
     real(c_double) :: x, y, z, w
  end type point
end module mo_pt
subroutine kern_aos(pts)
  use mo_pt
  implicit none
  type(point), intent(inout) :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = pts(i)%x + pts(i)%y
  end do
end subroutine kern_aos
"""
    auto = _auto(src, tmp_path, "kern_aos", "_QPkern_aos")
    assert len(auto.args) == 1
    arg = auto.args[0]
    assert arg.name == "pts" and arg.fortran_type == "type(point)"
    assert arg.struct_type == "point" and arg.rank == 1
    assert auto.used_modules == {"mo_pt": ("point",)}


def test_auto_iface_drives_compilable_binding(tmp_path):
    """``build_fortran_library`` with no ``iface``/``plan`` derives both from
    the SDFG and emits a binding that gfortran links against the kernel."""
    src = """
subroutine scale2(n, x)
  use iso_c_binding
  implicit none
  integer(c_int), intent(in)    :: n
  real(c_double), intent(inout) :: x(n)
  integer :: i
  do i = 1, n
     x(i) = x(i) * 2.0_c_double
  end do
end subroutine scale2
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="scale2", entry="_QPscale2").build()
    lib = build_fortran_library(sdfg, out_dir=str(tmp_path / "lib"), name="scale2")
    assert Path(lib.bindings_f90).exists()
    assert Path(lib.so_path).exists()
    # The wrapper preserves the caller's (n, x) signature.
    text = Path(lib.bindings_f90).read_text()
    assert "subroutine scale2_dace(n, x)" in text
