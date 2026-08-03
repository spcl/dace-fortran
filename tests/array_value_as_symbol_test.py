"""An array *element value* used as a data-access dimension becomes an SDFG symbol
(``__sym_<array>_<index>``) distinct from the array itself; the backing array must
stay constant in the symbol's scope or the build is refused.  Reproduces ICON's
``z_raylfac(nrdmax(jg))`` pattern (``mo_solve_nonhydro``)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_SRC = """
module array_value_as_dim_mod
  implicit none
contains
subroutine array_value_as_dim(sizes, sel, out)
  implicit none
  integer, intent(in) :: sizes(4)
  integer, intent(in) :: sel
  real(8), intent(inout) :: out(8)
  real(8) :: work(sizes(sel))
  integer :: i, n
  n = sizes(sel)
  do i = 1, n
    work(i) = out(i) * 2.0d0
  end do
  do i = 1, 8
    if (i <= n) out(i) = work(i)
  end do
end subroutine array_value_as_dim
end module array_value_as_dim_mod
"""


def test_array_value_as_dimension_symbol(tmp_path: Path):
    """``work(sizes(sel))`` extent becomes symbol ``__sym_sizes_sel`` (distinct from
    the ``sizes`` array); result matches the reference computation."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="avd", entry="array_value_as_dim_mod::array_value_as_dim").build()
    assert "sizes" in sdfg.arrays  # the array stays a data descriptor
    assert "sizes" not in sdfg.symbols, "array name leaked in as a symbol"
    assert "__sym_sizes_sel" in sdfg.symbols, \
        f"expected the value-symbol; got {sorted(sdfg.symbols)}"

    sizes = np.array([3, 5, 2, 7], dtype=np.int32)
    sel = 2  # 1-based -> sizes(2) = 5
    out0 = np.arange(1, 9, dtype=np.float64)

    out = out0.copy()
    sdfg(sizes=sizes, sel=np.int32(sel), out=out)

    # ``work(i) = out(i)*2`` for i in 1..sizes(sel); the rest of out is kept.
    ref = out0.copy()
    n = sizes[sel - 1]
    ref[:n] = out0[:n] * 2.0
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


_SRC_WRITTEN = """
module array_value_written_mod
  implicit none
contains
subroutine array_value_written(sizes, sel, out)
  implicit none
  integer, intent(inout) :: sizes(4)
  integer, intent(in) :: sel
  real(8), intent(inout) :: out(8)
  real(8) :: work(sizes(sel))
  integer :: i, n
  n = sizes(sel)
  sizes(1) = 99            ! write to the backing array -> stale value-symbol
  do i = 1, n
    work(i) = out(i) * 2.0d0
  end do
  do i = 1, 8
    if (i <= n) out(i) = work(i)
  end do
end subroutine array_value_written
end module array_value_written_mod
"""


def test_value_symbol_automatic_extent_source_write_allowed(tmp_path: Path):
    """``__sym_sizes_sel`` sizes an AUTOMATIC array (``work(sizes(sel))``).  Fortran
    evaluates that bound once at procedure entry and freezes it, so the later write
    ``sizes(1) = 99`` cannot change ``work``'s extent -- the entry snapshot is exact.
    Formerly refused as over-conservative; now builds and matches the reference,
    while the source write still takes effect on the ``sizes`` array itself."""
    sdfg = build_sdfg(_SRC_WRITTEN, tmp_path / "sdfg", name="avw",
                      entry="array_value_written_mod::array_value_written").build()
    assert "__sym_sizes_sel" in sdfg.symbols

    sizes = np.array([3, 5, 2, 7], dtype=np.int32)
    sel = 2  # 1-based -> sizes(2) = 5; sizes(1)=99 write is irrelevant to work's extent
    out0 = np.arange(1, 9, dtype=np.float64)

    sizes_s, out = sizes.copy(), out0.copy()
    sdfg(sizes=sizes_s, sel=np.int32(sel), out=out)

    ref = out0.copy()
    n = sizes[sel - 1]  # entry extent, unchanged by the write
    ref[:n] = out0[:n] * 2.0
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
    # the write landed on the array, but did not perturb the frozen extent above.
    np.testing.assert_array_equal(sizes_s, np.array([99, 5, 2, 7], dtype=np.int32))


_SRC_RESNAP = """
module resnap_mod
  implicit none
contains
subroutine resnap(sel, tab, z, out)
  implicit none
  integer, intent(in) :: sel
  integer, intent(inout) :: tab(4)
  real(8), intent(in) :: z(10)
  real(8), intent(inout) :: out(2)
  out(1) = z(tab(sel))       ! index z with tab(sel) -> per-site symbol tab_at0
  tab(sel) = tab(sel) + 3    ! write the backing array between the two uses
  out(2) = z(tab(sel))       ! second read -> tab_at1, must see the UPDATED tab(sel)
end subroutine resnap
end module resnap_mod
"""


def test_value_symbol_reaching_def_resnapshot(tmp_path: Path):
    """A mutable array element used as a data-access INDEX (``z(tab(sel))``) is
    projected PER READ SITE: the bridge mints ``tab_at0`` for the first read and
    ``tab_at1`` for the second (assigns.cpp path (b) / access.py), each re-reading
    the element at its own point.  So a write to ``tab`` between the two uses is
    handled -- not refused -- and the second index sees the updated value.  This is
    reaching-def value-projection SSA for the data-access case.  Oracle'd against f2py."""
    sdfg = build_sdfg(_SRC_RESNAP, tmp_path / "sdfg", name="rsnp", entry="resnap_mod::resnap").build()
    # One value symbol per read site: tab_at0 (pre-write), tab_at1 (post-write).
    assert sum(s.startswith("tab_at")
               for s in sdfg.symbols) >= 2, f"expected per-site symbols; got {sorted(sdfg.symbols)}"
    assert "tab" in sdfg.arrays

    sel = 1  # 1-based -> tab(1)
    tab = np.array([2, 5, 3, 7], dtype=np.int32)
    z = np.arange(1, 11, dtype=np.float64) * 10.0  # z(i) = 10*i

    tab_s, out_s = tab.copy(), np.zeros(2, dtype=np.float64)
    sdfg(sel=np.int32(sel), tab=tab_s, z=z, out=out_s)

    # first read tab(1)=2 -> z(2)=20; after tab(1)+=3 -> tab(1)=5 -> second read z(5)=50.
    assert out_s[0] == z[tab[sel - 1] - 1]  # tab_at0: pre-write element
    assert out_s[1] == z[tab[sel - 1] + 3 - 1]  # tab_at1: post-write element

    tab_r, out_r = tab.copy(), np.zeros(2, dtype=np.float64)
    mod = f2py_compile(_SRC_RESNAP, tmp_path / "ref", f"rsnp_ref_{tmp_path.name}")
    mod.resnap_mod.resnap(np.int32(sel), tab_r, z, out_r)
    np.testing.assert_array_equal(out_s, out_r)
    np.testing.assert_array_equal(tab_s, tab_r)
