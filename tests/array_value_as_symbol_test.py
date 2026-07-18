"""An array *element value* used as a data-access dimension becomes an SDFG symbol
(``__sym_<array>_<index>``) distinct from the array itself; the backing array must
stay constant in the symbol's scope or the build is refused.  Reproduces ICON's
``z_raylfac(nrdmax(jg))`` pattern (``mo_solve_nonhydro``)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

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


def test_value_symbol_backing_array_write_refused(tmp_path: Path):
    """Writing the array a value-symbol froze (``sizes``) would make the symbol
    stale; the constancy hook refuses the build."""
    with pytest.raises(ValueError, match=r"constant within the scope|stale value"):
        build_sdfg(_SRC_WRITTEN, tmp_path / "sdfg", name="avw",
                   entry="array_value_written_mod::array_value_written").build()
