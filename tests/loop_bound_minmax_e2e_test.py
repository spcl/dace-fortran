"""End-to-end faithfulness for a MIN/MAX-of-array-elements loop bound.

``do level = 1, min(arr(i, j), arr(k, l))`` -- a compound (``arith.select``)
loop bound over two array-element reads.  ``traceToDecl`` used to FOLLOW the
select unconditionally (a shortcut meant only for the ``max(ext, 0)`` extent
clamp), collapsing the whole bound to the bare array name and dropping BOTH the
other operand AND the subscripts -- so the generated code compared an ``int*``
pointer against an integer (a hard C++ compile error).  Restricting the
select-follow to the zero-clamp idiom (one branch is the constant ``0``) lets
the existing min/max idiom render ``min(arr[...], arr[...])``, which is hoisted
to a loop-bound symbol on an interstate edge.

This drives the kernel through its AUTO-GENERATED ``bind(c)`` binding and
compares every output buffer against the ORIGINAL Fortran on random input (the
same harness the ICON-O ocean kernels use).
"""
import shutil

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_SRC = """\
module lib
  implicit none
contains
  subroutine min_bound_loop(arr, i, j, k, l, out)
    integer, intent(in) :: arr(:, :)
    integer, intent(in) :: i, j, k, l
    real(8), intent(out) :: out(100)
    integer :: level
    out = 0.0d0
    do level = 1, min(arr(i, j), arr(k, l))
      out(level) = real(level, 8)
    end do
  end subroutine min_bound_loop
end module lib
"""


def test_minmax_array_element_loop_bound_e2e(tmp_path):
    src = tmp_path / "min_bound_loop.f90"
    src.write_text(_SRC)
    # ``arr`` is filled with random ints in [1, n]; the integer index scalars
    # i/j/k/l default to 0 (the harness scalar default), so pin them to valid
    # 1-based indices into the (n, n) array.  The bound min(arr(i,j), arr(k,l))
    # is then <= n < 100, so the ``out(level)`` writes stay in bounds.
    r = run_kernel_e2e(src, "lib::min_bound_loop", n=6, seed=3, scalar_overrides={"i": 2, "j": 3, "k": 4, "l": 5})
    assert r["passed"], f"build/run failed:\n{r['output'][-3000:]}"
    assert r["max_diff"] == 0.0, f"SDFG binding diverged from the Fortran reference, max|d|={r['max_diff']:.3e}"
