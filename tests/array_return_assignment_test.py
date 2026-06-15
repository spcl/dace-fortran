"""Whole-array assignment from a function returning a fixed-shape array.

The bridge does not yet lower

::

    real(8) :: tmp(3)
    tmp = make3(x)              ! LHS = function call returning real(8) :: r(3)

The right-hand side of the scalar-assign tasklet collapses to the bridge's
``?`` unresolved-expression placeholder, which then fails Python ``ast.parse``
inside :func:`dace.sdfg.SDFGState.add_tasklet`.

This pattern shows up in production benchmarks  --  NPB-style PURE FUNCTIONs
that return small fixed arrays as "multi-value" outputs (e.g. graupel's
``update = precip1(zeta, vc, ...)`` where ``precip1`` returns ``real(8) :: r(3)``
holding (new_qx, new_p, new_vt) in one call).

Pinned here as a ``strict=True`` xfail so the gap is tracked, with a faithful
minimal reproducer.  When the bridge learns to lower the pattern, the
assertion succeeds, ``strict=True`` raises ``XPASS`` -> ``FAILED``, and the
marker must be removed as part of the fix.
"""
from pathlib import Path

import pytest

from _util import have_flang

from dace_fortran import build_sdfg_from_files


pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


_SRC = """
module m_array_return
  implicit none
contains
  pure function make3(x) result(r)
    real(8), intent(in) :: x
    real(8) :: r(3)
    r(1) = x
    r(2) = x * 2.0d0
    r(3) = x * 3.0d0
  end function make3

  subroutine kern(out_arr, src, n)
    integer, intent(in) :: n
    real(8), intent(in) :: src(n)
    real(8), intent(out) :: out_arr(3, n)
    real(8) :: tmp(3)
    integer :: i
    do i = 1, n
      tmp = make3(src(i))
      out_arr(:, i) = tmp
    end do
  end subroutine kern
end module m_array_return
"""


def test_whole_array_assignment_from_function_return(tmp_path):
    """The bridge builds an SDFG for the ``tmp = make3(src(i))`` pattern."""
    src = tmp_path / "m.f90"
    src.write_text(_SRC)
    sdfg = build_sdfg_from_files(
        [src], entry="kern",
        name="array_return", out_dir=tmp_path / "build")
    sdfg.validate()
