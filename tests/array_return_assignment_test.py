"""Whole-array assignment from a function returning a fixed-shape array.

The bridge now lowers

::

    real(8) :: tmp(3)
    tmp = make3(x)              ! LHS = function call returning real(8) :: r(3)

(This once emitted the bridge's ``?`` unresolved-expression placeholder, which
failed Python ``ast.parse`` inside the scalar-assign tasklet.  That gap is
CLOSED -- the docstring previously claimed a ``strict=True`` xfail was pinned
here, but no marker was present and the pattern lowers correctly, so this test
now RUNS the SDFG and compares BIT-EXACT against gfortran/f2py instead of only
building + validating.)

This pattern shows up in production benchmarks  --  NPB-style PURE FUNCTIONs
that return small fixed arrays as "multi-value" outputs (e.g. graupel's
``update = precip1(zeta, vc, ...)`` where ``precip1`` returns ``real(8) :: r(3)``
holding (new_qx, new_p, new_vt) in one call).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang

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
    """``tmp = make3(src(i))`` (local array = fixed-shape fn return) lowers,
    runs, and matches gfortran BIT-EXACT: ``out_arr(:,i)=[x, 2x, 3x]``."""
    src = tmp_path / "m.f90"
    src.write_text(_SRC)
    sdfg = build_sdfg_from_files([src], entry="m_array_return::kern", name="array_return", out_dir=tmp_path / "build")
    sdfg.validate()

    n = 5
    rng = np.random.default_rng(3)
    src_arr = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros((3, n), order='F', dtype=np.float64)
    sdfg(out_arr=out, src=src_arr, n=np.int32(n))

    # gfortran reference of the same source (``out_arr`` intent(out) -> returned;
    # ``n`` inferred from ``src``).  Each column is a single-multiply per row, so
    # the SDFG and the reference are bit-identical.
    ref = f2py_compile(_SRC, tmp_path / "ref", "array_return_ref", only=("kern", ))
    out_ref = np.asfortranarray(ref.m_array_return.kern(src_arr.copy(order='F')))
    np.testing.assert_array_equal(out, out_ref)
