"""Whole-array assignment from a function returning a fixed-shape array: ``tmp = make3(x)`` where ``make3`` returns ``real(8) :: r(3)``.

Once emitted the bridge's unresolved-expression ``?`` placeholder and failed to parse; now lowers correctly and runs BIT-EXACT against gfortran/f2py.  Mirrors production patterns like graupel's ``precip1`` multi-value PURE FUNCTION returns.
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

    # gfortran reference: out_arr is intent(out)->returned, n inferred from src; single-multiply per row so SDFG and reference are bit-identical.
    ref = f2py_compile(_SRC, tmp_path / "ref", "array_return_ref", only=("kern", ))
    out_ref = np.asfortranarray(ref.m_array_return.kern(src_arr.copy(order='F')))
    np.testing.assert_array_equal(out, out_ref)
