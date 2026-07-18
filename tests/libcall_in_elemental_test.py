"""Bridge support for libcall results consumed by an enclosing ``hlfir.elemental`` body -- the ``2.0 - matmul(a, b)`` / ``1.0 - transpose(a)`` pattern.

Flang lowers this to an hlfir.matmul/transpose/dot_product producing an hlfir.expr value, read by the elemental body via hlfir.apply.  Without bridge support, buildExpr returns ``?`` for the apply and the tasklet fails to parse; the bridge materialises the libcall result into a ``_libtmp_<gid>`` transient and rewrites the apply as a regular array read.  Smallest possible programs so a regression here isolates to the bridge logic.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_one_minus_transpose(tmp_path: Path):
    """``res = 1.0 - transpose(a)`` -- libcall-in-elemental materialisation for hlfir.transpose."""
    src = """
subroutine main(a, res)
  double precision, dimension(5,4) :: a
  double precision, dimension(4,5) :: res
  res = 1.0 - transpose(a)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.empty((5, 4), order="F", dtype=np.float64)
    a[:] = np.arange(20).reshape(5, 4)
    res = np.zeros((4, 5), order="F", dtype=np.float64)
    sdfg(a=a, res=res)
    np.testing.assert_array_equal(res, 1.0 - a.T)


def test_two_minus_matmul(tmp_path: Path):
    """``res = 2.0 - matmul(a, b)``  --  same pattern, ``hlfir.matmul``."""
    src = """
subroutine main(a, b, res)
  double precision, dimension(5,3) :: a
  double precision, dimension(3,7) :: b
  double precision, dimension(5,7) :: res
  res = 2.0 - matmul(a, b)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    a = np.empty((5, 3), order="F", dtype=np.float64)
    a[:] = np.arange(15).reshape(5, 3)
    b = np.empty((3, 7), order="F", dtype=np.float64)
    b[:] = np.arange(21).reshape(3, 7)
    res = np.zeros((5, 7), order="F", dtype=np.float64)
    sdfg(a=a, b=b, res=res)
    np.testing.assert_array_equal(res, 2.0 - a @ b)
