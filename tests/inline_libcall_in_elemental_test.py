"""E2e probes for inline use of every HLFIR libcall the elemental + ``hlfir.apply``
materialisation handles.

``bridge/ast/control_flow.cpp::walkElementalBody`` pre-emits a ``_libtmp_<gid>``
transient when an elemental body's ``hlfir.apply`` reads a libcall expr-producer;
``libcallNameForExprOp`` (``bridge/ast/elementals.cpp``) is the gate -- anything not
listed there falls out as ``?`` in the tasklet body.

Covered: matmul, transpose, dot_product, matmul_transpose (QE vcut_get shape), count,
minloc/maxloc, cshift.
Still gaps (``?``): hlfir.reshape (not in the dispatch table); hlfir.sum/product/minval/
maxval/any/all with DIM (goes through buildSectionReduceAssign, not the libcall path).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_inline_matmul_transpose_in_elemental_division(tmp_path):
    """``MATMUL(TRANSPOSE(A), q) / scalar`` (QE vcut_get shape) -- optimised-bufferization
    fuses TRANSPOSE+MATMUL to ``hlfir.matmul_transpose``; ``libcallNameForExprOp``
    recognises it and materialises a ``Transpose + MatMul`` libcall pair, no source rewrite."""
    src = """
module m
contains
  subroutine qe_pattern(a, q, s, res)
    real(kind=8), intent(in) :: a(3, 3), q(3), s
    real(kind=8), intent(out) :: res(3)
    res = MATMUL(TRANSPOSE(a), q) / s
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="qe_pattern", entry="m::qe_pattern").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, s=np.float64(2.0), res=res)
    np.testing.assert_allclose(res, (A.T @ q) / 2.0)


def test_inline_cshift_in_elemental(tmp_path):
    """``2.0 - CSHIFT(arr, 1)`` -- bridge stashes the shift into ``options['shift']`` (so
    ``CShift`` gets the concrete shift, not a leaking ``__shift`` free-symbol fallback);
    d-face's ``ExpandCShiftPure`` lowers it to a single Map with a ``Mod``-rotated index."""
    src = """
module m
contains
  subroutine cshift_inline(arr, res)
    real(kind=8), intent(in) :: arr(5)
    real(kind=8), intent(out) :: res(5)
    res = 2.0d0 - CSHIFT(arr, 1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cshift_inline", entry="m::cshift_inline").build()
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order='F')
    res = np.zeros(5, dtype=np.float64, order='F')
    sdfg(arr=arr, res=res)
    expected = 2.0 - np.roll(arr, -1)
    np.testing.assert_allclose(res, expected)
