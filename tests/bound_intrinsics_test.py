"""``LBOUND``/``UBOUND``/``SIZE`` tests.  Flang lowers all three inline via the generic
``arith.select`` ternary fallback in ``buildExpr`` -- no dedicated HLFIR intrinsic op."""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_size_lbound_ubound_explicit_shape(tmp_path: Path):
    """``size(a)`` / ``size(a, dim)`` / ``lbound(a, 1)`` / ``ubound(a, 1)``
    on a 2-D explicit-shape dummy with symbolic extents."""
    src = """
subroutine probe(a, n, m, sz, sz1, sz2, lb, ub)
  integer, intent(in)    :: n, m
  real(8), intent(in)    :: a(n, m)
  integer, intent(out)   :: sz, sz1, sz2, lb, ub
  sz = size(a)
  sz1 = size(a, 1)
  sz2 = size(a, 2)
  lb = lbound(a, 1)
  ub = ubound(a, 1)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    rng = np.random.default_rng(0)
    n, m = 4, 5
    a = np.asfortranarray(rng.random((n, m)))
    sz = np.zeros(1, dtype=np.int32)
    sz1 = np.zeros(1, dtype=np.int32)
    sz2 = np.zeros(1, dtype=np.int32)
    lb = np.zeros(1, dtype=np.int32)
    ub = np.zeros(1, dtype=np.int32)
    sdfg(a=a, n=n, m=m, sz=sz, sz1=sz1, sz2=sz2, lb=lb, ub=ub)
    assert int(sz[0]) == n * m
    assert int(sz1[0]) == n
    assert int(sz2[0]) == m
    assert int(lb[0]) == 1
    assert int(ub[0]) == n


def test_lbound_ubound_custom_lower_bound(tmp_path: Path):
    """Custom Fortran lbound (``a(-2:2)``)  --  ``lbound`` must return ``-2``,
    ``ubound`` ``2``, ``size`` 5."""
    src = """
subroutine probe(a, lb, ub, sz)
  real(8), intent(in)   :: a(-2:2)
  integer, intent(out)  :: lb, ub, sz
  lb = lbound(a, 1)
  ub = ubound(a, 1)
  sz = size(a)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    a = np.asfortranarray(np.arange(5, dtype=np.float64))
    lb = np.zeros(1, dtype=np.int32)
    ub = np.zeros(1, dtype=np.int32)
    sz = np.zeros(1, dtype=np.int32)
    sdfg(a=a, lb=lb, ub=ub, sz=sz)
    assert int(lb[0]) == -2
    assert int(ub[0]) == 2
    assert int(sz[0]) == 5
