"""Scatter self-alias eval-order (audit crit#8).

``a(idx_w) = a(idx_r)`` is a vector-subscript scatter whose RHS gathers off the
SAME array; Fortran requires the whole RHS evaluated into a temp before any LHS
write, else an already-written element feeds a later read.  The scatter pass
(``ExpandVectorSubscriptScatter``) materialises a ``<dst>_scatter_<id>`` temp when
the LHS root aliases the RHS gather root -- the root must be the DATA array
(``a``), not the index array, or no temp is emitted and the scatter miscompiles.

Repro: a rotate (idx_w=[2,3,4,1], idx_r=[1,2,3,4]).  Without the temp the
sequential writes cascade the first value into every slot.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_scatter_self_alias_rotate(tmp_path: Path):
    src = """
subroutine rot(a, idx_w, idx_r)
  double precision, intent(inout) :: a(4)
  integer,          intent(in)    :: idx_w(4)
  integer,          intent(in)    :: idx_r(4)
  a(idx_w) = a(idx_r)
end subroutine rot
"""
    sdfg = build_sdfg(src, tmp_path, name='rot').build()
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    idx_w = np.array([2, 3, 4, 1], dtype=np.int32)
    idx_r = np.array([1, 2, 3, 4], dtype=np.int32)
    # Fortran semantics: RHS fully evaluated first.  a_new[idx_w-1] = a_old[idx_r-1].
    expected = a.copy()
    expected[idx_w - 1] = a[idx_r - 1]  # -> [40, 10, 20, 30]
    sdfg(a=a, idx_w=idx_w, idx_r=idx_r)
    np.testing.assert_allclose(a, expected, rtol=1e-12)


def test_scatter_self_alias_reverse(tmp_path: Path):
    """Full reverse ``a(4:1:-1) = a(1:4)`` via index arrays -- another overlap
    that collapses without an eval-order temp."""
    src = """
subroutine rev(a, idx_w, idx_r)
  double precision, intent(inout) :: a(4)
  integer,          intent(in)    :: idx_w(4)
  integer,          intent(in)    :: idx_r(4)
  a(idx_w) = a(idx_r)
end subroutine rev
"""
    sdfg = build_sdfg(src, tmp_path, name='rev').build()
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    idx_w = np.array([4, 3, 2, 1], dtype=np.int32)
    idx_r = np.array([1, 2, 3, 4], dtype=np.int32)
    expected = a.copy()
    expected[idx_w - 1] = a[idx_r - 1]  # -> [40, 30, 20, 10]
    sdfg(a=a, idx_w=idx_w, idx_r=idx_r)
    np.testing.assert_allclose(a, expected, rtol=1e-12)
