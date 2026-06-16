"""Array reductions (``SUM`` / ``MINVAL`` / ``MAXVAL`` / ``PRODUCT``) appearing
in a CONDITION are materialised into a scalar transient via a ``Reduce`` library
node BEFORE the branch (the reduction's elemental operands lower to element-wise
for-loops first), so the condition reads a scalar (``s > eps``) instead of
inline-unrolling the reduction into the condition expression.

These tests pin both the STRUCTURE (a ``Reduce`` lib-node + a scalar condition,
no bare whole-array operands) and END-TO-END correctness.  The reductions are
exercised inside ``DO`` loops and in loop conditions -- the safety cases the
materialisation must not break -- with the loop bodies incrementing an
in-bounds counter / accumulator array so a miscompile shows up as a wrong
result rather than out-of-bounds memory.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _util import build_sdfg, have_flang  # noqa: E402

_needs_flang = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _reduce_nodes(sdfg):
    from dace.libraries.standard.nodes import Reduce
    return [n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, Reduce)]


@_needs_flang
def test_if_sum_reduction_in_loop(tmp_path):
    """``IF (SUM((iv - rv)**2) > eps)`` inside a ``DO`` loop -- the Gate-G
    pattern.  Materialises a ``Reduce`` (sum) lib-node; the per-element squared
    difference must keep its subscript (no bare ``(iv - rv)`` whole-array term).
    Each hit increments an in-bounds counter."""
    src = """
subroutine driver(rv, cnt, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: rv(3)
  integer, intent(inout) :: cnt(n)
  integer :: iv(3), k
  iv = nint(rv)
  do k = 1, n
    if (sum((iv - rv) ** 2) > 1.0d-6) then
      cnt(k) = cnt(k) + 1
    end if
  end do
end subroutine driver
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert len(_reduce_nodes(sdfg)) >= 1, "SUM-in-condition should be a Reduce lib-node"
    n = 5
    cnt = np.zeros(n, dtype=np.int32)
    sdfg(rv=np.array([1.5, 2.0, 3.0], dtype=np.float64), cnt=cnt, n=np.int32(n))
    assert cnt.tolist() == [1] * n  # rv non-integer -> condition true every iter
    cnt2 = np.zeros(n, dtype=np.int32)
    sdfg(rv=np.array([1.0, 2.0, 3.0], dtype=np.float64), cnt=cnt2, n=np.int32(n))
    assert cnt2.tolist() == [0] * n  # integer-valued -> never


@_needs_flang
def test_if_maxval_of_array_diff(tmp_path):
    """``IF (MAXVAL(a - b) > thr)`` -- an array-op (elementwise subtract) feeding
    a MAXVAL reduction in the condition.  Materialises a ``Reduce`` (max)."""
    src = """
subroutine driver(a, b, cnt, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n), b(n)
  integer, intent(inout) :: cnt(n)
  integer :: k
  do k = 1, n
    if (maxval(a - b) > 0.5d0) then
      cnt(k) = cnt(k) + 1
    end if
  end do
end subroutine driver
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert len(_reduce_nodes(sdfg)) >= 1
    n = 4
    a = np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float64)
    b = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # max(a-b)=3 > 0.5
    cnt = np.zeros(n, dtype=np.int32)
    sdfg(a=a, b=b, cnt=cnt, n=np.int32(n))
    assert cnt.tolist() == [1] * n
    cnt2 = np.zeros(n, dtype=np.int32)
    b2 = a - 0.1  # max(a-b2)=0.1 < 0.5
    sdfg(a=a, b=b2.copy(), cnt=cnt2, n=np.int32(n))
    assert cnt2.tolist() == [0] * n


@_needs_flang
def test_reduction_in_loop_body(tmp_path):
    """A section reduction ``SUM(m(:, k))`` in the loop BODY (an assign, not a
    condition) -- the ordinary reduce path; guards that the condition-reduction
    change didn't perturb it."""
    src = """
subroutine driver(m, out, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: m(3, n)
  real(8), intent(out) :: out(n)
  integer :: k
  do k = 1, n
    out(k) = sum(m(:, k))
  end do
end subroutine driver
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    n = 4
    m = np.asfortranarray(np.arange(3 * n, dtype=np.float64).reshape(3, n))
    out = np.zeros(n, dtype=np.float64)
    sdfg(m=m, out=out, n=np.int32(n))
    assert np.allclose(out, m.sum(axis=0))


@_needs_flang
def test_minval_row_view_in_condition(tmp_path):
    """``IF (MINVAL(m(i, :)) > thr)`` -- a ROW section of a 2-D array feeds the
    MINVAL reduction.  The row becomes a DaCe VIEW (correct shape + column-major
    row stride) and the ``Reduce`` lib-node reduces the view (NOT the whole
    array).  Counts rows whose minimum exceeds the threshold."""
    src = """
subroutine driver(m, cnt, nr, nc)
  implicit none
  integer, intent(in) :: nr, nc
  real(8), intent(in) :: m(nr, nc)
  integer, intent(inout) :: cnt(nr)
  integer :: i
  do i = 1, nr
    if (minval(m(i, :)) > 0.5d0) then
      cnt(i) = cnt(i) + 1
    end if
  end do
end subroutine driver
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert len(_reduce_nodes(sdfg)) >= 1, "MINVAL of a row section should be a Reduce lib-node"
    from dace.data import View
    assert any(isinstance(d, View) for d in sdfg.arrays.values()), "row section should become a View"
    nr, nc = 3, 4
    m = np.asfortranarray(
        np.array(
            [
                [0.6, 0.7, 0.8, 0.9],  # row min 0.6 > 0.5 -> hit
                [0.1, 0.9, 0.9, 0.9],  # row min 0.1 -> no
                [0.51, 0.52, 0.53, 0.54]
            ],
            dtype=np.float64))  # min 0.51 -> hit
    cnt = np.zeros(nr, dtype=np.int32)
    sdfg(m=m, cnt=cnt, nr=np.int32(nr), nc=np.int32(nc))
    assert cnt.tolist() == [1, 0, 1]


@_needs_flang
def test_maxval_col_view_in_condition(tmp_path):
    """``IF (MAXVAL(m(:, j)) > thr)`` -- a COLUMN section (contiguous in
    column-major) becomes a VIEW reduced by the ``Reduce`` lib-node."""
    src = """
subroutine driver(m, cnt, nr, nc)
  implicit none
  integer, intent(in) :: nr, nc
  real(8), intent(in) :: m(nr, nc)
  integer, intent(inout) :: cnt(nc)
  integer :: j
  do j = 1, nc
    if (maxval(m(:, j)) > 0.5d0) then
      cnt(j) = cnt(j) + 1
    end if
  end do
end subroutine driver
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert len(_reduce_nodes(sdfg)) >= 1
    from dace.data import View
    assert any(isinstance(d, View) for d in sdfg.arrays.values()), "col section should become a View"
    nr, nc = 3, 4
    m = np.asfortranarray(np.array([[0.6, 0.1, 0.0, 0.9], [0.7, 0.2, 0.0, 0.9], [0.8, 0.3, 0.0, 0.9]],
                                   dtype=np.float64))
    # col maxes: 0.8, 0.3, 0.0, 0.9 -> >0.5: cols 0 and 3
    cnt = np.zeros(nc, dtype=np.int32)
    sdfg(m=m, cnt=cnt, nr=np.int32(nr), nc=np.int32(nc))
    assert cnt.tolist() == [1, 0, 0, 1]


@_needs_flang
def test_do_while_maxval_condition(tmp_path):
    """``DO WHILE (MAXVAL(a) > thr)`` -- a reduction in a LOOP condition,
    re-evaluated each iteration over a RUNTIME-extent array (which the old
    inline-unroll could not handle -> ``?`` for non-constant extent).  The body
    only advances scalars (``thr`` / ``iters``) so the array stays read-only and
    in-bounds; the loop runs ``ceil(maxval(a))`` times."""
    src = """
subroutine driver(a, iters, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  integer, intent(out) :: iters
  real(8) :: thr
  iters = 0
  thr = 0.0d0
  do while (maxval(a) > thr)
    thr = thr + 1.0d0
    iters = iters + 1
  end do
end subroutine driver
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert len(_reduce_nodes(sdfg)) >= 1, "MAXVAL in a loop condition should be a Reduce lib-node"
    n = 4
    a = np.array([3.0, 1.0, 2.5, 0.0], dtype=np.float64)  # maxval=3 -> thr 0,1,2 pass; stop at 3
    iters = np.zeros(1, dtype=np.int32)
    sdfg(a=a, iters=iters, n=np.int32(n))
    assert int(iters[0]) == 3
