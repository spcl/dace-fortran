"""Array reductions (SUM/MINVAL/MAXVAL/PRODUCT) in a CONDITION materialise into a scalar
transient via a Reduce lib-node BEFORE the branch, so the condition reads a scalar instead
of inline-unrolling the reduction.

Pins both STRUCTURE (Reduce lib-node + scalar condition) and end-to-end correctness, in DO
loops and loop conditions, with loop bodies touching only in-bounds memory so a miscompile
shows as a wrong result, not an OOB crash.
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
    """IF (SUM((iv-rv)**2) > eps) in a DO loop -- Gate-G pattern. Materialises a Reduce (sum)
    lib-node; the per-element squared diff must keep its subscript (no bare whole-array term)."""
    src = """
module driver_mod
  implicit none
contains
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
end module driver_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
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
    """IF (MAXVAL(a - b) > thr) -- elementwise subtract feeding a MAXVAL reduction; materialises a Reduce (max)."""
    src = """
module driver_mod
  implicit none
contains
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
end module driver_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
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
    """SUM(m(:, k)) in the loop BODY (not a condition) -- guards the ordinary reduce path
    against regression from the condition-reduction change."""
    src = """
module driver_mod
  implicit none
contains
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
end module driver_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
    n = 4
    m = np.asfortranarray(np.arange(3 * n, dtype=np.float64).reshape(3, n))
    out = np.zeros(n, dtype=np.float64)
    sdfg(m=m, out=out, n=np.int32(n))
    assert np.allclose(out, m.sum(axis=0))


@_needs_flang
def test_minval_row_view_in_condition(tmp_path):
    """IF (MINVAL(m(i, :)) > thr) -- a row section becomes a DaCe VIEW (correct shape + column-major
    stride) and Reduce reduces the view, not the whole array."""
    src = """
module driver_mod
  implicit none
contains
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
end module driver_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
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
module driver_mod
  implicit none
contains
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
end module driver_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
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
    """DO WHILE (MAXVAL(a) > thr) -- reduction re-evaluated each iteration over a runtime-extent
    array (old inline-unroll couldn't handle non-constant extent). Loop runs ceil(maxval(a)) times."""
    src = """
module driver_mod
  implicit none
contains
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
end module driver_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
    assert len(_reduce_nodes(sdfg)) >= 1, "MAXVAL in a loop condition should be a Reduce lib-node"
    n = 4
    a = np.array([3.0, 1.0, 2.5, 0.0], dtype=np.float64)  # maxval=3 -> thr 0,1,2 pass; stop at 3
    iters = np.zeros(1, dtype=np.int32)
    sdfg(a=a, iters=iters, n=np.int32(n))
    assert int(iters[0]) == 3
