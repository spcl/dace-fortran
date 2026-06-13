"""E2E test for the same-rank POINTER bounds-remap pattern:
``ptr(:) => parent(:, k)``.

The pointer and target have the SAME rank but the rebind picks a
specific column ``k`` of the parent.  Each rebind shifts ``ptr``'s
view to a different column slice; the bridge represents this as a
1D View with a DYNAMIC ``offset_<ptr>_d0`` symbol bound per
surrounding-loop iteration via interstate edges.

Pattern documented at ``MarkBoundsRemapViews.cpp`` for QE's
``addusxx_g`` ``prhoc_d`` rebinds.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_pointer_same_rank_column_remap_writes_to_correct_column(tmp_path):
    """``ptr(:) => arr2d(:, k)`` then ``ptr(i) = ...``.  Writes through
    the column-rebind must land in the correct column of the parent."""
    src = """
module m
  implicit none
  integer, parameter :: N = 4, K = 3
  double precision, target :: arr2d(N, K)
contains
  subroutine fill()
    double precision, pointer :: p(:)
    integer :: i, j
    do j = 1, K
      p(1:N) => arr2d(:, j)
      do i = 1, N
        p(i) = real(i + 10 * j, kind=8)
      end do
    end do
  end subroutine fill
end module m
"""
    N, K = 4, 3
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="_QMmPfill").build()
    # arr2d laid out in Fortran column-major
    arr = np.full((N, K), -1.0, dtype=np.float64, order='F')
    sdfg(arr2d=arr)
    expected = np.empty((N, K), dtype=np.float64, order='F')
    for j in range(1, K + 1):
        for i in range(1, N + 1):
            expected[i - 1, j - 1] = i + 10 * j
    np.testing.assert_array_equal(arr, expected)
