"""E2E numerical test for the POINTER rank-changing remap pattern: ``p(1:M, 1:K) =>
arr1d`` rebinds a multi-D pointer to a 1D target (flang: fir.embox+shape_shift).
The bridge detects it via MarkBoundsRemapViews, registers a View with column-major
reshape strides over the 1D buffer, and skips the redundant rebind tasklet. Pins
that every ``p(i, j)`` write lands at the correct flat offset inside ``arr1d``.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  integer, parameter :: M = 4, K = 3
  double precision, target :: arr1d(M*K)
contains
  subroutine fill()
    double precision, pointer :: p(:,:)
    integer :: i, j
    p(1:M, 1:K) => arr1d
    do j = 1, K
      do i = 1, M
        p(i, j) = real(i + 10 * j, kind=8)
      end do
    end do
  end subroutine fill
end module m
"""


def test_pointer_2d_view_of_1d_target_writes_at_correct_offsets(tmp_path):
    """``p(1:M, 1:K) => arr1d`` then ``p(i,j) = i+10*j``: column-major view strides
    (1, M) must flatten 2D [i,j] accesses to the correct linear offset."""
    M, K = 4, 3

    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="fill", entry="m::fill").build()

    arr = np.full((M * K, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr1d=arr)

    expected = np.empty(M * K, dtype=np.float64)
    for j in range(1, K + 1):
        for i in range(1, M + 1):
            lin = (i - 1) + M * (j - 1)
            expected[lin] = i + 10 * j

    np.testing.assert_array_equal(arr, expected)
