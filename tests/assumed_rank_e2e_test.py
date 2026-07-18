"""E2e for assumed-rank dummy + ``SELECT RANK`` (F2018 ``DIMENSION(..)``).

``hlfir-fold-assumed-rank-queries`` folds ``fir.box_rank``/``fir.is_assumed_size``
on the rank-erased box to constants; canonicalize then collapses the
``SELECT RANK`` dispatch to the single branch matching the actual's static
rank.  Pins that rank-2 writes land at the right offsets.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  integer, parameter :: N = 4, M = 3
  double precision :: arr2d(N, M)
contains
  subroutine inner(a)
    double precision, intent(inout) :: a(..)
    integer :: i, j
    select rank (a)
    rank (1)
      do i = 1, size(a)
        a(i) = real(i, kind=8)
      end do
    rank (2)
      ! ``size(a, 1)`` / ``size(a, 2)`` lowers to ``fir.box_dims``;
      ! the bridge's fold pass replaces it with the concrete extents
      ! when the box traces back to a static-shape actual.
      do j = 1, size(a, 2)
        do i = 1, size(a, 1)
          a(i, j) = real(i + 10 * j, kind=8)
        end do
      end do
    rank default
    end select
  end subroutine inner

  subroutine outer()
    call inner(arr2d)
  end subroutine outer
end module m
"""


def test_assumed_rank_dispatches_to_rank2_branch(tmp_path):
    """Rank-2 ``SELECT RANK`` branch runs for a 2D actual; each ``a(i,j) = i+10*j`` write lands at the right slot."""
    N, M = 4, 3
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="outer", entry="m::outer").build()

    arr_sdfg = np.full((N, M), -1.0, dtype=np.float64, order='F')
    sdfg(arr2d=arr_sdfg)

    expected = np.empty((N, M), dtype=np.float64, order='F')
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            expected[i - 1, j - 1] = i + 10 * j

    np.testing.assert_array_equal(arr_sdfg, expected)
