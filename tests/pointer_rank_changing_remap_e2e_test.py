"""E2E numerical test for the POINTER rank-changing remap pattern.

``p(1:M, 1:K) => arr1d`` rebinds a multi-D Fortran POINTER to a 1D
target.  Flang lowers this as ``fir.embox`` + ``fir.shape_shift`` of
the 1D source into a multi-D pointer box.  The bridge:

  * detects the rebind via ``MarkBoundsRemapViews`` (now handles both
    ``fir.embox`` and ``fir.rebox`` forms);
  * surfaces ``v.bounds_remap_source`` + ``v.bounds_remap_total_extent``
    via ``extract_vars``;
  * registers the View with the pointer's own shape (multi-D) and
    column-major reshape strides over the 1D source's flat buffer in
    ``descriptors.py``;
  * skips the redundant rebind tasklet in ``emit_tasklet`` (the View
    + source -> view linking edge already establish the alias);
  * wires the source AccessNode -> View ViewAccessNode edge via the
    canonical DaCe ``'views'`` connector in ``access.py``.

This test pins the runtime semantics: every ``p(i, j)`` write must
land at the correct flat offset inside ``arr1d``.
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
    """``p(1:M, 1:K) => arr1d`` then ``p(i, j) = i + 10*j``.  The
    column-major view strides ``(1, M)`` must flatten 2D ``[i, j]``
    accesses to linear offsets inside the 1D target."""
    M, K = 4, 3

    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="fill", entry="_QMmPfill").build()

    arr = np.full((M * K, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr1d=arr)

    expected = np.empty(M * K, dtype=np.float64)
    for j in range(1, K + 1):
        for i in range(1, M + 1):
            lin = (i - 1) + M * (j - 1)
            expected[lin] = i + 10 * j

    np.testing.assert_array_equal(arr, expected)
