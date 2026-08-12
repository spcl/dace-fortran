"""Regression: a whole-array ALLOCATABLE RHS assigned into a rank-reducing LHS section transposed.

The ICON halo-exchange body (``mo_sync``/``sync_patch_array``) packs an owned column into a local
allocatable buffer and lands a neighbour's column back through an assumed-shape dummy:

    sbuf = arr(:, 1)      ! pack owned column  -> whole allocatable
    arr(:, 2) = sbuf      ! land into halo column <- whole allocatable

``arr(:, 2) = sbuf`` is ``<rank-reducing section> = <whole allocatable>``. An allocatable RHS reaches
the ``hlfir.assign`` as a ``fir.load`` of its descriptor box, not a bare ``hlfir.declare`` -- so
``buildSectionToSectionAssign`` bailed and the dispatcher fell back to ``buildCopyNode``'s whole-array
copy, scattering the ``n1``-element buffer across all ``n1*nb`` cells of ``arr`` (the 2-rank ocean
``prog_h`` halo transpose). An AUTOMATIC ``sbuf(SIZE(arr,1))`` in the same role is a direct declare and
never hit the bug, so the two buffer kinds are the discriminator.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

# ``h`` is a POINTER member so the section is passed by copy-in/out and the dummy is assumed-shape;
# the buffer is ALLOCATABLE so its RHS box is a ``fir.load``, the shape the fix keys on.
_SRC = """
module mo_halo
  implicit none
  type diag_t
    real(8), pointer :: h(:, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine sync2d(arr)
    real(8), intent(inout) :: arr(:, :)
    real(8), allocatable :: sbuf(:)
    integer :: n1
    n1 = SIZE(arr, 1)
    allocate(sbuf(n1))
    sbuf = arr(:, 1)
    arr(:, 2) = sbuf
    deallocate(sbuf)
  end subroutine sync2d
  subroutine run(st)
    type(state_t), intent(inout) :: st
    call sync2d(st % p_diag % h)
  end subroutine run
end module mo_halo
"""


def test_allocatable_buffer_column_unpack(tmp_path: Path):
    """``arr(:, 2) = sbuf`` must copy column 1 into column 2 element-wise, not scatter the buffer."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="run", entry="mo_halo::run").build()
    sdfg.validate()

    n1, nb = 5, 3
    h = np.zeros((n1, nb), dtype=np.float64, order="F")
    h[:, 0] = np.arange(1, n1 + 1, dtype=np.float64) * 10.0
    sdfg(st_p_diag_h=h)

    expected = np.zeros((n1, nb), dtype=np.float64, order="F")
    expected[:, 0] = np.arange(1, n1 + 1, dtype=np.float64) * 10.0
    expected[:, 1] = expected[:, 0]  # arr(:, 2) = arr(:, 1) via the buffer
    np.testing.assert_array_equal(h, expected)
