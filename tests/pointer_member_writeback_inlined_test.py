"""Regression: writes through an inlined-callee dummy bound to a WHOLE POINTER struct member
must reach the member.

The ICON ``solve_free_sfc`` idiom: ``st%p_diag%vort`` (a ``REAL(8), POINTER :: (:,:)`` member)
is passed whole to an inlined worker with a contiguous explicit-shape dummy ``v(n1,n2)`` that
WRITES it. Flang guards contiguity with an ``hlfir.copy_in`` / ``hlfir.copy_out`` pair; the
bridge models neither, so the temp surfaced as a phantom argument ``v`` and the write-back was
dropped -- the member never changed (and calling with the member name raised ``KeyError: 'v'``).

``hlfir-fold-copy-in-out`` now reparents the alias accesses onto the member box directly
(``hlfir.designate %memberBox (i, j)``), which honours the component's strides and erases the
phantom ``v``. A per-element pattern (``i + 100*j``) catches a dropped/transposed index.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_wb
  implicit none
  type diag_t
    real(8), pointer :: vort(:, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine compute_vort(v, n1, n2)
    integer, intent(in) :: n1, n2
    real(8), intent(inout) :: v(n1, n2)
    integer :: i, j
    do j = 1, n2
      do i = 1, n1
        v(i, j) = real(i, 8) + 100.0d0 * real(j, 8)
      end do
    end do
  end subroutine compute_vort
  subroutine run(st, n1, n2)
    type(state_t), intent(inout) :: st
    integer, intent(in) :: n1, n2
    call compute_vort(st % p_diag % vort, n1, n2)
  end subroutine run
end module mo_wb
"""


def test_writeback_through_whole_pointer_member_inlined(tmp_path: Path):
    """The inlined worker's writes to ``st%p_diag%vort`` must land in the member. Without the
    reparent the member is a phantom ``v`` arg and the writes vanish (call raises KeyError)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="run", entry="mo_wb::run").build()
    sdfg.validate()

    # The member is the SDFG arg -- no phantom ``v``.
    assert "v" not in sdfg.arrays, f"phantom copy-in temp leaked as an SDFG array: {sorted(sdfg.arrays)}"

    n1, n2 = 3, 4
    vort = np.zeros((n1, n2), dtype=np.float64, order="F")
    sdfg(st_p_diag_vort=vort, n1=np.int32(n1), n2=np.int32(n2))

    i = np.arange(1, n1 + 1, dtype=np.float64)[:, None]
    j = np.arange(1, n2 + 1, dtype=np.float64)[None, :]
    expected = i + 100.0 * j
    np.testing.assert_array_equal(vort, expected)
