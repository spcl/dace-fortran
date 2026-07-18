"""Regression: two hlfir-inlined copies of a POINTER-rebind local must keep their own
rebind targets. Same-named copies from different call sites collapsed last-wins
(name-based keying end to end) and the pointer-view source trace stopped at the
inlined dummy's declare, so ``view_source`` had no SDFG descriptor -> bare View
AccessNodes -> validation failure. Fix: (1) suffix the 2nd+ pointer_view declare
with ``_pv<N>`` per copy, (2) peel through inlined-dummy declares to root storage.
Kernel inlines ``bump`` twice with different rebind targets; correct lowering
increments BOTH arrays.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_pv
  implicit none
contains
  subroutine bump(dst, n)
    integer, intent(in) :: n
    real(8), intent(inout), target :: dst(:)
    real(8), pointer :: p(:)
    integer :: i
    p => dst
    do i = 1, n
      p(i) = p(i) + 1.0d0
    end do
  end subroutine bump

  subroutine run(a, b, n)
    integer, intent(in) :: n
    real(8), intent(inout), target :: a(:), b(:)
    call bump(a, n)
    call bump(b, n)
  end subroutine run
end module mo_pv
"""


def test_inlined_pointer_rebind_copies_keep_their_targets(tmp_path: Path):
    """Both inlined copies' rebinds land on their own actual: a AND b bump."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="pvcopies", entry="mo_pv::run").build()
    n = 6
    a = np.zeros(n, dtype=np.float64, order="F")
    b = np.full(n, 10.0, dtype=np.float64, order="F")
    sdfg(a=a, b=b, n=np.int32(n), a_d0=n, b_d0=n)
    np.testing.assert_allclose(a, np.ones(n), rtol=0, atol=0)
    np.testing.assert_allclose(b, np.full(n, 11.0), rtol=0, atol=0)
