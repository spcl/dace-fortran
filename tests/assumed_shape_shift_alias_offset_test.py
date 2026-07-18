"""``a(lb:)`` (fir.shift) aliased through an inlined ``b(:)`` callee must rebase reads by ``(lb_outer - 1)``.

Regression: ``declareLowerBounds`` classified the fir.shift bound but returned empty, skipping the rebase (read off by ``lb-1``).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _src(lb: int) -> str:
    return f"""
module m
  implicit none
contains
  subroutine inner(b, out)
    real(8), intent(in)  :: b(:)
    real(8), intent(out) :: out
    out = b(2)
  end subroutine inner
  subroutine outer(a, out)
    real(8), intent(in)  :: a({lb}:)
    real(8), intent(out) :: out
    call inner(a, out)
  end subroutine outer
end module m
"""


@pytest.mark.parametrize("lb", [0, -2, 3])
def test_assumed_shape_shift_alias_offset(tmp_path: Path, lb: int):
    """``b(2)`` reads the 2nd element regardless of ``lb`` (sequence association aliases ``b(1)`` to ``a``'s first element)."""
    sdfg = build_sdfg(_src(lb), tmp_path, name='outer', entry='outer').build()
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    sdfg(a=a, out=out)
    # b(1) aliases a's first element (a(lb)); b(2) -> a's second element = 20.0.
    assert out[0] == 20.0, f"lb={lb}: b(2) should read the 2nd element (20.0); got {out[0]}"
