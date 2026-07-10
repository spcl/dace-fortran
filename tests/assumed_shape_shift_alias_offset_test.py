"""Assumed-shape alias offset rebase across an explicit-lower-bound outer
(audit: trace_utils declareLowerBounds dropped fir.shift bounds).

An assumed-shape dummy with an explicit lower bound (``a(0:)``, ``a(-2:)``)
lowers to an ``hlfir.declare`` carrying a ``fir.shift`` shape (per-dim lower
bound, extent from the box).  When that array is passed to an inlined callee
whose own dummy is a default-based assumed-shape (``b(:)``), the callee access
``b(k)`` is resolved to the outer array and must be rebased by
``(lb_outer - 1)`` so the memlet lands on the right shared element.

``declareLowerBounds`` classified the ``fir.shift`` bound into ``si.lbs`` but
then returned empty for that kind, so the rebase was skipped and the aliased
read was off by ``(lb - 1)``.  With ``a(0:)`` and ``b(2)`` the old path read
element 2 (30.0) instead of element 1 (20.0).
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
    """``a(lb:)`` (fir.shift) passed to inlined ``inner(b(:))``; ``b(2)`` reads
    the SECOND element of the shared storage regardless of ``lb`` -- Fortran
    sequence association makes ``b(1)`` alias the first element of ``a``."""
    sdfg = build_sdfg(_src(lb), tmp_path, name='outer', entry='outer').build()
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    sdfg(a=a, out=out)
    # b(1) aliases a's first element (a(lb)); b(2) -> a's second element = 20.0.
    assert out[0] == 20.0, f"lb={lb}: b(2) should read the 2nd element (20.0); got {out[0]}"
