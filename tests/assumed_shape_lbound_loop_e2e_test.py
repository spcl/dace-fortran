"""E10 regression: assumed-shape ``ubound`` lowering (``box_dims#0 + box_dims#1 - 1``)
left the lower-bound result unhandled in ``control_flow.cpp``'s loop-bound ``buildExpr``,
falling to the ``"?"`` sentinel and producing a SyntaxError in DaCe's
``unique_loop_iterators``.  Pins: assumed-shape sum builds and equals ``sum(a)``."""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ``a(10:)`` -> ``fir.shift %c10`` shape operand; ``ubound(a,1)`` lowers to
# ``box_dims#0(lb) + box_dims#1(extent) - 1`` -- lb must resolve to ``offset_a_d0``, not ``?``.
_SRC = """
module sum_as_mod
contains
subroutine sum_as(a, out)
  implicit none
  real(8), intent(in)  :: a(10:)
  real(8), intent(out) :: out
  integer :: i
  out = 0.0d0
  do i = lbound(a, 1), ubound(a, 1)
    out = out + a(i)
  end do
end subroutine sum_as
end module sum_as_mod
"""


def _bind_free_syms(sdfg, n: int) -> dict:
    """Kwargs for every ``a_d<i>`` extent (= n) and ``offset_a_d<i>`` lower bound
    (= 1, the assumed-shape default)."""
    out = {}
    for k in sdfg.arglist():
        if k.startswith("offset_") and k.endswith(tuple(f"_d{i}" for i in range(4))):
            out[k] = np.int64(1)
        elif k == "a" or k == "out":
            continue
        elif k.endswith(tuple(f"_d{i}" for i in range(4))):
            out[k] = np.int64(n)
    return out


def test_assumed_shape_lbound_ubound_loop(tmp_path: Path):
    """``do i = lbound(a,1), ubound(a,1); out += a(i)`` over assumed-shape ``a(10:)``.
    Builds (no ``?`` leak) and equals ``sum(a)``."""
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, d, name="sum_as", entry="sum_as_mod::sum_as").build()
    sdfg.validate()

    n = 7
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.random(n))
    out = np.zeros(1, dtype=np.float64, order="F")
    sdfg(a=a.copy(order="F"), out=out, **_bind_free_syms(sdfg, n))

    np.testing.assert_allclose(out[0], a.sum(), rtol=1e-12, atol=1e-12)
