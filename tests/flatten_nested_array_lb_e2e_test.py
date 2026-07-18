"""E8 regression: a flattened nested array member loses its non-default lower bound.

``outer_t.arr(2)`` of ``inner_t.v(0:3)``: ``hlfir-flatten-structs`` rewrites
``o%arr(i)%v(j)`` into flat companion ``o_arr_v`` whose synthesised declare
carries only extents (no bounds) -- the ``v(0:3)`` lower bound lived on the
per-access ``fir.shape_shift`` and was discarded, so ``resolveLowerBounds``
defaulted lb=1 and ``o%arr(1)%v(0)`` indexed element -1. ICON shape: nested
member + negative block lower bound. f2py can't wrap the dummy, so the
reference is the closed-form result; offset constants are the correctness signal.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mn
  implicit none
  type inner_t
    real(8) :: v(0:3)
  end type
  type outer_t
    type(inner_t) :: arr(2)
  end type
contains
  subroutine kn(o, out)
    type(outer_t), intent(inout) :: o
    real(8), intent(out) :: out
    out = o%arr(1)%v(0) + o%arr(2)%v(3)
  end subroutine kn
end module mn
"""


def test_flatten_nested_array_nondefault_lb(tmp_path: Path):
    """The flattened companion of ``inner%v(0:3)`` must carry lb 0 in
    its inner dimension (offset 0), not the default 1."""
    d = tmp_path / "sdfg"
    d.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, d, name="kn", entry="mn::kn").build()
    sdfg.validate()

    consts = dict(getattr(sdfg, "_fortran_offset_values", sdfg.constants))
    offs = {k: int(v) for k, v in consts.items() if k.startswith("offset_") and "arr_v" in k}
    # Companion is (arr dim, v dim): arr lb 1, v lb 0.
    assert offs.get("offset_o_arr_v_d0") == 1, offs
    assert offs.get("offset_o_arr_v_d1") == 0, (f"inner v(0:3) lower bound 0 lost in flattening; got {offs}")
