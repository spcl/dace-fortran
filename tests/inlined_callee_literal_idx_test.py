"""Lower-bound inference through an inlined subroutine call.

Pinpointed in ``velocity_full`` bisection: ICON passes a literal (e.g. ``-5``)
to an inlined callee, which stashes it into a local and indexes through it.
After ``hlfir-inline-all`` + ``hlfir-flatten-structs`` the designate index
becomes ``fir.load %local_decl`` rather than ``arith.constant -5``, so
``inferLowerBoundsFromLiteralAccesses`` misses it, the bridge defaults
``offset_arr_d0 = 1``, and ``arr(-5)`` lowers to ``arr[-6]`` -> runtime segfault.

Pins the inference to follow the inline-callee load/store chain.  Currently
only matches a direct ``arith.constant``; documents the gap as a regression gate.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_callee
  implicit none
  contains
  subroutine read_end_index(arr, irl_end, out)
    integer, allocatable, intent(in) :: arr(:)
    integer, intent(in) :: irl_end
    integer, intent(out) :: out
    integer :: local
    local = irl_end
    out = arr(local)
  end subroutine read_end_index
end module mo_callee

module outer_mod
  use mo_callee, only: read_end_index
  implicit none
  contains
  subroutine outer(arr, out)
    integer, allocatable, intent(in) :: arr(:)
    integer, intent(out) :: out
    call read_end_index(arr, -5, out)
  end subroutine outer
end module outer_mod
"""


def test_inlined_callee_propagates_negative_literal(tmp_path: Path):
    """Literal ``-5`` passed to an inlined callee, stashed into a local, then indexed:
    inference must trace the ``fir.load``/store chain to recover ``-5`` for ``offset_arr_d0``."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_SRC, sdfg_dir, name="outer", entry="outer_mod::outer").build()
    sdfg.validate()

    inferred_offset = dict(getattr(sdfg, "_fortran_offset_values", sdfg.constants)).get('offset_arr_d0')
    assert inferred_offset == -5, (f"expected offset_arr_d0 == -5 (literal propagated through "
                                   f"inlined subroutine + load/store chain); got {inferred_offset}.  "
                                   f"This is the bridge gap identified in velocity_full bisection.")

    arr = np.asfortranarray(np.array([100, 200, 300, 400, 500], dtype=np.int32))  # 5 elements
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out, arr_d0=np.int64(5))
    assert out[0] == 100, f"arr(-5) (first element with lb=-5) should be 100; got {out[0]}"
