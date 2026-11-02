# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``optimize`` turns order-free updates into reductions, so the loops carrying them map.

ICON's velocity tendencies accumulate a ``MAX``, count with ``clip_count = clip_count + 1`` and set
``levmask(jb, jk) = .TRUE.`` inside loops over cells. Each is a reduction whose result does not depend
on the order the iterations combine in, so the loop is parallel and the result stays bit-identical.
"""

import numpy as np
import pytest
from dace.sdfg.state import LoopRegion

from dace_fortran.pipelines import optimize
from tests._util import build_sdfg

_SRC = """
SUBROUTINE clip(n, a, b, amax, cnt, hit)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(INOUT) :: b(n)
  REAL(8), INTENT(OUT) :: amax(1)
  INTEGER, INTENT(OUT) :: cnt(1)
  LOGICAL, INTENT(OUT) :: hit(1)
  INTEGER :: i
  amax(1) = 0.0D0
  cnt(1) = 0
  hit(1) = .FALSE.
  DO i = 1, n
    IF (a(i) > 0.5D0) THEN
      cnt(1) = cnt(1) + 1
      hit(1) = .TRUE.
      amax(1) = MAX(amax(1), ABS(a(i)))
      b(i) = 2.0D0 * a(i)
    END IF
  END DO
END SUBROUTINE clip
"""


_RESET_EACH_ITERATION = """
SUBROUTINE nest(n, a, peak)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n, n, n)
  REAL(8), INTENT(OUT) :: peak(n)
  REAL(8) :: top
  INTEGER :: i, j, k, hits
  DO k = 1, n
    top = 0.0D0
    DO j = 1, n
      hits = 0
      DO i = 1, n
        IF (a(i, j, k) > 0.5D0) hits = hits + 1
      END DO
      IF (hits == 0) CYCLE
      DO i = 1, n
        top = MAX(top, a(i, j, k))
      END DO
    END DO
    peak(k) = top
  END DO
END SUBROUTINE nest
"""


def test_value_reset_in_outer_loop_is_a_reduction_of_the_inner_loops(tmp_path):
    """``top`` and ``hits`` restart inside ``k`` and ``j``: reading them there does not observe a partial reduction."""
    sdfg = build_sdfg(_RESET_EACH_ITERATION, tmp_path / "sdfg", name="nest", entry="nest").build()
    n = 6
    a = np.random.default_rng(0).random((n, n, n)).copy(order="F")
    reference = np.zeros(n)
    sdfg(n=n, a=a, peak=reference)
    optimize(sdfg)
    assert not [b for b in sdfg.all_control_flow_regions(recursive=True) if isinstance(b, LoopRegion)]
    peak = np.zeros(n)
    sdfg(n=n, a=a, peak=peak)
    assert np.array_equal(peak, reference)


def test_max_count_and_flag_updates_map(tmp_path):
    """The loop maps, and the optimized SDFG matches the unoptimized one bit for bit."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="clip", entry="clip").build()
    rng = np.random.default_rng(0)
    n = 64
    a = rng.random(n)
    inputs = {
        "n": np.int32(n),
        "a": a,
        "b": np.zeros(n),
        "amax": np.zeros(1),
        "cnt": np.zeros(1, dtype=np.int32),
        "hit": np.zeros(1, dtype=np.int32),
    }
    optimize(sdfg, verify_inputs=inputs)
    loops = [r.label for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]
    assert not loops, f"order-free updates left loops sequential: {loops}"

    b, amax, cnt, hit = np.zeros(n), np.zeros(1), np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)
    sdfg(n=np.int32(n), a=a, b=b, amax=amax, cnt=cnt, hit=hit)
    picked = a > 0.5
    assert amax[0] == a[picked].max()
    assert cnt[0] == picked.sum()
    assert hit[0]
    assert np.array_equal(b, np.where(picked, 2.0 * a, 0.0))


_LOOP_MUST_STAY = {
    "running_max": """
SUBROUTINE running_max(n, a, b, x)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: b(n), x(1)
  INTEGER :: i
  x(1) = 0.0D0
  DO i = 1, n
    x(1) = MAX(x(1), a(i))
    b(i) = x(1)
  END DO
END SUBROUTINE
""",
    "search_flag": """
SUBROUTINE search_flag(n, a, b, f)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(INOUT) :: b(n)
  LOGICAL, INTENT(OUT) :: f(1)
  INTEGER :: i
  f(1) = .FALSE.
  DO i = 1, n
    IF (.NOT. f(1)) b(i) = a(i)
    IF (a(i) > 0.5D0) f(1) = .TRUE.
  END DO
END SUBROUTINE
""",
    "last_store": """
SUBROUTINE last_store(n, a, f)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  LOGICAL, INTENT(OUT) :: f(1)
  INTEGER :: i
  f(1) = .FALSE.
  DO i = 1, n
    IF (a(i) > 0.5D0) THEN
      f(1) = .TRUE.
    ELSE
      f(1) = .FALSE.
    END IF
  END DO
END SUBROUTINE
""",
    "clamp": """
SUBROUTINE clamp(n, a, b, x)
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n), b(n)
  REAL(8), INTENT(OUT) :: x(1)
  INTEGER :: i
  x(1) = 0.0D0
  DO i = 1, n
    x(1) = MAX(x(1), a(i))
    x(1) = MIN(x(1), b(i))
  END DO
END SUBROUTINE
""",
}


@pytest.mark.parametrize("name", _LOOP_MUST_STAY)
def test_observed_or_mixed_update_keeps_the_loop_sequential(tmp_path, name):
    """A read of the partial value, or a second operator on the container, is order-dependent: no reduction forms."""
    sdfg = build_sdfg(_LOOP_MUST_STAY[name], tmp_path / "sdfg", name=name, entry=name).build()
    optimize(sdfg)
    assert [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)], "loop was mapped"


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_max_count_and_flag_updates_map(Path(tmp))
        for routine in _LOOP_MUST_STAY:
            test_observed_or_mixed_update_keeps_the_loop_sequential(Path(tmp) / routine, routine)
