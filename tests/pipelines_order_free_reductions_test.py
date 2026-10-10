# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``optimize`` turns order-free updates into reductions, so the loops carrying them map.

ICON's velocity tendencies accumulate a ``MAX``, count with ``clip_count = clip_count + 1`` and set
``levmask(jb, jk) = .TRUE.`` inside loops over cells. Each is a reduction whose result does not depend
on the order the iterations combine in, so the loop is parallel and the result stays bit-identical.
"""

import numpy as np
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


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_max_count_and_flag_updates_map(Path(tmp))
