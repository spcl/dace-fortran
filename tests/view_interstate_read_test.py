# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A View read by an interstate assignment: an assumed-size dummy ``ind(*)`` aliasing a rank-2 actual, read into a
scalar the next condition tests (QE h_psi's ``hpsort_eps`` on ``igk_exx``)."""

import numpy as np

from tests._util import build_sdfg

_SRC = """
MODULE view_read_mod
CONTAINS
SUBROUTINE pick(n, ind, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  INTEGER, INTENT(INOUT) :: ind(*)
  INTEGER, INTENT(OUT) :: out
  INTEGER :: l, iind
  out = 0
  DO l = 1, n
    iind = ind(l)
    IF (iind > 1) out = out + iind
  END DO
END SUBROUTINE
SUBROUTINE driver(igk, n, nks, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, nks
  INTEGER, INTENT(INOUT) :: igk(n, nks)
  INTEGER, INTENT(OUT) :: out
  INTEGER :: ik, o
  out = 0
  IF (nks == 1) THEN
    CALL pick(n, igk, o)
    out = o
  END IF
  IF (nks > 1) THEN
    DO ik = 1, nks
      CALL pick(n, igk(1, ik), o)
      out = out + o
    END DO
  END IF
END SUBROUTINE
END MODULE
"""


def test_assumed_size_view_read_on_an_interstate_edge(tmp_path):
    """Both calls: the whole actual (the View read on an interstate edge) and a column of it."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="view_interstate_read", entry="view_read_mod::driver").build()
    csdfg = sdfg.compile()
    for nks in (1, 2):
        igk = np.array(np.arange(1, 3 * nks + 1, dtype=np.int32).reshape(3, nks, order="F"), order="F")
        out = np.zeros(1, np.int32)
        csdfg(igk=igk, n=np.int32(3), nks=np.int32(nks), out=out)
        assert out[0] == sum(v for v in igk.ravel(order="F") if v > 1)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        test_assumed_size_view_read_on_an_interstate_edge(Path(tmp))
