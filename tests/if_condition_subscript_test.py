"""Reproducer for the residual icon_loopnest_4 numerical mismatch.

ICON velocity-advection loopnest 4 mismatches because the bridge's
IF-condition extraction loses array subscripts. The guard

    IF (levelmask(jk) .OR. levelmask(jk + 1)) THEN

surfaces in generated C++ as

    if_cond_5 = (levelmask || levelmask);

with both subscripts collapsed to the bare array name -- DaCe evaluates
against the array pointer (always non-zero), so every iteration enters the IF.

Isolates the pattern: a .OR. of two neighbouring array reads inside an IF
guarding a per-iteration write. xfail captures the gap so it's noticed if extract_ast regresses.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_IF_OR_SRC = """
SUBROUTINE if_logical_or_neighbour(mask, out, n)
  IMPLICIT NONE
  INTEGER(KIND=4), VALUE :: n
  LOGICAL, INTENT(IN) :: mask(n)
  REAL(KIND=8), INTENT(OUT) :: out(n)
  INTEGER :: i
  out = 0.0D0
  DO i = 1, n - 1
    IF (mask(i) .OR. mask(i + 1)) THEN
      out(i) = 1.0D0
    END IF
  END DO
END SUBROUTINE
"""


def test_if_condition_with_array_subscripts(tmp_path: Path):
    """The bridge must preserve ``mask(i)`` / ``mask(i + 1)`` subscripts
    when capturing an IF condition that ORs them together."""
    n = 6
    # mask=[F,T,F,F,T,F] -- i=3 (F OR F -> F) is the only index where the bug (always non-zero array-pointer test) would misfire.
    mask = np.array([False, True, False, False, True, False], dtype=np.bool_)
    expected = np.array([1, 1, 0, 1, 1, 0], dtype=np.float64)

    # gfortran reference -- confirms the test's intent.
    mod = f2py_compile(_IF_OR_SRC, tmp_path / "ref", "if_or_ref")
    out_ref = mod.if_logical_or_neighbour(mask, n=n)
    np.testing.assert_array_equal(out_ref, expected)

    # SDFG: xfail expects this assertion to fail until extract_ast subscript-preservation is fixed.
    sdfg = build_sdfg(_IF_OR_SRC, tmp_path, name="if_logical_or_neighbour").build()
    out = np.zeros(n, dtype=np.float64, order="F")
    sdfg(mask=mask, out=out, n=n, i=0)
    np.testing.assert_array_equal(out, expected)
