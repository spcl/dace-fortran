"""Fortran ``ALL`` / ``ANY`` lowering through the ``AllNode`` / ``AnyNode``
library nodes -- both as an assignment RHS and as an IF condition.

* Assignment ``res = ALL(mask)`` -> an ``AllNode`` writing a boolean
  scalar (``ALL`` / ``ANY`` return LOGICAL, mapped to ``bool``).
* ``IF (ALL(mask)) ...`` -> the reduction is materialised into a boolean
  scalar BEFORE the branch (``tryMaterialiseAllAnyCond`` in the bridge),
  and the IF reads that bare boolean scalar -- NOT the section inlined
  into the condition tasklet (which produced a malformed multi-dim
  memlet for QE's ``IF (ALL(odg(:)))``).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _libnode_names(sdfg):
    return {type(n).__name__ for st in sdfg.all_states() for n in st.nodes()}


@pytest.mark.parametrize("op,x,expected", [
    ("ALL", [1.0, 2.0, 3.0], True),
    ("ALL", [1.0, -1.0, 3.0], False),
    ("ANY", [-1.0, -2.0, -3.0], False),
    ("ANY", [-1.0, 2.0, -3.0], True),
])
def test_allany_assignment_emits_libnode(tmp_path, op, x, expected):
    """``res = ALL/ANY(mask)`` lowers to the AllNode/AnyNode and returns a
    boolean."""
    src = f"""
MODULE s_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE s(a, res, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  LOGICAL, INTENT(OUT) :: res
  LOGICAL :: mask(n)
  INTEGER :: i
  DO i = 1, n
    mask(i) = a(i) > 0.0D0
  END DO
  res = {op}(mask)
END SUBROUTINE
END MODULE s_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    assert {"AllNode", "AnyNode"} & _libnode_names(sdfg)
    res = np.zeros(1, dtype=np.bool_)
    a = np.asarray(x, dtype=np.float64, order="F")
    sdfg(a=a, res=res, n=np.int32(len(x)))
    assert bool(res[0]) == expected


@pytest.mark.parametrize("op,x,expected", [
    ("ALL", [1.0, 2.0, 3.0], 1),
    ("ALL", [1.0, -1.0, 3.0], 0),
    ("ANY", [-1.0, -2.0, -3.0], 0),
    ("ANY", [-1.0, 2.0, -3.0], 1),
])
def test_allany_in_if_condition(tmp_path, op, x, expected):
    """``IF (ALL/ANY(mask))`` materialises a boolean scalar via the libnode
    and branches on it."""
    src = f"""
MODULE s_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE s(a, res)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: a(3)
  INTEGER, INTENT(OUT) :: res
  LOGICAL :: mask(3)
  mask(1) = a(1) > 0.0D0
  mask(2) = a(2) > 0.0D0
  mask(3) = a(3) > 0.0D0
  IF ({op}(mask(:))) THEN
    res = 1
  ELSE
    res = 0
  END IF
END SUBROUTINE
END MODULE s_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    assert {"AllNode", "AnyNode"} & _libnode_names(sdfg)
    res = np.zeros(1, dtype=np.int32)
    a = np.asarray(x, dtype=np.float64, order="F")
    sdfg(a=a, res=res)
    assert int(res[0]) == expected
