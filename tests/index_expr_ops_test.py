"""Index-expression operator coverage for ``buildIndexExpr``.

``bridge/ast/assigns.cpp::buildIndexExpr`` renders integer arithmetic inside
``arr(<expr>)`` subscripts and must keep parity with the value path
(``buildExpr``); a missing op bottoms out at ``?`` and the SDFG build fails
on an unresolved free symbol. Covers ``MOD`` -> Python ``%`` and width-cast
index pass-through. NOT yet supported: bitwise IAND/IOR/IEOR/ISHFT in a subscript.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_mod_in_array_index(tmp_path):
    """``a(MOD(i, n) + 1)`` -- the MOD lowers to ``arith.remsi``; the
    subscript must render as ``(i % n) + 1`` (sympy ``Mod``)."""
    src = """
MODULE mod_idx_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE mod_idx(a, out, n, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: out(m)
  INTEGER :: i
  DO i = 1, m
    out(i) = a(MOD(i, n) + 1)
  END DO
END SUBROUTINE
END MODULE mod_idx_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mod_idx", entry="mod_idx_mod::mod_idx").build()
    a = np.array([10.0, 20.0, 30.0], dtype=np.float64, order="F")
    out = np.zeros(5, dtype=np.float64, order="F")
    sdfg(a=a, out=out, n=np.int32(3), m=np.int32(5))
    # i=1: a(mod(1,3)+1)=a(2)=20 ; i=2: a(mod(2,3)+1)=a(3)=30
    # i=3: a(mod(3,3)+1)=a(1)=10 ; i=4: a(mod(4,3)+1)=a(2)=20
    # i=5: a(mod(5,3)+1)=a(3)=30
    np.testing.assert_array_equal(out, [20.0, 30.0, 10.0, 20.0, 30.0])


def test_kind4_index_extended_to_i64(tmp_path):
    """``INTEGER(4)`` loop counter used as a subscript is extended to i64 by
    Flang (``arith.extsi``); the index path must treat the cast as transparent."""
    src = """
MODULE k4_idx_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE k4_idx(a, out, n)
  IMPLICIT NONE
  INTEGER(4), INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: out
  INTEGER(4) :: i
  out = 0.0_8
  DO i = 1, n
    out = out + a(i)
  END DO
END SUBROUTINE
END MODULE k4_idx_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="k4_idx", entry="k4_idx_mod::k4_idx").build()
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64, order="F")
    out = np.zeros(1, dtype=np.float64)
    sdfg(a=a, out=out, n=np.int32(4))
    assert out[0] == 10.0


def test_mod_plus_offset_index_2d(tmp_path):
    """``a(MOD(i, 2) + 1, j)`` -- MOD in one dim of a 2-D subscript."""
    src = """
MODULE mod_idx_2d_mod
  IMPLICIT NONE
CONTAINS
SUBROUTINE mod_idx_2d(a, out, n, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: a(2, m)
  REAL(8), INTENT(OUT) :: out(m)
  INTEGER :: j
  DO j = 1, m
    out(j) = a(MOD(j, 2) + 1, j)
  END DO
END SUBROUTINE
END MODULE mod_idx_2d_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mod_idx_2d", entry="mod_idx_2d_mod::mod_idx_2d").build()
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    out = np.zeros(3, dtype=np.float64, order="F")
    sdfg(a=a, out=out, n=np.int32(3), m=np.int32(3))
    # j=1: a(mod(1,2)+1, 1)=a(2,1)=4 ; j=2: a(mod(2,2)+1,2)=a(1,2)=2
    # j=3: a(mod(3,2)+1,3)=a(2,3)=6
    np.testing.assert_array_equal(out, [4.0, 2.0, 6.0])
