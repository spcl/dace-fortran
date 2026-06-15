"""Fortran ``CSHIFT`` (circular shift) end-to-end coverage.

The bridge routes ``hlfir.cshift`` to d-face's ``CShift`` library node;
``ExpandCShiftPure`` lowers it to a single Map whose source memlet
subset is ``Mod(Mod(i + shift, n) + n, n)`` (the doubled ``Mod`` keeps
a negative shift in range).  These tests pin the numerics against
``numpy.roll`` -- Fortran ``CSHIFT(arr, s)`` shifts LEFT by ``s``, i.e.
``np.roll(arr, -s)``.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(),
                                reason="flang-new-21 not on PATH")


def test_cshift_whole_array_positive(tmp_path):
    src = """
SUBROUTINE csh(arr, res, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  REAL(8), INTENT(OUT) :: res(n)
  res = CSHIFT(arr, 2)
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="csh", entry="csh").build()
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order="F")
    res = np.zeros(5, dtype=np.float64, order="F")
    sdfg(arr=arr, res=res, n=np.int32(5))
    np.testing.assert_allclose(res, np.roll(arr, -2))


def test_cshift_negative_shift(tmp_path):
    """Negative shift -- exercises the doubled ``Mod`` that keeps the
    rotated index non-negative."""
    src = """
SUBROUTINE csh_neg(arr, res, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  REAL(8), INTENT(OUT) :: res(n)
  res = CSHIFT(arr, -1)
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="csh_neg",
                      entry="csh_neg").build()
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order="F")
    res = np.zeros(5, dtype=np.float64, order="F")
    sdfg(arr=arr, res=res, n=np.int32(5))
    np.testing.assert_allclose(res, np.roll(arr, 1))


def test_cshift_inline_expr(tmp_path):
    """``2.0 - CSHIFT(arr, 1)`` -- expr-producer feeding an elemental."""
    src = """
module m
contains
  subroutine cshe(arr, res)
    real(kind=8), intent(in) :: arr(5)
    real(kind=8), intent(out) :: res(5)
    res = 2.0d0 - CSHIFT(arr, 1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cshe",
                      entry="cshe").build()
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order="F")
    res = np.zeros(5, dtype=np.float64, order="F")
    sdfg(arr=arr, res=res)
    np.testing.assert_allclose(res, 2.0 - np.roll(arr, -1))
