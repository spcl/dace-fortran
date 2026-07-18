"""Complex-constant lowering + 1j/variable-j disambiguation. Two invariants: (1) whole-array assignment
of a complex constant lowers to a mapped fill, not a scalar write; (2) the rendered `1j` must never be
confused with a real scalar/loop iterator literally named `j` (spurious _in_j connector)."""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_complex_2d_zero_fill(tmp_path):
    """COMPLEX(:,:); x = 0.0_dp -> whole 2-D array fill (not a 1-D scalar write)."""
    src = """
MODULE s_mod
CONTAINS
SUBROUTINE s(res, n, m)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  COMPLEX(8), INTENT(OUT) :: res(n, m)
  COMPLEX(8), ALLOCATABLE :: big(:, :)
  ALLOCATE(big(n, m))
  big = 0.0_8
  res = big
  DEALLOCATE(big)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    res = np.full((3, 2), 7.0 + 1j, dtype=np.complex128, order="F")
    sdfg(res=res, n=np.int32(3), m=np.int32(2))
    assert np.allclose(res, 0)


def test_complex_literal_fill(tmp_path):
    """``x = (1.0, 2.0)`` fills every element with ``1 + 2j``."""
    src = """
MODULE s_mod
CONTAINS
SUBROUTINE s(res, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  COMPLEX(8), INTENT(OUT) :: res(n)
  COMPLEX(8) :: a(n)
  a = (1.0_8, 2.0_8)
  res = a
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    res = np.zeros(5, dtype=np.complex128, order="F")
    sdfg(res=res, n=np.int32(5))
    assert np.allclose(res, 1.0 + 2.0j)


def test_complex_with_loop_iterator_j(tmp_path):
    """Loop iterator named `j` must coexist with the rendered `1j` -- imaginary unit is not the variable."""
    src = """
MODULE s_mod
CONTAINS
SUBROUTINE s(res, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  COMPLEX(8), INTENT(OUT) :: res(n)
  COMPLEX(8) :: a(n)
  INTEGER :: j
  a = (0.0_8, 0.0_8)
  DO j = 1, n
    a(j) = CMPLX(REAL(j, 8), 2.0_8, 8)
  END DO
  res = a
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    res = np.zeros(4, dtype=np.complex128, order="F")
    sdfg(res=res, n=np.int32(4))
    np.testing.assert_allclose(res, np.array([complex(i + 1, 2.0) for i in range(4)]))


def test_complex_scalar_j_plus_imaginary(tmp_path):
    """REAL scalar named `j` used alongside a complex constant -- `j` is read as data, `1j` is not."""
    src = """
MODULE s_mod
CONTAINS
SUBROUTINE s(j, res)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: j
  COMPLEX(8), INTENT(OUT) :: res
  res = j * (1.0_8, 0.0_8) + (0.0_8, 1.0_8)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    res = np.zeros(1, dtype=np.complex128)
    sdfg(j=3.0, res=res)
    np.testing.assert_allclose(res[0], 3.0 + 1.0j)
