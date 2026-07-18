"""Minimal reproducers for three HLFIR->SDFG bridge gaps surfaced by QE's ``vexx_bp_k_gpu``
kernel, each isolating one pattern as a fast regression guard independent of the full QE parse:

1. local allocatable SECTION bound (``bridge/ast/assigns.cpp`` box_dims handler)
2. PRESENT() on an inlined OPTIONAL dummy (``bridge/ast/expressions.cpp`` lowerIsPresent)
3. derived-type INTEGER member used as array size (``bridge/extract_vars.cpp`` shape-symbol snapshot)
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_local_allocatable_section_bound(tmp_path):
    """Whole-column SECTION of a local 2-D allocatable -> box_dims bound."""
    src = """
MODULE sect_mod
CONTAINS
SUBROUTINE sect(res, n, m, j)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, j
  REAL(8), INTENT(OUT) :: res(n)
  REAL(8), ALLOCATABLE :: arr(:, :)
  INTEGER :: i
  ALLOCATE(arr(n, m))
  DO i = 1, n
    arr(i, j) = REAL(i, 8) * 10.0D0
  END DO
  res = arr(:, j)
  DEALLOCATE(arr)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sect", entry="sect_mod::sect").build()
    n, m, j = 5, 3, 2
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(res=res, n=np.int32(n), m=np.int32(m), j=np.int32(j))
    np.testing.assert_allclose(res, (np.arange(1, n + 1) * 10.0))


def test_present_inlined_optional(tmp_path):
    """OPTIONAL array dummy of an inlined internal subprogram, called WITH the optional present -- ``PRESENT`` must fold to true."""
    src = """
MODULE m
CONTAINS
  SUBROUTINE driver(a, res, n)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    REAL(8), INTENT(IN) :: a(n)
    REAL(8), INTENT(OUT) :: res(n)
    CALL worker(res, n, a)
  END SUBROUTINE

  SUBROUTINE worker(res, n, opt)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: n
    REAL(8), INTENT(OUT) :: res(n)
    REAL(8), INTENT(IN), OPTIONAL :: opt(n)
    INTEGER :: i
    IF (PRESENT(opt)) THEN
      DO i = 1, n
        res(i) = opt(i) + 1.0D0
      END DO
    ELSE
      DO i = 1, n
        res(i) = -1.0D0
      END DO
    END IF
  END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    n = 4
    a = np.asarray(np.arange(1, n + 1), dtype=np.float64, order="F")
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(a=a, res=res, n=np.int32(n))
    np.testing.assert_allclose(res, a + 1.0)


def test_struct_int_member_as_size(tmp_path):
    """Derived-type INTEGER member used as an ALLOCATE size + loop bound."""
    src = """
MODULE m
  TYPE :: desc
    INTEGER :: n
  END TYPE
CONTAINS
  SUBROUTINE run(d, res)
    IMPLICIT NONE
    TYPE(desc), INTENT(IN) :: d
    REAL(8), INTENT(OUT) :: res(d%n)
    REAL(8), ALLOCATABLE :: tmp(:)
    INTEGER :: i
    ALLOCATE(tmp(d%n))
    DO i = 1, d%n
      tmp(i) = REAL(i, 8) * 2.0D0
    END DO
    DO i = 1, d%n
      res(i) = tmp(i)
    END DO
    DEALLOCATE(tmp)
  END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="run", entry="m::run").build()
    n = 6
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(d_n=np.int32(n), res=res)
    np.testing.assert_allclose(res, (np.arange(1, n + 1) * 2.0))


def test_float32_cast_with_symbol(tmp_path):
    """``inv = 1.0 / n`` mixes a REAL(4) literal with an INTEGER symbol, so the bridge emits
    ``dace.float32(...)`` casts; the bare ``dace`` name must NOT leak as a required free symbol."""
    src = """
MODULE s_mod
CONTAINS
SUBROUTINE s(n, a, res)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: res(n)
  REAL(8) :: inv
  INTEGER :: i
  inv = 1.0 / n
  DO i = 1, n
    res(i) = a(i) * inv
  END DO
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    n = 3
    a = np.asarray(np.arange(1, n + 1), dtype=np.float64, order="F")
    res = np.zeros(n, dtype=np.float64, order="F")
    sdfg(n=np.int32(n), a=a, res=res)
    np.testing.assert_allclose(res, a * (np.float32(1.0) / np.float32(n)))


def test_intrinsic_shadowing_local_variable_renamed(tmp_path):
    """LOCAL variable shadowing an intrinsic (``DOUBLE PRECISION :: max``) is renamed to
    ``var_max`` since flang resolved it as a variable (an ``hlfir.declare`` exists); a
    genuine ``max(...)`` intrinsic call is a separate op and unaffected."""
    src = """
MODULE shadow_mod
CONTAINS
SUBROUTINE shadow(a, b, res)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: a, b
  REAL(8), INTENT(OUT) :: res
  DOUBLE PRECISION :: max
  max = a * 2.0D0
  ! ``max`` here is the user variable; an intrinsic call on a DIFFERENT
  ! name still renders as the intrinsic.
  res = max + MIN(a, b)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="shadow", entry="shadow_mod::shadow").build()
    res = np.zeros(1, dtype=np.float64)
    sdfg(a=3.0, b=10.0, res=res)
    np.testing.assert_allclose(res[0], 3.0 * 2.0 + min(3.0, 10.0))


def test_intrinsic_shadowing_dummy_is_rejected(tmp_path):
    """DUMMY argument shadowing an intrinsic stays a hard error -- its name is the user-facing
    SDFG signature arg, so a silent rename would change the call ABI."""
    src = """
SUBROUTINE shadow_dummy(max, res)
  IMPLICIT NONE
  REAL(8), INTENT(IN) :: max
  REAL(8), INTENT(OUT) :: res
  res = max + 1.0D0
END SUBROUTINE
"""
    with pytest.raises(RuntimeError, match="collide with bridge-rendered"):
        build_sdfg(src, tmp_path / "sdfg", name="shadow_dummy", entry="shadow_dummy").build()


def test_rank_reducing_section_gather(tmp_path):
    """2-D rank-reducing-section vector-subscript gather (QE ``eigts1(mill(1, offset+1:blk), na)``):
    the gather index must render as the full 2-D element ``mill[1, offset+k]``, not the
    rank-deficient ``mill[k]`` -- a single index on a 2-D array is a RANGE, illegal on the
    interstate edge hosting the minted gather symbol. Fixed via ``buildIndexExpr`` +
    ``expandDesignateChain`` (bridge/ast/assigns.cpp).
    """
    src = """
MODULE g2d_mod
CONTAINS
SUBROUTINE g2d(eig, mill, na, n, offset, ld, res)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: na, n, offset, ld
  INTEGER, INTENT(IN) :: mill(3, n + offset)
  COMPLEX(8), INTENT(IN) :: eig(ld, na)
  COMPLEX(8), INTENT(OUT) :: res(n)
  res(1:n) = eig(mill(1, offset + 1:offset + n), na)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="g2d", entry="g2d_mod::g2d").build()
    na, n, offset, ld = 2, 4, 1, 6
    rng = np.random.default_rng(0)
    eig = np.asfortranarray(rng.standard_normal((ld, na)) + 1j * rng.standard_normal((ld, na)), dtype=np.complex128)
    mill = np.asfortranarray(np.zeros((3, n + offset), dtype=np.int32))
    mill[0, :] = rng.integers(1, ld + 1, size=n + offset).astype(np.int32)  # 1-based row indices
    res = np.zeros(n, dtype=np.complex128, order="F")
    sdfg(eig=eig, mill=mill, na=np.int32(na), n=np.int32(n), offset=np.int32(offset), ld=np.int32(ld), res=res)
    expected = np.array([eig[mill[0, offset + i] - 1, na - 1] for i in range(n)], dtype=np.complex128)
    np.testing.assert_allclose(res, expected)
