"""Pin various IF-condition shapes against the bridge -> SDFG path.

Covers scalar conditions, array-element conditions (lifted via the
per-occurrence-connector tasklet path), MAX/MIN over multiple array reads (the
graupel ``if_cond_38`` shape), and logical .AND./.OR. chains.

Failure mode: if the bridge lifts an array-referencing condition onto a plain
interstate-edge assignment, DaCe treats the array name as a Symbol -- the C++
codegen emits the data pointer where a scalar was expected and gfortran
rejects with ``double* > scalar`` type errors.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_if_scalar_compare_basic(tmp_path):
    """``IF (x > 0)`` -- trivial scalar condition, no lifting needed."""
    src = """
SUBROUTINE if_scalar(n, x, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: x
  REAL(8), INTENT(OUT) :: out(n)
  INTEGER :: i
  DO i = 1, n
    IF (x > 0.0d0) THEN
      out(i) = REAL(i, 8)
    ELSE
      out(i) = -REAL(i, 8)
    END IF
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_scalar', entry='if_scalar').build()
    out = np.zeros(4, dtype=np.float64, order='F')
    sdfg(n=np.int32(4), x=np.float64(2.5), out=out)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])


def test_if_array_element_compare(tmp_path):
    """``IF (a(i) > 0)`` -- array element read must lift via per-occurrence connector so codegen binds a scalar."""
    src = """
SUBROUTINE if_array_elem(n, a, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n)
  REAL(8), INTENT(OUT) :: out(n)
  INTEGER :: i
  DO i = 1, n
    IF (a(i) > 1.0d0) THEN
      out(i) = 100.0d0
    ELSE
      out(i) = 0.0d0
    END IF
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_array_elem', entry='if_array_elem').build()
    a = np.array([0.5, 1.5, 2.0, 0.1], dtype=np.float64, order='F')
    out = np.zeros(4, dtype=np.float64, order='F')
    sdfg(n=np.int32(4), a=a, out=out)
    np.testing.assert_array_equal(out, [0.0, 100.0, 100.0, 0.0])


def test_if_max_over_array_elements(tmp_path):
    """``IF (MAX(a(i), b(i), c(i)) > eps)`` -- the graupel ``if_cond_38`` shape; each read must lift via its own per-occurrence connector."""
    src = """
SUBROUTINE if_max3(n, a, b, c, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n), b(n), c(n)
  REAL(8), INTENT(OUT) :: out(n)
  REAL(8), PARAMETER :: eps = 1.0d-6
  INTEGER :: i
  DO i = 1, n
    IF (MAX(a(i), b(i), c(i)) > eps) THEN
      out(i) = 1.0d0
    ELSE
      out(i) = 0.0d0
    END IF
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_max3', entry='if_max3').build()
    a = np.array([0.0, 0.5, 1e-7, 0.0], dtype=np.float64, order='F')
    b = np.array([0.0, 0.0, 1e-8, 0.0], dtype=np.float64, order='F')
    c = np.array([1e-9, 0.0, 1e-9, 1e-7], dtype=np.float64, order='F')
    out = np.zeros(4, dtype=np.float64, order='F')
    sdfg(n=np.int32(4), a=a, b=b, c=c, out=out)
    # i=0: max(0, 0, 1e-9)=1e-9, not > 1e-6 -> 0
    # i=1: max(0.5, 0, 0)=0.5 > 1e-6 -> 1
    # i=2: max(1e-7, 1e-8, 1e-9)=1e-7, not > 1e-6 -> 0
    # i=3: max(0, 0, 1e-7)=1e-7, not > 1e-6 -> 0
    np.testing.assert_array_equal(out, [0.0, 1.0, 0.0, 0.0])


def test_if_logical_chain_mixed(tmp_path):
    """``IF (a(i) > 0 .AND. (b(i) < 1 .OR. c == 0))`` -- AND/OR chain mixing array reads and a scalar; each operand needs its own connector."""
    src = """
SUBROUTINE if_chain(n, a, b, c, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n), b(n)
  INTEGER, INTENT(IN) :: c
  REAL(8), INTENT(OUT) :: out(n)
  INTEGER :: i
  DO i = 1, n
    IF (a(i) > 0.0d0 .AND. (b(i) < 1.0d0 .OR. c == 0)) THEN
      out(i) = 1.0d0
    ELSE
      out(i) = 0.0d0
    END IF
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_chain', entry='if_chain').build()
    a = np.array([0.5, -0.5, 1.0, 0.1], dtype=np.float64, order='F')
    b = np.array([0.5, 0.5, 2.0, 2.0], dtype=np.float64, order='F')
    out = np.zeros(4, dtype=np.float64, order='F')
    sdfg(n=np.int32(4), a=a, b=b, c=np.int32(5), out=out)
    # i=0: a=0.5>0 AND (b=0.5<1 OR c=5==0) = T AND (T OR F) = T -> 1
    # i=1: a=-0.5>0? F -> 0
    # i=2: a=1.0>0 AND (b=2.0<1 OR c=5==0) = T AND (F OR F) = F -> 0
    # i=3: a=0.1>0 AND (b=2.0<1 OR c=5==0) = T AND (F OR F) = F -> 0
    np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 0.0])


def test_if_array_2d_with_indexed_read(tmp_path):
    """``IF (mat(i, j) > threshold)`` -- 2D array indexing in condition."""
    src = """
SUBROUTINE if_2d(n, m, mat, threshold, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: mat(n, m)
  REAL(8), INTENT(IN) :: threshold
  REAL(8), INTENT(OUT) :: out(n, m)
  INTEGER :: i, j
  DO j = 1, m
    DO i = 1, n
      IF (mat(i, j) > threshold) THEN
        out(i, j) = mat(i, j) * 2.0d0
      ELSE
        out(i, j) = 0.0d0
      END IF
    END DO
  END DO
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_2d', entry='if_2d').build()
    mat = np.array([[0.5, 2.5], [1.5, 0.1], [3.0, 0.0]], dtype=np.float64, order='F')
    out = np.zeros((3, 2), dtype=np.float64, order='F')
    sdfg(n=np.int32(3), m=np.int32(2), mat=mat, threshold=np.float64(1.0), out=out)
    expected = np.where(mat > 1.0, mat * 2.0, 0.0).astype(np.float64)
    np.testing.assert_allclose(out, expected)


def count_cond_hoists(sdfg) -> int:
    """How many interstate-edge assignments hoist a branch condition."""
    return sum(1 for e in sdfg.all_interstate_edges(recursive=True) for k in e.data.assignments
               if k.startswith('if_cond_'))


def test_if_repeated_guard_over_readonly_data_is_hoisted_once(tmp_path):
    """Two identical guards over never-written data share ONE hoisted symbol.

    QE's ``exx_bp`` guards both the definition and the use of ``qvan_init_nij``
    with ``IF (okvan .AND. .NOT. tqr)``.  Hoisting each occurrence separately
    left gcc unable to correlate the two, so the guarded use was reported
    ``-Wmaybe-uninitialized`` -- a false positive that a zero-warning build
    cannot absorb.  One symbol makes the correlation syntactic.
    """
    src = """
SUBROUTINE if_cse(n, flag_a, flag_b, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  LOGICAL, INTENT(IN) :: flag_a, flag_b
  REAL(8), INTENT(OUT) :: out(n)
  INTEGER :: i, k
  k = 0
  IF (flag_a .AND. .NOT. flag_b) THEN
    k = 3
  END IF
  DO i = 1, n
    out(i) = 0.0d0
  END DO
  IF (flag_a .AND. .NOT. flag_b) THEN
    DO i = 1, n
      out(i) = REAL(k, 8)
    END DO
  END IF
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_cse', entry='if_cse').build()
    assert count_cond_hoists(sdfg) == 1, "identical read-only guards must share one symbol"
    out = np.zeros(4, dtype=np.float64, order='F')
    sdfg(n=np.int32(4), flag_a=np.bool_(True), flag_b=np.bool_(False), out=out)
    np.testing.assert_array_equal(out, [3.0, 3.0, 3.0, 3.0])


def test_if_repeated_guard_over_written_data_is_not_shared(tmp_path):
    """A guard whose operand is REASSIGNED between the two IFs must be recomputed.

    Sharing here would freeze the first evaluation and take the wrong branch --
    the miscompile the read-only precondition exists to prevent.  ``x`` flips
    sign between the guards, so the two must disagree.
    """
    src = """
SUBROUTINE if_no_cse(n, x, out)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(INOUT) :: x
  REAL(8), INTENT(OUT) :: out(n)
  INTEGER :: i
  DO i = 1, n
    out(i) = 0.0d0
  END DO
  IF (x > 0.0d0) THEN
    DO i = 1, n
      out(i) = out(i) + 1.0d0
    END DO
  END IF
  x = -x
  IF (x > 0.0d0) THEN
    DO i = 1, n
      out(i) = out(i) + 10.0d0
    END DO
  END IF
END SUBROUTINE
"""
    sdfg = build_sdfg(src, tmp_path / 'sdfg', name='if_no_cse', entry='if_no_cse').build()
    assert count_cond_hoists(sdfg) == 2, "a guard over reassigned data must be recomputed"
    out = np.zeros(4, dtype=np.float64, order='F')
    x = np.array([2.5], dtype=np.float64)
    sdfg(n=np.int32(4), x=x, out=out)
    # x>0 on the first guard only: +1 each, never +10.
    np.testing.assert_array_equal(out, [1.0, 1.0, 1.0, 1.0])
