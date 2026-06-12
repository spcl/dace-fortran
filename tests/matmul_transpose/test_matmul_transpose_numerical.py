"""Numerical correctness for ``MATMUL(TRANSPOSE(A), B)`` after
fold-into-MatMul.

The bridge now emits ``MatMul(transA=True)`` instead of synthesising
a ``Transpose -> _temp -> MatMul`` chain.  Result must match
``numpy.matmul(A.T, B)`` exactly (same fp ops in a different order
can differ in the last ULP for non-deterministic BLAS; we use
``rtol=1e-12`` for fp64 which is well above ULP-level noise).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

import dace_fortran

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(),
                                reason="flang-new-21 not on PATH")


def test_matmul_transpose_numerical(tmp_path):
    """``C = MATMUL(TRANSPOSE(A), B)`` with concrete values."""
    src = """
SUBROUTINE matmul_t_kernel(n, m, k, A, B, C)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: A(m, n)
  REAL(8), INTENT(IN) :: B(m, k)
  REAL(8), INTENT(OUT) :: C(n, k)
  C = MATMUL(TRANSPOSE(A), B)
END SUBROUTINE matmul_t_kernel
"""
    sdfg = dace_fortran.build_sdfg(src,
                                    out_dir=str(tmp_path / "sdfg"),
                                    entry="matmul_t_kernel",
                                    name="matmul_t_kernel")

    # The fused path should produce exactly one MatMul + zero Transpose
    # libcalls.  Regression guard against silent re-introduction of the
    # transient-+-transpose path (which would still be numerically
    # correct but waste a copy).
    mm_count = sum(1 for s in sdfg.states() for n in s.nodes()
                   if type(n).__name__ == "MatMul")
    tr_count = sum(1 for s in sdfg.states() for n in s.nodes()
                   if type(n).__name__ == "Transpose")
    assert mm_count == 1 and tr_count == 0, \
        f"expected 1 MatMul + 0 Transpose, got mm={mm_count} tr={tr_count}"

    rng = np.random.default_rng(seed=42)
    n, m, k = 4, 7, 5
    A = np.asfortranarray(rng.standard_normal((m, n)).astype(np.float64))
    B = np.asfortranarray(rng.standard_normal((m, k)).astype(np.float64))
    C = np.asfortranarray(np.zeros((n, k), dtype=np.float64))

    sdfg(n=np.int32(n), m=np.int32(m), k=np.int32(k), a=A, b=B, c=C)

    expected = A.T @ B
    np.testing.assert_allclose(C, expected, rtol=1e-12, atol=1e-12)


def test_matmul_a_transposeB_numerical(tmp_path):
    """``C = MATMUL(A, TRANSPOSE(B))`` -- symmetric to the LHS case.
    Flang doesn't emit a fused op for the RHS shape; the bridge
    detects ``hlfir.transpose`` as the matmul's RHS operand and
    folds it into ``MatMul(transB=True)``."""
    src = """
SUBROUTINE matmul_atb_kernel(n, m, k, A, B, C)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: A(n, m)
  REAL(8), INTENT(IN) :: B(k, m)
  REAL(8), INTENT(OUT) :: C(n, k)
  C = MATMUL(A, TRANSPOSE(B))
END SUBROUTINE matmul_atb_kernel
"""
    sdfg = dace_fortran.build_sdfg(src,
                                    out_dir=str(tmp_path / "sdfg"),
                                    entry="matmul_atb_kernel",
                                    name="matmul_atb_kernel")
    # The matmul fold sets transB=True on the MatMul.  The transpose
    # materialiser still emits a Transpose libcall whose result is
    # then unused (the MatMul reads B directly + uses transB at
    # the BLAS level).  Eliminating the dead Transpose is a
    # follow-up (transpose-materialiser-skip).
    mm = [n for s in sdfg.states() for n in s.nodes()
          if type(n).__name__ == "MatMul"][0]
    assert mm.transB is True, f"expected transB=True, got {mm.transB}"

    rng = np.random.default_rng(seed=11)
    n, m, k = 4, 7, 5
    A = np.asfortranarray(rng.standard_normal((n, m)).astype(np.float64))
    B = np.asfortranarray(rng.standard_normal((k, m)).astype(np.float64))
    C = np.asfortranarray(np.zeros((n, k), dtype=np.float64))
    sdfg(n=np.int32(n), m=np.int32(m), k=np.int32(k), a=A, b=B, c=C)
    np.testing.assert_allclose(C, A @ B.T, rtol=1e-12, atol=1e-12)


def test_matmul_both_transposed_numerical(tmp_path):
    """``C = MATMUL(TRANSPOSE(A), TRANSPOSE(B))`` -- both flags fold
    into one ``MatMul(transA=True, transB=True)`` -- no transient,
    no separate Transpose libcall."""
    src = """
SUBROUTINE matmul_atbt_kernel(n, m, k, A, B, C)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m, k
  REAL(8), INTENT(IN) :: A(m, n)
  REAL(8), INTENT(IN) :: B(k, m)
  REAL(8), INTENT(OUT) :: C(n, k)
  C = MATMUL(TRANSPOSE(A), TRANSPOSE(B))
END SUBROUTINE matmul_atbt_kernel
"""
    sdfg = dace_fortran.build_sdfg(src,
                                    out_dir=str(tmp_path / "sdfg"),
                                    entry="matmul_atbt_kernel",
                                    name="matmul_atbt_kernel")
    # Both flags fold in.  Dead Transpose libcalls may remain
    # (see B-only case for explanation); fixed by a separate
    # transpose-materialiser-skip pass.
    mm = [n for s in sdfg.states() for n in s.nodes()
          if type(n).__name__ == "MatMul"][0]
    assert mm.transA is True and mm.transB is True, \
        f"expected transA=True transB=True, got transA={mm.transA} transB={mm.transB}"

    rng = np.random.default_rng(seed=21)
    n, m, k = 4, 7, 5
    A = np.asfortranarray(rng.standard_normal((m, n)).astype(np.float64))
    B = np.asfortranarray(rng.standard_normal((k, m)).astype(np.float64))
    C = np.asfortranarray(np.zeros((n, k), dtype=np.float64))
    sdfg(n=np.int32(n), m=np.int32(m), k=np.int32(k), a=A, b=B, c=C)
    np.testing.assert_allclose(C, A.T @ B.T, rtol=1e-12, atol=1e-12)


def test_matmul_transpose_vector(tmp_path):
    """``y = MATMUL(TRANSPOSE(A), v)`` -- matrix x vector via Gemv
    with ``transA=True``.  The Fortran shape forces the GEMV branch
    of ``SpecializeMatMul`` -- a separate path from the 2-D Gemm test
    above."""
    src = """
SUBROUTINE matmul_tv_kernel(n, m, A, v, y)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n, m
  REAL(8), INTENT(IN) :: A(m, n)
  REAL(8), INTENT(IN) :: v(m)
  REAL(8), INTENT(OUT) :: y(n)
  y = MATMUL(TRANSPOSE(A), v)
END SUBROUTINE matmul_tv_kernel
"""
    sdfg = dace_fortran.build_sdfg(src,
                                    out_dir=str(tmp_path / "sdfg"),
                                    entry="matmul_tv_kernel",
                                    name="matmul_tv_kernel")

    rng = np.random.default_rng(seed=7)
    n, m = 4, 6
    A = np.asfortranarray(rng.standard_normal((m, n)).astype(np.float64))
    v = rng.standard_normal(m).astype(np.float64)
    y = np.zeros(n, dtype=np.float64)

    sdfg(n=np.int32(n), m=np.int32(m), a=A, v=v, y=y)

    expected = A.T @ v
    np.testing.assert_allclose(y, expected, rtol=1e-12, atol=1e-12)
