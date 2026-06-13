"""End-to-end probes for ``MATMUL(TRANSPOSE(...))`` patterns and the
materialised-transient workaround the bridge currently expects.

Surfacing context: QE's ``vcut_get`` (and ``vexx_bp_k_gpu`` callees)
contain::

    i_real = (MATMUL(TRANSPOSE(vcut % a), q)) / tpi

When the matmul appears INLINE inside the larger expression, the
bridge's ``buildExpr`` returns ``?`` for the matmul value (the
libcall dispatcher only fires when the matmul is the WHOLE
assignment RHS).  ``hlfir-lift-reduction-operands`` now detects this
shape and emits a loud error pointing at the materialise-to-temp
workaround::

    ! WAS (silently broken): tasklet body contains ``?``
    ! res = MATMUL(TRANSPOSE(A), q) / scalar

    ! WORKAROUND (the temp gives the libcall dispatcher a target):
    DOUBLE PRECISION :: tmp(N)
    tmp = MATMUL(TRANSPOSE(A), q)   ! whole-assign -> GEMM lib node
    res = tmp / scalar              ! element-wise division on the temp

These probes pin the working transient pattern.  The bridge routes
the whole-assign case through ``hlfir.matmul_transpose`` (the
``hlfir-optimized-bufferization`` pass fuses the
``TRANSPOSE`` + ``MATMUL`` into a single op), then the libcall
dispatcher emits the corresponding GEMM lib node WITHOUT
materialising the transposed input matrix -- the transpose flag is
threaded through to the GEMM kernel.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_matmul_transpose_whole_assign_into_array_temp(tmp_path):
    """``tmp = MATMUL(TRANSPOSE(A), q)`` -- the libcall dispatcher
    sees the whole-assign, routes through the GEMM lib node with
    the transpose flag.  Result must match ``A.T @ q``."""
    src = """
module m
contains
  subroutine matmul_t(a, q, tmp)
    real(kind=8), intent(in) :: a(3, 3), q(3)
    real(kind=8), intent(out) :: tmp(3)
    tmp = MATMUL(TRANSPOSE(a), q)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="matmul_t", entry="_QMmPmatmul_t").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    tmp = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, tmp=tmp)
    expected = A.T @ q
    np.testing.assert_allclose(tmp, expected)


def test_matmul_transpose_via_temp_then_scalar_div(tmp_path):
    """The QE ``vcut_get`` pattern materialised to a temp::

        tmp = MATMUL(TRANSPOSE(A), q)   ! whole-assign -> GEMM
        res = tmp / scalar              ! element-wise

    Workaround for the inline ``MATMUL(TRANSPOSE(...)) / scalar``
    shape until the lift pass grows the array-result materialisation."""
    src = """
module m
contains
  subroutine vcut_pattern(a, q, scalar, res)
    real(kind=8), intent(in) :: a(3, 3), q(3), scalar
    real(kind=8), intent(out) :: res(3)
    real(kind=8) :: tmp(3)
    tmp = MATMUL(TRANSPOSE(a), q)
    res = tmp / scalar
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="vcut_pattern", entry="_QMmPvcut_pattern").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, scalar=np.float64(2.0), res=res)
    expected = (A.T @ q) / 2.0
    np.testing.assert_allclose(res, expected)


def test_matmul_no_transpose_whole_assign(tmp_path):
    """Regression: ``MATMUL(A, B)`` without TRANSPOSE still routes
    through the plain matmul lib node."""
    src = """
module m
contains
  subroutine mm(a, b, c)
    real(kind=8), intent(in) :: a(3, 4), b(4, 2)
    real(kind=8), intent(out) :: c(3, 2)
    c = MATMUL(a, b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mm", entry="_QMmPmm").build()
    A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float64, order='F')
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64, order='F')
    C = np.zeros((3, 2), dtype=np.float64, order='F')
    sdfg(a=A, b=B, c=C)
    expected = A @ B
    np.testing.assert_allclose(C, expected)


def test_inline_matmul_transpose_division_works_via_elemental_lift(tmp_path):
    """``res = MATMUL(TRANSPOSE(A), q) / scalar`` -- the bridge's
    elemental + ``hlfir.apply`` libcall materialisation
    (``control_flow.cpp::walkElementalBody``, with
    ``libcallNameForExprOp`` recognising ``hlfir.matmul_transpose``)
    pre-emits a ``_libtmp_<gid>`` transient holding the matmul
    result, and the consuming elemental reads it element-by-element
    for the division.  No Fortran-source rewrite needed.  QE's
    ``vcut_get`` was the surfacing case."""
    src = """
module m
contains
  subroutine inline_qe(a, q, scalar, res)
    real(kind=8), intent(in) :: a(3, 3), q(3), scalar
    real(kind=8), intent(out) :: res(3)
    res = MATMUL(TRANSPOSE(a), q) / scalar
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="inline_qe", entry="_QMmPinline_qe").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, scalar=np.float64(2.0), res=res)
    expected = (A.T @ q) / 2.0
    np.testing.assert_allclose(res, expected)
