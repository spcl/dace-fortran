"""End-to-end probes for ``MATMUL(TRANSPOSE(...))`` patterns and the
materialised-transient workaround the bridge currently expects.

QE's ``vcut_get`` has ``i_real = (MATMUL(TRANSPOSE(vcut%a), q)) / tpi`` --
when the matmul is inline inside a larger expression, ``buildExpr`` returns
``?`` (the libcall dispatcher only fires when matmul is the WHOLE assignment
RHS). ``hlfir-lift-reduction-operands`` detects this and errors, pointing at
the workaround: materialise to a temp (``tmp = MATMUL(TRANSPOSE(A), q)``,
whole-assign -> GEMM lib node) then divide the temp.

These probes pin that pattern: the bridge routes the whole-assign case through
``hlfir.matmul_transpose`` (TRANSPOSE+MATMUL fused by
``hlfir-optimized-bufferization``), and the libcall dispatcher emits GEMM with
the transpose flag threaded through -- no transposed-matrix materialisation.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_matmul_transpose_whole_assign_into_array_temp(tmp_path):
    """``tmp = MATMUL(TRANSPOSE(A), q)`` -- whole-assign routes through the GEMM lib
    node with the transpose flag. Result must match A.T @ q."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="matmul_t", entry="m::matmul_t").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    tmp = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, tmp=tmp)
    expected = A.T @ q
    np.testing.assert_allclose(tmp, expected)


def test_matmul_transpose_via_temp_then_scalar_div(tmp_path):
    """QE's vcut_get pattern materialised to a temp: tmp=MATMUL(TRANSPOSE(A),q)
    (whole-assign->GEMM), then res=tmp/scalar (element-wise). Workaround until the
    lift pass grows array-result materialisation for the inline form."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="vcut_pattern", entry="m::vcut_pattern").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, scalar=np.float64(2.0), res=res)
    expected = (A.T @ q) / 2.0
    np.testing.assert_allclose(res, expected)


def test_matmul_no_transpose_whole_assign(tmp_path):
    """Regression: ``MATMUL(A, B)`` without TRANSPOSE still routes through the plain matmul lib node."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mm", entry="m::mm").build()
    A = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float64, order='F')
    B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64, order='F')
    C = np.zeros((3, 2), dtype=np.float64, order='F')
    sdfg(a=A, b=B, c=C)
    expected = A @ B
    np.testing.assert_allclose(C, expected)


def test_inline_matmul_transpose_division_works_via_elemental_lift(tmp_path):
    """``res = MATMUL(TRANSPOSE(A), q) / scalar`` -- the bridge's elemental +
    hlfir.apply libcall materialisation (control_flow.cpp::walkElementalBody,
    libcallNameForExprOp recognising hlfir.matmul_transpose) pre-emits a
    _libtmp_<gid> transient, read element-by-element for the division. No
    Fortran-source rewrite needed; QE's vcut_get was the surfacing case."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="inline_qe", entry="m::inline_qe").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, scalar=np.float64(2.0), res=res)
    expected = (A.T @ q) / 2.0
    np.testing.assert_allclose(res, expected)


def test_matmul_transpose_libnode_is_dace_compatible(tmp_path):
    """Emitted MATMUL(TRANSPOSE(A), q) node must be DaCe-ABI compatible: transA set
    as a declared Property, NOT an __init__ kwarg (raised "MatMul.__init__() got
    an unexpected keyword argument 'transA'" on DaCe builds lacking it). Also pins
    the canonical _a/_b/_c connectors."""
    src = """
module m
contains
  subroutine mmt_node(a, q, tmp)
    real(kind=8), intent(in) :: a(3, 3), q(3)
    real(kind=8), intent(out) :: tmp(3)
    tmp = MATMUL(TRANSPOSE(a), q)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mmt_node", entry="m::mmt_node").build()
    matmuls = [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__ == "MatMul"]
    assert len(matmuls) == 1, f"expected exactly one MatMul lib node, got {len(matmuls)}"
    mm = matmuls[0]
    # The transpose folded onto the node as its declared Property (A-side).
    assert getattr(mm, "transA", False) is True, "transA must be set as a Property"
    assert getattr(mm, "transB", False) is False
    # Canonical DaCe matmul connector names.
    assert set(mm.in_connectors) == {"_a", "_b"}, dict(mm.in_connectors)
    assert set(mm.out_connectors) == {"_c"}, dict(mm.out_connectors)


def test_matmul_transpose_b_side_whole_assign(tmp_path):
    """B-side fold ``C = MATMUL(A, TRANSPOSE(B))``: same hlfir.matmul_transpose op
    (operands swapped), bridge threads transB=True onto the GEMM node -- no
    transient B^T materialised. Result must match A @ B.T."""
    src = """
module m
contains
  subroutine mmt_b(a, b, c)
    real(kind=8), intent(in) :: a(4, 3), b(5, 3)
    real(kind=8), intent(out) :: c(4, 5)
    c = MATMUL(a, TRANSPOSE(b))
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mmt_b", entry="m::mmt_b").build()
    rng = np.random.default_rng(2)
    A = np.asfortranarray(rng.standard_normal((4, 3)))
    B = np.asfortranarray(rng.standard_normal((5, 3)))
    C = np.zeros((4, 5), dtype=np.float64, order='F')
    sdfg(a=A, b=B, c=C)
    np.testing.assert_allclose(C, A @ B.T)


def test_matmul_transpose_b_side_libnode_transB(tmp_path):
    """B-side MATMUL(A, TRANSPOSE(B)) folds onto a single GEMM node with transB set
    (transA clear), no transient Transpose node -- symmetric to the A-side fold."""
    src = """
module m
contains
  subroutine mmt_b_node(a, b, c)
    real(kind=8), intent(in) :: a(4, 3), b(5, 3)
    real(kind=8), intent(out) :: c(4, 5)
    c = MATMUL(a, TRANSPOSE(b))
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mmt_b_node", entry="m::mmt_b_node").build()
    matmuls = [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__ == "MatMul"]
    assert len(matmuls) == 1, f"expected exactly one MatMul lib node, got {len(matmuls)}"
    mm = matmuls[0]
    assert getattr(mm, "transB", False) is True, "transB must be set as a Property"
    assert getattr(mm, "transA", False) is False
    transposes = [n for n, _ in sdfg.all_nodes_recursive() if "Transpose" in type(n).__name__]
    assert not transposes, f"B-side fold must allocate no transient transpose, got {transposes}"
    assert set(mm.in_connectors) == {"_a", "_b"}, dict(mm.in_connectors)
    assert set(mm.out_connectors) == {"_c"}, dict(mm.out_connectors)
