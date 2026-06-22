"""End-to-end coverage for whole-array ("array-wide") operations and
inline library operands.

The bridge lowers whole-array Fortran expressions to element-wise DaCe
maps, and library intrinsics (reductions, matmul, transpose,
dot_product) to library nodes.  This module pins that BOTH compose
freely inside larger expressions:

  * chains of whole-array arithmetic  (``d = a + b - c``)
  * a reduction whose scalar result is re-broadcast over another
    whole-array op  (``out = c + MAXVAL(a + b - 1)``  --  the surfacing
    pattern: a library operand captured into the body of the consuming
    ``hlfir.elemental``, which the LiftReductionOperands pass must reach
    by descending into the elemental's region, not just its operands)
  * a library op nested inside whole-array arithmetic in either
    direction  (``c + MATMUL(a,b)`` and ``MATMUL(a+b, q)``)
  * whole-array ops on a STRUCT-COMPONENT subset  (``a(i)%w = a(i)%w -
    1`` must touch only row ``i`` of the flattened ``a_w`` companion,
    i.e. ``a_w[i, :] -= 1``)
  * an inline reduction inside a ``do while`` body (reached through the
    scf.while body walker, which now shares the structured reduction
    dispatch)

See ``while_loop_counter_e2e_test.py`` for the plain do-while
whole-array element-wise regression (the sourceless-copy miscompile).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


# --------------------------------------------------------------------------
# Chains of whole-array arithmetic
# --------------------------------------------------------------------------
def test_whole_array_chain_add_sub(tmp_path):
    """``d = a + b - c`` -- a three-operand whole-array chain."""
    src = """
subroutine chain3(n, a, b, c, d)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n), b(n), c(n)
  real(kind=8), intent(out) :: d(n)
  d = a + b - c
end subroutine
"""
    N = 5
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="chain3", entry="chain3").build()
    a = np.arange(1, N + 1, dtype=np.float64)
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    c = np.full(N, 2.0)
    d = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), b=np.asfortranarray(b), c=np.asfortranarray(c), d=d)
    np.testing.assert_allclose(d, a + b - c)


def test_whole_array_chain_mul_div(tmp_path):
    """``e = (a + b) * c - a / 2`` -- mixed precedence + scalar division
    over a whole-array chain."""
    src = """
subroutine chain_mul(n, a, b, c, e)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n), b(n), c(n)
  real(kind=8), intent(out) :: e(n)
  e = (a + b) * c - a / 2.0d0
end subroutine
"""
    N = 4
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="chain_mul", entry="chain_mul").build()
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 5.0, 5.0, 5.0])
    c = np.array([2.0, 3.0, 4.0, 5.0])
    e = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), b=np.asfortranarray(b), c=np.asfortranarray(c), e=e)
    np.testing.assert_allclose(e, (a + b) * c - a / 2.0)


# --------------------------------------------------------------------------
# Reduction -> scalar -> re-broadcast over a whole-array op
# --------------------------------------------------------------------------
def test_inline_maxval_rebroadcast(tmp_path):
    """``out = c + MAXVAL(a + b - 1)`` -- the user's surfacing example.

    The ``MAXVAL`` reduces a whole-array chain to a scalar, then the
    scalar is re-broadcast across ``c``.  In HLFIR the loop-invariant
    reduction is computed once OUTSIDE the consuming ``c + <scalar>``
    elemental and referenced as a region CAPTURE -- so the lift pass had
    to learn to descend into the elemental's body to find it (an
    operands-only walk missed it and the inline MAXVAL rendered ``?``)."""
    src = """
subroutine max_rb(n, a, b, c, out)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n), b(n), c(n)
  real(kind=8), intent(out) :: out(n)
  out = c + maxval(a + b - 1.0d0)
end subroutine
"""
    N = 4
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="max_rb", entry="max_rb").build()
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 5.0, 5.0, 5.0])
    c = np.array([100.0, 200.0, 300.0, 400.0])
    out = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), b=np.asfortranarray(b), c=np.asfortranarray(c), out=out)
    np.testing.assert_allclose(out, c + np.max(a + b - 1.0))


def test_inline_sum_rebroadcast(tmp_path):
    """``out = c * SUM(a * b)`` -- inline SUM over a whole-array product,
    re-broadcast multiplicatively."""
    src = """
subroutine sum_rb(n, a, b, c, out)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n), b(n), c(n)
  real(kind=8), intent(out) :: out(n)
  out = c * sum(a * b)
end subroutine
"""
    N = 4
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sum_rb", entry="sum_rb").build()
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 5.0, 5.0, 5.0])
    c = np.array([100.0, 200.0, 300.0, 400.0])
    out = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), b=np.asfortranarray(b), c=np.asfortranarray(c), out=out)
    np.testing.assert_allclose(out, c * np.sum(a * b))


def test_inline_two_reductions(tmp_path):
    """``out = c + MAXVAL(a) - MINVAL(b)`` -- two distinct reductions, each
    lifted to its own scalar temp and re-broadcast."""
    src = """
subroutine two_rb(n, a, b, c, out)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n), b(n), c(n)
  real(kind=8), intent(out) :: out(n)
  out = c + maxval(a) - minval(b)
end subroutine
"""
    N = 4
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="two_rb", entry="two_rb").build()
    a = np.array([1.0, 9.0, 3.0, 4.0])
    b = np.array([5.0, 2.0, 7.0, 8.0])
    c = np.array([100.0, 200.0, 300.0, 400.0])
    out = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), b=np.asfortranarray(b), c=np.asfortranarray(c), out=out)
    np.testing.assert_allclose(out, c + np.max(a) - np.min(b))


# --------------------------------------------------------------------------
# Library op nested in whole-array arithmetic (both directions)
# --------------------------------------------------------------------------
def test_inline_matmul_in_arithmetic(tmp_path):
    """``res = c + MATMUL(a, b)`` -- a matmul array result added to a
    whole-array operand."""
    src = """
module m
contains
  subroutine arr_plus_mm(a, b, c, res)
    real(kind=8), intent(in) :: a(2, 2), b(2, 2), c(2, 2)
    real(kind=8), intent(out) :: res(2, 2)
    res = c + matmul(a, b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="arr_plus_mm", entry="m::arr_plus_mm").build()
    A = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')
    B = np.array([[5.0, 6.0], [7.0, 8.0]], order='F')
    C = np.array([[100.0, 200.0], [300.0, 400.0]], order='F')
    res = np.zeros((2, 2), dtype=np.float64, order='F')
    sdfg(a=A, b=B, c=C, res=res)
    np.testing.assert_allclose(res, C + A @ B)


def test_inline_matmul_of_chain(tmp_path):
    """``res = MATMUL(a + b, q)`` -- the matmul's LEFT operand is itself a
    whole-array chain (materialised before the GEMM)."""
    src = """
module m
contains
  subroutine mm_of_chain(a, b, q, res)
    real(kind=8), intent(in) :: a(3, 3), b(3, 3), q(3)
    real(kind=8), intent(out) :: res(3)
    res = matmul(a + b, q)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mm_of_chain", entry="m::mm_of_chain").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], order='F')
    B = np.ones((3, 3), order='F')
    q = np.array([1.0, 2.0, 3.0], order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, b=B, q=q, res=res)
    np.testing.assert_allclose(res, (A + B) @ q)


def test_inline_dot_product(tmp_path):
    """``out = c + DOT_PRODUCT(a, b)`` -- a scalar-result library op
    (dot_product) lifted to a scalar temp and re-broadcast."""
    src = """
subroutine dot_inline(n, a, b, c, out)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n), b(n), c(n)
  real(kind=8), intent(out) :: out(n)
  out = c + dot_product(a, b)
end subroutine
"""
    N = 3
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="dot_inline", entry="dot_inline").build()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([10.0, 20.0, 30.0])
    out = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), b=np.asfortranarray(b), c=np.asfortranarray(c), out=out)
    np.testing.assert_allclose(out, c + np.dot(a, b))


def test_inline_transpose_in_arithmetic(tmp_path):
    """``res = b + TRANSPOSE(a)`` -- a transpose array result added to a
    whole-array operand."""
    src = """
module m
contains
  subroutine tr_inline(a, b, res)
    real(kind=8), intent(in) :: a(2, 2), b(2, 2)
    real(kind=8), intent(out) :: res(2, 2)
    res = b + transpose(a)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="tr_inline", entry="m::tr_inline").build()
    A = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')
    B = np.array([[10.0, 20.0], [30.0, 40.0]], order='F')
    res = np.zeros((2, 2), dtype=np.float64, order='F')
    sdfg(a=A, b=B, res=res)
    np.testing.assert_allclose(res, B + A.T)


# --------------------------------------------------------------------------
# Whole-array op on a struct-component subset  ->  a_w[i, :] op
# --------------------------------------------------------------------------
def test_struct_component_subset_decrement(tmp_path):
    """``a(i)%w = a(i)%w - 1`` where ``w`` is an array component.

    The flattened companion ``a_w`` has shape ``(K, len(w))``; the
    selector ``a(i)%w`` must lower to the ROW subset ``a_w[i-1, :]`` (0-
    based) so only row ``i`` is decremented and every other row is left
    untouched."""
    src = """
module m
  implicit none
  type t
    real(kind=8) :: w(3)
  end type t
contains
  subroutine dec_member(arr, i, k)
    integer, intent(in) :: i, k
    type(t), intent(inout) :: arr(k)
    arr(i)%w = arr(i)%w - 1.0d0
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="dec_member", entry="m::dec_member").build()
    assert "arr_w" in sdfg.arrays, f"expected flattened companion arr_w, got {list(sdfg.arrays)}"
    K = 4
    arr_w = np.arange(1, K * 3 + 1, dtype=np.float64).reshape(K, 3).copy(order='F')
    before = arr_w.copy()
    i = 2  # Fortran 1-based -> 0-based row 1
    sdfg(arr_w=arr_w, i=np.int32(i), k=np.int32(K), arr_w_d0=K)
    expected = before.copy()
    expected[i - 1, :] -= 1.0
    np.testing.assert_array_equal(arr_w, expected)
    # every other row untouched
    others = [r for r in range(K) if r != i - 1]
    np.testing.assert_array_equal(arr_w[others], before[others])


def test_struct_component_subset_chain(tmp_path):
    """``a(i)%w = a(i)%w * 2 + b(i)%w`` -- a whole-array CHAIN on a
    struct-component subset, mixing two different AoS companions on the
    same row ``i``."""
    src = """
module m
  implicit none
  type t
    real(kind=8) :: w(3)
  end type t
contains
  subroutine fuse_member(a, b, i, k)
    integer, intent(in) :: i, k
    type(t), intent(inout) :: a(k)
    type(t), intent(in) :: b(k)
    a(i)%w = a(i)%w * 2.0d0 + b(i)%w
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fuse_member", entry="m::fuse_member").build()
    assert "a_w" in sdfg.arrays and "b_w" in sdfg.arrays, list(sdfg.arrays)
    K = 3
    a_w = np.arange(1, K * 3 + 1, dtype=np.float64).reshape(K, 3).copy(order='F')
    b_w = np.full((K, 3), 10.0, dtype=np.float64, order='F')
    before = a_w.copy()
    i = 3
    sdfg(a_w=a_w, b_w=b_w, i=np.int32(i), k=np.int32(K), a_w_d0=K, b_w_d0=K)
    expected = before.copy()
    expected[i - 1, :] = before[i - 1, :] * 2.0 + 10.0
    np.testing.assert_allclose(a_w, expected)
    others = [r for r in range(K) if r != i - 1]
    np.testing.assert_array_equal(a_w[others], before[others])


# --------------------------------------------------------------------------
# Inline reduction inside a do-while body (scf.while body walker)
# --------------------------------------------------------------------------
def test_inline_reduction_in_do_while(tmp_path):
    """``out = out + MAXVAL(a)`` inside a ``do while`` body.

    Reaches the scf.while body walker, which now routes the lifted
    ``_QQred_lift_N = MAXVAL(a)`` temp through the SAME reduction
    dispatch the structured path uses.  Three iterations accumulate
    ``3 * max(a)`` into every slot of ``out``."""
    src = """
subroutine redux_in_while(n, a, out)
  integer, intent(in) :: n
  real(kind=8), intent(in) :: a(n)
  real(kind=8), intent(inout) :: out(n)
  integer :: i
  i = 0
  do while (i < 3)
    out = out + maxval(a)
    i = i + 1
  end do
end subroutine
"""
    N = 4
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="redux_in_while", entry="redux_in_while").build()
    a = np.array([1.0, 9.0, 3.0, 4.0])
    out = np.zeros(N, dtype=np.float64, order='F')
    sdfg(n=np.int32(N), a=np.asfortranarray(a), out=out)
    np.testing.assert_allclose(out, np.full(N, 3.0 * np.max(a)))


# --------------------------------------------------------------------------
# Known gap: a library op feeding ANOTHER library op (matmul -> reduction)
# --------------------------------------------------------------------------
@pytest.mark.xfail(reason="reduction directly over an array-result library op "
                   "(SUM(MATMUL(a,b))) needs the matmul materialised into a "
                   "named transient before the reduce; tracked as M2/array-"
                   "result-linalg-lift. Library ops compose with ARITHMETIC in "
                   "both directions (c+MATMUL, MATMUL(a+b,q)); only library-"
                   "feeding-library is unsupported. Real code writes "
                   "tmp=MATMUL(a,b); s=SUM(tmp).",
                   strict=True)
def test_inline_reduction_of_matmul(tmp_path):
    """``out = c + SUM(MATMUL(a, b))`` -- a reduction whose source is an
    array-result library op.  The matmul result has no Fortran-source
    name, so the reduce node's source resolves empty."""
    src = """
module m
contains
  subroutine sum_of_mm(a, b, c, out)
    real(kind=8), intent(in) :: a(2, 2), b(2, 2), c(2)
    real(kind=8), intent(out) :: out(2)
    out = c + sum(matmul(a, b))
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="sum_of_mm", entry="m::sum_of_mm").build()
    A = np.array([[1.0, 2.0], [3.0, 4.0]], order='F')
    B = np.array([[5.0, 6.0], [7.0, 8.0]], order='F')
    c = np.array([1.0, 2.0], order='F')
    out = np.zeros(2, dtype=np.float64, order='F')
    sdfg(a=A, b=B, c=c, out=out)
    np.testing.assert_allclose(out, c + np.sum(A @ B))
