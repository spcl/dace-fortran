"""SUM/PRODUCT/MINVAL/MAXVAL of inline elementals must materialise via buildElementalAnyAllReduce.
Before the fix, the dispatcher only routed hlfir.any/hlfir.all through elemental-materialisation; SUM
et al. fell to buildReduceNode -> traceToDecl on the elemental's result (no backing declare) ->
"reduction source '' not registered as SDFG data". QE's vcut_get (vexx_bp_k_gpu) hits this via three
SUMs of inline elementals.

Fix (dispatch.cpp Mode-C routing): any reduction op whose first operand is an hlfir.elemental routes
through buildElementalAnyAllReduce, not just hlfir.any/hlfir.all."""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_sum_of_pow(tmp_path):
    """``SUM(q ** 2)`` -- inline elemental computing pow."""
    src = """
module m
contains
  subroutine driver(q, out)
    real(kind=8), intent(in) :: q(3)
    real(kind=8), intent(out) :: out
    out = SUM(q ** 2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(q=q, out=out)
    np.testing.assert_allclose(out[0], np.sum(q**2))


def test_sum_of_difference_squared(tmp_path):
    """SUM((a - b) ** 2) -- inline elemental over (a-b) composed with pow; QE's L2-norm-squared shape."""
    src = """
module m
contains
  subroutine driver(a, b, out)
    real(kind=8), intent(in) :: a(3), b(3)
    real(kind=8), intent(out) :: out
    out = SUM((a - b) ** 2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    b = np.array([0.5, 1.5, 2.5], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(a=a, b=b, out=out)
    np.testing.assert_allclose(out[0], np.sum((a - b)**2))


def test_sum_of_element_product(tmp_path):
    """SUM(a * b) -- inline elemental element-wise product; QE's dot-product shape."""
    src = """
module m
contains
  subroutine driver(a, b, out)
    real(kind=8), intent(in) :: a(3), b(3)
    real(kind=8), intent(out) :: out
    out = SUM(a * b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(a=a, b=b, out=out)
    np.testing.assert_allclose(out[0], np.sum(a * b))


def test_product_of_elemental(tmp_path):
    """PRODUCT(arr + 1) -- non-SUM reduction verifies the op-agnostic routing covers all reductions."""
    src = """
module m
contains
  subroutine driver(arr, out)
    integer, intent(in) :: arr(4)
    integer, intent(out) :: out
    out = PRODUCT(arr + 1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    arr = np.array([1, 2, 3, 4], dtype=np.int32, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out[0], np.prod(arr + 1))


def test_minval_of_elemental(tmp_path):
    """``MINVAL(arr - 2)`` -- MINVAL over elemental."""
    src = """
module m
contains
  subroutine driver(arr, out)
    integer, intent(in) :: arr(4)
    integer, intent(out) :: out
    out = MINVAL(arr - 2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    arr = np.array([5, 3, 7, 1], dtype=np.int32, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out[0], np.min(arr - 2))
