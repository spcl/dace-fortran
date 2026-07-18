"""Multiple calls to the same inlined function in ONE expression must each get a
distinct result temporary. Flang emits distinct fir.call SSA results; the bridge must
not collapse them onto a single result scalar (else every read sees the last call).

Repro of the CLOUDSC scc_k_caching zbeta miscompile: ZPOW called 5x in one expression
aliased onto one zpow_res scalar, so pow(ztp1,2) got overwritten by pow(zlambda,c3r)
(~170x wrong), seeding the whole-kernel divergence.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_two_calls_distinct_args(tmp_path: Path):
    """y = sq(a) + sq(b): the two results must not alias -> a**2 + b**2, not 2*b**2."""
    src = """
module m
contains
  pure function sq(x) result(r)
    real(8), intent(in) :: x
    real(8) :: r
    r = x * x
  end function sq
  subroutine kern(a, b, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n), b(n)
    real(8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = sq(a(i)) + sq(b(i))
    end do
  end subroutine kern
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kern", entry="m::kern").build()
    n = 8
    rng = np.random.default_rng(0)
    a = np.asfortranarray(rng.random(n))
    b = np.asfortranarray(rng.random(n))
    y = np.zeros(n, order="F")
    sdfg(a=a, b=b, y=y, n=n)
    np.testing.assert_allclose(y, a**2 + b**2, atol=1e-13, rtol=0)


def test_power_helper_distinct_exponents(tmp_path: Path):
    """The CLOUDSC pattern: one helper called with different (base, exponent) pairs in
    a single expression. Each pw result is distinct; aliasing collapses them."""
    src = """
module m
contains
  pure function pw(base, expo) result(r)
    real(8), intent(in) :: base, expo
    real(8) :: r
    r = base ** expo
  end function pw
  subroutine kern(t, l, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: t(n), l(n)
    real(8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = pw(t(i), 2.0d0) * (0.78d0 / pw(l(i), 0.5d0) + pw(l(i), 0.25d0) / pw(t(i), 3.0d0))
    end do
  end subroutine kern
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kern", entry="m::kern").build()
    n = 8
    rng = np.random.default_rng(1)
    t = np.asfortranarray(2.0 + rng.random(n))
    l = np.asfortranarray(2.0 + rng.random(n))
    y = np.zeros(n, order="F")
    sdfg(t=t, l=l, y=y, n=n)
    ref = t**2.0 * (0.78 / l**0.5 + l**0.25 / t**3.0)
    np.testing.assert_allclose(y, ref, atol=0, rtol=1e-13)


def test_three_calls_same_arg_different_exponent(tmp_path: Path):
    """Same base, three different exponents in one expression -> three distinct results."""
    src = """
module m
contains
  pure function pw(base, expo) result(r)
    real(8), intent(in) :: base, expo
    real(8) :: r
    r = base ** expo
  end function pw
  subroutine kern(x, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = pw(x(i), 2.0d0) + pw(x(i), 3.0d0) + pw(x(i), 0.5d0)
    end do
  end subroutine kern
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kern", entry="m::kern").build()
    n = 8
    rng = np.random.default_rng(2)
    x = np.asfortranarray(2.0 + rng.random(n))
    y = np.zeros(n, order="F")
    sdfg(x=x, y=y, n=n)
    np.testing.assert_allclose(y, x**2.0 + x**3.0 + x**0.5, atol=0, rtol=1e-13)


def test_elemental_scalar_multi_call(tmp_path: Path):
    """ELEMENTAL (not just pure) form: two elemental calls on scalars in one expression."""
    src = """
module m
contains
  elemental function sq(x) result(r)
    real(8), intent(in) :: x
    real(8) :: r
    r = x * x
  end function sq
  subroutine kern(a, b, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n), b(n)
    real(8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = sq(a(i)) - sq(b(i))
    end do
  end subroutine kern
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kern", entry="m::kern").build()
    n = 8
    rng = np.random.default_rng(3)
    a = np.asfortranarray(rng.random(n))
    b = np.asfortranarray(rng.random(n))
    y = np.zeros(n, order="F")
    sdfg(a=a, b=b, y=y, n=n)
    np.testing.assert_allclose(y, a**2 - b**2, atol=1e-13, rtol=0)


def test_function_returning_array_multi_call(tmp_path: Path):
    """Bug-class breadth: a function returning an ARRAY, called twice in one expression.
    The two result arrays must not alias onto one SDFG transient."""
    src = """
module m
contains
  pure function scaled(v, s) result(r)
    real(8), intent(in) :: v(3)
    real(8), intent(in) :: s
    real(8) :: r(3)
    r = v * s
  end function scaled
  subroutine kern(a, b, y)
    real(8), intent(in) :: a(3), b(3)
    real(8), intent(out) :: y(3)
    y = scaled(a, 2.0d0) + scaled(b, 3.0d0)
  end subroutine kern
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kern", entry="m::kern").build()
    rng = np.random.default_rng(4)
    a = np.asfortranarray(rng.random(3))
    b = np.asfortranarray(rng.random(3))
    y = np.zeros(3, order="F")
    sdfg(a=a, b=b, y=y)
    np.testing.assert_allclose(y, a * 2.0 + b * 3.0, atol=1e-13, rtol=0)


def test_multi_call_extra_local_temp(tmp_path: Path):
    """Bug-class breadth: callee has a LOCAL TEMP besides the result; both must be
    disambiguated across the two inlined call sites in one expression."""
    src = """
module m
contains
  pure function f(x) result(r)
    real(8), intent(in) :: x
    real(8) :: t, r
    t = x + 1.0d0
    r = t * t
  end function f
  subroutine kern(a, b, y, n)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n), b(n)
    real(8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = f(a(i)) * f(b(i))
    end do
  end subroutine kern
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name="kern", entry="m::kern").build()
    n = 8
    rng = np.random.default_rng(5)
    a = np.asfortranarray(rng.random(n))
    b = np.asfortranarray(rng.random(n))
    y = np.zeros(n, order="F")
    sdfg(a=a, b=b, y=y, n=n)
    np.testing.assert_allclose(y, (a + 1.0)**2 * (b + 1.0)**2, atol=1e-13, rtol=0)
