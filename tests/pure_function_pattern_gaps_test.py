"""PURE FUNCTION / ELEMENTAL array-return call-site shapes -- now lowered.

Each test below pins a distinct call-site shape that surfaces in production
NPB / climate benchmarks (graupel's ``update = precip1(...)``, NPB-LU's
``snow_*`` PURE-function chains, etc.).  Together with
:mod:`array_return_assignment_test` they form the bridge's PURE-function
array-return coverage set.

Status: these once emitted the bridge's ``?`` unresolved-expression
placeholder in a scalar-assign tasklet (``ast.parse`` then rejected the code),
and this file's docstring claimed each was pinned as a ``strict=True`` xfail.
Reality diverged: no xfail marker was ever present, and every pattern here
lowers correctly.  So each test now RUNS the compiled SDFG on numpy inputs and
compares BIT-EXACT against the same source compiled by gfortran/f2py -- a
build-and-validate-only check would have let a miscompile pass silently.  If a
future regression breaks the lowering the run/compare fails loudly.

Related patterns that also lower cleanly (covered elsewhere, listed for
context):

* PURE FUNCTION call as an actual ``CALL`` argument
  (``call use_vec(make3(x), out)``) -- the function-temp lowers cleanly.
* User-defined ELEMENTAL over a whole-array argument (``b = sqr(a)``).
* User-defined ELEMENTAL over array slices on either side
  (``b(1:n) = sqr(a(1:n))``, ``b(2:n-1) = sqr(a(2:n-1))``).
* User-defined ELEMENTAL in a composite slice expression
  (``c(1:n) = sqr(a(1:n)) + b(1:n)``).
* PURE FUNCTION taking a fixed-shape array slice as an argument
  (``s = pair_sum(a(i:i+1))``).
* ``WHERE`` construct guarding a user-defined ELEMENTAL
  (``where (a > 0) b = sqr(a)``).

Patterns deliberately out of scope (not anchored):

* RECURSIVE PURE FUNCTION (``r = n + sumto(n - 1)``).  ``hlfir-inline-all``
  cannot inline a self-recursive callee, and the bridge has no fallback
  for treating the self-call as an actual call.  A future pre-pass
  rejecting recursion with a clear diagnostic is a cleaner answer than
  silently emitting ``?``; tracked separately.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang

from dace_fortran import build_sdfg_from_files

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------------------
# Pattern A -- array fn return used inside an arithmetic expression.
# ---------------------------------------------------------------------------

_PAT_A = """
module m_pat_a
  implicit none
contains
  pure function make3(x) result(r)
    real(8), intent(in) :: x
    real(8) :: r(3)
    r(1) = x; r(2) = x * 2.0d0; r(3) = x * 3.0d0
  end function make3

  subroutine kern(out_arr, src, n)
    integer, intent(in) :: n
    real(8), intent(in) :: src(n)
    real(8), intent(out) :: out_arr(3, n)
    real(8) :: a(3)
    integer :: i
    a = (/ 1.0d0, 2.0d0, 3.0d0 /)
    do i = 1, n
      out_arr(:, i) = a + make3(src(i))    ! arr + fn(...)  in expression
    end do
  end subroutine kern
end module m_pat_a
"""


def test_array_fn_return_in_arithmetic(tmp_path):
    """Array RHS = arr + array_fn(...): ``out_arr(:,i) = a + make3(src(i))``.
    Runs + bit-exact vs gfortran (single multiply then add per element)."""
    src = tmp_path / "m.f90"
    src.write_text(_PAT_A)
    sdfg = build_sdfg_from_files([src], entry="m_pat_a::kern", name="pat_a", out_dir=tmp_path / "build")
    sdfg.validate()

    n = 5
    rng = np.random.default_rng(3)
    src_arr = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros((3, n), order='F', dtype=np.float64)
    sdfg(out_arr=out, src=src_arr, n=np.int32(n))

    ref = f2py_compile(_PAT_A, tmp_path / "ref", "pat_a_ref", only=("kern", ))
    out_ref = np.asfortranarray(ref.m_pat_a.kern(src_arr.copy(order='F')))
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Pattern C -- PURE FUNCTION return shape derived from a dummy argument.
# ---------------------------------------------------------------------------

_PAT_C = """
module m_pat_c
  implicit none
contains
  pure function makeN(x, k) result(r)
    integer, intent(in) :: k
    real(8), intent(in) :: x
    real(8) :: r(k)               ! shape derived from dummy ``k``
    integer :: i
    do i = 1, k
      r(i) = x * real(i, 8)
    end do
  end function makeN

  subroutine kern(out_arr, src, n, k)
    integer, intent(in) :: n, k
    real(8), intent(in) :: src(n)
    real(8), intent(out) :: out_arr(k, n)
    real(8) :: tmp(k)
    integer :: i
    do i = 1, n
      tmp = makeN(src(i), k)
      out_arr(:, i) = tmp
    end do
  end subroutine kern
end module m_pat_c
"""


def test_dummy_shaped_fn_return(tmp_path):
    """Local array = fn(...) where fn's return shape is a dummy expression
    (``r(k)``).  Runs + bit-exact vs gfortran: ``out_arr(j,i)=src(i)*j``."""
    src = tmp_path / "m.f90"
    src.write_text(_PAT_C)
    sdfg = build_sdfg_from_files([src], entry="m_pat_c::kern", name="pat_c", out_dir=tmp_path / "build")
    sdfg.validate()

    n, k = 5, 4
    rng = np.random.default_rng(7)
    src_arr = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros((k, n), order='F', dtype=np.float64)
    sdfg(out_arr=out, src=src_arr, n=np.int32(n), k=np.int32(k))

    ref = f2py_compile(_PAT_C, tmp_path / "ref", "pat_c_ref", only=("kern", ))
    out_ref = np.asfortranarray(ref.m_pat_c.kern(src_arr.copy(order='F'), np.int32(k)))
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Pattern F -- PURE FUNCTION return as a derived type.
# ---------------------------------------------------------------------------

_PAT_F = """
module m_pat_f_types
  implicit none
  type :: vec3
    real(8) :: x, y, z
  end type
end module m_pat_f_types

module m_pat_f
  use m_pat_f_types
  implicit none
contains
  pure function make_vec(a, b, c) result(r)
    real(8), intent(in) :: a, b, c
    type(vec3) :: r
    r%x = a; r%y = b; r%z = c
  end function make_vec

  subroutine kern(out_x, src_a, src_b, src_c, n)
    integer, intent(in) :: n
    real(8), intent(in) :: src_a(n), src_b(n), src_c(n)
    real(8), intent(out) :: out_x(n)
    type(vec3) :: p
    integer :: i
    do i = 1, n
      p = make_vec(src_a(i), src_b(i), src_c(i))
      out_x(i) = p%x + p%y + p%z
    end do
  end subroutine kern
end module m_pat_f
"""


def test_fn_returns_derived_type(tmp_path):
    """A PURE FUNCTION whose return is a small derived type.  Runs + bit-exact
    vs gfortran: ``out_x(i) = p%x+p%y+p%z`` with ``p = make_vec(a,b,c)``."""
    src = tmp_path / "m.f90"
    src.write_text(_PAT_F)
    sdfg = build_sdfg_from_files([src], entry="m_pat_f::kern", name="pat_f", out_dir=tmp_path / "build")
    sdfg.validate()

    n = 5
    rng = np.random.default_rng(11)
    a = np.asfortranarray(rng.standard_normal(n))
    b = np.asfortranarray(rng.standard_normal(n))
    c = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros(n, order='F', dtype=np.float64)
    sdfg(out_x=out, src_a=a, src_b=b, src_c=c, n=np.int32(n))

    ref = f2py_compile(_PAT_F, tmp_path / "ref", "pat_f_ref", only=("kern", ))
    out_ref = np.asfortranarray(ref.m_pat_f.kern(a.copy(order='F'), b.copy(order='F'), c.copy(order='F')))
    np.testing.assert_array_equal(out, out_ref)


# ---------------------------------------------------------------------------
# Pattern I -- slice LHS = fixed-shape PURE FUNCTION return.
# ---------------------------------------------------------------------------

_PAT_I = """
module m_pat_i
  implicit none
contains
  pure function make3(x) result(r)
    real(8), intent(in) :: x
    real(8) :: r(3)
    r(1) = x; r(2) = x * 2.0d0; r(3) = x * 3.0d0
  end function make3

  subroutine kern(b, src, n)
    integer, intent(in) :: n
    real(8), intent(in) :: src(n)
    real(8), intent(out) :: b(3, n)
    integer :: i
    do i = 1, n
      b(1:3, i) = make3(src(i))      ! slice LHS = fixed-shape fn return
    end do
  end subroutine kern
end module m_pat_i
"""


def test_slice_lhs_array_fn_return(tmp_path):
    """``arr(1:3, i) = fn(...)`` where fn returns a fixed-shape array.  Runs +
    bit-exact vs gfortran: ``b(:,i) = [x, 2x, 3x]``."""
    src = tmp_path / "m.f90"
    src.write_text(_PAT_I)
    sdfg = build_sdfg_from_files([src], entry="m_pat_i::kern", name="pat_i", out_dir=tmp_path / "build")
    sdfg.validate()

    n = 5
    rng = np.random.default_rng(13)
    src_arr = np.asfortranarray(rng.standard_normal(n))
    out = np.zeros((3, n), order='F', dtype=np.float64)
    sdfg(b=out, src=src_arr, n=np.int32(n))

    ref = f2py_compile(_PAT_I, tmp_path / "ref", "pat_i_ref", only=("kern", ))
    out_ref = np.asfortranarray(ref.m_pat_i.kern(src_arr.copy(order='F')))
    np.testing.assert_array_equal(out, out_ref)
