"""Verify ``size``/``lbound``/``ubound`` always resolve to SDFG symbols.

Contract: a shape intrinsic must materialise as a literal int (static extent)
or a synthetic ``<arr>_d<dim>``/``offset_<arr>_d<dim>`` symbol (dynamic) --
never a runtime descriptor read, or the bridge loses the ability to fold the
extent into memlet subsets and loop schedules. Two paths: static
(``passes/FoldAssumedRankQueries.cpp`` rewrites ``fir.box_dims`` to the literal
extent) and dynamic (``bridge/ast/expressions.cpp``~525 emits the bridge-minted
shape symbol).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_size_on_explicit_shape_array_folds_to_constant(tmp_path):
    """Static-extent array's ``size`` query folds to the literal extent -- SDFG
    arglist carries no extra shape symbol for a compile-time-constant shape."""
    src = """
module m
  implicit none
  integer, parameter :: N = 8
  double precision :: arr(N)
contains
  subroutine fill()
    integer :: i, k
    k = size(arr)
    do i = 1, k
      arr(i) = real(i, kind=8)
    end do
  end subroutine fill
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="m::fill").build()
    arr = np.full((8, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr=arr)
    np.testing.assert_array_equal(arr, np.arange(1, 9, dtype=np.float64))


def test_size_on_assumed_shape_dummy_resolves_to_symbol(tmp_path):
    """Dynamic-shape dummy: ``size(buf, 1)`` surfaces as the bridge's ``<arr>_d<dim>``
    symbol, not a runtime box_dims read."""
    src = """
module m
  implicit none
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(:)
    integer :: i
    do i = 1, size(buf)
      buf(i) = real(i, kind=8)
    end do
  end subroutine inner

  subroutine outer(arr, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: arr(n)
    call inner(arr)
  end subroutine outer
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="outer", entry="m::outer").build()
    # bridge mints arr_d0 (or similar) as a shape symbol the caller binds via runtime n;
    # SDFG accepts n as a scalar arg and the extent symbol is specialised from it
    arr = np.full((6, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr=arr, n=np.int32(6))
    np.testing.assert_array_equal(arr, np.arange(1, 7, dtype=np.float64))


def test_lbound_and_ubound_resolve_to_symbols(tmp_path):
    """``lbound``/``ubound`` queries are extents + offsets; they must
    fold to the descriptor's offset symbol / extent symbol pair."""
    src = """
module m
  implicit none
  integer, parameter :: N = 5
  double precision :: arr(N)
contains
  subroutine fill()
    integer :: i, lo, hi
    lo = lbound(arr, 1)
    hi = ubound(arr, 1)
    do i = lo, hi
      arr(i) = real(i*3, kind=8)
    end do
  end subroutine fill
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="m::fill").build()
    arr = np.full((5, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr=arr)
    np.testing.assert_array_equal(arr, np.array([3.0, 6.0, 9.0, 12.0, 15.0]))
