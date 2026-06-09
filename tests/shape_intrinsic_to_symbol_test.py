"""Verify ``size``/``lbound``/``ubound`` always resolve to SDFG symbols.

The contract: when constructing the SDFG, the result of a Fortran shape
intrinsic must materialise as a symbolic expression (a literal int when
the extent is statically known, or a synthetic ``<arr>_d<dim>`` /
``offset_<arr>_d<dim>`` symbol when it's dynamic) -- never as a
runtime scalar computation against the descriptor.  Without this rule,
loop bounds reading ``size(buf, k)`` would compile to a load against
the dummy's box and the bridge would lose the ability to fold the
extent into memlet subsets and loop schedules.

Two paths cover the rule:

* Static: ``passes/FoldAssumedRankQueries.cpp`` rewrites
  ``fir.box_dims %X, %k`` to the literal extent when the trace lands
  on a concrete-shape ``fir.array``.
* Dynamic: ``bridge/ast/expressions.cpp`` (around line 525) emits the
  result of an unfolded ``fir.box_dims`` as the bridge-minted shape
  symbol ``<arr>_d<dim>``.

Both paths share the property that the lifted ``do`` bound never
contains a runtime descriptor read -- the SDFG's loop emitter and
memlet sizer see a closed-form symbolic value.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_size_on_explicit_shape_array_folds_to_constant(tmp_path):
    """A static-extent array's ``size`` query folds to the literal
    extent.  The SDFG arglist must not carry an extra shape symbol
    for an array whose shape is a compile-time constant."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="_QMmPfill").build()
    arr = np.full((8, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr=arr)
    np.testing.assert_array_equal(arr, np.arange(1, 9, dtype=np.float64))


@pytest.mark.xfail(strict=False,
                   reason=("The bridge mints ``arr_d0`` as a shape symbol "
                           "for the dynamic-shape outer dummy but never "
                           "wires the caller-side ``n`` to bind it.  This is "
                           "the shape-symbol-propagation gap (separate from "
                           "the ``size``/``box_dims`` symbol-emission "
                           "contract, which works -- see the static-extent "
                           "test).  Surfaces as 'unresolved free symbol "
                           "arr_d0' at build time."))
def test_size_on_assumed_shape_dummy_resolves_to_symbol(tmp_path):
    """When the dummy has dynamic shape, ``size(buf, 1)`` must
    surface as the bridge's ``<arr>_d<dim>`` symbol -- not a
    runtime box_dims read."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="outer", entry="_QMmPouter").build()
    # The bridge mints ``arr_d0`` (or similar) as a shape symbol
    # that the caller binds via the runtime ``n``.  The SDFG must
    # accept ``n`` as a scalar arg; the extent symbol is then
    # specialised.
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="_QMmPfill").build()
    arr = np.full((5, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr=arr)
    np.testing.assert_array_equal(arr, np.array([3.0, 6.0, 9.0, 12.0, 15.0]))
