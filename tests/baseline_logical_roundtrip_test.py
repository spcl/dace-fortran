"""Pinned coverage for the bridge's LOGICAL type mapping.

Contract: Fortran LOGICAL(KIND=N) (any kind) surfaces on the SDFG signature as np.bool_
(bool ops render directly, no (x != 0) coercion). The bindings wrapper translates to/from
the original Fortran LOGICAL(KIND=N) image (e.g. 4-byte int32, -1/0 encoding) at the boundary.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_logical_array_copy_in_copy_out_roundtrip(tmp_path: Path):
    """b = a over LOGICAL arrays; bit-exact np.bool_ round-trip."""
    src = """
subroutine roundtrip(a, b, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: a(n)
  logical, intent(out) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = a(i)
  end do
end subroutine roundtrip
"""
    sdfg = build_sdfg(src, tmp_path, name='roundtrip', entry='roundtrip').build()

    n = 5
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    b = np.zeros(n, dtype=np.bool_)
    sdfg(a=a, b=b, n=n)
    np.testing.assert_array_equal(b, a)


def test_logical_invert_per_element(tmp_path: Path):
    """b(i) = .not. a(i): per-element logical NOT lowers through the bool-typed SDFG path."""
    src = """
subroutine not_kernel(a, b, n)
  implicit none
  integer, intent(in)  :: n
  logical, intent(in)  :: a(n)
  logical, intent(out) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = .not. a(i)
  end do
end subroutine not_kernel
"""
    sdfg = build_sdfg(src, tmp_path, name='not_kernel', entry='not_kernel').build()

    n = 5
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    b = np.zeros(n, dtype=np.bool_)
    sdfg(a=a, b=b, n=n)
    np.testing.assert_array_equal(b, np.logical_not(a))


def test_logical_array_inplace_invert_roundtrip(tmp_path: Path):
    """In-place mask = .not. mask; intent(inout) dummy means caller's buffer is both read and written."""
    src = """
subroutine invert_in_place(mask, n)
  implicit none
  integer, intent(in)    :: n
  logical, intent(inout) :: mask(n)
  integer :: i
  do i = 1, n
    mask(i) = .not. mask(i)
  end do
end subroutine invert_in_place
"""
    sdfg = build_sdfg(src, tmp_path, name='invert_in_place', entry='invert_in_place').build()

    n = 6
    original = np.array([True, False, True, True, False, True], dtype=np.bool_)
    mask = original.copy()
    sdfg(mask=mask, n=n)
    # SDFG read the original pattern, wrote back the inverted one.
    np.testing.assert_array_equal(mask, np.logical_not(original))
    # Symmetry: invoking again restores the original.
    sdfg(mask=mask, n=n)
    np.testing.assert_array_equal(mask, original)
