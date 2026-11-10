# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fortran fixes explicit bounds at entry: a body assigning an extent leaves the array's bounds alone."""

import numpy as np
import pytest

from tests._util import build_sdfg


@pytest.mark.parametrize(
    ("declaration", "write"),
    [
        ("real(8), intent(inout) :: a(n, 3)", "a(n + 1, 2) = 5.0d0"),
        ("real(8), intent(inout) :: a(4, 3)\n  real(8) :: w(n, 2)", "w(n + 1, 2) = 5.0d0\n  a(4, 2) = w(n + 1, 2)"),
    ],
    ids=["dummy", "automatic"],
)
def test_writing_an_extent_symbol_keeps_the_bounds(tmp_path, declaration, write):
    src = f"""
subroutine shrink(n, a)
  integer, intent(inout) :: n
  {declaration}
  n = n - 1
  {write}
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="shrink", entry="shrink").build()
    a = np.zeros((4, 3), order="F")
    sdfg(n=np.int32(4), a=a)
    assert a[3, 1] == 5.0 and np.count_nonzero(a) == 1


def test_extent_of_an_allocatable_may_be_assigned(tmp_path):
    """``ALLOCATE`` fixes the extent, so reassigning the name changes nothing."""
    src = """
subroutine grow(n, s)
  integer, intent(inout) :: n
  real(8), intent(out) :: s(1)
  real(8), allocatable :: w(:)
  allocate (w(n))
  w = 2.0d0
  n = n + 1
  s(1) = sum(w)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="grow", entry="grow").build()
    s = np.zeros(1)
    sdfg(n=np.int32(3), s=s)
    assert s[0] == 6.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
