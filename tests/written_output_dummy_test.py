# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A written integer output dummy, read as a symbol in the body, reaches the caller."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg


def test_a_counted_down_dummy_reaches_the_caller(tmp_path: Path):
    src = """
subroutine dec(n, a)
  integer, intent(inout) :: n
  double precision, intent(inout) :: a(10)
  do while (n > 0)
    a(n) = 1d0
    n = n - 1
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="dec").build()
    n, a = np.array([3], np.int32), np.zeros(10)
    sdfg(n=n, a=a)
    assert n[0] == 0 and a.tolist() == [1.0] * 3 + [0.0] * 7


def test_a_do_bound_dummy_assigned_in_the_body_reaches_the_caller(tmp_path: Path):
    """Fortran fixes the trip count at entry: ``n`` bounds the loop, the body assigns it, and the caller sees 2."""
    src = """
subroutine bnd(n, r)
  integer, intent(inout) :: n
  integer, intent(out) :: r
  integer :: i
  r = 0
  do i = 1, n
    n = 2
    r = r + 1
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="bnd").build()
    n, r = np.array([5], np.int32), np.zeros(1, np.int32)
    sdfg(n=n, r=r)
    assert (n[0], r[0]) == (2, 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
