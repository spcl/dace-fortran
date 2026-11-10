# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reductions with MASK=, IEEE class tests, exact integer(8) initial data, and an expression lower bound."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg


def test_masked_reductions(tmp_path: Path):
    src = """
subroutine masked(x, s, m)
  double precision, intent(in) :: x(6)
  double precision, intent(out) :: s, m
  s = sum(x, mask=x > 0d0)
  m = maxval(x, mask=x < 3d0)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="masked").build()
    s, m = np.zeros(1), np.zeros(1)
    sdfg(x=np.array([1.0, -2.0, 3.0, -4.0, 2.5, 5.0]), s=s, m=m)
    assert (s[0], m[0]) == (11.5, 2.5)


def test_ieee_class_tests(tmp_path: Path):
    src = """
subroutine classes(x, flags)
  use, intrinsic :: ieee_arithmetic
  double precision, intent(in) :: x(4)
  integer, intent(out) :: flags(4)
  integer :: i
  do i = 1, 4
    flags(i) = merge(1, 0, ieee_is_nan(x(i))) + 2 * merge(1, 0, ieee_is_finite(x(i)))
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="classes").build()
    flags = np.zeros(4, np.int32)
    sdfg(x=np.array([np.nan, 1.0, np.inf, -0.0]), flags=flags)
    assert flags.tolist() == [1, 2, 0, 2]


def test_integer8_initial_data_is_exact(tmp_path: Path):
    src = """
subroutine big(r)
  integer(8), intent(out) :: r(2)
  integer(8), parameter :: v(2) = [9007199254740993_8, -9007199254740995_8]
  r = v
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="big").build()
    r = np.zeros(2, np.int64)
    sdfg(r=r)
    assert r.tolist() == [9007199254740993, -9007199254740995]


def test_expression_lower_bound(tmp_path: Path):
    src = """
subroutine shifted(lo, x, r)
  integer, intent(in) :: lo
  double precision, intent(in) :: x(lo + 1:lo + 4)
  double precision, intent(out) :: r
  r = x(lo + 2)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="shifted").build()
    r = np.zeros(1)
    sdfg(lo=np.int32(5), x=np.array([1.0, 2.0, 3.0, 4.0]), r=r)
    assert r[0] == 2.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
