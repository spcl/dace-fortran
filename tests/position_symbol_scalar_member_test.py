# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A subscript read from a written scalar member of a module variable seeds its position symbol without an offset."""

import numpy as np
import pytest

from tests._util import build_sdfg

SRC = """
module m
  implicit none
  type :: d
    integer :: mype = 0
    integer :: mype2 = 0
    integer, allocatable :: ir(:)
    integer, allocatable :: nr2p(:)
  end type
  type(d) :: g
contains
  subroutine s(desc, n)
    type(d), intent(inout) :: desc
    integer, intent(in) :: n
    integer :: i
    desc % mype2 = desc % mype
    allocate(desc % ir(n))
    allocate(desc % nr2p(n))
    do i = 1, n
      desc % nr2p(i) = 3
    end do
    do i = 1, n
      desc % ir(i) = desc % nr2p(desc % mype2 + 1)
    end do
  end subroutine
  subroutine top(n, out)
    integer, intent(in) :: n
    integer, intent(out) :: out(n)
    integer :: i
    call s(g, n)
    do i = 1, n
      out(i) = g % ir(i)
    end do
  end subroutine
end module
"""


def test_scalar_member_subscript_builds_without_an_offset_symbol(tmp_path):
    """The member is one element with lower bound 1: its read named an ``offset_g_mype2_d0`` nothing defines."""
    sdfg = build_sdfg(SRC, tmp_path, name="top", entry="m::top").build()
    sdfg.validate()
    assert "offset_g_mype2_d0" not in {str(s) for s in sdfg.free_symbols}


def test_length_one_position_array_reads_its_element(tmp_path):
    """``x(1:p(1))`` with a one-element ``p``: the position symbol reads ``p[1 - 1]``, which must collapse to the scalar
    ``p`` becomes rather than subscript it."""
    src = """
subroutine len1(p, x, r)
  integer, intent(in) :: p(1)
  double precision, intent(in) :: x(8)
  double precision, intent(out) :: r
  r = sum(x(1:p(1)))
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="len1").build()
    r = np.zeros(1)
    sdfg(p=np.array([3], np.int32), x=np.arange(1.0, 9.0), r=r)
    assert r[0] == 6.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
