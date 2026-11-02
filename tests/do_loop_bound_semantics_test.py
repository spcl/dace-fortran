# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A DO loop follows Fortran's iteration count: fixed at entry, with the sign of the step known only at run time."""

import numpy as np
from dace.sdfg.state import LoopRegion

from tests._util import build_sdfg


def _loops(sdfg):
    return [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]


def test_nested_subscript_in_a_bound_is_zero_based_at_every_level(tmp_path):
    src = """
subroutine bound(n, cnt, map, b)
  integer, intent(in) :: n
  integer, intent(in) :: cnt(n), map(n)
  real(8), intent(inout) :: b(n)
  integer :: i, j
  do i = 1, n
    do j = 1, cnt(map(i))
      b(i) = b(i) + 1.0d0
    end do
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="bound", entry="bound").build()
    hoist = next(
        value
        for edge in sdfg.all_interstate_edges(recursive=True)
        for name, value in edge.data.assignments.items()
        if name.startswith("loopend") and "cnt" in value
    )
    assert "map[_loop_it_0 - 1]" in hoist, f"inner subscript still 1-based: {hoist}"
    cnt, indirect, b = np.array([1, 2, 3, 4], dtype=np.int32), np.array([4, 1, 3, 2], dtype=np.int32), np.zeros(4)
    sdfg(n=np.int32(4), cnt=cnt, map=indirect, b=b)
    np.testing.assert_array_equal(b, [4.0, 1.0, 3.0, 2.0])


def test_runtime_negative_step_walks_down_and_keeps_the_exit_value(tmp_path):
    src = """
subroutine down(n, s, b, last)
  integer, intent(in) :: n, s
  real(8), intent(inout) :: b(n)
  integer, intent(out) :: last(1)
  integer :: i
  do i = n, 1, s
    b(i) = i
  end do
  last(1) = i
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="down", entry="down").build()
    for step, values, exit_value in ((-2, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0], 0), (2, [0.0] * 6, 6)):
        b, last = np.zeros(6), np.zeros(1, dtype=np.int32)
        sdfg(n=np.int32(6), s=np.int32(step), b=b, last=last)
        np.testing.assert_array_equal(b, values)
        assert last[0] == exit_value


def test_exit_keeps_the_iterator_of_a_runtime_step(tmp_path):
    src = """
subroutine leave(n, s, last)
  integer, intent(in) :: n, s
  integer, intent(out) :: last(1)
  integer :: i
  do i = 1, n, s
    if (i > 4) exit
  end do
  last(1) = i
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="leave", entry="leave").build()
    last = np.zeros(1, dtype=np.int32)
    sdfg(n=np.int32(20), s=np.int32(3), last=last)
    assert last[0] == 7


def test_bound_assigned_in_the_body_is_fixed_at_entry(tmp_path):
    src = """
subroutine shrink(m0, c)
  integer, intent(in) :: m0
  integer, intent(out) :: c(1)
  integer :: i, m
  m = m0
  c(1) = 0
  do i = 1, m
    c(1) = c(1) + 1
    m = m - 1
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="shrink", entry="shrink").build()
    conditions = [loop.loop_condition.as_string for loop in _loops(sdfg)]
    assert all("loopend" in condition for condition in conditions), conditions
    c = np.zeros(1, dtype=np.int32)
    sdfg(m0=np.int32(10), c=c)
    assert c[0] == 10


def test_step_assigned_in_the_body_is_fixed_at_entry(tmp_path):
    src = """
subroutine stretch(c)
  integer, intent(out) :: c(1)
  integer :: i, k
  k = 2
  c(1) = 0
  do i = 1, 30, k
    c(1) = c(1) + 1
    k = k + 1
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="stretch", entry="stretch").build()
    c = np.zeros(1, dtype=np.int32)
    sdfg(c=c)
    assert c[0] == 15


def test_bound_the_body_leaves_alone_stays_a_bare_name(tmp_path):
    """No ceremony for the common case: the loop keeps reading ``n``, so loop analyses see it."""
    src = """
subroutine plain(n, b)
  integer, intent(in) :: n
  real(8), intent(inout) :: b(n)
  integer :: i
  do i = 1, n
    b(i) = 1.0d0
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="plain", entry="plain").build()
    assert "loopend" not in _loops(sdfg)[0].loop_condition.as_string


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
