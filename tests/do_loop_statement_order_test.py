# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A DO body of plain assignments keeps source order: no symbol assign or index load moves past a statement it follows."""

import numpy as np
from dace import nodes

from tests._util import build_sdfg


def _assign_edge(sdfg, name):
    return next(edge for edge in sdfg.all_interstate_edges(recursive=True) if name in edge.data.assignments)


def _has_tasklet(state):
    return any(isinstance(node, nodes.Tasklet) for node in state.nodes())


def test_symbol_update_after_a_compute_stays_after_it(tmp_path):
    """``b(i)=a(k); k=k+1`` reads ``a(k)`` before the update."""
    src = """
subroutine cursor(n, a, b)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(out) :: b(n)
  integer :: i, k
  k = 1
  do i = 1, n
    b(i) = a(k)
    k = k + 1
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="cursor", entry="cursor").build()
    update = next(
        edge
        for edge in sdfg.all_interstate_edges(recursive=True)
        if "k" in edge.data.assignments and "k" in edge.data.assignments["k"]
    )
    assert _has_tasklet(update.src), "the update must leave the state of the compute, not precede it"
    a, b = np.arange(1.0, 6.0), np.zeros(5)
    sdfg(n=np.int32(5), a=a, b=b)
    np.testing.assert_array_equal(b, a)


def test_index_array_written_by_the_body_is_read_after_the_write(tmp_path):
    src = """
subroutine reindex(n, a, idx, y)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  integer, intent(inout) :: idx(n)
  real(8), intent(out) :: y(n)
  integer :: i
  do i = 1, n
    idx(i) = n + 1 - i
    y(i) = a(idx(i))
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="reindex", entry="reindex").build()
    a, idx, y = np.arange(1.0, 6.0), np.ones(5, dtype=np.int32), np.zeros(5)
    sdfg(n=np.int32(5), a=a, idx=idx, y=y)
    np.testing.assert_array_equal(y, a[::-1])


def test_nested_index_of_a_symbol_assign_uses_the_inner_load(tmp_path):
    """``k=i2(i1(i))``: the edge reads the symbol holding ``i1(i)``, never a raw 1-based ``i1`` subscript."""
    src = """
subroutine nested(n, i1, i2, b)
  integer, intent(in) :: n
  integer, intent(in) :: i1(n), i2(n)
  real(8), intent(inout) :: b(n)
  integer :: i, k
  do i = 1, n
    k = i2(i1(i))
    b(k) = i
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="nested", entry="nested").build()
    assert "i1[" not in _assign_edge(sdfg, "k").data.assignments["k"]
    i1, i2, b = np.array([2, 3, 4, 1], dtype=np.int32), np.array([4, 3, 2, 1], dtype=np.int32), np.zeros(4)
    sdfg(n=np.int32(4), i1=i1, i2=i2, b=b)
    np.testing.assert_array_equal(b, [3.0, 2.0, 1.0, 4.0])


def test_index_load_repeats_after_its_symbol_changes(tmp_path):
    """The same text ``idx(k)`` names two elements around ``k=k+1``."""
    src = """
subroutine twice(n, a, idx, y, z)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  integer, intent(in) :: idx(n)
  real(8), intent(inout) :: y(n), z(n)
  integer :: i, k
  k = 1
  do i = 1, n - 1
    y(i) = a(idx(k))
    k = k + 1
    z(i) = a(idx(k))
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="twice", entry="twice").build()
    a, idx, y, z = 10.0 * np.arange(1.0, 6.0), np.array([3, 1, 4, 2, 5], dtype=np.int32), np.zeros(5), np.zeros(5)
    sdfg(n=np.int32(5), a=a, idx=idx, y=y, z=z)
    np.testing.assert_array_equal(y[:4], a[idx[:4] - 1])
    np.testing.assert_array_equal(z[:4], a[idx[1:] - 1])


def test_index_load_repeats_after_its_array_changes(tmp_path):
    src = """
subroutine rewrite(n, a, idx, y, z)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  integer, intent(inout) :: idx(n)
  real(8), intent(inout) :: y(n), z(n)
  integer :: i
  do i = 1, n
    y(i) = a(idx(i))
    idx(i) = n + 1 - i
    z(i) = a(idx(i))
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="rewrite", entry="rewrite").build()
    a, y, z = 10.0 * np.arange(1.0, 6.0), np.zeros(5), np.zeros(5)
    idx = np.array([3, 1, 4, 2, 5], dtype=np.int32)
    first = a[idx - 1]
    sdfg(n=np.int32(5), a=a, idx=idx, y=y, z=z)
    np.testing.assert_array_equal(y, first)
    np.testing.assert_array_equal(z, a[::-1])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
