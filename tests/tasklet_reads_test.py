# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tasklet wiring: a value read of an index array keeps its connector, and two scalar writes keep their order."""

import networkx as nx
import numpy as np

from tests._util import build_sdfg


def test_value_read_of_an_array_also_used_as_an_index(tmp_path):
    src = """
subroutine both(n, a, idx, x, y)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  integer, intent(in) :: idx(n)
  real(8), intent(out) :: x(n), y(n)
  integer :: i
  do i = 1, n
    x(i) = idx(i)
    y(i) = a(idx(i))
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="both", entry="both").build()
    a, idx, x, y = 10.0 * np.arange(1.0, 5.0), np.array([4, 3, 2, 1], dtype=np.int32), np.zeros(4), np.zeros(4)
    sdfg(n=np.int32(4), a=a, idx=idx, x=x, y=y)
    np.testing.assert_array_equal(x, idx.astype(np.float64))
    np.testing.assert_array_equal(y, a[idx - 1])


def test_two_writes_to_a_scalar_are_two_ordered_nodes(tmp_path):
    src = """
subroutine rewrite(y, x)
  real(8), intent(in) :: y
  real(8), intent(out) :: x(1)
  real(8) :: t
  t = y * 2.0d0
  t = y + 1.0d0
  x(1) = t
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="scalar_rewrite", entry="rewrite").build()
    state = next(s for s in sdfg.states() if any(n.data == "t" for n in s.data_nodes()))
    written = [n for n in state.data_nodes() if n.data == "t"]
    assert len(written) == 2 and all(state.in_degree(n) == 1 for n in written)
    graph = nx.DiGraph((e.src, e.dst) for e in state.edges())
    assert nx.has_path(graph, *written) or nx.has_path(graph, *written[::-1]), "the writes are unordered"
    x = np.zeros(1)
    sdfg(y=np.float64(3.0), x=x)
    assert x[0] == 4.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
