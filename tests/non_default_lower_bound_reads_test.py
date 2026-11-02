# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reads staged outside tasklets (position symbols, spliced section-alias slots) honor the array's lower bound."""

import numpy as np

from tests._util import build_sdfg


def test_position_symbol_of_a_zero_based_array(tmp_path):
    src = """
subroutine window(pos, a, s)
  integer, intent(in) :: pos(0:3)
  real(8), intent(in) :: a(8)
  real(8), intent(out) :: s(1)
  s(1) = sum(a(pos(1):pos(2)))
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="window", entry="window").build()
    reads = [
        value
        for edge in sdfg.all_interstate_edges(recursive=True)
        for name, value in edge.data.assignments.items()
        if name.startswith("__sym_pos")
    ]
    assert sorted(reads) == ["pos[1 - 0]", "pos[2 - 0]"], reads
    pos, a, s = np.array([1, 2, 3, 4], dtype=np.int32), np.arange(1.0, 9.0), np.zeros(1)
    sdfg(pos=pos, a=a, s=s)
    assert s[0] == 5.0


def test_section_alias_slot_of_a_zero_based_dimension(tmp_path):
    """``a(:, k)`` passed down: a condition reading it splices ``k`` against ``a``'s own lower bound."""
    src = """
subroutine outer(n, m, k, a, r)
  integer, intent(in) :: n, m, k
  real(8), intent(inout) :: a(n, 0:m)
  real(8), intent(inout) :: r
  call inner(a(:, k), n, r)
contains
  subroutine inner(x, l, r)
    integer, intent(in) :: l
    real(8), intent(inout) :: x(l)
    real(8), intent(inout) :: r
    if (x(1) > 0.0d0 .and. x(2) > 0.0d0) r = r + 1.0d0
  end subroutine
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="outer", entry="outer").build()
    a = np.asfortranarray(np.array([[0.0, 5.0, 0.0], [0.0, 2.0, 0.0]]))
    r = np.zeros(1)
    sdfg(n=np.int32(2), m=np.int32(2), k=np.int32(1), a=a, r=r)
    assert r[0] == 1.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
