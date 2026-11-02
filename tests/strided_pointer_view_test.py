# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""A pointer onto a strided section keeps the stride, literal or symbolic, and has the extent of the section."""

import numpy as np

from tests._util import build_sdfg


def test_rank_one_stride_reaches_the_view(tmp_path):
    src = """
subroutine onto(n, a)
  integer, intent(in) :: n
  real(8), intent(inout), target :: a(n)
  real(8), pointer :: p(:)
  p => a(1:n:2)
  p(2) = 1.0d0
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="onto", entry="onto").build()
    assert sdfg.arrays["p"].strides == (2,)
    a = np.zeros(6)
    sdfg(n=np.int32(6), a=a)
    np.testing.assert_array_equal(a, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_symbolic_stride_reaches_the_view(tmp_path):
    src = """
subroutine onto(n, k, a)
  integer, intent(in) :: n, k
  real(8), intent(inout), target :: a(n, 3)
  real(8), pointer :: p(:)
  p => a(1:n:k, 2)
  p(2) = 1.0d0
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="onto_sym", entry="onto").build()
    a = np.zeros((6, 3), order="F")
    sdfg(n=np.int32(6), k=np.int32(2), a=a)
    assert a[2, 1] == 1.0 and a.sum() == 1.0


def test_whole_view_write_covers_the_section_only(tmp_path):
    src = """
subroutine fill(n, a)
  integer, intent(in) :: n
  real(8), intent(inout), target :: a(n)
  real(8), pointer :: p(:)
  p => a(2:n:2)
  p = 1.0d0
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="fill", entry="fill").build()
    fill = next(n for s in sdfg.states() for n in s.nodes() if n.__class__.__name__ == "MapEntry")
    assert str(fill.map.range.size()[0]).startswith("int_floor"), fill.map.range
    a = np.zeros(7)
    sdfg(n=np.int32(7), a=a)
    np.testing.assert_array_equal(a, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
