# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""An array element at a run-time index sizes an explicit-shape dummy and bounds a loop."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg


def test_element_sizes_a_compound_extent_and_bounds_a_loop(tmp_path: Path):
    src = """
subroutine element_extent(k, m, x)
  integer, intent(in) :: k, m(3)
  double precision, intent(inout) :: x(2 * m(k))
  integer :: i
  do i = 1, m(k)
    x(i) = x(i) + 1d0
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="element_extent").build()
    x = np.zeros(10)
    sdfg(k=np.int32(2), m=np.array([1, 3, 5], np.int32), x=x)
    assert x.tolist() == [1.0] * 3 + [0.0] * 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
