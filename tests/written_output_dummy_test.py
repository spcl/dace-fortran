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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
