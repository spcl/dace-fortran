# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""IF conditions: an unrendered one is refused, and an else-if the earlier branches guard is not staged ahead of them."""

import numpy as np
import pytest
from dace.sdfg.state import ConditionalBlock

from tests._util import build_sdfg


def _conditionals(sdfg):
    return [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock)]


@pytest.mark.parametrize(("x", "expected"), [(float("nan"), 0.0), (1.0, 7.0)])
def test_an_ieee_class_test_guards_its_branch(tmp_path, x, expected):
    """``is.fpclass`` lowers to a test of ``x``; the branch is taken only for a NaN, never unconditionally."""
    src = """
subroutine guard(x, y)
  use ieee_arithmetic
  real(8), intent(in) :: x
  real(8), intent(inout) :: y(1)
  if (ieee_is_nan(x)) y(1) = 0.0d0
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="guard", entry="guard").build()
    y = np.array([7.0])
    sdfg(x=x, y=y)
    assert y[0] == expected


@pytest.mark.parametrize(
    ("condition", "declaration"),
    [("m / n > 3", ""), ("a(n) > 3.0d0", "real(8), intent(in) :: a(*)")],
)
def test_guarded_else_if_stays_in_the_else_region(tmp_path, condition, declaration):
    """A division or array read the first branch guards is evaluated only once that branch failed."""
    src = f"""
subroutine ladder(n, m, a, x)
  integer, intent(in) :: n, m
  {declaration or "real(8), intent(in) :: a(*)"}
  real(8), intent(inout) :: x(1)
  if (n == 0) then
    x(1) = 0.0d0
  else if ({condition}) then
    x(1) = 1.0d0
  end if
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="ladder", entry="ladder").build()
    blocks = _conditionals(sdfg)
    assert len(blocks) == 2, "the else-if must nest so its condition is staged inside the else region"
    assert max(len(b.branches) for b in blocks) == 2


def test_total_else_if_stays_flat(tmp_path):
    src = """
subroutine flat(n, x)
  integer, intent(in) :: n
  real(8), intent(inout) :: x(1)
  if (n == 0) then
    x(1) = 0.0d0
  else if (n == 1) then
    x(1) = 1.0d0
  else
    x(1) = 2.0d0
  end if
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="flat", entry="flat").build()
    assert [len(b.branches) for b in _conditionals(sdfg)] == [3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
