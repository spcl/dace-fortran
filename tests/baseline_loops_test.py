"""Baseline HLFIR coverage: DO WHILE and SELECT CASE (split from ported_from_f2dace_windmill_test.py)."""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src: str, tmp: Path, name: str):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline="hlfir-propagate-shapes").build()


def test_do_while(tmp_path):
    """DO WHILE needs the full pipeline (lift-cf-to-scf), not the minimal `_build` helper."""
    src = """
subroutine while_count(res)
  implicit none
  real(4), intent(inout) :: res(2)
  integer :: i
  i = 0
  res(1) = 0.0
  do while (i < 10)
    res(1) = res(1) + 1.0
    i = i + 1
  end do
end subroutine while_count
"""
    mod = f2py_compile(src, tmp_path / "ref", "while_count")
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="while_count").build()

    r_ref = np.zeros(2, order="F", dtype=np.float32)
    mod.while_count(r_ref)
    r_sdfg = np.zeros(2, dtype=np.float32)
    sdfg(res=r_sdfg)
    np.testing.assert_allclose(r_sdfg, r_ref)


def test_select_case(tmp_path):
    """Simple SELECT CASE on a scalar dummy."""
    src = """
subroutine case_pick(v, out)
  implicit none
  integer, intent(in)    :: v
  integer, intent(inout) :: out
  select case (v)
  case (1)
    out = 10
  case (2)
    out = 20
  case default
    out = -1
  end select
end subroutine case_pick
"""
    mod = f2py_compile(src, tmp_path / "ref", "case_pick")
    sdfg = _build(src, tmp_path / "sdfg", name="case_pick")

    o_ref = np.zeros(1, order="F", dtype=np.int32)
    mod.case_pick(2, o_ref)
    # intent(in) -> plain Scalar; intent(inout) -> stays a length-1 Array (write-back)
    o_sdfg = np.zeros(1, dtype=np.int32)
    sdfg(v=2, out=o_sdfg)
    assert int(o_sdfg[0]) == int(o_ref[0])
