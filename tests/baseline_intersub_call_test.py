"""Baseline HLFIR coverage: inter-subroutine calls (caller->callee inlining) and OPTIONAL scalar dummies with PRESENT().  Split out of ``ported_from_f2dace_windmill_test.py``."""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def _build(src: str, tmp: Path, name: str, entry: str | None = None, pipeline: str | None = "hlfir-propagate-shapes"):
    tmp.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, tmp, name=name, pipeline=pipeline, entry=entry).build()


def test_intersub_call(tmp_path):
    """Two subroutines, outer calls inner -- exercises hlfir-inline-all writeback; f2py wraps every subroutine, we pick outer."""
    src = """
subroutine inner(d)
  implicit none
  real(8), intent(inout) :: d(4)
  d(2) = 4.2d0
end subroutine inner

subroutine outer(d)
  implicit none
  real(8), intent(inout) :: d(4)
  d(2) = 5.5d0
  call inner(d)
end subroutine outer
"""
    mod = f2py_compile(src, tmp_path / "ref", "outer_mod")
    # default pipeline (hlfir-inline-all runs) -- required so privatised callees inline before symbol-dce drops them.
    sdfg = _build(src, tmp_path / "sdfg", name="outer", entry="outer", pipeline=None)

    d_ref = np.zeros(4, order="F")
    mod.outer(d_ref)
    d_sdfg = np.zeros(4, dtype=np.float64)
    sdfg(d=d_sdfg)
    np.testing.assert_allclose(d_sdfg, d_ref)


def test_optional_arg(tmp_path):
    """PRESENT() on a scalar OPTIONAL resolves to a companion ``<name>_present`` ABI symbol (non-zero=present); covers both if-present() branches."""
    src = """
subroutine opt_sum(res, a)
  implicit none
  integer, intent(inout) :: res(2)
  integer, optional      :: a
  if (present(a)) then
    res(1) = a
  else
    res(1) = 0
  end if
end subroutine opt_sum
"""
    mod = f2py_compile(src, tmp_path / "ref", "opt_sum")
    sdfg = _build(src, tmp_path / "sdfg", name="opt_sum")

    # present branch: caller supplies a and sets a_present=1 (OPTIONAL scalar dummy lands as a plain Scalar on the SDFG signature).
    r_ref = np.zeros(2, order="F", dtype=np.int32)
    mod.opt_sum(r_ref, 5)
    r_sdfg = np.zeros(2, dtype=np.int32)
    sdfg(res=r_sdfg, a=5, a_present=1)
    np.testing.assert_array_equal(r_sdfg, r_ref)

    # absent branch: reference omits the argument; SDFG passes a placeholder for a and a_present=0 (callee never reads a here).
    r_ref_absent = np.zeros(2, order="F", dtype=np.int32)
    mod.opt_sum(r_ref_absent)
    r_sdfg_absent = np.zeros(2, dtype=np.int32)
    sdfg(res=r_sdfg_absent, a=0, a_present=0)
    np.testing.assert_array_equal(r_sdfg_absent, r_ref_absent)
