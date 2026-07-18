"""Edge-case probes for the bridge's offset-propagation logic.

Each test exercises a Fortran pattern that could mis-infer a lower bound and
trigger an OOB load.  Passing cases pin bridge coverage; ``xfail`` cases
document gaps for Phase 1 of the offset-propagation fix (see
``project_hlfir_offset_propagation_plan``).
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp_path: Path, entry: str):
    """Compile ``src`` to an SDFG."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=entry.split('P')[-1], entry=entry).build()
    sdfg.validate()
    return sdfg


def test_arith_computed_negative_index(tmp_path: Path):
    """Edge case 3: index expr ``lb - 1`` (``lb`` a negative PARAMETER). Works
    today via ALLOCATE's embox shape_shift (``lowerBoundsFromAllocSite``), not
    by folding ``arith.subi``.  Regression gate."""
    src = """
subroutine arith_idx(out)
  implicit none
  integer, parameter :: lb = -5
  integer, intent(out) :: out
  integer, allocatable :: arr(:)
  allocate(arr(-6:5))
  arr(-6) = 1234
  out = arr(lb - 1)
  deallocate(arr)
end subroutine arith_idx
"""
    sdfg = _build(src, tmp_path, "arith_idx")
    consts = dict(getattr(sdfg, "_fortran_offset_values", sdfg.constants))
    assert consts.get('offset_arr_d0') == -6, (f"expected offset_arr_d0 == -6 (from ALLOCATE shape_shift); "
                                               f"got {consts.get('offset_arr_d0')}")
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(out=out)
    assert out[0] == 1234


def test_multi_allocate_different_bounds(tmp_path: Path):
    """Edge case 4: deallocate + re-allocate with different bounds. Bridge
    creates a separate alias (``arr_alloc1``) per ALLOCATE site; Phase A wired
    ``lowerBoundsFromAllocSite`` into the alias entry so each gets its own
    offset.  Regression gate: ``offset_arr_d0 == -5`` and ``offset_arr_alloc1_d0 == 0``."""
    src = """
subroutine multi_alloc(out)
  implicit none
  integer, intent(out) :: out
  integer, allocatable :: arr(:)
  integer :: i
  allocate(arr(-5:5))
  arr(-5) = 42
  deallocate(arr)
  allocate(arr(0:10))
  do i = 0, 10
    arr(i) = i + 1000
  end do
  out = arr(3)
end subroutine multi_alloc
"""
    sdfg = _build(src, tmp_path, "multi_alloc")
    # Each ALLOCATE produces a separate SDFG alias -- verify offsets directly.
    consts = dict(getattr(sdfg, "_fortran_offset_values", sdfg.constants))
    assert consts.get('offset_arr_d0') == -5, (f"first alias offset should be -5; got {consts.get('offset_arr_d0')}")
    # REAL gap: second-allocate alias defaults to offset=1 instead of the actual
    # bound 0 -- lowerBoundsFromAllocSite isn't called for the alias entry.
    assert consts.get('offset_arr_alloc1_d0') == 0, (
        f"second alias offset should be 0 (matching ``allocate(arr(0:10))``); "
        f"got {consts.get('offset_arr_alloc1_d0')}.  Bridge gap.")
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(out=out)
    # Correct semantics: arr(3) in the second allocation = 1003.
    assert out[0] == 1003


def test_loop_iv_into_negative_bound_local_works(tmp_path: Path):
    """Edge case 6 (local-allocatable): loop iv as index, local ALLOCATABLE --
    ``lowerBoundsFromAllocSite`` recovers the bound from the embox's shape_shift, so this WORKS today.  Regression gate."""
    src = """
subroutine loop_iv_local(out)
  implicit none
  integer, intent(out) :: out
  integer, allocatable :: arr(:)
  integer :: i, total
  allocate(arr(-3:3))
  do i = -3, 3
    arr(i) = i + 100
  end do
  total = 0
  do i = -3, 3
    total = total + arr(i)
  end do
  out = total
  deallocate(arr)
end subroutine loop_iv_local
"""
    sdfg = _build(src, tmp_path, "loop_iv_local")
    consts = dict(getattr(sdfg, "_fortran_offset_values", sdfg.constants))
    assert consts.get('offset_arr_d0') == -3, (f"expected offset_arr_d0 == -3 from local ALLOCATE shape_shift; "
                                               f"got {consts.get('offset_arr_d0')}")
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(out=out)
    # Sum of (i + 100) for i in -3..3 = 0 + 7*100 = 700
    assert out[0] == 700


def test_loop_iv_into_negative_bound_dummy(tmp_path: Path):
    """Edge case 6 (dummy-arg): loop iv as index, dummy ALLOCATABLE, no literal
    hint.  Bridge correctly leaves ``offset_arr_d0`` as a free symbol -- caller
    supplies the bound at call time; loop-iv auto-inference is NOT needed."""
    src = """
subroutine loop_iv_dummy(arr, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -3, 3
    total = total + arr(i)
  end do
  out = total
end subroutine loop_iv_dummy
"""
    sdfg = _build(src, tmp_path, "loop_iv_dummy")
    # Bridge leaves the offset as a free symbol -- caller binds.
    assert 'offset_arr_d0' in sdfg.arglist(), (f"expected free offset symbol for dummy-arg + no-literal pattern; "
                                               f"arglist: {[k for k in sdfg.arglist() if 'offset' in k]}")
    # arr_buf[0..6] holds Fortran arr(-3..3).
    arr_buf = np.asfortranarray(np.array([i + 100 for i in range(-3, 4)], dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr_buf, out=out, arr_d0=np.int64(7), offset_arr_d0=np.int64(-3))
    # Sum of (i + 100) for i in -3..3 = 0 + 7*100 = 700.
    assert out[0] == 700


def test_indirect_table_negative_offset_caller_supplied(tmp_path: Path):
    """Edge case 7: ``arr(idx_table(j))`` with statically-unknowable negative
    values -- caller MUST supply ``offset_arr_d0``.  Verifies the free-symbol fallback."""
    src = """
subroutine indirect_neg(arr, idx_table, n_idx, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(in) :: n_idx
  integer, intent(in) :: idx_table(n_idx)
  integer, intent(out) :: out(n_idx)
  integer :: j
  do j = 1, n_idx
    out(j) = arr(idx_table(j))
  end do
end subroutine indirect_neg
"""
    sdfg = _build(src, tmp_path, "indirect_neg")
    # Bridge can't infer the bound -- expect the offset stays free.
    assert 'offset_arr_d0' in sdfg.arglist(), (f"expected free symbol for unresolvable indirect access; "
                                               f"arglist: {list(sdfg.arglist().keys())}")

    # Caller fills the buffer so arr_buf[0] corresponds to Fortran arr(-3) (offset_arr_d0 = -3).
    arr_buf = np.asfortranarray(np.array([-30, -20, -10, 0, 10, 20, 30], dtype=np.int32))  # arr(-3..3) values
    idx_table = np.asfortranarray(np.array([-3, 0, 3], dtype=np.int32))
    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(arr=arr_buf, idx_table=idx_table, n_idx=np.int32(3), out=out, arr_d0=np.int64(7), offset_arr_d0=np.int64(-3))
    np.testing.assert_array_equal(out, [-30, 0, 30])
