"""Phase H: skip the per-allocatable <arr>_allocated tracker + post_<arr>_allocated_<n> init state
when the kernel body never queries ALLOCATED/ASSOCIATED and has no ALLOCATE/DEALLOCATE site (the ICON
already-allocated-dummy pattern; velocity_tendencies shed 30 dead trackers + 30 orphan states).
Complementary case (tracker kept when ALLOCATED IS queried) is pinned by intrinsic_allocated_test.py."""

from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_unread_allocatable_dummy_skips_tracker(tmp_path: Path):
    """Allocatable dummy + no ALLOCATE/ALLOCATED in body -> no data_allocated symbol, no
    post_data_allocated_* init state."""
    src = """
subroutine kernel(data, out, n)
  implicit none
  integer, intent(in) :: n
  integer, allocatable, intent(in) :: data(:)
  integer, intent(out) :: out
  out = data(1) + data(n)
end subroutine kernel
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    assert "data_allocated" not in sdfg.arrays, \
        "tracker scalar should not be emitted when no kernel ALLOCATED reader exists"
    assert "data_allocated" not in sdfg.symbols, \
        "tracker symbol should not be emitted when no kernel ALLOCATED reader exists"
    orphan_states = [s.label for s in sdfg.all_states() if s.label.startswith("post_data_allocated")]
    assert not orphan_states, \
        f"orphan post_data_allocated_* init state(s) survived: {orphan_states}"


def test_queried_allocatable_keeps_tracker(tmp_path: Path):
    """Allocatable queried by ALLOCATED(arr) MUST keep its tracker -- the read needs the symbol to resolve to."""
    src = """
subroutine kernel(out)
  implicit none
  integer, allocatable :: data(:)
  integer, intent(out) :: out
  ALLOCATE(data(8))
  out = MERGE(1, 0, ALLOCATED(data))
end subroutine kernel
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name='kernel').build()
    # tracker registers as a symbol (role=symbol in extract_vars), not an array -- both paths kept for safety.
    has_tracker = ("data_allocated" in sdfg.symbols or "data_allocated" in sdfg.arrays)
    assert has_tracker, \
        "tracker MUST be emitted when ALLOCATED(...) reader exists"
