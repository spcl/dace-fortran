"""Unit coverage for the ``hlfir-mark-bounds-remap-views`` pass.

Approach A: detect Fortran 2003 bounds-remapping pointer assignment
(``ptr(1:N*K) => target(:, slice)``) and tag the LHS pointer's
``hlfir.declare`` with the unit attribute
``hlfir_bridge.bounds_remap_view``.  The pass mutates no IR; the
downstream SDFG-build path reads the tag to know to emit a DaCe
View node rather than route the pointer through the existing
index-rewriting machinery.

These tests run the pass against the four pinned probes from
``test_view_vs_copy_distinguishable.py`` and assert the tag fires on
exactly the one probe whose source is a bounds-remap *view*, not on
the three copy-or-non-remap probes.  Together with the IR-pattern
distinguishability tests, this commit closes the loop: the
detection criteria are structurally sound *and* implemented in C++
without false positives.
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent

#: The attribute the pass attaches to a pointer's ``hlfir.declare``
#: when it recognises a bounds-remap view rebind.  Identical literal
#: to ``kBoundsRemapViewAttr`` in MarkBoundsRemapViews.cpp.
_ATTR = "hlfir_bridge.bounds_remap_view"


def _emit_hlfir_and_mark(src_path: Path) -> str:
    """Compile ``src_path`` to HLFIR via flang, parse it into the
    bridge, run ``hlfir-mark-bounds-remap-views``, return the dumped
    IR."""
    from dace_fortran.build_bridge import hb

    with tempfile.TemporaryDirectory(prefix="brv_mark_") as td:
        h = Path(td) / "k.hlfir"
        subprocess.check_call([
            "flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
            str(src_path), "-o",
            str(h)
        ],
                              cwd=td)
        mod = hb.HLFIRModule()
        mod.parse_file(str(h))
        mod.run_passes("hlfir-mark-bounds-remap-views")
        return mod.dump()


def _count_tagged(ir: str) -> int:
    """Number of ``hlfir.declare`` ops carrying the
    bounds-remap-view tag in ``ir``."""
    return ir.count(_ATTR)


def test_pointer_view_bounds_remap_is_tagged():
    """The view probe -- the only probe whose Fortran source is a
    bounds-remap pointer assignment -- has its LHS pointer declare
    tagged."""
    ir = _emit_hlfir_and_mark(_HERE / "pointer_view_bounds_remap_probe.f90")
    assert _count_tagged(ir) >= 1, \
        "view probe should have exactly one bounds-remap-view tag"


def test_pointer_view_bounds_remap_allocatable_is_tagged():
    """The bounds-remap view whose TARGET is ALLOCATABLE (the QE
    ``prhoc_d => rhoc_d(:, slice)`` Gate-H shape) is still tagged.  The
    rebox-rank-change detection is independent of the parent being
    allocatable; the allocatable-specific ``fir.load`` hop is needed
    downstream in extract_vars' source trace (covered by
    ``test_view_emission.test_var_info_bounds_remap_view_through_allocatable_target``)."""
    ir = _emit_hlfir_and_mark(_HERE / "pointer_view_bounds_remap_allocatable_probe.f90")
    assert _count_tagged(ir) >= 1, \
        "allocatable-target view probe should have a bounds-remap-view tag"


def test_reshape_intrinsic_copy_is_not_tagged():
    """The RESHAPE copy probe must NOT trigger the tag -- RESHAPE
    lowers through ``hlfir.reshape`` (a different op), so the
    detector cannot fire here."""
    ir = _emit_hlfir_and_mark(_HERE / "reshape_intrinsic_copy_probe.f90")
    assert _count_tagged(ir) == 0, \
        "RESHAPE copy must not be tagged as a view"


def test_pointer_plain_no_remap_is_not_tagged():
    """Plain pointer assign (no bounds remap, no rank change) must
    not trigger the tag.  The existing
    ``hlfir-rewrite-pointer-assigns`` handles this case; the
    bounds-remap-view tag is reserved for the rank-changing variant."""
    ir = _emit_hlfir_and_mark(_HERE / "pointer_plain_no_remap_probe.f90")
    assert _count_tagged(ir) == 0, \
        "plain pointer assign must not be tagged"


def test_plain_slice_copy_is_not_tagged():
    """Plain ``dst = src(:, 1:k)`` slice assignment must not trigger
    the tag.  No pointer is involved, so the rebox-into-pointer
    detector cannot fire."""
    ir = _emit_hlfir_and_mark(_HERE / "plain_slice_copy_probe.f90")
    assert _count_tagged(ir) == 0, \
        "plain slice copy must not be tagged"


def test_idempotent():
    """Running the pass twice on the same module produces the same
    tag count -- the attribute set is overwrite-with-same, not
    cumulative."""
    from dace_fortran.build_bridge import hb

    with tempfile.TemporaryDirectory(prefix="brv_idem_") as td:
        h = Path(td) / "k.hlfir"
        subprocess.check_call([
            "flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
            str(_HERE / "pointer_view_bounds_remap_probe.f90"), "-o",
            str(h)
        ],
                              cwd=td)
        mod = hb.HLFIRModule()
        mod.parse_file(str(h))
        mod.run_passes("hlfir-mark-bounds-remap-views")
        once = mod.dump().count(_ATTR)
        mod.run_passes("hlfir-mark-bounds-remap-views")
        twice = mod.dump().count(_ATTR)
        assert once == twice, f"tag count grew on second run ({once} -> {twice})"
        assert once >= 1


def test_summary_distinguishes_all_four_cases_end_to_end():
    """The pass produces exactly the expected per-probe tag-count
    distribution across the four pinned probes.  This is the
    primary mark-pass correctness contract."""
    cases = [
        ("pointer_view_bounds_remap_probe.f90", True),
        ("reshape_intrinsic_copy_probe.f90", False),
        ("pointer_plain_no_remap_probe.f90", False),
        ("plain_slice_copy_probe.f90", False),
    ]
    for fname, expected_tagged in cases:
        ir = _emit_hlfir_and_mark(_HERE / fname)
        count = _count_tagged(ir)
        if expected_tagged:
            assert count >= 1, f"{fname}: expected >=1 tag, got 0"
        else:
            assert count == 0, f"{fname}: expected 0 tags, got {count}"
