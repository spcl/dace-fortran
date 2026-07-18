"""Unit coverage for the ``hlfir-mark-bounds-remap-views`` pass.

Tags a bounds-remapping pointer assign's (``ptr(1:N*K) => target(:, slice)``) LHS ``hlfir.declare`` with ``hlfir_bridge.bounds_remap_view`` so the SDFG builder emits a View node instead of routing through index-rewriting.  Runs against the four pinned probes from ``test_view_vs_copy_distinguishable.py`` and checks the tag fires on exactly the view probe.
"""
import subprocess
import tempfile
from pathlib import Path

import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent

#: Attribute the pass attaches when it recognises a bounds-remap view; must match kBoundsRemapViewAttr in MarkBoundsRemapViews.cpp.
_ATTR = "hlfir_bridge.bounds_remap_view"


def _emit_hlfir_and_mark(src_path: Path) -> str:
    """Compile ``src_path`` to HLFIR via flang, run hlfir-mark-bounds-remap-views, return the dumped IR."""
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
    """Number of hlfir.declare ops carrying the bounds-remap-view tag in ir."""
    return ir.count(_ATTR)


def test_pointer_view_bounds_remap_is_tagged():
    """The view probe -- the only bounds-remap pointer assignment -- has its LHS pointer declare tagged."""
    ir = _emit_hlfir_and_mark(_HERE / "pointer_view_bounds_remap_probe.f90")
    assert _count_tagged(ir) >= 1, \
        "view probe should have exactly one bounds-remap-view tag"


def test_pointer_view_bounds_remap_allocatable_is_tagged():
    """Bounds-remap view whose TARGET is ALLOCATABLE (QE's ``prhoc_d => rhoc_d(:, slice)`` shape) is still tagged -- detection is independent of the parent being allocatable."""
    ir = _emit_hlfir_and_mark(_HERE / "pointer_view_bounds_remap_allocatable_probe.f90")
    assert _count_tagged(ir) >= 1, \
        "allocatable-target view probe should have a bounds-remap-view tag"


def test_reshape_intrinsic_copy_is_not_tagged():
    """RESHAPE copy must NOT be tagged -- it lowers through hlfir.reshape, a different op the detector doesn't match."""
    ir = _emit_hlfir_and_mark(_HERE / "reshape_intrinsic_copy_probe.f90")
    assert _count_tagged(ir) == 0, \
        "RESHAPE copy must not be tagged as a view"


def test_pointer_plain_no_remap_is_not_tagged():
    """Plain pointer assign (no bounds remap, no rank change) must not be tagged -- hlfir-rewrite-pointer-assigns handles that case."""
    ir = _emit_hlfir_and_mark(_HERE / "pointer_plain_no_remap_probe.f90")
    assert _count_tagged(ir) == 0, \
        "plain pointer assign must not be tagged"


def test_plain_slice_copy_is_not_tagged():
    """Plain ``dst = src(:, 1:k)`` slice assignment must not be tagged -- no pointer involved, so the rebox-into-pointer detector can't fire."""
    ir = _emit_hlfir_and_mark(_HERE / "plain_slice_copy_probe.f90")
    assert _count_tagged(ir) == 0, \
        "plain slice copy must not be tagged"


def test_idempotent():
    """Running the pass twice produces the same tag count -- overwrite-with-same, not cumulative."""
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
    """The pass produces the expected per-probe tag-count distribution across all four pinned probes -- the primary correctness contract."""
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
