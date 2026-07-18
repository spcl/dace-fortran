"""Pins that view-semantics and copy-semantics Fortran array ops are always
structurally distinguishable in HLFIR -- the load-bearing assumption behind
Approach A for bounds-remapping pointer assignment (``ptr(lo:hi) =>
target(:, slice)``, a view with no copy).  A misdetected copy would silently
alias instead of copy.

Four probes, each asserted for its op-pattern signature:
  * pointer_view_bounds_remap_probe.f90 -- view: rank-changing ``fir.rebox``
    into ``!fir.box<!fir.ptr<...>>``.
  * reshape_intrinsic_copy_probe.f90 -- copy: ``hlfir.reshape``, no
    rebox-into-ptr.
  * pointer_plain_no_remap_probe.f90 -- plain pointer assign: ``fir.embox``,
    same rank; handled by the existing ``hlfir-rewrite-pointer-assigns`` pass.
  * plain_slice_copy_probe.f90 -- plain copy: ``hlfir.assign`` between
    same-rank boxes, no rebox/embox/reshape.

If these drift on a future flang release, the bounds-remap-view detector
needs re-validation.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from _util import have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_HERE = Path(__file__).resolve().parent


def _emit_hlfir(src_path: Path) -> str:
    """Compile ``src_path`` to HLFIR text via flang."""
    with tempfile.TemporaryDirectory(prefix="brv_") as td:
        out = Path(td) / "k.hlfir"
        subprocess.check_call([
            "flang-new-21", "-fc1", "-fintrinsic-modules-path", "/usr/lib/llvm-21/include/flang", "-emit-hlfir",
            str(src_path), "-o",
            str(out)
        ],
                              cwd=td)
        return out.read_text()


#: ``fir.rebox`` into ``!fir.box<!fir.ptr<...>>``; type spelling alone distinguishes a pointer
#: rebox from any other use.  ``re.DOTALL`` spans flang's wrapped multi-line type prints.
_REBOX_INTO_PTR_RE = re.compile(r"fir\.rebox\s+%\w+(?:\([^)]*\))?\s*:\s*\([^)]*\)\s*->\s*!fir\.box<!fir\.ptr<",
                                re.DOTALL)


#: Rank change = differing count of ``?`` (dynamic-extent placeholders) between input/output.
def _has_rank_changing_rebox_into_ptr(ir: str) -> bool:
    """Detect ``fir.rebox %X(...) : (!fir.box<!fir.array<INPUT>>, ...)
    -> !fir.box<!fir.ptr<!fir.array<OUTPUT>>>`` where rank(INPUT) !=
    rank(OUTPUT)."""
    for m in re.finditer(
            r"fir\.rebox[^:]*:\s*"
            r"\(\s*!fir\.box<!fir\.array<([?\dx]+)x[^>]+>>[^)]*\)\s*->\s*"
            r"!fir\.box<!fir\.ptr<!fir\.array<([?\dx]+)x[^>]+>>",
            ir,
    ):
        in_dims = m.group(1).count("?") + m.group(1).count("x") + 1 - m.group(1).count("x")
        out_dims = m.group(2).count("?") + m.group(2).count("x") + 1 - m.group(2).count("x")
        # heuristic above is flaky on flang's type printer -- count ? directly instead.
        in_dims = m.group(1).count("?")
        out_dims = m.group(2).count("?")
        if in_dims != out_dims and in_dims > 0 and out_dims > 0:
            return True
    return False


def test_pointer_view_bounds_remap_has_rebox_into_pointer():
    """The view case: rank-changing rebox into a pointer-typed box.
    This is the signature approach A keys on."""
    ir = _emit_hlfir(_HERE / "pointer_view_bounds_remap_probe.f90")
    assert _REBOX_INTO_PTR_RE.search(ir), \
        "view probe missing fir.rebox into !fir.box<!fir.ptr<...>>"
    assert _has_rank_changing_rebox_into_ptr(ir), \
        "view probe missing the rank-change between rebox input and output"
    # And critically -- no hlfir.reshape (that would be the copy path).
    assert "hlfir.reshape" not in ir, \
        "view probe shouldn't have hlfir.reshape (that's copy semantics)"


def test_reshape_intrinsic_copy_uses_hlfir_reshape_not_rebox():
    """RESHAPE copy: ``hlfir.reshape`` present, no rebox-into-pointer -- must not be confused
    with the view case."""
    ir = _emit_hlfir(_HERE / "reshape_intrinsic_copy_probe.f90")
    assert "hlfir.reshape" in ir, \
        "RESHAPE probe missing the hlfir.reshape op"
    # detector keys on rebox-into-pointer; this probe must not match it.
    assert not _REBOX_INTO_PTR_RE.search(ir), \
        "RESHAPE probe must not have fir.rebox into !fir.box<!fir.ptr<...>> -- " \
        "the bounds-remap-view detector would false-positive on copy semantics"


def test_pointer_plain_no_remap_uses_embox_not_rebox():
    """Plain pointer assign (no bounds remap): ``fir.embox``, same rank -- handled by
    ``hlfir-rewrite-pointer-assigns``; bounds-remap-view detector must not trigger."""
    ir = _emit_hlfir(_HERE / "pointer_plain_no_remap_probe.f90")
    assert "fir.embox" in ir, "plain pointer assign should use fir.embox"
    assert not _has_rank_changing_rebox_into_ptr(ir), \
        "plain pointer assign must not have rank-changing rebox -- " \
        "the bounds-remap-view detector would false-positive"


def test_plain_slice_copy_has_no_pointer_box_at_all():
    """Plain ``dst = src(:, 1:k)``: same-rank ``hlfir.assign``, no pointer box at all --
    detector must not trigger."""
    ir = _emit_hlfir(_HERE / "plain_slice_copy_probe.f90")
    assert "!fir.box<!fir.ptr<" not in ir, \
        "plain slice copy must not introduce a pointer box descriptor"
    assert not _REBOX_INTO_PTR_RE.search(ir)
    assert "hlfir.assign" in ir, \
        "plain slice copy should land as an hlfir.assign between boxes"


def test_detector_distinguishes_all_four_cases():
    """Across all four probes, rebox-into-pointer-with-rank-change is present only for the
    view case."""
    cases = [
        ("pointer_view_bounds_remap_probe.f90", True),
        ("reshape_intrinsic_copy_probe.f90", False),
        ("pointer_plain_no_remap_probe.f90", False),
        ("plain_slice_copy_probe.f90", False),
    ]
    for fname, expected in cases:
        ir = _emit_hlfir(_HERE / fname)
        got = _has_rank_changing_rebox_into_ptr(ir)
        assert got is expected, \
            f"{fname}: detector said {got}, expected {expected}"
