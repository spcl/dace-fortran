"""Pin the IR-pattern claim that view-semantics and copy-semantics
Fortran array operations are always distinguishable at the HLFIR
level.

Approach A for handling Fortran 2003 bounds-remapping pointer
assignment (``ptr(lo:hi) => target(:, slice)`` -- a *view*, no copy)
relies on a single load-bearing claim: the op pattern flang emits
for the view case is structurally distinct from every copy-shaped
op pattern (``RESHAPE`` intrinsic, plain slice assignment).  If a
copy pattern could be mistaken for a view pattern the bridge would
silently turn copies into aliases, corrupting any kernel that
relies on copy semantics.

This test exercises four representative shapes and asserts the
op-pattern signature of each:

  * pointer_view_bounds_remap_probe.f90 -- the view case.
    Required signature: ``fir.rebox`` producing a
    ``!fir.box<!fir.ptr<...>>`` of different rank than its input.

  * reshape_intrinsic_copy_probe.f90 -- the RESHAPE copy.
    Required signature: ``hlfir.reshape`` op present; NO
    ``fir.rebox`` into a pointer.

  * pointer_plain_no_remap_probe.f90 -- plain pointer assign,
    no bounds remap, no rank change.
    Required signature: ``fir.embox`` into a pointer, same
    rank; the existing ``hlfir-rewrite-pointer-assigns`` pass
    handles this.

  * plain_slice_copy_probe.f90 -- plain slice assignment.
    Required signature: ``hlfir.assign`` between two
    same-rank boxes; no rebox/embox/reshape at all.

If any of these shapes shift in a future flang release the test
fails loudly -- the detector design in the bounds-remap-view
pass would need to be re-validated against the new pattern.
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


#: ``fir.rebox`` producing a ``!fir.box<!fir.ptr<...>>``.  The
#: shape-operand projection inside the parens isn't captured here --
#: the type spelling alone is enough to distinguish a pointer rebox
#: from any other fir.rebox use.  ``re.DOTALL`` lets the regex span
#: lines because flang sometimes wraps long type prints.
_REBOX_INTO_PTR_RE = re.compile(r"fir\.rebox\s+%\w+(?:\([^)]*\))?\s*:\s*\([^)]*\)\s*->\s*!fir\.box<!fir\.ptr<",
                                re.DOTALL)


#: A rank-changing rebox: input is ``!fir.array<?x?x...>`` (multiple
#: ``?``), output is ``!fir.array<?x...>`` (fewer ``?``).  Matched by
#: counting question marks in the input vs output types -- the
#: question marks are the dynamic-extent placeholders flang prints,
#: one per dim.
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
        # The above heuristics are flaky on flang's type printer; use a
        # simpler check: count ``?`` directly (every dyn dim).
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
    """The RESHAPE copy case: ``hlfir.reshape`` op; no
    ``fir.rebox`` into a pointer.  Detection cannot confuse this
    with the view case."""
    ir = _emit_hlfir(_HERE / "reshape_intrinsic_copy_probe.f90")
    assert "hlfir.reshape" in ir, \
        "RESHAPE probe missing the hlfir.reshape op"
    # The bounds-remap-view detector keys on rebox-into-pointer; this
    # probe must NOT have that pattern.
    assert not _REBOX_INTO_PTR_RE.search(ir), \
        "RESHAPE probe must not have fir.rebox into !fir.box<!fir.ptr<...>> -- " \
        "the bounds-remap-view detector would false-positive on copy semantics"


def test_pointer_plain_no_remap_uses_embox_not_rebox():
    """Plain pointer assign (no bounds remap): ``fir.embox`` (not
    rebox) into a pointer, same rank in/out.  The existing
    ``hlfir-rewrite-pointer-assigns`` pass handles this; the
    bounds-remap-view detector must not trigger."""
    ir = _emit_hlfir(_HERE / "pointer_plain_no_remap_probe.f90")
    assert "fir.embox" in ir, "plain pointer assign should use fir.embox"
    assert not _has_rank_changing_rebox_into_ptr(ir), \
        "plain pointer assign must not have rank-changing rebox -- " \
        "the bounds-remap-view detector would false-positive"


def test_plain_slice_copy_has_no_pointer_box_at_all():
    """Plain ``dst = src(:, 1:k)``: same-rank ``hlfir.assign`` between
    boxes; no pointer involved, so no rebox/embox into a pointer.
    The bounds-remap-view detector must not trigger."""
    ir = _emit_hlfir(_HERE / "plain_slice_copy_probe.f90")
    # No pointer box anywhere.
    assert "!fir.box<!fir.ptr<" not in ir, \
        "plain slice copy must not introduce a pointer box descriptor"
    assert not _REBOX_INTO_PTR_RE.search(ir)
    assert "hlfir.assign" in ir, \
        "plain slice copy should land as an hlfir.assign between boxes"


def test_detector_distinguishes_all_four_cases():
    """End-to-end: across all four probes, the rebox-into-pointer-
    with-rank-change signature is present in exactly one (the view)
    and absent in the other three (the copies / non-view-pointer)."""
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
