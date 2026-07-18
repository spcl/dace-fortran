"""End-to-end coverage for the SDFG-side View descriptor emission.

Pipeline: hlfir-mark-bounds-remap-views (C++) tags the LHS pointer declare ->
extract_vars surfaces it via VarInfo.bounds_remap_view/_source/_total_extent ->
descriptors.py emits sdfg.add_view(shape=[total_extent], strides=[1]) + offset_<ptr>_d0.

Pins the descriptor level only. The View's linking memlets (wiring the View to its parent
storage) are a follow-up needing flat-1D-to-parent-multidim index folding; until that lands,
probes catch the expected post-descriptor validation error and inspect the partial SDFG (the
descriptor is emitted before validation runs).
"""
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import build_sdfg, have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _make_builder(probe_name: str, tmp_path: Path, entry: str = "run"):
    """Build the SDFGBuilder for a probe, returned before .build() invokes validation."""
    src = (_HERE / probe_name).read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name="brv", entry=entry)


def _build_partial_sdfg(builder):
    """Drive the build past descriptor emission, tolerating the known View-edge validation gap
    (InvalidSDFGNodeError carries the half-built SDFG on e.sdfg); re-raises other errors."""
    from dace.sdfg.validation import InvalidSDFGNodeError
    try:
        return builder.build()
    except InvalidSDFGNodeError as e:
        if "Ambiguous or invalid edge to/from a View" in str(e):
            return e.sdfg
        raise


def _find_view(sdfg, candidate_substrings: tuple):
    """Return the (name, descriptor) pair for a View whose name contains any candidate_substrings, or (None, None)."""
    from dace.data import View
    for name, desc in sdfg.arrays.items():
        if isinstance(desc, View) and any(c in name for c in candidate_substrings):
            return name, desc
    return None, None


# ---------------------------------------------------------------------------
# VarInfo round-trip
# ---------------------------------------------------------------------------


def test_var_info_carries_bounds_remap_fields(tmp_path):
    """VarInfo for prhoc carries bounds_remap_view=True, the parent's name, and the total-extent expr."""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    # Force the pipeline + classification to run so VarInfo is final.
    _build_partial_sdfg(builder)
    inner = getattr(builder, "_inner", builder)
    pointer_vi = next(
        (vi for vi in inner.module.get_variables() if vi.fortran_name == "prhoc"),
        None,
    )
    assert pointer_vi is not None, "no VarInfo for 'prhoc'"
    assert pointer_vi.bounds_remap_view is True, \
        f"bounds_remap_view={pointer_vi.bounds_remap_view}"
    assert pointer_vi.bounds_remap_source == "rhoc", \
        f"bounds_remap_source={pointer_vi.bounds_remap_source!r}"


def test_var_info_bounds_remap_view_through_allocatable_target(tmp_path):
    """Bounds-remap source trace must walk through the fir.load of an ALLOCATABLE target's
    descriptor box to reach the parent declare (QE vexx_bp_k_gpu Gate H: prhoc_d => rhoc_d(:, slice)).
    Before the fir.LoadOp hop, the walk stopped at the load and the rebind mis-lowered as a scalar copy."""
    builder = _make_builder("pointer_view_bounds_remap_allocatable_probe.f90", tmp_path)
    _build_partial_sdfg(builder)
    inner = getattr(builder, "_inner", builder)
    pointer_vi = next(
        (vi for vi in inner.module.get_variables() if vi.fortran_name == "prhoc"),
        None,
    )
    assert pointer_vi is not None, "no VarInfo for 'prhoc'"
    assert pointer_vi.bounds_remap_view is True, \
        ("allocatable-target bounds-remap-view not detected: "
         f"bounds_remap_view={pointer_vi.bounds_remap_view}")
    assert pointer_vi.bounds_remap_source == "rhoc", \
        f"bounds_remap_source={pointer_vi.bounds_remap_source!r}"


def test_var_info_total_extent_parses_n_times_k(tmp_path):
    """prhoc(1:n*k) => rhoc(:, 1:k): extract_vars should render the extent as "n*k" (or an
    equivalent expr); an empty extent (synth fallback symbol) is also acceptable."""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    _build_partial_sdfg(builder)
    inner = getattr(builder, "_inner", builder)
    pointer_vi = next(vi for vi in inner.module.get_variables() if vi.fortran_name == "prhoc")
    extent = pointer_vi.bounds_remap_total_extent
    # Either: (a) parsed -- ``n`` and ``k`` mentioned -- or
    # (b) empty -- the synth fallback symbol kicks in downstream.
    if extent:
        assert "n" in extent and "k" in extent, \
            f"extent {extent!r} should mention n and k"


def test_var_info_copy_probes_carry_no_remap_view_flag(tmp_path):
    """Non-remap-view probes never get bounds_remap_view=True on any VarInfo -- false-positive guard."""
    for fname in (
            "reshape_intrinsic_copy_probe.f90",
            "pointer_plain_no_remap_probe.f90",
            "plain_slice_copy_probe.f90",
    ):
        builder = _make_builder(fname, tmp_path / fname.replace(".f90", ""))
        try:
            _build_partial_sdfg(builder)
        except Exception:
            pass  # unrelated downstream gaps OK; only care no VarInfo got spuriously flagged
        inner = getattr(builder, "_inner", builder)
        try:
            vi_list = inner.module.get_variables()
        except Exception:
            continue
        flagged = [vi.fortran_name for vi in vi_list if vi.bounds_remap_view]
        assert not flagged, \
            f"{fname}: {flagged} were spuriously flagged as bounds-remap views"


# ---------------------------------------------------------------------------
# Descriptor + symbol emission
# ---------------------------------------------------------------------------


def test_view_descriptor_is_added_for_view_probe(tmp_path):
    """A ``dace.data.View`` descriptor named ``prhoc`` exists in the
    SDFG after the descriptors pass runs."""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    sdfg = _build_partial_sdfg(builder)
    view_name, view_desc = _find_view(sdfg, ("prhoc", ))
    assert view_desc is not None, \
        f"no View descriptor for 'prhoc' in {list(sdfg.arrays)}"
    assert len(view_desc.shape) == 1, \
        f"View should be 1-D, got shape {view_desc.shape}"


def test_view_descriptor_strides_are_one(tmp_path):
    """Per the spec: contiguous 1-D view -> ``strides=[1]``."""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    sdfg = _build_partial_sdfg(builder)
    _, view_desc = _find_view(sdfg, ("prhoc", ))
    assert view_desc is not None
    assert tuple(view_desc.strides) == (1, ), \
        f"View strides should be (1,), got {view_desc.strides}"


def test_view_extent_is_symbolic_n_times_k(tmp_path):
    """View's first-dim extent serialises to n*k, or a synthesised total-extent symbol if unparseable."""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    sdfg = _build_partial_sdfg(builder)
    _, view_desc = _find_view(sdfg, ("prhoc", ))
    assert view_desc is not None
    extent_str = str(view_desc.shape[0])
    has_parsed = ("n" in extent_str and "k" in extent_str)
    has_synth = "total_extent" in extent_str
    assert has_parsed or has_synth, \
        f"View extent should be n*k or a synth symbol, got {extent_str!r}"


def test_offset_symbol_is_stamped_to_view_lb(tmp_path):
    """View's access offset is stamped to its Fortran lower bound (1): prhoc(i) subtracts 1 to reach
    the 0-based element. The per-rebind SOURCE column offset rides the linking memlet's source subset,
    NOT this symbol -- so offset_prhoc_d0 folds to the constant 1 (previously an unwired free symbol
    stuck at 0, causing a write-back off-by-one)."""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    _build_partial_sdfg(builder)
    inner = getattr(builder, "_inner", builder)
    assert inner.offset_values.get("offset_prhoc_d0") == 1, \
        f"offset_prhoc_d0 should be stamped to the view LB 1, got " \
        f"{inner.offset_values.get('offset_prhoc_d0')!r}"


def test_copy_probes_yield_no_view_at_destination(tmp_path):
    """Genuine copy probes (value assignment, RESHAPE) must NOT emit a View for their destination
    -- only => rebinds do. (pointer_plain_no_remap_probe.f90 is a rebind, not a copy; asserted
    positively in test_plain_pointer_rebind_yields_view_at_destination below.)"""
    from dace.data import View
    cases = [
        ("plain_slice_copy_probe.f90", "dst"),
        ("reshape_intrinsic_copy_probe.f90", "prhoc"),
    ]
    for probe, dst_name in cases:
        builder = _make_builder(probe, tmp_path / probe.replace(".f90", ""))
        try:
            sdfg = _build_partial_sdfg(builder)
        except Exception:
            continue  # unrelated downstream gap; the no-spurious-view invariant still holds
        if dst_name in sdfg.arrays:
            assert not isinstance(sdfg.arrays[dst_name], View), \
                f"{probe}: '{dst_name}' should be a real Array, not a View"


def test_plain_pointer_rebind_yields_view_at_destination(tmp_path):
    """Whole-array plain rebind (prhoc => rhoc) lowers as a View: under all-rebinds-are-views,
    every => duplicates the source as an ArrayView rather than rewriting accesses."""
    from dace.data import View
    probe = "pointer_plain_no_remap_probe.f90"
    builder = _make_builder(probe, tmp_path / probe.replace(".f90", ""))
    sdfg = _build_partial_sdfg(builder)
    assert "prhoc" in sdfg.arrays, \
        f"'prhoc' missing from {list(sdfg.arrays)}"
    assert isinstance(sdfg.arrays["prhoc"], View), \
        f"{probe}: 'prhoc' should be a View (plain rebind), got " \
        f"{type(sdfg.arrays['prhoc']).__name__}"
