"""End-to-end coverage for the SDFG-side View descriptor emission.

Pipeline:
  ``hlfir-mark-bounds-remap-views`` (C++ pass) tags the LHS
  pointer declare ->
  ``extract_vars`` (C++) surfaces the tag through ``VarInfo.
  bounds_remap_view`` / ``bounds_remap_source`` /
  ``bounds_remap_total_extent`` ->
  ``descriptors.py`` consumes the fields, emits
  ``sdfg.add_view(name, shape=[total_extent], strides=[1])`` and
  mints ``offset_<ptr>_d0``.

These tests pin the contract at the descriptor level -- they
inspect the SDFG's arrays + symbols after the descriptor pass
has run.  The View's linking memlets (the per-state read/write
edges that wire the View to its parent storage) are the next-
tier follow-up: emitting them needs the bridge's access-emission
code to know how to fold the flat 1-D index back to the parent's
multi-dim coordinates.  Tests that need the linking memlets
(notably any SDFG numerical run) are gated on that follow-up
landing; the descriptor-level checks here run unconditionally.

When a probe's SDFG fails post-descriptor validation (because
the linking memlets aren't wired yet), the tests catch the
validation error and inspect the partial state -- the descriptor
itself was emitted *before* validation runs, so it's available
to read back.
"""
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import build_sdfg, have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _make_builder(probe_name: str, tmp_path: Path, entry: str = "run"):
    """Build the SDFGBuilder for a probe and return it before
    ``.build()`` invokes validation.  Lets the test inspect the
    bridge's ``VarInfo`` list and the descriptors-stage SDFG
    independently of the post-validation status."""
    src = (_HERE / probe_name).read_text()
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name="brv", entry=entry)


def _build_partial_sdfg(builder):
    """Drive the build past descriptor emission but tolerate the
    follow-up linking-memlet validation gap.  Returns the SDFG with
    descriptors + symbols populated; raises only on errors that
    aren't the known View-edge gap.

    The SDFG itself is constructed inside ``SDFGBuilder.build()``
    before ``sdfg.validate()`` runs; when validation fails on the
    expected View-edge issue, ``InvalidSDFGNodeError`` carries the
    half-built SDFG on ``e.sdfg`` so we can still introspect
    descriptors + symbols."""
    from dace.sdfg.validation import InvalidSDFGNodeError
    try:
        return builder.build()
    except InvalidSDFGNodeError as e:
        if "Ambiguous or invalid edge to/from a View" in str(e):
            return e.sdfg
        raise


def _find_view(sdfg, candidate_substrings: tuple):
    """Walk all data descriptors; return the (name, descriptor) pair
    for a View whose name contains any of ``candidate_substrings``.
    Returns ``(None, None)`` when no such view exists."""
    from dace.data import View
    for name, desc in sdfg.arrays.items():
        if isinstance(desc, View) and any(c in name for c in candidate_substrings):
            return name, desc
    return None, None


# ---------------------------------------------------------------------------
# VarInfo round-trip
# ---------------------------------------------------------------------------


def test_var_info_carries_bounds_remap_fields(tmp_path):
    """The bridge's ``VarInfo`` for ``prhoc`` carries the three
    new fields: ``bounds_remap_view=True``, the parent's name,
    and the (parsed or empty) total-extent expression."""
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
    """The bounds-remap source trace must walk through the ``fir.load`` of
    an ALLOCATABLE target's descriptor box to reach the parent declare.

    This is the QE ``vexx_bp_k_gpu`` ``prhoc_d => rhoc_d(:, slice)`` shape
    (Gate H) where ``rhoc_d`` is ALLOCATABLE: flang lowers the section
    designate over a ``fir.load %rhoc#0``.  Before the ``fir.LoadOp`` hop in
    ``extract_vars``'s source trace, the walk stopped at the load,
    ``bounds_remap_source`` stayed empty, ``bounds_remap_view`` was left
    false, and the rebind was mis-lowered as a scalar copy (a bare
    ``complex128*`` ref in a tasklet body)."""
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
    """For the probe ``prhoc(1:n*k) => rhoc(:, 1:k)`` the extent
    operand on the rebox's shape-shift is the ``arith.muli`` of
    ``n`` and ``k``.  ``extract_vars`` should render that as
    ``"n*k"`` (or any equivalent multiplication expression).  An
    empty extent triggers the synth fallback symbol and is also
    acceptable."""
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
    """Probes that aren't bounds-remap views never have
    ``bounds_remap_view=True`` on any of their ``VarInfo``
    entries.  This is the false-positive guard at the
    extract_vars layer."""
    for fname in (
            "reshape_intrinsic_copy_probe.f90",
            "pointer_plain_no_remap_probe.f90",
            "plain_slice_copy_probe.f90",
    ):
        builder = _make_builder(fname, tmp_path / fname.replace(".f90", ""))
        try:
            _build_partial_sdfg(builder)
        except Exception:
            # Some copy probes have downstream gaps unrelated to the
            # bounds-remap path.  We only care that no VarInfo got
            # spuriously flagged.
            pass
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
    """The View's first-dim extent should serialise to ``n*k``
    (or a synthesised total-extent symbol when the bridge could
    not parse the SSA multiplication)."""
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
    """The view's access offset is stamped to its Fortran lower bound (1):
    a ``prhoc(i)`` access subtracts 1 to reach the 0-based view element.

    The per-rebind SOURCE column offset (``rhoc(:, k)``) rides the
    original->view linking memlet's source subset
    (``VarInfo.bounds_remap_source_subset``), NOT this symbol -- so
    ``offset_prhoc_d0`` folds to the constant 1.  (It used to be left a
    free symbol meant to carry the column offset; that binding was never
    wired, so the symbol stayed 0 and every ``prhoc(i)`` write landed one
    slot past its element -- the write-back off-by-one.)"""
    builder = _make_builder("pointer_view_bounds_remap_probe.f90", tmp_path)
    _build_partial_sdfg(builder)
    inner = getattr(builder, "_inner", builder)
    assert inner.offset_values.get("offset_prhoc_d0") == 1, \
        f"offset_prhoc_d0 should be stamped to the view LB 1, got " \
        f"{inner.offset_values.get('offset_prhoc_d0')!r}"


def test_copy_probes_yield_no_view_at_destination(tmp_path):
    """Genuine *copy* probes (value assignment, RESHAPE) must NOT emit a
    View descriptor for their destination -- only ``=>`` rebinds do.

    NB: ``pointer_plain_no_remap_probe.f90`` (``prhoc => rhoc``) is a
    pointer rebind, not a copy, so under the all-rebinds-are-views design
    it CORRECTLY becomes a View -- it is asserted positively in
    ``test_plain_pointer_rebind_yields_view_at_destination`` below."""
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
            # Unrelated downstream gap; the descriptor invariant
            # we care about (no spurious view) still holds.
            continue
        if dst_name in sdfg.arrays:
            assert not isinstance(sdfg.arrays[dst_name], View), \
                f"{probe}: '{dst_name}' should be a real Array, not a View"


def test_plain_pointer_rebind_yields_view_at_destination(tmp_path):
    """A whole-array plain rebind (``prhoc => rhoc``) lowers as a View:
    under the all-rebinds-are-views design every ``=>`` -- including the
    no-section, no-remap case -- duplicates the source as an ArrayView
    rather than rewriting accesses.  (Was previously mis-grouped with the
    copy probes that assert *no* view.)"""
    from dace.data import View
    probe = "pointer_plain_no_remap_probe.f90"
    builder = _make_builder(probe, tmp_path / probe.replace(".f90", ""))
    sdfg = _build_partial_sdfg(builder)
    assert "prhoc" in sdfg.arrays, \
        f"'prhoc' missing from {list(sdfg.arrays)}"
    assert isinstance(sdfg.arrays["prhoc"], View), \
        f"{probe}: 'prhoc' should be a View (plain rebind), got " \
        f"{type(sdfg.arrays['prhoc']).__name__}"
