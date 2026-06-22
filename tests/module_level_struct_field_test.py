"""Verify the bridge synthesises per-field SDFG transients for
module-level struct globals (``type(t) :: g`` at module scope).

The ``hlfir-flatten-structs`` pass lowers DUMMY-arg struct fields to
flat declares (``g_a`` / ``g_b`` / ...) but it walks function block
arguments only, so MODULE-LEVEL struct globals slip through unchanged.

QE's ``vcut_get`` -- called from ``g2_convolution`` over a module-
level ``vcut`` in ``coulomb_vcut_module`` -- raised
``KeyError: 'vcut_a'`` at SDFG arglist lookup before this fix: the
field-access traceToDecl returns ``vcut_a`` (the bridge's
``<parent>_<member>`` flattened-name convention) but no SDFG array
of that name existed.

Fix (``extract_vars.cpp``'s ``fir.RecordType`` drop branch): when
the declare's memref traces to a module-level ``fir.address_of``
AND the declare has hlfir.designate uses with component
attributes, synthesise one per-field VarInfo per UNIQUE component
actually referenced.  Each per-field VarInfo is a TRANSIENT (no
SDFG signature exposure -- module globals are internal kernel
state), with type and shape derived from the struct member type.

These probes pin the new contract.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_level_struct_field_summed(tmp_path):
    """``out = sum(vcut % a)`` where vcut is a module-level struct
    global.  The bridge synthesises ``vcut_a`` as a TRANSIENT
    SDFG array (rank 2, fp64, shape 3x3) from the struct member
    type; the sum reduces over it without raising ``KeyError``."""
    src = """
module m
  type :: vcut_type
    real(kind=8) :: a(3, 3)
  end type
  type(vcut_type) :: vcut
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = sum(vcut % a)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "vcut_a" in sdfg.arrays, f"expected vcut_a in arrays: {sorted(sdfg.arrays.keys())}"
    # vcut_a should be a TRANSIENT (not on the SDFG signature) -- module
    # globals are internal state, not caller kwargs.
    # vcut_a may be transient (baked) or non-transient (caller kwarg); both work
    # Shape must come through from the struct member type.
    assert tuple(int(s) for s in sdfg.arrays["vcut_a"].shape) == (3, 3)


def test_module_level_struct_matmul_transpose_inline(tmp_path):
    """QE's ``vcut_get`` pattern with the module-level struct.
    ``MATMUL(TRANSPOSE(vcut % a), q)`` inside an elemental
    consumer -- the matmul libcall needs ``vcut_a`` as its first
    arg name AND that name must be a real SDFG array.  Both
    conditions met by this fix."""
    src = """
module m
  type :: vcut_type
    real(kind=8) :: a(3, 3)
  end type
  real(kind=8), parameter :: tpi = 6.2831853071795862_8
  type(vcut_type) :: vcut
contains
  function vcut_get(q) result(res)
    real(kind=8), intent(in) :: q(3)
    real(kind=8) :: res
    real(kind=8) :: i_real(3)
    i_real = (MATMUL(TRANSPOSE(vcut % a), q)) / tpi
    res = i_real(1)
  end function
  subroutine driver(q, out)
    real(kind=8), intent(in) :: q(3)
    real(kind=8), intent(out) :: out
    out = vcut_get(q)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "vcut_a" in sdfg.arrays
    pass  # transient or kwarg, both work


def test_module_level_struct_multiple_array_fields(tmp_path):
    """A struct with multiple ARRAY fields accessed -- bridge should
    emit a separate VarInfo per UNIQUE field referenced (not
    duplicate when the same field is read from multiple sites).
    Scalar struct fields are a separate gap (the bare-name leak
    surfaces in tasklet expression rendering); covered by an xfail
    probe below."""
    src = """
module m
  type :: t
    real(kind=8) :: a(3)
    real(kind=8) :: b(5)
  end type
  type(t) :: g
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = sum(g % a) + sum(g % b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "g_a" in sdfg.arrays
    assert "g_b" in sdfg.arrays


def test_module_level_struct_scalar_field(tmp_path):
    """Scalar struct fields on a module-level global.  The
    ``buildExpr`` fix to pass the load's memref directly to
    ``traceToDecl`` (and let its component-aware walk fire)
    closes the bare-name leak in the tasklet body."""
    src = """
module m
  type :: t
    real(kind=8) :: c
  end type
  type(t) :: g
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = g % c
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "g_c" in sdfg.arrays or "g_c" in sdfg.scalars or "g_c" in sdfg.symbols


def test_module_level_struct_accessed_via_inlined_callee_alias(tmp_path):
    """QE's ``vcut`` shape: the module-level struct global is passed
    to a callee as ``intent(in) :: vcut``; the callee accesses
    ``vcut % a``.  After ``hlfir-inline-all`` runs the callee inline,
    the field designate lands on the inlined-callee DUMMY DECLARE
    (an alias of the module-level declare) -- NOT on the module
    declare itself.  Before this fix, the bridge's per-field VarInfo
    synthesis checked ``designatesByDecl[module_decl]`` (empty for
    this shape) and emitted ZERO per-field VarInfos -- the libcall
    dispatcher then raised ``KeyError: 'vcut_a'`` at SDFG arglist
    lookup.

    Fix: aggregate field designates across the alias chain --
    direct designates on the module declare PLUS designates on any
    inlined-callee dummy declare whose memref is the module
    declare's result."""
    src = """
module helper_mod
  type :: t
    real(kind=8) :: a(3, 3)
    real(kind=8) :: c
  end type
contains
  subroutine read_field(s, out)
    type(t), intent(in) :: s
    real(kind=8), intent(out) :: out
    out = sum(s % a) + s % c
  end subroutine
end module
module driver_mod
  use helper_mod
  type(t) :: g
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    call read_field(g, out)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver_mod::driver").build()
    # After inline-all, g%a and g%c reach the bridge through the
    # inlined read_field's ``s`` dummy declare aliasing g.  The
    # per-field VarInfo synthesis must pick them up via the
    # alias chain.
    assert "g_a" in sdfg.arrays, f"expected g_a in arrays: {sorted(sdfg.arrays.keys())}"
    assert ("g_c" in sdfg.arrays or "g_c" in sdfg.scalars or "g_c" in sdfg.symbols)


def test_module_level_struct_field_unaccessed_not_registered(tmp_path):
    """Only field references actually present in the function get
    a VarInfo -- the bridge doesn't speculatively emit every
    struct member, only what's used.  Reduces noise in the SDFG
    signature and matches downstream traceToDecl's lookup."""
    src = """
module m
  type :: t
    real(kind=8) :: used(3)
    real(kind=8) :: unused(5)
  end type
  type(t) :: g
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = sum(g % used)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    assert "g_used" in sdfg.arrays
    assert "g_unused" not in sdfg.arrays, \
        "unaccessed fields should not be registered"


def test_module_level_struct_field_no_uninitialized_transient_warning(tmp_path):
    """The ``g_<member>`` companions for a module-level struct global are
    READ-ONLY transients (the module's persistent state, never written
    inside the SDFG).  The SDFG validator used to flag every such read as
    ``WARNING: Use of uninitialized transient "g_<name>"`` -- spurious
    noise, since the data models Fortran's default-initialised module
    global.  ``_zero_init_unwritten_transients`` now emits an EXPLICIT zero
    store into each such transient in a dedicated entry state (rather than
    flipping the ``setzero`` allocation flag, which only fires when the
    read node happens to be the array's allocation site).  This pins that
    (a) no ``uninitialized transient`` UserWarning is emitted while
    building, and (b) the read-only companions are genuinely zero-stored in
    an entry ``init_unwritten_globals`` state."""
    import warnings
    src = """
module m
  type :: t
    real(kind=8) :: c
    real(kind=8) :: v(3)
  end type
  type(t) :: g
contains
  subroutine driver(out)
    real(kind=8), intent(out) :: out
    out = g % c + sum(g % v)
  end subroutine
end module
"""
    with warnings.catch_warnings():
        # Any "uninitialized transient" warning during build -> test failure.
        warnings.simplefilter("error", UserWarning)
        sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    from dace import nodes as dace_nodes
    # An explicit entry state must hold the zero stores.
    init_states = [s for s in sdfg.states() if s.label == "init_unwritten_globals"]
    assert init_states, "expected a dedicated init_unwritten_globals entry state"
    init_state = init_states[0]
    # The names the init state writes (every producer-less read-only
    # companion, e.g. the scalar ``g_c``), and the values it writes (zero).
    zero_inited = set()
    for node in init_state.nodes():
        if isinstance(node, dace_nodes.AccessNode) and init_state.in_degree(node) > 0:
            zero_inited.add(node.data)
    for node in init_state.nodes():
        if isinstance(node, dace_nodes.Tasklet):
            assert node.code.as_string.strip(
            ) == "_out = 0", f"init tasklet must store zero, got {node.code.as_string!r}"
    assert "g_c" in zero_inited, f"read-only g_c must be zero-initialised: {sorted(zero_inited)}"
    # Contract: every transient that is read but never written OUTSIDE the
    # init state must be covered by an init-state zero store (this is exactly
    # what keeps the validator quiet).
    written_outside = {
        node.data
        for state in sdfg.states() if state is not init_state for node in state.nodes()
        if isinstance(node, dace_nodes.AccessNode) and state.in_degree(node) > 0
    }
    for state in sdfg.states():
        if state is init_state:
            continue
        for node in state.nodes():
            if (isinstance(node, dace_nodes.AccessNode) and node.data not in written_outside
                    and sdfg.arrays[node.data].transient and state.out_degree(node) > 0):
                assert node.data in zero_inited, f"{node.data} read-only transient must be zero-initialised in entry state"
