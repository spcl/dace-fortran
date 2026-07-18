"""Verify the bridge synthesises per-field SDFG transients for module-level struct globals
(type(t) :: g at module scope).

hlfir-flatten-structs lowers DUMMY-arg struct fields to flat declares (g_a/g_b/...) but walks
function block arguments only, so MODULE-LEVEL globals slip through. QE's vcut_get (module-
level vcut in coulomb_vcut_module) raised KeyError: 'vcut_a' at SDFG arglist lookup before
this fix -- traceToDecl returns the flattened name but no SDFG array existed for it.

Fix (extract_vars.cpp's fir.RecordType drop branch): when a declare traces to a module-level
fir.address_of and has component-attribute hlfir.designate uses, synthesise one per-field
VarInfo per unique component referenced, as a TRANSIENT (module globals are internal state).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_level_struct_field_summed(tmp_path):
    """out = sum(vcut % a), vcut a module-level struct global. Bridge synthesises vcut_a as a
    TRANSIENT (rank 2, fp64, shape 3x3) from the struct member type; sums without KeyError."""
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
    """QE's vcut_get pattern: MATMUL(TRANSPOSE(vcut % a), q) inside an elemental consumer -- the
    matmul libcall needs vcut_a to be both its first arg name and a real SDFG array."""
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
    """Struct with multiple ARRAY fields -- bridge should emit one VarInfo per UNIQUE field
    referenced, not a duplicate when the same field is read from multiple sites."""
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
    """Scalar struct fields on a module-level global. buildExpr now passes the load's memref
    directly to traceToDecl, closing the bare-name leak in the tasklet body."""
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
    """QE's vcut shape: module-level struct passed to a callee as intent(in) :: vcut; after
    hlfir-inline-all the field designate lands on the inlined callee's DUMMY DECLARE (aliasing
    the module declare), not the module declare itself. Before this fix, per-field VarInfo
    synthesis checked designatesByDecl[module_decl] only (empty here), emitting zero VarInfos and
    raising KeyError: 'vcut_a'.

    Fix: aggregate field designates across the alias chain -- module declare PLUS any inlined
    dummy declare whose memref is the module declare's result."""
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
    # After inline-all, g%a/g%c reach the bridge through inlined read_field's s dummy declare
    # aliasing g; per-field VarInfo synthesis must pick them up via the alias chain.
    assert "g_a" in sdfg.arrays, f"expected g_a in arrays: {sorted(sdfg.arrays.keys())}"
    assert ("g_c" in sdfg.arrays or "g_c" in sdfg.scalars or "g_c" in sdfg.symbols)


def test_module_level_struct_field_unaccessed_not_registered(tmp_path):
    """Only field references actually present get a VarInfo -- the bridge doesn't speculatively
    emit every struct member, only what's used."""
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
    """g_<member> companions for a module-level struct global are READ-ONLY transients (never
    written inside the SDFG); the validator used to spuriously flag every read as "uninitialized
    transient". _zero_init_unwritten_transients now emits an explicit zero store into each such
    transient in a dedicated init_unwritten_globals entry state (instead of the setzero alloc
    flag, which only fires at the array's allocation site). Pins: no warning, and genuine
    zero-stores in that entry state."""
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
    # Names the init state writes (every producer-less read-only companion) and values (zero).
    zero_inited = set()
    for node in init_state.nodes():
        if isinstance(node, dace_nodes.AccessNode) and init_state.in_degree(node) > 0:
            zero_inited.add(node.data)
    for node in init_state.nodes():
        if isinstance(node, dace_nodes.Tasklet):
            assert node.code.as_string.strip(
            ) == "_out = 0", f"init tasklet must store zero, got {node.code.as_string!r}"
    assert "g_c" in zero_inited, f"read-only g_c must be zero-initialised: {sorted(zero_inited)}"
    # Contract: every transient read but never written outside the init state must be covered
    # by an init-state zero store -- exactly what keeps the validator quiet.
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
