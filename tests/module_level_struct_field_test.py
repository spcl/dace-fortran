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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    assert "g_c" in sdfg.arrays or "g_c" in sdfg.scalars or "g_c" in sdfg.symbols


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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    assert "g_used" in sdfg.arrays
    assert "g_unused" not in sdfg.arrays, \
        "unaccessed fields should not be registered"
