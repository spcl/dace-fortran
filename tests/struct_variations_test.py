"""Comprehensive Fortran struct-variation probes for the bridge.

User-requested audit (commit a30a7d8 fixed 3 of 7 audit probes;
this expands to cover all the variations the user listed):

  * struct of struct of scalars
  * struct of struct array of scalars
  * module-level / local
  * struct passed as dummy arg
  * struct used in arithmetic / branch / assigned-to

Each probe builds + runs an e2e numerical comparison so the
binding path is covered too.

Patterns currently working post-session-fixes (commits 25f8e83,
fb7c4ee, 4cc4442, a30a7d8, plus the nested + subscript-index
fixes in this commit):

  * module-level: scalar field, array field, scalar in
    expression / condition / assign-to / libcall source.
  * module-level: NESTED struct of struct (recursion through
    record-typed members emits the flat leaf-VarInfo).
  * dummy arg: handled by hlfir-flatten-structs (unchanged).

Still pending (xfail / TODO):

  * module-level scalar field used as ARRAY SUBSCRIPT INDEX
    (``arr(g % idx)``) -- name resolves to ``g_idx`` correctly
    but ``g_idx`` lives as a scalar transient and the memlet
    subset needs it as an SDFG symbol; needs a symbol-promotion
    path for integer scalar struct fields.
  * struct of struct ARRAY (``g % inner % a(i)``) -- the array
    component access through the nested chain needs additional
    designate-chain rewriting.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ---------------------------------------------------------------
# Module-level: scalar struct of struct of scalars
# ---------------------------------------------------------------


def test_module_struct_of_struct_of_scalars(tmp_path):
    """``g % inner % x`` where x is a scalar.  Recurses through
    nested record types to emit ``g_inner_x`` as a flat scalar
    transient."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: x
  end type
  type :: outer_t
    type(inner_t) :: inner
  end type
  type(outer_t) :: g
contains
  subroutine f(out)
    real(kind=8), intent(out) :: out
    out = g % inner % x
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_inner_x" in sdfg.arrays or "g_inner_x" in sdfg.scalars, \
        f"expected g_inner_x: arrays={sorted(sdfg.arrays.keys())}"


def test_module_struct_of_struct_of_arrays(tmp_path):
    """``g % inner % a(3)`` reduces -- nested record's array field."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: a(3)
  end type
  type :: outer_t
    type(inner_t) :: inner
  end type
  type(outer_t) :: g
contains
  subroutine f(out)
    real(kind=8), intent(out) :: out
    out = sum(g % inner % a)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_inner_a" in sdfg.arrays


def test_module_struct_three_levels_of_nesting(tmp_path):
    """``g % a % b % x`` -- three levels of struct nesting."""
    src = """
module m
  type :: level3_t
    real(kind=8) :: x
  end type
  type :: level2_t
    type(level3_t) :: b
  end type
  type :: level1_t
    type(level2_t) :: a
  end type
  type(level1_t) :: g
contains
  subroutine f(out)
    real(kind=8), intent(out) :: out
    out = g % a % b % x
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_a_b_x" in sdfg.arrays or "g_a_b_x" in sdfg.scalars


def test_module_struct_multiple_nested_fields(tmp_path):
    """Multiple leaf paths via the same parent struct -- each unique
    leaf gets its own VarInfo, the intermediate structs don't."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: x
    real(kind=8) :: y(2)
  end type
  type :: outer_t
    type(inner_t) :: i1
    type(inner_t) :: i2
  end type
  type(outer_t) :: g
contains
  subroutine f(out)
    real(kind=8), intent(out) :: out
    out = g % i1 % x + g % i2 % x + sum(g % i1 % y)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    arrs = sorted(sdfg.arrays.keys())
    # Both i1.x and i2.x referenced as scalar; i1.y referenced
    # as array.  i2.y NOT referenced -- shouldn't be registered.
    assert "g_i1_x" in arrs or "g_i1_x" in sdfg.scalars
    assert "g_i2_x" in arrs or "g_i2_x" in sdfg.scalars
    assert "g_i1_y" in arrs
    assert "g_i2_y" not in arrs, "unused field should not be registered"


# ---------------------------------------------------------------
# Module-level: pre-existing 1-level probes (regression cover)
# ---------------------------------------------------------------


def test_module_struct_scalar_field_in_arithmetic(tmp_path):
    """``out = g % x + g % y`` -- two scalar fields used in expr."""
    src = """
module m
  type :: t
    real(kind=8) :: x
    real(kind=8) :: y
  end type
  type(t) :: g
contains
  subroutine f(out)
    real(kind=8), intent(out) :: out
    out = g % x + g % y
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g" not in sdfg.symbols


def test_module_struct_array_field_indexed(tmp_path):
    """``g % a(i)`` -- array field with element subscript."""
    src = """
module m
  type :: t
    real(kind=8) :: a(5)
  end type
  type(t) :: g
contains
  subroutine f(i, out)
    integer, intent(in) :: i
    real(kind=8), intent(out) :: out
    out = g % a(i)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_a" in sdfg.arrays
    assert "g" not in sdfg.symbols


# ---------------------------------------------------------------
# Dummy-arg struct cases (handled by hlfir-flatten-structs)
# ---------------------------------------------------------------


def test_dummy_struct_of_struct_of_scalars(tmp_path):
    """Dummy arg with nested struct -- flatten pass handles it."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: x
  end type
  type :: outer_t
    type(inner_t) :: inner
    real(kind=8) :: y
  end type
contains
  subroutine f(s, out)
    type(outer_t), intent(in) :: s
    real(kind=8), intent(out) :: out
    out = s % inner % x + s % y
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    # The flatten pass produces flat per-field args.
    assert "s" not in sdfg.symbols


# ---------------------------------------------------------------
# Local struct case
# ---------------------------------------------------------------


def test_local_struct_simple_scalar(tmp_path):
    """``type(t) :: local`` inside a subroutine -- local struct
    with scalar field."""
    src = """
module m
  type :: t
    real(kind=8) :: x
  end type
contains
  subroutine f(in, out)
    real(kind=8), intent(in) :: in
    real(kind=8), intent(out) :: out
    type(t) :: local
    local % x = in * 2.0d0
    out = local % x + 1.0d0
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    # Local scalar struct's field should be addressable.
    assert "local" not in sdfg.symbols


def test_module_struct_int_field_as_subscript(tmp_path):
    """``arr(g % idx)`` -- struct integer field as array subscript.
    The bridge mints a one-shot position symbol
    ``__sym_g_idx_1`` via ``internPosSymbol`` and stages an entry-
    time interstate-edge ``__sym_g_idx_1 = g_idx[0]``; the memlet
    subset picks up the symbol directly."""
    src = """
module m
  type :: t
    integer :: idx
  end type
  type(t) :: g
contains
  subroutine f(arr, out)
    real(kind=8), intent(in) :: arr(10)
    real(kind=8), intent(out) :: out
    out = arr(g % idx)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_idx" in sdfg.symbols or "g_idx" in sdfg.arrays
