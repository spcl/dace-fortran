"""Audit struct-field name resolution across the bridge.

traceToDecl fix (25f8e83) flattens ``<parent>_<member>`` for
``hlfir.designate`` with a component attr, but other call sites call
``traceToDecl`` directly, bypassing that branch. Each test probes one
such site for a leaked bare struct base.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_struct_field_read_in_tasklet(tmp_path):
    """Scalar struct field read in a tasklet; fixed via expressions.cpp simplification (4cc4442)."""
    src = """
module m
  type :: t
    real(kind=8) :: x
  end type
  type(t) :: g
contains
  subroutine f(in, out)
    real(kind=8), intent(in) :: in
    real(kind=8), intent(out) :: out
    out = in + g % x
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    # Should NOT have ``g`` as a free symbol -- flat ``g_x`` is the access.
    assert "g" not in sdfg.symbols, f"struct base leaked as symbol: {sdfg.symbols}"


def test_module_struct_field_in_condition(tmp_path):
    """Scalar struct field as predicate -- interstate-edge conditional emitter must use flat name."""
    src = """
module m
  type :: t
    real(kind=8) :: threshold
  end type
  type(t) :: g
contains
  subroutine f(x, out)
    real(kind=8), intent(in) :: x
    real(kind=8), intent(out) :: out
    if (x > g % threshold) then
      out = 1.0d0
    else
      out = 0.0d0
    end if
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g" not in sdfg.symbols


def test_module_struct_field_in_subscript(tmp_path):
    """Struct field as array-index subscript -- bridge mints a position symbol via internPosSymbol for a closed-form memlet subset."""
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
    assert "g" not in sdfg.symbols


def test_module_struct_array_field_subscript(tmp_path):
    """g % a(i): array-typed struct field with element subscript -- different designate path than whole-array access."""
    src = """
module m
  type :: t
    real(kind=8) :: a(10)
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
    # flat g_a registered; element access g_a[i-1] lands as memlet subset.
    assert "g_a" in sdfg.arrays, f"flat g_a missing: {sorted(sdfg.arrays.keys())}"
    assert "g" not in sdfg.symbols


def test_module_struct_field_assigned_to(tmp_path):
    """Writing to a struct field -- assign-emitter target-name resolution must produce the flat name."""
    src = """
module m
  type :: t
    real(kind=8) :: y
  end type
  type(t) :: g
contains
  subroutine f(in)
    real(kind=8), intent(in) :: in
    g % y = in * 2.0d0
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_y" in sdfg.arrays or "g_y" in sdfg.scalars or "g_y" in sdfg.symbols
    assert "g" not in sdfg.symbols


def test_module_struct_field_as_libcall_source(tmp_path):
    """Struct field passed as matmul libcall source -- traceToDecl on the arg site must produce the flat name."""
    src = """
module m
  type :: t
    real(kind=8) :: a(3, 3)
  end type
  type(t) :: g
contains
  subroutine f(b, out)
    real(kind=8), intent(in) :: b(3, 3)
    real(kind=8), intent(out) :: out(3, 3)
    out = MATMUL(g % a, b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_a" in sdfg.arrays
    A = np.eye(3, dtype=np.float64, order='F')
    B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    out = np.zeros((3, 3), dtype=np.float64, order='F')
    sdfg(g_a=A, b=B, out=out)
    np.testing.assert_allclose(out, A @ B)


def test_module_struct_nested_field(tmp_path):
    """Nested struct member access (g % inner % a) -- fixed by recursing extract_vars's per-field synthesis through record-typed members; each leaf emits its own flat VarInfo."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: a(3)
  end type
  type :: t
    type(inner_t) :: inner
  end type
  type(t) :: g
contains
  subroutine f(out)
    real(kind=8), intent(out) :: out
    out = sum(g % inner % a)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    assert "g_inner_a" in sdfg.arrays
