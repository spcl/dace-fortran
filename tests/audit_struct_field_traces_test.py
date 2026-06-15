"""Audit probes for struct-field name resolution across the bridge.

Session's traceToDecl fix (commit 25f8e83) built the flattened
``<parent>_<member>`` name for ``hlfir.designate`` ops with a
component attribute.  Several other call sites in the bridge call
``traceToDecl(<dg|ld>.getMemref())`` directly -- bypassing the
component branch.  These probes test each context to find which
ones now correctly produce the flat name and which still leak the
bare struct base.

Audited contexts:

  * struct field read in tasklet body (expressions.cpp:1437 -- fixed)
  * struct field as libcall source (elementals.cpp:298)
  * struct field as section/element subscript
  * struct field in branch condition
  * struct field assigned to from RHS
  * NESTED struct field (``g % inner % a``)
  * struct field with array element index (``g % a(i)``)
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_struct_field_read_in_tasklet(tmp_path):
    """Scalar struct field read in a plain arithmetic tasklet (fixed
    via expressions.cpp simplification in 4cc4442)."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    # Should NOT have ``g`` as a free symbol -- flat ``g_x`` is the access.
    assert "g" not in sdfg.symbols, f"struct base leaked as symbol: {sdfg.symbols}"


def test_module_struct_field_in_condition(tmp_path):
    """Scalar struct field as predicate -- the bridge's interstate-
    edge / conditional emitter must also use the flat name."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    assert "g" not in sdfg.symbols


def test_module_struct_field_in_subscript(tmp_path):
    """Scalar struct field used as ARRAY INDEX in subscript.  The
    bridge mints a one-shot position symbol via ``internPosSymbol``
    so the memlet subset gets a closed-form expression."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    assert "g" not in sdfg.symbols


def test_module_struct_array_field_subscript(tmp_path):
    """``g % a(i)`` -- struct field that's ITSELF an array, accessed
    with an element subscript.  This goes through a different
    designate path than a whole-array field access."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    # The flat array ``g_a`` should be registered; element access
    # ``g_a[i-1]`` lands as memlet subset.
    assert "g_a" in sdfg.arrays, f"flat g_a missing: {sorted(sdfg.arrays.keys())}"
    assert "g" not in sdfg.symbols


def test_module_struct_field_assigned_to(tmp_path):
    """Writing to a struct field via the bridge's assign emitter --
    target-name resolution must produce the flat name."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    assert "g_y" in sdfg.arrays or "g_y" in sdfg.scalars or "g_y" in sdfg.symbols
    assert "g" not in sdfg.symbols


def test_module_struct_field_as_libcall_source(tmp_path):
    """Pass a struct field as a libcall (matmul) source.  Calls
    ``traceToDecl`` on the libcall arg site -- must produce the
    flat name."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    assert "g_a" in sdfg.arrays
    A = np.eye(3, dtype=np.float64, order='F')
    B = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    out = np.zeros((3, 3), dtype=np.float64, order='F')
    sdfg(g_a=A, b=B, out=out)
    np.testing.assert_allclose(out, A @ B)


def test_module_struct_nested_field(tmp_path):
    """Nested struct member access (``g % inner % a``).  Fixed by
    extending extract_vars's per-field synthesis to RECURSE through
    record-typed members -- each leaf path emits its own flat
    VarInfo (``g_inner_a``)."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="f").build()
    assert "g_inner_a" in sdfg.arrays
