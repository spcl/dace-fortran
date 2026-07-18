"""Fortran ``POINTER`` rebinding under the strict-no-aliasing assumption: the bridge
collapses ``ptr => target`` in ``hlfir-rewrite-pointer-assigns`` so every read/write of the
pointer becomes an access to the rebind target's storage -- unsafe if the program relies on
aliasing (the pass warns per firing)."""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")


def test_pointer_to_scalar_local(tmp_path: Path):
    """``tmp => x; tmp = 13; res = tmp + 1``  --  pointer to a scalar local."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out
  integer, target  :: x
  integer, pointer :: tmp
  x = 0
  tmp => x
  tmp = 13
  out = tmp + 1
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_to_scalar_local")
    out_ref = mod.main()
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert int(out[0]) == int(out_ref) == 14


def test_scalar_rebind_lowers_as_length_one_view(tmp_path: Path):
    """``tmp => x`` lowers as a length-1-array View of ``x``: writes through ``tmp`` alias
    ``x``, and ``x`` is a length-1 ``Array`` (not ``Scalar``) so the view can write back."""
    import dace.data as dd
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out
  integer, target  :: x
  integer, pointer :: tmp
  x = 7
  tmp => x
  tmp = 13
  out = x
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "scalar_view_alias")
    out_ref = int(mod.main())
    b = build_sdfg(src, tmp_path, name='main', entry='main')
    sdfg = b.build()
    # tmp is a View; x is a length-1 Array (not a Scalar).
    assert isinstance(sdfg.arrays["tmp"], dd.View)
    assert isinstance(sdfg.arrays["x"], dd.Array) and not isinstance(sdfg.arrays["x"], dd.View)
    assert tuple(sdfg.arrays["x"].shape) == (1, )
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    # The write through tmp must reach x: out = x = 13, NOT the initial 7.
    assert int(out[0]) == out_ref == 13


def test_dead_store_rebind_is_collapsed(tmp_path: Path):
    """Sequential dead-store rebinds (``ptr=>A; ptr=>B; use ptr``): only the LAST is
    observable, pass erases the earlier dead store -- mirrors ICON's aggregate-rebind
    lowering (multiple stores to one pointer slot)."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out
  integer, target  :: x, y
  integer, pointer :: tmp
  x = 1
  y = 2
  tmp => x
  tmp => y
  out = tmp
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    out = np.zeros(1, dtype=np.int32)
    sdfg(out=out)
    assert int(out[0]) == 2  # last rebind wins


def test_unsupported_interleaved_rebinds_raises(tmp_path: Path):
    """Loud-failure contract: a READ between two rebinds observes the earlier target, so
    collapsing to one rebind would lose that semantics -- must raise, not silently coalesce."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out(2)
  integer, target  :: x, y
  integer, pointer :: tmp
  x = 1
  y = 2
  tmp => x
  out(1) = tmp     ! read of tmp -> x
  tmp => y
  out(2) = tmp     ! read of tmp -> y
end subroutine main
"""
    with pytest.raises(RuntimeError):
        build_sdfg(src, tmp_path, name='main', entry='main').build()


def test_unsupported_interleaved_rebind_across_blocks_raises(tmp_path: Path):
    """Interleaved read sits inside an ``IF`` body (nested ``scf`` region), so an
    intra-block check would miss it. Dominance-based detection catches it: ``out(1)``'s
    read is not dominated by the final rebind, so the pass rejects loudly."""
    src = """
subroutine main(out, cond)
  implicit none
  integer, intent(out) :: out(2)
  logical, intent(in) :: cond
  integer, target :: x, y
  integer, pointer :: tmp
  x = 11
  y = 22
  tmp => x
  if (cond) then
    out(1) = tmp
  end if
  tmp => y
  out(2) = tmp
end subroutine main
"""
    with pytest.raises(RuntimeError):
        build_sdfg(src, tmp_path, name='main', entry='main').build()


def test_bounds_remap_lb_rebase_lowers_as_view(tmp_path: Path):
    """``w(0:n-1) => src(1:n)`` rebases the lower bound to 0: lowers as a View whose access
    offset is the captured lb (``w(0)`` aliases ``src(1)``). Was a loud-failure rejection
    before the index-rewrite could model the lb shift."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(3)
  real(8), target :: src(10)
  real(8), pointer :: w(:)
  integer :: i
  do i = 1, 10
    src(i) = real(i, 8)
  end do
  w(0:9) => src(1:10)
  out(1) = w(0)
  out(2) = w(5)
  out(3) = w(9)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "bounds_remap_lb0")
    ref = np.asarray(mod.main(), dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    build_sdfg(src, tmp_path, name='main', entry='main').build()(out=out)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, [1.0, 6.0, 10.0])


def test_bounds_remap_nonzero_lb_rebase_view(tmp_path: Path):
    """``w(5:14) => src(1:10)``: lb rebased to 5 (neither 0 nor Fortran default 1) --
    verifies the view's access offset is the *captured* lb, not a hardcoded default."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(2)
  real(8), target :: src(10)
  real(8), pointer :: w(:)
  integer :: i
  do i = 1, 10
    src(i) = real(i * 10, 8)
  end do
  w(5:14) => src(1:10)
  out(1) = w(5)
  out(2) = w(14)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "bounds_remap_lb5")
    ref = np.asarray(mod.main(), dtype=np.float64)
    out = np.zeros(2, dtype=np.float64)
    build_sdfg(src, tmp_path, name='main', entry='main').build()(out=out)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, [10.0, 100.0])


def test_bounds_remap_nonconstant_lb_still_rejected(tmp_path: Path):
    """Bounds remap with a NON-constant lower bound (runtime ``k``) has no fixed View
    access offset -- the pass still rejects."""
    src = """
subroutine main(n, k, src, res)
  implicit none
  integer, intent(in)        :: n, k
  real(8), intent(in), target :: src(n)
  real(8), intent(out)       :: res
  real(8), pointer           :: w(:)
  w(k : k + n - 1) => src(1:n)
  res = w(k)
end subroutine main
"""
    with pytest.raises(RuntimeError):
        build_sdfg(src, tmp_path, name='main', entry='main').build()


def test_pointer_rebind_to_array_slice(tmp_path: Path):
    """``w => store(1:n); res = w(2) + w(4)``: pointer rebound to a triplet section
    (``fir.rebox(hlfir.designate(...))``, not ``fir.embox(declare)``) -- ``traceToDecl``
    walks the chain back to the parent so ``w(i)`` reads ``parent(i)`` directly. No
    runtime ALLOCATED checks."""
    src = """
subroutine main(n, store, res)
  implicit none
  integer, intent(in)             :: n
  real, intent(in), target        :: store(n)
  real, intent(out)               :: res
  real, pointer                   :: w(:)
  w => store(1:n)
  res = w(2) + w(4)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    n = 5
    store = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    res = np.zeros(1, dtype=np.float32)
    sdfg(n=n, store=store, res=res)
    assert res[0] == 6.0  # store[1] + store[3] = 2 + 4


def test_pointer_to_struct_scalar_field(tmp_path: Path):
    """``tmp => s%a; tmp = 13``: pointer rebound onto a scalar struct field.
    flatten-structs runs first (``s%a`` -> flat ``s_a`` declare); rewrite-pointer-assigns
    traces the rebind through the box+embox chain to ``s_a``."""
    src = """
module lib
  implicit none
  type simple_type
    integer :: a
  end type simple_type
end module lib

subroutine main(d)
  use lib
  implicit none
  real, intent(inout) :: d(2)
  type(simple_type), target :: s
  integer, pointer :: tmp
  s%a = 0
  tmp => s%a
  tmp = 13
  d(1) = real(s%a)
  d(2) = real(tmp)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "ptr_to_struct_field")
    d_ref = np.zeros(2, order="F", dtype=np.float32)
    mod.main(d_ref)
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    d = np.zeros(2, dtype=np.float32)
    sdfg(d=d)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(d, [13.0, 13.0])
