"""Array shapes whose extent is a non-trivial value -- the "value in a symbolic context is a symbol" rule applied to allocation/declaration size expressions.

Extent from a constant-indexed array element (``buf(dims(1))``), an arithmetic expression over one, multi-dim allocation, an automatic array, or a function result must lift to a position symbol (``__sym_dims_1``, read once on an interstate edge) rather than promote the whole source array -- promoting the array would collide it with its own data descriptor.

Each case builds an SDFG and an f2py reference from the same source, writes back the realised size, and compares numerically so a wrong extent shows up as a wrong value.  Assumes the element-extent source array is intent(in) -- reading it at SDFG entry is then equivalent to reading it at the allocation point.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _run(tmp_path, src, dims, *, entry="probe_mod::probe", shape_of=None):
    """Build src through the bridge and through f2py, run both on the same dims table, return (sdfg_out, ref_out).  out(n) must match len(dims).

    ``shape_of``: array name to also return its SDFG descriptor shape, so a test can assert the extent expression directly (e.g. a genuine MAX survives as Max(...)).
    """
    n = len(dims)
    dims = np.asfortranarray(np.asarray(dims, dtype=np.int32))
    out_sdfg = np.zeros(n, dtype=np.float64)
    out_ref = np.zeros(n, dtype=np.float64)

    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry=entry).build()
    if shape_of is not None:
        shape_str = str(sdfg.arrays[shape_of].shape)
    sdfg(n=np.int32(n), dims=dims, out=out_sdfg)

    # unique f2py module name per test -- a shared name would be served from Python's import cache, handing later tests the first test's compiled kernel.
    mod = f2py_compile(src, tmp_path / "ref", f"size_ref_{tmp_path.name}")
    mod.probe_mod.probe(dims, out_ref)
    if shape_of is not None:
        return out_sdfg, out_ref, shape_str
    return out_sdfg, out_ref


def _alloc_iter_by(size_expr: str, iter_idx: str) -> str:
    """Allocate buf(size_expr) but iterate/read via the bare element iter_idx (<= realised size) -- exercises a MAX/MIN extent without also routing MAX/MIN through the index/bound path."""
    return f"""
module probe_mod
contains
subroutine probe(n, dims, out)
  implicit none
  integer, intent(in) :: n, dims(n)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: buf(:)
  integer :: i
  allocate(buf({size_expr}))
  do i = 1, {iter_idx}
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf({iter_idx})
  out(2) = buf(1)
  deallocate(buf)
end subroutine probe
end module probe_mod
"""


# out(1) = buf(<size>) makes the realised extent observable; out(2) is the buffer head. <size> is the only thing varying across the 1-D allocatable cases.
def _alloc_1d(size_expr: str) -> str:
    return f"""
module probe_mod
contains
subroutine probe(n, dims, out)
  implicit none
  integer, intent(in) :: n, dims(n)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: buf(:)
  integer :: i
  allocate(buf({size_expr}))
  do i = 1, {size_expr}
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf({size_expr})
  out(2) = buf(1)
  deallocate(buf)
end subroutine probe
end module probe_mod
"""


def test_size_bare_array_element(tmp_path):
    """``allocate(buf(dims(1)))`` -- the bare element extent."""
    s, r = _run(tmp_path, _alloc_1d("dims(1)"), [5, 3, 7])
    assert s[0] == 2 * 5 and s[1] == 2
    np.testing.assert_array_equal(s, r)


def test_size_arith_on_element(tmp_path):
    """``allocate(buf(2*dims(1) + 3))`` -- arithmetic around the element."""
    s, r = _run(tmp_path, _alloc_1d("2*dims(1) + 3"), [5, 3, 7])
    assert s[0] == 2 * (2 * 5 + 3)
    np.testing.assert_array_equal(s, r)


def test_size_sum_of_two_elements(tmp_path):
    """``allocate(buf(dims(1) + dims(2)))`` -- two position symbols."""
    s, r = _run(tmp_path, _alloc_1d("dims(1) + dims(2)"), [5, 4, 7])
    assert s[0] == 2 * (5 + 4)
    np.testing.assert_array_equal(s, r)


def test_size_scalar_hop(tmp_path):
    """k = dims(1); allocate(buf(k)) -- element via a scalar (already a symbol; verifies the hop stays consistent)."""
    src = """
module probe_mod
contains
subroutine probe(n, dims, out)
  implicit none
  integer, intent(in) :: n, dims(n)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: buf(:)
  integer :: i, k
  k = dims(1)
  allocate(buf(k))
  do i = 1, k
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf(k)
  deallocate(buf)
end subroutine probe
end module probe_mod
"""
    s, r = _run(tmp_path, src, [6, 3])
    assert s[0] == 2 * 6
    np.testing.assert_array_equal(s, r)


def test_size_multidim_alloc(tmp_path):
    """``allocate(mat(dims(1), dims(2)))`` -- one position symbol per dim."""
    src = """
module probe_mod
contains
subroutine probe(n, dims, out)
  implicit none
  integer, intent(in) :: n, dims(n)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: mat(:,:)
  integer :: i, j
  allocate(mat(dims(1), dims(2)))
  do j = 1, dims(2)
    do i = 1, dims(1)
      mat(i, j) = real(i + 10*j, 8)
    end do
  end do
  out(1) = mat(dims(1), dims(2))
  out(2) = mat(1, 1)
  deallocate(mat)
end subroutine probe
end module probe_mod
"""
    s, r = _run(tmp_path, src, [4, 3])
    assert s[0] == 4 + 10 * 3 and s[1] == 1 + 10
    np.testing.assert_array_equal(s, r)


def test_size_automatic_array(tmp_path):
    """real(8) :: tmp(dims(1)) -- automatic (explicit-shape local) array sized by an element; no ALLOCATE."""
    src = """
module probe_mod
contains
subroutine probe(n, dims, out)
  implicit none
  integer, intent(in) :: n, dims(n)
  real(8), intent(inout) :: out(n)
  real(8) :: tmp(dims(1))
  integer :: i
  do i = 1, dims(1)
    tmp(i) = real(2*i, 8)
  end do
  out(1) = tmp(dims(1))
  out(2) = tmp(1)
end subroutine probe
end module probe_mod
"""
    s, r = _run(tmp_path, src, [6, 3])
    assert s[0] == 2 * 6 and s[1] == 2
    np.testing.assert_array_equal(s, r)


def _run_scalar_args(tmp_path, src, kwargs, ref_args, *, nout=2):
    """Variant of :func:`_run` for kernels whose only inputs are plain scalars -- shape comes from a function of those scalars, not a dims table."""
    out_sdfg = np.zeros(nout, dtype=np.float64)
    out_ref = np.zeros(nout, dtype=np.float64)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry="probe_mod::probe").build()
    sdfg(out=out_sdfg, **kwargs)
    mod = f2py_compile(src, tmp_path / "ref", f"size_ref_{tmp_path.name}")
    mod.probe_mod.probe(*ref_args, out_ref)
    return out_sdfg, out_ref


def test_size_function_of_scalars(tmp_path):
    """allocate(buf(fsz(a, b))) -- the function result is the symbol; its scalar inputs stay plain scalars."""
    src = """
module probe_mod
contains
subroutine probe(a, b, out)
  implicit none
  integer, intent(in) :: a, b
  real(8), intent(inout) :: out(2)
  real(8), allocatable :: buf(:)
  integer :: i
  allocate(buf(fsz(a, b)))
  do i = 1, fsz(a, b)
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf(fsz(a, b))
  out(2) = buf(1)
  deallocate(buf)
contains
  pure integer function fsz(x, y) result(r)
    integer, intent(in) :: x, y
    r = x*y + 1
  end function fsz
end subroutine probe
end module probe_mod
"""
    s, r = _run_scalar_args(tmp_path, src, {"a": np.int32(3), "b": np.int32(4)}, (3, 4))
    assert s[0] == 2 * (3 * 4 + 1) and s[1] == 2
    np.testing.assert_array_equal(s, r)


def test_size_function_no_input(tmp_path):
    """``allocate(buf(fc()))`` -- size from a no-argument function."""
    src = """
module probe_mod
contains
subroutine probe(out)
  implicit none
  real(8), intent(inout) :: out(2)
  real(8), allocatable :: buf(:)
  integer :: i
  allocate(buf(fc()))
  do i = 1, fc()
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf(fc())
  out(2) = buf(1)
  deallocate(buf)
contains
  pure integer function fc() result(r)
    r = 7
  end function fc
end subroutine probe
end module probe_mod
"""
    s, r = _run_scalar_args(tmp_path, src, {}, ())
    assert s[0] == 2 * 7 and s[1] == 2
    np.testing.assert_array_equal(s, r)


def test_size_function_of_array_element(tmp_path):
    """allocate(buf(fsz(dims(1), dims(2)))) -- function fed array elements, result used as both size and index.

    r = dims(1)*dims(2)+1 is a scalar promoted to a symbol; lifting its full compound RHS onto the interstate edge (not just the first read) keeps every term.
    """
    src = """
module probe_mod
contains
subroutine probe(n, dims, out)
  implicit none
  integer, intent(in) :: n, dims(n)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: buf(:)
  integer :: i
  allocate(buf(fsz(dims(1), dims(2))))
  do i = 1, fsz(dims(1), dims(2))
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf(fsz(dims(1), dims(2)))
  deallocate(buf)
contains
  pure integer function fsz(x, y) result(r)
    integer, intent(in) :: x, y
    r = x*y + 1
  end function fsz
end subroutine probe
end module probe_mod
"""
    s, r = _run(tmp_path, src, [3, 4])
    assert s[0] == 2 * (3 * 4 + 1)  # buf(13) = 26
    np.testing.assert_array_equal(s, r)


# clampdim checks: Flang's non-negativity clamp max(ext,0) is dropped, but a genuine two-operand MAX/MIN extent must survive.


def test_size_max_element_const(tmp_path):
    """allocate(buf(max(dims(1), 1))) -- genuine MAX(element, const); realised extent is Max(1, dims(1)), not the dropped clamp."""
    s, r, shape = _run(tmp_path, _alloc_iter_by("max(dims(1), 1)", "dims(1)"), [5, 3], shape_of="buf")
    assert "Max" in shape, shape
    assert s[0] == 2 * 5 and s[1] == 2
    np.testing.assert_array_equal(s, r)


def test_size_max_two_elements(tmp_path):
    """allocate(buf(max(dims(1), dims(2)))) -- MAX of two elements; wrong handling (dropping the max / false arm) would under-size the buffer and make buf(dims(1)) out of bounds."""
    s, r, shape = _run(tmp_path, _alloc_iter_by("max(dims(1), dims(2))", "dims(1)"), [5, 3], shape_of="buf")
    assert "Max" in shape, shape
    assert s[0] == 2 * 5
    np.testing.assert_array_equal(s, r)


def test_size_min_two_elements(tmp_path):
    """allocate(buf(min(dims(1), dims(2)))) -- MIN of two elements; verified through the descriptor (Min(...)) and a read at the smaller extent."""
    s, r, shape = _run(tmp_path, _alloc_iter_by("min(dims(1), dims(2))", "dims(2)"), [5, 3], shape_of="buf")
    assert "Min" in shape, shape
    assert s[0] == 2 * 3
    np.testing.assert_array_equal(s, r)


def test_size_multidim_element_via_scalar(tmp_path):
    """dim = shp(i,j,k); allocate(buf(dim)) -- multi-dim element via a scalar hop; the read RHS shp(ii,jj,kk) lifts whole onto the interstate edge, so this needs no multi-dim position-symbol machinery."""
    src = """
module probe_mod
contains
subroutine probe(ii, jj, kk, shp, out)
  implicit none
  integer, intent(in) :: ii, jj, kk
  integer, intent(in) :: shp(2,2,2)
  real(8), intent(inout) :: out(2)
  real(8), allocatable :: buf(:)
  integer :: i, dim
  dim = shp(ii, jj, kk)
  allocate(buf(dim))
  do i = 1, dim
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf(dim)
  out(2) = buf(1)
  deallocate(buf)
end subroutine probe
end module probe_mod
"""
    shp = np.asfortranarray(np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2))
    out_s = np.zeros(2)
    out_r = np.zeros(2)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry="probe_mod::probe").build()
    sdfg(ii=np.int32(1), jj=np.int32(2), kk=np.int32(1), shp=shp, out=out_s)
    assert out_s[0] == 2 * 3  # shp(1,2,1) == 3 (column-major), buf(3) = 6
    mod = f2py_compile(src, tmp_path / "ref", f"size_ref_{tmp_path.name}")
    mod.probe_mod.probe(1, 2, 1, shp, out_r)
    np.testing.assert_array_equal(out_s, out_r)


def test_size_multidim_element_inline(tmp_path):
    """allocate(buf(shp(1,2,1))) -- multi-dim element inline as the extent (no intermediate scalar); lifts to a multi-dim position symbol __sym_shp_1_2_1 (read once at entry), so the shape stays symbolic."""
    src = """
module probe_mod
contains
subroutine probe(n, shp, out)
  implicit none
  integer, intent(in) :: n, shp(2,2,2)
  real(8), intent(inout) :: out(n)
  real(8), allocatable :: buf(:)
  integer :: i
  allocate(buf(shp(1,2,1)))
  do i = 1, shp(1,2,1)
    buf(i) = real(2*i, 8)
  end do
  out(1) = buf(shp(1,2,1))
  deallocate(buf)
end subroutine probe
end module probe_mod
"""
    n = 4
    shp = np.asfortranarray(np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2))
    out_s = np.zeros(n)
    out_r = np.zeros(n)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry="probe_mod::probe").build()
    assert "__sym_shp_1_2_1" in sdfg.symbols
    sdfg(n=np.int32(n), shp=shp, out=out_s)
    assert out_s[0] == 2 * 3  # shp(1,2,1) == 3 (column-major), buf(3) = 6
    mod = f2py_compile(src, tmp_path / "ref", f"size_ref_{tmp_path.name}")
    mod.probe_mod.probe(shp, out_r)
    np.testing.assert_array_equal(out_s, out_r)
