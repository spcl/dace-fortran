"""Array shapes whose extent is a non-trivial value -- the "value in a
symbolic context is a symbol" rule applied to allocation / declaration
size expressions.

A DaCe array shape is symbolic, so any value reaching an extent must be
a symbol.  A bare scalar already is one; the interesting cases are when
the extent is a *constant-indexed array element* (``buf(dims(1))``), an
arithmetic expression over one, a multi-dimensional allocation, an
automatic (explicit-shape local) array, or a function result.  Each must
lift the element read to a position symbol (``__sym_dims_1``, read once
on an interstate edge) rather than promote the whole source array -- the
latter collides the array with its own data descriptor.

Every case builds an SDFG and an f2py reference from the same source and
checks the output numerically; the kernels write back the realised size
so a wrong extent shows up as a wrong value, not just a build error.

Assumes the source array of an element extent is read-only (``intent(in)``
dimension tables, the universal shape-source shape); reading it at SDFG
entry is then equivalent to reading it at the allocation point.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _run(tmp_path, src, dims, *, entry="_QPprobe"):
    """Build ``src`` through the bridge and through f2py, run both on the
    same ``dims`` table, and return ``(sdfg_out, ref_out)``.  ``out`` is
    dimensioned ``out(n)`` in the kernel, so it must match ``len(dims)``."""
    n = len(dims)
    dims = np.asfortranarray(np.asarray(dims, dtype=np.int32))
    out_sdfg = np.zeros(n, dtype=np.float64)
    out_ref = np.zeros(n, dtype=np.float64)

    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry=entry).build()
    sdfg(n=np.int32(n), dims=dims, out=out_sdfg)

    # Unique f2py module name per test: a shared name would be served from
    # Python's import cache, handing every later test the first test's
    # compiled kernel as its reference.
    mod = f2py_compile(src, tmp_path / "ref", f"size_ref_{tmp_path.name}")
    mod.probe(dims, out_ref)
    return out_sdfg, out_ref


# ``out(1)`` is written ``buf(<size>) = 2*<size>`` so the realised extent
# is observable; ``out(2)`` is the buffer head.  ``<size>`` placeholder is
# the only thing that varies between the 1-D allocatable cases.
def _alloc_1d(size_expr: str) -> str:
    return f"""
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
    """``k = dims(1); allocate(buf(k))`` -- element via a scalar (the
    scalar is already a symbol; verifies the hop stays consistent)."""
    src = """
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
"""
    s, r = _run(tmp_path, src, [6, 3])
    assert s[0] == 2 * 6
    np.testing.assert_array_equal(s, r)


def test_size_multidim_alloc(tmp_path):
    """``allocate(mat(dims(1), dims(2)))`` -- one position symbol per dim."""
    src = """
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
"""
    s, r = _run(tmp_path, src, [4, 3])
    assert s[0] == 4 + 10 * 3 and s[1] == 1 + 10
    np.testing.assert_array_equal(s, r)


def test_size_automatic_array(tmp_path):
    """``real(8) :: tmp(dims(1))`` -- automatic (explicit-shape local)
    array sized by an element; no ALLOCATE."""
    src = """
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
"""
    s, r = _run(tmp_path, src, [6, 3])
    assert s[0] == 2 * 6 and s[1] == 2
    np.testing.assert_array_equal(s, r)


def _run_scalar_args(tmp_path, src, kwargs, ref_args, *, nout=2):
    """Variant of :func:`_run` for kernels whose only inputs are plain
    scalars (``probe(a, b, out)`` / ``probe(out)``) -- the shape comes
    from a function of those scalars, not from a ``dims`` table."""
    out_sdfg = np.zeros(nout, dtype=np.float64)
    out_ref = np.zeros(nout, dtype=np.float64)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry="_QPprobe").build()
    sdfg(out=out_sdfg, **kwargs)
    mod = f2py_compile(src, tmp_path / "ref", f"size_ref_{tmp_path.name}")
    mod.probe(*ref_args, out_ref)
    return out_sdfg, out_ref


def test_size_function_of_scalars(tmp_path):
    """``allocate(buf(fsz(a, b)))`` -- the function result is the symbol;
    its scalar inputs stay plain scalars (input is a scalar, output is a
    symbol)."""
    src = """
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
"""
    s, r = _run_scalar_args(tmp_path, src,
                            {"a": np.int32(3), "b": np.int32(4)}, (3, 4))
    assert s[0] == 2 * (3 * 4 + 1) and s[1] == 2
    np.testing.assert_array_equal(s, r)


def test_size_function_no_input(tmp_path):
    """``allocate(buf(fc()))`` -- size from a no-argument function."""
    src = """
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
"""
    s, r = _run_scalar_args(tmp_path, src, {}, ())
    assert s[0] == 2 * 7 and s[1] == 2
    np.testing.assert_array_equal(s, r)


@pytest.mark.xfail(strict=True, reason="function fed an array element "
                   "(foo(a(1))): the inlined call result used as size / index "
                   "drops terms (reads buf(dims(1)) instead of buf(dims(1)*"
                   "dims(2)+1)) -- a function-inlining issue distinct from the "
                   "shape-symbol lifting, tracked separately")
def test_size_function_of_array_element(tmp_path):
    """``allocate(buf(fsz(dims(1), dims(2))))`` -- function fed array
    elements (the ``foo(a(0))`` shape)."""
    src = """
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
"""
    s, r = _run(tmp_path, src, [3, 4])
    np.testing.assert_array_equal(s, r)


@pytest.mark.xfail(strict=True, reason="extent wrapped in MAX(elem, k): the "
                   "select's false-arm is k != 0, so traceExtentExpr's "
                   "max(ext,0)-clamp peel doesn't fire and the element read "
                   "still resolves to the whole array name")
def test_size_intrinsic_max_element(tmp_path):
    """``allocate(buf(max(dims(1), 1)))`` -- element inside an intrinsic."""
    s, r = _run(tmp_path, _alloc_1d("max(dims(1), 1)"), [5, 3])
    np.testing.assert_array_equal(s, r)


@pytest.mark.xfail(strict=True, reason="multi-dimensional element extent "
                   "shp(i,j,k): constIndexedElementLoad only lifts a single "
                   "1-D index; a multi-dim position symbol is not yet minted")
def test_size_multidim_element(tmp_path):
    """``allocate(buf(shp(1,2,1)))`` -- size from a multi-dim element."""
    src = """
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
"""
    n = 4
    shp = np.asfortranarray(np.arange(1, 9, dtype=np.int32).reshape(2, 2, 2))
    out_s = np.zeros(n); out_r = np.zeros(n)
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="probe", entry="_QPprobe").build()
    sdfg(n=np.int32(n), shp=shp, out=out_s)
    mod = f2py_compile(src, tmp_path / "ref", "size_ref_md")
    mod.probe(shp, out_r)
    np.testing.assert_array_equal(out_s, out_r)
