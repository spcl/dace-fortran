"""Comprehensive numerical coverage for Fortran ``POINTER`` rebind shapes.

Each test rebinds a ``POINTER`` to some target shape, exercises it
(read-through, write-through, or whole-array copy), and compares the
SDFG result against a gfortran/f2py reference of the *same* source --
so a numeric regression surfaces instead of a builds-but-wrong silent
failure.

The matrix spans the axes that drive different bridge lowering paths:

  * **rank relationship** -- same-rank slice; rank-reducing (a 2-D
    column -> 1-D, a 2-D/3-D *section* flattened -> 1-D, a 3-D plane
    -> 2-D); rank-increasing (1-D -> 2-D / 3-D);
  * **bounds** -- literal vs variable section lower bound; the LHS
    explicit ``(1 : extent)`` bounds-remap;
  * **element type** -- ``real(8)``, ``integer``, ``complex(8)``;
  * **use** -- read-through, write-through (alias write-back), copy.

Shapes the bridge already lowers correctly act as regression guards.
The rank-*reducing* flatten of a multi-D **section** to a 1-D view
(``p(1:n*k) => arr(:, c0:c1)`` -- QE's ``prhoc`` FFT-feed trick) is the
open ``bounds-remap-view`` follow-up: the View descriptor is emitted
(``tests/bounds_remap_view/test_view_emission.py``) but the per-state
linking memlets that fold the flat 1-D index back to the parent's
multi-D coordinates are not wired yet, so a numerical *run* is wrong.
Those cases carry an ``xfail`` naming the follow-up; they flip to
``xpass`` (then the marker is removed) once the linking-memlet work
lands.

Every subroutine fills its own target deterministically so the test is
self-contained -- only the ``intent(out)`` result (and any scalar
``intent(in)`` index) crosses the boundary.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# --- Known-unsupported shape classes, each with a precise reason. The
#     matrix below pins exactly where the bridge's pointer-rebind support
#     ends; every xfail names the single facet that breaks so the marker
#     can be removed the moment that facet is wired.

#: A bare *whole-array* read of a plain (non-bounds-remap) rank-reducing
#: rebind: `p => a(:, j); out = p`.  Subscripted reads `p(i)` and writes
#: `p(i) = ...` forward correctly, but a bare `p` on an assignment RHS is
#: not rewritten to its target -- `out` reads the uninitialised pointer
#: storage.  Same root as QE's `res = p` whole-array copy (gate-H).
_PLAIN_REBIND_WHOLE_READ = ("plain rank-reducing rebind: a bare whole-array read `out = p` is not "
                            "forwarded to the target (only subscripted `p(i)` access is rewritten)")

#: Bounds-remap-view flatten read with a *variable* section lower bound:
#: `p(1:n*k) => a(:, c0:c1)`.  The constant-offset read works (see the
#: passing C/G cases); a runtime `c0` leaves the View's offset symbol
#: unresolved.  This is QE's exact `prhoc_d(...) => rhoc_d(:, off+1:...)`.
_VIEW_VAR_OFFSET = ("bounds-remap-view: a variable section lower bound leaves the View's "
                    "offset symbol unbound (constant-offset flatten reads already pass)")

#: Write-back through a flattened bounds-remap view: `p(1:n*k) =>
#: a(:, c0:c1); p(i) = ...` (or a callee mutating `p`).  The read-side
#: View linking memlet is wired; the write-side fold-back to the parent's
#: multi-D coordinates is not.
_VIEW_WRITEBACK = ("bounds-remap-view: write-back through a flattened view (direct or via "
                   "a callee) lacks the per-state linking memlet folding the flat index "
                   "to the parent's multi-D coordinates")


def _build(src: str, tmp: Path, entry: str = "_QPmain"):
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    return build_sdfg(src, sdfg_dir, name="m", entry=entry).build()


# ===========================================================================
# Family A -- same-rank rank-1 slice rebind over a plain local array.
# Sibling of the struct-member-slice tests but on a non-flattened target.
# ===========================================================================


def test_a_same_rank_slice_read(tmp_path: Path):
    """``p(:) => store(3:7)`` then read ``p(1)``, ``p(5)``."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(2)
  real(8), target :: store(10)
  real(8), pointer :: p(:)
  integer :: i
  do i = 1, 10
    store(i) = real(i, 8)
  end do
  p => store(3:7)
  out(1) = p(1)
  out(2) = p(5)
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "a_slice_read")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(2, dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, [3.0, 7.0])


def test_a_same_rank_slice_write_through(tmp_path: Path):
    """Write through ``p(:) => store(3:7)``; read back via ``store``."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(10)
  real(8), target :: store(10)
  real(8), pointer :: p(:)
  integer :: i
  do i = 1, 10
    store(i) = real(i, 8)
  end do
  p => store(3:7)
  do i = 1, 5
    p(i) = real(100 + i, 8)
  end do
  out = store
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "a_slice_write")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(10, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, [1, 2, 101, 102, 103, 104, 105, 8, 9, 10])


# ===========================================================================
# Family B -- rank-reducing single column (2-D -> 1-D) over a plain array.
# `p(:) => a(:, j)` keeps a unit stride; the section's fixed dim collapses.
# ===========================================================================


@pytest.mark.xfail(reason=_PLAIN_REBIND_WHOLE_READ, strict=False)
def test_b_column_rebind_read(tmp_path: Path):
    """``p(:) => a(:, 3)`` then copy the whole view out (`out = p`)."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(4)
  real(8), target :: a(4, 5)
  real(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      a(i, j) = real(10 * j + i, 8)
    end do
  end do
  p => a(:, 3)
  out = p
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "b_col_read")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(4, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, [31, 32, 33, 34])


def test_b_column_rebind_write_through(tmp_path: Path):
    """Write through ``p(:) => a(:, 3)``; the other columns stay put."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(20)
  real(8), target :: a(4, 5)
  real(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      a(i, j) = real(10 * j + i, 8)
    end do
  end do
  p => a(:, 3)
  do i = 1, 4
    p(i) = real(900 + i, 8)
  end do
  out = reshape(a, [20])
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "b_col_write")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(20, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    # Only column 3 (flat indices 8..11) changes to 901..904.
    assert list(out[8:12]) == [901, 902, 903, 904]


# ===========================================================================
# Family C -- rank-reducing flatten of a 2-D *section* -> 1-D view.
# This is the QE `prhoc(1:n*k) => rhoc(:, c0:c1)` gate-H pattern.
# Column-major flatten: p((j-1)*M + i) == a(i, c0-1+j).
# ===========================================================================


def test_c_section_flatten_read_literal_lb(tmp_path: Path):
    """``p(1:4*3) => a(:, 1:3)`` -- literal lower bound 1.  The
    constant-offset flatten *read* path is wired (regression guard)."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(12)
  real(8), target :: a(4, 5)
  real(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      a(i, j) = real(10 * j + i, 8)
    end do
  end do
  p(1 : 4 * 3) => a(:, 1:3)
  out = p
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "c_sec_read")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(12, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    # Column-major flatten of a(:, 1:3): cols 1,2,3.
    np.testing.assert_array_equal(out, [11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 34])


@pytest.mark.xfail(reason=_VIEW_WRITEBACK, strict=False)
def test_c_section_flatten_write_through(tmp_path: Path):
    """Write through the flattened section view; alias write-back must
    land at the column-major offsets inside ``a``."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(20)
  real(8), target :: a(4, 5)
  real(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      a(i, j) = real(10 * j + i, 8)
    end do
  end do
  p(1 : 4 * 3) => a(:, 1:3)
  do i = 1, 12
    p(i) = real(1000 + i, 8)
  end do
  out = reshape(a, [20])
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "c_sec_write")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(20, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    # First 12 flat slots (cols 1..3) overwritten; cols 4,5 unchanged.
    assert list(out[0:12]) == [1000 + i for i in range(1, 13)]
    assert list(out[12:20]) == [41, 42, 43, 44, 51, 52, 53, 54]


@pytest.mark.xfail(reason=_VIEW_VAR_OFFSET, strict=False)
def test_c_section_flatten_variable_lb(tmp_path: Path):
    """``p(1:12) => a(:, c0:c0+2)`` with a *variable* lower bound ``c0``
    -- the section's offset must stay symbolic (QE feeds a loop index)."""
    src = """
subroutine main(c0, out)
  implicit none
  integer, intent(in) :: c0
  real(8), intent(out) :: out(12)
  real(8), target :: a(4, 6)
  real(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 6
    do i = 1, 4
      a(i, j) = real(10 * j + i, 8)
    end do
  end do
  p(1 : 12) => a(:, c0 : c0 + 2)
  out = p
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "c_sec_varlb")
    ref = np.asarray(mod.main(2), dtype=np.float64)

    out = np.zeros(12, order="F", dtype=np.float64)
    _build(src, tmp_path)(c0=2, out=out)
    np.testing.assert_array_equal(out, ref)
    # c0=2 -> columns 2,3,4.
    np.testing.assert_array_equal(out, [21, 22, 23, 24, 31, 32, 33, 34, 41, 42, 43, 44])


def test_c_section_flatten_complex(tmp_path: Path):
    """``complex(8)`` constant-offset flatten read -- QE's exact element
    type for ``prhoc`` (regression guard for the wired read path)."""
    src = """
subroutine main(out)
  implicit none
  complex(8), intent(out) :: out(8)
  complex(8), target :: a(4, 5)
  complex(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      a(i, j) = cmplx(real(i, 8), real(j, 8), 8)
    end do
  end do
  p(1 : 4 * 2) => a(:, 1:2)
  out = p
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "c_sec_cplx")
    ref = np.asarray(mod.main(), dtype=np.complex128)

    out = np.zeros(8, order="F", dtype=np.complex128)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)


# ===========================================================================
# Family D -- rank-increasing 1-D -> 2-D (embox form), element-type spread.
# The real(8) base case lives in pointer_rank_changing_remap_e2e_test.py.
# ===========================================================================


def test_d_rank_increase_integer(tmp_path: Path):
    """``integer`` ``p(1:4,1:3) => arr1d`` -- column-major (1,M) strides."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out(12)
  integer, target :: arr1d(12)
  integer, pointer :: p(:, :)
  integer :: i, j
  do i = 1, 12
    arr1d(i) = i
  end do
  p(1:4, 1:3) => arr1d
  do j = 1, 3
    do i = 1, 4
      p(i, j) = 100 * j + i
    end do
  end do
  out = arr1d
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "d_rank_inc_int")
    ref = np.asarray(mod.main(), dtype=np.int32)

    out = np.zeros(12, order="F", dtype=np.int32)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    # Column-major: arr1d[(j-1)*4 + (i-1)] = 100*j + i.
    np.testing.assert_array_equal(out, [101, 102, 103, 104, 201, 202, 203, 204, 301, 302, 303, 304])


def test_d_rank_increase_complex(tmp_path: Path):
    """``complex(8)`` ``p(1:3,1:2) => arr1d``."""
    src = """
subroutine main(out)
  implicit none
  complex(8), intent(out) :: out(6)
  complex(8), target :: arr1d(6)
  complex(8), pointer :: p(:, :)
  integer :: i, j
  do i = 1, 6
    arr1d(i) = (0.0d0, 0.0d0)
  end do
  p(1:3, 1:2) => arr1d
  do j = 1, 2
    do i = 1, 3
      p(i, j) = cmplx(real(i, 8), real(j, 8), 8)
    end do
  end do
  out = arr1d
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "d_rank_inc_cplx")
    ref = np.asarray(mod.main(), dtype=np.complex128)

    out = np.zeros(6, order="F", dtype=np.complex128)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)


# ===========================================================================
# Family E -- scalar pointer to an element, *variable* index.
# Literal `p => arr(3)` is in pointer_flat_subset_test.py; add a runtime idx.
# ===========================================================================


def test_e_scalar_element_variable_index(tmp_path: Path):
    """``p => arr(idx)`` with ``idx`` a runtime scalar."""
    src = """
subroutine main(idx, out)
  implicit none
  integer, intent(in) :: idx
  integer, intent(out) :: out
  integer, target :: arr(5)
  integer, pointer :: p
  integer :: i
  do i = 1, 5
    arr(i) = i * 10
  end do
  p => arr(idx)
  out = p + 1
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "e_scalar_varidx")
    ref = np.asarray(mod.main(4), dtype=np.int32)

    out = np.zeros(1, dtype=np.int32)
    _build(src, tmp_path)(idx=4, out=out)
    np.testing.assert_array_equal(out, ref)
    assert out[0] == 41  # arr(4) = 40; +1.


def test_e_scalar_element_write_through(tmp_path: Path):
    """Write through a scalar element pointer; read back via the host."""
    src = """
subroutine main(out)
  implicit none
  integer, intent(out) :: out(5)
  integer, target :: arr(5)
  integer, pointer :: p
  integer :: i
  do i = 1, 5
    arr(i) = i
  end do
  p => arr(3)
  p = 777
  out = arr
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "e_scalar_write")
    ref = np.asarray(mod.main(), dtype=np.int32)

    out = np.zeros(5, order="F", dtype=np.int32)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    np.testing.assert_array_equal(out, [1, 2, 777, 4, 5])


# ===========================================================================
# Family F -- rank-reducing 3-D plane -> 2-D view (no flatten, stride kept).
# `p(:,:) => a(:,:,k)` collapses the last fixed dim but keeps the rank-2
# section contiguous; distinct from the Family C flatten.
# ===========================================================================


@pytest.mark.xfail(reason=_PLAIN_REBIND_WHOLE_READ, strict=False)
def test_f_3d_plane_to_2d_read(tmp_path: Path):
    """``p(:,:) => a(:,:,2)`` then copy the whole plane out (`reshape(p)`)."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(6)
  real(8), target :: a(2, 3, 4)
  real(8), pointer :: p(:, :)
  integer :: i, j, k
  do k = 1, 4
    do j = 1, 3
      do i = 1, 2
        a(i, j, k) = real(100 * k + 10 * j + i, 8)
      end do
    end do
  end do
  p => a(:, :, 2)
  out = reshape(p, [6])
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "f_plane_read")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(6, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
    # Plane k=2, column-major over (i=1..2, j=1..3): 211,212,221,222,231,232.
    np.testing.assert_array_equal(out, [211, 212, 221, 222, 231, 232])


# ===========================================================================
# Family G -- rank-reducing flatten of a 3-D section -> 1-D view.
# `p(1:M*N*kk) => a(:,:,k0:k1)` -- the Family C flatten one rank up.
# ===========================================================================


def test_g_3d_section_flatten_read(tmp_path: Path):
    """``p(1:2*3*2) => a(:,:,1:2)`` -- flatten the first two planes
    (constant-offset read; regression guard)."""
    src = """
subroutine main(out)
  implicit none
  real(8), intent(out) :: out(12)
  real(8), target :: a(2, 3, 4)
  real(8), pointer :: p(:)
  integer :: i, j, k
  do k = 1, 4
    do j = 1, 3
      do i = 1, 2
        a(i, j, k) = real(100 * k + 10 * j + i, 8)
      end do
    end do
  end do
  p(1 : 2 * 3 * 2) => a(:, :, 1:2)
  out = p
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "g_3d_flatten")
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(12, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)


# ===========================================================================
# Family H -- pass the rebound pointer to a callee (QE's `CALL fwfft(...,
# prhoc, ...)`).  Pins that a flattened-section view survives as a call
# argument, not just a local read/write.
# ===========================================================================


@pytest.mark.xfail(reason=_VIEW_WRITEBACK, strict=False)
def test_h_flattened_view_passed_to_callee(tmp_path: Path):
    """``p(1:12) => a(:,1:3); call scale(p, 12)`` -- the callee doubles
    every element in place; the writes must land back in ``a``."""
    src = """
subroutine scale(v, n)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: v(n)
  integer :: i
  do i = 1, n
    v(i) = v(i) * 2.0d0
  end do
end subroutine scale

subroutine main(out)
  implicit none
  real(8), intent(out) :: out(20)
  real(8), target :: a(4, 5)
  real(8), pointer :: p(:)
  integer :: i, j
  do j = 1, 5
    do i = 1, 4
      a(i, j) = real(10 * j + i, 8)
    end do
  end do
  p(1 : 4 * 3) => a(:, 1:3)
  call scale(p, 12)
  out = reshape(a, [20])
end subroutine main
"""
    mod = f2py_compile(src, tmp_path / "ref", "h_call", only=("main", ))
    ref = np.asarray(mod.main(), dtype=np.float64)

    out = np.zeros(20, order="F", dtype=np.float64)
    _build(src, tmp_path)(out=out)
    np.testing.assert_array_equal(out, ref)
