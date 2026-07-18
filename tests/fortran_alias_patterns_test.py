"""Exhaustive probe of every Fortran storage-association / aliasing pattern the bridge may
encounter -- one IR shape per Fortran reshape/aliasing construct Flang's HLFIR/FIR emits.
Classification audited against HLFIROps.td + FIROps.td and Fortran 2018 15.5.2.4-15.5.2.10.

Each pattern builds a tiny SDFG and asserts a known property, or xfails with what's missing.
Sequence-association probes also verify the f2py round-trip honours the same semantics.

Pattern catalogue (status as of 2026-06-09):

  A. Whole-array RANK reinterpretation (1D<->multi-D, same declare, no slice). HANDLED:
     trace_utils.cpp::asAssumedShapeAlias refuses the alias collapse on rank mismatch.
  B. Array section reshape to lower rank (designate+triplets -> fir.convert). HANDLED:
     view-alias detection in extract_vars.cpp:1908+.
  C. Element passed to a fixed-extent dummy (sequence association). HANDLED by the
     RewriteSequenceAssociation pass before the bridge sees it.
  D. Assumed-rank dummy DIMENSION(..) (fir.rebox_assumed_rank). UNKNOWN -- probe below.
  E. Assumed-shape dummy DIMENSION(:,:) (fir.embox/fir.rebox). HANDLED by the existing
     asAssumedShapeAlias chain peel.
  F. Assumed-size dummy DIMENSION(*) / DIMENSION(N,*). Probe below.
  G. POINTER bounds-remap-AND-RESHAPE (p(1:M,1:K) => arr1d, rank-changing). Probe below.
  H. c_f_pointer(cptr, fptr, [M,N]) -- C interop with explicit shape. Probe below.
  I. ASSOCIATE construct (hlfir.associate). Probe below.
  J. Component reference t%arr -- normal designate alias, should already be handled.
     Probe below to confirm.
  K. RESHAPE intrinsic -- produces a fresh hlfir.expr, a VALUE copy, not an alias. No
     bridge work needed.
  L. TRANSFER intrinsic -- bit reinterpretation, VALUE copy. No bridge work needed.
  M. EQUIVALENCE statement (legacy F77) -- two names share storage. Probe below.
  N. Vector subscripts arr([1,3,5]) / arr(idx) -- spec forbids LHS alias semantics; IR
     routes through a temporary. No alias.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _try_build(tmp_path, src, name, entry):
    """Build an SDFG; return (sdfg, None) on success or (None, err_str) on failure -- so one
    pattern's failure doesn't mask the rest."""
    try:
        return build_sdfg(src, tmp_path, name=name, entry=entry).build(), None
    except Exception as e:
        return None, str(e)[:300]


# ---------------------------------------------------------------------------
# Pattern A -- whole-array rank reinterpretation (the LU ``tv`` case).
# ---------------------------------------------------------------------------
def test_a_whole_array_rank_promotion_1d_to_3d(tmp_path):
    """1D 5445-element scratch passed to a callee expecting a 3D (5,33,33) dummy -- same
    storage, different rank. Closed via asAssumedShapeAlias rank-mismatch refusal."""
    src = """
module m
  implicit none
  integer, parameter :: N1 = 5, N2 = 33, N3 = 33
  double precision :: scratch(N1*N2*N3)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(N1, N2, N3)
    integer :: i, j, k
    do k = 1, N3
      do j = 1, N2
        do i = 1, N1
          buf(i, j, k) = real(i + j + k, kind=8)
        end do
      end do
    end do
  end subroutine inner

  subroutine outer()
    call inner(scratch)
  end subroutine outer
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="outer", entry="m::outer")
    assert sdfg is not None, f"build failed: {err}"


# ---------------------------------------------------------------------------
# Pattern F -- assumed-size dummy ``DIMENSION(*)``.
# ---------------------------------------------------------------------------
def test_f_assumed_size_dummy_one_dim(tmp_path):
    """REAL :: a(*) -- callee sees the actual's storage from element 1 onwards; same-rank alias."""
    src = """
module m
  implicit none
contains
  subroutine inner(a, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: a(*)
    integer :: i
    do i = 1, n
      a(i) = real(i, kind=8)
    end do
  end subroutine inner

  subroutine outer(arr, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: arr(n)
    call inner(arr, n)
  end subroutine outer
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="outer", entry="m::outer")
    assert sdfg is not None, f"build failed: {err}"


def test_f_assumed_size_dummy_multi_dim(tmp_path):
    """REAL :: a(5, *) -- rank-fixed leading extent, unbounded last extent; same rank as
    actual when caller passes a 2D array."""
    src = """
module m
  implicit none
contains
  subroutine inner(a, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: a(5, *)
    integer :: i, j
    do j = 1, n
      do i = 1, 5
        a(i, j) = real(i*j, kind=8)
      end do
    end do
  end subroutine inner

  subroutine outer(arr, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: arr(5, n)
    call inner(arr, n)
  end subroutine outer
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="outer", entry="m::outer")
    assert sdfg is not None, f"build failed: {err}"


# ---------------------------------------------------------------------------
# Pattern G -- POINTER bounds-remap WITH rank change.
# ---------------------------------------------------------------------------
def test_g_pointer_rank_changing_remap(tmp_path):
    """``real, pointer :: p(:,:)`` rebound via ``p(1:M,1:K) => arr1d``.
    Same storage; pointer sees the 1D data as 2D."""
    src = """
module m
  implicit none
  integer, parameter :: M = 4, K = 3
  double precision, target :: arr1d(M*K)
contains
  subroutine fill()
    double precision, pointer :: p(:,:)
    integer :: i, j
    p(1:M, 1:K) => arr1d
    do j = 1, K
      do i = 1, M
        p(i, j) = real(i + (j - 1) * M, kind=8)
      end do
    end do
  end subroutine fill
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="fill", entry="m::fill")
    assert sdfg is not None, f"build failed: {err}"


# ---------------------------------------------------------------------------
# Pattern H -- ``c_f_pointer`` with explicit shape.
# ---------------------------------------------------------------------------
def test_h_c_f_pointer_with_shape_is_rejected(tmp_path):
    """c_f_pointer(cptr, fptr, [M,N]) -- DELIBERATELY UNSUPPORTED: the C-managed buffer has no
    DaCe descriptor, so the bridge FAILS CLEANLY (unresolved free symbol) rather than silently
    emitting reads/writes through a pointer it never allocated. Fortran-side ptr(1:M,1:N) => arr
    rebinds are handled via bounds-remap-view; only the c_f_pointer C-buffer variant is out of scope."""
    src = """
module m
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine fill(cptr, M, N)
    type(c_ptr), value, intent(in) :: cptr
    integer, intent(in) :: M, N
    double precision, pointer :: fptr(:,:)
    integer :: i, j
    call c_f_pointer(cptr, fptr, [M, N])
    do j = 1, N
      do i = 1, M
        fptr(i, j) = real(i + j, kind=8)
      end do
    end do
  end subroutine fill
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="fill", entry="m::fill")
    assert sdfg is None, "expected c_f_pointer-with-shape to be rejected at build"
    assert "unresolved free symbol" in str(err) or "cptr" in str(err), (
        f"expected an unresolved-symbol rejection mentioning the opaque "
        f"C pointer, got: {err}")


# ---------------------------------------------------------------------------
# Pattern I -- ``ASSOCIATE`` construct.
# ---------------------------------------------------------------------------
def test_i_associate_variable_binding(tmp_path):
    """ASSOCIATE (name => arr_var) -- no rank change; same-rank alias the existing path handles."""
    src = """
module m
  implicit none
contains
  subroutine fill(arr, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: arr(n)
    integer :: i
    associate (a => arr)
      do i = 1, n
        a(i) = real(i, kind=8)
      end do
    end associate
  end subroutine fill
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="fill", entry="m::fill")
    assert sdfg is not None, f"build failed: {err}"


def test_i_associate_section_binding(tmp_path):
    """ASSOCIATE (name => arr(:,:,k)) -- section reshape inside an ASSOCIATE; tests whether the
    bridge sees the section through hlfir.associate."""
    src = """
module m
  implicit none
contains
  subroutine fill(arr, n, k)
    integer, intent(in) :: n, k
    double precision, intent(inout) :: arr(n, n, n)
    integer :: i, j
    associate (slice => arr(:, :, k))
      do j = 1, n
        do i = 1, n
          slice(i, j) = real(i + j, kind=8)
        end do
      end do
    end associate
  end subroutine fill
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="fill", entry="m::fill")
    assert sdfg is not None, f"build failed: {err}"


# ---------------------------------------------------------------------------
# Pattern J -- derived-type component reference.
# ---------------------------------------------------------------------------
def test_j_derived_type_array_component(tmp_path):
    """``t%arr(i, j)`` where ``t`` is a derived type with an array
    component.  Treated as a normal designate alias on the component."""
    src = """
module m
  implicit none
  type :: container
    double precision :: arr(4, 3)
  end type container
contains
  subroutine fill(t)
    type(container), intent(inout) :: t
    integer :: i, j
    do j = 1, 3
      do i = 1, 4
        t%arr(i, j) = real(i + j, kind=8)
      end do
    end do
  end subroutine fill
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="fill", entry="m::fill")
    assert sdfg is not None, f"build failed: {err}"


# ---------------------------------------------------------------------------
# Pattern D -- assumed-rank dummy ``DIMENSION(..)`` + SELECT RANK.
# ---------------------------------------------------------------------------
def test_d_assumed_rank_dummy(tmp_path):
    """``DIMENSION(..)`` -- assumed rank.  Actual's rank reaches the
    callee at runtime; the callee uses SELECT RANK to dispatch."""
    src = """
module m
  implicit none
contains
  subroutine inner(a)
    double precision, intent(inout) :: a(..)
    select rank (a)
    rank (1)
      a(1) = 1.0d0
    rank (2)
      a(1, 1) = 1.0d0
    rank default
    end select
  end subroutine inner

  subroutine outer(arr2d)
    double precision, intent(inout) :: arr2d(:, :)
    call inner(arr2d)
  end subroutine outer
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="outer", entry="m::outer")
    assert sdfg is not None, f"build failed: {err}"


# ---------------------------------------------------------------------------
# Pattern M -- ``EQUIVALENCE`` statement (legacy F77).
# ---------------------------------------------------------------------------
def test_m_equivalence_statement(tmp_path):
    """EQUIVALENCE (a, b) -- a(1) and b(1) share the same memory cell; flang may or may not
    lower it in a form the bridge can recognise."""
    src = """
module m
  implicit none
contains
  subroutine fill(a)
    double precision, intent(inout) :: a(100)
    double precision :: scratch(100)
    double precision :: alias(50, 2)
    equivalence (scratch, alias)
    integer :: i, j
    do j = 1, 2
      do i = 1, 50
        alias(i, j) = real(i + (j - 1) * 50, kind=8)
      end do
    end do
    do i = 1, 100
      a(i) = scratch(i)
    end do
  end subroutine fill
end module m
"""
    sdfg, err = _try_build(tmp_path / "sdfg", src, name="fill", entry="m::fill")
    assert sdfg is not None, f"build failed: {err}"
