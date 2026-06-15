"""Stability probes for the rank-reinterpretation view-alias path.

Each test exercises a Fortran source pattern that could plausibly
crash the bridge or misclassify the view.  Build success without
runtime error is the contract; numerical correctness is covered by
``rank_promotion_view_e2e_test.py``.

The intent is to lock down the corner cases so ingesting new Fortran
code doesn't surprise us: ``asAssumedShapeAlias`` returns null on rank
mismatch, ``extract_vars`` mints a view_alias VarInfo, ``descriptors.py``
synthesises column-major strides, ``access.py`` wires the source ->
view edge.  Each step must not crash on the patterns below.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _builds(tmp_path, src, name, entry):
    """Build an SDFG; succeed iff no exception is raised."""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name=name, entry=entry).build()
    sdfg.validate()
    return sdfg


# ---------------------------------------------------------------------------
# Same callee invoked from two call sites with same-shape actuals.
# Each call should reuse the same view-alias classification.
# ---------------------------------------------------------------------------
def test_multiple_call_sites_same_actual_shape(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5
  double precision :: arr1(A*B*C), arr2(A*B*C)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(A, B, C)
    buf(1, 1, 1) = 1.0d0
  end subroutine inner

  subroutine outer()
    call inner(arr1)
    call inner(arr2)
  end subroutine outer
end module m
"""
    _builds(tmp_path, src, name="outer", entry="outer")


# ---------------------------------------------------------------------------
# Nested rank change: 4D source -> 2D first dummy -> 1D second dummy.
# Each level of inlining must not collide aliases.
# ---------------------------------------------------------------------------
def test_chained_rank_changes(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5, D = 2
  double precision :: arr_4d(A, B, C, D)
contains
  subroutine bottom(flat)
    double precision, intent(inout) :: flat(A*B*C*D)
    flat(1) = 7.0d0
  end subroutine bottom

  subroutine middle(buf)
    double precision, intent(inout) :: buf(A*B, C*D)
    call bottom(buf)
  end subroutine middle

  subroutine top()
    call middle(arr_4d)
  end subroutine top
end module m
"""
    _builds(tmp_path, src, name="top", entry="top")


# ---------------------------------------------------------------------------
# Same-rank pass-through must NOT be misclassified as a view alias.
# The bridge should resolve accesses through ``traceToDecl`` to the
# source's name (no separate VarInfo for the dummy).
# ---------------------------------------------------------------------------
def test_same_rank_pass_through_not_a_view(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: M = 5, N = 3
  double precision :: arr_2d(M, N)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(M, N)
    buf(1, 1) = 9.0d0
  end subroutine inner

  subroutine outer()
    call inner(arr_2d)
  end subroutine outer
end module m
"""
    sdfg = _builds(tmp_path, src, name="outer", entry="outer")
    # Same-rank pass-through: only ``arr_2d`` should appear; no
    # separately-classified ``buf`` view alias.
    assert 'arr_2d' in sdfg.arrays
    # ``buf`` may or may not appear depending on the bridge's
    # collapse logic, but if it does it must NOT be a View.
    if 'buf' in sdfg.arrays:
        import dace.data as dt
        assert not isinstance(sdfg.arrays['buf'],
                              dt.View), ("Same-rank pass-through misclassified as a View; the bridge "
                                         "should resolve accesses through traceToDecl instead.")


# ---------------------------------------------------------------------------
# Caller passes an array section that the callee sees as a different
# rank.  Combines the section-reshape path with rank change.
# ---------------------------------------------------------------------------
def test_section_actual_with_rank_change(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5
  double precision :: arr_3d(A, B, C)
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(A, B)
    buf(1, 1) = 5.0d0
  end subroutine inner

  subroutine outer()
    integer :: k
    do k = 1, C
      call inner(arr_3d(:, :, k))
    end do
  end subroutine outer
end module m
"""
    _builds(tmp_path, src, name="outer", entry="outer")


# ---------------------------------------------------------------------------
# Optional dummy with rank reinterpretation: the OPTIONAL attribute
# adds a presence-flag companion -- the view-alias path must not
# crash on it.
# ---------------------------------------------------------------------------
def test_optional_dummy_rank_change(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: A = 5, B = 7, C = 4
  double precision :: arr_1d(A*B*C)
contains
  subroutine inner(buf)
    double precision, intent(inout), optional :: buf(A, B, C)
    if (present(buf)) then
      buf(1, 1, 1) = 1.0d0
    end if
  end subroutine inner

  subroutine outer()
    call inner(arr_1d)
  end subroutine outer
end module m
"""
    _builds(tmp_path, src, name="outer", entry="outer")


# ---------------------------------------------------------------------------
# Source is a LOCAL (alloca) 1D scratch, not a module global.
# ---------------------------------------------------------------------------
def test_local_scratch_with_rank_change(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5
contains
  subroutine inner(buf)
    double precision, intent(inout) :: buf(A, B, C)
    buf(1, 1, 1) = 11.0d0
  end subroutine inner

  subroutine outer()
    double precision :: scratch(A*B*C)
    scratch = 0.0d0
    call inner(scratch)
  end subroutine outer
end module m
"""
    _builds(tmp_path, src, name="outer", entry="outer")


# ---------------------------------------------------------------------------
# Pass-by-VALUE scalar args at the call site must not confuse the
# view detector (they aren't arrays).
# ---------------------------------------------------------------------------
def test_callee_with_mixed_scalar_and_array_dummies(tmp_path):
    src = """
module m
  implicit none
  integer, parameter :: A = 4, B = 3, C = 5
  double precision :: arr_1d(A*B*C)
contains
  subroutine inner(buf, n, x)
    double precision, intent(inout) :: buf(A, B, C)
    integer, intent(in) :: n
    double precision, intent(in) :: x
    buf(1, 1, 1) = x + real(n, kind=8)
  end subroutine inner

  subroutine outer()
    call inner(arr_1d, 42, 1.5d0)
  end subroutine outer
end module m
"""
    _builds(tmp_path, src, name="outer", entry="outer")
