"""Nested cartesian-array struct member read inside an inlined libcall. ICON-O's
veloc_adv_horz_mimetic_rot calls the inlined coriolis routine with p_diag%p_vn_dual (a POINTER
member, not a direct dummy) and does DOT_PRODUCT(p_vn_dual(i)%x, ...); cartesian-member flattening
(gates #4-7) only registered the SoA companion <arr>_x for DIRECT dummies, so reached through p_diag%
it never registered -> KeyError: '<arr>_x' at emit_libcall.

Gate #12: rootedAtStructDummy/walkMemberChain now walk THROUGH the inlined-call alias to the caller's
struct (reusing gate #11's leadsToComponentDesignate), and the libcall operand-subset builder renders
the whole-member read as the element slice diag_pvd_x[(i-1), 0:3] instead of the whole multi-dim
companion (which 1-D-only dot_product rejects)."""
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  type :: t_cc
    real(8) :: x(3)
  end type t_cc
  type :: t_diag
    type(t_cc), pointer, dimension(:) :: pvd
  end type t_diag
contains
  subroutine inner(pvd, n, out)
    type(t_cc), intent(in) :: pvd(:)
    integer, intent(in) :: n
    real(8), intent(out) :: out(n)
    integer :: i
    do i = 1, n
      out(i) = dot_product(pvd(i) % x, pvd(i) % x)
    end do
  end subroutine inner
  subroutine outer(diag, n, out)
    type(t_diag), intent(inout) :: diag
    integer, intent(in) :: n
    real(8), intent(out) :: out(n)
    call inner(diag % pvd, n, out)
  end subroutine outer
end module m
"""


def test_nested_cartesian_member_in_inlined_dot_product(tmp_path):
    """diag%pvd(i)%x reached through an inlined DOT_PRODUCT lowers: SoA companion diag_pvd_x
    registers and the dot-product reads a 1-D element slice."""
    sdfg = build_sdfg(_SRC, entry="m::outer", name="nested_cc_dot", out_dir=str(tmp_path))
    assert "diag_pvd_x" in sdfg.arrays, f"nested cartesian companion not registered: {sorted(sdfg.arrays)}"


# Gate #12b (multi-level inlining + hlfir.copy_in): outer passes the POINTER member diag%pvd through
# an intermediate inlined mid before inner's dot-product. Flang copies the non-contiguous pointer
# actual into a contiguous temp (hlfir.copy_in) at the call boundary, so the alias walk must peel
# copy_in AND hop dummy->dummy (asAssumedShapeAlias) through the chain to reach diag%pvd -- mirrors
# ICON-O veloc_adv's veloc -> coriolis_3d -> coriolis_fast_scalar -> rot_vertex nest.
_SRC_MULTI = """\
module m2
  implicit none
  type :: t_cc
    real(8) :: x(3)
  end type t_cc
  type :: t_diag
    type(t_cc), pointer, dimension(:) :: pvd
  end type t_diag
contains
  subroutine inner(pvd, n, out)
    type(t_cc), intent(in) :: pvd(:)
    integer, intent(in) :: n
    real(8), intent(out) :: out(n)
    integer :: i
    do i = 1, n
      out(i) = dot_product(pvd(i) % x, pvd(i) % x)
    end do
  end subroutine inner
  subroutine mid(pvd, n, out)
    type(t_cc), intent(in) :: pvd(:)
    integer, intent(in) :: n
    real(8), intent(out) :: out(n)
    call inner(pvd, n, out)
  end subroutine mid
  subroutine outer(diag, n, out)
    type(t_diag), intent(inout) :: diag
    integer, intent(in) :: n
    real(8), intent(out) :: out(n)
    call mid(diag % pvd, n, out)
  end subroutine outer
end module m2
"""


def test_nested_cartesian_member_through_multilevel_inline_copyin(tmp_path):
    """Cartesian companion still registers when the pointer member flows through MULTIPLE inlined
    levels + a hlfir.copy_in contiguous temp."""
    sdfg = build_sdfg(_SRC_MULTI, entry="m2::outer", name="nested_cc_multi", out_dir=str(tmp_path))
    assert "diag_pvd_x" in sdfg.arrays, f"nested cartesian companion not registered: {sorted(sdfg.arrays)}"
