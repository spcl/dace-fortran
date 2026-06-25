"""Nested cartesian-array struct member read inside an inlined libcall.

ICON-O ``veloc_adv_horz_mimetic_rot`` calls the (inlined) coriolis routine
with ``p_diag % p_vn_dual`` -- a ``type(t_cartesian_coordinates), POINTER,
DIMENSION(:,:,:)`` *member* of ``p_diag`` -- and the inlined body does
``DOT_PRODUCT(p_vn_dual(i) % x, ...)``.  The cartesian-member flattening
(gates #4-7) registers the SoA companion ``<arr>_x`` only when ``p_vn_dual``
is a DIRECT dummy; reached through ``p_diag %`` it is a nested struct member,
so ``rootedAtStructDummy`` / ``walkMemberChain`` stopped at the inlined dummy
(whose own type is a record ARRAY, not a RecordType) and never registered the
companion -> ``KeyError: '<arr>_x'`` at ``emit_libcall``.

Gate #12: ``rootedAtStructDummy`` / ``walkMemberChain`` now walk THROUGH the
inlined-call alias to the caller's struct (reusing gate #11's
``leadsToComponentDesignate``), so the nested companion registers; and the
libcall operand-subset builder renders the whole-member read
``diag % pvd(i) % x`` as the element slice ``diag_pvd_x[(i-1), 0:3]`` instead
of the whole multi-dim companion (which a 1-D-only ``dot_product`` rejects).
"""
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
    """A cartesian-array struct member ``diag % pvd(i) % x`` reached through an
    inlined callee's ``DOT_PRODUCT`` lowers: the SoA companion ``diag_pvd_x``
    registers and the dot-product reads a 1-D element slice."""
    sdfg = build_sdfg(_SRC, entry="m::outer", name="nested_cc_dot", out_dir=str(tmp_path))
    assert "diag_pvd_x" in sdfg.arrays, f"nested cartesian companion not registered: {sorted(sdfg.arrays)}"
