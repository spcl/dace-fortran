"""Negative tests -- patterns the HLFIR bridge deliberately refuses to lower,
each producing a clear error naming the failing source location and reason.
Live tests (not just READMEs) keep the contract enforced in CI.

Currently covered: symbolic-extent noncontiguous gather (no compile-time-
constant size). hlfir-expand-vector-subscript-gather only lowers a constant-
integer extent; otherwise the pass aborts with op.emitError.

Higher-rank noncontiguous gathers and INTENT(out) scatter-back fail the same
way; once implemented, the xfails in noncontig_pardecls_test.py move here as positive tests.
"""

from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# scatter/gather variants (incl. symbolic extents) are supported in Phase 1.5;
# positive tests in noncontig_gather_scatter_test.py. Remaining unsupported case:
# ALIASED self-assignment (same source/dest array, RHS-to-temp order) -- see
# test_gather_scatter_aliasing_same_array there (xfailed).


def test_placeholder_for_future_unsupported_cases(tmp_path: Path):
    """Stub -- add bail-out tests here when new deliberately-unsupported patterns are introduced."""
    pass


def test_virtual_dispatch_bails_loudly(tmp_path: Path):
    """Genuinely runtime-polymorphic fir.dispatch -- a polymorphic dummy
    class(t) the function itself dispatches on. Flang's static devirtualisation
    can't resolve it (concrete type only known at the call site, outside the
    compiled function); surviving fir.dispatch after fir-polymorphic-op trips
    hlfir-reject-polymorphism.

    Asserts the pipeline raises RuntimeError naming polymorphism.
    """
    src = """
module shapes
  implicit none
  type, abstract :: shape_t
  contains
    procedure(area_iface), deferred :: area
  end type shape_t

  abstract interface
    function area_iface(this) result(a)
      import :: shape_t
      class(shape_t), intent(in) :: this
      real :: a
    end function
  end interface

  type, extends(shape_t) :: circle_t
    real :: r
  contains
    procedure :: area => circle_area
  end type circle_t

contains
  function circle_area(this) result(a)
    class(circle_t), intent(in) :: this
    real :: a
    a = 3.141592 * this%r * this%r
  end function
end module shapes

subroutine main(p, out)
  use shapes
  implicit none
  ! ``p`` is a CLASS-typed dummy argument — a polymorphic dummy whose
  ! concrete runtime type is determined by the caller, *outside* the
  ! function being compiled.  ``p%area()`` here is a true virtual
  ! dispatch that ``fir-polymorphic-op`` cannot statically resolve;
  ! it lowers to an indirect ``fir.call`` through the type-info
  ! dispatch table, which our reject pass catches via the leftover
  ! ``fir.box_tdesc`` marker.
  class(shape_t), intent(in) :: p
  real, intent(out) :: out
  out = p%area()
end subroutine main
"""
    with pytest.raises(RuntimeError) as exc:
        build_sdfg(src, tmp_path, name='main', entry='main').build()
    msg = str(exc.value)
    assert "pipeline failed" in msg or "polymorphism" in msg, (
        f"expected a pipeline-failed message naming polymorphism, "
        f"got: {msg}")
