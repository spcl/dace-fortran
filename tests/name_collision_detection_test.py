"""Verify the bridge's name-collision detection between
``builder.arrays`` / ``builder.scalars`` / ``builder.symbols``.

A Fortran short name that appears as MULTIPLE VarInfos with
different ``role`` -- typically an entry-kernel block-arg ARRAY
whose name matches a SCALAR dummy of an inlined helper -- used
to leak into ``builder.scalars`` AND ``builder.arrays``
simultaneously, causing emit_tasklet to add a spurious 1D scalar
memlet ``arr[0]`` on top of the correct ND array memlet.  Surfaced
as graupel's ``InvalidSDFGEdgeError: Memlet subset does not match
node dimension (expected 2, got 1)``.

Two-layer defense:
  * builder-init de-collision: ARRAY wins over SCALAR over
    SYMBOL when the same name is present in multiple
    role-keyed dicts.
  * loud-fail post-condition: a RuntimeError at builder-init
    time if the three role-keyed dicts aren't disjoint -- so
    the gap is caught at extract time, not at SDFG validation
    200 states later with an opaque InvalidSDFGEdgeError.
"""
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_inlined_callee_scalar_shadowing_outer_array(tmp_path):
    """Reproduces the graupel pattern: an outer subroutine has a 2D
    array ``arr``; an inlined helper takes a scalar ``arr`` dummy.
    After the inliner runs, both the array and the scalar VarInfo
    end up in the entry function -- the bridge must classify the
    name as ARRAY (rank>0 wins) and not duplicate it across role
    dicts.  Direct probe of the de-collision logic."""
    src = """
module m
contains
  subroutine helper(arr)
    real(kind=8), intent(inout) :: arr   ! scalar dummy with name 'arr'
    arr = arr * 2.0d0
  end subroutine
  subroutine outer(arr, out)
    real(kind=8), intent(inout) :: arr(3, 3)   ! 2D array with name 'arr'
    real(kind=8), intent(out) :: out
    real(kind=8) :: tmp
    tmp = arr(1, 1)
    call helper(tmp)
    out = tmp + arr(2, 2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="outer", entry="m::outer").build()
    # ``arr`` must be in arrays (2D), NOT in scalars (the inlined
    # callee's collision must lose to the outer array).
    assert "arr" in sdfg.arrays
    assert tuple(int(s) for s in sdfg.arrays["arr"].shape) == (3, 3)
    # And the spurious 1D ``arr[0]`` memlet must NOT appear -- if it
    # did, validation would have raised
    # ``InvalidSDFGEdgeError: Memlet subset does not match node
    # dimension``.
    sdfg.validate()


def test_collision_loud_fail_is_loud(tmp_path):
    """The role-keyed dicts must remain disjoint after the
    de-collision logic.  Builder init raises RuntimeError if not --
    we can't easily force the failure without crafting a synthetic
    VarInfo list, so this is a smoke test that builder init doesn't
    erroneously raise on a normal kernel with NO collisions."""
    src = """
module m
contains
  subroutine f(a, b, out)
    real(kind=8), intent(in) :: a(3), b(3)
    real(kind=8), intent(out) :: out
    out = sum(a * b)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f", entry="m::f").build()
    # No exception -> no false-positive collision.
    assert "a" in sdfg.arrays
    assert "b" in sdfg.arrays
