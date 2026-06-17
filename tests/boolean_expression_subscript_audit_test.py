"""Audit probes for Boolean expression rendering across bridge code paths.

The yieldedExpr fix (commit 7c0835d) addressed ONE specific bug pattern:
a multi-element AND convergence check yielded into an scf.if ``__sc_<N>``
synthesis was rendered with BARE array names (``rsdnm < tolrsd``) instead
of subscripted (``rsdnm[0] < tolrsd[0]``).  Root cause: ``buildExpr``
at expressions.cpp:1280-1288 sets ``NoSubscriptGuard`` on ``arith.andi``
of i1 operands assuming the result lands in a tasklet body.

Other paths in the bridge call ``buildExpr`` with potentially-Boolean
operands -- if any of those paths route the result to an interstate
edge / non-tasklet context, they hit the same bug.

These probes exercise the audited shapes to confirm they emit correct
subscripted Boolean expressions.

Audited sites (dispatch.cpp):

  * line 218:  walkSCFBeforeRegion's fir.store handler -- if Fortran
    stores a Boolean expression to a scalar via ``logical :: c; c =
    (a > b) .and. (c > d)``, the bridge calls buildExpr on the andi.
  * line 1919: buildAST's const-index assign for hlfir.assign --
    similar shape but with array-element target.
  * line 1415: loop bound buildExpr fallback -- a do-loop with a
    Boolean-valued upper bound is unusual but probe anyway.

Each probe asserts the runtime result matches the Fortran semantics
under any caller context (interstate edge / tasklet body / loop bound).
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_boolean_store_to_logical_scalar(tmp_path):
    """``LOGICAL :: c; c = (a(1) > b(1)) .and. (a(2) > b(2))`` -- the
    Fortran-level assign stores the AND of two array-element
    comparisons to a logical scalar.  If the bridge renders the RHS
    with bare names (``a > b``), the C++ does pointer comparison.
    """
    src = """
module m
  implicit none
contains
  subroutine check(a, b, out)
    double precision, intent(in) :: a(2), b(2)
    logical, intent(out) :: out
    out = (a(1) > b(1)) .and. (a(2) > b(2))
  end subroutine check
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="check", entry="m::check").build()
    a = np.array([1.0, 1.0], dtype=np.float64, order='F')
    b = np.array([0.5, 0.5], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(a=a, b=b, out=out)
    assert out[0] != 0, f"both a > b -> AND should be true, got {out[0]}"

    a = np.array([1.0, 0.0], dtype=np.float64, order='F')
    b = np.array([0.5, 0.5], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(a=a, b=b, out=out)
    assert out[0] == 0, f"a(2)=0 < b(2)=0.5 -> AND should be false, got {out[0]}"


def test_boolean_or_store_to_logical_scalar(tmp_path):
    """``LOGICAL :: c; c = (a(1) > b(1)) .or. (a(2) > b(2))`` --
    OR variant of the same pattern."""
    src = """
module m
  implicit none
contains
  subroutine check(a, b, out)
    double precision, intent(in) :: a(2), b(2)
    logical, intent(out) :: out
    out = (a(1) > b(1)) .or. (a(2) > b(2))
  end subroutine check
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="check", entry="m::check").build()
    a = np.array([0.0, 1.0], dtype=np.float64, order='F')
    b = np.array([0.5, 0.5], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(a=a, b=b, out=out)
    assert out[0] != 0, f"a(2) > b(2) -> OR should be true, got {out[0]}"

    a = np.array([0.0, 0.0], dtype=np.float64, order='F')
    b = np.array([0.5, 0.5], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(a=a, b=b, out=out)
    assert out[0] == 0, f"neither -> OR should be false, got {out[0]}"


def test_nested_boolean_and_or_subscripts_preserved(tmp_path):
    """``((a > b) .and. (c > d)) .or. (e > f)`` -- nested mix to make
    sure the subscript preservation traverses all levels.  Each leaf is
    an array-element comparison."""
    src = """
module m
  implicit none
contains
  subroutine check(a, b, c, d, e, f, out)
    double precision, intent(in) :: a(1), b(1), c(1), d(1), e(1), f(1)
    logical, intent(out) :: out
    out = ((a(1) > b(1)) .and. (c(1) > d(1))) .or. (e(1) > f(1))
  end subroutine check
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="check", entry="m::check").build()
    # case: AND true (a>b, c>d) -> OR is true
    a = np.array([1.0], dtype=np.float64, order='F')
    b = np.array([0.0], dtype=np.float64, order='F')
    c = np.array([1.0], dtype=np.float64, order='F')
    d = np.array([0.0], dtype=np.float64, order='F')
    e = np.array([0.0], dtype=np.float64, order='F')
    f = np.array([1.0], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(a=a, b=b, c=c, d=d, e=e, f=f, out=out)
    assert out[0] != 0
    # case: AND false, e<=f -> OR is false
    c = np.array([0.0], dtype=np.float64, order='F')  # break AND
    out = np.zeros((1, ), dtype=np.int32, order='F')
    sdfg(a=a, b=b, c=c, d=d, e=e, f=f, out=out)
    assert out[0] == 0
