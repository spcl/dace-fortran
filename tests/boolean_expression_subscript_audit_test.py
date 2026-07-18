"""Boolean-expression-rendering audit across bridge code paths.

commit 7c0835d fixed ``buildExpr``'s NoSubscriptGuard (expressions.cpp:1280-1288,
assumed tasklet-body context) rendering an AND-convergence check with bare array
names instead of subscripted. Probes cover the other call sites that could hit
the same bug: dispatch.cpp:218 (fir.store handler), :1919 (const-index assign),
:1415 (loop-bound fallback)."""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_boolean_store_to_logical_scalar(tmp_path):
    """``c = (a(1)>b(1)) .and. (a(2)>b(2))`` stored to LOGICAL scalar -- bare-name
    RHS would do pointer comparison instead."""
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
    """OR variant of the AND pattern above."""
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
    """Nested AND/OR mix of array-element comparisons -- subscript preservation
    must traverse all levels."""
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
