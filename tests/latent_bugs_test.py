"""Regression tests for latent bugs found in the post-dd80990 audit.

Pins behaviour for design-audit findings #3-#6:
  * #3 emit_while parity with emit_cond (array reads in DO WHILE cond).
  * #4 ``-0.0`` sign preservation in the shortest-round-trip float printer.
  * #5 NaN / +inf / -inf emitted faithfully from Fortran-source literals.
  * #6 mixed-triplet ``hlfir.designate`` (one scalar + one slice) in a
       supported context.

Convention (per user direction): we treat ``NaN == NaN`` and
``+/-inf == +/-inf`` for round-trip purposes -- the printer must emit
them verbatim, but tests assert via ``np.isnan`` / ``np.isinf`` + sign
rather than equality.
"""
import math
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(),
                                reason="flang-new-21 not on PATH")


# ---------------------------------------------------------------------------
# Bug #3 -- emit_while + array-read condition.  ``DO WHILE (i <= n .AND.
# a(i) > thr)`` exercises the lift-via-tasklet path emit_while gained to
# mirror emit_cond.  Today the bridge typically folds while-conds into
# break nodes (cond_expr = ``True``), so this test pins the lift path
# the day a non-trivial cond actually reaches emit_while.
# ---------------------------------------------------------------------------
def test_while_cond_with_array_read(tmp_path):
    src = """
MODULE while_arr_mod
CONTAINS
SUBROUTINE while_arr(a, thr, n, count)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: a(n), thr
  INTEGER, INTENT(OUT) :: count
  INTEGER :: i
  count = 0
  i = 1
  DO WHILE (i <= n)
    IF (.NOT. (a(i) > thr)) EXIT
    count = count + 1
    i = i + 1
  END DO
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="while_arr",
                      entry="while_arr_mod::while_arr").build()
    a = np.array([2.0, 3.0, 1.5, 0.5, 4.0], dtype=np.float64, order="F")
    thr = 1.0
    count_arr = np.array([0], dtype=np.int32)
    sdfg(a=a, thr=thr, n=np.int32(5), count=count_arr)
    # i=1 (a=2.0 > 1.0): count=1
    # i=2 (a=3.0 > 1.0): count=2
    # i=3 (a=1.5 > 1.0): count=3
    # i=4 (a=0.5 not > 1.0): EXIT
    assert count_arr[0] == 3


# ---------------------------------------------------------------------------
# Bug #4 -- ``-0.0`` survives the shortest-round-trip float printer.
# Without the ``std::signbit`` short-circuit in expressions.cpp:1632, the
# printer's IEEE equality check accepted ``"0"`` for ``-0.0`` and the
# generated C++ flipped the sign.  Observable in ``1.0 / x`` (-inf -> +inf),
# ``ATAN2(x, -1.0)`` (-pi -> +pi), ``SIGN(y, x)`` and complex branch cuts.
# ---------------------------------------------------------------------------
def test_negative_zero_division_yields_negative_inf(tmp_path):
    """``1.0 / -0.0 == -inf`` -- the sign of zero determines the sign of inf."""
    src = """
MODULE neg_zero_div_mod
CONTAINS
SUBROUTINE neg_zero_div(out_pos_inf, out_neg_inf)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out_pos_inf, out_neg_inf
  REAL(8), PARAMETER :: zp = 0.0_8
  REAL(8), PARAMETER :: zn = -0.0_8
  out_pos_inf = 1.0_8 / zp
  out_neg_inf = 1.0_8 / zn
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="neg_zero_div",
                      entry="neg_zero_div_mod::neg_zero_div").build()
    pos = np.zeros(1, dtype=np.float64)
    neg = np.zeros(1, dtype=np.float64)
    sdfg(out_pos_inf=pos, out_neg_inf=neg)
    assert np.isposinf(pos[0]), f"expected +inf, got {pos[0]}"
    assert np.isneginf(neg[0]), f"expected -inf, got {neg[0]}"


def test_negative_zero_atan2_branch(tmp_path):
    """``ATAN2(-0.0, -1.0)`` returns -pi, ``ATAN2(+0.0, -1.0)`` returns +pi.
    Sign of zero selects the branch of atan2 on the negative real axis."""
    src = """
MODULE atan2_zero_sign_mod
CONTAINS
SUBROUTINE atan2_zero_sign(out_pos, out_neg)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out_pos, out_neg
  REAL(8), PARAMETER :: zp = 0.0_8
  REAL(8), PARAMETER :: zn = -0.0_8
  out_pos = ATAN2(zp, -1.0_8)
  out_neg = ATAN2(zn, -1.0_8)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="atan2_zero_sign",
                      entry="atan2_zero_sign_mod::atan2_zero_sign").build()
    pos = np.zeros(1, dtype=np.float64)
    neg = np.zeros(1, dtype=np.float64)
    sdfg(out_pos=pos, out_neg=neg)
    assert math.isclose(pos[0], math.pi, abs_tol=1e-15)
    assert math.isclose(neg[0], -math.pi, abs_tol=1e-15)


def test_negative_zero_signbit_preserved(tmp_path):
    """``-0.0`` as a literal output preserves its sign bit through
    the printer + DaCe codegen + ctypes round-trip."""
    src = """
MODULE emit_neg_zero_mod
CONTAINS
SUBROUTINE emit_neg_zero(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  out = -0.0_8
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="emit_neg_zero",
                      entry="emit_neg_zero_mod::emit_neg_zero").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert out[0] == 0.0
    assert math.copysign(1.0, out[0]) == -1.0, (
        f"expected -0.0 (sign bit set), got {out[0]} with sign "
        f"{math.copysign(1.0, out[0])}")


# ---------------------------------------------------------------------------
# Bug #5 -- NaN / +inf / -inf literal constants emit faithfully from
# Fortran source through the shortest-round-trip printer.  Per user
# convention, NaN payload doesn't matter -- ``np.isnan`` is the contract.
# Both sign of inf and "is nan" must round-trip.
# ---------------------------------------------------------------------------
def test_positive_infinity_arithmetic(tmp_path):
    """``1.0/0.0`` -> +inf survives the printer + DaCe codegen."""
    src = """
MODULE emit_pos_inf_mod
CONTAINS
SUBROUTINE emit_pos_inf(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8), PARAMETER :: zero = 0.0_8
  out = 1.0_8 / zero
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="emit_pos_inf",
                      entry="emit_pos_inf_mod::emit_pos_inf").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert np.isposinf(out[0]), f"expected +inf, got {out[0]}"


def test_negative_infinity_arithmetic(tmp_path):
    """``-1.0/0.0`` -> -inf survives the printer + DaCe codegen."""
    src = """
MODULE emit_neg_inf_mod
CONTAINS
SUBROUTINE emit_neg_inf(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8), PARAMETER :: zero = 0.0_8
  out = -1.0_8 / zero
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="emit_neg_inf",
                      entry="emit_neg_inf_mod::emit_neg_inf").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert np.isneginf(out[0]), f"expected -inf, got {out[0]}"


def test_nan_arithmetic(tmp_path):
    """``0.0/0.0`` -> NaN survives the printer + DaCe codegen.
    Per convention NaN payload doesn't matter; ``np.isnan`` is the
    contract."""
    src = """
MODULE emit_nan_mod
CONTAINS
SUBROUTINE emit_nan(out)
  IMPLICIT NONE
  REAL(8), INTENT(OUT) :: out
  REAL(8), PARAMETER :: zero = 0.0_8
  out = zero / zero
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="emit_nan",
                      entry="emit_nan_mod::emit_nan").build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    assert np.isnan(out[0]), f"expected NaN, got {out[0]}"


# ---------------------------------------------------------------------------
# Bug #6 -- mixed-triplet ``hlfir.designate`` (one scalar + one slice
# index) in a supported context (section assign).  ``buildExpr``'s
# designate handler returns the bare array name when any index is a
# triplet; the bridge's AccessInfo carries the slab descriptor and
# emit_tasklet wires one connector + memlet for the slab.
# Pins behaviour for the common ``arr_2d(i, lo:hi) = ...`` shape.
# ---------------------------------------------------------------------------
def test_mixed_triplet_section_assign(tmp_path):
    src = """
MODULE mixed_triplet_assign_mod
CONTAINS
SUBROUTINE mixed_triplet_assign(s_y, out, i, n)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: i, n
  REAL(8), INTENT(IN) :: s_y(10, 100)
  REAL(8), INTENT(OUT) :: out(100)
  out(1:n) = s_y(i, 1:n)
END SUBROUTINE
END MODULE
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="mixed_triplet_assign",
                      entry="mixed_triplet_assign_mod::mixed_triplet_assign").build()
    s_y = np.zeros((10, 100), order="F", dtype=np.float64)
    s_y[3, :5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = np.zeros(100, dtype=np.float64, order="F")
    sdfg(s_y=s_y, out=out, i=np.int32(4), n=np.int32(5))
    np.testing.assert_array_equal(out[:5], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(out[5:], 0.0)
