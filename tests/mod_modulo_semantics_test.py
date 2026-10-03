# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
"""Fortran ``MOD`` truncates (the result has the sign of the dividend), ``MODULO`` floors (the sign of the
divisor) and integer ``/`` truncates toward zero. The bridge lowers them to the SDFG's ``FtnMod`` /
``FtnModulo`` and the constant folder evaluates them with the same rules; every sign combination of the
operands is checked, for integer and real kinds, in tasklets, in index expressions and at fold time."""

from pathlib import Path

import numpy as np
import pytest

from _helpers import sdfg_call_args
from _util import build_sdfg, have_flang
from dace_fortran.inliner.ast_desugaring import optimizations
from inliner.fortran_test_helper import SourceCodeBuilder, parse_and_improve

#: Every sign combination of dividend and divisor, exact and inexact.
_DIVIDENDS = [7, -7, 7, -7, 6, -6, 0, 1]
_DIVISORS = [3, 3, -3, -3, 3, -3, 5, -4]


def _truncated_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - np.fmod(a, b)) // b).astype(a.dtype)


@pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")
@pytest.mark.parametrize("kind,dtype", [(4, np.int32), (8, np.int64)])
def test_integer_mod_modulo_division_negative_operands(tmp_path: Path, kind: int, dtype: type):
    """Integer MOD lowers to ``arith.remsi`` (``FtnMod``), MODULO to the inlined floored idiom (``FtnModulo``)."""
    src = f"""
subroutine probe(n, a, b, r_mod, r_modulo, r_div)
  integer, intent(in) :: n
  integer({kind}), intent(in) :: a(n), b(n)
  integer({kind}), intent(out) :: r_mod(n), r_modulo(n), r_div(n)
  integer :: i
  do i = 1, n
    r_mod(i) = mod(a(i), b(i))
    r_modulo(i) = modulo(a(i), b(i))
    r_div(i) = a(i) / b(i)
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name=f"int_mod_{kind}").build()
    a = np.array(_DIVIDENDS, dtype=dtype)
    b = np.array(_DIVISORS, dtype=dtype)
    r_mod, r_modulo, r_div = (np.zeros_like(a) for _ in range(3))
    sdfg(a=a, b=b, r_mod=r_mod, r_modulo=r_modulo, r_div=r_div, **sdfg_call_args(sdfg, {"n": a.size}))
    np.testing.assert_array_equal(r_mod, np.fmod(a, b))
    np.testing.assert_array_equal(r_modulo, np.mod(a, b))
    np.testing.assert_array_equal(r_div, _truncated_div(a, b))


@pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")
@pytest.mark.parametrize("kind,dtype", [(4, np.float32), (8, np.float64)])
def test_real_mod_modulo_negative_operands(tmp_path: Path, kind: int, dtype: type):
    """Real MOD / MODULO lower to the ``_FortranAMod*`` / ``_FortranAModulo*`` runtime calls."""
    src = f"""
subroutine probe(n, a, b, r_mod, r_modulo)
  integer, intent(in) :: n
  real({kind}), intent(in) :: a(n), b(n)
  real({kind}), intent(out) :: r_mod(n), r_modulo(n)
  integer :: i
  do i = 1, n
    r_mod(i) = mod(a(i), b(i))
    r_modulo(i) = modulo(a(i), b(i))
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name=f"real_mod_{kind}").build()
    a = np.array(_DIVIDENDS, dtype=dtype) + dtype(0.5)
    b = np.array(_DIVISORS, dtype=dtype)
    r_mod, r_modulo = np.zeros_like(a), np.zeros_like(a)
    sdfg(a=a, b=b, r_mod=r_mod, r_modulo=r_modulo, **sdfg_call_args(sdfg, {"n": a.size}))
    np.testing.assert_allclose(r_mod, np.fmod(a, b), rtol=1e-6)
    np.testing.assert_allclose(r_modulo, np.mod(a, b), rtol=1e-6)


@pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")
def test_mod_and_division_in_index_expression_negative_dividend(tmp_path: Path):
    """``a(mod(i, 3) + 3)`` and ``a(i / 2 + 4)`` over ``i = -5..5``: floored rounding reads other elements."""
    src = """
subroutine probe(a, out_mod, out_div)
  integer, intent(in) :: a(6)
  integer, intent(out) :: out_mod(11), out_div(11)
  integer :: i
  do i = -5, 5
    out_mod(i + 6) = a(mod(i, 3) + 3)
    out_div(i + 6) = a(i / 2 + 4)
  end do
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name="mod_index_negative").build()
    a = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    out_mod, out_div = np.zeros(11, dtype=np.int32), np.zeros(11, dtype=np.int32)
    sdfg(a=a, out_mod=out_mod, out_div=out_div)
    i = np.arange(-5, 6)
    np.testing.assert_array_equal(out_mod, a[np.fmod(i, 3) + 2])
    np.testing.assert_array_equal(out_div, a[_truncated_div(i, np.full_like(i, 2)) + 3])


def test_constant_fold_mod_modulo_division_negative_operands():
    """The fparser constant folder applies the same truncated / floored rules as the compiled code."""
    sources, _ = (
        SourceCodeBuilder()
        .add_file("""
subroutine main(r)
  implicit none
  integer, parameter :: m = mod(-7, 3), mo = modulo(-7, 3), q = -7 / 2
  real, parameter :: rm = mod(-7.5, 3.0), rmo = modulo(-7.5, 3.0)
  real, intent(out) :: r(5)
  r(1) = m
  r(2) = mo
  r(3) = q
  r(4) = rm
  r(5) = rmo
end subroutine main
""")
        .check_with_gfortran()
        .get()
    )
    got = optimizations.const_eval_nodes(parse_and_improve(sources)).tofortran()
    assert "r(1) = (- 1)\n" in got, got
    assert "r(2) = 2\n" in got, got
    assert "r(3) = (- 3)\n" in got, got
    assert "r(4) = (- 1.5)\n" in got, got
    assert "r(5) = 1.5\n" in got, got


if __name__ == "__main__":
    import tempfile

    for kind, dtype in [(4, np.int32), (8, np.int64)]:
        with tempfile.TemporaryDirectory() as d:
            test_integer_mod_modulo_division_negative_operands(Path(d), kind, dtype)
    for kind, dtype in [(4, np.float32), (8, np.float64)]:
        with tempfile.TemporaryDirectory() as d:
            test_real_mod_modulo_negative_operands(Path(d), kind, dtype)
    with tempfile.TemporaryDirectory() as d:
        test_mod_and_division_in_index_expression_negative_dividend(Path(d))
    test_constant_fold_mod_modulo_division_negative_operands()
