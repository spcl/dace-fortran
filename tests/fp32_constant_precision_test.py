"""Bridge wraps fp32 constants via ``dace.float32(...)`` so DaCe's symbolic engine +
codegen propagate the right precision (see ``expressions.cpp:1505``).  Without the
wrap, tasklet code would render ``1.4`` as a Python float (fp64) and downstream
multiplications would silently promote.

Fortran's default ``REAL`` is fp32 unless suffixed ``d0``/``_8``/``kind=8``; flang
lowers ``1.4`` to the fp32 approximation (``0x3FB33333`` = 1.399999976158142 widened).

Pinned contract: fp32-in-fp32 stays fp32 (bit-exact to ``numpy.float32``); fp32
assigned to fp64 widens through convert (wrap must not block it); fp64 constants
(``1.4d0``) get no ``dace.float32`` wrap at all.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fp32_constant_wrapped_in_dace_float32(tmp_path):
    """``REAL :: out; out = 1.4 * 2.0`` (default-real fp32 literals) -- bit-exact to
    ``np.float32(1.4) * np.float32(2.0)``."""
    src = """
module m
  implicit none
contains
  subroutine f32_const(out)
    real, intent(out) :: out
    out = 1.4 * 2.0
  end subroutine f32_const
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f32_const", entry="m::f32_const").build()
    out = np.zeros((1, ), dtype=np.float32, order='F')
    sdfg(out=out)
    expected = np.float32(1.4) * np.float32(2.0)
    assert out[0] == expected, f"fp32 product mismatch: got {out[0]} expected {expected}"


def test_fp64_d_suffix_constant_keeps_full_precision(tmp_path):
    """``DOUBLE PRECISION :: out; out = 1.4d0 * 2.0d0`` (``d0`` = genuine fp64) -- bit-exact
    to ``np.float64(1.4) * np.float64(2.0)``, not the fp32-widened form."""
    src = """
module m
  implicit none
contains
  subroutine f64_const(out)
    double precision, intent(out) :: out
    out = 1.4d0 * 2.0d0
  end subroutine f64_const
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="f64_const", entry="m::f64_const").build()
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(out=out)
    expected = np.float64(1.4) * np.float64(2.0)
    assert out[0] == expected, f"fp64 product mismatch: got {out[0]} expected {expected}"


def test_fp32_param_widened_to_fp64_uses_f32_precision(tmp_path):
    """``parameter (c = 1.4)`` (fp32 default-real literal) widened to fp64 at assignment
    must produce 1.399999976158142... (the fp32-widened value), matching gfortran's
    compile-time semantic -- not exact 1.4 (impossible in fp64)."""
    src = """
module m
  implicit none
contains
  subroutine widen(out)
    double precision, intent(out) :: out
    double precision, parameter :: c = 1.4
    out = c * 2.0d0
  end subroutine widen
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="widen", entry="m::widen").build()
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(out=out)
    expected = np.float64(np.float32(1.4)) * np.float64(2.0)
    assert out[0] == expected, f"widened fp32 parameter mismatch: got {out[0]} expected {expected}"


def test_fp64_d_suffix_param_keeps_precision(tmp_path):
    """Same as above but ``c = 1.4d0`` (fp64 suffix) -- matches ``np.float64(1.4) * 2``,
    not the fp32-widened form."""
    src = """
module m
  implicit none
contains
  subroutine fp64param(out)
    double precision, intent(out) :: out
    double precision, parameter :: c = 1.4d0
    out = c * 2.0d0
  end subroutine fp64param
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fp64param", entry="m::fp64param").build()
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(out=out)
    expected = np.float64(1.4) * np.float64(2.0)
    assert out[0] == expected, f"fp64 param mismatch: got {out[0]} expected {expected}"
