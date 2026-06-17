"""Verify the bridge correctly wraps fp32 constants via ``dace.float32(...)``.

Fortran's default ``REAL`` is single precision (fp32) unless the user
suffixes a literal with ``d0`` / ``_8`` / ``kind=8`` etc.  Flang lowers
``1.4`` as ``arith.constant 1.4 : f32`` -- the binary representation
is the fp32 approximation of 1.4 (``0x3FB33333``), which rounds to
``1.399999976158142`` when widened to fp64.

The bridge MUST emit f32 constants wrapped as ``dace.float32(...)``
so DaCe's symbolic engine + codegen propagate the right precision.
Without the wrap, the tasklet code would render ``1.4`` as a
Python float (fp64) and downstream multiplications would silently
promote.  See ``expressions.cpp:1505`` for the wrap site.

These probes pin the contract:

  * fp32 constants in fp32 expressions stay fp32 (bit-exact to
    ``numpy.float32(...)`` arithmetic).
  * fp32 constants assigned to fp64 variables widen through the
    convert chain; the bridge's wrap must NOT prevent the convert.
  * fp64 constants (``1.4d0``) emit as plain ``1.4`` (or 17-digit
    precision); no ``dace.float32`` wrap should appear.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fp32_constant_wrapped_in_dace_float32(tmp_path):
    """``REAL :: out; out = 1.4 * 2.0`` -- 1.4 and 2.0 are default
    real (fp32) literals.  The product should be bit-exact to
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
    """``DOUBLE PRECISION :: out; out = 1.4d0 * 2.0d0`` -- the ``d0``
    suffix makes the literals genuine fp64.  The product is
    bit-exact to ``np.float64(1.4) * np.float64(2.0)``, not the
    fp32-widened form."""
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
    """``DOUBLE PRECISION :: c; parameter (c = 1.4); out = c * 2.0d0``
    -- the parameter's literal 1.4 is fp32 (default real), then
    widened to fp64 at assignment.  Without the bridge preserving
    the fp32 precision, the widened value would be exact 1.4
    (impossible -- 1.4 has no exact fp64 representation).  Bridge
    should produce the fp32-widened value 1.399999976158142...
    matching gfortran's compile-time semantic."""
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
    # gfortran semantic: 1.4 is fp32, widened to fp64 = 1.399999976...
    expected = np.float64(np.float32(1.4)) * np.float64(2.0)
    assert out[0] == expected, f"widened fp32 parameter mismatch: got {out[0]} expected {expected}"


def test_fp64_d_suffix_param_keeps_precision(tmp_path):
    """Same as above but ``c = 1.4d0`` -- the fp64 suffix gives full
    fp64 precision.  Result should match ``np.float64(1.4) * 2``
    (NOT the fp32-widened form)."""
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
