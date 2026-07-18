"""Elementwise math intrinsics not covered by ``elemwise_intrinsics_test.py``: each test exercises one intrinsic family, compares SDFG output to a numpy reference.

- Hyperbolic: sinh/cosh/tanh (sinh lowers to a fir.call runtime call, not math.*).
- Inverse trig: asin/acos/atan/atan2 (math.* ops).
- Conversion: int/nint/aint/anint/floor -- nint via llvm.lround, aint via llvm.trunc; bridge maps to dace::int{32,64} casts and trunc/round.
- Modulo: mod (truncated) / modulo (floored) -- both lower to fir.call @_FortranAMod*Real8, both map to Python %, C++ codegen picks semantics per operand type.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_hyperbolic(tmp_path: Path):
    """sinh / cosh / tanh elementwise."""
    src = """
subroutine probe(x, out)
  real(8), intent(in)  :: x
  real(8), intent(out) :: out(3)
  out(1) = sinh(x)
  out(2) = cosh(x)
  out(3) = tanh(x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    out = np.zeros(3, dtype=np.float64)
    sdfg(x=0.7, out=out)
    np.testing.assert_allclose(out, [np.sinh(0.7), np.cosh(0.7), np.tanh(0.7)], rtol=1e-12)


def test_inverse_trig(tmp_path: Path):
    """asin / acos / atan / atan2."""
    src = """
subroutine probe(x, y, out)
  real(8), intent(in)  :: x, y
  real(8), intent(out) :: out(4)
  out(1) = asin(x)
  out(2) = acos(x)
  out(3) = atan(x)
  out(4) = atan2(y, x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    out = np.zeros(4, dtype=np.float64)
    sdfg(x=0.5, y=0.3, out=out)
    np.testing.assert_allclose(out, [np.arcsin(0.5), np.arccos(0.5), np.arctan(0.5), np.arctan2(0.3, 0.5)], rtol=1e-12)


def test_floor_aint(tmp_path: Path):
    """floor / aint -- round toward -inf / 0 respectively, return real of the same kind."""
    src = """
subroutine probe(x, out)
  real(8), intent(in)  :: x
  real(8), intent(out) :: out(2)
  out(1) = floor(x)
  out(2) = aint(x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    for v in (3.7, -3.7, 0.5):
        out = np.zeros(2, dtype=np.float64)
        sdfg(x=v, out=out)
        assert out[0] == np.floor(v)  # floor: -inf rounding
        assert out[1] == np.trunc(v)  # aint: trunc toward 0


def test_int_nint(tmp_path: Path):
    """int (truncating cast) / nint (rounding cast)."""
    src = """
subroutine probe(x, out_int, out_nint)
  real(8), intent(in)  :: x
  integer, intent(out) :: out_int, out_nint
  out_int = int(x)
  out_nint = nint(x)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    for v, expected_int, expected_nint in [(3.4, 3, 3), (3.6, 3, 4), (-3.4, -3, -3), (-3.6, -3, -4)]:
        out_int = np.zeros(1, dtype=np.int32)
        out_nint = np.zeros(1, dtype=np.int32)
        sdfg(x=v, out_int=out_int, out_nint=out_nint)
        assert int(out_int[0]) == expected_int, f"int({v}) = {out_int[0]}, want {expected_int}"
        assert int(out_nint[0]) == expected_nint, f"nint({v}) = {out_nint[0]}, want {expected_nint}"


def test_mod_modulo(tmp_path: Path):
    """Fortran MOD (truncated) and MODULO (floored) -- both are Python % at the bridge level; C++ codegen picks the right semantics per type."""
    src = """
subroutine probe(a, b, out)
  real(8), intent(in)  :: a, b
  real(8), intent(out) :: out(2)
  out(1) = mod(a, b)
  out(2) = modulo(a, b)
end subroutine
"""
    sdfg = build_sdfg(src, tmp_path, name='probe').build()
    # MOD truncated, MODULO floored: (-7,3) -> mod=-7-3*int(-7/3)=-1, modulo=-7-3*floor(-7/3)=2
    out = np.zeros(2, dtype=np.float64)
    sdfg(a=-7.0, b=3.0, out=out)
    # both bridge to %; codegen lowers it to fmod (truncated) for MOD positions and a floor-helper for MODULO positions -- verify against the floored numpy result.
    np.testing.assert_allclose(out[0], np.fmod(-7.0, 3.0), rtol=1e-12)
    np.testing.assert_allclose(out[1], -7.0 - 3.0 * np.floor(-7.0 / 3.0), rtol=1e-12)
