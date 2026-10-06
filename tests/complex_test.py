"""E2e tests for Fortran COMPLEX(4)/COMPLEX(8) through the HLFIR bridge. All tests use 1-D arrays
even for "scalar" cases: ctypes has no c_double_complex, so DaCe can't pass complex by-value."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import build_sdfg

# ---------------------------------------------------------------------------
# Arithmetic  --  parametrised over (kind, op, numpy-equivalent)
# ---------------------------------------------------------------------------

_COMPLEX_BIN_OPS = [
    # (fortran_op, numpy_op), id
    pytest.param("+", np.add, id="add"),
    pytest.param("-", np.subtract, id="sub"),
    pytest.param("*", np.multiply, id="mul"),
    # complex '/' lowers to __divdc3/__divsc3 (Smith's algorithm); bridge recognizes the 4-real call
    # shape and reconstructs the complex operands.
    pytest.param("/", np.divide, id="div"),
]

_COMPLEX_KINDS = [
    # (kind, np_dtype, fortran_decl), id
    pytest.param(4, np.complex64, "complex(4)", id="c4"),
    pytest.param(8, np.complex128, "complex(8)", id="c8"),
]


@pytest.mark.parametrize("kind,np_dtype,decl", _COMPLEX_KINDS)
@pytest.mark.parametrize("fop,np_op", _COMPLEX_BIN_OPS)
def test_complex_arithmetic(tmp_path: Path, kind, np_dtype, decl, fop, np_op):
    """Complex arithmetic on length-N arrays, compared against numpy."""
    src = f"""
subroutine main(n, a, b, out)
  integer,    intent(in)  :: n
  {decl}, intent(in)  :: a(n), b(n)
  {decl}, intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = a(i) {fop} b(i)
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    n = 8
    rng = np.random.default_rng(0)
    a = (rng.random(n) + 1j * rng.random(n) + 0.1).astype(np_dtype)
    b = (rng.random(n) + 1j * rng.random(n) + 0.1).astype(np_dtype)
    out = np.zeros(n, dtype=np_dtype)
    sdfg(n=n, a=a, b=b, out=out)
    rtol = 1e-6 if kind == 4 else 1e-12
    np.testing.assert_allclose(out, np_op(a, b), rtol=rtol)


# ---------------------------------------------------------------------------
# Transcendentals  --  Fortran intrinsic vs numpy
# ---------------------------------------------------------------------------

_COMPLEX_UNARY_FUNCS = [
    # (fortran_intrinsic, numpy_func), id
    pytest.param("SIN", np.sin, id="sin"),
    pytest.param("COS", np.cos, id="cos"),
    pytest.param("TAN", np.tan, id="tan"),
    pytest.param("SINH", np.sinh, id="sinh"),
    pytest.param("COSH", np.cosh, id="cosh"),
    pytest.param("TANH", np.tanh, id="tanh"),
    pytest.param("EXP", np.exp, id="exp"),
    pytest.param("LOG", np.log, id="log"),
    pytest.param("SQRT", np.sqrt, id="sqrt"),
]


@pytest.mark.parametrize("kind,np_dtype,decl", _COMPLEX_KINDS)
@pytest.mark.parametrize("fname,np_func", _COMPLEX_UNARY_FUNCS)
def test_complex_transcendentals(tmp_path: Path, kind, np_dtype, decl, fname, np_func):
    """Each complex transcendental lowers to c<func> (kind=8) / c<func>f (kind=4); bridge maps both
    to the bare Python name."""
    src = f"""
subroutine main(n, a, out)
  integer,    intent(in)  :: n
  {decl}, intent(in)  :: a(n)
  {decl}, intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = {fname}(a(i))
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    n = 6
    rng = np.random.default_rng(1)
    # tame domain avoids branch-cut mismatches vs numpy (e.g. imag~=+/-pi/2 for tan, real<=0 for log).
    a = (0.1 + 0.5 * rng.random(n) + 1j * (0.1 + 0.4 * rng.random(n))).astype(np_dtype)
    out = np.zeros(n, dtype=np_dtype)
    sdfg(n=n, a=a, out=out)
    rtol = 1e-5 if kind == 4 else 1e-12
    np.testing.assert_allclose(out, np_func(a), rtol=rtol)


def test_complex8_abs_returns_real(tmp_path: Path):
    """``ABS(complex(8))`` returns ``real(8)``  --  lowered as ``cabs``."""
    src = """
subroutine main(n, a, out)
  integer,    intent(in)  :: n
  complex(8), intent(in)  :: a(n)
  real(8),    intent(out) :: out(n)
  integer :: i
  do i = 1, n
    out(i) = abs(a(i))
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name="main").build()
    n = 4
    a = np.array([3 + 4j, 5 + 12j, -1 + 0j, 0 + 1j], dtype=np.complex128)
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=n, a=a, out=out)
    np.testing.assert_allclose(out, np.abs(a), rtol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
