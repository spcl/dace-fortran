"""Quantum-ESPRESSO ``zaxpy`` indirect complex-AXPY layout matrix.

The QE / SC26-layout ``zaxpy`` micro-kernel is a complex
``y += x`` accumulation under gather / scatter index maps, swept over
data layouts.  This test mirrors that matrix end-to-end through the
bridge: every layout x indirection variant is built into an SDFG and
checked against a numpy reference on the same data.

Layouts:

* **AoS** -- a Fortran ``complex(8)`` array (real / imag interleaved
  in memory, the natural Fortran complex layout).
* **SoA** -- two ``real(8)`` arrays (``*_re`` / ``*_im``), the
  layout-transformed form the SC26 paper sweeps.

Indirection (matching the C kernels ``kern_aos_*`` / ``kern_soa_*``):

* **direct**       -- ``y(i)        += x(i)``
* **gather**       -- ``y(i)        += x(xmap(i))``     (single, on x)
* **scatter**      -- ``y(ymap(i))  += x(i)``           (single, on y)
* **double**       -- ``y(ymap(i))  += x(xmap(i))``     (gather + scatter)

Index maps are distinct-target permutation samples (as QE's
``uniformSample`` produces), so the scatter accumulation order is
irrelevant and the comparison is exact.  Input values come from the
same xorshift64 PRNG the SC26 artifacts use (``Xor64Rng`` /
``splitmix64``); see :func:`_prng.xor64_uniform01`.
"""
import numpy as np
import pytest

from _prng import complex_stream
from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# (kind, indirection) -> Fortran kernel.  ``n`` = iteration count;
# ``ymap`` / ``xmap`` are 1-based index maps; AoS uses complex(8),
# SoA uses paired real(8) re/im arrays.
_AOS = {
    "direct": "y(i) = y(i) + x(i)",
    "gather": "y(i) = y(i) + x(xmap(i))",
    "scatter": "y(ymap(i)) = y(ymap(i)) + x(i)",
    "double": "y(ymap(i)) = y(ymap(i)) + x(xmap(i))",
}
_SOA = {
    "direct": "yr(i) = yr(i) + xr(i)\n    yi(i) = yi(i) + xi(i)",
    "gather": "yr(i) = yr(i) + xr(xmap(i))\n    yi(i) = yi(i) + xi(xmap(i))",
    "scatter": "yr(ymap(i)) = yr(ymap(i)) + xr(i)\n    yi(ymap(i)) = yi(ymap(i)) + xi(i)",
    "double": "yr(ymap(i)) = yr(ymap(i)) + xr(xmap(i))\n    yi(ymap(i)) = yi(ymap(i)) + xi(xmap(i))",
}


def _aos_src(body: str) -> str:
    return f"""
module zaxpy_aos_mod
  implicit none
contains
subroutine zaxpy_aos(n, ymap, xmap, x, y)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: ymap(n), xmap(n)
  complex(8), intent(in) :: x(:)
  complex(8), intent(inout) :: y(:)
  integer :: i
  do i = 1, n
    {body}
  end do
end subroutine zaxpy_aos
end module zaxpy_aos_mod
"""


def _soa_src(body: str) -> str:
    return f"""
module zaxpy_soa_mod
  implicit none
contains
subroutine zaxpy_soa(n, ymap, xmap, xr, xi, yr, yi)
  implicit none
  integer, intent(in) :: n
  integer, intent(in) :: ymap(n), xmap(n)
  real(8), intent(in) :: xr(:), xi(:)
  real(8), intent(inout) :: yr(:), yi(:)
  integer :: i
  do i = 1, n
    {body}
  end do
end subroutine zaxpy_soa
end module zaxpy_soa_mod
"""


# Small symbolic problem sizes -- this is a correctness test, not a
# benchmark, so a handful of elements exercises every memlet shape.
_N, _NX, _NY = 6, 12, 12


def _index_maps():
    """Distinct-target 1-based permutation samples (QE uniformSample
    shape) so scatter accumulation order can't perturb the result."""
    rng = np.random.default_rng(0)
    ymap = (rng.permutation(_NY)[:_N] + 1).astype(np.int32)
    xmap = (rng.permutation(_NX)[:_N] + 1).astype(np.int32)
    return np.asfortranarray(ymap), np.asfortranarray(xmap)


def _ref(body_key: str, ymap, xmap, x, y):
    """numpy reference accumulation for one indirection pattern."""
    yo = y.copy()
    for i in range(_N):
        yi = (ymap[i] - 1) if body_key in ("scatter", "double") else i
        xi = (xmap[i] - 1) if body_key in ("gather", "double") else i
        yo[yi] += x[xi]
    return yo


@pytest.mark.parametrize("indir", ["direct", "gather", "scatter", "double"])
def test_zaxpy_aos(tmp_path, indir):
    """AoS complex(8) AXPY -- direct + single (gather/scatter) + double
    indirection."""
    ymap, xmap = _index_maps()
    x = np.asfortranarray(complex_stream(_NX, seed=1))
    y = np.asfortranarray(complex_stream(_NY, seed=2))
    ref = _ref(indir, ymap, xmap, x, y)

    sdfg = build_sdfg(_aos_src(_AOS[indir]), tmp_path, name="zaxpy_aos", entry="zaxpy_aos_mod::zaxpy_aos").build()
    sdfg(n=np.int32(_N), ymap=ymap, xmap=xmap, x=x, y=y)
    np.testing.assert_allclose(y, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("indir", ["direct", "gather", "scatter", "double"])
def test_zaxpy_soa(tmp_path, indir):
    """SoA paired real(8) re/im AXPY -- same indirection matrix as AoS;
    the layout-transformed variant."""
    ymap, xmap = _index_maps()
    x = complex_stream(_NX, seed=1)
    y = complex_stream(_NY, seed=2)
    ref = _ref(indir, ymap, xmap, x, y)
    xr = np.asfortranarray(x.real.copy())
    xi = np.asfortranarray(x.imag.copy())
    yr = np.asfortranarray(y.real.copy())
    yi = np.asfortranarray(y.imag.copy())

    sdfg = build_sdfg(_soa_src(_SOA[indir]), tmp_path, name="zaxpy_soa", entry="zaxpy_soa_mod::zaxpy_soa").build()
    sdfg(n=np.int32(_N), ymap=ymap, xmap=xmap, xr=xr, xi=xi, yr=yr, yi=yi)
    np.testing.assert_allclose(yr, ref.real, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(yi, ref.imag, rtol=1e-12, atol=1e-12)
