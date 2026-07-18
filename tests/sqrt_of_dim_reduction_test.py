"""SQRT(SUM(arr, dim)) and similar: an inline elemental wrapping a dim-reduction result.  From QE's vcut_spheric_get: ``rcut = 0.5 * MINVAL(SQRT(SUM(a**2, 1)))``.

Was broken: materialiseElementalToTransient's walk of the SQRT elemental hit hlfir.apply on an hlfir.sum source, which isn't in kHlfirExprToTransient, so buildExpr fell through to ``?`` -- the QE test_vexx_bp_k_gpu_parses xfail's blocker.  Fixed by pre-walking for hlfir.apply ops over sum/product/minval/maxval, materialising the inner elemental to a ``_libtmp_<gid>`` transient, and registering the reduction in kHlfirExprToTransient.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_minval_of_sqrt_of_sum_dim(tmp_path):
    """QE ``vcut_spheric_get`` pattern -- the original blocker."""
    src = """
module m
contains
  subroutine driver(a, out)
    real(kind=8), intent(in) :: a(3, 3)
    real(kind=8), intent(out) :: out
    out = MINVAL(SQRT(SUM(a ** 2, 1)))
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(a=a, out=out)
    # SUM(a**2, 1) reduces along Fortran dim 1 (numpy axis 0).
    np.testing.assert_allclose(out[0], np.min(np.sqrt(np.sum(a**2, axis=0))))


def test_maxval_of_sqrt_of_sum_dim(tmp_path):
    """MAXVAL counterpart -- same dim-reduction materialisation with the max identity literal."""
    src = """
module m
contains
  subroutine driver(a, out)
    real(kind=8), intent(in) :: a(4, 4)
    real(kind=8), intent(out) :: out
    out = MAXVAL(SQRT(SUM(a ** 2, 2)))
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    a = np.arange(16.0, dtype=np.float64).reshape(4, 4, order='F') + 0.5
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(a=a, out=out)
    np.testing.assert_allclose(out[0], np.max(np.sqrt(np.sum(a**2, axis=1))))


def test_sum_of_log_of_sum_dim(tmp_path):
    """SUM(LOG(SUM(a, 1) + 1.0)) -- 3-level nest (inner SUM-dim -> +1.0 elemental -> LOG elemental -> outer SUM); verifies the pre-walk descends through nested elemental bodies and isn't sqrt-specific."""
    src = """
module m
contains
  subroutine driver(a, out)
    real(kind=8), intent(in) :: a(3, 3)
    real(kind=8), intent(out) :: out
    out = SUM(LOG(SUM(a, 1) + 1.0d0))
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(a=a, out=out)
    np.testing.assert_allclose(out[0], np.sum(np.log(np.sum(a, axis=0) + 1.0)))
