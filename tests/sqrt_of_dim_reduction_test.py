"""Verify ``SQRT(SUM(arr, dim))`` and similar shapes where an inline
elemental wraps a dim-reduction result.

QE's ``vcut_spheric_get`` (line 52) has
``rcut = 0.5 * MINVAL(SQRT(SUM(a ** 2, 1)))``.  Pipeline:

  * ``a ** 2`` -> inline elemental returning 3x3 expr
  * ``SUM(..., 1)`` -> ``hlfir.sum`` with DIM operand returning 3-vector
  * ``SQRT(...)`` -> outer elemental applying sqrt element-wise
  * ``MINVAL(...)`` -> reduction over the SQRT elemental

Before this fix, ``materialiseElementalToTransient`` walked the SQRT
elemental body and called ``buildExpr`` on ``math.sqrt %x`` where
``%x = hlfir.apply %sum_result, %i``.  The ``hlfir.sum`` source isn't an
inner elemental and isn't in ``kHlfirExprToTransient``, so the apply
branch fell through and returned ``?``, producing
``_out__mask_0 = sqrt(?)`` at emit_tasklet validation time -- the QE
``test_vexx_bp_k_gpu_parses`` xfail's current blocker.

Fix: ``materialiseElementalToTransient`` pre-walks the body for
``hlfir.apply`` ops whose source is an ``hlfir.sum`` / ``hlfir.product`` /
``hlfir.minval`` / ``hlfir.maxval`` (dim-reduction returning a vector).
For each, materialise the inner source elemental into a ``_libtmp_<gid>``
transient (via ``materialiseElementalForLibcall``) and emit a
``kind="reduce"`` AST node writing to a sibling transient.  Register the
reduction op in ``kHlfirExprToTransient`` so ``buildExpr``'s apply branch
renders the apply as the transient's name -- no ``?`` placeholder
escapes into the tasklet body.
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
    """MAXVAL counterpart -- exercises the same dim-reduction
    materialisation with the max identity literal."""
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
    """``SUM(LOG(SUM(a, 1) + 1.0))`` -- a 3-level nest
    (inner-SUM-dim -> +1.0 elemental -> LOG elemental -> outer SUM).
    The pre-walk in ``materialiseElementalToTransient`` now descends
    through nested elemental bodies, so the inner SUM-dim is
    materialised to its own transient and the chain resolves
    transient-by-transient (verifies the materialisation isn't
    sqrt-specific and handles arbitrary elemental nesting)."""
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
