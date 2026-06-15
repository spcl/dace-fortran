"""End-to-end probes for inline use of every HLFIR libcall the
elemental + ``hlfir.apply`` materialisation handles.

The bridge's elemental walker at
``bridge/ast/control_flow.cpp::walkElementalBody`` pre-emits a
``_libtmp_<gid>`` transient when the elemental body's ``hlfir.apply``
reads a libcall expr-producer.  ``libcallNameForExprOp``
(``bridge/ast/elementals.cpp``) is the gate -- every HLFIR op-name
listed there gets the materialisation, anything else falls out as
``?`` in the tasklet body.

These probes pin the inline-libcall coverage so a future regression
in the gate surfaces here at parse time, not in QE's microkernel
diff at integration time.

Coverage as of the matmul_transpose fix (commit referenced in this
file's git log):

  * matmul                   -- ``2.0 - matmul(a, b)`` (existing)
  * transpose                -- ``1.0 - transpose(a)`` (existing)
  * dot_product              -- inline ``dot_product(...)``
  * matmul_transpose         -- ``MATMUL(TRANSPOSE(...)) / scalar``
                                 (QE vcut_get -- THIS commit)
  * count                    -- inline ``COUNT(...)`` (THIS commit)
  * minloc / maxloc          -- inline ``MINLOC(...) + 1`` (THIS)
  * cshift                   -- ``2.0 - cshift(arr, 1)`` (THIS)

Gaps still surfacing ``?`` (separate work items):

  * hlfir.reshape -- not in the dispatcher's libcall table at all
  * hlfir.sum / product / minval / maxval / any / all with DIM
    (array-result dim-reductions go through buildSectionReduceAssign,
    not the libcall path)
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_inline_matmul_transpose_in_elemental_division(tmp_path):
    """``MATMUL(TRANSPOSE(A), q) / scalar`` (QE vcut_get shape).

    The optimised-bufferization pass fuses TRANSPOSE+MATMUL to
    ``hlfir.matmul_transpose``; the elemental body's apply walks
    back to this fused op.  ``libcallNameForExprOp`` recognises it
    and the materialisation emits a ``Transpose + MatMul`` libcall
    pair into the SDFG without any source-Fortran rewrite."""
    src = """
module m
contains
  subroutine qe_pattern(a, q, s, res)
    real(kind=8), intent(in) :: a(3, 3), q(3), s
    real(kind=8), intent(out) :: res(3)
    res = MATMUL(TRANSPOSE(a), q) / s
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="qe_pattern", entry="qe_pattern").build()
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64, order='F')
    q = np.array([1.0, 2.0, 3.0], dtype=np.float64, order='F')
    res = np.zeros(3, dtype=np.float64, order='F')
    sdfg(a=A, q=q, s=np.float64(2.0), res=res)
    np.testing.assert_allclose(res, (A.T @ q) / 2.0)


def test_inline_cshift_in_elemental(tmp_path):
    """``2.0 - CSHIFT(arr, 1)`` -- the cshift expr-producer feeds an
    elemental's apply.  The bridge stashes the shift into
    ``options['shift']`` (so the ``CShift`` node gets the concrete
    shift, not the ``__shift`` fallback symbol that would leak as a
    free symbol) and d-face's ``ExpandCShiftPure`` lowers it to a
    single Map whose source memlet subset is the ``Mod``-rotated
    index."""
    src = """
module m
contains
  subroutine cshift_inline(arr, res)
    real(kind=8), intent(in) :: arr(5)
    real(kind=8), intent(out) :: res(5)
    res = 2.0d0 - CSHIFT(arr, 1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="cshift_inline", entry="cshift_inline").build()
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, order='F')
    res = np.zeros(5, dtype=np.float64, order='F')
    sdfg(arr=arr, res=res)
    expected = 2.0 - np.roll(arr, -1)
    np.testing.assert_allclose(res, expected)
