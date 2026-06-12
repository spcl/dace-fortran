"""Frontend-recognition test for ``MATMUL(TRANSPOSE(A), B)``.

Confirms that the default ``flang-new -fc1`` HLFIR -- which emits a
separate ``hlfir.transpose`` followed by ``hlfir.matmul`` -- produces
both the ``Transpose`` and ``MatMul`` library nodes.

The optimised ``hlfir.matmul_transpose`` op only emerges under the
``hlfir-optimized-bufferization`` pass; the bridge raises a clear
``NotImplementedError`` on it via the surfaced libcall miss until a
dedicated lowering lands.
"""
from pathlib import Path
import sys

import pytest

import dace_fortran

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(probe_name: str, entry: str, tmp_path):
    src = (_HERE / probe_name).read_text()
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry=entry, name=entry)
    sdfg.validate()
    return sdfg


def _classes(sdfg):
    return {type(n).__name__ for s in sdfg.states() for n in s.nodes()}


def _count(sdfg, class_name):
    return sum(1 for s in sdfg.states() for n in s.nodes() if type(n).__name__ == class_name)


def test_matmul_of_transpose_lhs_default_lowering(tmp_path):
    """``C = MATMUL(TRANSPOSE(A), B)`` -- under Flang's optimised
    bufferisation the LHS transpose fuses into ``hlfir.matmul_transpose``,
    which the bridge now lowers to a single ``MatMul`` carrying
    ``transA=True``.  No separate Transpose libcall is emitted; the
    BLAS-level transpose flag handles it in-place."""
    sdfg = _build("matmul_transpose_probe.f90", "matmul_transpose_run", tmp_path)
    classes = _classes(sdfg)
    assert "MatMul" in classes, sorted(classes)
    mm_nodes = [n for s in sdfg.states() for n in s.nodes()
                if type(n).__name__ == "MatMul"]
    assert any(getattr(n, 'transA', False) is True for n in mm_nodes), \
        "expected at least one MatMul with transA=True (A-transpose folded in)"
    # No Transpose libcall should appear -- the fused path replaces it.
    assert _count(sdfg, "Transpose") == 0, \
        f"expected zero Transpose libcalls (A-transpose folds into MatMul), got {_count(sdfg, 'Transpose')}"


def test_matmul_with_transpose_rhs(tmp_path):
    """``C = MATMUL(A, TRANSPOSE(B))`` -- folds into ``MatMul(transB=True)``.
    The libcall dispatcher now SKIPS materialising the
    ``hlfir.transpose`` operand when it feeds a matmul (the BLAS
    flag handles the transpose in-place), so no Transpose libcall
    is emitted."""
    sdfg = _build("matmul_a_transposeb_probe.f90", "matmul_a_transposeb", tmp_path)
    assert _count(sdfg, "MatMul") == 1, \
        f"expected exactly one MatMul, got {_count(sdfg, 'MatMul')}"
    assert _count(sdfg, "Transpose") == 0, \
        f"expected zero Transpose libcalls (B-transpose folds via transB), got {_count(sdfg, 'Transpose')}"
    mm = [n for s in sdfg.states() for n in s.nodes()
          if type(n).__name__ == "MatMul"][0]
    assert mm.transB is True, f"expected transB=True, got {mm.transB}"


def test_matmul_both_transposed(tmp_path):
    """``C = MATMUL(TRANSPOSE(A), TRANSPOSE(B))`` -- both flags fold,
    zero Transpose libcalls.  The materialiser-skip + buildLibCallNode
    detection together produce one ``MatMul(transA=True, transB=True)``
    with NO transient -- BLAS does both transposes in place."""
    sdfg = _build("matmul_transpose_both_probe.f90", "matmul_transpose_both", tmp_path)
    assert _count(sdfg, "Transpose") == 0, \
        f"expected zero Transpose libcalls (both fold via BLAS flags), got {_count(sdfg, 'Transpose')}"
    assert _count(sdfg, "MatMul") == 1, \
        f"expected exactly one MatMul, got {_count(sdfg, 'MatMul')}"
    mm_nodes = [n for s in sdfg.states() for n in s.nodes()
                if type(n).__name__ == "MatMul"]
    assert mm_nodes[0].transA is True and mm_nodes[0].transB is True, \
        f"MatMul should have transA=True transB=True, got transA={mm_nodes[0].transA} transB={mm_nodes[0].transB}"
