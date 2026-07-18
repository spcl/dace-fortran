"""Frontend-recognition test for MATMUL(TRANSPOSE(A), B).

Confirms the default flang-new -fc1 HLFIR (separate hlfir.transpose + hlfir.matmul) produces
both Transpose and MatMul library nodes. The optimised hlfir.matmul_transpose op only emerges
under hlfir-optimized-bufferization; the bridge raises NotImplementedError on it via the
surfaced libcall miss until a dedicated lowering lands.
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
    """C = MATMUL(TRANSPOSE(A), B) -- LHS transpose fuses into hlfir.matmul_transpose, lowered to
    a single MatMul(transA=True); no separate Transpose libcall (BLAS flag handles it in-place)."""
    sdfg = _build("matmul_transpose_probe.f90", "matmul_transpose_run", tmp_path)
    classes = _classes(sdfg)
    assert "MatMul" in classes, sorted(classes)
    mm_nodes = [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == "MatMul"]
    assert any(getattr(n, 'transA', False) is True for n in mm_nodes), \
        "expected at least one MatMul with transA=True (A-transpose folded in)"
    # No Transpose libcall should appear -- the fused path replaces it.
    assert _count(sdfg, "Transpose") == 0, \
        f"expected zero Transpose libcalls (A-transpose folds into MatMul), got {_count(sdfg, 'Transpose')}"


def test_matmul_with_transpose_rhs(tmp_path):
    """C = MATMUL(A, TRANSPOSE(B)) -- folds into MatMul(transB=True); dispatcher skips
    materialising the hlfir.transpose operand when it feeds a matmul, so no Transpose libcall."""
    sdfg = _build("matmul_a_transposeb_probe.f90", "matmul_a_transposeb", tmp_path)
    assert _count(sdfg, "MatMul") == 1, \
        f"expected exactly one MatMul, got {_count(sdfg, 'MatMul')}"
    assert _count(sdfg, "Transpose") == 0, \
        f"expected zero Transpose libcalls (B-transpose folds via transB), got {_count(sdfg, 'Transpose')}"
    mm = [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == "MatMul"][0]
    assert mm.transB is True, f"expected transB=True, got {mm.transB}"


def test_matmul_both_transposed(tmp_path):
    """C = MATMUL(TRANSPOSE(A), TRANSPOSE(B)) -- both flags fold to one
    MatMul(transA=True, transB=True) with no transient and zero Transpose libcalls."""
    sdfg = _build("matmul_transpose_both_probe.f90", "matmul_transpose_both", tmp_path)
    assert _count(sdfg, "Transpose") == 0, \
        f"expected zero Transpose libcalls (both fold via BLAS flags), got {_count(sdfg, 'Transpose')}"
    assert _count(sdfg, "MatMul") == 1, \
        f"expected exactly one MatMul, got {_count(sdfg, 'MatMul')}"
    mm_nodes = [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == "MatMul"]
    assert mm_nodes[0].transA is True and mm_nodes[0].transB is True, \
        f"MatMul should have transA=True transB=True, got transA={mm_nodes[0].transA} transB={mm_nodes[0].transB}"
