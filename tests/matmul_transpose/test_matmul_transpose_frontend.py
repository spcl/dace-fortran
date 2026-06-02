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
    """``C = MATMUL(TRANSPOSE(A), B)`` -- default pipeline lowers as
    a separate ``hlfir.transpose(A)`` + ``hlfir.matmul``."""
    sdfg = _build("matmul_transpose_probe.f90", "matmul_transpose_run", tmp_path)
    classes = _classes(sdfg)
    assert "Transpose" in classes and "MatMul" in classes, sorted(classes)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MATMUL(A, TRANSPOSE(B)) keeps a separate hlfir.transpose+hlfir.matmul pair "
        "(the optimised pass only fuses LHS transpose into hlfir.matmul_transpose).  "
        "The bridge's matmul libcall handler can't yet materialise a transpose-result "
        "operand on the fly -- needs the materialiseElementalForLibcall pattern "
        "extended to hlfir.transpose."))
def test_matmul_with_transpose_rhs(tmp_path):
    """``C = MATMUL(A, TRANSPOSE(B))`` -- gap: transpose-result operand."""
    sdfg = _build("matmul_a_transposeb_probe.f90", "matmul_a_transposeb", tmp_path)
    classes = _classes(sdfg)
    assert "Transpose" in classes and "MatMul" in classes, sorted(classes)


@pytest.mark.xfail(
    strict=True,
    reason="Same gap as MATMUL(A, TRANSPOSE(B)) -- the RHS transpose-result operand "
    "lookup yields an empty array name and emit_libcall trips on it.")
def test_matmul_both_transposed(tmp_path):
    """``C = MATMUL(TRANSPOSE(A), TRANSPOSE(B))`` -- two transposes."""
    sdfg = _build("matmul_transpose_both_probe.f90", "matmul_transpose_both", tmp_path)
    assert _count(sdfg, "Transpose") == 2
    assert _count(sdfg, "MatMul") == 1
