"""Frontend-recognition tests for Fortran ``CSHIFT``."""
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


def _find_lib_nodes(sdfg, class_name: str):
    return [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == class_name]


def test_cshift_1d_const_shift_recognised(tmp_path):
    """``out = CSHIFT(arr, 1)`` -> CShift with dim=1, shift=1."""
    sdfg = _build("cshift_1d_const_shift_probe.f90", "cshift_1d_const", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    assert nodes[0].dim == 1
    assert str(nodes[0].shift) == "1"


def test_cshift_1d_var_shift_recognised(tmp_path):
    """``out = CSHIFT(arr, shift)`` -> CShift with shift = symbolic."""
    sdfg = _build("cshift_1d_var_shift_probe.f90", "cshift_1d_var", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    # shift expression is a symbolic reference to the Fortran dummy.
    assert nodes[0].shift is not None


def test_cshift_2d_dim2_recognised(tmp_path):
    """``out = CSHIFT(arr, 1, dim=2)`` -> CShift with dim=2."""
    sdfg = _build("cshift_2d_dim2_probe.f90", "cshift_2d_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    assert nodes[0].dim == 2
