"""Frontend-recognition tests for Fortran ``RESHAPE``: each test asserts the SDFG
carries a :class:`CopyLibraryNode` (the flat copy an ``hlfir.reshape`` with preserved
element count produces)."""
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


def _node_class_names(sdfg):
    return {type(n).__name__ for s in sdfg.states() for n in s.nodes()}


def test_reshape_2d_to_1d_recognised(tmp_path):
    """``out = RESHAPE(arr(n, m), SHAPE=[n*m])`` -> CopyLibraryNode."""
    sdfg = _build("reshape_2d_to_1d_probe.f90", "reshape_2d_to_1d", tmp_path)
    classes = _node_class_names(sdfg)
    assert "CopyLibraryNode" in classes, \
        f"expected RESHAPE to lower as CopyLibraryNode; got {sorted(classes)!r}"


def test_reshape_1d_to_2d_recognised(tmp_path):
    """``out(n, m) = RESHAPE(arr(n*m), SHAPE=[n, m])`` -> CopyLibraryNode."""
    sdfg = _build("reshape_1d_to_2d_probe.f90", "reshape_1d_to_2d", tmp_path)
    classes = _node_class_names(sdfg)
    assert "CopyLibraryNode" in classes
