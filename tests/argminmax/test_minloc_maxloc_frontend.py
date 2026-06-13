"""Frontend-recognition tests for Fortran ``MINLOC`` / ``MAXLOC``.

Each test drives one isolated Fortran pattern through the bridge and
asserts the resulting SDFG carries an :class:`ArgMin` /
:class:`ArgMax` library node with the right ``dim``, ``back``, and
``one_based`` configuration.  Numerical correctness of the lib node
itself is covered in d-face's ``tests/library/argminmax_test.py``.
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


def _find_lib_nodes(sdfg, class_name: str):
    """Return every library-node instance of ``class_name`` in ``sdfg``."""
    hits = []
    for state in sdfg.states():
        for node in state.nodes():
            if type(node).__name__ == class_name:
                hits.append(node)
    return hits


def test_minloc_whole_array_recognised(tmp_path):
    """``idx = MINLOC(arr)`` -> a single ArgMin lib node, no dim, back=False."""
    sdfg = _build("minloc_whole_array_probe.f90", "minloc_whole", tmp_path)
    nodes = _find_lib_nodes(sdfg, "ArgMin")
    assert len(nodes) == 1, f"expected one ArgMin lib node, got {len(nodes)}"
    n = nodes[0]
    assert n.dim is None, f"whole-array MINLOC should leave dim=None, got {n.dim!r}"
    assert n.back is False
    assert n.one_based is True


def test_maxloc_whole_array_recognised(tmp_path):
    """``idx = MAXLOC(arr)`` -> a single ArgMax lib node."""
    sdfg = _build("maxloc_whole_array_probe.f90", "maxloc_whole", tmp_path)
    nodes = _find_lib_nodes(sdfg, "ArgMax")
    assert len(nodes) == 1
    n = nodes[0]
    assert n.dim is None
    assert n.back is False


def test_minloc_2d_whole_recognised(tmp_path):
    """2-D ``MINLOC(arr)`` returns a rank-1 (length-2) multi-dim subscript."""
    sdfg = _build("minloc_2d_whole_probe.f90", "minloc_2d_whole", tmp_path)
    nodes = _find_lib_nodes(sdfg, "ArgMin")
    assert len(nodes) == 1
    assert nodes[0].dim is None


def test_minloc_dim1_recognised(tmp_path):
    """``MINLOC(arr, dim=1)`` -> ArgMin with dim=1, output rank R-1."""
    sdfg = _build("minloc_dim_probe.f90", "minloc_dim1", tmp_path)
    nodes = _find_lib_nodes(sdfg, "ArgMin")
    assert len(nodes) == 1, f"expected one ArgMin, got {len(nodes)}"
    assert nodes[0].dim == 1


def test_maxloc_dim2_recognised(tmp_path):
    """``MAXLOC(arr, dim=2)`` -> ArgMax with dim=2."""
    sdfg = _build("maxloc_dim_probe.f90", "maxloc_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "ArgMax")
    assert len(nodes) == 1
    assert nodes[0].dim == 2


def test_minloc_back_recognised(tmp_path):
    """``MINLOC(arr, back=.true.)`` -> ArgMin with back=True."""
    sdfg = _build("minloc_back_probe.f90", "minloc_back", tmp_path)
    nodes = _find_lib_nodes(sdfg, "ArgMin")
    assert len(nodes) == 1
    assert nodes[0].back is True
