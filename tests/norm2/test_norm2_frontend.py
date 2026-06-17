"""Frontend-recognition test for Fortran ``NORM2``.

Drives :file:`norm2_whole_probe.f90`'s ``norm2_whole`` entry through
the bridge and asserts the resulting SDFG carries a single
:class:`dace.libraries.standard.nodes.norm2.Norm2` lib node.
"""
from pathlib import Path
import sys

import pytest

import dace_fortran

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_norm2_whole_array_recognised(tmp_path):
    """``r = NORM2(v)`` -> a single Norm2 lib node, dim=None (whole-array)."""
    src = (_HERE / "norm2_whole_probe.f90").read_text()
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"),
                                   entry="norm2_whole_mod::norm2_whole", name="norm2_whole")
    sdfg.validate()
    nodes = [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == "Norm2"]
    assert len(nodes) == 1, f"expected one Norm2 lib node, got {len(nodes)}"
    assert nodes[0].dim is None
