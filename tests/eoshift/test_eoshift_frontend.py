"""Frontend-recognition test for Fortran ``EOSHIFT``.

The bridge recognises the heap-result runtime call
``fir.call @_FortranAEoshiftVector(...)`` and routes it through an
:class:`dace.libraries.standard.nodes.eoshift.EOShift` lib node.
"""
from pathlib import Path
import sys

import pytest

import dace_fortran

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from _util import have_flang  # noqa: E402

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_eoshift_1d_recognised(tmp_path):
    """``EOSHIFT(v, SHIFT=1, BOUNDARY=0.0)`` -> a single EOShift lib node."""
    src = (_HERE / "eoshift_1d_probe.f90").read_text()
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"),
                                   entry="eoshift_1d", name="eoshift_1d")
    sdfg.validate()
    nodes = [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == "EOShift"]
    assert len(nodes) == 1, f"expected one EOShift lib node, got {len(nodes)}"
