"""Frontend-recognition + memlet-subset tests for Fortran ``EOSHIFT``.

``fir.call @_FortranAEoshiftVector`` routes to an
:class:`dace.libraries.standard.nodes.eoshift.EOShift` lib node; connector
memlets must cover the full shift axis (needed for the boundary-fill
expansion, itself a stub -- tests stop at SDFG-build time).
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
    return [(s, n) for s in sdfg.states() for n in s.nodes() if type(n).__name__ == class_name]


def _connector_memlets(sdfg, state, node):
    out = {}
    for e in state.in_edges(node):
        out[e.dst_conn] = e.data
    for e in state.out_edges(node):
        out[e.src_conn] = e.data
    return out


def test_eoshift_1d_recognised(tmp_path):
    """``EOSHIFT(v, SHIFT=1, BOUNDARY=0.0)`` -> a single EOShift lib node."""
    sdfg = _build("eoshift_1d_probe.f90", "eoshift_1d", tmp_path)
    nodes = _find_lib_nodes(sdfg, "EOShift")
    assert len(nodes) == 1


def test_eoshift_2d_dim2_recognised(tmp_path):
    """``EOSHIFT(arr, 1, BOUNDARY=-1.0, dim=2)`` -> EOShift with dim=2."""
    sdfg = _build("eoshift_2d_dim2_probe.f90", "eoshift_2d_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "EOShift")
    assert len(nodes) == 1


def test_eoshift_negative_shift_recognised(tmp_path):
    """``EOSHIFT(v, SHIFT=-2, BOUNDARY=99.0)`` wires an EOShift node.

    Constant-folding doesn't yet trace the literal shift through Flang's
    alloca+store+load chain, so ``n.shift`` may stay ``None``; pins recognition only.
    """
    sdfg = _build("eoshift_negative_shift_probe.f90", "eoshift_negative", tmp_path)
    nodes = _find_lib_nodes(sdfg, "EOShift")
    assert len(nodes) == 1


def test_eoshift_memlets_cover_full_arrays(tmp_path):
    """EOShift connectors get whole-array memlets (boundary fill needs the full extent), same as CShift."""
    import sympy
    sdfg = _build("eoshift_2d_dim2_probe.f90", "eoshift_2d_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "EOShift")
    assert len(nodes) == 1
    state, node = nodes[0]
    memlets = _connector_memlets(sdfg, state, node)
    assert "_x" in memlets and "_out" in memlets
    x_desc = sdfg.arrays[memlets["_x"].data]
    o_desc = sdfg.arrays[memlets["_out"].data]
    assert sympy.simplify(memlets["_x"].subset.num_elements() - x_desc.total_size) == 0
    assert sympy.simplify(memlets["_out"].subset.num_elements() - o_desc.total_size) == 0
