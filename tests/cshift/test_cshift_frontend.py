"""Frontend-recognition + memlet-subset tests for Fortran ``CSHIFT``: each test asserts
the SDFG carries the right :class:`CShift` lib node with memlets covering the full shift
dimension.  Tests stop at SDFG-build time -- the lib node's pure expansion stub
(``NotImplementedError``) never runs."""
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
    """Return ``(state, node)`` tuples for every lib node of ``class_name``."""
    return [(s, n) for s in sdfg.states() for n in s.nodes() if type(n).__name__ == class_name]


def _connector_memlets(sdfg, state, node):
    """Return ``{conn_name: memlet}`` for every in/out connector on ``node``."""
    out = {}
    for e in state.in_edges(node):
        out[e.dst_conn] = e.data
    for e in state.out_edges(node):
        out[e.src_conn] = e.data
    return out


def test_cshift_1d_const_shift_recognised(tmp_path):
    """``out = CSHIFT(arr, 1)`` -> CShift with dim=1, shift=1."""
    sdfg = _build("cshift_1d_const_shift_probe.f90", "cshift_1d_const", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    _, n = nodes[0]
    assert n.dim == 1
    assert str(n.shift) == "1"


def test_cshift_1d_var_shift_recognised(tmp_path):
    """``out = CSHIFT(arr, shift)`` -> CShift with shift = symbolic."""
    sdfg = _build("cshift_1d_var_shift_probe.f90", "cshift_1d_var", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    _, n = nodes[0]
    assert n.shift is not None


def test_cshift_2d_dim2_recognised(tmp_path):
    """``out = CSHIFT(arr, 1, dim=2)`` -> CShift with dim=2."""
    sdfg = _build("cshift_2d_dim2_probe.f90", "cshift_2d_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    _, n = nodes[0]
    assert n.dim == 2


def test_cshift_2d_dim1_recognised(tmp_path):
    """``out = CSHIFT(arr, 2, dim=1)`` -> CShift with dim=1."""
    sdfg = _build("cshift_2d_dim1_probe.f90", "cshift_2d_dim1", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    _, n = nodes[0]
    assert n.dim == 1


def test_cshift_3d_dim2_recognised(tmp_path):
    """``out = CSHIFT(arr, -1, dim=2)`` -> CShift with dim=2 over rank-3 source."""
    sdfg = _build("cshift_3d_dim2_probe.f90", "cshift_3d_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    _, n = nodes[0]
    assert n.dim == 2


def test_cshift_memlets_cover_full_arrays(tmp_path):
    """CShift connectors receive whole-array memlets, not single-element subsets --
    a single-element subset would defeat the lib node's eventual expansion (needs the
    full axis to compute the wrapped reads)."""
    sdfg = _build("cshift_2d_dim2_probe.f90", "cshift_2d_dim2", tmp_path)
    nodes = _find_lib_nodes(sdfg, "CShift")
    assert len(nodes) == 1
    state, node = nodes[0]
    import sympy
    memlets = _connector_memlets(sdfg, state, node)
    assert "_x" in memlets and "_out" in memlets
    # subset volume == full descriptor volume; sympy.simplify needed since direct ``==``
    # is structural and would reject e.g. ``n*(m-1)+n`` vs ``m*n``.
    x_desc = sdfg.arrays[memlets["_x"].data]
    o_desc = sdfg.arrays[memlets["_out"].data]
    assert sympy.simplify(memlets["_x"].subset.num_elements() - x_desc.total_size) == 0
    assert sympy.simplify(memlets["_out"].subset.num_elements() - o_desc.total_size) == 0
