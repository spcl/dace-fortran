"""Frontend-recognition test for Fortran ``SPREAD``.

``fir.call @_FortranASpread(...)`` routes to a
:class:`dace.libraries.standard.nodes.broadcast.Broadcast` lib node.
"""

from pathlib import Path


import dace_fortran

_HERE = Path(__file__).resolve().parent


def test_spread_1d_to_2d_recognised(tmp_path):
    """``SPREAD(v, DIM=1, NCOPIES=3)`` -> a single Broadcast lib node, dim=1."""
    src = (_HERE / "spread_1d_to_2d_probe.f90").read_text()
    sdfg = dace_fortran.build_sdfg(
        src, out_dir=str(tmp_path / "sdfg"), entry="spread_1d_to_2d_mod::spread_1d_to_2d", name="spread_1d_to_2d"
    )
    sdfg.validate()
    nodes = [n for s in sdfg.states() for n in s.nodes() if type(n).__name__ == "Broadcast"]
    assert len(nodes) == 1, f"expected one Broadcast lib node, got {len(nodes)}"
    assert nodes[0].dim == 1


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
