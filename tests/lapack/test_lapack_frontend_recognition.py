"""Frontend-recognition tests for LAPACK routine calls.

Each test drives one ``run_<routine>`` entry of
:file:`lapack_probes.f90` through the bridge and asserts the resulting
SDFG contains the matching :mod:`dace.libraries.lapack` library node.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "lapack_probes.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, \
        f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def test_dgetrf_recognised(tmp_path):
    _build_and_assert("lapack_probes::run_dgetrf", "Getrf", tmp_path)


def test_dpotrf_recognised(tmp_path):
    _build_and_assert("lapack_probes::run_dpotrf", "Potrf", tmp_path)
