"""E2E frontend-recognition tests for the LAPACK extension lib nodes
(:class:`Potrs`, :class:`Geqrf`, :class:`Orgqr`)."""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "lapack_extension_probes.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, \
        f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def test_dpotrs_recognised(tmp_path):
    _build_and_assert("lapack_extension_probes::run_dpotrs", "Potrs", tmp_path)


def test_dgeqrf_recognised(tmp_path):
    _build_and_assert("lapack_extension_probes::run_dgeqrf", "Geqrf", tmp_path)


def test_dorgqr_recognised(tmp_path):
    _build_and_assert("lapack_extension_probes::run_dorgqr", "Orgqr", tmp_path)
