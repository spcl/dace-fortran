"""E2E frontend-recognition tests for the BLAS extension lib nodes.

One test per recognised Fortran routine added in this session
(``dcopy``, ``dswap``, ``dger``, ``dtrsv``, ``dtrmv``, ``dsymv``,
``dtrsm``, ``dtrmm``, ``dsymm``, ``dsyrk``).  Each test drives the
``run_<routine>`` entry of :file:`blas_extension_probes.f90` through
the bridge and asserts the matching :mod:`dace.libraries.blas` lib
node lands in the SDFG.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "blas_extension_probes.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, \
        f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def test_dcopy_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dcopy", "Copy", tmp_path)


def test_dswap_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dswap", "Swap", tmp_path)


def test_dger_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dger", "Ger", tmp_path)


def test_dtrsv_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dtrsv", "Trsv", tmp_path)


def test_dtrmv_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dtrmv", "Trmv", tmp_path)


def test_dsymv_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dsymv", "Symv", tmp_path)


def test_dtrsm_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dtrsm", "Trsm", tmp_path)


def test_dtrmm_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dtrmm", "Trmm", tmp_path)


def test_dsymm_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dsymm", "Symm", tmp_path)


def test_dsyrk_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dsyrk", "Syrk", tmp_path)
