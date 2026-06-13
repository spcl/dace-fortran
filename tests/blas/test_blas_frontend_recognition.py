"""Frontend-recognition tests for BLAS routine calls.

Each test drives one ``run_<routine>`` entry of :file:`blas_probes.f90`
through the bridge and asserts the resulting SDFG contains the matching
:mod:`dace.libraries.blas` library node. Numerical correctness of the
lib node itself is covered by the per-node unit tests in d-face
(``tests/library/blas/blas_extensions_openblas_test.py``).
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "blas_probes.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    """Build the SDFG and assert ``expected_node`` is one of the lib-node classes."""
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, \
        f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def test_daxpy_recognised(tmp_path):
    _build_and_assert("blas_probes::run_daxpy", "Axpy", tmp_path)


def test_dscal_recognised(tmp_path):
    _build_and_assert("blas_probes::run_dscal", "Scal", tmp_path)


def test_dgemv_recognised(tmp_path):
    _build_and_assert("blas_probes::run_dgemv", "Gemv", tmp_path)


def test_dgemm_recognised(tmp_path):
    _build_and_assert("blas_probes::run_dgemm", "Gemm", tmp_path)
