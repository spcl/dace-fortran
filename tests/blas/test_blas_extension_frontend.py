"""E2E frontend-recognition tests for the BLAS extension routines.

One test per routine; drives ``run_<routine>`` through the bridge and asserts the matching
:mod:`dace.libraries.blas` lib node lands in the SDFG. ``dcopy`` and ``dswap`` have no BLAS node and
lower to a standard copy node and a map, so they run against numpy. The triangular routines have no
library node at all and are rejected as unsupported library calls.
"""
from pathlib import Path

import numpy as np
import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "blas_extension_probes.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, \
        f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def _assert_unsupported(entry: str, routine: str, tmp_path):
    name = entry.split("::")[-1]
    with pytest.raises(NotImplementedError, match=routine) as exc:
        dace_fortran.build_sdfg(_SRC.read_text(), out_dir=str(tmp_path / name), entry=entry, name=name)
    assert "BLAS" in str(exc.value)


def test_dcopy_lowers_to_a_copy_node(tmp_path):
    _build_and_assert("blas_extension_probes::run_dcopy", "CopyLibraryNode", tmp_path)


def test_dcopy_numerical(tmp_path):
    sdfg = dace_fortran.build_sdfg(_SRC.read_text(),
                                   out_dir=str(tmp_path / "dcopy"),
                                   entry="blas_extension_probes::run_dcopy",
                                   name="run_dcopy")
    x = np.random.default_rng(0).standard_normal(9)
    y = np.zeros(9)
    sdfg(n=np.int32(9), x=x, y=y)
    np.testing.assert_array_equal(y, x)


def test_dswap_lowers_to_a_map(tmp_path):
    _build_and_assert("blas_extension_probes::run_dswap", "MapEntry", tmp_path)


def test_dswap_numerical(tmp_path):
    sdfg = dace_fortran.build_sdfg(_SRC.read_text(),
                                   out_dir=str(tmp_path / "dswap"),
                                   entry="blas_extension_probes::run_dswap",
                                   name="run_dswap")
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(9), rng.standard_normal(9)
    x0, y0 = x.copy(), y.copy()
    sdfg(n=np.int32(9), x=x, y=y)
    np.testing.assert_array_equal(x, y0)
    np.testing.assert_array_equal(y, x0)


def test_dger_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dger", "Ger", tmp_path)


def test_dtrsv_is_unsupported(tmp_path):
    _assert_unsupported("blas_extension_probes::run_dtrsv", "dtrsv", tmp_path)


def test_dtrmv_is_unsupported(tmp_path):
    _assert_unsupported("blas_extension_probes::run_dtrmv", "dtrmv", tmp_path)


def test_dsymv_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dsymv", "Symv", tmp_path)


def test_dtrsm_is_unsupported(tmp_path):
    _assert_unsupported("blas_extension_probes::run_dtrsm", "dtrsm", tmp_path)


def test_dtrmm_is_unsupported(tmp_path):
    _assert_unsupported("blas_extension_probes::run_dtrmm", "dtrmm", tmp_path)


def test_dsymm_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dsymm", "Symm", tmp_path)


def test_dsyrk_recognised(tmp_path):
    _build_and_assert("blas_extension_probes::run_dsyrk", "Syrk", tmp_path)


if __name__ == "__main__":
    import tempfile

    for test in (test_dcopy_lowers_to_a_copy_node, test_dcopy_numerical, test_dswap_lowers_to_a_map,
                 test_dswap_numerical, test_dger_recognised, test_dtrsv_is_unsupported, test_dtrmv_is_unsupported,
                 test_dsymv_recognised, test_dtrsm_is_unsupported, test_dtrmm_is_unsupported, test_dsymm_recognised,
                 test_dsyrk_recognised):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
