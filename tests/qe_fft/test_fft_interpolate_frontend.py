"""E2E frontend-recognition tests for QE's ``fft_interpolate_*`` calls.

Each test drives one ``run_fft_interpolate_<dtype>`` entry of
:file:`fft_interpolate_probe.f90` through the bridge and asserts the
resulting SDFG contains a single
:class:`dace.libraries.fft.nodes.FFTInterpolate` lib node.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "fft_interpolate_probe.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert(entry: str, expected_kind: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    matches = [n for s in sdfg.states() for n in s.nodes()
               if type(n).__name__ == "FFTInterpolate"]
    assert matches, "no FFTInterpolate lib node emitted"
    assert matches[0].dtype_kind == expected_kind, \
        f"expected dtype_kind={expected_kind!r}, got {matches[0].dtype_kind!r}"


def test_fft_interpolate_complex_recognised(tmp_path):
    _build_and_assert("fft_interpolate_probe::run_fft_interpolate_complex",
                      "complex", tmp_path)


def test_fft_interpolate_real_recognised(tmp_path):
    _build_and_assert("fft_interpolate_probe::run_fft_interpolate_real",
                      "real", tmp_path)
