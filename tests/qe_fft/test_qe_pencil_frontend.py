"""E2E frontend-recognition tests for QE's parallel pencil-pipeline routines.

Drives one entry of :file:`qe_pencil_probe.f90` per routine and asserts the SDFG carries the
matching lib node: ``cft_1x``/``cft_1y``/``cft_1z`` -> :class:`FFT`; ``fft_scatter_xy``/
``fft_scatter_yz`` -> :class:`Alltoall`.

Recognition only -- the buffer-to-3-D-grid reinterpretation for a full parallel 3-D FFT SDFG
is a separate gap (``axis`` tag carried on the ASTNode for ``emit_fft`` to consume later).
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "qe_pencil_probe.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, \
        f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def test_cft_1z_recognised(tmp_path):
    _build_and_assert("qe_pencil_probe::run_cft_1z", "FFT", tmp_path)


def test_cft_1y_recognised(tmp_path):
    _build_and_assert("qe_pencil_probe::run_cft_1y", "FFT", tmp_path)


def test_cft_1x_recognised(tmp_path):
    _build_and_assert("qe_pencil_probe::run_cft_1x", "FFT", tmp_path)


def test_fft_scatter_xy_recognised(tmp_path):
    _build_and_assert("qe_pencil_probe::run_fft_scatter_xy", "Alltoall", tmp_path)


def test_fft_scatter_yz_recognised(tmp_path):
    _build_and_assert("qe_pencil_probe::run_fft_scatter_yz", "Alltoall", tmp_path)
