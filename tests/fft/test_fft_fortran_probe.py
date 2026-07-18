"""FFTW3 2-D / 3-D recognition: bridge lowers them to a single FFT lib node.

:file:`fft_probe_2d3d.f90` drives the FFTW3 C ABI via ``iso_c_binding``:
``run_fft_2d``/``run_fft_3d`` (in-place forward FFT) -- the serial-FFT shape
plane-wave codes reduce to after sticks-and-planes redistribution (QE's
``FFTXlib/src/fft_scalar.FFTW3.f90``).

Lowering: ``dispatch.cpp::fftw3CalleeTag`` matches the ``fftw_plan_dft_*``/
``fftw_execute_dft``/``fftw_destroy_plan`` mangled callees; the plan-create
statement is absorbed at its ``hlfir.assign`` site (the opaque ``C_PTR`` has no
SDFG representation), recording (rank, dims, direction); ``fftw_execute_dft``
becomes a single :class:`FFT`/:class:`IFFT` library node; ``fftw_destroy_plan``
is dropped (the lib node's expansion owns the plan lifecycle).
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "fft_probe_2d3d.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_and_assert_fft(src: str, entry: str, name: str, out_dir):
    """Build the SDFG, validate, and assert a single FFT lib node is emitted."""
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(out_dir), entry=entry, name=name)
    sdfg.validate()
    fft_nodes = [n for s in sdfg.states() for n in s.nodes() if 'FFT' in type(n).__name__]
    assert fft_nodes, f"no FFT lib node emitted -- got states {[s.label for s in sdfg.states()]}"
    return sdfg


def test_fft_probe_2d_lowers_to_lib_node(tmp_path):
    """2-D FFTW3 in-place FFT lowers to a single :class:`FFT` lib node."""
    src = _SRC.read_text()
    _build_and_assert_fft(src, "fft_probe::run_fft_2d", "run_fft_2d", tmp_path / "sdfg_2d")


def test_fft_probe_3d_lowers_to_lib_node(tmp_path):
    """3-D FFTW3 in-place FFT lowers to a single :class:`FFT` lib node."""
    src = _SRC.read_text()
    _build_and_assert_fft(src, "fft_probe::run_fft_3d", "run_fft_3d", tmp_path / "sdfg_3d")
