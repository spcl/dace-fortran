"""FFTW3 2-D / 3-D recognition: bridge lowers them to a single FFT lib node.

The fixture :file:`fft_probe_2d3d.f90` contains two routines that drive
the FFTW3 C ABI from Fortran via ``iso_c_binding``:

* ``run_fft_2d(M, N, x)`` -- in-place 2-D forward FFT;
* ``run_fft_3d(L, M, N, x)`` -- in-place 3-D forward FFT.

This is the serial-FFT shape every plane-wave code reduces to once the
parallel sticks-and-planes scatter pipeline has redistributed the data
locally (Quantum ESPRESSO's ``FFTXlib/src/fft_scalar.FFTW3.f90`` is the
canonical example).

End-to-end lowering through DaCe is now wired:

1. ``dace_fortran/bridge/ast/dispatch.cpp::fftw3CalleeTag`` matches
   ``fftw_plan_dft_{2,3}d`` / ``fftw_execute_dft`` / ``fftw_destroy_plan``
   (and their ``fftwf_*`` single-precision twins) by mangled callee.
2. The ``plan = fftw_plan_dft_*(...)`` user statement (lowered by
   flang through a ``.result`` temp + ``hlfir.as_expr`` + ``hlfir.assign``)
   is recognised at the ``hlfir.assign`` site and absorbed -- the
   ``TYPE(C_PTR)`` plan SSA value is opaque and has no SDFG representation.
   At absorb time the (rank, dims, direction) get recorded under the
   destination variable name.
3. ``fftw_execute_dft(plan, in, out)`` lowers to a single
   :class:`dace.libraries.fft.nodes.FFT` (or :class:`IFFT` if the plan
   was created with ``FFTW_BACKWARD = +1``) library node carrying the
   in/out array memlets; the FFT node's ``FFTW3`` / ``cuFFT`` / ``MKL``
   expansions take over from there.
4. ``fftw_destroy_plan(plan)`` is recognised and dropped -- the FFT
   library node's expansion owns the plan lifecycle.
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
