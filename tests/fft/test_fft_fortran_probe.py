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

import numpy as np
import pytest

import dace_fortran

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "fft_probe_2d3d.f90"


def _build_and_assert_fft(src: str, entry: str, name: str, out_dir):
    """Build the SDFG, validate, and assert a single FFT lib node is emitted."""
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(out_dir), entry=entry, name=name)
    sdfg.validate()
    fft_nodes = [n for s in sdfg.states() for n in s.nodes() if "FFT" in type(n).__name__]
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


_PLAN_ON_BUFFER = """
  SUBROUTINE {name}(x)
    COMPLEX(C_DOUBLE_COMPLEX), INTENT(INOUT) :: x({extents})
    TYPE(C_PTR) :: plan
    plan = fftw_plan_dft_2d(3, 2, x, x, {direction}, FFTW_ESTIMATE)
    CALL fftw_execute_dft(plan, x, x)
    CALL fftw_destroy_plan(plan)
  END SUBROUTINE {name}
"""


def _probe_with(name: str, extents: str, direction: str) -> str:
    head = _SRC.read_text().split("CONTAINS")[0]
    return (
        f"{head}CONTAINS{_PLAN_ON_BUFFER.format(name=name, extents=extents, direction=direction)}END MODULE fft_probe\n"
    )


def test_fftw_backward_stays_unnormalized(tmp_path):
    """FFTW scales neither direction: backward over 6 points is 6 * ifft2."""
    src = _probe_with("run_backward", "2, 3", "FFTW_BACKWARD")
    sdfg = _build_and_assert_fft(src, "fft_probe::run_backward", "run_backward", tmp_path / "sdfg")
    start = (np.arange(6.0) + 1j * np.arange(6.0)[::-1]).reshape(2, 3, order="F")
    x = start.copy(order="F")
    sdfg(x=x)
    np.testing.assert_allclose(x, np.fft.ifft2(start) * 6, atol=1e-12)


def test_plan_rank_must_match_the_buffer(tmp_path):
    """A 2-D plan on a flat buffer has no faithful lowering yet, so it stops rather than transforming one axis."""
    src = _probe_with("run_flat", "6", "FFTW_FORWARD")
    with pytest.raises(NotImplementedError, match="rank 2"):
        dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"), entry="fft_probe::run_flat", name="run_flat")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
