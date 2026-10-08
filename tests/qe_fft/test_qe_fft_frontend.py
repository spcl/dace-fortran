"""Frontend-recognition tests for QE fwfft/invfft call sites: each ``run_<dir>`` entry of qe_fft_probe.f90 must lower to a single FFT (forward) or IFFT (backward) library node.

3-D-grid extraction from QE's fft_type_descriptor is out of scope -- only recognition + lowering is pinned here.
"""

from pathlib import Path

import dace_fortran

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "qe_fft_probe.f90"


def _build_and_assert(entry: str, expected_node: str, tmp_path):
    src = _SRC.read_text()
    name = entry.split("::")[-1]
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / name), entry=entry, name=name)
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert expected_node in classes, f"{entry}: expected a {expected_node!r} lib node, got {sorted(classes)!r}"


def test_qe_fwfft_recognised(tmp_path):
    """``fwfft_y`` call site lowers to a forward :class:`FFT` lib node."""
    _build_and_assert("qe_fft_probe::run_fwfft", "FFT", tmp_path)


def test_qe_invfft_recognised(tmp_path):
    """``invfft_y`` call site lowers to an :class:`IFFT` lib node."""
    _build_and_assert("qe_fft_probe::run_invfft", "IFFT", tmp_path)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
