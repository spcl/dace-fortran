"""Frontend-recognition tests for QE fwfft/invfft call sites: each ``run_<dir>`` entry of qe_fft_probe.f90 must lower to a single FFT (forward) or IFFT (backward) library node.

3-D-grid extraction from QE's fft_type_descriptor is out of scope -- only recognition + lowering is pinned here.
"""

from pathlib import Path

import dace_fortran

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "qe_fft_probe.f90"
_SECTION_SRC = _HERE / "qe_fft_section_probe.f90"


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


def run_section_probe(entry: str, tmp_path):
    """Build ``qe_fft_section_probe::<entry>`` and run it on a seeded 5 x 3 complex array; returns (input, output)."""
    import numpy as np

    sdfg = dace_fortran.build_sdfg(
        _SECTION_SRC.read_text(), out_dir=str(tmp_path / entry), entry=f"qe_fft_section_probe::{entry}", name=entry
    )
    rng = np.random.default_rng(0)
    n, m = 5, 3
    f_in = np.asfortranarray(rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m)))
    f = f_in.copy(order="F")
    sdfg(n=n, m=m, f=f)
    return f_in, f


def test_qe_invfft_on_columns_transforms_each_column(tmp_path):
    """``invfft_y(..., f(:, j), ...)`` transforms column j alone (inverse, 1/N), not the whole 2-D array."""
    import numpy as np

    f_in, f = run_section_probe("run_invfft_columns", tmp_path)
    np.testing.assert_allclose(f, np.fft.ifft(f_in, axis=0), rtol=1e-12, atol=1e-12)


def test_qe_fwfft_on_rows_transforms_each_strided_row(tmp_path):
    """``fwfft_y(..., f(i, :), ...)`` transforms the strided row i alone (forward, unnormalized)."""
    import numpy as np

    f_in, f = run_section_probe("run_fwfft_rows", tmp_path)
    np.testing.assert_allclose(f, np.fft.fft(f_in, axis=1), rtol=1e-12, atol=1e-12)


def test_qe_fwfft_on_allocatable_section_transforms_that_column(tmp_path):
    """``fwfft_y(..., work(:, 2, m), ...)`` on an allocatable transforms that one column: only f's last column changes."""
    import numpy as np

    f_in, f = run_section_probe("run_fwfft_allocatable_column", tmp_path)
    want = f_in.copy(order="F")
    want[:, -1] = np.fft.fft(f_in[:, -1])
    np.testing.assert_allclose(f, want, rtol=1e-12, atol=1e-12)


def test_qe_fwfft_on_remapped_pointer_transforms_its_flat_extent(tmp_path):
    """``p(1:n*jc) => f(:, 2:1+jc); fwfft_y(..., p, ...)`` with a run-time ``jc`` (QE vexx ``prhoc_d``): one flat
    transform over the n*jc elements of columns 2..1+jc; column 1 is untouched."""
    import numpy as np

    f_in, f = run_section_probe("run_fwfft_remapped_columns", tmp_path)
    want = f_in.copy(order="F")
    want[:, 1:3] = np.fft.fft(f_in[:, 1:3].reshape(-1, order="F")).reshape(-1, 2, order="F")
    np.testing.assert_allclose(f, want, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    import tempfile

    for test in (test_qe_fwfft_recognised, test_qe_invfft_recognised):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
    for test in (
        test_qe_invfft_on_columns_transforms_each_column,
        test_qe_fwfft_on_rows_transforms_each_strided_row,
        test_qe_fwfft_on_allocatable_section_transforms_that_column,
        test_qe_fwfft_on_remapped_pointer_transforms_its_flat_extent,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
