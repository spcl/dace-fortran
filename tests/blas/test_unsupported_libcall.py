"""An unrecognised routine within a known library convention (MPI/FFTW3/BLAS/LAPACK)
must raise ``NotImplementedError``, not silently degrade to an invalid ``_out = ?``
tasklet. When the bridge gains support for the probed routine, swap it for a still-
unsupported one.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "unsupported_blas_probe.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_drot_raises_clear_error(tmp_path):
    """Unsupported BLAS routine yields a precise NotImplementedError."""
    src = _SRC.read_text()
    with pytest.raises(NotImplementedError) as exc:
        dace_fortran.build_sdfg(src,
                                out_dir=str(tmp_path / "sdfg"),
                                entry="unsupported_blas_probe::run_drot",
                                name="run_drot")
    msg = str(exc.value)
    assert "drot" in msg.lower(), f"error should mention the routine name: {msg!r}"
    assert "blas" in msg.lower(), f"error should identify the library family: {msg!r}"
