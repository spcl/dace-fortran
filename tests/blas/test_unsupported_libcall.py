"""Verify the bridge raises a CLEAR error for an unrecognised library call.

When a Fortran call site matches a recognised library's call convention
(MPI / FFTW3 / BLAS / LAPACK) but the specific routine isn't in the
bridge's supported subset yet, the bridge MUST surface a precise
``NotImplementedError`` -- not silently degrade to a generic ``call``
lowering that mints an invalid ``_out = ?`` tasklet body.

This test pins that contract.  When the bridge gains support for the
probed routine the assertion flips to "no error" and the test must be
updated (or the probe routine swapped to a still-unsupported one).
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
