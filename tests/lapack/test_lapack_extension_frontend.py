"""E2E tests for the LAPACK routines beyond ``getrf``/``potrf``: ``potrs``, ``geqrf`` and ``orgqr`` have no
library node and are rejected as unsupported library calls."""

from pathlib import Path

import pytest

import dace_fortran

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "lapack_extension_probes.f90"


def _assert_unsupported(entry: str, routine: str, tmp_path):
    name = entry.split("::")[-1]
    with pytest.raises(NotImplementedError, match=routine) as exc:
        dace_fortran.build_sdfg(_SRC.read_text(), out_dir=str(tmp_path / name), entry=entry, name=name)
    assert "LAPACK" in str(exc.value)


def test_dpotrs_is_unsupported(tmp_path):
    _assert_unsupported("lapack_extension_probes::run_dpotrs", "dpotrs", tmp_path)


def test_dgeqrf_is_unsupported(tmp_path):
    _assert_unsupported("lapack_extension_probes::run_dgeqrf", "dgeqrf", tmp_path)


def test_dorgqr_is_unsupported(tmp_path):
    _assert_unsupported("lapack_extension_probes::run_dorgqr", "dorgqr", tmp_path)


if __name__ == "__main__":
    import tempfile

    for test in (test_dpotrs_is_unsupported, test_dgeqrf_is_unsupported, test_dorgqr_is_unsupported):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
