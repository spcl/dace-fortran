"""NPB3.4 LU: single-file SDFG build (lu.F90 only, entry=lu::dolu, no driver wrapper).
Companion to test_lu_multi_file_build; same kernel-presence smoke check.
"""
import json
from pathlib import Path

import pytest

from _util import have_flang

from dace_fortran import build_sdfg_from_files

_HERE = Path(__file__).resolve().parent

# module::procedure name; resolves to flang's mangled _QMluPdolu.
_ENTRY = "lu::dolu"

# Kernels that must appear in the SDFG, else the build silently dropped lu.F90's body.
_LU_KERNELS = ("ssor", "rhs", "jacld", "jacu", "blts", "buts", "erhs")

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_lu_single_file_builds(tmp_path):
    """Ingesting just lu.F90 emits a valid SDFG rooted at lu::dolu."""
    sdfg = build_sdfg_from_files(
        [_HERE / "lu.F90"],
        entry=_ENTRY,
        name="npb_lu_single",
        out_dir=tmp_path / "build",
    )
    sdfg.validate()
    sdfg_text = json.dumps(sdfg.to_json()).lower()
    assert any(k in sdfg_text for k in _LU_KERNELS), (f"built SDFG does not reference any of {_LU_KERNELS}; the "
                                                      "single-file build likely dropped lu.F90's body.")
