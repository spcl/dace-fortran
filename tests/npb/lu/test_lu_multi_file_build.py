"""NPB3.4 LU: multi-file SDFG build (lu.F90 + useapplu.F90) via build_sdfg_from_files.
Must produce a valid SDFG under both merge engines (fparser, regex).
"""
import json
from pathlib import Path

import pytest

from _util import have_flang

from dace_fortran import build_sdfg_from_files

_HERE = Path(__file__).resolve().parent

# module::procedure name; bridge resolves to flang's mangled _QMuseappluPcall_dolu.
_ENTRY = "useapplu::call_dolu"

_LU_SOURCES = [_HERE / "lu.F90", _HERE / "useapplu.F90"]

# Kernels dolu sequences; if none survive the merge, lu.F90's body was silently dropped.
_LU_KERNELS = ("ssor", "rhs", "jacld", "jacu", "blts", "buts", "erhs")

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.mark.parametrize("merge_engine", ["fparser", "regex"])
def test_lu_multi_file_builds(tmp_path, merge_engine):
    """Ingesting [lu.F90, useapplu.F90] emits a valid SDFG rooted at useapplu::call_dolu, both merge engines."""
    sdfg = build_sdfg_from_files(
        _LU_SOURCES,
        entry=_ENTRY,
        name="npb_lu",
        out_dir=tmp_path / "build",
        merge_engine=merge_engine,
    )
    sdfg.validate()
    # At least one LU kernel must appear in the serialized SDFG, else the merge silently dropped lu.F90's body.
    sdfg_text = json.dumps(sdfg.to_json()).lower()
    assert any(k in sdfg_text for k in _LU_KERNELS), (f"built SDFG does not reference any of {_LU_KERNELS}; the "
                                                      "multi-file merge likely dropped lu.F90's body.")
