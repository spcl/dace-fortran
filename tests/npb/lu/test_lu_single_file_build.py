"""NAS Parallel Benchmark LU -- single-file SDFG build.

Companion to :mod:`test_lu_multi_file_build`.  Where that test drives the
bridge through a two-file project (``lu.F90`` + ``useapplu.F90``), this
one targets ``lu.F90`` standalone and exercises ``dolu`` directly as the
SDFG entry  --  no driver-wrapper around it.

This pins the single-source path: ``build_sdfg_from_files([lu.F90],
entry="_QMluPdolu")``.  flang's name-mangling lower-cases the module
name, so the entry symbol is ``_QMluPdolu`` (capital-M + ``lu`` module +
capital-P + ``dolu`` procedure).

Why a separate test alongside the multi-file one:

  * The multi-file variant routes the call through ``useapplu::call_dolu``
    and validates that ``merge_used_modules`` correctly resolves
    ``USE lu, ONLY: dolu`` across files (the ICON-style shape).
  * This variant validates the same source with the bridge's entry
    resolver pointed straight at the module procedure (no driver),
    which is the simpler invocation many corpus-driven tests will use.

The smoke check is identical to the multi-file test: at least one of LU's
compute kernels (``ssor`` / ``rhs`` / ``jacld`` / ``blts`` / ``buts`` /
``jacu`` / ``erhs``) must show up in the serialised SDFG so a silent
body-drop fails loudly.
"""
import json
from pathlib import Path

import pytest

from _util import have_flang

from dace_fortran import build_sdfg_from_files


_HERE = Path(__file__).resolve().parent

# Flang lowercases module names: ``MODULE lu`` -> ``_QMlu``, public procedure
# ``dolu`` -> ``P dolu``.
_ENTRY = "_QMluPdolu"

# Same compute kernels as the multi-file variant; one or more must appear in
# the serialised SDFG so a regression that silently strips lu.F90's body is
# caught.
_LU_KERNELS = ("ssor", "rhs", "jacld", "jacu", "blts", "buts", "erhs")

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.mark.long
@pytest.mark.xfail(strict=False,
                   reason=("Same as test_lu_multi_file_builds: surfaces "
                           "ssor's function-scope dynamic-size local "
                           "``tv(5*isiz1*isiz2)`` whose ``ssor_tv_d0`` / "
                           "``ssor_tv_d1`` shape symbols + ``offset_*`` "
                           "offsets aren't yet resolved by the bound-"
                           "resolution path for routine-scope locals."))
def test_lu_single_file_builds(tmp_path):
    """The bridge ingests just ``lu.F90`` and emits an SDFG rooted at
    ``lu::dolu``."""
    sdfg = build_sdfg_from_files(
        [_HERE / "lu.F90"],
        entry=_ENTRY,
        name="npb_lu_single",
        out_dir=tmp_path / "build",
    )
    sdfg.validate()
    sdfg_text = json.dumps(sdfg.to_json()).lower()
    assert any(k in sdfg_text for k in _LU_KERNELS), (
        f"built SDFG does not reference any of {_LU_KERNELS}; the "
        "single-file build likely dropped lu.F90's body.")
