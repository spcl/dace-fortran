"""Multi-file SDFG build for the AES graupel microphysics scheme.

``mo_aes_graupel`` (ICON AES warm/cold-cloud microphysics) drives a 4-module
build: the scheme proper (7 PURE FUNCTIONs + ``graupel_run`` driver),
``mo_aes_thermo`` (saturation/derivatives), ``mo_kind``, ``mo_physical_constants``.

Regression gate: the AoS-of-pointer-records gather temp (``t_qx_ptr%x``) used
to leave unbound extent symbols (unresolved ``?`` in the post-inline body)
until the ``fir.box_dims -> <name>_d<dim>`` extent resolution closed it.
"""
from pathlib import Path

import pytest

from _util import have_flang

from dace_fortran import build_sdfg_from_files

_HERE = Path(__file__).resolve().parent

_GRAUPEL_SOURCES = [
    _HERE / "aes_graupel" / "mo_aes_graupel.f90",
    _HERE / "aes_graupel" / "mo_aes_thermo.f90",
    _HERE / "aes_graupel" / "mo_kind.f90",
    _HERE / "aes_graupel" / "mo_physical_constants.f90",
]

# Mangled flang symbol for ``mo_aes_graupel::graupel_run``.
_ENTRY = "mo_aes_graupel::graupel_run"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_aes_graupel_multi_file_build(tmp_path):
    """The bridge ingests the 4-module aes_graupel project and emits a
    validated SDFG rooted at ``mo_aes_graupel::graupel_run``."""
    sdfg = build_sdfg_from_files(
        _GRAUPEL_SOURCES,
        entry=_ENTRY,
        name="aes_graupel",
        out_dir=tmp_path / "build",
    )
    sdfg.validate()
