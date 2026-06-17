"""Multi-file SDFG build for the AES graupel microphysics scheme.

``mo_aes_graupel`` is the warm/cold-cloud microphysics parameterisation
from ICON's AES (Atmospheric Earth System) physics package.  This test
drives its 4-module project through the bridge as a multi-file build:

  * ``mo_aes_graupel.f90``       -- the scheme proper (1521 LoC,
                                    7 PURE FUNCTIONs invoked from the
                                    main ``graupel_run`` driver).
  * ``mo_aes_thermo.f90``        -- thermodynamic helper functions
                                    (saturation, derivatives).
  * ``mo_kind.f90``              -- working-precision kinds.
  * ``mo_physical_constants.f90``-- gas-law / latent-heat / etc.

Status: green.  The pipeline used to surface an unresolved ``?``
placeholder in the post-inline body of ``graupel_run`` -- the
AoS-of-pointer-records gather temp (``t_qx_ptr%x``), whose inner extents
are recovered via ``fir.box_dims``, was left with unbound extent symbols.
The ``fir.box_dims -> <name>_d<dim>`` extent resolution closed it; this
build test is the fast regression gate for anyone touching the graupel
pipeline.
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


@pytest.mark.long
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
