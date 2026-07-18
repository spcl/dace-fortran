"""Parse + SDFG generation + numerical e2e for the ICON-O PPM vertical tracer flux
kernel ``upwind_vflux_ppm_onBlock``, extracted as a self-contained single TU
(``ppm_vflux_single_tu.f90``) -- built directly, no icon-model submodule / USE-closure
merge. Seeds random inputs and compares the SDFG against a plain-gfortran reference
through the generated Fortran binding (both sides called with the SAME ``ppmcoeffs``
derived-type struct, nine POINTER(:,:) members).
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

from dace_fortran.bindings.build_fortran_library import build_fortran_library

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = Path(__file__).parent / "ppm_vflux_single_tu.f90"
_ENTRY = "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onblock"
_NAME = "upwind_vflux_ppm"

# ppmcoeffs derived-type members in DECLARATION order (drivers point/fill them from m1..m9).
_MEMBERS = [
    "cellheightratio_this_tobelow",
    "cellheightratio_this_tothisbelow",
    "cellheight_2xbelow_x_ratiothis_tothisbelow",
    "cellheightratio_this_tothisabovebelow",
    "cellheightratio_2xaboveplusthis_tothisbelow",
    "cellheightratio_2xbelowplusthis_tothisabove",
    "cellheightratio_thisabove_to2xthisplusbelow",
    "cellheightratio_thisbelow_to2xthisplusabove",
    "cellheight_inv_thisabovebelow2below",
]


def test_ppm_vflux_single_tu_builds_and_validates(tmp_path):
    """The extracted single TU parses and lowers to a validated DaCe SDFG, with
    the ``ppmcoeffs`` derived type flattened to one array per POINTER member."""
    builder = build_sdfg(_SRC.read_text(), tmp_path / "sdfg", name=_NAME, entry=_ENTRY)
    sdfg = builder.build()
    sdfg.validate()
    for m in _MEMBERS:
        assert f"ppmcoeffs_{m}" in sdfg.arrays, f"struct member {m} not flattened"
    assert "flux_div_vert" in sdfg.arrays
    assert "tracer" in sdfg.arrays


def _drivers() -> str:
    """Two bind(c) drivers (binding wrapper vs original subroutine), each rebuilding
    ppmcoeffs from the same flat member arrays m1..m9 so both see identical contents."""
    decl_m = "\n".join(f"  real(c_double) :: m{i}(np, nz)" for i in range(1, 10))
    build_struct = "\n".join(f"  allocate(ppmcoeffs % {name}(np, nz)); ppmcoeffs % {name} = m{i}"
                             for i, name in enumerate(_MEMBERS, start=1))
    margs = ", ".join(f"m{i}" for i in range(1, 10))
    common = f"""
  integer(c_int), value :: np, nz, vlt, si, ei
  real(c_double), value :: dtime
  real(c_double) :: tracer(np, nz), w(np, nz + 1), ct(np, nz), cih(np, nz), flux(np, nz)
{decl_m}
  integer(c_int) :: nlev(np)
  type(t_verticaladvection_ppm_coefficients) :: ppmcoeffs
  nproma = np
  n_zlev = nz
{build_struct}"""
    return f"""
subroutine run_ppm_binding(np, nz, tracer, w, dtime, vlt, ct, cih, {margs}, flux, si, ei, nlev) &
    bind(c, name="run_ppm_binding")
  use iso_c_binding
  use mo_parallel_config, only: nproma
  use mo_ocean_nml, only: n_zlev
  use mo_ocean_types, only: t_verticaladvection_ppm_coefficients
  use upwind_vflux_ppm_dace_bindings, only: upwind_vflux_ppm_dace, upwind_vflux_ppm_dace_finalize
  implicit none
{common}
  call upwind_vflux_ppm_dace(tracer, w, dtime, vlt, ct, cih, ppmcoeffs, flux, si, ei, nlev, &
                             logical(.false., c_bool))
  call upwind_vflux_ppm_dace_finalize()
end subroutine run_ppm_binding

subroutine run_ppm_ref(np, nz, tracer, w, dtime, vlt, ct, cih, {margs}, flux, si, ei, nlev) &
    bind(c, name="run_ppm_ref")
  use iso_c_binding
  use mo_parallel_config, only: nproma
  use mo_ocean_nml, only: n_zlev
  use mo_ocean_types, only: t_verticaladvection_ppm_coefficients
  use mo_ocean_tracer_transport_vert, only: upwind_vflux_ppm_onblock
  implicit none
{common}
  call upwind_vflux_ppm_onblock(tracer, w, dtime, vlt, ct, cih, ppmcoeffs, flux, si, ei, nlev)
end subroutine run_ppm_ref
"""


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH")
def test_ppm_vflux_numerical_e2e_via_binding(tmp_path):
    """Random inputs through the generated binding (``ppmcoeffs`` struct) must match a
    plain-gfortran reference calling the original kernel with the same struct."""
    nproma, n_zlev = 6, 5
    rng = np.random.default_rng(0)

    def randf(shape):
        return np.asfortranarray(rng.standard_normal(shape))

    tracer = randf((nproma, n_zlev))
    w = randf((nproma, n_zlev + 1))
    ct = np.asfortranarray(rng.uniform(0.5, 1.5, (nproma, n_zlev)))  # thickness > 0
    cih = np.asfortranarray(1.0 / ct)
    members = [randf((nproma, n_zlev)) for _ in _MEMBERS]
    nlev = np.asfortranarray(np.full(nproma, n_zlev, dtype=np.int32))
    dtime, vlt, si, ei = 100.0, 1, 1, nproma

    # Build the SDFG + the Fortran binding, link in the two drivers.
    builder = build_sdfg(_SRC.read_text(), tmp_path / "sdfg", name=_NAME, entry=_ENTRY)
    sdfg = builder.build()
    sdfg.name = _NAME
    src_f90 = tmp_path / "ppm_kernel.f90"
    src_f90.write_text(_SRC.read_text())
    drv_f90 = tmp_path / "ppm_drivers.f90"
    drv_f90.write_text(_drivers())
    lib = build_fortran_library(sdfg,
                                None,
                                None,
                                str(tmp_path / "lib"),
                                name=_NAME,
                                prelude_sources=[src_f90],
                                extra_sources=[drv_f90])
    dl = lib.load()

    def _call(symbol):
        flux = np.zeros((nproma, n_zlev), dtype=np.float64, order="F")
        cd = lambda a: a.ctypes.data_as(ctypes.c_void_p)
        args = [
            ctypes.c_int(nproma),
            ctypes.c_int(n_zlev),
            cd(tracer),
            cd(w),
            ctypes.c_double(dtime),
            ctypes.c_int(vlt),
            cd(ct),
            cd(cih)
        ]
        args += [cd(m) for m in members]
        args += [cd(flux), ctypes.c_int(si), ctypes.c_int(ei), cd(nlev)]
        fn = getattr(dl, symbol)
        fn.restype = None
        fn(*args)
        return flux

    flux_binding = _call("run_ppm_binding")
    flux_ref = _call("run_ppm_ref")
    assert np.all(np.isfinite(flux_ref)), "reference produced non-finite values"
    np.testing.assert_allclose(flux_binding, flux_ref, rtol=1e-10, atol=1e-12)
