"""End-to-end numerical correctness for the ICON-O ocean kernels.

Each kernel's committed single-TU is lowered to a DaCe SDFG, compiled, and
driven through its AUTO-GENERATED ``bind(c)`` Fortran binding (the e2e binding
the deployment uses -- NOT a Python-direct SDFG call).  The reference is the
ORIGINAL Fortran kernel reached through the same shim retargeted to call it
directly (see :mod:`_ocean_e2e`).  Both run on identical random inputs and
every output buffer must match bit-closely AND at least one output must have
actually changed.

This exercises the full ocean binding path including the shim
module-variable-extent forwarding fix (``tracer(nproma, n_zlev)`` ->
``c_f_pointer`` resolves) and the struct-member AoS->SoA marshalling
(``t_verticaladvection_ppm_coefficients``'s nine pointer members).

ppm_vflux lowers + binds today.  coriolis_pv / veloc_adv_horz currently fail to
LOWER (a bridge HLFIR pass mis-types the ``edgeOfVertex_index`` indirect-index
scalar: ``'hlfir.declare' op first result type is inconsistent ... expected
'!fir.ref<i32>'``) -- a bridge bug, not a binding gap -- so they are xfail until
that lands; veloc_adv_horz embeds the same coriolis routine, hence the same
root cause.
"""
import shutil

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

_HERE = __import__("pathlib").Path(__file__).resolve().parent

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# key, single-TU file, entry, per-kernel scalar-dummy overrides (endindex = N
# so the column loop actually runs; vertical_limiter_type selects a real PPM
# limiter arm; dtime a physical step).  N is the harness default (8).
_KERNELS = [
    pytest.param("ppm_vflux",
                 "ppm_vflux_single_tu.f90",
                 "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock", {
                     "startindex": 1,
                     "endindex": 8,
                     "vertical_limiter_type": 1,
                     "dtime": 60.0,
                 },
                 id="ppm_vflux"),
    pytest.param("coriolis_pv",
                 "coriolis_pv_single_tu.f90",
                 "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar", {},
                 marks=pytest.mark.xfail(reason="hlfir-flatten-structs mis-types the i32 scalar "
                                         "edgeOfVertex_index (read from the flattened allocatable member "
                                         "patch%verts%edge_idx(...)): 'hlfir.declare op first result type "
                                         "inconsistent ... expected !fir.ref<i32>'.  Bridge pass bug, not "
                                         "a binding gap; needs a FlattenStructs.cpp fix + rebuild.",
                                         strict=True),
                 id="coriolis_pv"),
    pytest.param("ocean_veloc_adv",
                 "ocean_veloc_adv_single_tu.f90",
                 "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", {},
                 marks=pytest.mark.xfail(reason="embeds nonlinear_coriolis_3d_fast_scalar; same "
                                         "hlfir-flatten-structs edgeOfVertex_index mis-typing bug",
                                         strict=True),
                 id="ocean_veloc_adv"),
]


@pytest.mark.xdist_group("ocean_fparser")
@pytest.mark.parametrize("key,fname,entry,overrides", _KERNELS)
def test_ocean_kernel_numerical_e2e(key, fname, entry, overrides):
    """SDFG binding output == original-kernel reference on random inputs."""
    res = run_kernel_e2e(_HERE / fname, entry, scalar_overrides=overrides)
    assert res["passed"], f"{key}: build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, f"{key}: no output buffer changed -- the kernel did no work (test is vacuous)"
    assert res["max_diff"] <= 1e-9, f"{key}: SDFG binding diverged from reference, max|d|={res['max_diff']:.3e}"
