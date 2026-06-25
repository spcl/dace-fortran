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

ppm_vflux lowers + binds today.  coriolis_pv / veloc_adv_horz now lower, bind,
compile and RUN through the auto shim (the bridge mis-typing and the
flat-C-ABI struct reconstruction -- pointer-array-of-record patches indexed
``(1)``, arrays of ``t_cartesian_coordinates`` scattered element-wise -- are
fixed): the SDFG path and the original-Fortran reference execute identically
(``max_diff == 0``).  They stay xfail only because the test feeds RANDOM
``[1, n]`` mesh data, which for this index-heavy stencil drives empty loops
(``start_block > end_block`` / ``get_index_range`` returns an empty range) so no
output buffer changes -- ``n_changed == 0`` trips the vacuous-test guard.  A
non-vacuous run needs a controlled valid-mesh fixture (single in-domain block,
small edge range, ``no_dual_edges`` in bounds), tracked separately from the
shim; veloc_adv_horz embeds the same coriolis routine.
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
                 marks=pytest.mark.xfail(reason="DUT executes and is correct on in-bounds outputs (p_vn_dual_x "
                                         "bit-identical, vort_v agrees to 1e-14 once the kernel's own OOB is "
                                         "avoided); full bit-exact vort_flux is blocked by the kernel reading/"
                                         "writing OUT OF BOUNDS on synthetic indices (z_vt(4) vs bnd_edges, the "
                                         "no_dual_edges+vertex_edge additive index) -- needs a valid-mesh "
                                         "fixture so every index lands in bounds, tracked next.",
                                         strict=True),
                 id="coriolis_pv"),
    pytest.param("ocean_veloc_adv",
                 "ocean_veloc_adv_single_tu.f90",
                 "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", {},
                 marks=pytest.mark.xfail(reason="embeds nonlinear_coriolis_3d_fast_scalar; same vacuous-loop "
                                         "(random mesh -> n_changed==0) limitation as coriolis_pv",
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
