"""End-to-end numerical correctness for the ICON-O ocean kernels: each committed
single-TU is lowered to an SDFG and driven through its auto-generated bind(c)
binding, compared against the original Fortran kernel via the same shim (see
_ocean_e2e). All three kernels are BIT-EXACT and pass. ``module_seeds`` pins ICON
grid-dimension module globals (nproma/n_zlev) that some kernels' extent-derivation
can't reach on its own (unsourced inlined-callee dummies), mirroring ICON's
namelist init -- else automatic locals size to zero -> OOB. ``int_fill`` pins a
degenerate in-bounds mesh for veloc_adv, whose composite indices exceed a random
mesh's bounds; coriolis_pv/ppm_vflux run on the random mesh directly.
"""
import shutil

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

_HERE = __import__("pathlib").Path(__file__).resolve().parent

pytestmark = [
    # NOT a `long` test: builds the checked-in extracted single-TU kernels
    # (no ICON-from-source), so it belongs in the fast lane.
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# key, single-TU file, entry, scalar-dummy overrides, int_fill, module_seeds.
# int_fill=None -> random in-bounds [1,n] mesh; an int pins every connectivity/
# count/bound array to a controlled degenerate valid mesh (veloc_adv needs this:
# its composite indices exceed n on a random mesh). endindex=N runs the column
# loop; vertical_limiter_type picks a real PPM arm.
# module_seeds ({fortran_sym: value}) pins ICON grid-dimension module globals some
# kernels' extent-derivation can't reach (see module docstring); n=8 here, so
# every extent is 8.
_KERNELS = [
    pytest.param("ppm_vflux",
                 "ppm_vflux_single_tu.f90",
                 "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock", {
                     "startindex": 1,
                     "endindex": 8,
                     "vertical_limiter_type": 1,
                     "dtime": 60.0
                 },
                 None, {},
                 id="ppm_vflux"),
    pytest.param("coriolis_pv",
                 "coriolis_pv_single_tu.f90",
                 "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar", {},
                 None, {},
                 id="coriolis_pv"),
    pytest.param("ocean_veloc_adv",
                 "ocean_veloc_adv_single_tu.f90",
                 "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", {},
                 1, {
                     "n_zlev": 8,
                     "nproma": 8
                 },
                 id="ocean_veloc_adv"),
]


@pytest.mark.xdist_group("ocean_fparser")
@pytest.mark.parametrize("key,fname,entry,overrides,int_fill,module_seeds", _KERNELS)
def test_ocean_kernel_numerical_e2e(key, fname, entry, overrides, int_fill, module_seeds):
    """SDFG binding output == original-kernel reference on random inputs."""
    res = run_kernel_e2e(_HERE / fname,
                         entry,
                         scalar_overrides=overrides,
                         int_fill=int_fill,
                         module_seeds=module_seeds or None)
    assert res["passed"], f"{key}: build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, f"{key}: no output buffer changed -- the kernel did no work (test is vacuous)"
    assert res["max_diff"] <= 1e-9, f"{key}: SDFG binding diverged from reference, max|d|={res['max_diff']:.3e}"
