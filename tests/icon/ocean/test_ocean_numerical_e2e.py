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

**ppm_vflux and coriolis_pv are BIT-EXACT (``max_diff == 0``) and PASS.**  Three
fixes landed the coriolis class: (1) the ``edge2edge_viavert_coeff`` d3 offset
(a struct-member lower-bound mis-inferred as 0 from a mutated accumulator's init
-- ``traceConstIntThroughLoad`` no longer folds a target with a non-constant
write); (2) section zero-fills (``rot_vec_v(:, :, blockno) = 0``) now write only
the section via the loop broadcast instead of a whole-array memset that clobbered
the caller's out-of-domain ``intent(inout)`` data; (3) the reference shim seeds
the ICON grid-dimension module globals (``nproma`` / ``n_zlev``) from the C-ABI
extents, mirroring ICON's namelist init (else the isolated reference reads them
as 0 and sizes its automatic locals to zero -> OOB).  No ``int_fill`` needed:
random in-bounds ``[1, n]`` indices are read identically on both sides.

veloc_adv_horz still xfails -- it embeds the (now-correct) coriolis routine but
adds its own velocity-advection connectivity that reads out of bounds on the
random synthetic mesh (a remaining veloc_adv-specific lowering bug); tracked in
``project_ocean_e2e_uninit_module_dims_rootcause``.
"""
import shutil

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

_HERE = __import__("pathlib").Path(__file__).resolve().parent

pytestmark = [
    # NOT a ``long`` test: these build the checked-in extracted single-TU
    # kernels (no ICON-from-source / submodule), so they belong in the fast
    # lane next to the other self-contained single-TU e2e correctness tests.
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
                 id="coriolis_pv"),
    pytest.param("ocean_veloc_adv",
                 "ocean_veloc_adv_single_tu.f90",
                 "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", {},
                 marks=pytest.mark.xfail(reason="coriolis_pv + ppm_vflux are now bit-exact (edge2edge d3 offset + "
                                         "section-zero-fill bridge fixes + reference module-dim init).  veloc_adv "
                                         "embeds nonlinear_coriolis_3d_fast_scalar AND adds its own velocity-advection "
                                         "connectivity, which still reads OUT OF BOUNDS on the random synthetic mesh "
                                         "(max|d|~2e183 overflow); a degenerate in-bounds fixture (int_fill=1) avoids "
                                         "the OOB but leaves a residual ~1.72 -> a remaining lowering bug specific to "
                                         "veloc_adv, tracked in project_ocean_e2e_uninit_module_dims_rootcause.",
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
