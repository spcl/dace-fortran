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

**All three kernels are BIT-EXACT (``max_diff == 0``) and PASS.**  Four fixes
landed them: (1) the ``edge2edge_viavert_coeff`` d3 offset (a struct-member
lower-bound mis-inferred as 0 from a mutated accumulator's init -- the bridge no
longer folds a target with a non-constant write); (2) section zero-fills
(``rot_vec_v(:, :, blockno) = 0``) now write only the section via the loop
broadcast instead of a whole-array memset that clobbered the caller's
out-of-domain ``intent(inout)`` data; (3) the reference shim seeds the ICON
grid-dimension module globals (``nproma`` / ``n_zlev``) that the DUT binding
derives from array extents, mirroring ICON's namelist init (else the isolated
reference reads them as 0 and sizes its automatic locals to zero -> OOB);
(4) for veloc_adv the DUT binding *also* reads ``n_zlev`` / ``nproma`` straight
from the (zero) module -- its extent-derivation can't reach them because the
only args carrying ``n_zlev`` (``psi_c`` / ``vort_v``, inlined-callee dummies)
are unsourced placeholders -- so ``module_seeds`` pins both the DUT and the
reference .so to the mesh size (``_KERNELS`` below), else ``z_vort_internal(
n_zlev)`` sizes to zero and the level reads run out of bounds into stray heap
(an intermittent divergence).

coriolis_pv and ppm_vflux run on the RANDOM ``[1, n]`` mesh (every index is in
bounds).  veloc_adv_horz forms COMPOSITE indices that exceed ``n`` on a random
mesh, so it uses the pinned in-bounds mesh (``int_fill=1``) plus the
``module_seeds`` grid dims; on it the SDFG and the reference are then identical.
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

# key, single-TU file, entry, scalar-dummy overrides, int_fill, module_seeds.
# ``int_fill`` is None for a RANDOM in-bounds ``[1, n]`` mesh (coriolis_pv /
# ppm_vflux read every index in bounds there); an int pins every connectivity /
# count / bound array to that value = a controlled in-bounds "degenerate valid
# mesh".  veloc_adv forms COMPOSITE indices (e.g. ``no_dual_edges +
# vertex_edge``) that exceed n on a random mesh -> OOB, so it needs the pinned
# mesh.  ``endindex`` = N so the column loop runs; ``vertical_limiter_type``
# picks a real PPM arm.
#
# ``module_seeds`` ({fortran_sym: value}) pins ICON grid-dimension module globals
# the DUT binding reads STRAIGHT from the module (``n_zlev = int(n_zlev__mod)``)
# rather than deriving from an array extent -- the extent-derivation path can't
# reach ``n_zlev`` here because the only args carrying it (``psi_c`` / ``vort_v``,
# inlined-callee dummies of grad_fd_norm / nonlinear_coriolis) are unsourced
# placeholders, so the bridge sources it from ``mo_ocean_nml::n_zlev`` instead.
# That global is 0 in an isolated kernel, sizing the ``z_vort_internal(n_zlev)``
# scratch to zero -> out-of-bounds level reads (an intermittent heap-garbage
# divergence).  Real ICON sets it at namelist init; the harness mirrors that by
# seeding both the DUT and reference .so to the mesh size (see
# ``_resolve_module_seeds``).  ``n = 8`` in this harness, so every extent is 8.
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
