"""End-to-end numerical correctness for the ICON atmosphere velocity kernel.

``velocity_advection_inlined_single_tu.f90``'s ``velocity_tendencies`` (the kernel
``solve_nh`` calls) is lowered to a DaCe SDFG and driven through its auto-generated
``bind(c)`` binding; the reference is the original Fortran kernel via the same shim
retargeted to call it directly.  Both run on identical inputs; every output buffer
must match bit-exactly (``max_diff == 0``), with at least one output changed.

Atmosphere counterpart to :mod:`test_ocean_numerical_e2e`; exercises three things the
ocean kernels don't:

  * **Negative array lower bounds.**  ICON's refinement-control index arrays
    (``p_patch % {verts,cells,edges} % {start,end}_{block,index}``) are
    ``ALLOCATABLE(:)`` allocated ``(min_rl : max_rl)`` and read at negative levels
    (``end_block(-10)``).  The ``bind(c)`` shim carries each array's lower bound
    (``<arr>_lb<i>``) and derives ``offset_<arr>_d<i>`` from the MEMBER (not its
    1-based ``c_f_pointer`` alias) so ``arr[(idx) - offset]`` stays in bounds.

  * **ICON config module globals** (``nproma``, per-domain ``nflatlev``/``nrdmax``)
    the DUT reads straight from the module.  An isolated kernel reads BSS 0 ->
    zero-sized locals / ``DO jk = 0, ...`` -> OOB; the harness seeds them on both
    the DUT and the reference ``.so`` (``module_seeds``), mirroring namelist init.

  * **A non-degenerate vertical column** (``nlev=7``, ``nlevp1=8`` within the
    ``n=8`` buffer): a fully degenerate ``nlev=1`` leaves the kernel's vertical
    loops empty and its automatic transients read uninitialised (reference stack
    automatics diverge from the DUT's zero-initialised DaCe transients).

Bit-exactness holds because the harness pins ``-ffp-contract=off`` on both sides
(DaCe defaults to ``-ffast-math``, which would FMA-contract the rbf / cells2verts
dot products and round ~1 ulp off the reference).
"""

import pytest

from tests.icon.atmosphere._atmo_harness import VELOCITY_TU_VARIANTS
from tests.icon.atmosphere._icon_mesh import velocity_patch
from tests.icon.ocean._ocean_e2e import _REFIN_CTRL_EXTENT, run_kernel_e2e

_HERE = __import__("pathlib").Path(__file__).resolve().parent


@pytest.mark.xdist_group("atmo_velocity_fparser")
@pytest.mark.parametrize(
    "tu_name,loop_exchange",
    VELOCITY_TU_VARIANTS,
    ids=[("loopexch" if le else "noloopexch") for _, le in VELOCITY_TU_VARIANTS],
)
def test_velocity_tendencies_numerical_e2e(tu_name, loop_exchange):
    """SDFG binding output == original-kernel reference, bit-exact, on a random 8 x 8 block patch with
    locality-biased connectivity (``_icon_mesh``) and a real vertical column.

    Run for both ``__LOOP_EXCHANGE`` variants: the define swaps the automatic transients from
    ``(nlev, nproma, nblks)`` to ``(nproma, nlev, nblks)``, so the two are different kernels and a
    layout bug shows up in only one of them.  The fixture is deliberately non-square
    (``nlev=7`` vs ``nproma=8``) so a transposed access runs out of bounds instead of aliasing."""
    patch = velocity_patch(refin_extent=_REFIN_CTRL_EXTENT)
    # nflatlev is a DO lower bound; BSS 0 would start jk at 0 -> OOB.
    patch["module_seeds"] |= {"nflatlev": 1, "nrdmax": 1}
    res = run_kernel_e2e(_HERE / tu_name, "mo_velocity_advection::velocity_tendencies", **patch)
    assert res["passed"], f"velocity_tendencies build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"
    assert res["max_diff"] == 0.0, f"SDFG binding diverged from reference, max|d|={res['max_diff']:.3e} (must be 0)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
