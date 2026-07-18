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
import shutil

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

_HERE = __import__("pathlib").Path(__file__).resolve().parent

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


@pytest.mark.xdist_group("atmo_velocity_fparser")
def test_velocity_tendencies_numerical_e2e():
    """SDFG binding output == original-kernel reference, bit-exact, on a single
    in-domain block with a real vertical column."""
    res = run_kernel_e2e(
        _HERE / "velocity_advection_inlined_single_tu.f90",
        "mo_velocity_advection::velocity_tendencies",
        int_fill=1,  # one in-domain block; every connectivity index -> element 1
        # nflatlev is a DO lower bound; BSS 0 would start jk at 0 -> OOB.
        module_seeds={
            "nproma": 8,
            "nflatlev": 1,
            "nrdmax": 1
        },
        # real vertical column within the n=8 buffers so the vertical loops run
        # and fully initialise the kernel's automatic transients.
        array_overrides={
            "p_patch_nlev": 7,
            "p_patch_nlevp1": 8
        },
    )
    assert res["passed"], f"velocity_tendencies build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"
    assert res["max_diff"] == 0.0, f"SDFG binding diverged from reference, max|d|={res['max_diff']:.3e} (must be 0)"
