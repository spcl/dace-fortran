"""End-to-end numerical correctness for the ICON atmosphere velocity kernel.

The committed ``velocity_advection_inlined_single_tu.f90`` (the inner
``velocity_tendencies`` the ``solve_nh`` dycore calls) is lowered to a DaCe
SDFG, compiled, and driven through its AUTO-GENERATED ``bind(c)`` binding; the
reference is the ORIGINAL Fortran kernel reached through the same shim
retargeted to call it directly (see :mod:`icon.ocean._ocean_e2e`).  Both run on
identical inputs and every output buffer must match **bit-exactly**
(``max_diff == 0``), with at least one output actually changed.

This is the atmosphere counterpart to :mod:`test_ocean_numerical_e2e`, and it
exercises three things the ocean kernels do not:

  * **Non-default (negative) array lower bounds.**  ICON's refinement-control
    index arrays -- ``p_patch % {verts,cells,edges} % {start,end}_{block,index}``
    -- are ``ALLOCATABLE(:)`` allocated ``(min_rl : max_rl)`` and read at
    negative refinement levels (``end_block(-10)``).  The ``bind(c)`` shim now
    carries each array's lower bound (``<arr>_lb<i>``) and rebuilds the member at
    its true bounds, and the binding derives ``offset_<arr>_d<i>`` from the
    MEMBER (not its 1-based ``c_f_pointer`` alias), so the SDFG's
    ``arr[(idx) - offset]`` indexing is in bounds.  The harness sizes these
    arrays to span the level range and gives them the negative lower bound.

  * **ICON config module globals the DUT reads straight from the module**
    (``nproma`` block size, per-domain ``nflatlev`` / ``nrdmax`` vertical
    indices).  An isolated kernel reads them as BSS 0 -> zero-sized locals /
    ``DO jk = 0, ...`` -> OOB; the harness seeds them on BOTH the DUT and the
    reference ``.so`` (``module_seeds``), mirroring ICON's namelist init.

  * **A non-degenerate vertical column** (``nlev = 7``, ``nlevp1 = 8``, both
    within the ``n = 8`` buffer): with a fully degenerate ``nlev = 1`` the
    kernel's vertical loops are empty and its automatic transients are read
    uninitialised (the reference's stack automatics diverge from the DUT's
    zero-initialised DaCe transients).  A real column runs those loops.

Bit-exactness holds because the harness pins ``-ffp-contract=off`` on both
sides (DaCe defaults to ``-ffast-math``, which would contract the rbf /
cells2verts dot products into FMAs and round ~1 ulp off the reference).
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
        # ICON namelist: block size, and per-domain vertical indices >= 1
        # (nflatlev is a DO lower bound; BSS 0 would start jk at 0 -> OOB).
        module_seeds={
            "nproma": 8,
            "nflatlev": 1,
            "nrdmax": 1
        },
        # Real vertical column within the n=8 buffers so the vertical loops run
        # and fully initialise the kernel's automatic transients.
        array_overrides={
            "p_patch_nlev": 7,
            "p_patch_nlevp1": 8
        },
    )
    assert res["passed"], f"velocity_tendencies build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"
    assert res["max_diff"] == 0.0, f"SDFG binding diverged from reference, max|d|={res['max_diff']:.3e} (must be 0)"
