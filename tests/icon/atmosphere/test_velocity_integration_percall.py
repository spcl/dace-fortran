"""ICON integration: per-call ORIGINAL-vs-BINDING bit-exact for a dycore call site.

The ``integration`` lane's contract: duplicate the input state, run ORIGINAL ICON Fortran
and OUR SDFG binding on identical data, require bit-exact agreement (``max_diff==0``) --
the single-rank half (the multi-rank halo-exchange half carries the ``mpi`` marker).

Anchored on the atmosphere ``velocity_tendencies`` kernel, the fastest real dycore call
site. ``run_kernel_e2e`` drives the DUT (auto-generated ``bind(c)`` binding) and the REF
(stock Fortran kernel via the same shim) on one seeded input and compares every output
buffer element-for-element. Build detail shared verbatim with
:mod:`test_velocity_numerical_e2e`; only the ``integration`` marker differs."""
import shutil
from pathlib import Path

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

_HERE = Path(__file__).resolve().parent
_TU = _HERE / "velocity_advection_inlined_single_tu.f90"
_ENTRY = "mo_velocity_advection::velocity_tendencies"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


@pytest.mark.xdist_group("atmo_velocity_integration")
def test_velocity_percall_orig_vs_binding_bitexact():
    """Duplicate the velocity call's input, run ORIGINAL ICON Fortran + OUR SDFG binding on
    it, require bit-exact agreement -- the per-call orig-vs-binding contract, single-rank."""
    res = run_kernel_e2e(
        _TU,
        _ENTRY,
        int_fill=1,
        module_seeds={
            "nproma": 8,
            "nflatlev": 1,
            "nrdmax": 1
        },
        array_overrides={
            "p_patch_nlev": 7,
            "p_patch_nlevp1": 8
        },
    )
    assert res["passed"], f"velocity orig-vs-binding build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, "no output buffer changed -- the call did no work (integration check is vacuous)"
    assert res["max_diff"] == 0.0, f"binding diverged from original ICON, max|d|={res['max_diff']:.3e} (must be 0)"
