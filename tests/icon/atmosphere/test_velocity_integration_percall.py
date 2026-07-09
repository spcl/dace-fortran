"""ICON integration: per-call ORIGINAL-vs-BINDING bit-exact for a dycore call site.

The ``integration`` lane's charter is the ICON orig-vs-binding contract stated
per dycore *call site*: for each call, duplicate the input state, run the
ORIGINAL ICON Fortran once and OUR SDFG binding once on that identical data, and
require bit-exact agreement (``max_diff == 0``, no tolerance) before ICON would
continue.  This is the SHORT, single-rank half of that contract (the multi-rank
halo-exchange half carries the ``mpi`` marker and runs under ``mpirun``).

The atmosphere ``velocity_tendencies`` kernel is the fastest real dycore call
site, so it anchors the lane.  ``run_kernel_e2e`` already IS the per-call
duplicate-run engine: it drives the DUT (the auto-generated ``bind(c)`` binding
of the ``velocity_advection_inlined_single_tu.f90`` SDFG) and the REF (the stock
Fortran kernel reached through the same shim retargeted at it) on one identical
seeded input and compares every output buffer element-for-element.  The build
detail (negative refinement-control lower bounds, ``module_seeds`` config
globals, a real vertical column, ``-ffp-contract=off`` on both sides) is shared
verbatim with :mod:`test_velocity_numerical_e2e`; the difference here is only
the ``integration`` marker -- this test exists to give that lane real coverage
and to pin the orig-vs-binding invariant as a first-class ICON-integration gate.
"""
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
    """Duplicate the velocity call's input, run ORIGINAL ICON Fortran + OUR SDFG
    binding on it, and require bit-exact agreement -- the per-call ICON
    orig-vs-binding integration contract, single-rank."""
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
