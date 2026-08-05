# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON ocean velocity-family kernels through the full parallelization pipeline.

Same three kernels and the same fixture set as ``icon/ocean/test_ocean_numerical_e2e``, but with
``pipelines.optimize`` inserted between the DUT SDFG build and its compile (via
``OCEAN_E2E_SDFG_HOOK``). The reference stays the untouched gfortran kernel, so this asserts that
short-unroll + simplify + state fusion + loop2map + map fusion preserve the numerics EXACTLY. No
specialize and no scalar-fission: these kernels map without either.
"""
import shutil

import pytest

from _util import have_flang
from icon.ocean._ocean_e2e import run_kernel_e2e

_OCEAN = __import__("pathlib").Path(__file__).resolve().parents[1] / "icon" / "ocean"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Mirrors _KERNELS in icon/ocean/test_ocean_numerical_e2e -- see that module's docstring for what
# int_fill and module_seeds pin and why.
_KERNELS = [
    pytest.param("ppm_vflux",
                 "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock", {
                     "startindex": 1,
                     "endindex": 8,
                     "vertical_limiter_type": 1,
                     "dtime": 60.0
                 },
                 None, {},
                 id="ppm_vflux"),
    pytest.param("coriolis_pv", "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar", {}, None, {}, id="coriolis_pv"),
    pytest.param("ocean_veloc_adv",
                 "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", {},
                 1, {
                     "n_zlev": 8,
                     "nproma": 8
                 },
                 id="ocean_veloc_adv"),
]


@pytest.mark.xdist_group("ocean_fparser")
@pytest.mark.parametrize("key,entry,overrides,int_fill,module_seeds", _KERNELS)
def test_velocity_pipeline_numerical_e2e(key, entry, overrides, int_fill, module_seeds, monkeypatch):
    """Optimized SDFG binding output == untouched-reference output, to fp64 precision."""
    # The harness forks a child that re-execs _ocean_e2e.py; the child inherits os.environ and
    # resolves the hook against tests/ on its PYTHONPATH.
    monkeypatch.setenv("OCEAN_E2E_SDFG_HOOK", "e2e._hooks:velocity_optimize")

    res = run_kernel_e2e(_OCEAN / f"{key}_single_tu.f90",
                         entry,
                         scalar_overrides=overrides,
                         int_fill=int_fill,
                         module_seeds=module_seeds or None)

    assert res["passed"], f"{key}: optimized build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, f"{key}: no output buffer changed -- the kernel did no work (test is vacuous)"
    # max_diff of 0.0 is a PERFECT match: test against None explicitly, never for truthiness.
    assert res["max_diff"] is not None, f"{key}: harness reported no diff value"
    assert res["max_diff"] <= 1e-11, (f"{key}: pipeline changed the numerics, max|d|={res['max_diff']:.3e}")
