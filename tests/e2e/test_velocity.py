# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON atmosphere ``velocity_tendencies`` through the full parallelization pipeline.

Same kernel and the same fixture set as ``icon/atmosphere/test_velocity_numerical_e2e``, but with
``pipelines.optimize`` inserted between the DUT SDFG build and its compile (via
``OCEAN_E2E_SDFG_HOOK``). The reference stays the untouched gfortran kernel, so this asserts that
short-unroll + simplify + state fusion + loop2map + map fusion preserve the numerics EXACTLY. No
specialize and no scalar-fission: this kernel maps without either.

``velocity_tendencies`` is the dycore kernel ``solve_nh`` calls, and the paper's ICON data point.
See the borrowed test's docstring for what the fixture pins and why -- in particular the negative
array lower bounds, the config-module globals seeded on both sides, and the non-degenerate vertical
column without which the kernel's automatic transients stay uninitialised.
"""
import shutil
from pathlib import Path

import pytest

from _util import have_flang
from icon.atmosphere._atmo_harness import VELOCITY_TU_VARIANTS
from icon.ocean._ocean_e2e import run_kernel_e2e

_ATMO = Path(__file__).resolve().parents[1] / "icon" / "atmosphere"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


@pytest.mark.xdist_group("atmo_velocity_fparser")
@pytest.mark.parametrize("tu_name,loop_exchange",
                         VELOCITY_TU_VARIANTS,
                         ids=[("loopexch" if le else "noloopexch") for _, le in VELOCITY_TU_VARIANTS])
def test_velocity_tendencies_pipeline_numerical_e2e(monkeypatch, tu_name, loop_exchange):
    """Optimized SDFG binding output == untouched-reference output, bit-exact.

    Both ``__LOOP_EXCHANGE`` variants: the define swaps the automatic transients' data layout, so
    the pipeline sees different stride patterns and can form different maps in each."""
    # The harness forks a child that re-execs _ocean_e2e.py; the child inherits os.environ and
    # resolves the hook against tests/ on its PYTHONPATH.
    monkeypatch.setenv("OCEAN_E2E_SDFG_HOOK", "e2e._hooks:velocity_optimize")

    res = run_kernel_e2e(
        _ATMO / tu_name,
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

    assert res["passed"], f"velocity_tendencies: optimized build/lower/run failed:\n{res['output'][-3500:]}"
    assert res["n_changed"] > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"
    # max_diff of 0.0 is a PERFECT match: test against None explicitly, never for truthiness.
    assert res["max_diff"] is not None, "harness reported no diff value"
    # Bit-exact, same bar the unoptimized kernel already clears: the pipeline reorders statements
    # and forms maps but never reassociates arithmetic, and the flags pin FMA contraction off.
    assert res["max_diff"] == 0.0, (f"pipeline changed the numerics, max|d|={res['max_diff']:.3e} (must be 0)")
