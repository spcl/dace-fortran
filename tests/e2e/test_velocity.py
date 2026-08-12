# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON atmosphere ``velocity_tendencies`` through the full parallelization pipeline.

Same kernel and the same fixture set as ``icon/atmosphere/test_velocity_numerical_e2e``, but the
DUT SDFG goes through ``pipelines.optimize`` before it is compiled (via ``OCEAN_E2E_SDFG_HOOK``).

Each variant is run TWICE on one seeded random input set -- once with the pipeline and once
without -- and both are compared against the same untouched gfortran kernel. Both must be
bit-exact, which pins the pre- vs post-optimize differential exactly (equality at ``max_diff ==
0.0`` is transitive): short-unroll + unique-loop-iterators + scalar-fission + simplify + state
fusion + loop2map + map fusion preserve the numerics. No specialize: this kernel maps without it.

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

# One RNG stream per variant, never shared. ``__LOOP_EXCHANGE`` swaps the automatic transients'
# data layout, so the two variants are different kernels that happen to share a dummy-argument ABI;
# driving both from one stream would let a layout-sensitive bug hide behind inputs tuned to the
# other variant.
_SEEDS = {True: 1001, False: 1002}

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


def run_variant(seed: int, tu_name: str):
    """One harness run of ``tu_name`` on the ``seed`` input set, against the gfortran reference."""
    return run_kernel_e2e(
        _ATMO / tu_name,
        "mo_velocity_advection::velocity_tendencies",
        seed=seed,
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


@pytest.mark.xdist_group("atmo_velocity_fparser")
@pytest.mark.parametrize("tu_name,loop_exchange",
                         VELOCITY_TU_VARIANTS,
                         ids=[("loopexch" if le else "noloopexch") for _, le in VELOCITY_TU_VARIANTS])
def test_velocity_tendencies_pipeline_numerical_e2e(monkeypatch, tu_name, loop_exchange):
    """Pre-optimize and post-optimize bindings both match the reference bit-exactly.

    Both ``__LOOP_EXCHANGE`` variants: the define swaps the automatic transients' data layout, so
    the pipeline sees different stride patterns and can form different maps in each."""
    seed = _SEEDS[loop_exchange]

    # Unoptimized first, with the hook explicitly OFF -- an inherited hook from another test would
    # silently turn this leg into a second optimized run and the differential into a tautology.
    monkeypatch.delenv("OCEAN_E2E_SDFG_HOOK", raising=False)
    pre = run_variant(seed, tu_name)
    assert pre["passed"], f"velocity_tendencies: unoptimized build/lower/run failed:\n{pre['output'][-3500:]}"
    assert pre["n_changed"] > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"
    assert pre["max_diff"] == 0.0, (f"unoptimized SDFG diverged from reference, max|d|={pre['max_diff']:.3e}")

    # The harness forks a child that re-execs _ocean_e2e.py; the child inherits os.environ and
    # resolves the hook against tests/ on its PYTHONPATH.
    monkeypatch.setenv("OCEAN_E2E_SDFG_HOOK", "e2e._hooks:velocity_optimize")
    opt = run_variant(seed, tu_name)

    assert opt["passed"], f"velocity_tendencies: optimized build/lower/run failed:\n{opt['output'][-3500:]}"
    assert opt["n_changed"] > 0, "no output buffer changed -- the kernel did no work (test is vacuous)"
    # max_diff of 0.0 is a PERFECT match: test against None explicitly, never for truthiness.
    assert opt["max_diff"] is not None, "harness reported no diff value"
    # Bit-exact, same bar the unoptimized leg above just cleared: the pipeline reorders statements
    # and forms maps but never reassociates arithmetic, and the flags pin FMA contraction off. With
    # both legs at exactly 0.0 against one reference, pre and post are bit-identical to each other.
    assert opt["max_diff"] == 0.0, (f"pipeline changed the numerics, max|d|={opt['max_diff']:.3e} (must be 0)")
