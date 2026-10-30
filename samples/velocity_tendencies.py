# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON ``velocity_tendencies``: ingest the Fortran, parallelize it, run it and check the numbers.

The dycore kernel ``solve_nh`` calls, extracted with its modules into one translation unit, in both
``__LOOP_EXCHANGE`` layouts. Every loop becomes a map, the ``clip_count`` counter included (an integer sum
reduction).

The kernel reads ICON configuration globals and derived-type dummies, so it runs through its generated Fortran
binding next to the untouched gfortran kernel, in a child process (``run_kernel_e2e``). The child builds the SDFG and
calls :func:`optimize` on it before compiling, via ``OCEAN_E2E_SDFG_HOOK``. Each layout runs twice on one seeded
input set, unoptimized and optimized, and both must match gfortran bit for bit, so they match each other too. Run from
the repository root: ``python -m samples.velocity_tendencies``.
"""

import os
from pathlib import Path

from dace.sdfg import SDFG

from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.pipelines import num_loops, num_maps
from dace_fortran.pipelines import optimize as optimize_pipeline
from tests.icon.atmosphere._atmo_harness import VELOCITY_TU_VARIANTS
from tests.icon.ocean._ocean_e2e import run_kernel_e2e

ATMOSPHERE = Path(__file__).resolve().parents[1] / "tests" / "icon" / "atmosphere"
ENTRY = "mo_velocity_advection::velocity_tendencies"
HOOK = "samples.velocity_tendencies:optimize"
MAPS = {True: 39, False: 39}  # per ``__LOOP_EXCHANGE`` layout
# One input stream per layout: the define swaps the automatic transients' layout, so the two are different kernels
# sharing an ABI, and inputs tuned to one could hide a layout-sensitive bug in the other.
SEEDS = {True: 1001, False: 1002}


def optimize(sdfg: SDFG) -> None:
    """The hook the child runs on the built SDFG; raising here fails that run with the child's output."""
    optimize_pipeline(sdfg)
    loop_exchange = "no_loop_exchange" not in sdfg.name  # the SDFG is named after the translation unit
    counts = (num_maps(sdfg), num_loops(sdfg))
    if counts != (MAPS[loop_exchange], 0):
        raise AssertionError(f"{sdfg.name}: {counts[0]} maps, {counts[1]} loops")
    # The binding is generated from the build-time signature snapshot; the optimized SDFG keeps the ABI
    refreeze(sdfg)


def run_layout(tu_name: str, loop_exchange: bool, optimized: bool) -> dict:
    """One child run of ``tu_name``, optimized or as built, against the gfortran kernel."""
    if optimized:
        os.environ["OCEAN_E2E_SDFG_HOOK"] = HOOK
    else:
        os.environ.pop("OCEAN_E2E_SDFG_HOOK", None)
    result = run_kernel_e2e(
        ATMOSPHERE / tu_name,
        ENTRY,
        seed=SEEDS[loop_exchange],
        int_fill=1,  # one in-domain block: every connectivity index points at element 1
        # nflatlev is a DO lower bound, so its BSS 0 would start jk at 0, out of bounds
        module_seeds={"nproma": 8, "nflatlev": 1, "nrdmax": 1},
        # a real vertical column within the n=8 buffers, so the vertical loops initialise the automatic transients
        array_overrides={"p_patch_nlev": 7, "p_patch_nlevp1": 8},
    )
    leg = f"{tu_name} {'optimized' if optimized else 'unoptimized'}"
    assert result["passed"], f"{leg} failed:\n{result['output'][-3500:]}"
    assert result["n_changed"] > 0, f"{leg}: no output changed, the kernel did no work"
    assert result["max_diff"] == 0.0, f"{leg} differs from gfortran, max |d| = {result['max_diff']:.3e}"
    return result


def run() -> None:
    try:
        for tu_name, loop_exchange in VELOCITY_TU_VARIANTS:
            run_layout(tu_name, loop_exchange, optimized=False)
            run_layout(tu_name, loop_exchange, optimized=True)
            print(f"velocity_tendencies {tu_name}: {MAPS[loop_exchange]} maps, no loops, bit-exact against gfortran")
    finally:
        os.environ.pop("OCEAN_E2E_SDFG_HOOK", None)


if __name__ == "__main__":
    run()
