# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""DUT transform hooks for the ocean e2e harness.

``run_kernel_e2e`` forks a subprocess that re-execs ``_ocean_e2e.py``, so the pipeline can't be
handed over as a callable -- it goes through ``OCEAN_E2E_SDFG_HOOK="module:func"``, which the child
imports and runs on the DUT SDFG before compile. ``tests/`` is on the child's PYTHONPATH
(``_ocean_e2e`` puts it there), so this module resolves as ``e2e._hooks``.
"""
import dace

from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.pipelines import optimize


def velocity_optimize(sdfg):
    """Ocean velocity-family kernels: the full pipeline, no specialize, no scalar-fission.

    The harness has already stripped -ffast-math and pinned -ffp-contract=off for the bit-exact
    differential; only add the libm-vectorization subset, which changes no value.
    """
    args = dace.Config.get("compiler", "cpu", "args")
    for flag in ("-fno-math-errno", "-fno-trapping-math"):
        if flag not in args:
            args += " " + flag
    dace.Config.set("compiler", "cpu", "args", value=args)

    optimize(sdfg)
    # The harness builds the Fortran binding right after this hook and verifies the build-time
    # signature snapshot. Re-snapshot so the binding regenerates against the optimized SDFG: args
    # stay ABI-identical and free symbols may only shrink, else refreeze raises.
    refreeze(sdfg)
