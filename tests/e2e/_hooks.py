# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""DUT transform hooks for the ``run_kernel_e2e`` harness.

The harness forks a subprocess that re-execs ``_ocean_e2e.py``, so the pipeline can't be handed
over as a callable -- it goes through ``OCEAN_E2E_SDFG_HOOK="module:func"``, which the child
imports and runs on the DUT SDFG before compile. ``tests/`` is on the child's PYTHONPATH
(``_ocean_e2e`` puts it there), so this module resolves as ``e2e._hooks``.
"""
from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.pipelines import num_maps, optimize


def velocity_optimize(sdfg):
    """ICON velocity kernels: the full pipeline, no specialize, no scalar-fission.

    Compiler flags are not this hook's business: the harness already pinned ``BITEXACT_CPU_ARGS``
    (which carries -O3 -march=native and the IEEE-strict set) before calling us.
    """
    optimize(sdfg)
    # Same non-vacuity bar as the vexx / cloudsc lanes: a numerically-identical result proves
    # nothing if the pipeline parallelized nothing. Raising here surfaces through the harness as
    # passed=False with the child's captured output, since this runs in the forked child.
    if num_maps(sdfg) == 0:
        raise AssertionError("pipeline produced no maps -- nothing was parallelized")
    # The harness builds the Fortran binding right after this hook and verifies the build-time
    # signature snapshot. Re-snapshot so the binding regenerates against the optimized SDFG: args
    # stay ABI-identical and free symbols may only shrink, else refreeze raises.
    refreeze(sdfg)
