# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ICON AES graupel microphysics: ingest the Fortran, parallelize it, run it and check the numbers.

The per-column fused graupel (``tests/icon/graupel/aes_graupel_fused/graupel.f90``) and the three modules it
``USE``s become one SDFG. The column loop ``DO iv = ivstart, ivend`` becomes a map, and so does the level loop that
finds ``kmin``, a min-reduction; the sedimentation loop carries its flux from level to level and stays sequential. The
other maps are the whole-array loops around them.

``optimize`` replays the SDFG on physical columns before and after every structural stage and requires bit-exact
agreement; the result must then match the gfortran build on every regime column, over the full column range and over
a sub-range starting below the top level. Run from the repository root: ``python -m samples.graupel``.
"""

import tempfile
from pathlib import Path

import dace
import numpy as np
from dace.sdfg import nodes

import dace_fortran
from dace_fortran.pipelines import assert_parallelized, num_loops, num_maps, optimize
from tests._util import BITEXACT_CPU_ARGS
from tests.icon.graupel._graupel_harness import (
    DEP_SOURCES,
    ENTRY,
    FUSED_SOURCE,
    SCENARIOS,
    Config,
    assert_families_fire,
    assert_match,
    compile_reference,
    copy_fields,
    physical_columns,
    run_reference,
    run_sdfg,
    sdfg_args,
    zero_outputs,
)

LEVELS = 20
MAX_MAPS = 14
SEQUENTIAL_LOOPS = 1  # sedimentation


def run(work_dir: Path) -> None:
    sdfg = dace_fortran.build_sdfg_from_files([*DEP_SOURCES, FUSED_SOURCE], entry=ENTRY, name="graupel")
    # DaCe's default flags let gcc contract floating-point arithmetic differently in the unoptimized and the optimized
    # program; comparing them bit for bit needs value-preserving flags (-O3, no FMA contraction).
    dace.Config.set("compiler", "cpu", "args", value=BITEXACT_CPU_ARGS)
    full = Config(ivstart=1, ivend=len(SCENARIOS), kstart=1)
    verify = physical_columns(LEVELS)
    zero_outputs(verify)
    loops_before = num_loops(sdfg)
    optimize(sdfg, verify_inputs=sdfg_args(verify, full))
    assert_parallelized(sdfg, loops_before=loops_before, max_maps=MAX_MAPS, sequential=SEQUENTIAL_LOOPS)
    assert any(
        isinstance(n, nodes.MapEntry) and str(n.map.range.ranges[0][0]) == "ivstart"
        for n, _ in sdfg.all_nodes_recursive()
    ), "the iv column loop is not a map"

    reference = compile_reference(work_dir, [FUSED_SOURCE])  # gfortran -O0
    for cfg in (full, Config(ivstart=2, ivend=len(SCENARIOS) - 1, kstart=4)):
        inputs = physical_columns(LEVELS)
        want, got = copy_fields(inputs), copy_fields(inputs)
        zero_outputs(want)
        zero_outputs(got)
        run_reference(reference, want, cfg)
        run_sdfg(sdfg, got, cfg)
        if cfg == full:
            assert_families_fire(inputs, want)
        # gfortran -O0 and the C++ build differ by at most 1e-15 relative, except the 1e-19-sized decay tails
        assert_match(want, got, rtol=1e-10, atol=1e-14)
        assert np.array_equal(got["t"][:, : cfg.kstart - 1], inputs["t"][:, : cfg.kstart - 1])
    print(f"graupel: {num_maps(sdfg)} maps, {num_loops(sdfg)} sequential loops, matches gfortran")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp))
