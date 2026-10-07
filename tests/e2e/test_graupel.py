# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-column fused ICON AES graupel (``icon/graupel/aes_graupel_fused/graupel.f90``) through the
full parallelization pipeline, against the gfortran build of the same source.

Same source, reference and physical input columns as ``icon/graupel/test_aes_graupel_fused_full``,
with ``pipelines.optimize`` inserted between the SDFG build and the run. Nothing is specialized:
the species counts CloudSC bakes are Fortran ``PARAMETER``s here, folded by the build already.

The column loop ``DO iv = ivstart, ivend`` is the parallel one; the two ``k`` loops inside it
carry ``kmin`` and the sedimentation flux from level to level and stay sequential.
"""

import numpy as np
import pytest

from dace.sdfg import nodes
from dace_fortran import build_sdfg_from_files
from dace_fortran.pipelines import num_maps, optimize
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

pytestmark = pytest.mark.e2e

KE = 20
# Same bound and evidence as test_aes_graupel_fused_full: gfortran -O0 vs the DaCe C++ build differ
# by at most 1e-15 relative, except the 1e-19-sized decay tails.
RTOL = 1e-10
ATOL = 1e-14


def _column_maps(sdfg) -> list[nodes.MapEntry]:
    """Maps over the ``iv`` column range ``ivstart..ivend``."""
    return [
        n
        for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and str(n.map.range.ranges[0][0]) == "ivstart"
    ]


def test_graupel_pipeline_numerical_e2e(tmp_path, e2e_cpu_args):
    """Optimized SDFG output == gfortran output, the column loop is a map, and the pre- and
    post-optimize SDFGs agree BIT-EXACTLY on the same inputs (``verify_inputs``)."""
    sdfg = build_sdfg_from_files(
        [*DEP_SOURCES, FUSED_SOURCE], entry=ENTRY, name="graupel_e2e", out_dir=tmp_path / "build"
    )
    full = Config(1, len(SCENARIOS), 1)
    verify = physical_columns(KE)
    zero_outputs(verify)
    optimize(sdfg, verify_inputs=sdfg_args(verify, full))

    assert num_maps(sdfg) > 0, "pipeline produced no maps -- nothing was parallelized"
    assert _column_maps(sdfg), "the iv column loop was not parallelized"

    reference = compile_reference(tmp_path / "ref", [FUSED_SOURCE])
    for cfg in (full, Config(2, len(SCENARIOS) - 1, 4)):
        inputs = physical_columns(KE)
        ref, got = copy_fields(inputs), copy_fields(inputs)
        zero_outputs(ref)
        zero_outputs(got)
        run_reference(reference, ref, cfg)
        run_sdfg(sdfg, got, cfg)
        if cfg == full:
            assert_families_fire(inputs, ref)
        assert_match(ref, got, RTOL, ATOL)
        assert np.array_equal(got["t"][:, : cfg.kstart - 1], inputs["t"][:, : cfg.kstart - 1])


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from tests._util import BITEXACT_CPU_ARGS

    import dace

    dace.Config.set("compiler", "cpu", "args", value=BITEXACT_CPU_ARGS)
    with tempfile.TemporaryDirectory() as tmp:
        test_graupel_pipeline_numerical_e2e(Path(tmp), BITEXACT_CPU_ARGS)
