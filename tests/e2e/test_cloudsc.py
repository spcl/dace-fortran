# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full ECMWF CLOUDSC through the full parallelization pipeline.

Same source, reference and seeded physical inputs as ``cloudsc/full/test_cloudsc_full`` (see it for
why the inputs must be in-regime rather than uniform-random), with ``pipelines.optimize`` inserted
between the SDFG build and the run.

CLOUDSC needs the ``specialize`` knob the ocean and QE kernels do without: it bakes the species
count NCLV so downstream shape and branch folding have literals. ``scalar_fission`` runs
unconditionally in the pipeline; it is what splits the scalar-carried loop bodies so CLOUDSC's
block loop can map at all.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from dace_fortran.pipelines import num_maps, optimize
from tests._util import build_sdfg
from tests.cloudsc.full._harness import f2py_reference, run_against_reference
from tests.cloudsc.full._registries import program_outputs

pytestmark = pytest.mark.e2e

_SRC = Path(__file__).resolve().parents[1] / "cloudsc" / "full" / "cloudsc.F90"

# The species count, the one species constant that reaches the SDFG as a free symbol: CLOUDSCOUTER
# sizes its arrays by its NCLV dummy, while CLOUDSC reads NCLV and the NCLDQ* species indices from
# YOECLDP's PARAMETERs, which the build folds to literals (CLOUDSCOUTER's NCLDQ* dummies are unused).
_SPECIALIZE = {"nclv": 5}


@pytest.fixture(scope="module")
def _f2py_ref(tmp_path_factory):
    """The untouched gfortran reference, built once (see cloudsc/full/test_cloudsc_full)."""
    return f2py_reference(tmp_path_factory.mktemp("cloudsc_e2e_ref"))


def test_cloudsc_pipeline_numerical_e2e(tmp_path, _f2py_ref, e2e_cpu_args):
    """Optimized SDFG output == untouched-reference output, to fp64 precision.

    Replaying the unoptimized SDFG adds the second, stricter question on the same seeded inputs:
    the pre- and post-optimize SDFGs must agree BIT-EXACTLY. The reference comparison below can only
    be a 1e-11 one (frontend vs gfortran differ in evaluation order), so on its own a pipeline bug
    worth less than 1e-11 passes; the differential has no such floor.
    """
    sdfg = build_sdfg(_SRC.read_text(), tmp_path / "sdfg", name="cloudsc", entry="cloudscouter").build()
    unoptimized = copy.deepcopy(sdfg)
    optimize(sdfg, symbols=_SPECIALIZE)
    assert num_maps(sdfg) > 0, "pipeline produced no maps -- nothing was parallelized"

    outputs_sdfg, outputs_ref = run_against_reference(sdfg, _f2py_ref, unoptimized=unoptimized)

    rtol = atol = 1e-11  # fp64 precision guard
    report: list[str] = []
    for name in program_outputs:
        a = np.asarray(outputs_sdfg[name.lower()])
        b = np.asarray(outputs_ref[name.lower()])
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if bad.any():
            report.append(
                f"{name}: {int(bad.sum())} cell(s) exceed rtol={rtol} (max |Δ|={np.abs(a - b)[bad].max():.3e})"
            )
    assert not report, "cloudsc pipeline numerical mismatch:\n" + "\n".join(report)


if __name__ == "__main__":
    import tempfile

    import dace

    from tests._util import BITEXACT_CPU_ARGS

    dace.Config.set("compiler", "cpu", "args", value=BITEXACT_CPU_ARGS)
    with tempfile.TemporaryDirectory() as tmp:
        test_cloudsc_pipeline_numerical_e2e(Path(tmp) / "run", f2py_reference(Path(tmp) / "ref"), BITEXACT_CPU_ARGS)
