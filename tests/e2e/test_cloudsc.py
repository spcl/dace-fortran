# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Full ECMWF CLOUDSC through the full parallelization pipeline.

Same source, reference and seeded physical inputs as ``cloudsc/full/test_cloudsc_full`` (see it for
why the inputs must be in-regime rather than uniform-random), with ``pipelines.optimize`` inserted
between the SDFG build and the run.

CLOUDSC needs both knobs the ocean and QE kernels do without: ``specialize`` bakes the species
counts so downstream shape and branch folding have literals, and ``scalar_fission`` splits the
scalar-carried loop bodies so the block loop can map at all.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc.full._harness import run_cloudsc
from cloudsc.full._registries import CLOUDSC_F90FLAGS, program_outputs
from dace_fortran.pipelines import num_maps, optimize

_SRC = Path(__file__).resolve().parents[1] / "cloudsc" / "full" / "cloudsc.F90"

# Species counts CLOUDSC treats as compile-time constants. flang lowercases identifiers and some
# of these fold away during the build, so the names are matched against the live SDFG rather than
# assumed present.
_SPECIALIZE = {"NCLV": 5, "NCLDQI": 2, "NCLDQL": 1, "NCLDQR": 3, "NCLDQS": 4}

pytestmark = [pytest.mark.e2e, pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")]


def _split_specialize(sdfg):
    """Split ``_SPECIALIZE`` into (symbols, scalars) by how each name appears in ``sdfg``.

    A name already folded away by the build matches neither and is simply not specialized.
    """
    from dace.data import Scalar

    present_syms = {str(s) for s in sdfg.free_symbols} | set(sdfg.symbols)
    syms, scalars = {}, {}
    for name, val in _SPECIALIZE.items():
        for cand in (name, name.lower(), name.upper()):
            if cand in present_syms:
                syms[cand] = val
                break
            if cand in sdfg.arrays and isinstance(sdfg.arrays[cand], Scalar):
                scalars[cand] = val
                break
    return syms, scalars


@pytest.fixture(scope="module")
def _f2py_ref(tmp_path_factory):
    """The untouched gfortran reference, built once (see cloudsc/full/test_cloudsc_full)."""
    return f2py_compile(_SRC.read_text(),
                        tmp_path_factory.mktemp("cloudsc_e2e_ref"),
                        "cloudsc_ref",
                        extra_f90flags=CLOUDSC_F90FLAGS,
                        only=("cloudscouter", ))


def test_cloudsc_pipeline_numerical_e2e(tmp_path, _f2py_ref, e2e_cpu_args):
    """Optimized SDFG output == untouched-reference output, to fp64 precision."""
    maps = {}

    def transform(sdfg):
        syms, scalars = _split_specialize(sdfg)
        optimize(sdfg, symbols=syms, scalars=scalars, scalar_fission=True)
        maps["n"] = num_maps(sdfg)

    outputs_sdfg, outputs_ref = run_cloudsc(_SRC.read_text(),
                                            "cloudsc",
                                            _f2py_ref,
                                            tmp_path / "sdfg",
                                            transform=transform)

    assert maps["n"] > 0, "pipeline produced no maps -- nothing was parallelized"

    rtol = atol = 1e-11  # fp64 precision guard
    report: list[str] = []
    for name in program_outputs:
        a = np.asarray(outputs_sdfg[name.lower()])
        b = np.asarray(outputs_ref[name.lower()])
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if bad.any():
            report.append(f"{name}: {int(bad.sum())} cell(s) exceed rtol={rtol} "
                          f"(max |Δ|={np.abs(a - b)[bad].max():.3e})")
    assert not report, "cloudsc pipeline numerical mismatch:\n" + "\n".join(report)
