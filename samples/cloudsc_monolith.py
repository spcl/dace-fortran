# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Monolithic ECMWF CLOUDSC: ingest the Fortran, parallelize it, run it and check the numbers.

``tests/cloudsc/full/cloudsc.F90`` is the whole scheme in one file, ``CLOUDSCOUTER`` driving ``CLOUDSC``. The
species count ``NCLV`` is the only species constant that reaches the SDFG as a symbol (the ``NCLDQ*`` indices are
``YOECLDP`` ``PARAMETER``s, folded by the build), so it is specialized to 5. Every column (``JL``) loop becomes a map;
the vertical sweep and the species-ordered solver stay sequential.

The optimized SDFG must match the gfortran build to fp64 precision and the unoptimized SDFG bit for bit on the same
seeded physical state. Run from the repository root: ``python -m samples.cloudsc_monolith``.
"""

import copy
import tempfile
from pathlib import Path

import dace
import numpy as np

import dace_fortran
from dace_fortran.pipelines import assert_parallelized, num_loops, num_maps, optimize
from tests._util import BITEXACT_CPU_ARGS
from tests.cloudsc.full._harness import f2py_reference, run_against_reference
from tests.cloudsc.full._registries import program_outputs

SOURCE = Path(__file__).resolve().parents[1] / "tests" / "cloudsc" / "full" / "cloudsc.F90"
MAX_MAPS = 264
SEQUENTIAL_LOOPS = 8


def run(work_dir: Path) -> None:
    sdfg = dace_fortran.build_sdfg(SOURCE.read_text(), entry="cloudscouter", name="cloudsc_monolith")
    unoptimized = copy.deepcopy(sdfg)
    loops_before = num_loops(sdfg)
    optimize(sdfg, symbols={"nclv": 5})
    assert_parallelized(sdfg, loops_before=loops_before, max_maps=MAX_MAPS, sequential=SEQUENTIAL_LOOPS)

    # DaCe's default flags let gcc contract floating-point arithmetic differently in the unoptimized and the optimized
    # program; comparing them bit for bit needs value-preserving flags (-O3, no FMA contraction).
    dace.Config.set("compiler", "cpu", "args", value=BITEXACT_CPU_ARGS)
    got, want = run_against_reference(sdfg, f2py_reference(work_dir), unoptimized=unoptimized)
    for name in program_outputs:
        np.testing.assert_allclose(got[name.lower()], want[name.lower()], rtol=1e-11, atol=1e-11, err_msg=name)
    print(f"cloudsc_monolith: {num_maps(sdfg)} maps, {num_loops(sdfg)} sequential loops, matches gfortran")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(tmp))
