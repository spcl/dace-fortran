"""Full-source CLOUDSC end-to-end test for the HLFIR bridge.

Drives the entire ECMWF CLOUDSC microphysics kernel through the bridge and compares against a
gfortran/f2py reference compiled from the same source -- catches integration regressions
per-loopnest tests miss (state hoisting across the block loop, the 100+ scalar-arg signature,
nested ELEMENTAL, rank-4 ``PCLV(:, :, :, JKGLO)`` per-block slicing).

Source: ``cloudsc.F90``, preprocessor directives pre-expanded inline (flang-new-21
``-emit-hlfir`` has no ``-cpp``). The DT-of-constants wrapper pattern is handled by
``hlfir-flatten-structs`` (minimal pinned version: ``struct_of_scalars_test.py``).

Reference: f2py-compiled Fortran from the same source, non-transformed (``feedback_e2e_numerical``). Inputs: seeded random data via ``_registries.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc.full._registries import (
    CLOUDSC_F90FLAGS,
    program_outputs,
)
from cloudsc.full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_ref(tmp_path_factory):
    """Build the f2py reference once per session (3541-line compile ~30-90s).

    FP flags: flang-portable core ``-O0 -fno-fast-math -ffp-contract=off``. Neither side
    zero-fills locals, so an uninitialised-read divergence is a real bug, not a flag artifact.
    ``-ffree-line-length-none`` is the sole gfortran-only flag (long-line source; flang has no
    line limit).
    """
    src = (_HERE / "cloudsc.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_ref")
    # only=('cloudscouter',): hides inner CLOUDSC from crackfortran -- its TYPE(TOMCST/...)
    # dummies map to 'void' and crash f2py (KeyError: 'void')
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_ref",
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


def test_cloudsc_full_numerical(tmp_path, _f2py_ref, _strict_fp_cpu_args):
    """End-to-end SDFG-vs-gfortran equivalence on full CLOUDSC under physically-plausible inputs (element-wise at ``rtol=atol=1e-12``).

    Physical inputs matter: the old uniform-random ``get_inputs`` drove ``ZINEW`` to NaN, and
    Fortran MIN/MAX with a NaN operand is processor-dependent by the standard -- gfortran vs
    flang/DaCe's ``arith.maxnumf`` legitimately differ there. That's not a bridge defect;
    validating in-regime is the correct contract.
    """
    src = (_HERE / "cloudsc.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc", _f2py_ref, tmp_path / "sdfg")

    # strict rtol=atol=1e-15 every cell; PCOVPTOT's old 1-cell tolerance removed once fixes
    # 2e38612c3/acb337a81 closed the ULP chain that justified it
    rtol = atol = 1e-15
    report: list[str] = []
    for name in program_outputs:
        a = np.asarray(outputs_sdfg[name.lower()])
        b = np.asarray(outputs_ref[name.lower()])
        bad = ~np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        nbad = int(bad.sum())
        if nbad == 0:
            continue
        report.append(f"{name}: {nbad} cell(s) exceed rtol={rtol} "
                      f"(max |Δ|={np.abs(a - b)[bad].max():.3e})")
    assert not report, "cloudsc_full numerical mismatch:\n" + "\n".join(report)
