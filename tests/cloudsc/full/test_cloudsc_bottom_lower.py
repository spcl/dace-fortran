"""Bottom-LOWER reproducer: keeps solvers + flux + tendency (cloudsc.F90:3356-3710), drops
bottom-upper physics (2620-3355) -- ZSOLQA/ZSOLQB/ZCOVPTOT stay at their zero reset.
Expect PASS at rtol=atol=1e-14. Bisection chain: top_half PASS, bottom_half FAIL,
bottom_upper FAIL, bottom_upper_a PASS (bug in 4.5 EVAP), bottom_lower PASS -> bug not in
solvers/flux/tendency.
"""
from pathlib import Path
import numpy as np
import pytest
from _util import f2py_compile, have_flang
from cloudsc.full._registries import CLOUDSC_F90FLAGS, program_outputs
from cloudsc.full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_lo(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_lower.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_lower_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_lower_ref",
        # -ffree-line-length-none is gfortran-only (parser necessity for long lines); flang has no line limit. FP set is otherwise flang-portable.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ))


def test_cloudsc_bottom_lower_numerical(tmp_path, _f2py_lo, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_lower.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_lower", _f2py_lo, tmp_path / "sdfg")

    # PASS at strict tolerance; compares every program_output (solvers+flux+tendency writes them all).
    for name in program_outputs:
        np.testing.assert_allclose(outputs_sdfg[name.lower()],
                                   outputs_ref[name.lower()],
                                   rtol=1e-14,
                                   atol=1e-14,
                                   err_msg=f"PCOVPTOT mismatch in bottom-lower (solvers/flux/tendency only): {name}")
