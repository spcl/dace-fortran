"""Sub-bisection of test_cloudsc_bottom_upper.py's 51/548 PCOVPTOT mismatch: drops 4.5 EVAPORATION,
keeps Sedimentation/Autoconv/Melt/Freeze (4.2-4.4). Pass -> bug in 4.5; fail -> bug in 4.2-4.4."""
from pathlib import Path
import numpy as np
import pytest
from _util import f2py_compile, have_flang
from cloudsc.full._registries import CLOUDSC_F90FLAGS
from cloudsc.full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_a(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_upper_a.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_upper_a_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_upper_a_ref",
        # -ffree-line-length-none is the sole gfortran-only flag (long-line source; flang has no
        # line limit); the rest of CLOUDSC_F90FLAGS is flang-portable.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ))


# Physical (NaN-free) inputs: the bridge matches gfortran to tight tolerance here.
def test_cloudsc_bottom_upper_a_numerical(tmp_path, _f2py_a, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_upper_a.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_upper_a", _f2py_a, tmp_path / "sdfg")

    np.testing.assert_allclose(outputs_sdfg["pcovptot"],
                               outputs_ref["pcovptot"],
                               rtol=1e-14,
                               atol=1e-14,
                               err_msg="PCOVPTOT mismatch in bottom-upper-A (Sed/Autoconv/Melt/Freeze)")
