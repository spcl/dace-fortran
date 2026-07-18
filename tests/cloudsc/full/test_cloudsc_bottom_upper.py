"""Bisection step: bottom-UPPER reproducer.

Splits the bottom half (fails with 26/548 PCOVPTOT divergence, same as
test_cloudsc_full) at the physics/solver boundary: keeps bottom-half physics
(sedimentation/autoconversion/melting/freezing/evaporation, lines 2620-3355)
and drops solvers/flux/tendency (lines 3356-3710), replacing them with a
direct PCOVPTOT(JL,JK) = ZCOVPTOT(JL) writeback.

PASS -> bug is in bottom-lower (3356-3710). Same-mismatch FAIL -> bug is here.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc.full._registries import (
    CLOUDSC_F90FLAGS, )
from cloudsc.full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent
pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def _f2py_bottom_upper(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_upper.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_upper_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_upper_ref",
        # -ffree-line-length-none is gfortran-only (long-line source); flang has no line limit, needs no equivalent.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter", ),
    )


# Physical (NaN-free) inputs: the bridge matches gfortran to tight tolerance here.
def test_cloudsc_bottom_upper_numerical(tmp_path, _f2py_bottom_upper, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_upper.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_upper", _f2py_bottom_upper, tmp_path / "sdfg")

    # only PCOVPTOT is meaningful -- other outputs zero, their writes dropped with the solver/flux section.
    np.testing.assert_allclose(
        outputs_sdfg["pcovptot"],
        outputs_ref["pcovptot"],
        rtol=1e-14,
        atol=1e-14,
        err_msg="PCOVPTOT mismatch in bottom-upper (sedimentation/physics) reproducer",
    )
