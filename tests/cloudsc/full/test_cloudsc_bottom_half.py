"""Bottom-half CLOUDSC reproducer for the cloudsc_full xfail bisection. Companion to
``test_cloudsc_top_half.py`` (passes at 1e-14, isolating the divergence to the bottom half:
sedimentation + LU solver + flux/tendency updates). Source/sink accumulation block (orig
lines 1879-2617) is deleted so ZSOLQA/ZSOLQB stay zero and the LU solver factors a
near-identity matrix; init state above that point is preserved. SDFG vs f2py at strict
tolerance: agreement here means the cloudsc_full bug is cross-talk-only between halves."""

from pathlib import Path

import numpy as np
import pytest

from tests._util import f2py_compile
from tests.cloudsc.full._registries import (
    CLOUDSC_F90FLAGS,
    program_outputs,
)
from tests.cloudsc.full._harness import run_cloudsc

_HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def _f2py_bottom_half(tmp_path_factory):
    src = (_HERE / "cloudsc_bottom_half.F90").read_text()
    ref_dir = tmp_path_factory.mktemp("cloudsc_bottom_half_ref")
    return f2py_compile(
        src,
        ref_dir,
        "cloudsc_bottom_half_ref",
        # -ffree-line-length-none is the only gfortran-only flag (LLVM-flang has no line
        # limit); rest of the FP set is flang-portable.
        extra_f90flags=CLOUDSC_F90FLAGS,
        only=("cloudscouter",),
    )


# Physical (NaN-free) inputs: the bridge matches gfortran to tight tolerance here.
def test_cloudsc_bottom_half_numerical(tmp_path, _f2py_bottom_half, _strict_fp_cpu_args):
    src = (_HERE / "cloudsc_bottom_half.F90").read_text()
    outputs_sdfg, outputs_ref = run_cloudsc(src, "cloudsc_bottom_half", _f2py_bottom_half, tmp_path / "sdfg")

    for name in program_outputs:
        np.testing.assert_allclose(
            outputs_sdfg[name.lower()],
            outputs_ref[name.lower()],
            rtol=1e-14,
            atol=1e-14,
            err_msg=f"mismatch on output {name}",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
