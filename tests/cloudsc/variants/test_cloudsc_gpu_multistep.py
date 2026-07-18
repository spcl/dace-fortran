"""End-to-end value-correctness test for the CLOUDSC-GPU multistep variant: forward-Euler
NSTEPS substeps around the byte-for-byte-upstream kernel+driver (CLOUDSCOUTER); the substep
loop itself is transcribed from dwarf_cloudsc_gpu_multistep.F90, not vendored verbatim.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc.full._registries import CLOUDSC_F90FLAGS, program_outputs
from cloudsc.variants._harness import extract_variant_tu, mismatch_report, run_cloudsc_gpu

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "cloudsc_outer_multistep.F90"
NAME = "cloudsc_gpu_multistep"

# Integrated in place across substeps; each leg needs its own copy from the shared initial
# state so both start identical.
PROGNOSTIC_STATE = ("PT", "PQ", "PA", "PCLV")

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def variant_tu(tmp_path_factory):
    """Single-TU Fortran text for the multistep variant (built once)."""
    out_dir = tmp_path_factory.mktemp("multistep_tu")
    return extract_variant_tu(WRAPPER, out_dir, NAME)


@pytest.fixture(scope="module")
def f2py_ref(variant_tu, tmp_path_factory):
    """gfortran/f2py reference; only=('cloudscouter',) for the same crackfortran reason as scc_k_caching."""
    ref_dir = tmp_path_factory.mktemp("multistep_ref")
    return f2py_compile(variant_tu, ref_dir, "cloudsc_gpu_multistep_ref", extra_f90flags=CLOUDSC_F90FLAGS,
                        only=("cloudscouter", ))


@pytest.mark.parametrize("simplify", [False, True], ids=["raw", "simplify"])
def test_cloudsc_gpu_multistep_numerical(tmp_path, variant_tu, f2py_ref, _strict_fp_cpu_args, simplify):
    """SDFG-vs-gfortran equivalence on the CLOUDSC-GPU multistep variant."""
    outputs_sdfg, outputs_ref = run_cloudsc_gpu(variant_tu, NAME, f2py_ref, tmp_path / "sdfg",
                                                simplify=simplify,
                                                state_arrays=PROGNOSTIC_STATE)

    rtol = atol = 1e-12
    names = list(program_outputs) + list(PROGNOSTIC_STATE)
    report = mismatch_report(outputs_sdfg, outputs_ref, names, rtol=rtol, atol=atol)
    assert not report, "cloudsc_gpu_multistep numerical mismatch:\n" + "\n".join(report)
    # Guard against a degenerate no-op integration.
    assert np.any(np.abs(np.asarray(outputs_sdfg["pt"])) > 0.0), \
        "PT is all-zero -- the integration did not run"
