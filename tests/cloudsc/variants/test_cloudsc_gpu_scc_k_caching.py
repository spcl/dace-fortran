"""End-to-end value-correctness test for the CLOUDSC-GPU scc_k_caching variant: SDFG output
(via CLOUDSCOUTER, the byte-for-byte-upstream kernel+driver) vs gfortran/f2py on the same
single-TU source and seeded inputs. Species-index PARAMETERs bake to literals, never args.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import f2py_compile, have_flang
from cloudsc.full._registries import CLOUDSC_F90FLAGS, program_outputs
from cloudsc.variants._harness import GPU_SRC, extract_variant_tu, mismatch_report, run_cloudsc_gpu

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "cloudsc_outer_scc_k_caching.F90"
NAME = "cloudsc_gpu_scc_k_caching"

# Kernel's primary result (unpacked from driver's BUFFER_LOC); each leg needs its own copy
# since these are integrated in place.
LOC_TENDENCIES = ("tendency_loc_T", "tendency_loc_a", "tendency_loc_q", "tendency_loc_cld")

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.fixture(scope="module")
def variant_tu(tmp_path_factory):
    """Single-TU Fortran text for the scc_k_caching variant (built once)."""
    out_dir = tmp_path_factory.mktemp("scc_tu")
    return extract_variant_tu(WRAPPER, out_dir, NAME)


@pytest.fixture(scope="module")
def f2py_ref(variant_tu, tmp_path_factory):
    """gfortran/f2py reference from the same single-TU source.

    only=('cloudscouter',): crackfortran crashes on the inner driver/kernel's TYPE(...) dummies.
    """
    ref_dir = tmp_path_factory.mktemp("scc_ref")
    return f2py_compile(variant_tu, ref_dir, "cloudsc_gpu_scc_ref", extra_f90flags=CLOUDSC_F90FLAGS,
                        only=("cloudscouter", ))


@pytest.mark.parametrize("simplify", [False, True], ids=["raw", "simplify"])
def test_cloudsc_gpu_scc_k_caching_numerical(tmp_path, variant_tu, f2py_ref, _strict_fp_cpu_args, simplify):
    """SDFG-vs-gfortran equivalence on the CLOUDSC-GPU scc_k_caching variant."""
    outputs_sdfg, outputs_ref = run_cloudsc_gpu(variant_tu, NAME, f2py_ref, tmp_path / "sdfg",
                                                simplify=simplify,
                                                state_arrays=LOC_TENDENCIES)

    rtol = atol = 1e-12
    names = list(program_outputs) + list(LOC_TENDENCIES)
    report = mismatch_report(outputs_sdfg, outputs_ref, names, rtol=rtol, atol=atol)
    assert not report, "cloudsc_gpu_scc_k_caching numerical mismatch:\n" + "\n".join(report)
    # Guard against a degenerate all-zero comparison.
    assert np.any(np.abs(np.asarray(outputs_sdfg["tendency_loc_t"])) > 0.0), \
        "tendency_loc_T is all-zero -- the kernel did not run"
