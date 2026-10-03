"""End-to-end numerical correctness for ICON's AES graupel scheme (original Muphys layout).

Compiles ``mo_aes_graupel + mo_aes_thermo + mo_kind + mo_physical_constants`` as a gfortran reference (via the C-bound ``graupel_caller.f90`` wrapper) and the SAME sources through the bridge as one SDFG, runs both on the physical input columns of ``_graupel_harness.physical_columns`` (cold ice/snow, mixed phase, melting, warm-rain evaporation, warm/dry no-op) and compares every INOUT and OUT array.  The per-column fused variant is covered by ``test_aes_graupel_fused_full.py``.

Regression gate: the AoS-of-pointer-records gather temp (``t_qx_ptr%x``) used to size from unbound extents that call-time auto-fill defaulted to 1, under-allocating and overflowing the heap, until the ``fir.box_dims -> <name>_d<dim>`` extent resolution closed it.
"""

import numpy as np
import pytest

from tests._util import have_flang
from dace_fortran import build_sdfg_from_files

from ._graupel_harness import (
    DEP_SOURCES,
    ENTRY,
    ORIGINAL_SOURCE,
    SCENARIOS,
    Config,
    assert_families_fire,
    assert_match,
    compile_reference,
    copy_fields,
    physical_columns,
    run_reference,
    run_sdfg,
    zero_outputs,
)

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

RTOL = 1e-10
ATOL = 1e-14


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return compile_reference(tmp_path_factory.mktemp("graupel_orig_ref"), [ORIGINAL_SOURCE])


@pytest.fixture(scope="module")
def sdfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("graupel_orig_sdfg")
    built = build_sdfg_from_files(
        [*DEP_SOURCES, ORIGINAL_SOURCE], entry=ENTRY, name="graupel_run", out_dir=out / "build"
    )
    built.validate()
    return built


def _run_both(reference, sdfg, cfg: Config):
    inputs = physical_columns()
    ref, got = copy_fields(inputs), copy_fields(inputs)
    zero_outputs(ref)
    zero_outputs(got)
    run_reference(reference, ref, cfg)
    run_sdfg(sdfg, got, cfg)
    return inputs, ref, got


def test_aes_graupel_e2e_numerical(reference, sdfg):
    """Every INOUT prognostic and OUT diagnostic of ``graupel_run`` matches gfortran on all five scenarios."""
    inputs, ref, got = _run_both(reference, sdfg, Config(1, len(SCENARIOS), 1))
    assert_families_fire(inputs, ref)
    assert_match(ref, got, RTOL, ATOL)


def test_aes_graupel_e2e_offsets(reference, sdfg):
    """``ivstart > 1``, ``ivend < nvec`` and ``kstart > 1`` match gfortran; columns and levels outside the range are untouched."""
    inputs, ref, got = _run_both(reference, sdfg, Config(2, len(SCENARIOS) - 1, 4))
    assert_match(ref, got, RTOL, ATOL)
    for n in ("t", "qv", "qc", "qi", "qr", "qs", "qg"):
        np.testing.assert_array_equal(got[n][:, :3], inputs[n][:, :3])
        np.testing.assert_array_equal(got[n][[0, len(SCENARIOS) - 1]], inputs[n][[0, len(SCENARIOS) - 1]])


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ref_lib = compile_reference(Path(tmp) / "ref", [ORIGINAL_SOURCE])
        built = build_sdfg_from_files(
            [*DEP_SOURCES, ORIGINAL_SOURCE], entry=ENTRY, name="graupel_run", out_dir=Path(tmp) / "build"
        )
        built.validate()
        test_aes_graupel_e2e_numerical(ref_lib, built)
        test_aes_graupel_e2e_offsets(ref_lib, built)
