"""Whole per-column fused ICON AES graupel (``aes_graupel_fused/graupel.f90``, 1556 lines) through the frontend into ONE SDFG, compared against the gfortran build of the same source.

The fused variant runs one ``iv`` loop per column: microphysics over ``k`` followed by the sedimentation scan over ``k`` (``kmin(np)``, scalar ``eflx``, ``vt(np)``).  The source is the user's file and stays verbatim, including its quirks, which the test preserves: ``qnc(ivstart)`` is used for every column, ``pflx(iv,k)`` is written only for ``k >= MINVAL(kmin)`` (outputs are zeroed before both calls) and ``kstart`` skips the upper levels.

Input columns are the physical scenarios of ``_graupel_harness.physical_columns``; each process family must visibly fire in the reference so the comparison cannot pass on a no-op path.

Marker ``long``: the build is a large-SDFG frontend + compile run (CI runs it, local sweeps exclude it with ``-m "not long"``).  ``integration`` is reserved for the ICON orig-vs-binding runner and ``e2e`` for the parallelization-pipeline kernels, which this is not.

Run directly with ``python -m tests.icon.graupel.test_aes_graupel_fused_full`` from the repo root (explicit calls in ``__main__``).

Tolerance evidence: see ``RTOL``/``ATOL`` below.
"""

import numpy as np
import pytest

from dace_fortran import build_sdfg_from_files

from ._graupel_harness import (
    DEP_SOURCES,
    ENTRY,
    FUSED_SOURCE,
    ORIGINAL_SOURCE,
    SCENARIOS,
    Config,
    assert_families_fire,
    assert_match,
    compile_reference,
    copy_fields,
    physical_columns,
    random_state,
    run_reference,
    run_sdfg,
    zero_outputs,
)

pytestmark = pytest.mark.long


KE = 20
DT = 30.0
# gfortran -O0 -ffp-contract=off vs the DaCe C++ build, measured on these columns (and 1/2 configs): max |diff| t 1.1e-13 K (scale 294), pre_gsp 9e-13 W/m2 (scale 4e3), mass fields and fluxes <= 1.4e-19 (scale 1e-4..1e-3); relative error 4e-16..1.4e-15 everywhere except the qs/qg/prs/prg decay tails (|x| ~ 1e-19), where the same absolute 1e-19 error reads as 2e-11.  RTOL covers the tails' relative error with 5x margin; ATOL (1e-14 on kg/kg, kg/m2/s) is 5 orders above the largest mass-field absolute error and 8 orders below the signals (>= 1e-6) the families assert on.
RTOL = 1e-10
ATOL = 1e-14
# Random states hit cancellation-heavy branches (rain sedimentation near an empty layer): seed 20261003 trial 6 gives |qr diff| 3.9e-14 (rel 1.8e-9 at qr ~ 1e-5), which the energy flux amplifies to pre_gsp |diff| 1.3e-7 (rel 1.6e-10, scale 1e4).  A gfortran -O3 -march=native -ffp-contract=fast build of the same source deviates from the -O0 reference by exactly the same amounts (same trial, same fields), so this is floating-point reassociation/FMA noise, not a logic difference; over 42 random trials nothing else exceeds 4e-12.  Tolerance 1e-8 / 1e-12 leaves >50x margin on the observed values while staying 9 orders below the 1e-3 kg/kg signal scale.
RANDOM_RTOL = 1e-8
RANDOM_ATOL = 1e-12


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return compile_reference(tmp_path_factory.mktemp("graupel_fused_ref"), [FUSED_SOURCE])


@pytest.fixture(scope="module")
def sdfg(tmp_path_factory):
    """The single SDFG for the whole fused module (built once, reused for every configuration)."""
    out = tmp_path_factory.mktemp("graupel_fused_sdfg")
    sdfg = build_sdfg_from_files([*DEP_SOURCES, FUSED_SOURCE], entry=ENTRY, name="graupel_fused", out_dir=out / "build")
    sdfg.validate()
    return sdfg


def _run_both(reference, sdfg, cfg: Config, **columns):
    inputs = physical_columns(**{"ke": KE, **columns})
    ref, got = copy_fields(inputs), copy_fields(inputs)
    zero_outputs(ref)
    zero_outputs(got)
    run_reference(reference, ref, cfg)
    run_sdfg(sdfg, got, cfg)
    return inputs, ref, got


def test_fused_graupel_full_columns(reference, sdfg):
    """All five scenarios (one column each), full vertical range: every INOUT and OUT array matches gfortran."""
    inputs, ref, got = _run_both(reference, sdfg, Config(1, len(SCENARIOS), 1, DT))
    assert_families_fire(inputs, ref)
    assert_match(ref, got, RTOL, ATOL)


def test_fused_graupel_ivstart_offset_and_kstart(reference, sdfg):
    """``ivstart > 1`` (qnc(ivstart) quirk, untouched columns), ``ivend < nvec`` and ``kstart > 1`` (upper levels skipped) match gfortran."""
    inputs, ref, got = _run_both(reference, sdfg, Config(2, len(SCENARIOS) - 1, 4, DT))
    assert_match(ref, got, RTOL, ATOL)
    # columns outside [ivstart, ivend] and levels above kstart are untouched
    for n in ("t", "qv", "qc", "qi", "qr", "qs", "qg"):
        np.testing.assert_array_equal(got[n][:, :3], inputs[n][:, :3])
        np.testing.assert_array_equal(got[n][[0, len(SCENARIOS) - 1]], inputs[n][[0, len(SCENARIOS) - 1]])
    # the offset must actually change the physics relative to the full-range call
    _, ref_full, _ = _run_both(reference, sdfg, Config(1, len(SCENARIOS), 1, DT))
    assert not np.allclose(ref["qc"], ref_full["qc"])


def test_fused_graupel_other_shape_and_dt(reference, sdfg):
    """One SDFG serves other runtime extents: ``ke=12``, ten columns, thicker layers and ``dt=60`` still match gfortran (and are not a no-op)."""
    inputs, ref, got = _run_both(reference, sdfg, Config(1, 10, 1, 60.0), ke=12, dz0=400.0, repeats=2)
    assert any((ref[n] != inputs[n]).any() for n in ("t", "qv", "qc", "qi", "qr", "qs", "qg"))
    assert ref["pflx"].any()
    assert_match(ref, got, RTOL, ATOL)


def test_fused_graupel_random_states(reference, sdfg):
    """Seeded random states (mixed species, T 215-300 K, random extents/``ivstart``/``ivend``/``kstart``/``dt``) match gfortran: the scenario columns cannot cover every branch combination."""
    rng = np.random.default_rng(20261003)
    for _ in range(12):
        inputs, cfg = random_state(rng)
        ref, got = copy_fields(inputs), copy_fields(inputs)
        zero_outputs(ref)
        zero_outputs(got)
        run_reference(reference, ref, cfg)
        run_sdfg(sdfg, got, cfg)
        assert_match(ref, got, RANDOM_RTOL, RANDOM_ATOL)


def test_fused_layout_matches_original_layout(reference, tmp_path):
    """The fused per-column source computes the same physics as the original Muphys layout (gfortran vs gfortran), so the fused SDFG test also pins the original scheme's results."""
    original = compile_reference(tmp_path / "orig_ref", [ORIGINAL_SOURCE])
    cfg = Config(1, len(SCENARIOS), 1, DT)
    inputs = physical_columns(KE)
    fused, orig = copy_fields(inputs), copy_fields(inputs)
    zero_outputs(fused)
    zero_outputs(orig)
    run_reference(reference, fused, cfg)
    run_reference(original, orig, cfg)
    assert_match(fused, orig, RTOL, ATOL)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ref_lib = compile_reference(Path(tmp) / "ref", [FUSED_SOURCE])
        built = build_sdfg_from_files(
            [*DEP_SOURCES, FUSED_SOURCE], entry=ENTRY, name="graupel_fused", out_dir=Path(tmp) / "build"
        )
        built.validate()
        test_fused_graupel_full_columns(ref_lib, built)
        test_fused_graupel_ivstart_offset_and_kstart(ref_lib, built)
        test_fused_graupel_other_shape_and_dt(ref_lib, built)
        test_fused_graupel_random_states(ref_lib, built)
        test_fused_layout_matches_original_layout(ref_lib, Path(tmp))
