"""fparser single-TU round-trip for the AES graupel scheme: inline the 4-module ``aes_graupel`` project into one ``.f90`` via fparser, build it through the bridge, compare numerics against a gfortran reference from the original multi-file sources.

Previously surfaced a codegen gap: the AoS-of-pointer-records gather temp (``t_qx_ptr%x``) was sized from unbound extent symbols defaulting to 1, overflowing the heap.  Fixed by ``fir.box_dims -> <name>_d<dim>`` extent resolution.  Inlining itself is asserted unconditionally up front so an inliner regression surfaces as a hard failure independent of the numerical compare.
"""
import pytest

from tests._util import have_flang
from dace_fortran import build_sdfg_from_files, inline_to_single_tu

from ._graupel_harness import (DEP_SOURCES, ENTRY, ORIGINAL_SOURCE, SCENARIOS, Config, assert_families_fire,
                               assert_match, compile_reference, copy_fields, physical_columns, run_reference, run_sdfg,
                               zero_outputs)

pytestmark = pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH")

_SOURCES = [ORIGINAL_SOURCE, *DEP_SOURCES]


def test_aes_graupel_inline_single_tu(tmp_path):
    """The 4-module project inlines into one valid, self-contained Fortran TU that still defines graupel_run."""
    out = inline_to_single_tu(_SOURCES, entry=ENTRY, out_dir=tmp_path, name="graupel_inlined")
    assert out.is_file()
    text = out.read_text()
    assert "graupel_run" in text.lower()
    # self-contained TU: every needed module is inlined, no stray external module beyond intrinsic stubs.
    assert "FUNCTION" in text.upper() or "SUBROUTINE" in text.upper()


def test_aes_graupel_inline_roundtrip_numerical(tmp_path):
    """graupel_run reference vs SDFG built from the fparser-inlined single TU: element-wise compare of every INOUT prognostic + OUT diagnostic on the physical scenario columns."""
    reference = compile_reference(tmp_path / "ref", [ORIGINAL_SOURCE])
    single_tu = inline_to_single_tu(_SOURCES, entry=ENTRY, out_dir=tmp_path / "inlined", name="graupel_run")
    sdfg = build_sdfg_from_files([single_tu], entry=ENTRY, name="graupel_run", out_dir=tmp_path / "build")

    cfg = Config(1, len(SCENARIOS), 1)
    inputs = physical_columns()
    ref, got = copy_fields(inputs), copy_fields(inputs)
    zero_outputs(ref)
    zero_outputs(got)
    run_reference(reference, ref, cfg)
    run_sdfg(sdfg, got, cfg)
    assert_families_fire(inputs, ref)
    assert_match(ref, got, 1e-10, 1e-14)
