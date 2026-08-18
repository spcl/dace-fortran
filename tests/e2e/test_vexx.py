# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""QE ``exx_bp::vexx_bp_k_gpu`` through the full parallelization pipeline.

Same build and comparison as ``qe/exx_bp/test_vexx_bp_k_gpu_parse::test_vexx_bp_k_gpu_numerical_
correctness`` (see it for why the e2e path goes through the generated binding rather than calling
DaCe directly), with ``pipelines.optimize`` inserted between the SDFG build and the binding build.
That test's helpers are imported rather than duplicated, so the fixture stays in one place.

Two bars on one set of seeded random inputs:

* pre-optimize binding vs post-optimize binding, BIT-EXACT -- the pipeline differential;
* post-optimize binding vs the gfortran reference, to fp64 precision -- the frontend check.

The differential is the strict one. Frontend and gfortran differ in evaluation order so their
comparison can only ever be a 1e-11 one, under which a pipeline bug worth less than 1e-11 passes.
The kernel's ~80 USE-imported module globals rule out calling either SDFG directly (that is why
``pipelines.verify_numerics`` cannot be used here), so the differential is taken through two
independently built bindings instead.

No specialize: this kernel maps without it. ``scalar_fission`` is no longer a caller knob -- it
always runs as part of the pipeline.
"""
import copy
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.flatten_plan import FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.pipelines import num_maps, optimize
from qe.exx_bp import test_vexx_bp_k_gpu_parse as vexx

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


def binding_entry(sdfg, plan, src_path: Path, out: Path, lib_name: str):
    """Build ``sdfg``'s Fortran binding under ``out`` and return its loaded ``run_vexx_dace_c``.

    The names the driver imports from the generated binding -- ``<name>_dace_bindings``,
    ``<name>_dace``, ``<name>_dace_finalize`` -- all derive from the SDFG name, so retargeting the
    driver at a renamed SDFG is one substitution on the ``vexx_bp_k_gpu_dace`` stem. It must NOT be
    the bare kernel stem: ``init_vexx_bp_k_gpu_state_c`` is defined by ``_CALLER``, not by the
    binding, and renaming its declaration leaves the .so with an undefined symbol.

    The rename is what keeps the two builds apart: DaCe keys its build folder on the SDFG name, so
    two same-named SDFGs would compile over each other.
    """
    out.mkdir(parents=True, exist_ok=True)
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)
    driver_path = out / "driver.f90"
    driver_path.write_text(vexx._SDFG_DRIVER.replace("vexx_bp_k_gpu_dace", f"{sdfg.name}_dace"))
    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(out / "lib"),
        name=lib_name,
        prelude_sources=[src_path],
        extra_sources=[vexx._CALLER, driver_path],
        # pruned qvan2 has an implicit-interface COMPLEX->REAL arg-kind mismatch (qg) behind
        # IF(okvan) -- never run on the no-op path.
        extra_flags=["-fallow-argument-mismatch"])

    fn = lib.load().run_vexx_dace_c
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_void_p]
    return fn


def test_vexx_pipeline_numerical_e2e(tmp_path, e2e_cpu_args):
    """Pre- vs post-optimize bit-exact, and post-optimize vs gfortran to fp64 precision."""
    lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1

    # gfortran reference (identity on the no-op path -- see the borrowed test's docstring).
    _, init, run = vexx._compile_reference(tmp_path)
    init(lda, n, m, npol, max_ibands)
    psi_ref, hpsi_ref = vexx._make_random_inputs(lda, npol, max_ibands)
    psi_pre, hpsi_pre = psi_ref.copy(order="F"), hpsi_ref.copy(order="F")
    psi_dace, hpsi_dace = psi_ref.copy(order="F"), hpsi_ref.copy(order="F")
    run(lda, n, m, psi_ref.ctypes.data, hpsi_ref.ctypes.data)

    src = vexx._make_paw_flag_public(vexx._restore_fft_interfaces(vexx._SRC.read_text()))
    src_path = tmp_path / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, tmp_path / "sdfg", name="vexx_bp_k_gpu", entry=vexx._ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()

    # Snapshot before the pipeline touches anything; the rename gives it its own build folder.
    preopt = copy.deepcopy(sdfg)
    preopt.name = "vexx_bp_k_gpu_preopt"
    refreeze(preopt)

    optimize(sdfg)
    # Re-snapshot the binding signature against the optimized SDFG: args must stay ABI-identical,
    # free symbols may only shrink (refreeze raises otherwise).
    refreeze(sdfg)
    assert num_maps(sdfg) > 0, "pipeline produced no maps -- nothing was parallelized"

    sdfg.name = "vexx_bp_k_gpu"
    pre_fn = binding_entry(preopt, plan, src_path, tmp_path / "preopt", "vexx_lib_preopt")
    pre_fn(lda, n, m, npol, max_ibands, psi_pre.ctypes.data, hpsi_pre.ctypes.data)

    opt_fn = binding_entry(sdfg, plan, src_path, tmp_path / "opt", "vexx_lib")
    opt_fn(lda, n, m, npol, max_ibands, psi_dace.ctypes.data, hpsi_dace.ctypes.data)

    # The pipeline reorders statements and forms maps but never reassociates arithmetic, and the
    # e2e flags pin FMA contraction off -- so anything short of bit-identical is a bug.
    assert np.array_equal(hpsi_dace, hpsi_pre), ("optimize() changed the numerics (must be bit-exact): "
                                                 f"max|d|={np.abs(hpsi_dace - hpsi_pre).max():.6e}")

    np.testing.assert_allclose(hpsi_dace, hpsi_ref, rtol=1e-11, atol=1e-11)
