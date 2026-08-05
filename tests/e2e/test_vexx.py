# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""QE ``exx_bp::vexx_bp_k_gpu`` through the full parallelization pipeline.

Same build and comparison as ``qe/exx_bp/test_vexx_bp_k_gpu_parse::test_vexx_bp_k_gpu_numerical_
correctness`` (see it for why the e2e path goes through the generated binding rather than calling
DaCe directly), with ``pipelines.optimize`` inserted between the SDFG build and the binding build.
That test's helpers are imported rather than duplicated, so the fixture stays in one place.

No specialize and no scalar-fission: this kernel maps without either.
"""
import ctypes
import shutil

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
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


def test_vexx_pipeline_numerical_e2e(tmp_path, e2e_cpu_args):
    """Optimized SDFG binding output == gfortran reference, to fp64 precision."""
    lda, n, m, npol, max_ibands = 4, 4, 1, 1, 1

    # gfortran reference (identity on the no-op path -- see the borrowed test's docstring).
    _, init, run = vexx._compile_reference(tmp_path)
    init(lda, n, m, npol, max_ibands)
    psi_ref, hpsi_ref = vexx._make_random_inputs(lda, npol, max_ibands)
    psi_dace = psi_ref.copy(order="F")
    hpsi_dace = hpsi_ref.copy(order="F")
    run(lda, n, m, psi_ref.ctypes.data, hpsi_ref.ctypes.data)

    # SDFG-via-binding, WITH the pipeline between the SDFG build and the interface build.
    src = vexx._make_paw_flag_public(vexx._restore_fft_interfaces(vexx._SRC.read_text()))
    src_path = tmp_path / "qe.f90"
    src_path.write_text(src)

    builder = build_sdfg(src, tmp_path / "sdfg", name="vexx_bp_k_gpu", entry=vexx._ENTRY)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()

    optimize(sdfg)
    # Re-snapshot the binding signature against the optimized SDFG: args must stay ABI-identical,
    # free symbols may only shrink (refreeze raises otherwise).
    refreeze(sdfg)
    assert num_maps(sdfg) > 0, "pipeline produced no maps -- nothing was parallelized"

    sdfg.name = "vexx_bp_k_gpu"
    iface = build_auto_interface(sdfg._fortran_interface_raw, sdfg.name)

    driver_path = tmp_path / "driver.f90"
    driver_path.write_text(vexx._SDFG_DRIVER)
    lib = build_fortran_library(
        sdfg,
        iface,
        plan,
        str(tmp_path / "lib"),
        name="vexx_lib",
        prelude_sources=[src_path],
        extra_sources=[vexx._CALLER, driver_path],
        # pruned qvan2 has an implicit-interface COMPLEX->REAL arg-kind mismatch (qg) behind
        # IF(okvan) -- never run on the no-op path.
        extra_flags=["-fallow-argument-mismatch"])

    fn = lib.load().run_vexx_dace_c
    fn.restype = None
    fn.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p, ctypes.c_void_p]
    fn(lda, n, m, npol, max_ibands, psi_dace.ctypes.data, hpsi_dace.ctypes.data)

    np.testing.assert_allclose(hpsi_dace, hpsi_ref, rtol=1e-11, atol=1e-11)
