# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenMP-offload binding, end to end on an AMD GPU.

A host Fortran caller hands its buffers to the generated binding; the SDFG behind it was offloaded (its arrays live
in ``GPU_Global``, its maps run on the device).  With ``Directive.OPENMP`` the binding stages the buffers with
``!$omp target enter/exit data`` and hands the SDFG their device addresses through ``use_device_addr`` -- built here
with ROCm's ``amdflang -fopenmp --offload-arch=<gfx>``, the compiler that has no OpenACC offload.

Needs an AMD GPU, ``amdflang`` and DaCe's HIP backend: run it in the agent-mi300 container (``-m gpu``).
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dace

from dace_fortran.bindings.acc_transfers import Directive
from dace_fortran.bindings.build_fortran_library import CompilerFamily, FortranCompiler, build_fortran_library
from dace_fortran.bindings.frozen_signature import refreeze
from dace_fortran.pipelines import num_maps, optimize
from tests._util import build_sdfg

pytestmark = pytest.mark.gpu

_KERNEL = """
module kern
contains
  subroutine axpy(n, a, x, y)
    integer, intent(in) :: n
    real(8), intent(in) :: a
    real(8), intent(in) :: x(n)
    real(8), intent(inout) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = a * x(i) + y(i)
    end do
  end subroutine axpy
end module kern
"""

# The host caller: ordinary Fortran holding host arrays, calling the generated binding.
_DRIVER = """
subroutine run_axpy_c(n, a, x, y) bind(c, name="run_axpy_c")
  use, intrinsic :: iso_c_binding
  use axpy_dace_bindings, only: axpy_dace, axpy_dace_finalize
  implicit none
  integer(c_int), value :: n
  real(c_double), value :: a
  real(c_double), intent(in) :: x(n)
  real(c_double), intent(inout) :: y(n)
  call axpy_dace(n, a, x, y)
  call axpy_dace_finalize()
end subroutine run_axpy_c
"""


def _offload_arch() -> str:
    """The first GPU's ``gfx`` target, as ROCm's ``amdgpu-arch`` reports it."""
    tool = shutil.which("amdgpu-arch") or "/opt/rocm/llvm/bin/amdgpu-arch"
    archs = subprocess.run([tool], capture_output=True, text=True, check=True).stdout.split()
    assert archs, "amdgpu-arch reports no AMD GPU"
    return archs[0]


def test_openmp_offload_binding_runs_the_offloaded_sdfg(tmp_path: Path):
    """``y = a*x + y`` on the GPU through an OpenMP-offload binding matches numpy, and ``x`` is left untouched."""
    amdflang = shutil.which("amdflang")
    assert amdflang is not None, "the gpu lane needs ROCm's amdflang"

    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="axpy", entry="kern::axpy").build()
    sdfg.name = "axpy"  # the driver uses axpy_dace_bindings; the test builder suffixes the name per test
    optimize(sdfg)
    assert num_maps(sdfg) > 0, "the loop did not become a map, so nothing would run on the GPU"
    for desc in sdfg.arrays.values():
        if not desc.transient and isinstance(desc, dace.data.Array):
            desc.storage = dace.StorageType.GPU_Global
    sdfg.apply_gpu_transformations()
    refreeze(sdfg)

    driver = tmp_path / "driver.f90"
    driver.write_text(_DRIVER)
    lib = build_fortran_library(
        sdfg,
        out_dir=tmp_path / "lib",
        extra_sources=[driver],
        directive=Directive.OPENMP,
        fortran_compiler=FortranCompiler(amdflang, CompilerFamily.LLVM),
        extra_flags=[f"--offload-arch={_offload_arch()}"],
    )
    text = lib.bindings_f90.read_text()
    assert "!$omp target enter data" in text and "!$ACC" not in text

    run = lib.load().run_axpy_c
    run.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
    run.restype = None
    rng = np.random.default_rng(0)
    n, a = 1000, 2.5
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    x0, expected = x.copy(), a * x + y
    run(n, a, x.ctypes.data, y.ctypes.data)

    np.testing.assert_array_equal(x, x0)
    np.testing.assert_allclose(y, expected, rtol=1e-14, atol=0)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_openmp_offload_binding_runs_the_offloaded_sdfg(Path(tmp))
