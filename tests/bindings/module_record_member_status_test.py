# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""``ALLOCATED`` of MODULE storage the kernel only tests, as in QE h_psi.

* An allocatable member of a module record (``dfftp_exx % nsp``): the status read becomes the tracker symbol
  ``<record>_<member>_allocated``, seeded from the host member itself -- the member's local copy is always
  allocated, so testing it would always say "allocated".
* A whole module allocatable array of records the kernel never indexes (``.NOT. ALLOCATED(comm_recv)``): the
  tracker ``<array>_allocated`` carries the module provenance and is seeded from ``allocated(<array>)``.
"""

import ctypes
from pathlib import Path

from dace_fortran.bindings.build_fortran_library import build_fortran_library
from tests._util import build_sdfg

_MODULE = """
module fftmod
  type :: desc
    integer, allocatable :: nsp(:)
    integer :: n
  end type
  type(desc) :: dfft
end module fftmod
"""

_KERNEL = (
    _MODULE
    + """
module m
contains
  subroutine k(out)
    use fftmod, only: dfft
    real(8), intent(out) :: out
    if (allocated(dfft % nsp)) then
      out = 1
    else
      out = 0
    end if
  end subroutine
end module m
"""
)

# The host caller sets the member's allocation status, then calls the binding.
_DRIVER = """
subroutine run_c(flag, out) bind(c, name="run_c")
  use, intrinsic :: iso_c_binding
  use fftmod, only: dfft
  use k_dace_bindings, only: k_dace, k_dace_finalize
  integer(c_int), value :: flag
  real(c_double), intent(out) :: out
  if (allocated(dfft % nsp)) deallocate(dfft % nsp)
  if (flag /= 0) allocate(dfft % nsp(3))
  call k_dace(out)
  call k_dace_finalize()
end subroutine
"""


def test_kernel_sees_the_host_members_allocation_status(tmp_path: Path):
    sdfg = build_sdfg(_KERNEL, tmp_path / "sdfg", name="k", entry="m::k").build()
    sdfg.name = "k"  # the driver uses k_dace_bindings; the test builder suffixes the name per test
    assert "dfft_nsp_allocated" in sdfg.symbols
    module, driver = tmp_path / "mod.f90", tmp_path / "driver.f90"
    module.write_text(_MODULE)
    driver.write_text(_DRIVER)
    lib = build_fortran_library(sdfg, out_dir=tmp_path / "lib", prelude_sources=[module], extra_sources=[driver])

    run = lib.load().run_c
    run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    for allocated, expected in ((1, 1.0), (0, 0.0), (1, 1.0)):
        out = ctypes.c_double(-1.0)
        run(allocated, ctypes.byref(out))
        assert out.value == expected, f"host member allocated={allocated}: kernel saw {out.value}"


_RECORDS_MODULE = """
module commmod
  type :: packet
    integer :: size
    real(8), allocatable :: msg(:)
  end type
  type(packet), allocatable :: comm(:, :)
end module commmod
"""

_RECORDS_KERNEL = (
    _RECORDS_MODULE
    + """
module m
contains
  subroutine k(out)
    use commmod, only: comm
    real(8), intent(out) :: out
    if (.not. allocated(comm)) then
      out = 1
    else
      out = 0
    end if
  end subroutine
end module m
"""
)

_RECORDS_DRIVER = """
subroutine run_records_c(flag, out) bind(c, name="run_records_c")
  use, intrinsic :: iso_c_binding
  use commmod, only: comm
  use kr_dace_bindings, only: kr_dace, kr_dace_finalize
  integer(c_int), value :: flag
  real(c_double), intent(out) :: out
  if (allocated(comm)) deallocate(comm)
  if (flag /= 0) allocate(comm(2, 3))
  call kr_dace(out)
  call kr_dace_finalize()
end subroutine
"""


def test_kernel_sees_the_host_record_arrays_allocation_status(tmp_path: Path):
    sdfg = build_sdfg(_RECORDS_KERNEL, tmp_path / "sdfg", name="kr", entry="m::k").build()
    sdfg.name = "kr"  # own library name: both tests' libraries load into one process
    assert "comm_allocated" in sdfg.symbols
    module, driver = tmp_path / "mod.f90", tmp_path / "driver.f90"
    module.write_text(_RECORDS_MODULE)
    driver.write_text(_RECORDS_DRIVER)
    lib = build_fortran_library(sdfg, out_dir=tmp_path / "lib", prelude_sources=[module], extra_sources=[driver])

    run = lib.load().run_records_c
    run.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
    for allocated, expected in ((1, 0.0), (0, 1.0), (1, 0.0)):
        out = ctypes.c_double(-1.0)
        run(allocated, ctypes.byref(out))
        assert out.value == expected, f"host array allocated={allocated}: kernel saw {out.value}"


if __name__ == "__main__":
    import tempfile

    for test in (
        test_kernel_sees_the_host_members_allocation_status,
        test_kernel_sees_the_host_record_arrays_allocation_status,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            test(Path(tmp))
