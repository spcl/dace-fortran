"""Standalone Fortran dycore + no-op sync external: bit-exact gfortran reference vs SDFG.

Minimal companion to the velocity dycore + external e2e: one Fortran source bundle
(``mo_sync_helpers_noop`` + ``mo_standalone_dycore``) with a sync subroutine WITHOUT
``bind(c)`` (routed to via ``keep_external`` through a hand-authored ``bind(c)``
wrapper) and a dycore subroutine that calls it twice around an in-place stencil
computation.  Compiled twice (gfortran reference, DaCe HLFIR bridge SDFG), both driven
from ctypes with random input, output buffers compared bit-exact.

The sync body is a no-op so the comparison catches divergence in the SDFG's lowering of
the computation itself or the external-call routing (the bridge must not optimise
``keep_external`` away, but the no-op keeps data untouched for a clean assertion).

Companion to ``test_dycore_velocity_external_e2e.py`` (full ICON-style sync prints + AoS
marshalling) and ``test_dycore_struct_ext_e2e.py`` (small struct-external sibling).
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import dace
import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import build_fortran_library
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external

# ``-O0 -fno-fast-math -ffp-contract=off`` pinned on every layer (gfortran reference
# link, SDFG-side gfortran link of the binding wrapper, DaCe's C++ codegen -- overriding
# DaCe's default ``-O3 -march=native -ffast-math`` which would FMA-contract ``a*b + c``
# and skew the comparison by ~1 ULP) is what makes the 1-ULP check below tight enough to
# catch a real codegen regression while absorbing residual conversion-ordering at the
# ``sqrt(real(i+j+k, c_double))`` site.
_O0_FFLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none")
_O0_CXX_FLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-Wno-unused-parameter", "-Wno-unused-label")

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# "original" sync_patch_array_noop has no bind(c) (dycore calls it via the regular
# Fortran interface); the companion sync_patch_array_noop_c IS bind(c) and rebuilds the
# assumed-shape descriptor from flat C-ABI args before forwarding.
_SYNC_NOOP_SRC = """
module mo_sync_helpers_noop
  use iso_c_binding
  implicit none
contains
  subroutine sync_patch_array_noop(tag, field)
    integer(c_int), intent(in) :: tag
    real(c_double), intent(inout) :: field(:, :, :)
    ! Intentionally a no-op so the SDFG-vs-gfortran comparison stays
    ! bit-exact while still exercising the external-call routing.
  end subroutine sync_patch_array_noop

  subroutine sync_patch_array_noop_c(tag, d0, d1, d2, field_p) &
    bind(c, name='sync_patch_array_noop_c')
    integer(c_int), value :: tag, d0, d1, d2
    type(c_ptr), value :: field_p
    real(c_double), pointer :: field_local(:, :, :)
    call c_f_pointer(field_p, field_local, [d0, d1, d2])
    call sync_patch_array_noop(tag, field_local)
  end subroutine sync_patch_array_noop_c
end module mo_sync_helpers_noop
"""

# two sync_patch_array_noop CALLs flank a triply-nested in-place computation; shape comes
# from size(field, dim=...) at runtime, so bind_c_shim generates a dynamic-extents ABI.
_DYCORE_SRC = """
module mo_standalone_dycore
  use iso_c_binding
  use mo_sync_helpers_noop, only: sync_patch_array_noop
  implicit none
contains
  subroutine standalone_dycore(field, alpha)
    real(c_double), intent(inout) :: field(:, :, :)
    real(c_double), intent(in) :: alpha
    integer :: i, j, k

    call sync_patch_array_noop(1_c_int, field)
    do k = 1, size(field, 3)
      do j = 1, size(field, 2)
        do i = 1, size(field, 1)
          field(i, j, k) = field(i, j, k) * alpha + &
                           sqrt(real(i + j + k, c_double))
        end do
      end do
    end do
    call sync_patch_array_noop(2_c_int, field)
  end subroutine standalone_dycore
end module mo_standalone_dycore
"""

# mirrors bind_c_shim's convention exactly (extents first, pointer next, scalars last)
# so the same ctypes call site drives both the reference and the SDFG-emitted wrapper.
_REF_DRIVER_SRC = """
subroutine standalone_dycore_ref_c(field_d0, field_d1, field_d2, &
                                   field_p, alpha) &
  bind(c, name='standalone_dycore_ref_c')
  use iso_c_binding
  use mo_standalone_dycore, only: standalone_dycore
  integer(c_int), value :: field_d0, field_d1, field_d2
  type(c_ptr), value :: field_p
  real(c_double), value :: alpha
  real(c_double), pointer :: field(:, :, :)
  call c_f_pointer(field_p, field, [field_d0, field_d1, field_d2])
  call standalone_dycore(field, alpha)
end subroutine standalone_dycore_ref_c
"""


def _build_sync_lib(tmp_path: Path) -> tuple[Path, Path]:
    """Pre-compile the sync no-op library so the SDFG's link can resolve
    ``sync_patch_array_noop_c``.  Returns ``(.so path, build dir)``; the build dir
    doubles as a ``-J`` for a later consumer needing ``mo_sync_helpers_noop.mod``."""
    build_dir = tmp_path / "sync_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    src = build_dir / "sync_noop.f90"
    src.write_text(_SYNC_NOOP_SRC)
    so_path = build_dir / "libsync_noop.so"
    subprocess.check_call(
        ["gfortran", "-shared", "-fPIC", "-O2", f"-J{build_dir}",
         str(src), "-o", str(so_path)], cwd=build_dir)
    return so_path, build_dir


def test_standalone_dycore_with_sync_external(tmp_path: Path):
    """Compile the same Fortran dycore source bundle two ways (gfortran reference,
    SDFG via ``keep_external`` for the no-``bind(c)`` sync routine) and drive both
    from ctypes with the same random input; bit-exact match confirms the SDFG's
    stencil lowering + external-call routing."""
    # 1. Pre-build the no-op sync library.
    sync_so, sync_build_dir = _build_sync_lib(tmp_path)

    # 2. register BEFORE the SDFG build so the bridge externalises it instead of inlining.
    clear_external_registry()
    keep_external(
        "sync_patch_array_noop",
        c_name="sync_patch_array_noop_c",
        args=(
            Arg(kind="scalar", dtype="int32", intent="in"),  # tag
            Arg(kind="array", dtype="float64", intent="inout"),  # field
        ),
        libraries=(str(sync_so), ),
        dynamic_extents_abi=True,
    )
    # 2b. pin DaCe's C++ codegen to O0/no-fast-math so arithmetic order matches gfortran;
    # save + restore so later tests in this session aren't affected.
    _orig_cxx_args = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args", value=" ".join(_O0_CXX_FLAGS))
    try:
        # 3. Build the SDFG of the standalone dycore + bind_c_shim.
        sdfg_dir = tmp_path / "sdfg"
        sdfg_dir.mkdir(parents=True, exist_ok=True)
        # bridge needs the sync module's body to type-check the `use` in mo_standalone_dycore.
        full_src = _SYNC_NOOP_SRC + _DYCORE_SRC
        sdfg = build_sdfg(full_src, sdfg_dir, name="standalone_dycore", entry="standalone_dycore").build()
        sdfg.name = "standalone_dycore"
        sdfg.build_folder = str(sdfg_dir / "dacecache")
        iface = build_auto_interface(sdfg._fortran_interface_raw, "standalone_dycore")
        # sync source must also be a prelude so bind_c_shim's `use` resolves at link time.
        sync_prelude = sdfg_dir / "sync_noop.f90"
        sync_prelude.write_text(_SYNC_NOOP_SRC)
        sdfg_lib = build_fortran_library(
            sdfg,
            iface=iface,
            out_dir=str(tmp_path / "sdfg_lib"),
            name="standalone_dycore_wrap",
            prelude_sources=[sync_prelude],
            bind_c_shim=True,
            # same FP-conservative flags as the reference path -- ABI translation must
            # not itself contribute ULP-level drift.
            flags=_O0_FFLAGS,
        )
    finally:
        clear_external_registry()
        dace.Config.set("compiler", "cpu", "args", value=_orig_cxx_args)
    assert sdfg_lib.bind_c_shim_f90 is not None

    sdfg_so_lib = ctypes.CDLL(str(sdfg_lib.so_path))

    # 4. gfortran reference: same source, plus the hand-authored bind(c) driver.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    sync_ref = ref_dir / "sync_noop.f90"
    sync_ref.write_text(_SYNC_NOOP_SRC)
    dycore_ref = ref_dir / "dycore.f90"
    dycore_ref.write_text(_DYCORE_SRC)
    driver_ref = ref_dir / "driver.f90"
    driver_ref.write_text(_REF_DRIVER_SRC)
    ref_so = ref_dir / "libdycore_ref.so"
    gfortran_compile_so(ref_so, sync_ref, dycore_ref, driver_ref, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # 5. Fortran-order arrays so the contiguous-stride pattern matches both C ABIs.
    rng = np.random.default_rng(42)
    n0, n1, n2 = 11, 7, 5
    alpha = 2.5
    field_init = np.asfortranarray(rng.standard_normal((n0, n1, n2)))
    field_sdfg = field_init.copy(order='F')
    field_ref = field_init.copy(order='F')

    # bind_c_shim's flat-arg convention: dynamic extents ahead of the pointer, scalars
    # last -- the ref driver mirrors this so the same argtypes apply to both.
    argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_double]
    sdfg_fn = sdfg_so_lib.standalone_dycore_c
    sdfg_fn.restype = None
    sdfg_fn.argtypes = argtypes
    ref_fn = ref_lib.standalone_dycore_ref_c
    ref_fn.restype = None
    ref_fn.argtypes = argtypes

    sdfg_fn(n0, n1, n2, field_sdfg.ctypes.data, ctypes.c_double(alpha))
    ref_fn(n0, n1, n2, field_ref.ctypes.data, ctypes.c_double(alpha))

    # 1-ULP tight: the two paths are bit-identical today (verified rtol=atol=0).
    # rtol=2**-52 is a safety buffer for a future flang/gfortran reordering at
    # sqrt(real(i+j+k, c_double)) without masking a real codegen regression.
    one_ulp_rtol = 2**-52  # ~2.22e-16
    np.testing.assert_allclose(field_sdfg, field_ref, rtol=one_ulp_rtol, atol=0.0)
    # hard guarantee today: bit-exact.  Loosen to assert_allclose if a future flang
    # reorders the a*b + sqrt(c) chain and only 1 ULP can be sustained.
    np.testing.assert_array_equal(field_sdfg, field_ref)
