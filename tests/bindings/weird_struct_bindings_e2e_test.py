"""E2e F90-binding coverage for a struct with MIXED-RANK members: a rank-3 array member
plus a rank-0 scalar member (the scalar lifts to a length-1 ``Array`` on the SDFG surface).
Pins the same defect class ``0318f9efe`` fixed for rank-0 scalar struct members. Checks both
the dace-generated F90 binding and the direct flat-ABI SDFG call against a gfortran
reference; FlattenPlan asserted non-empty so a silent unpack regression fails loudly."""

import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import (
    FlattenPlan,
    emit_bindings,
)
from dace_fortran.bindings.fortran_interface import build_auto_interface

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_NX, _NY, _NZ = 4, 3, 2

_TYPES_SRC = f"""
module mo_w
  use iso_c_binding
  implicit none
  integer, parameter :: NX = {_NX}, NY = {_NY}, NZ = {_NZ}
  type :: t_w
     real(c_double) :: vol(NX, NY, NZ)
     real(c_double) :: coef
  end type t_w
end module mo_w
"""

_KERNEL_SRC = """
subroutine scale3d(w)
  use mo_w
  implicit none
  type(t_w), intent(inout) :: w
  integer :: i, j, k
  do k = 1, NZ
    do j = 1, NY
      do i = 1, NX
        w%vol(i, j, k) = w%vol(i, j, k) * w%coef + real(i + j + k, c_double)
      end do
    end do
  end do
end subroutine scale3d
"""

_SRC = _TYPES_SRC + _KERNEL_SRC

_SDFG_DRIVER = """
subroutine run_scale3d(vol, coef) bind(c, name='run_scale3d')
  use iso_c_binding
  use mo_w, only: t_w, NX, NY, NZ
  use scale3d_dace_bindings
  implicit none
  real(c_double), intent(inout) :: vol(NX, NY, NZ)
  real(c_double), value :: coef
  type(t_w), target :: w
  w%vol = vol
  w%coef = coef
  call scale3d_dace(w)
  call scale3d_dace_finalize()
  vol = w%vol
end subroutine run_scale3d
"""

_REF_DRIVER = """
subroutine run_scale3d_ref(vol, coef) bind(c, name='run_scale3d_ref')
  use iso_c_binding
  use mo_w, only: t_w, NX, NY, NZ
  implicit none
  real(c_double), intent(inout) :: vol(NX, NY, NZ)
  real(c_double), value :: coef
  type(t_w) :: w
  external :: scale3d
  w%vol = vol
  w%coef = coef
  call scale3d(w)
  vol = w%vol
end subroutine run_scale3d_ref
"""


def test_e2e_mixed_rank_struct(tmp_path: Path):
    """``type(t_w)`` with a rank-3 array member ``vol`` and a rank-0 scalar member ``coef``:
    ``w%vol = w%vol * w%coef + (i+j+k)``. Binding AND direct SDFG vs gfortran (rtol=1e-12);
    FlattenPlan must be non-empty."""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    # Entry kept BARE: scale3d is a free subroutine; a module wrapper would break both the
    # gfortran reference link (`external :: scale3d`) and the emitted binding.
    builder = build_sdfg(_SRC, sdfg_dir, name="scale3d", entry="scale3d")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = "scale3d"
    # Auto-derived caller interface (rank-0 ``type(t_w)`` dummy from mo_w).
    iface = build_auto_interface(sdfg._fortran_interface_raw, "scale3d")
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    flat_targets = {fn for e in plan.entries for fn in e.recipe.flat_names}
    assert plan.entries, "mixed-rank struct dummy must produce a non-empty FlattenPlan"
    assert any("vol" in t for t in flat_targets), flat_targets
    assert any("coef" in t for t in flat_targets), flat_targets

    bindings_path = tmp_path / "scale3d_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, plan, str(bindings_path))
    types_path = tmp_path / "scale3d_types.f90"
    types_path.write_text(_TYPES_SRC)
    drv_path = tmp_path / "scale3d_driver.f90"
    drv_path.write_text(_SDFG_DRIVER)

    build_dir = tmp_path / "bind_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    drv_so = build_dir / "scale3d_drv.so"
    gfortran_compile_so(drv_so, types_path, bindings_path, drv_path, mod_dir=build_dir, link_so=so_path)
    lib = ctypes.CDLL(str(drv_so))

    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    rt = ref_dir / "scale3d_types.f90"
    rt.write_text(_TYPES_SRC)
    rk = ref_dir / "scale3d_k.f90"
    rk.write_text(_KERNEL_SRC)
    rd = ref_dir / "scale3d_d.f90"
    rd.write_text(_REF_DRIVER)
    ref_so = ref_dir / "scale3d_ref.so"
    gfortran_compile_so(ref_so, rt, rk, rd, mod_dir=ref_dir)
    ref = ctypes.CDLL(str(ref_so))

    rng = np.random.default_rng(43)
    vol0 = np.asfortranarray(rng.standard_normal((_NX, _NY, _NZ)))
    coef = 0.625
    dp = ctypes.POINTER(ctypes.c_double)

    vol_ref = vol0.copy(order="F")
    fref = ref.run_scale3d_ref
    fref.restype = None
    fref.argtypes = [dp, ctypes.c_double]
    fref(vol_ref.ctypes.data_as(dp), ctypes.c_double(coef))

    vol_bind = vol0.copy(order="F")
    fbind = lib.run_scale3d
    fbind.restype = None
    fbind.argtypes = [dp, ctypes.c_double]
    fbind(vol_bind.ctypes.data_as(dp), ctypes.c_double(coef))
    np.testing.assert_allclose(vol_bind, vol_ref, rtol=1e-12, atol=1e-12)

    # Direct SDFG path: struct members as separate flat companions. HLFIR-bridged arrays
    # keep Fortran (column-major) layout, so the companion must be F-ordered.
    vol_d = vol0.copy(order="F")
    coef_d = np.array([coef], dtype=np.float64)
    sdfg(w_vol=vol_d, w_coef=coef_d)
    np.testing.assert_allclose(vol_d, vol_ref, rtol=1e-12, atol=1e-12)
