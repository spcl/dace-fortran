"""Dycore + sibling-SDFG e2e for a VALUE-RECORD-ARRAY struct member: numerical counterpart to the
build-only ABI pin in marshal_shim_abi_alignment_test.py. A derived-type member that is an allocatable
array of a value record (box<heap<array<record>>>) -- the shape of ICON's t_patch%edges%primal_normal_cell
-- which hlfir-marshal-external-structs v2 expands into one per-record-field SoA leaf each (pnc_v1/pnc_v2).
This test RUNS both paths (not just checks ABI order) and verifies the fields route BIT-EXACT: a
mis-scattered SoA leaf or off-by-one stride changes the numerical output and trips here.

Inner SDFG (inner_vrec) reads the value-record fields, built bind_c_shim=True (per-field SoA C ABI).
Outer SDFG (outer_vrec) scales around a call to inner_vrec as a keep_external(c_abi='per_member_soa')
so a dropped/miswired external changes the result. Reference is gfortran-compiled untransformed kernels
+ a hand bind(c) driver sharing the exact same C ABI, scattering/gathering SoA v1/v2 into AoS pnc(i)%v1/%v2.

-O0 -fno-fast-math -ffp-contract=off is pinned across DaCe codegen, the binding link, and the gfortran
reference so arithmetic order matches and the comparison (assert_array_equal) is genuinely bit-exact,
not tolerance-based.
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

import dace
from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import FlattenPlan, build_fortran_library
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# matches the reference's arithmetic order across all three build layers: DaCe's default -O3
# -ffast-math would contract a*b+c into an FMA (~1 ULP drift), breaking the bit-exact comparison.
_O0_FFLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none")
_O0_CXX_FLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-Wno-unused-parameter")

_TYPES_SRC = """
module m_vrec
  use iso_c_binding
  implicit none
  type :: tv
    real(c_double) :: v1
    real(c_double) :: v2
  end type
  type :: patch_t
    real(c_double), allocatable :: out(:)
    type(tv),       allocatable :: pnc(:)
  end type
end module
"""

# inner: reads BOTH value-record fields with distinct coefficients, so a v1/v2 SoA mis-scatter changes the result.
_INNER_SRC = """
subroutine inner_vrec(p)
  use m_vrec
  implicit none
  type(patch_t), intent(inout) :: p
  integer :: i
  do i = 1, size(p%out)
    p%out(i) = p%out(i) + 2.0d0 * p%pnc(i)%v1 - 3.0d0 * p%pnc(i)%v2
  end do
end subroutine inner_vrec
"""

# outer (dycore stand-in): scales out by 10 before / 0.1 after the inner call, so the external sits
# between two transforms the reference must reproduce exactly. Value record forwarded untouched.
_OUTER_SRC = """
subroutine outer_vrec(p)
  use m_vrec
  implicit none
  type(patch_t), intent(inout) :: p
  integer :: i
  interface
    subroutine inner_vrec(p)
      use m_vrec
      type(patch_t), intent(inout) :: p
    end subroutine
  end interface
  do i = 1, size(p%out)
    p%out(i) = 10.0d0 * p%out(i)
  end do
  call inner_vrec(p)
  do i = 1, size(p%out)
    p%out(i) = 0.1d0 * p%out(i)
  end do
end subroutine outer_vrec
"""

# Reference C-ABI driver matching outer_vrec_c's exact shim entry. `out` is a plain allocatable
# member so the shim carries its lower bound (out_lb0) ahead of the extent (bind_c_shim reconstructs
# every dynamic member at its TRUE bounds); the value-record `pnc` crosses as per-field SoA leaves
# (v1/v2) with no lower-bound slot. Scatters the SoA fields into AoS pnc(i)%v1/%v2 before the call,
# gathers them back after -- the copy _emit_value_record_array does automatically on the marshalled path.
_REF_DRIVER_SRC = """
subroutine outer_vrec_c(out_lb0, out_d0, out_p, v1_d0, v1_p, v2_d0, v2_p) bind(c, name="outer_vrec_c")
  use iso_c_binding
  use m_vrec
  implicit none
  integer(c_int), value :: out_lb0, out_d0, v1_d0, v2_d0
  type(c_ptr), value :: out_p, v1_p, v2_p
  type(patch_t), target :: p
  real(c_double), pointer :: out(:), v1(:), v2(:)
  integer :: i
  external :: outer_vrec
  call c_f_pointer(out_p, out, [out_d0])
  call c_f_pointer(v1_p, v1, [v1_d0])
  call c_f_pointer(v2_p, v2, [v2_d0])
  ! ``out`` reconstructed at its true bounds ``(lb : lb + d - 1)``; the
  ! whole-array copy is by position (shape-conformable), so ``p%out(out_lb0)``
  ! takes ``out(1)``.  ``pnc`` is 1-based (no lb slot).
  allocate(p%out(out_lb0 : out_lb0 + out_d0 - 1), p%pnc(out_d0))
  p%out = out
  do i = 1, out_d0
    p%pnc(i)%v1 = v1(i)
    p%pnc(i)%v2 = v2(i)
  end do
  call outer_vrec(p)
  out = p%out
  do i = 1, out_d0
    v1(i) = p%pnc(i)%v1
    v2(i) = p%pnc(i)%v2
  end do
end subroutine outer_vrec_c
"""


def _build_wrap(tmp_path: Path, tag: str, src: str, entry: str):
    """Build entry from src into a bind_c_shim .so; return the lib."""
    d = tmp_path / tag
    (d / "sdfg").mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, d / "sdfg", name=entry, entry=entry).build()
    sdfg.name = entry
    sdfg.build_folder = str(d / "dacecache")
    iface = build_auto_interface(sdfg._fortran_interface_raw, entry)
    plan = FlattenPlan.from_dict(sdfg._flatten_plan_raw or {})
    types_f90 = d / "lib_types.f90"
    types_f90.write_text(_TYPES_SRC)
    return build_fortran_library(sdfg,
                                 iface=iface,
                                 plan=plan,
                                 out_dir=str(d / "lib"),
                                 name=f"{entry}_wrap",
                                 bind_c_shim=True,
                                 prelude_sources=[types_f90],
                                 flags=_O0_FFLAGS)


def test_dycore_struct_ext_value_record_array_e2e(tmp_path: Path):
    """Outer SDFG calls inner SDFG passing patch_t whose pnc member is an allocatable array of the
    value record tv{v1,v2}; marshal expansion crosses it as per-field SoA leaves (pnc_v1/pnc_v2),
    numerical output is bit-exact against the gfortran reference."""
    _orig_cxx = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args", value=" ".join(_O0_CXX_FLAGS))
    clear_external_registry()
    try:
        # ---- 1. Inner value-record kernel -> bind_c_shim .so ----
        inner_lib = _build_wrap(tmp_path, "inner", _TYPES_SRC + _INNER_SRC, "inner_vrec")
        assert inner_lib.bind_c_shim_f90 is not None

        # ---- 2. Register inner as a per-member-SoA external ----
        keep_external(
            "inner_vrec",
            c_name="inner_vrec_c",
            args=(Arg(kind="aos", intent="inout", c_abi="per_member_soa"), ),
            libraries=(str(inner_lib.so_path), ),
            dynamic_extents_abi=True,
        )
        # ---- 3. Outer dycore kernel -> bind_c_shim .so (dispatches to inner) ----
        outer_lib = _build_wrap(tmp_path, "outer", _TYPES_SRC + _INNER_SRC + _OUTER_SRC, "outer_vrec")
    finally:
        clear_external_registry()
        dace.Config.set("compiler", "cpu", "args", value=_orig_cxx)

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    # ---- 4. gfortran reference ----
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "m_vrec.f90").write_text(_TYPES_SRC)
    (ref_dir / "inner_vrec.f90").write_text(_INNER_SRC)
    (ref_dir / "outer_vrec.f90").write_text(_OUTER_SRC)
    (ref_dir / "ref_driver.f90").write_text(_REF_DRIVER_SRC)
    ref_so = ref_dir / "libouter_vrec_ref.so"
    gfortran_compile_so(ref_so,
                        ref_dir / "m_vrec.f90",
                        ref_dir / "inner_vrec.f90",
                        ref_dir / "outer_vrec.f90",
                        ref_dir / "ref_driver.f90",
                        mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # ---- 5. Drive both through the shared shim C ABI + compare bit-exact ----
    n = 8
    rng = np.random.default_rng(23)
    out_init = np.asfortranarray(rng.standard_normal(n))
    v1_init = np.asfortranarray(rng.standard_normal(n))
    v2_init = np.asfortranarray(rng.standard_normal(n))
    out_sdfg, v1_sdfg, v2_sdfg = (a.copy(order="F") for a in (out_init, v1_init, v2_init))
    out_ref, v1_ref, v2_ref = (a.copy(order="F") for a in (out_init, v1_init, v2_init))

    # ABI: out_lb0, out_d0, out_p, v1_d0, v1_p, v2_d0, v2_p -- out (plain allocatable) carries a
    # lower bound ahead of its extent; the per-field value-record leaves carry only an extent.
    argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    ]
    for so in (sdfg_so, ref_lib):
        so.outer_vrec_c.restype = None
        so.outer_vrec_c.argtypes = argtypes
    sdfg_so.outer_vrec_c(1, n, out_sdfg.ctypes.data, n, v1_sdfg.ctypes.data, n, v2_sdfg.ctypes.data)
    ref_lib.outer_vrec_c(1, n, out_ref.ctypes.data, n, v1_ref.ctypes.data, n, v2_ref.ctypes.data)

    # out_final = 0.1*(10*out + 2*v1 - 3*v2) = out + 0.2*v1 - 0.3*v2 (v1/v2 read-only).
    expected = out_init + 0.2 * v1_init - 0.3 * v2_init
    np.testing.assert_allclose(out_ref, expected, rtol=1e-12, atol=1e-12)
    # real gate: SDFG (per-field SoA marshal) is bit-exact vs the gfortran AoS reference; read-only fields untouched.
    np.testing.assert_array_equal(out_sdfg, out_ref)
    np.testing.assert_array_equal(v1_sdfg, v1_ref)
    np.testing.assert_array_equal(v2_sdfg, v2_ref)
