"""Dycore + sibling-SDFG E2E for a struct-shaped external call.

Scales the flat-arg architecture proof in test_dycore_ext_velocity_e2e.py (inner_axpy(n,a,x,y))
to a struct-shaped inner (type(state_t)) -- the velocity_tendencies shape at small scale.

What Arg(kind='aos', c_abi='per_member_soa') enables and this test pins:
  * inner's emit_bind_c_shim expands type(state_t) to one c_ptr per member.
  * outer's emit_call forwards the marshal-expanded per-member SoA flats verbatim to the C
    call site -- no stack AoS buffer, no pack/unpack copy.
  * both signatures are derived from the same Fortran state_t through the same pipeline, so
    they coincide by construction -- no hand-authored shim file.

Outer driven from Fortran via build_fortran_library bindings; reference is a gfortran linkage
of inner+outer+a bind(c) driver sharing the same C ABI.
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
    build_fortran_library,
)
from dace_fortran.external import Arg, clear_external_registry, keep_external

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# state_t: two static-shape array members -- smallest struct exercising the bind_c shim's
# 2 per-member C-ABI slots and the marshal expansion's 2 SoA flats.
_TYPES_SRC = """
module m_state
  use iso_c_binding
  implicit none
  integer, parameter :: N = 8
  type, bind(c) :: state_t
    real(c_double) :: u(N)
    real(c_double) :: v(N)
  end type
end module
"""

# Inner: u(i) += v(i) -- smallest write-pattern proving per-member SoA forwarding lands right.
_INNER_KERNEL_SRC = """
subroutine inner_state(s)
  use m_state
  implicit none
  type(state_t), intent(inout) :: s
  integer :: i
  do i = 1, N
    s%u(i) = s%u(i) + s%v(i)
  end do
end subroutine inner_state
"""

# Outer: doubles s%u, calls inner, halves s%u -- pre/post work makes the comparison sensitive
# to external-call wiring (a passthrough wouldn't catch a miswired external).
_OUTER_KERNEL_SRC = """
subroutine outer_state(s)
  use m_state
  implicit none
  type(state_t), intent(inout) :: s
  integer :: i
  interface
    subroutine inner_state(s)
      use m_state
      type(state_t), intent(inout) :: s
    end subroutine
  end interface
  do i = 1, N
    s%u(i) = 2.0d0 * s%u(i)
  end do
  call inner_state(s)
  do i = 1, N
    s%u(i) = 0.5d0 * s%u(i)
  end do
end subroutine outer_state
"""

# Reference driver: same flat layout as the emitted outer_state_dace bindings entry (one
# pointer per member, declaration order), so ctypes drives both libraries identically.
_REF_DRIVER_SRC = """
subroutine outer_state_c(u_p, v_p) bind(c, name="outer_state_c")
  use iso_c_binding
  use m_state
  implicit none
  type(c_ptr), value :: u_p, v_p
  type(state_t), target :: s
  real(c_double), pointer :: u(:), v(:)
  external :: outer_state
  call c_f_pointer(u_p, u, [N])
  call c_f_pointer(v_p, v, [N])
  s%u = u
  s%v = v
  call outer_state(s)
  u = s%u
  v = s%v
end subroutine outer_state_c
"""


def test_dycore_struct_outer_calls_inner_via_sibling_sdfg(tmp_path: Path):
    """Outer SDFG with a type(state_t) arg calls inner SDFG with the same arg shape via
    per-member SoA pointers -- no hand-authored shim; both C-ABI sides derive from the same
    Fortran source so they coincide bit-for-bit."""
    # ---- 1. Inner SDFG (the velocity_tendencies stand-in) ----
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    inner_src = _TYPES_SRC + _INNER_KERNEL_SRC
    inner_sdfg = build_sdfg(inner_src, inner_sdfg_dir, name="inner_state", entry="inner_state").build()
    inner_sdfg.name = "inner_state"
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    inner_iface = OriginalInterface(
        entry="inner_state",
        args=(OriginalArg(name="s", fortran_type="type(state_t)", rank=0, intent="inout", struct_type="state_t"), ),
        struct_types={
            "state_t":
            DerivedType(name="state_t",
                        module="m_state",
                        members=(
                            Member(name="u", fortran_type="real(c_double)", rank=1, shape=("N", )),
                            Member(name="v", fortran_type="real(c_double)", rank=1, shape=("N", )),
                        ))
        },
        used_modules={"m_state": ("state_t", "N")},
    )
    inner_types_f90 = inner_dir / "lib_types.f90"
    inner_types_f90.write_text(_TYPES_SRC)
    inner_lib = build_fortran_library(
        inner_sdfg,
        iface=inner_iface,
        out_dir=str(inner_dir / "lib"),
        name="inner_state_wrap",
        bind_c_shim=True,
        prelude_sources=[inner_types_f90],
    )
    assert inner_lib.bind_c_shim_f90 is not None

    # ---- 2. Register the inner as a per-member-SoA external ----
    # c_name is the inner's bind_c_shim entry; Arg(kind='aos', c_abi='per_member_soa') tells
    # emit_call to forward the marshal-expanded leaves verbatim, matching the shim's inputs.
    keep_external(
        "inner_state",
        c_name="inner_state_c",
        args=(Arg(kind="aos", intent="inout", c_abi="per_member_soa"), ),
        libraries=(str(inner_lib.so_path), ),
    )
    try:
        # ---- 3. Outer SDFG (the dycore stand-in) ----
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg_dir = outer_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        outer_src = _TYPES_SRC + _INNER_KERNEL_SRC + _OUTER_KERNEL_SRC
        outer_sdfg = build_sdfg(outer_src, outer_sdfg_dir, name="outer_state", entry="outer_state").build()
        outer_sdfg.name = "outer_state"
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_iface = OriginalInterface(
            entry="outer_state",
            args=(OriginalArg(name="s", fortran_type="type(state_t)", rank=0, intent="inout", struct_type="state_t"), ),
            struct_types={
                "state_t":
                DerivedType(name="state_t",
                            module="m_state",
                            members=(
                                Member(name="u", fortran_type="real(c_double)", rank=1, shape=("N", )),
                                Member(name="v", fortran_type="real(c_double)", rank=1, shape=("N", )),
                            ))
            },
            used_modules={"m_state": ("state_t", "N")},
        )
        outer_types_f90 = outer_dir / "lib_types.f90"
        outer_types_f90.write_text(_TYPES_SRC)
        outer_lib = build_fortran_library(
            outer_sdfg,
            iface=outer_iface,
            out_dir=str(outer_dir / "lib"),
            name="outer_state_wrap",
            bind_c_shim=True,
            prelude_sources=[outer_types_f90],
        )
    finally:
        clear_external_registry()

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    # ---- 4. gfortran reference ----
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    types_ref = ref_dir / "m_state.f90"
    types_ref.write_text(_TYPES_SRC)
    inner_ref = ref_dir / "inner_state.f90"
    inner_ref.write_text(_INNER_KERNEL_SRC)
    outer_ref = ref_dir / "outer_state.f90"
    outer_ref.write_text(_OUTER_KERNEL_SRC)
    ref_drv = ref_dir / "ref_driver.f90"
    ref_drv.write_text(_REF_DRIVER_SRC)
    ref_so = ref_dir / "libouter_ref.so"
    gfortran_compile_so(ref_so, types_ref, inner_ref, outer_ref, ref_drv, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # ---- 5. Drive both libraries through the same ctypes wiring ----
    # Auto-generated outer_state_c takes one c_ptr per member; the hand-written reference
    # driver mirrors that.
    n = 8
    rng = np.random.default_rng(17)
    u_init = np.asfortranarray(rng.standard_normal(n))
    v_init = np.asfortranarray(rng.standard_normal(n))
    u_sdfg = u_init.copy(order="F")
    v_sdfg = v_init.copy(order="F")
    u_ref = u_init.copy(order="F")
    v_ref = v_init.copy(order="F")

    for so in (sdfg_so, ref_lib):
        fn = so.outer_state_c
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    sdfg_so.outer_state_c(u_sdfg.ctypes.data, v_sdfg.ctypes.data)
    ref_lib.outer_state_c(u_ref.ctypes.data, v_ref.ctypes.data)

    # outer pre (u<-2u) -> inner (u<-u+v) -> outer post (u<-0.5u) => u_final = u + 0.5*v; v read-only.
    expected_u = u_init + 0.5 * v_init
    expected_v = v_init
    np.testing.assert_allclose(u_ref, expected_u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_ref, expected_v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u_sdfg, u_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_sdfg, v_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Dynamic-shape (ALLOCATABLE) variant: same pattern but runtime-sized members exercise the
# bind_c_shim's per-dim int extents + allocate + element copy-in/copy-out.
# ---------------------------------------------------------------------------

_DYN_TYPES_SRC = """
module m_state_dyn
  use iso_c_binding
  implicit none
  type :: state_dyn_t
    real(c_double), allocatable :: u(:)
    real(c_double), allocatable :: v(:)
  end type
end module
"""

_DYN_INNER_KERNEL_SRC = """
subroutine inner_state_dyn(s)
  use m_state_dyn
  implicit none
  type(state_dyn_t), intent(inout) :: s
  integer :: i
  do i = 1, size(s%u)
    s%u(i) = s%u(i) + s%v(i)
  end do
end subroutine inner_state_dyn
"""

_DYN_OUTER_KERNEL_SRC = """
subroutine outer_state_dyn(s)
  use m_state_dyn
  implicit none
  type(state_dyn_t), intent(inout) :: s
  integer :: i
  interface
    subroutine inner_state_dyn(s)
      use m_state_dyn
      type(state_dyn_t), intent(inout) :: s
    end subroutine
  end interface
  do i = 1, size(s%u)
    s%u(i) = 2.0d0 * s%u(i)
  end do
  call inner_state_dyn(s)
  do i = 1, size(s%u)
    s%u(i) = 0.5d0 * s%u(i)
  end do
end subroutine outer_state_dyn
"""

_DYN_REF_DRIVER_SRC = """
subroutine outer_state_dyn_c(n, u_p, v_p) bind(c, name="outer_state_dyn_c")
  use iso_c_binding
  use m_state_dyn
  implicit none
  integer(c_int), value :: n
  type(c_ptr), value :: u_p, v_p
  type(state_dyn_t), target :: s
  real(c_double), pointer :: u(:), v(:)
  external :: outer_state_dyn
  call c_f_pointer(u_p, u, [n])
  call c_f_pointer(v_p, v, [n])
  allocate(s%u(n))
  allocate(s%v(n))
  s%u = u
  s%v = v
  call outer_state_dyn(s)
  u = s%u
  v = s%v
end subroutine outer_state_dyn_c
"""


def test_dycore_struct_ext_dynamic_shape_e2e(tmp_path: Path):
    """Outer SDFG with type(state_dyn_t) (ALLOCATABLE members) calls inner via per-member SoA
    pointers + per-dim int extents. Mirrors the static-shape test but exercises the allocate +
    element copy-in/copy-out path; dynamic_extents_abi=True tells emit_call to prepend the
    per-dim extents the shim's c_f_pointer needs."""
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    inner_src = _DYN_TYPES_SRC + _DYN_INNER_KERNEL_SRC
    inner_sdfg = build_sdfg(inner_src, inner_sdfg_dir, name="inner_state_dyn", entry="inner_state_dyn").build()
    inner_sdfg.name = "inner_state_dyn"
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    inner_iface = OriginalInterface(
        entry="inner_state_dyn",
        args=(OriginalArg(name="s", fortran_type="type(state_dyn_t)", rank=0, intent="inout",
                          struct_type="state_dyn_t"), ),
        struct_types={
            "state_dyn_t":
            DerivedType(name="state_dyn_t",
                        module="m_state_dyn",
                        members=(
                            Member(name="u", fortran_type="real(c_double)", rank=1, shape=("?", )),
                            Member(name="v", fortran_type="real(c_double)", rank=1, shape=("?", )),
                        ))
        },
        used_modules={"m_state_dyn": ("state_dyn_t", )},
    )
    inner_types_f90 = inner_dir / "lib_types.f90"
    inner_types_f90.write_text(_DYN_TYPES_SRC)
    inner_lib = build_fortran_library(
        inner_sdfg,
        iface=inner_iface,
        out_dir=str(inner_dir / "lib"),
        name="inner_state_dyn_wrap",
        bind_c_shim=True,
        prelude_sources=[inner_types_f90],
    )
    assert inner_lib.bind_c_shim_f90 is not None

    keep_external(
        "inner_state_dyn",
        c_name="inner_state_dyn_c",
        args=(Arg(kind="aos", intent="inout", c_abi="per_member_soa"), ),
        libraries=(str(inner_lib.so_path), ),
        dynamic_extents_abi=True,
    )
    try:
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg_dir = outer_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        outer_src = _DYN_TYPES_SRC + _DYN_INNER_KERNEL_SRC + _DYN_OUTER_KERNEL_SRC
        outer_sdfg = build_sdfg(outer_src, outer_sdfg_dir, name="outer_state_dyn", entry="outer_state_dyn").build()
        outer_sdfg.name = "outer_state_dyn"
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_iface = OriginalInterface(
            entry="outer_state_dyn",
            args=(OriginalArg(name="s",
                              fortran_type="type(state_dyn_t)",
                              rank=0,
                              intent="inout",
                              struct_type="state_dyn_t"), ),
            struct_types={
                "state_dyn_t":
                DerivedType(name="state_dyn_t",
                            module="m_state_dyn",
                            members=(
                                Member(name="u", fortran_type="real(c_double)", rank=1, shape=("?", )),
                                Member(name="v", fortran_type="real(c_double)", rank=1, shape=("?", )),
                            ))
            },
            used_modules={"m_state_dyn": ("state_dyn_t", )},
        )
        outer_types_f90 = outer_dir / "lib_types.f90"
        outer_types_f90.write_text(_DYN_TYPES_SRC)
        outer_lib = build_fortran_library(
            outer_sdfg,
            iface=outer_iface,
            out_dir=str(outer_dir / "lib"),
            name="outer_state_dyn_wrap",
            bind_c_shim=True,
            prelude_sources=[outer_types_f90],
        )
    finally:
        clear_external_registry()

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    types_ref = ref_dir / "m_state_dyn.f90"
    types_ref.write_text(_DYN_TYPES_SRC)
    inner_ref = ref_dir / "inner_state_dyn.f90"
    inner_ref.write_text(_DYN_INNER_KERNEL_SRC)
    outer_ref = ref_dir / "outer_state_dyn.f90"
    outer_ref.write_text(_DYN_OUTER_KERNEL_SRC)
    ref_drv = ref_dir / "ref_driver.f90"
    ref_drv.write_text(_DYN_REF_DRIVER_SRC)
    ref_so = ref_dir / "libouter_dyn_ref.so"
    gfortran_compile_so(ref_so, types_ref, inner_ref, outer_ref, ref_drv, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    n = 8
    rng = np.random.default_rng(17)
    u_init = np.asfortranarray(rng.standard_normal(n))
    v_init = np.asfortranarray(rng.standard_normal(n))
    u_sdfg = u_init.copy(order="F")
    v_sdfg = v_init.copy(order="F")
    u_ref = u_init.copy(order="F")
    v_ref = v_init.copy(order="F")

    # SDFG side: outer's bind_c_shim takes per dynamic member (lb, extent, ptr) --
    # (s_u_lb0, s_u_d0, s_u_p, s_v_lb0, s_v_d0, s_v_p) -- since the shim reconstructs each
    # member at its TRUE bounds (here both default to 1). Reference driver instead factors
    # the single shared extent into one leading n; call each lib with its own signature.
    sdfg_fn = sdfg_so.outer_state_dyn_c
    sdfg_fn.restype = None
    sdfg_fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    sdfg_fn(1, n, u_sdfg.ctypes.data, 1, n, v_sdfg.ctypes.data)
    ref_fn = ref_lib.outer_state_dyn_c
    ref_fn.restype = None
    ref_fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    ref_fn(n, u_ref.ctypes.data, v_ref.ctypes.data)

    expected_u = u_init + 0.5 * v_init
    expected_v = v_init
    np.testing.assert_allclose(u_ref, expected_u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_ref, expected_v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u_sdfg, u_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_sdfg, v_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# LOGICAL kind variants -- exercise source_logical_kind bridging for struct members.
#
#   * LOGICAL :: flag (default kind=4): wrapper bridges through a logical(c_bool),
#     allocatable, target scratch -- SDFG bool slot must NOT alias the wider 4-byte LOGICAL
#     field directly (aliasing caused the "free(): invalid next size" glibc crash in ICON e2e).
#   * LOGICAL(c_bool) :: flag (kind=1): existing aliasable c_f_pointer path, zero-copy.
#
# Kernel: if flag, inner doubles u; outer pre-multiplies by 3, calls inner, post-multiplies
# by 0.5. Checked bit-exact against gfortran for both flag=.TRUE. and flag=.FALSE.
# ---------------------------------------------------------------------------


def _logical_test_sources(logical_decl: str, suffix: str) -> dict:
    """Render type/kernel/reference-driver sources for one LOGICAL kind variant. suffix brands
    type/subroutine/bind(c) names so the two variants link side by side without symbol clashes."""
    s = suffix
    return dict(
        types=f"""
module m_logstate_{s}
  use iso_c_binding
  implicit none
  integer, parameter :: N = 8
  type :: state_log_{s}_t
    real(c_double) :: u(N)
    {logical_decl} :: flag
  end type
end module
""",
        inner=f"""
subroutine inner_state_log_{s}(s)
  use m_logstate_{s}
  implicit none
  type(state_log_{s}_t), intent(inout) :: s
  integer :: i
  if (s%flag) then
    do i = 1, N
      s%u(i) = 2.0d0 * s%u(i)
    end do
  end if
end subroutine inner_state_log_{s}
""",
        outer=f"""
subroutine outer_state_log_{s}(s)
  use m_logstate_{s}
  implicit none
  type(state_log_{s}_t), intent(inout) :: s
  integer :: i
  interface
    subroutine inner_state_log_{s}(s)
      use m_logstate_{s}
      type(state_log_{s}_t), intent(inout) :: s
    end subroutine
  end interface
  if (s%flag) then
    do i = 1, N
      s%u(i) = 3.0d0 * s%u(i)
    end do
  end if
  call inner_state_log_{s}(s)
  if (s%flag) then
    do i = 1, N
      s%u(i) = 0.5d0 * s%u(i)
    end do
  end if
end subroutine outer_state_log_{s}
""",
        ref_driver=f"""
subroutine outer_state_log_{s}_c(u_p, flag_p) bind(c, name="outer_state_log_{s}_c")
  use iso_c_binding
  use m_logstate_{s}
  implicit none
  type(c_ptr), value :: u_p, flag_p
  type(state_log_{s}_t), target :: s
  real(c_double), pointer :: u(:)
  logical(c_bool), pointer :: flag_cbool
  external :: outer_state_log_{s}
  call c_f_pointer(u_p, u, [N])
  call c_f_pointer(flag_p, flag_cbool)
  s%u = u
  s%flag = flag_cbool
  call outer_state_log_{s}(s)
  u = s%u
  flag_cbool = s%flag
end subroutine outer_state_log_{s}_c
""",
    )


def _run_logical_kind_variant(tmp_path: Path, suffix: str, logical_decl: str, member_fortran_type: str):
    """Build inner + outer SDFGs for one LOGICAL-kind variant + a
    gfortran reference, then assert SDFG output == reference for
    both ``flag = .TRUE.`` and ``flag = .FALSE.``."""
    srcs = _logical_test_sources(logical_decl, suffix)
    s = suffix
    inner_name = f"inner_state_log_{s}"
    outer_name = f"outer_state_log_{s}"
    type_name = f"state_log_{s}_t"
    type_mod = f"m_logstate_{s}"

    iface_struct = {
        type_name:
        DerivedType(name=type_name,
                    module=type_mod,
                    members=(
                        Member(name="u", fortran_type="real(c_double)", rank=1, shape=("N", )),
                        Member(name="flag", fortran_type=member_fortran_type, rank=0, shape=()),
                    ))
    }

    # ---- 1. Inner ----
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    (inner_dir / "sdfg").mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    inner_src = srcs["types"] + srcs["inner"]
    inner_sdfg = build_sdfg(inner_src, inner_dir / "sdfg", name=inner_name, entry=f"_QP{inner_name}").build()
    inner_sdfg.name = inner_name
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    inner_iface = OriginalInterface(
        entry=inner_name,
        args=(OriginalArg(name="s", fortran_type=f"type({type_name})", rank=0, intent="inout",
                          struct_type=type_name), ),
        struct_types=iface_struct,
        used_modules={type_mod: (type_name, "N")},
    )
    types_f90 = inner_dir / "lib_types.f90"
    types_f90.write_text(srcs["types"])
    inner_lib = build_fortran_library(inner_sdfg,
                                      iface=inner_iface,
                                      out_dir=str(inner_dir / "lib"),
                                      name=f"inner_state_log_{s}_wrap",
                                      bind_c_shim=True,
                                      prelude_sources=[types_f90])
    assert inner_lib.bind_c_shim_f90 is not None

    keep_external(
        inner_name,
        c_name=f"{inner_name}_c",
        args=(Arg(kind="aos", intent="inout", c_abi="per_member_soa"), ),
        libraries=(str(inner_lib.so_path), ),
    )
    try:
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        (outer_dir / "sdfg").mkdir(parents=True, exist_ok=True)
        outer_src = srcs["types"] + srcs["inner"] + srcs["outer"]
        outer_sdfg = build_sdfg(outer_src, outer_dir / "sdfg", name=outer_name, entry=f"_QP{outer_name}").build()
        outer_sdfg.name = outer_name
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_iface = OriginalInterface(
            entry=outer_name,
            args=(OriginalArg(name="s",
                              fortran_type=f"type({type_name})",
                              rank=0,
                              intent="inout",
                              struct_type=type_name), ),
            struct_types=iface_struct,
            used_modules={type_mod: (type_name, "N")},
        )
        outer_types_f90 = outer_dir / "lib_types.f90"
        outer_types_f90.write_text(srcs["types"])
        outer_lib = build_fortran_library(outer_sdfg,
                                          iface=outer_iface,
                                          out_dir=str(outer_dir / "lib"),
                                          name=f"outer_state_log_{s}_wrap",
                                          bind_c_shim=True,
                                          prelude_sources=[outer_types_f90])
    finally:
        clear_external_registry()

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / f"{type_mod}.f90").write_text(srcs["types"])
    (ref_dir / f"{inner_name}.f90").write_text(srcs["inner"])
    (ref_dir / f"{outer_name}.f90").write_text(srcs["outer"])
    (ref_dir / "ref_driver.f90").write_text(srcs["ref_driver"])
    ref_so = ref_dir / f"libouter_log_{s}_ref.so"
    gfortran_compile_so(ref_so,
                        ref_dir / f"{type_mod}.f90",
                        ref_dir / f"{inner_name}.f90",
                        ref_dir / f"{outer_name}.f90",
                        ref_dir / "ref_driver.f90",
                        mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    n = 8
    rng = np.random.default_rng(31)

    for flag_in in (np.int8(1), np.int8(0)):
        u_init = np.asfortranarray(rng.standard_normal(n))
        u_sdfg = u_init.copy(order="F")
        u_ref = u_init.copy(order="F")
        flag_sdfg = ctypes.c_int8(int(flag_in))
        flag_ref = ctypes.c_int8(int(flag_in))

        sdfg_fn = getattr(sdfg_so, f"{outer_name}_c")
        sdfg_fn.restype = None
        sdfg_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        sdfg_fn(u_sdfg.ctypes.data, ctypes.addressof(flag_sdfg))

        ref_fn = getattr(ref_lib, f"{outer_name}_c")
        ref_fn.restype = None
        ref_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ref_fn(u_ref.ctypes.data, ctypes.addressof(flag_ref))

        np.testing.assert_allclose(u_sdfg, u_ref, rtol=1e-12, atol=1e-12, err_msg=f"flag={flag_in}: u diverged")


def test_dycore_struct_ext_logical_default_kind_e2e(tmp_path: Path):
    """Variant a) LOGICAL :: flag (default kind, 4 bytes). Wrapper goes through the
    source_logical_kind > 1 width-bridging scratch (allocatable c_bool target, element copy
    with kind conversion before/after the SDFG call)."""
    _run_logical_kind_variant(tmp_path, suffix="def", logical_decl="logical", member_fortran_type="logical")


def test_dycore_struct_ext_logical_cbool_e2e(tmp_path: Path):
    """Variant b) LOGICAL(c_bool) :: flag (1-byte kind). Wrapper stays on the zero-copy aliasable
    path (source_logical_kind == 1 short-circuits the bridge); SDFG bool* aliases the source directly."""
    _run_logical_kind_variant(tmp_path,
                              suffix="cbool",
                              logical_decl="logical(c_bool)",
                              member_fortran_type="logical(c_bool)")
