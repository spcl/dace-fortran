"""Dycore + sibling-SDFG E2E for a *struct-shaped* external call.

The architecture proof in [test_dycore_ext_velocity_e2e.py] uses a
flat-arg inner (`inner_axpy(n, a, x, y)`).  This test scales the same
pattern to a struct-shaped inner -- the actual velocity_tendencies
shape at small scale.  The inner kernel takes a single derived-type
dummy (`type(state_t)`); the outer dycore stand-in takes the same
type and calls the inner via the C++/C-ABI external boundary.

What the new ``Arg(kind='aos', c_abi='per_member_soa')`` enables (and
what this test pins):

  * The inner SDFG's :func:`emit_bind_c_shim` entry expands the
    ``type(state_t)`` dummy to one ``c_ptr`` per member -- two
    pointers for ``state_t{u(8), v(8)}``.
  * The outer SDFG's :func:`emit_call`, with the inner registered as
    ``Arg(kind='aos', c_abi='per_member_soa')``, forwards the
    marshal-expanded per-member SoA flats *verbatim* to the C call
    site.  No stack AoS buffer, no pack/unpack copy.
  * The two signatures coincide by construction: both sides are
    derived from the same Fortran ``state_t`` through the same
    pipeline.  No hand-authored shim file.

The dycore (outer) is driven from Fortran via the standard
``build_fortran_library`` bindings; the reference is a gfortran
linkage of inner + outer + a ``bind(c)`` driver sharing the same C
ABI as the SDFG path.
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


# Shared derived-type module.  ``state_t`` has two static-shape array
# members of the same scalar dtype -- the smallest struct the bind_c
# shim emits as 2 per-member C-ABI slots, and that the marshal
# expansion lowers to 2 SoA flats at the call site.
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


# Inner kernel: increments each ``u(i)`` by ``v(i)`` -- the smallest
# write-pattern that proves the per-member SoA forwarding lands on
# the right storage.
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


# Outer (dycore) kernel: doubles ``s%u`` before the inner call,
# halves it after, and calls the inner on the struct.  The pre/post
# work makes the SDFG-vs-reference comparison sensitive to the
# external-call wiring (a passthrough wouldn't catch a miswired
# external).
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


# Reference C-ABI driver around ``outer_state``.  Same flat layout as
# the SDFG's emitted ``outer_state_dace`` bindings entry (two
# pointers, one per member, in declaration order), so ``ctypes`` can
# drive both libraries identically.
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
    """Outer SDFG with a ``type(state_t)`` arg calls inner SDFG with
    the same arg shape, via per-member SoA pointers.  No
    hand-authored shim -- both sides of the C ABI are derived from
    the same Fortran source, so the marshal-expanded per-leaf args
    on the outer side coincide bit-for-bit with the bind_c shim's
    per-member entry on the inner side."""
    # ---- 1. Inner SDFG (the velocity_tendencies stand-in) ----
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    inner_src = _TYPES_SRC + _INNER_KERNEL_SRC
    inner_sdfg = build_sdfg(inner_src, inner_sdfg_dir, name="inner_state",
                            entry="_QPinner_state").build()
    inner_sdfg.name = "inner_state"
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    inner_iface = OriginalInterface(
        entry="inner_state",
        args=(OriginalArg(name="s", fortran_type="type(state_t)", rank=0,
                          intent="inout", struct_type="state_t"), ),
        struct_types={
            "state_t":
            DerivedType(name="state_t", module="m_state",
                        members=(
                            Member(name="u", fortran_type="real(c_double)",
                                   rank=1, shape=("N", )),
                            Member(name="v", fortran_type="real(c_double)",
                                   rank=1, shape=("N", )),
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
    # ``c_name="inner_state_c"`` is the bind_c_shim entry on the
    # inner's wrapper ``.so``; ``Arg(kind='aos', c_abi='per_member_soa')``
    # tells the outer's emit_call to forward the marshal-expanded
    # leaves verbatim, matching what the shim entry receives.
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
        outer_sdfg = build_sdfg(outer_src, outer_sdfg_dir,
                                name="outer_state",
                                entry="_QPouter_state").build()
        outer_sdfg.name = "outer_state"
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_iface = OriginalInterface(
            entry="outer_state",
            args=(OriginalArg(name="s", fortran_type="type(state_t)", rank=0,
                              intent="inout", struct_type="state_t"), ),
            struct_types={
                "state_t":
                DerivedType(name="state_t", module="m_state",
                            members=(
                                Member(name="u", fortran_type="real(c_double)",
                                       rank=1, shape=("N", )),
                                Member(name="v", fortran_type="real(c_double)",
                                       rank=1, shape=("N", )),
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
    gfortran_compile_so(ref_so, types_ref, inner_ref, outer_ref, ref_drv,
                        mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # ---- 5. Drive both libraries through the same ``ctypes`` wiring ----
    # The auto-generated ``outer_state_c`` (from outer's bind_c_shim)
    # takes one ``c_ptr`` per member (``s_u_p``, ``s_v_p``); the
    # hand-written reference driver mirrors that.
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

    # Expected per-element walk: outer pre (u <- 2u) -> inner (u <- u+v) ->
    # outer post (u <- 0.5*u) ==> u_final = 0.5*(2*u + v) = u + 0.5*v.
    # v is read-only on the path.
    expected_u = u_init + 0.5 * v_init
    expected_v = v_init
    np.testing.assert_allclose(u_ref, expected_u, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_ref, expected_v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u_sdfg, u_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v_sdfg, v_ref, rtol=1e-12, atol=1e-12)
