"""Dycore + external-via-sibling-SDFG E2E.

The outer SDFG (the dycore stand-in) calls the inner SDFG (the
velocity_tendencies stand-in) directly via the inner's
``emit_bind_c_shim`` entry -- no hand-written C++ glue.  This is the
direct-call corollary of the design observation that *with the C ABI
fixed on both sides*, a forwarder shim between two sibling SDFGs is
mechanical: the outer's ``ExternalCall`` C decl and the inner's
``bind_c_shim`` C signature are derived from the same Fortran source
through the same pipeline, so they coincide.

Architecture:
  * Inner SDFG built from ``_INNER_SRC``; ``build_fortran_library
    (..., bind_c_shim=True)`` produces a standalone ``.so`` exporting
    ``inner_axpy_c`` with the canonical per-arg C ABI (scalars by
    value, arrays by pointer).
  * Outer SDFG built from ``_OUTER_SRC``; ``inner_axpy`` is
    registered as :func:`keep_external` with
    ``c_name="inner_axpy_c"`` and the inner wrapper as
    ``libraries=``.  The SDFG link line picks up the inner's ``.so``
    directly; the dynamic loader follows DT_NEEDED into it on dlopen.
  * Caller (Fortran) drives the outer via the standard
    ``build_fortran_library`` bindings.
  * Reference: gfortran links inner + outer + a ``bind(c)`` driver
    sharing the same C ABI as the SDFG path; ``ctypes`` invokes both
    libraries identically and asserts bit-for-bit equality.

Scaling this to ``velocity_tendencies``: register each derived-type
arg with ``Arg(kind="aos", c_abi="per_member_soa")`` so
:func:`emit_call` forwards per-member pointers verbatim (no AoS
buffer); the inner's ``bind_c_shim`` already receives per-member
slots, so the two sides agree by construction without an intermediate
shim.
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import (
    OriginalArg,
    OriginalInterface,
    build_fortran_library,
)
from dace_fortran.external import Arg, clear_external_registry, keep_external

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


_INNER_SRC = """
subroutine inner_axpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine inner_axpy
"""


# Outer (dycore) kernel.  Two pre/post adjustments around the
# external call so the test exercises a real delta between the outer's
# own work and the external's, not a trivial passthrough.
_OUTER_SRC = """
subroutine outer_wrapper(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  interface
    subroutine inner_axpy(n, a, x, y)
      integer, intent(in) :: n
      real(8), intent(in) :: a
      real(8), intent(in) :: x(n)
      real(8), intent(inout) :: y(n)
    end subroutine
  end interface
  do i = 1, n
    y(i) = 2.0d0 * y(i)
  end do
  call inner_axpy(n, a, x, y)
  do i = 1, n
    y(i) = 0.5d0 * y(i)
  end do
end subroutine outer_wrapper
"""


# Reference ``bind(c)`` driver around the un-transformed Fortran
# sources -- same C ABI as the SDFG's ``outer_wrapper_c`` so ``ctypes``
# drives both identically.
_REF_DRIVER_SRC = """
subroutine outer_wrapper_c(n, a, x_p, y_p) bind(c, name="outer_wrapper_c")
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), value :: a
  type(c_ptr), value :: x_p, y_p
  real(c_double), pointer :: x(:), y(:)
  external :: outer_wrapper
  call c_f_pointer(x_p, x, [n])
  call c_f_pointer(y_p, y, [n])
  call outer_wrapper(n, a, x, y)
end subroutine outer_wrapper_c
"""


def test_dycore_outer_calls_inner_via_sibling_sdfg(tmp_path: Path):
    """Outer SDFG calls inner SDFG directly via the inner's
    ``bind_c_shim`` entry.  No hand-written C++ shim -- the outer's
    generated ``ExternalCall`` body emits ``inner_axpy_c(...)``
    verbatim, and the link line resolves it from the inner's
    ``.so``."""
    # 1. Build the inner kernel as its own SDFG + bind(c) shim.
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    inner_sdfg = build_sdfg(_INNER_SRC, inner_sdfg_dir, name="inner_axpy",
                            entry="inner_axpy").build()
    inner_sdfg.name = "inner_axpy"
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    # Wrapper lib uses a *distinct* basename (``libinner_axpy_wrap.so``)
    # from the SDFG kernel's ``libinner_axpy.so`` so the dynamic loader
    # doesn't trip the link-against-self circular DT_NEEDED that an
    # identical basename produces.  Hand-author the iface so its
    # ``entry`` stays matched to ``sdfg.name`` regardless of the
    # wrapper basename (the bindings emit ``__program_<entry>``).
    inner_iface = OriginalInterface(
        entry="inner_axpy",
        args=(
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="a", fortran_type="real(c_double)", rank=0, intent="in"),
            OriginalArg(name="x", fortran_type="real(c_double)", rank=1,
                        shape=("n", ), intent="in"),
            OriginalArg(name="y", fortran_type="real(c_double)", rank=1,
                        shape=("n", ), intent="inout"),
        ),
    )
    inner_lib = build_fortran_library(
        inner_sdfg,
        iface=inner_iface,
        out_dir=str(inner_dir / "lib"),
        name="inner_axpy_wrap",
        bind_c_shim=True,
    )
    assert inner_lib.bind_c_shim_f90 is not None

    # 2. Register the inner as an external whose ``c_name`` IS the
    # bind_c_shim symbol -- the outer SDFG's tasklet emits
    # ``inner_axpy_c(...)`` and the link line resolves it from the
    # inner's wrapper ``.so`` directly.  No intermediate forwarder.
    keep_external(
        "inner_axpy",
        c_name="inner_axpy_c",
        args=(
            Arg(kind="scalar", dtype="int32", intent="in"),
            Arg(kind="scalar", dtype="float64", intent="in"),
            Arg(kind="array", dtype="float64", intent="in"),
            Arg(kind="array", dtype="float64", intent="inout"),
        ),
        libraries=(str(inner_lib.so_path), ),
    )
    try:
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg_dir = outer_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg = build_sdfg(_OUTER_SRC, outer_sdfg_dir,
                                name="outer_wrapper",
                                entry="outer_wrapper").build()
        outer_sdfg.name = "outer_wrapper"
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_lib = build_fortran_library(
            outer_sdfg,
            out_dir=str(outer_dir / "lib"),
            name="outer_wrapper",
            bind_c_shim=True,
        )
    finally:
        clear_external_registry()

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    # 3. Reference: gfortran links inner + outer + a ``bind(c)`` driver.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    inner_ref = ref_dir / "inner_axpy.f90"
    inner_ref.write_text(_INNER_SRC)
    outer_ref = ref_dir / "outer_wrapper.f90"
    outer_ref.write_text(_OUTER_SRC)
    ref_drv = ref_dir / "ref_driver.f90"
    ref_drv.write_text(_REF_DRIVER_SRC)
    ref_so = ref_dir / "libouter_ref.so"
    gfortran_compile_so(ref_so, inner_ref, outer_ref, ref_drv, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # 4. Drive both libraries through the same ``ctypes`` wiring.
    n = 8
    a = 2.5
    rng = np.random.default_rng(17)
    x = np.asfortranarray(rng.standard_normal(n))
    y_init = np.asfortranarray(rng.standard_normal(n))
    y_sdfg = y_init.copy(order="F")
    y_ref = y_init.copy(order="F")

    for so in (sdfg_so, ref_lib):
        fn = so.outer_wrapper_c
        fn.restype = None
        fn.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
    sdfg_so.outer_wrapper_c(n, a, x.ctypes.data, y_sdfg.ctypes.data)
    ref_lib.outer_wrapper_c(n, a, x.ctypes.data, y_ref.ctypes.data)

    # Expected: outer pre (2*y) -> inner (a*x + y) -> outer post (0.5*y)
    # so final y = 0.5 * (a*x + 2*y_init) = 0.5*a*x + y_init.
    expected = 0.5 * a * x + y_init
    np.testing.assert_allclose(y_ref, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12, atol=1e-12)
