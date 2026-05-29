"""Dycore + external-velocity E2E.

The dycore SDFG calls velocity_tendencies as an *external C++ symbol*.
The velocity SDFG is built separately and exposes a stable C-ABI entry
via :func:`emit_bind_c_shim` (``velocity_tendencies_c``); a small
hand-authored C++ shim re-exports it under the dycore's expected
external name (``velocity_tendencies``), bridging the dycore SDFG's
``ExternalCall`` to the velocity SDFG's ``.so``.  The dycore itself is
driven from Fortran through the standard ``build_fortran_library``
bindings.

This commit lands the *architecture-proof* variant on a small
``inner_axpy`` kernel that shares the contract velocity_tendencies
will eventually fill: an outer SDFG calls an inner SDFG via a C++
shim that forwards into the inner's ``bind_c_shim`` entry point.
Numerical equivalence to a gfortran reference (outer + inner compiled
together) anchors the e2e.

Scaling this up to the actual ``velocity_tendencies`` signature is a
follow-up commit -- the path is identical, but the marshal expansion
produces a much longer flat-arg list.
"""
import ctypes
import shutil
import subprocess
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
    pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not on PATH"),
]


# Inner axpy kernel.  No ``bind(c)`` -- standard Fortran procedure,
# mangled by the compiler.  The reference gfortran build calls it
# directly via Fortran mangling.  The SDFG path treats it as a
# ``keep_external`` with an explicit C-side ``c_name`` that the
# generated C tasklet emits and the hand-authored C++ shim
# implements.
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


# Outer kernel.  Two pre/post adjustments around the external call
# (scale ``y`` by 2 before, by ``0.5`` after) so the test exercises a
# real delta between the outer's own work and the external's.  The
# interface block is plain Fortran -- the SDFG path lowers the
# ``call inner_axpy`` site to an ``ExternalCall`` whose C entry name
# we route via ``keep_external``'s ``c_name``.
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


# C++ shim bridging the dycore SDFG's ``ExternalCall`` (which calls
# ``dace_inner_axpy_ext`` -- the ``c_name`` registered with
# ``keep_external``) to the inner SDFG's ``bind_c_shim`` entry point
# (``inner_axpy_c``).  The dispatch symbol is deliberately *distinct*
# from any Fortran-mangled inner_axpy on the reference path -- the
# reference resolves its inner_axpy call via gfortran's mangling
# entirely separately, so we never collide.
_CPP_SHIM_SRC = """
// Hand-authored C++ shim: forwards the dycore SDFG's call to
// dace_inner_axpy_ext (the registered external symbol) into the
// inner SDFG's inner_axpy_c (the bind_c_shim entry).  The same
// forward pattern scales to the full velocity_tendencies signature
// -- only the arg list grows.  ``n`` is passed as a const-int*
// because the SDFG's ExternalCall emits scalar args as pointers
// (matches the ``kind="scalar"`` Arg shape in keep_external).
extern "C" {
  void inner_axpy_c(int n, double a, void* x_p, void* y_p);
  // The SDFG's generated C tasklet emits scalars by-value (matching
  // ``Arg(kind="scalar")``) and arrays by-pointer (matching
  // ``Arg(kind="array")``).  Mirror that exactly in the shim
  // signature; an int / double-by-pointer mismatch crashes the call
  // on entry.
  void dace_inner_axpy_ext(int n, double a, void* x_p, void* y_p) {
    inner_axpy_c(n, a, x_p, y_p);
  }
}
"""


# Reference C-ABI driver.  Same flat shape as the SDFG's exported
# ``outer_wrapper_dace`` wrapper -- when both are loaded via ctypes the
# call sites are bit-identical, and any divergence is the SDFG's.
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


def _gpp_compile(src: Path, out_so: Path, link_so: Path):
    """g++ ``-shared -fPIC`` compile the C++ shim into a ``.so`` that
    links against the inner SDFG's library at load time.

    ``-Wl,--no-as-needed`` pins the DT_NEEDED record for ``link_so``
    onto the resulting ``.so``: without it, modern binutils strips
    libraries the linker can't prove are *directly* needed at link
    time (the shim only references ``inner_axpy_c`` indirectly via
    the C++ function it defines), so the dynamic loader never reaches
    the inner SDFG's wrapper at load time and the call site fails
    with ``undefined symbol: inner_axpy_c``."""
    subprocess.check_call([
        "g++", "-shared", "-fPIC", "-o", str(out_so), str(src),
        "-Wl,--no-as-needed",
        f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}",
        f"-l:{link_so.name}",
    ])


def test_dycore_outer_calls_inner_via_cpp_shim(tmp_path: Path):
    """Outer SDFG calls inner SDFG via a hand-authored C++ shim.

    Architecture proof for the dycore + external-velocity_tendencies
    pattern: the inner SDFG ships as a standalone ``.so`` with a
    ``bind_c_shim`` entry; the dycore SDFG sees the inner as an
    opaque external (``keep_external``) with a C-ABI ``c_name``; a
    one-file C++ shim forwards the call.  No Fortran intermediate
    glue -- the dycore SDFG's ``ExternalCall`` body emits the C call
    directly into the shim's symbol."""
    # 1. Build the inner kernel as its own SDFG + bind(c) shim.
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    inner_sdfg = build_sdfg(_INNER_SRC, inner_sdfg_dir, name="inner_axpy",
                            entry="_QPinner_axpy").build()
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

    # 2. Compile the C++ shim into its own ``.so`` linked against the
    # inner SDFG's library; this re-exports ``inner_axpy`` (the
    # dycore-expected external) by forwarding to ``inner_axpy_c``.
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim_src = shim_dir / "inner_axpy_shim.cpp"
    shim_src.write_text(_CPP_SHIM_SRC)
    shim_so = shim_dir / "libinner_axpy_shim.so"
    _gpp_compile(shim_src, shim_so, inner_lib.so_path)

    # 3. Build the outer (dycore) SDFG.  ``inner_axpy`` is registered
    # as an external whose ``c_name`` is itself (the symbol the C++
    # shim exports) and whose library is the shim .so -- so the SDFG's
    # link line resolves the symbol at compile time without any
    # LD_PRELOAD.
    keep_external(
        "inner_axpy",
        c_name="dace_inner_axpy_ext",
        args=(
            Arg(kind="scalar", dtype="int32", intent="in"),
            Arg(kind="scalar", dtype="float64", intent="in"),
            Arg(kind="array", dtype="float64", intent="in"),
            Arg(kind="array", dtype="float64", intent="inout"),
        ),
        libraries=(str(shim_so), ),
    )
    try:
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg_dir = outer_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg = build_sdfg(_OUTER_SRC, outer_sdfg_dir,
                                name="outer_wrapper",
                                entry="_QPouter_wrapper").build()
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

    # 4. Reference: g++ links a flat-C-ABI driver around the
    # un-transformed outer + inner Fortran sources.  Same C calling
    # convention as the SDFG path, so ctypes drives both identically.
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

    # 5. Drive both libraries through the same ``ctypes`` wiring.  The
    # SDFG entry point is ``outer_wrapper_c`` (the auto-generated
    # bind(c) shim); the reference uses the same C ABI under the
    # equivalently-named symbol.
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
