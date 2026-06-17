"""End-to-end struct-dummy emit-bindings tests.

``tests/hlfir/bindings/emit_bindings_test.py`` covers three struct-flatten
fixtures via string-match assertions only:
    * ``_two_real_array_struct``  --  ``type(t_fields)`` with two plain
      ``real(c_double)`` members; everything aliases.
    * ``_complex_split_struct``   --  ``complex(c_double)`` member split
      into re/im scratch + copy loops, plus a plain real member that
      still aliases.
    * ``_nested_struct``          --  two-level nested struct (``st%a%v``
      / ``st%b%v``) where the ``%``-path is preserved through ``c_loc``.

Per ``feedback_e2e_valid_fortran``, every bindings test whose input is
a valid Fortran program must compile-and-run, not only string-match.
This module is the e2e companion: each test builds the SDFG, lets the
bridge stamp the real ``hlfir.flatten_plan`` attribute, lifts that
into a Python ``FlattenPlan``, runs the emitter, gfortran-compiles
the wrapper + a Fortran driver, links to the SDFG ``.so``, and
asserts numerical equality against a gfortran-compiled reference of
the same source.

Both the SDFG-via-bindings library and the gfortran reference library
are compiled by gfortran into ``.so`` files and loaded via ctypes.
f2py's crackfortran can't parse the bindings wrapper -- its dummy is
``type(t_fields)``, which maps to ``'void'`` in f2py's C-type table
and crashes lookup -- and the same struct-typed kernel dummy would
crash the reference build too.  Skipping f2py for both paths keeps
the test surface uniform and dodges that parser limitation entirely.
"""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings import (
    FlattenPlan,
    OriginalInterface,
    emit_bindings,
)
from dace_fortran.bindings.fortran_interface import build_auto_interface

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


def _compile_so(out_so: Path, *sources: Path, mod_dir: Path, link_so: Path | None = None):
    """gfortran-compile ``sources`` into ``out_so``, writing ``.mod``
    files to ``mod_dir`` and (optionally) linking against ``link_so``.

    We always set ``cwd=mod_dir`` and ``-J<mod_dir>`` so gfortran
    doesn't search the repository root for ``.mod`` files -- earlier
    f2py probes leak stale modules there and the format isn't
    cross-compiler compatible.
    """
    # No strict-FP flags here by design (structural ABI test, not a
    # numeric compare); just lift the free-form column cap so the long
    # generated signatures compile on gfortran <=12.
    cmd = ["gfortran", "-shared", "-fPIC", "-ffree-line-length-none", f"-J{mod_dir}"]
    cmd.extend(str(s) for s in sources)
    cmd.extend(["-o", str(out_so)])
    if link_so is not None:
        cmd.extend([
            f"-L{link_so.parent}",
            f"-Wl,-rpath,{link_so.parent}",
            f"-l:{link_so.name}",
        ])
    subprocess.check_call(cmd, cwd=mod_dir)


def _build_sdfg_lib(
    tmp_path: Path,
    *,
    kernel_src: str,
    types_src: str,
    name: str,
    entry: str,
    iface: OriginalInterface = None,
    driver_src: str,
):
    """SDFG-via-bindings path: build SDFG, emit bindings, gfortran-link
    the types + bindings + driver into one ``.so`` against the SDFG
    library, return the loaded ctypes lib.

    ``FlattenPlan`` is read off the bridge module after the pass
    pipeline runs, so the emitter sees the same recipe
    ``hlfir-flatten-structs`` actually recorded.  ``iface`` defaults to
    the SDFG's auto-derived caller interface (these AoS kernels all flatten
    explicit-shape struct dummies, which the snapshot names correctly);
    pass an explicit one only for a shape the snapshot can't recover.
    """
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(kernel_src, sdfg_dir, name=name, entry=entry)
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.name = name
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)
    fs = sdfg._frozen_signature
    if iface is None:
        iface = build_auto_interface(sdfg._fortran_interface_raw, name)

    bindings_path = tmp_path / f"{name}_bindings.f90"
    emit_bindings(fs, iface, plan, str(bindings_path))
    types_path = tmp_path / f"{name}_types.f90"
    types_path.write_text(types_src)
    driver_path = tmp_path / f"{name}_driver.f90"
    driver_path.write_text(driver_src)

    build_dir = tmp_path / "sdfg_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    driver_so = build_dir / f"{name}_driver.so"
    _compile_so(driver_so, types_path, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)
    return ctypes.CDLL(str(driver_so))


def _build_reference_lib(
    tmp_path: Path,
    *,
    types_src: str,
    kernel_src: str,
    ref_driver_src: str,
    name: str,
):
    """gfortran reference path: compile types + plain Fortran kernel +
    reference driver into one ``.so``.  The reference driver uses the
    same ``bind(c)`` raw-pointer entry-point convention as the SDFG
    driver so we can swap them with ctypes."""
    types_path = tmp_path / f"{name}_ref_types.f90"
    types_path.write_text(types_src)
    kernel_path = tmp_path / f"{name}_ref_kernel.f90"
    kernel_path.write_text(kernel_src)
    driver_path = tmp_path / f"{name}_ref_driver.f90"
    driver_path.write_text(ref_driver_src)

    build_dir = tmp_path / "ref_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    ref_so = build_dir / f"{name}_ref.so"
    _compile_so(ref_so, types_path, kernel_path, driver_path, mod_dir=build_dir)
    return ctypes.CDLL(str(ref_so))


# ---------------------------------------------------------------------------
# Two-real-array struct  --  zero-copy alias path
# ---------------------------------------------------------------------------

_TWO_REAL_TYPES_SRC = """
module mo_fields
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_fields
     real(c_double) :: a(NX, NY)
     real(c_double) :: b(NX, NY)
  end type t_fields
end module mo_fields
"""

_TWO_REAL_KERNEL_SRC = """
module kernel_two_real_mod
contains
subroutine kernel_two_real(fld)
  use mo_fields
  use iso_c_binding
  implicit none
  type(t_fields), intent(inout) :: fld
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        fld%a(i, j) = fld%a(i, j) + fld%b(i, j)
     end do
  end do
end subroutine kernel_two_real
end module kernel_two_real_mod
"""

_TWO_REAL_REF_DRIVER_SRC = """
! Reference C-callable driver: same Fortran kernel ``kernel_two_real``,
! same raw-pointer ABI as the SDFG driver, so the test calls one or
! the other via ctypes and they're directly comparable.
subroutine run_two_real_ref(a_ptr, b_ptr) bind(c, name='run_two_real_ref')
  use iso_c_binding
  use mo_fields, only: t_fields, NX, NY
  use kernel_two_real_mod, only: kernel_two_real
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_fields) :: fld
  fld%a = a_ptr
  fld%b = b_ptr
  call kernel_two_real(fld)
  a_ptr = fld%a
  b_ptr = fld%b
end subroutine run_two_real_ref
"""

# Full source the bridge consumes: types + kernel.
_TWO_REAL_SRC = _TWO_REAL_TYPES_SRC + _TWO_REAL_KERNEL_SRC

_TWO_REAL_DRIVER = """
! C-callable driver that loads ``a``, ``b`` into a ``type(t_fields)``,
! calls the bindings wrapper, copies the post-call values back out.
! Linking is via ctypes (the bindings module's ``type(t_fields)`` arg
! defeats f2py's crackfortran), so the entry point is bind(c) with
! raw c_double pointers.
subroutine run_two_real(a_ptr, b_ptr) bind(c, name='run_two_real')
  use iso_c_binding
  use mo_fields, only: t_fields, NX, NY
  use kernel_two_real_dace_bindings
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_fields), target :: fld
  fld%a = a_ptr
  fld%b = b_ptr
  call kernel_two_real_dace(fld)
  a_ptr = fld%a
  b_ptr = fld%b
  call kernel_two_real_dace_finalize()
end subroutine run_two_real
"""


def test_e2e_two_real_array_struct(tmp_path: Path):
    """``type(t_fields)`` with two static ``real(c_double)`` members.
    Both members alias zero-copy through ``c_loc``.  ``kernel`` does
    ``fld%a = fld%a + fld%b``; reference and SDFG paths must produce
    identical ``fld%a`` post-call."""
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_TWO_REAL_SRC,
        types_src=_TWO_REAL_TYPES_SRC,
        name="kernel_two_real",
        entry="kernel_two_real_mod::kernel_two_real",
        driver_src=_TWO_REAL_DRIVER,
    )
    sdfg_lib.run_two_real.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sdfg_lib.run_two_real.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_TWO_REAL_TYPES_SRC,
        kernel_src=_TWO_REAL_KERNEL_SRC,
        ref_driver_src=_TWO_REAL_REF_DRIVER_SRC,
        name="kernel_two_real",
    )
    ref_lib.run_two_real_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    ref_lib.run_two_real_ref.restype = None

    rng = np.random.default_rng(17)
    nx, ny = 4, 5
    a_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    b_init = np.asfortranarray(rng.standard_normal((nx, ny)))

    a_ref = a_init.copy(order="F")
    b_ref = b_init.copy(order="F")
    ref_lib.run_two_real_ref(
        a_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    a_sdfg = a_init.copy(order="F")
    b_sdfg = b_init.copy(order="F")
    sdfg_lib.run_two_real(
        a_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    np.testing.assert_array_equal(a_sdfg, a_ref)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# Nested struct ``st%a%v`` / ``st%b%v``  --  bridge gap (xfail)
# ---------------------------------------------------------------------------
#
# ``passes/FlattenStructs.cpp`` calls ``replaceStructArgNested`` for the
# nested case and intentionally does NOT call ``recordStructArgEntry`` --
# the comment in that block reads:
#
#     "Bindings-side FlattenEntry emission for outer_kind='dummy_nested'
#      is a separate follow-up -- Python-side callers can pass the flat
#      companions directly via kwargs today; the Fortran caller wrapper
#      needs the recipe to pack the nested struct's path-form members
#      on its end."
#
# Consequence: the bridge produces ``st_a_v``/``st_b_v`` flat dummies on
# the SDFG but ``get_flatten_plan()`` returns ``{'entries': []}``.  The
# emitter therefore emits no ``c_f_pointer`` aliases and the wrapper
# call site passes uninitialised ``st_a_v``/``st_b_v`` pointers.
#
# Pinning the e2e shape here so the test flips green when nested-plan
# emission lands in FlattenStructs.cpp.

_NESTED_TYPES_SRC = """
module mo_nested
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_inner
     real(c_double) :: v(NX, NY)
  end type t_inner
  type :: t_outer
     type(t_inner) :: a
     type(t_inner) :: b
  end type t_outer
end module mo_nested
"""

_NESTED_KERNEL_SRC = """
module kernel_nested_mod
contains
subroutine kernel_nested(st)
  use mo_nested
  use iso_c_binding
  implicit none
  type(t_outer), intent(inout) :: st
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        st%a%v(i, j) = st%a%v(i, j) + st%b%v(i, j)
     end do
  end do
end subroutine kernel_nested
end module kernel_nested_mod
"""

_NESTED_REF_DRIVER_SRC = """
subroutine run_nested_ref(a_ptr, b_ptr) bind(c, name='run_nested_ref')
  use iso_c_binding
  use mo_nested, only: t_outer, NX, NY
  use kernel_nested_mod, only: kernel_nested
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_outer) :: st
  st%a%v = a_ptr
  st%b%v = b_ptr
  call kernel_nested(st)
  a_ptr = st%a%v
  b_ptr = st%b%v
end subroutine run_nested_ref
"""

_NESTED_SRC = _NESTED_TYPES_SRC + _NESTED_KERNEL_SRC

_NESTED_DRIVER = """
subroutine run_nested(a_ptr, b_ptr) bind(c, name='run_nested')
  use iso_c_binding
  use mo_nested, only: t_outer, NX, NY
  use kernel_nested_dace_bindings
  implicit none
  real(c_double), intent(inout) :: a_ptr(NX, NY), b_ptr(NX, NY)
  type(t_outer), target :: st
  st%a%v = a_ptr
  st%b%v = b_ptr
  call kernel_nested_dace(st)
  a_ptr = st%a%v
  b_ptr = st%b%v
  call kernel_nested_dace_finalize()
end subroutine run_nested
"""


def test_e2e_nested_struct(tmp_path: Path):
    """``type(t_outer)`` containing two ``type(t_inner)`` members, each
    with a static ``real(c_double)`` array.  The kernel does
    ``st%a%v = st%a%v + st%b%v``.  ``recordNestedStructArgEntry`` in
    ``FlattenStructs.cpp`` emits one FlattenEntry whose recipe carries
    a flat name + a dotted read_expr per leaf, so the bindings emitter
    aliases each leaf via ``c_f_pointer(c_loc(st%a%v), st_a_v, [...])``."""
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_NESTED_SRC,
        types_src=_NESTED_TYPES_SRC,
        name="kernel_nested",
        entry="kernel_nested_mod::kernel_nested",
        driver_src=_NESTED_DRIVER,
    )
    sdfg_lib.run_nested.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sdfg_lib.run_nested.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_NESTED_TYPES_SRC,
        kernel_src=_NESTED_KERNEL_SRC,
        ref_driver_src=_NESTED_REF_DRIVER_SRC,
        name="kernel_nested",
    )
    ref_lib.run_nested_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    ref_lib.run_nested_ref.restype = None

    rng = np.random.default_rng(23)
    nx, ny = 4, 5
    a_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    b_init = np.asfortranarray(rng.standard_normal((nx, ny)))

    a_ref = a_init.copy(order="F")
    b_ref = b_init.copy(order="F")
    ref_lib.run_nested_ref(
        a_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    a_sdfg = a_init.copy(order="F")
    b_sdfg = b_init.copy(order="F")
    sdfg_lib.run_nested(
        a_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    np.testing.assert_array_equal(a_sdfg, a_ref)
    np.testing.assert_array_equal(b_sdfg, b_ref)


# ---------------------------------------------------------------------------
# Complex member struct  --  complex stays as a native complex128 SDFG dtype
# ---------------------------------------------------------------------------
#
# ``complex(c_double)`` struct members flatten to a single
# ``complex128`` companion array, NOT into a re/im pair.  Per the
# user's policy "complex types are supported in DaCe; complex arrays
# should be flattened using the complex dtype": DaCe handles complex
# arithmetic on a single ``std::complex<T>`` array natively.
#
# Bridge plumbing:
#   * ``FlattenStructs.cpp::isSimpleScalar`` accepts ``ComplexType``.
#   * ``dtypeName`` maps to ``complex64`` / ``complex128``.
#   * ``recordStructArgEntry`` emits one FlattenEntry per member so
#     mixed-dtype structs (complex + real) carry the right
#     ``scratch_dtype`` per entry.
#   * AST extractor ``ast/expressions.cpp`` recognises standalone
#     ``fir.extract_value`` from a complex value and emits
#     ``<z>.real()`` / ``<z>.imag()``; cppunparse renders these as
#     ``std::complex<T>::real()`` / ``::imag()`` method calls in C++.

_COMPLEX_TYPES_SRC = """
module mo_state
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_state
     complex(c_double) :: z(NX, NY)
     real(c_double)    :: u(NX, NY)
  end type t_state
end module mo_state
"""

_COMPLEX_KERNEL_SRC = """
module kernel_complex_mod
contains
subroutine kernel_complex(st)
  use mo_state
  use iso_c_binding
  implicit none
  type(t_state), intent(inout) :: st
  integer :: i, j
  do j = 1, NY
     do i = 1, NX
        st%u(i, j) = real(st%z(i, j), kind=c_double) + aimag(st%z(i, j))
     end do
  end do
end subroutine kernel_complex
end module kernel_complex_mod
"""

_COMPLEX_REF_DRIVER_SRC = """
subroutine run_complex_ref(z_re_ptr, z_im_ptr, u_ptr) bind(c, name='run_complex_ref')
  use iso_c_binding
  use mo_state, only: t_state, NX, NY
  use kernel_complex_mod, only: kernel_complex
  implicit none
  real(c_double), intent(in)    :: z_re_ptr(NX, NY)
  real(c_double), intent(in)    :: z_im_ptr(NX, NY)
  real(c_double), intent(inout) :: u_ptr(NX, NY)
  type(t_state) :: st
  st%z = cmplx(z_re_ptr, z_im_ptr, kind=c_double)
  st%u = u_ptr
  call kernel_complex(st)
  u_ptr = st%u
end subroutine run_complex_ref
"""

_COMPLEX_SRC = _COMPLEX_TYPES_SRC + _COMPLEX_KERNEL_SRC

_COMPLEX_DRIVER = """
subroutine run_complex(z_re_ptr, z_im_ptr, u_ptr) bind(c, name='run_complex')
  use iso_c_binding
  use mo_state, only: t_state, NX, NY
  use kernel_complex_dace_bindings
  implicit none
  real(c_double), intent(in)    :: z_re_ptr(NX, NY)
  real(c_double), intent(in)    :: z_im_ptr(NX, NY)
  real(c_double), intent(inout) :: u_ptr(NX, NY)
  type(t_state), target :: st
  st%z = cmplx(z_re_ptr, z_im_ptr, kind=c_double)
  st%u = u_ptr
  call kernel_complex_dace(st)
  u_ptr = st%u
  call kernel_complex_dace_finalize()
end subroutine run_complex
"""


def test_e2e_complex_member_struct(tmp_path: Path):
    """``type(t_state)`` with ``complex(c_double)`` and ``real(c_double)``
    array members.  Kernel does ``st%u = real(st%z) + aimag(st%z)``.
    The complex member flattens to a single ``complex128`` companion
    (NOT split into re/im); the bindings emitter aliases it via
    ``c_f_pointer(c_loc(st%z), st_z, [...])`` and DaCe's tasklet
    codegen handles the ``.real()`` / ``.imag()`` method calls on
    ``std::complex<double>``."""
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_COMPLEX_SRC,
        types_src=_COMPLEX_TYPES_SRC,
        name="kernel_complex",
        entry="kernel_complex_mod::kernel_complex",
        driver_src=_COMPLEX_DRIVER,
    )
    sdfg_lib.run_complex.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    sdfg_lib.run_complex.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_COMPLEX_TYPES_SRC,
        kernel_src=_COMPLEX_KERNEL_SRC,
        ref_driver_src=_COMPLEX_REF_DRIVER_SRC,
        name="kernel_complex",
    )
    ref_lib.run_complex_ref.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    ref_lib.run_complex_ref.restype = None

    rng = np.random.default_rng(29)
    nx, ny = 4, 5
    z_re_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    z_im_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    u_init = np.asfortranarray(rng.standard_normal((nx, ny)))

    u_ref = u_init.copy(order="F")
    ref_lib.run_complex_ref(
        z_re_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_im_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        u_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    u_sdfg = u_init.copy(order="F")
    sdfg_lib.run_complex(
        z_re_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_im_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        u_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )

    np.testing.assert_allclose(u_sdfg, u_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Array-of-structs with scalar members  --  DEEPCOPY path (strided gather)
# ---------------------------------------------------------------------------
#
# ``type(point) :: pts(N)`` with four scalar members.  Each SoA companion
# ``pts_x``..``pts_w`` is a *strided* view of the interleaved AoS (stride =
# struct size), so it CANNOT be a zero-copy ``c_f_pointer`` alias -- the
# binding must allocate contiguous companions and scatter/gather them
# (``aliasable=False`` -> render_copy_in_loop / render_copy_out_loop).  This
# exercises the deepcopy path end-to-end and value-checks the gather/scatter.

_AOS_TYPES_SRC = """
module mo_pt
  use iso_c_binding
  implicit none
  integer, parameter :: N = 6
  type :: point
     real(c_double) :: x, y, z, w
  end type point
end module mo_pt
"""

_AOS_KERNEL_SRC = """
module kern_aos_mod
contains
subroutine kern_aos(pts)
  use mo_pt
  use iso_c_binding
  implicit none
  type(point), intent(inout) :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = pts(i)%x + pts(i)%y * pts(i)%z - pts(i)%w
  end do
end subroutine kern_aos
end module kern_aos_mod
"""

_AOS_REF_DRIVER_SRC = """
subroutine run_aos_ref(p_ptr) bind(c, name='run_aos_ref')
  use iso_c_binding
  use mo_pt, only: point, N
  use kern_aos_mod, only: kern_aos
  implicit none
  real(c_double), intent(inout) :: p_ptr(4, N)
  type(point) :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = p_ptr(1, i); pts(i)%y = p_ptr(2, i)
     pts(i)%z = p_ptr(3, i); pts(i)%w = p_ptr(4, i)
  end do
  call kern_aos(pts)
  do i = 1, N
     p_ptr(1, i) = pts(i)%x; p_ptr(2, i) = pts(i)%y
     p_ptr(3, i) = pts(i)%z; p_ptr(4, i) = pts(i)%w
  end do
end subroutine run_aos_ref
"""

_AOS_SRC = _AOS_TYPES_SRC + _AOS_KERNEL_SRC

_AOS_DRIVER = """
subroutine run_aos(p_ptr) bind(c, name='run_aos')
  use iso_c_binding
  use mo_pt, only: point, N
  use kern_aos_dace_bindings
  implicit none
  real(c_double), intent(inout) :: p_ptr(4, N)
  type(point), target :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = p_ptr(1, i); pts(i)%y = p_ptr(2, i)
     pts(i)%z = p_ptr(3, i); pts(i)%w = p_ptr(4, i)
  end do
  call kern_aos_dace(pts)
  do i = 1, N
     p_ptr(1, i) = pts(i)%x; p_ptr(2, i) = pts(i)%y
     p_ptr(3, i) = pts(i)%z; p_ptr(4, i) = pts(i)%w
  end do
  call kern_aos_dace_finalize()
end subroutine run_aos
"""


def test_e2e_array_of_scalar_structs_deepcopy(tmp_path: Path):
    """``type(point) :: pts(N)`` (4 scalar members) -> 4 SoA arrays via the
    strided-gather DEEPCOPY path.  The binding allocates ``pts_x``..``pts_w``,
    scatters the interleaved AoS in, runs the kernel, gathers back out; the
    result must match a gfortran reference of the same kernel on the AoS."""
    sdfg_lib = _build_sdfg_lib(
        tmp_path,
        kernel_src=_AOS_SRC,
        types_src=_AOS_TYPES_SRC,
        name="kern_aos",
        entry="kern_aos_mod::kern_aos",
        driver_src=_AOS_DRIVER,
    )
    sdfg_lib.run_aos.argtypes = [ctypes.POINTER(ctypes.c_double)]
    sdfg_lib.run_aos.restype = None

    ref_lib = _build_reference_lib(
        tmp_path,
        types_src=_AOS_TYPES_SRC,
        kernel_src=_AOS_KERNEL_SRC,
        ref_driver_src=_AOS_REF_DRIVER_SRC,
        name="kern_aos",
    )
    ref_lib.run_aos_ref.argtypes = [ctypes.POINTER(ctypes.c_double)]
    ref_lib.run_aos_ref.restype = None

    rng = np.random.default_rng(31)
    p_init = np.asfortranarray(rng.standard_normal((4, 6)))

    p_ref = p_init.copy(order="F")
    ref_lib.run_aos_ref(p_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    p_sdfg = p_init.copy(order="F")
    sdfg_lib.run_aos(p_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    np.testing.assert_allclose(p_sdfg, p_ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# AoS with MIXED-TYPE scalar members  --  deepcopy across dtypes
# ---------------------------------------------------------------------------
#
# ``type(item) :: items(N)`` with a real(c_double) and an integer(c_int)
# member.  Each flattens to its own typed SoA companion (items_a : f64,
# items_n : i32), both deep-copied (strided gather/scatter).  Exercises the
# copy-in/out path with two distinct element types in one struct.

_MIX_TYPES_SRC = """
module mo_mix
  use iso_c_binding
  implicit none
  integer, parameter :: N = 5
  type :: item
     real(c_double)  :: a
     integer(c_int)  :: n
  end type item
end module mo_mix
"""

_MIX_KERNEL_SRC = """
module kern_mix_mod
contains
subroutine kern_mix(items)
  use mo_mix
  use iso_c_binding
  implicit none
  type(item), intent(inout) :: items(N)
  integer :: i
  do i = 1, N
     items(i)%a = items(i)%a * real(items(i)%n, c_double)
     items(i)%n = items(i)%n + 1
  end do
end subroutine kern_mix
end module kern_mix_mod
"""

_MIX_REF_DRIVER_SRC = """
subroutine run_mix_ref(a_ptr, n_ptr) bind(c, name='run_mix_ref')
  use iso_c_binding
  use mo_mix, only: item, N
  use kern_mix_mod, only: kern_mix
  implicit none
  real(c_double), intent(inout)   :: a_ptr(N)
  integer(c_int), intent(inout)   :: n_ptr(N)
  type(item) :: items(N)
  integer :: i
  do i = 1, N
     items(i)%a = a_ptr(i); items(i)%n = n_ptr(i)
  end do
  call kern_mix(items)
  do i = 1, N
     a_ptr(i) = items(i)%a; n_ptr(i) = items(i)%n
  end do
end subroutine run_mix_ref
"""

_MIX_SRC = _MIX_TYPES_SRC + _MIX_KERNEL_SRC

_MIX_DRIVER = """
subroutine run_mix(a_ptr, n_ptr) bind(c, name='run_mix')
  use iso_c_binding
  use mo_mix, only: item, N
  use kern_mix_dace_bindings
  implicit none
  real(c_double), intent(inout)   :: a_ptr(N)
  integer(c_int), intent(inout)   :: n_ptr(N)
  type(item), target :: items(N)
  integer :: i
  do i = 1, N
     items(i)%a = a_ptr(i); items(i)%n = n_ptr(i)
  end do
  call kern_mix_dace(items)
  do i = 1, N
     a_ptr(i) = items(i)%a; n_ptr(i) = items(i)%n
  end do
  call kern_mix_dace_finalize()
end subroutine run_mix
"""


def test_e2e_array_of_mixed_type_structs_deepcopy(tmp_path: Path):
    """``type(item){a:real, n:int} :: items(N)`` -> two typed SoA companions,
    both deep-copied.  Kernel updates both members; result must match the
    gfortran reference for both the real and integer arrays."""
    sdfg_lib = _build_sdfg_lib(tmp_path, kernel_src=_MIX_SRC, types_src=_MIX_TYPES_SRC, name="kern_mix",
                               entry="kern_mix_mod::kern_mix", driver_src=_MIX_DRIVER)
    sdfg_lib.run_mix.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    sdfg_lib.run_mix.restype = None
    ref_lib = _build_reference_lib(tmp_path, types_src=_MIX_TYPES_SRC, kernel_src=_MIX_KERNEL_SRC,
                                   ref_driver_src=_MIX_REF_DRIVER_SRC, name="kern_mix")
    ref_lib.run_mix_ref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
    ref_lib.run_mix_ref.restype = None

    rng = np.random.default_rng(37)
    a_init = np.asfortranarray(rng.standard_normal(5))
    n_init = np.asfortranarray(rng.integers(1, 9, size=5).astype(np.int32))

    a_ref, n_ref = a_init.copy(order="F"), n_init.copy(order="F")
    ref_lib.run_mix_ref(a_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        n_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    a_sdfg, n_sdfg = a_init.copy(order="F"), n_init.copy(order="F")
    sdfg_lib.run_mix(a_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                     n_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    np.testing.assert_allclose(a_sdfg, a_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(n_sdfg, n_ref)


# ---------------------------------------------------------------------------
# AoS as an intent(in) argument  --  copy-IN only, no copy-out
# ---------------------------------------------------------------------------
#
# When the struct array is read-only the deepcopy must scatter the members
# IN but emit no copy-back (and the input must be left unchanged).  The
# kernel writes a separate plain output array.

_RO_KERNEL_SRC = """
module kern_ro_mod
contains
subroutine kern_ro(pts, outv)
  use mo_pt
  use iso_c_binding
  implicit none
  type(point), intent(in)  :: pts(N)
  real(c_double), intent(out) :: outv(N)
  integer :: i
  do i = 1, N
     outv(i) = pts(i)%x + pts(i)%y * pts(i)%z - pts(i)%w
  end do
end subroutine kern_ro
end module kern_ro_mod
"""

_RO_REF_DRIVER_SRC = """
subroutine run_ro_ref(p_ptr, o_ptr) bind(c, name='run_ro_ref')
  use iso_c_binding
  use mo_pt, only: point, N
  use kern_ro_mod, only: kern_ro
  implicit none
  real(c_double), intent(in)    :: p_ptr(4, N)
  real(c_double), intent(out)   :: o_ptr(N)
  type(point) :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = p_ptr(1, i); pts(i)%y = p_ptr(2, i)
     pts(i)%z = p_ptr(3, i); pts(i)%w = p_ptr(4, i)
  end do
  call kern_ro(pts, o_ptr)
end subroutine run_ro_ref
"""

_RO_SRC = _AOS_TYPES_SRC + _RO_KERNEL_SRC

_RO_DRIVER = """
subroutine run_ro(p_ptr, o_ptr) bind(c, name='run_ro')
  use iso_c_binding
  use mo_pt, only: point, N
  use kern_ro_dace_bindings
  implicit none
  real(c_double), intent(in)    :: p_ptr(4, N)
  real(c_double), intent(out)   :: o_ptr(N)
  type(point), target :: pts(N)
  integer :: i
  do i = 1, N
     pts(i)%x = p_ptr(1, i); pts(i)%y = p_ptr(2, i)
     pts(i)%z = p_ptr(3, i); pts(i)%w = p_ptr(4, i)
  end do
  call kern_ro_dace(pts, o_ptr)
  call kern_ro_dace_finalize()
end subroutine run_ro
"""


def test_e2e_array_of_structs_read_only_copy_in(tmp_path: Path):
    """``type(point) :: pts(N)`` passed ``intent(in)``: the deepcopy scatters
    the members in (no copy-back) and the kernel writes a separate output
    array.  Output must match the reference."""
    sdfg_lib = _build_sdfg_lib(tmp_path, kernel_src=_RO_SRC, types_src=_AOS_TYPES_SRC, name="kern_ro",
                               entry="kern_ro_mod::kern_ro", driver_src=_RO_DRIVER)
    sdfg_lib.run_ro.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    sdfg_lib.run_ro.restype = None
    ref_lib = _build_reference_lib(tmp_path, types_src=_AOS_TYPES_SRC, kernel_src=_RO_KERNEL_SRC,
                                   ref_driver_src=_RO_REF_DRIVER_SRC, name="kern_ro")
    ref_lib.run_ro_ref.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    ref_lib.run_ro_ref.restype = None

    rng = np.random.default_rng(41)
    p_init = np.asfortranarray(rng.standard_normal((4, 6)))

    o_ref = np.zeros(6, dtype=np.float64, order="F")
    p_ref = p_init.copy(order="F")
    ref_lib.run_ro_ref(p_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       o_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    o_sdfg = np.zeros(6, dtype=np.float64, order="F")
    p_sdfg = p_init.copy(order="F")
    sdfg_lib.run_ro(p_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    o_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    np.testing.assert_allclose(o_sdfg, o_ref, rtol=1e-12, atol=1e-12)
    # intent(in): the input AoS must be left unchanged.
    np.testing.assert_array_equal(p_sdfg, p_init)


# ---------------------------------------------------------------------------
# Jagged AoS + allocatable member  --  the runtime-cap pack/unpack path
# ---------------------------------------------------------------------------
# ``type(bag) :: a(NB)`` where ``bag`` has an ``allocatable :: w(:)`` member
# and each instance is allocated a DIFFERENT length.  The flatten pass packs
# the jagged member into an ELLPACK companion ``a_w(NB, cap)`` whose inner
# extent is a runtime ``cap_a_w`` symbol the binding fills via ``max`` over
# the per-instance sizes.  This exercises the extent-detection fix:
# ``size(a(i)%w)`` (the inner loop bound) must resolve to that companion cap
# symbol, not leak the original struct's ``a_d0`` extent.  The kernel scales
# in place (padding rows stay zero -> harmless), and the binding's pack-out
# copies back only each instance's live ``1:size`` region.

_JAG_TYPES_SRC = """
module mo_bag
  use iso_c_binding
  implicit none
  integer, parameter :: NB = 3
  type :: bag
     real(c_double), allocatable :: w(:)
  end type bag
end module mo_bag
"""

_JAG_KERNEL_SRC = """
module kern_jag_mod
contains
subroutine kern_jag(a)
  use mo_bag
  use iso_c_binding
  implicit none
  type(bag), intent(inout) :: a(NB)
  integer :: i, j
  do i = 1, NB
     do j = 1, size(a(i)%w)
        a(i)%w(j) = a(i)%w(j) * 2.0_c_double
     end do
  end do
end subroutine kern_jag
end module kern_jag_mod
"""

# Both drivers allocate the same jagged shape (sizes 2, 4, 3 -> cap 4) and
# move the flat 9-element buffer in/out of the per-instance members.
_JAG_DRIVER = """
subroutine run_jag(p_ptr) bind(c, name='run_jag')
  use iso_c_binding
  use mo_bag, only: bag, NB
  use kern_jag_dace_bindings
  implicit none
  real(c_double), intent(inout) :: p_ptr(9)
  type(bag), target :: a(NB)
  integer :: i, off, sizes(NB)
  sizes = [2, 4, 3]
  off = 0
  do i = 1, NB
     allocate(a(i)%w(sizes(i)))
     a(i)%w = p_ptr(off+1:off+sizes(i))
     off = off + sizes(i)
  end do
  call kern_jag_dace(a)
  off = 0
  do i = 1, NB
     p_ptr(off+1:off+sizes(i)) = a(i)%w
     off = off + sizes(i)
     deallocate(a(i)%w)
  end do
  call kern_jag_dace_finalize()
end subroutine run_jag
"""

_JAG_REF_DRIVER = """
subroutine run_jag_ref(p_ptr) bind(c, name='run_jag_ref')
  use iso_c_binding
  use mo_bag, only: bag, NB
  use kern_jag_mod, only: kern_jag
  implicit none
  real(c_double), intent(inout) :: p_ptr(9)
  type(bag) :: a(NB)
  integer :: i, off, sizes(NB)
  sizes = [2, 4, 3]
  off = 0
  do i = 1, NB
     allocate(a(i)%w(sizes(i)))
     a(i)%w = p_ptr(off+1:off+sizes(i))
     off = off + sizes(i)
  end do
  call kern_jag(a)
  off = 0
  do i = 1, NB
     p_ptr(off+1:off+sizes(i)) = a(i)%w
     off = off + sizes(i)
     deallocate(a(i)%w)
  end do
end subroutine run_jag_ref
"""


def test_e2e_array_of_jagged_alloc_structs_deepcopy(tmp_path: Path):
    """``type(bag){allocatable w(:)} :: a(NB)`` with per-instance sizes
    (2, 4, 3) -> ELLPACK companion ``a_w(NB, cap_a_w)`` (cap = 4 via the
    binding's ``max``).  Verifies the runtime-cap pack/unpack round-trips
    the live data and that ``size(a(i)%w)`` resolved to the companion cap
    (no ``a_d0`` symbol leak), against a gfortran reference."""
    sdfg_lib = _build_sdfg_lib(tmp_path, kernel_src=_JAG_TYPES_SRC + _JAG_KERNEL_SRC,
                               types_src=_JAG_TYPES_SRC, name="kern_jag",
                               entry="kern_jag_mod::kern_jag", driver_src=_JAG_DRIVER)
    sdfg_lib.run_jag.argtypes = [ctypes.POINTER(ctypes.c_double)]
    sdfg_lib.run_jag.restype = None

    ref_lib = _build_reference_lib(tmp_path, types_src=_JAG_TYPES_SRC,
                                   kernel_src=_JAG_KERNEL_SRC,
                                   ref_driver_src=_JAG_REF_DRIVER, name="kern_jag")
    ref_lib.run_jag_ref.argtypes = [ctypes.POINTER(ctypes.c_double)]
    ref_lib.run_jag_ref.restype = None

    rng = np.random.default_rng(7)
    p_init = np.asfortranarray(rng.standard_normal(9))

    p_ref = p_init.copy(order="F")
    ref_lib.run_jag_ref(p_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    p_sdfg = p_init.copy(order="F")
    sdfg_lib.run_jag(p_sdfg.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    np.testing.assert_allclose(p_sdfg, p_ref, rtol=1e-12, atol=1e-12)
    # Sanity: the live data really was scaled (not a no-op match).
    np.testing.assert_allclose(p_ref, p_init * 2.0, rtol=1e-12, atol=1e-12)
