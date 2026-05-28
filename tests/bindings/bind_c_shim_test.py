"""Auto-generated ``bind(c)`` shim correctness.

:func:`dace_fortran.bindings.emit_bind_c_shim` and the
``build_fortran_library(..., bind_c_shim=True)`` option produce a
``<entry>_c.f90`` shim that wraps the binding module's
``<entry>_dace`` procedure under a stable C-ABI symbol so a ``ctypes``
or C caller can drive the SDFG ``.so`` without any hand-authored
Fortran glue.  These tests cover three things:

1. The emitter's text output for the representative flat shapes
   (scalar-in, scalar-out, rank-1 array, rank-2 array) -- structural
   check, no compile.
2. ``UnsupportedShimInterfaceError`` on a derived-type
   :class:`OriginalInterface` -- the MVP shim refuses anything its
   struct-construction extension doesn't yet handle.
3. End-to-end: build the SDFG, link with ``bind_c_shim=True``,
   ctypes-call the auto-generated symbol, assert numeric equivalence
   to a gfortran reference compiled from the same source.
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
    UnsupportedShimInterfaceError,
    build_fortran_library,
    emit_bind_c_shim,
)

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]


# ---------------------------------------------------------------------------
#  Emitter structural checks (text-only -- no compile, no link).
# ---------------------------------------------------------------------------


def test_emit_shim_scalar_in_array_in_out(tmp_path: Path):
    """A rank-1 input + rank-1 output + scalar-input dim ``n``: the
    scalar rides by value, the arrays ride as ``c_ptr`` with
    ``c_f_pointer`` aliases sized by ``n``."""
    iface = OriginalInterface(
        entry="kern",
        args=(
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="x", fortran_type="real(c_double)", rank=1,
                        shape=("n", ), intent="in"),
            OriginalArg(name="y", fortran_type="real(c_double)", rank=1,
                        shape=("n", ), intent="out"),
        ),
    )
    out = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90"))
    text = out.read_text()
    # bind(c) header + correct extern symbol name.
    assert "subroutine kern_c(n, x_p, y_p) bind(c, name='kern_c')" in text
    # Scalar input by value.
    assert "integer(c_int), value :: n" in text
    # Arrays come through as c_ptr.
    assert "type(c_ptr), value :: x_p" in text
    assert "type(c_ptr), value :: y_p" in text
    # c_f_pointer aliases use the scalar dim ``n`` directly.
    assert "real(c_double), pointer :: x(:)" in text
    assert "call c_f_pointer(x_p, x, [n])" in text
    assert "real(c_double), pointer :: y(:)" in text
    assert "call c_f_pointer(y_p, y, [n])" in text
    # The call uses the Fortran-side local aliases (not the ``_p`` names).
    assert "call kern_dace(n, x, y)" in text
    assert "call kern_dace_finalize()" in text
    # USE statement targets the right binding module.
    assert ("use kern_dace_bindings, only: kern_dace, kern_dace_finalize"
            in text)


def test_emit_shim_scalar_output_is_length1_array(tmp_path: Path):
    """A rank-0 ``intent(out)`` rides as ``c_ptr`` + ``c_f_pointer(..., [1])``
    -- matches ``feedback_scalar_io_convention`` (outputs = length-1
    array on the descriptor side)."""
    iface = OriginalInterface(
        entry="reduce",
        args=(
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="x", fortran_type="real(c_double)", rank=1,
                        shape=("n", ), intent="in"),
            OriginalArg(name="s", fortran_type="real(c_double)", rank=0, intent="out"),
        ),
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "reduce_c.f90")).read_text()
    assert "type(c_ptr), value :: s_p" in text
    assert "real(c_double), pointer :: s(:)" in text
    assert "call c_f_pointer(s_p, s, [1])" in text


def test_emit_shim_rank2_array_extents(tmp_path: Path):
    """Multi-dim shape extents are threaded verbatim into the
    ``c_f_pointer`` shape constructor in declaration order."""
    iface = OriginalInterface(
        entry="mat",
        args=(
            OriginalArg(name="m", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="a", fortran_type="real(c_double)", rank=2,
                        shape=("m", "n"), intent="inout"),
        ),
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "mat_c.f90")).read_text()
    assert "real(c_double), pointer :: a(:, :)" in text
    assert "call c_f_pointer(a_p, a, [m, n])" in text


def test_emit_shim_rejects_dynamic_shape_struct_member(tmp_path: Path):
    """A struct member whose extent is a Fortran symbol (here ``'n'``)
    -- rather than a literal -- is treated as dynamic and refused: the
    auto-shim's static-shape ``c_f_pointer`` alias cannot spell a
    runtime extent without an extra dim-passing convention the MVP
    doesn't define."""
    iface = OriginalInterface(
        entry="kern",
        args=(
            OriginalArg(name="st", fortran_type="type(t_state)", rank=0,
                        intent="inout", struct_type="t_state"),
        ),
        struct_types={
            "t_state":
            DerivedType(
                name="t_state",
                module="mo_state",
                members=(Member(name="u",
                                fortran_type="real(c_double)",
                                rank=1,
                                shape=("?", )), ),
            )
        },
        used_modules={"mo_state": ("t_state", )},
    )
    with pytest.raises(UnsupportedShimInterfaceError):
        emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90"))


def test_emit_shim_struct_with_static_array_members(tmp_path: Path):
    """A struct dummy whose every member is a static-shape array of
    scalar expands to one C-ABI slot per member, plus a local
    instance of the derived type assembled by copy-in and (for
    ``inout``) written back by copy-out."""
    iface = OriginalInterface(
        entry="kern",
        args=(
            OriginalArg(name="fld", fortran_type="type(t_fields)", rank=0,
                        intent="inout", struct_type="t_fields"),
        ),
        struct_types={
            "t_fields":
            DerivedType(
                name="t_fields",
                module="mo_fields",
                members=(
                    Member(name="a", fortran_type="real(c_double)",
                           rank=2, shape=("NX", "NY")),
                    Member(name="b", fortran_type="real(c_double)",
                           rank=2, shape=("NX", "NY")),
                ),
            )
        },
        used_modules={"mo_fields": ("t_fields", "NX", "NY")},
    )
    text = emit_bind_c_shim(iface,
                            str(tmp_path / "kern_c.f90")).read_text()
    # One C-ABI slot per member.
    assert "subroutine kern_c(fld_a_p, fld_b_p) bind(c, name='kern_c')" in text
    assert "type(c_ptr), value :: fld_a_p" in text
    assert "type(c_ptr), value :: fld_b_p" in text
    # Aliases sized to the struct module's shape constants.
    assert "real(c_double), pointer :: fld_a(:, :)" in text
    assert "call c_f_pointer(fld_a_p, fld_a, [NX, NY])" in text
    # Local struct + copy-in.
    assert "type(t_fields), target :: fld" in text
    assert "fld%a = fld_a" in text
    assert "fld%b = fld_b" in text
    # Copy-out (intent=inout).
    assert "fld_a = fld%a" in text
    assert "fld_b = fld%b" in text
    # The shim calls _dace with the struct, not the flat slots.
    assert "call kern_dace(fld)" in text
    # use-line for the struct's module survived.
    assert "use mo_fields, only: t_fields, NX, NY" in text


# ---------------------------------------------------------------------------
#  End-to-end: the auto-shim produces a Fortran-callable ``.so`` whose
#  ``<entry>_c`` symbol numerically matches a gfortran reference.
# ---------------------------------------------------------------------------

_KERNEL_SRC = """
subroutine axpy(n, a, x, y)
  implicit none
  integer, intent(in) :: n
  real(8), intent(in) :: a
  real(8), intent(in) :: x(n)
  real(8), intent(inout) :: y(n)
  integer :: i
  do i = 1, n
    y(i) = a * x(i) + y(i)
  end do
end subroutine axpy
"""

# Plain Fortran ``bind(c)`` reference driver: same flat C ABI as the
# auto-generated shim so the two ``.so``\\s are compared apples-to-apples
# through the same ctypes invocation.
_REF_DRIVER = """
subroutine axpy_c(n, a, x_p, y_p) bind(c, name="axpy_c")
  use iso_c_binding
  implicit none
  integer(c_int), value :: n
  real(c_double), value :: a
  type(c_ptr), value :: x_p, y_p
  real(c_double), pointer :: x(:), y(:)
  call c_f_pointer(x_p, x, [n])
  call c_f_pointer(y_p, y, [n])
  call axpy(n, a, x, y)
end subroutine axpy_c
"""


def test_bind_c_shim_e2e_axpy(tmp_path: Path):
    """``build_fortran_library(..., bind_c_shim=True)`` produces a
    ``.so`` with an ``axpy_c`` symbol the test invokes via
    ``ctypes``.  Compared bit-for-bit (``a*x + y_pre``) against a
    gfortran reference built from the same source through the same C
    ABI."""
    name = "axpy"

    # SDFG path: bridge build -> bind_fortran_library with shim.
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_KERNEL_SRC, sdfg_dir, name=name, entry=f"_QP{name}")
    sdfg = builder.build()
    sdfg.name = name
    sdfg.build_folder = str(tmp_path / "dacecache")
    lib = build_fortran_library(
        sdfg,
        out_dir=str(tmp_path / "lib"),
        name=name,
        bind_c_shim=True,
    )
    assert lib.bind_c_shim_f90 is not None and lib.bind_c_shim_f90.exists()
    sdfg_so = ctypes.CDLL(str(lib.so_path))

    # Reference path: gfortran-compile the kernel + a hand-written
    # ``axpy_c`` driver sharing the same C ABI as the auto-shim.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    kern_src = ref_dir / "axpy.f90"
    kern_src.write_text(_KERNEL_SRC)
    drv_src = ref_dir / "driver.f90"
    drv_src.write_text(_REF_DRIVER)
    ref_so = ref_dir / "libaxpy_ref.so"
    gfortran_compile_so(ref_so, kern_src, drv_src, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # Drive both through identical ctypes wiring.
    n = 8
    a = 2.5
    x = np.asfortranarray(np.arange(1, n + 1, dtype=np.float64))
    y_sdfg = np.full(n, 10.0, dtype=np.float64, order="F")
    y_ref = y_sdfg.copy(order="F")

    for so in (sdfg_so, ref_lib):
        fn = so.axpy_c
        fn.restype = None
        fn.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]
    sdfg_so.axpy_c(n, a, x.ctypes.data, y_sdfg.ctypes.data)
    ref_lib.axpy_c(n, a, x.ctypes.data, y_ref.ctypes.data)

    expected = a * x + 10.0
    np.testing.assert_allclose(y_ref, expected, rtol=0, atol=0)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
#  End-to-end: derived-type kernel with all-static-array members.
# ---------------------------------------------------------------------------

_STRUCT_KERNEL_SRC = """
module mo_shim_fields
  use iso_c_binding
  implicit none
  integer, parameter :: NX = 4, NY = 5
  type :: t_shim_fields
    real(c_double) :: a(NX, NY)
    real(c_double) :: b(NX, NY)
  end type t_shim_fields
end module mo_shim_fields

subroutine shim_kern(fld)
  use mo_shim_fields
  implicit none
  type(t_shim_fields), intent(inout) :: fld
  integer :: i, j
  do j = 1, NY
    do i = 1, NX
      fld%a(i, j) = fld%a(i, j) + fld%b(i, j)
    end do
  end do
end subroutine shim_kern
"""

# Plain Fortran ``bind(c)`` reference with the same flat C ABI the
# auto-shim produces: one ``c_ptr`` per member.  The reference loads
# the flat pointers into a local ``type(t_shim_fields)``, calls the
# un-transformed kernel, copies the post-call values back out.
_STRUCT_REF_DRIVER = """
subroutine shim_kern_c(fld_a_p, fld_b_p) bind(c, name='shim_kern_c')
  use iso_c_binding
  use mo_shim_fields, only: t_shim_fields, NX, NY
  implicit none
  type(c_ptr), value :: fld_a_p, fld_b_p
  real(c_double), pointer :: fld_a(:, :), fld_b(:, :)
  type(t_shim_fields), target :: fld
  external :: shim_kern
  call c_f_pointer(fld_a_p, fld_a, [NX, NY])
  call c_f_pointer(fld_b_p, fld_b, [NX, NY])
  fld%a = fld_a
  fld%b = fld_b
  call shim_kern(fld)
  fld_a = fld%a
  fld_b = fld%b
end subroutine shim_kern_c
"""

# Hand-authored interface kept as a regression anchor: the bridge
# member-layout snapshot (see ``test_auto_iface_struct_members``) is the
# preferred path; the explicit interface is the fallback when the
# bridge's snapshot doesn't carry enough info (a future v2 shape -- a
# nested struct, an allocatable member, ...).
_STRUCT_IFACE = OriginalInterface(
    entry="shim_kern",
    args=(OriginalArg(name="fld", fortran_type="type(t_shim_fields)",
                      rank=0, intent="inout",
                      struct_type="t_shim_fields"), ),
    struct_types={
        "t_shim_fields":
        DerivedType(name="t_shim_fields", module="mo_shim_fields",
                    members=(
                        Member(name="a", fortran_type="real(c_double)",
                               rank=2, shape=("NX", "NY")),
                        Member(name="b", fortran_type="real(c_double)",
                               rank=2, shape=("NX", "NY")),
                    ))
    },
    used_modules={"mo_shim_fields": ("t_shim_fields", "NX", "NY")},
)


def test_bind_c_shim_e2e_struct_two_real_array(tmp_path: Path):
    """``build_fortran_library(..., bind_c_shim=True)`` on a kernel
    whose only dummy is a ``type(t_shim_fields)`` with two static
    ``real(c_double)`` array members produces a ``shim_kern_c`` symbol
    that ``ctypes``-invokes through the same C ABI as a gfortran
    reference compiled from the same source."""
    name = "shim_kern"

    # Types module must precede the binding on the gfortran command
    # line; the shim ``USE``\\s ``mo_shim_fields`` so the types source
    # rides as a ``prelude_sources`` entry.
    types_path = tmp_path / "kernel_types.f90"
    types_path.write_text(_STRUCT_KERNEL_SRC)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_STRUCT_KERNEL_SRC, sdfg_dir, name=name,
                         entry=f"_QP{name}")
    plan_dict = builder.module.get_flatten_plan()
    sdfg = builder.build()
    sdfg.name = name
    sdfg.build_folder = str(tmp_path / "dacecache")
    from dace_fortran.bindings import FlattenPlan
    plan = FlattenPlan.from_dict(plan_dict)
    lib = build_fortran_library(
        sdfg,
        iface=_STRUCT_IFACE,
        plan=plan,
        out_dir=str(tmp_path / "lib"),
        name=name,
        prelude_sources=[types_path],
        bind_c_shim=True,
    )
    assert lib.bind_c_shim_f90 is not None and lib.bind_c_shim_f90.exists()
    sdfg_so = ctypes.CDLL(str(lib.so_path))

    # Reference: same source, hand-authored shim_kern_c shim.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    kern_src = ref_dir / "shim_kern.f90"
    kern_src.write_text(_STRUCT_KERNEL_SRC)
    drv_src = ref_dir / "driver.f90"
    drv_src.write_text(_STRUCT_REF_DRIVER)
    ref_so = ref_dir / "libshim_kern_ref.so"
    gfortran_compile_so(ref_so, kern_src, drv_src, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    rng = np.random.default_rng(17)
    nx, ny = 4, 5
    a_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    b_init = np.asfortranarray(rng.standard_normal((nx, ny)))
    a_sdfg = a_init.copy(order="F")
    b_sdfg = b_init.copy(order="F")
    a_ref = a_init.copy(order="F")
    b_ref = b_init.copy(order="F")

    for so in (sdfg_so, ref_lib):
        fn = so.shim_kern_c
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    sdfg_so.shim_kern_c(a_sdfg.ctypes.data, b_sdfg.ctypes.data)
    ref_lib.shim_kern_c(a_ref.ctypes.data, b_ref.ctypes.data)

    np.testing.assert_allclose(a_sdfg, a_ref, rtol=0, atol=0)
    np.testing.assert_allclose(b_sdfg, b_ref, rtol=0, atol=0)
    # Sanity-check that the kernel actually fired (vs. pass-through).
    np.testing.assert_allclose(a_ref, a_init + b_init, rtol=0, atol=0)
