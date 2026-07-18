"""Auto-generated ``bind(c)`` shim correctness.

Covers: (1) emitter text-output structural checks (no compile), (2)
``UnsupportedShimInterfaceError`` on unsupported derived-type interfaces, (3)
e2e -- build+link+ctypes-call the shim, compare against a gfortran reference.
"""
import ctypes
import re
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
    """Rank-1 in + rank-1 out + scalar dim ``n``: scalar rides by value,
    arrays ride as ``c_ptr`` with ``c_f_pointer`` aliases sized by ``n``."""
    iface = OriginalInterface(
        entry="kern",
        args=(
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="x", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="in"),
            OriginalArg(name="y", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="out"),
        ),
    )
    out = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90"))
    text = out.read_text()
    assert "subroutine kern_c(n, x_p, y_p) bind(c, name='kern_c')" in text
    assert "integer(c_int), value :: n" in text
    assert "type(c_ptr), value :: x_p" in text
    assert "type(c_ptr), value :: y_p" in text
    assert "real(c_double), pointer :: x(:)" in text
    assert "call c_f_pointer(x_p, x, [n])" in text
    assert "real(c_double), pointer :: y(:)" in text
    assert "call c_f_pointer(y_p, y, [n])" in text
    assert "call kern_dace(n, x, y)" in text
    assert "call kern_dace_finalize()" in text
    assert ("use kern_dace_bindings, only: kern_dace, kern_dace_finalize" in text)


def test_emit_shim_scalar_output_is_length1_array(tmp_path: Path):
    """Rank-0 ``intent(out)`` rides as ``c_ptr`` + ``c_f_pointer(..., [1])`` --
    matches ``feedback_scalar_io_convention`` (outputs = length-1 array)."""
    iface = OriginalInterface(
        entry="reduce",
        args=(
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="x", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="in"),
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
            OriginalArg(name="a", fortran_type="real(c_double)", rank=2, shape=("m", "n"), intent="inout"),
        ),
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "mat_c.f90")).read_text()
    assert "real(c_double), pointer :: a(:, :)" in text
    assert "call c_f_pointer(a_p, a, [m, n])" in text


def test_emit_shim_forwards_module_var_array_extents(tmp_path: Path):
    """Array dummy whose static shape references *module* vars (ICON ocean
    ``tracer(nproma, n_zlev)``) forwards them as ``integer(c_int), value`` args
    (pre-fix: bare names reached gfortran undeclared). Extents forward once,
    prepended to the arg list; an in-scope scalar dummy is NOT re-forwarded."""
    iface = OriginalInterface(
        entry="ppm",
        args=(
            OriginalArg(name="vt", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="tracer", fortran_type="real(c_double)", rank=2, shape=("nproma", "n_zlev"), intent="in"),
            OriginalArg(name="w", fortran_type="real(c_double)", rank=2, shape=("nproma", "n_zlev + 1"), intent="in"),
            OriginalArg(name="flux", fortran_type="real(c_double)", rank=2, shape=("nproma", "n_zlev"), intent="out"),
        ),
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "ppm_c.f90")).read_text()
    # Each distinct module-var extent forwarded once, as a value arg.
    assert "integer(c_int), value :: nproma" in text
    assert "integer(c_int), value :: n_zlev" in text
    assert text.count("integer(c_int), value :: nproma") == 1
    assert text.count("integer(c_int), value :: n_zlev") == 1
    # Prepended (extents first), ahead of the scalar dummy + array ptrs.
    assert ("subroutine ppm_c(nproma, n_zlev, vt, tracer_p, w_p, flux_p) "
            "bind(c, name='ppm_c')" in text)
    # c_f_pointer shapes resolve against the forwarded names; n_zlev + 1 rides verbatim.
    assert "call c_f_pointer(tracer_p, tracer, [nproma, n_zlev])" in text
    assert "call c_f_pointer(w_p, w, [nproma, n_zlev + 1])" in text
    assert "call c_f_pointer(flux_p, flux, [nproma, n_zlev])" in text


def test_emit_shim_scalar_dummy_extent_not_forwarded(tmp_path: Path):
    """Array sized by a scalar *dummy* (``a(n)``) stays on the existing path --
    ``n`` is already in scope, so it's NOT also forwarded as a module-var extent."""
    iface = OriginalInterface(
        entry="kern",
        args=(
            OriginalArg(name="n", fortran_type="integer(c_int)", rank=0, intent="in"),
            OriginalArg(name="a", fortran_type="real(c_double)", rank=1, shape=("n", ), intent="inout"),
        ),
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    # Header is unchanged from the pre-fix shape (no spurious extent arg).
    assert "subroutine kern_c(n, a_p) bind(c, name='kern_c')" in text
    assert "call c_f_pointer(a_p, a, [n])" in text


def test_emit_shim_dynamic_shape_struct_member(tmp_path: Path):
    """Struct member with dynamic extent (``'?'``): each dim rides as a
    LOWER-BOUND/EXTENT value pair (``<flat>_lb<i>``/``<flat>_d<i>``). ALLOCATABLE/
    POINTER members can't ``=>``-alias uniformly, so the shim ``allocate``s at
    TRUE bounds ``(lb : lb + d - 1)`` and element-copies in/out per ``intent``."""
    iface = OriginalInterface(
        entry="kern",
        args=(OriginalArg(name="st", fortran_type="type(t_state)", rank=0, intent="inout", struct_type="t_state"), ),
        struct_types={
            "t_state":
            DerivedType(
                name="t_state",
                module="mo_state",
                members=(Member(name="u", fortran_type="real(c_double)", rank=1, shape=("?", )), ),
            )
        },
        used_modules={"mo_state": ("t_state", )},
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    # Per-dim lower-bound then extent arg precede the pointer arg.
    assert "integer(c_int), value :: st_u_lb0" in text
    assert "integer(c_int), value :: st_u_d0" in text
    assert "type(c_ptr), value :: st_u_p" in text
    # The c_f_pointer shape constructor references the extent arg.
    assert "call c_f_pointer(st_u_p, st_u, [st_u_d0])" in text
    # Allocate at TRUE bounds (lb : lb + d - 1), element copy-in/out -- valid for POINTER + ALLOCATABLE.
    assert "allocate(st%u(st_u_lb0 : st_u_lb0 + st_u_d0 - 1))" in text
    assert "st%u = st_u" in text
    assert "st_u = st%u" in text


def test_emit_shim_struct_with_static_array_members(tmp_path: Path):
    """Struct dummy with all-static-shape-array members expands to one C-ABI
    slot per member, plus a local instance assembled by copy-in/copy-out."""
    iface = OriginalInterface(
        entry="kern",
        args=(OriginalArg(name="fld", fortran_type="type(t_fields)", rank=0, intent="inout", struct_type="t_fields"), ),
        struct_types={
            "t_fields":
            DerivedType(
                name="t_fields",
                module="mo_fields",
                members=(
                    Member(name="a", fortran_type="real(c_double)", rank=2, shape=("NX", "NY")),
                    Member(name="b", fortran_type="real(c_double)", rank=2, shape=("NX", "NY")),
                ),
            )
        },
        used_modules={"mo_fields": ("t_fields", "NX", "NY")},
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    assert "subroutine kern_c(fld_a_p, fld_b_p) bind(c, name='kern_c')" in text
    assert "type(c_ptr), value :: fld_a_p" in text
    assert "type(c_ptr), value :: fld_b_p" in text
    # Aliases sized to the struct module's shape constants.
    assert "real(c_double), pointer :: fld_a(:, :)" in text
    assert "call c_f_pointer(fld_a_p, fld_a, [NX, NY])" in text
    assert "type(t_fields), target :: fld" in text
    assert "fld%a = fld_a" in text
    assert "fld%b = fld_b" in text
    # Copy-out (intent=inout).
    assert "fld_a = fld%a" in text
    assert "fld_b = fld%b" in text
    # calls _dace with the struct, not the flat slots.
    assert "call kern_dace(fld)" in text
    assert "use mo_fields, only: t_fields, NX, NY" in text


def test_emit_shim_value_record_array_scatters_elementwise(tmp_path: Path):
    """ARRAY of a flat *value* record (e.g. ``t_cartesian_coordinates%x(3)``)
    reconstructs element-wise: local allocatable + flat companion (rank outer+
    member, outer-dims-first) + nested scatter/gather loops -- NOT an illegal
    whole-array ``arr%x`` descent."""
    iface = OriginalInterface(
        entry="kern",
        args=(OriginalArg(name="p",
                          fortran_type="type(t_cc)",
                          rank=2,
                          intent="inout",
                          struct_type="t_cc",
                          shape=(":", ":")), ),
        struct_types={
            "t_cc":
            DerivedType(name="t_cc",
                        module="mo_cc",
                        members=(Member(name="x", fortran_type="real(c_double)", rank=1, shape=("3", )), ))
        },
        used_modules={"mo_cc": ("t_cc", )},
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    # Local allocatable + PER-FIELD outer-extent value args (<flat>_<field>_d<i>),
    # matching the SDFG's one-companion-per-field flatten.
    assert "type(t_cc), allocatable, target :: p(:, :)" in text
    assert "integer(c_int), value :: p_x_d0" in text
    assert "integer(c_int), value :: p_x_d1" in text
    # Flat companion: rank outer(2) + member(1), sized [outer..., 3].
    assert "real(c_double), pointer :: p_x(:, :, :)" in text
    assert "call c_f_pointer(p_x_p, p_x, [p_x_d0, p_x_d1, 3])" in text
    # Shared allocate uses the first (only) field's extents.
    assert "allocate(p(p_x_d0, p_x_d1))" in text
    # Element-wise scatter (copy-in) then gather (copy-out, inout).
    assert "p(p_x_i0, p_x_i1)%x(p_x_i2) = p_x(p_x_i0, p_x_i1, p_x_i2)" in text
    assert "p_x(p_x_i0, p_x_i1, p_x_i2) = p(p_x_i0, p_x_i1)%x(p_x_i2)" in text
    # struct array, not flat slots, passed to _dace.
    assert "call kern_dace(p)" in text


def test_emit_shim_value_record_array_multifield_per_field_extents(tmp_path: Path):
    """MULTI-field value record (``t_tangent_vectors {v1, v2}``) array member
    emits PER-FIELD extents: each field's ``<flat>_<field>_d<i>`` block precedes
    its pointer, matching ``emit_library``'s per-leaf ABI order slot-for-slot.
    One shared allocate (array-of-records) uses the first field's extents."""
    iface = OriginalInterface(
        entry="kern",
        args=(OriginalArg(name="s", fortran_type="type(t_edges)", rank=0, intent="in", struct_type="t_edges"), ),
        struct_types={
            "t_edges":
            DerivedType(name="t_edges",
                        module="mo_edges",
                        members=(Member(name="pnc",
                                        fortran_type="type(t_tv)",
                                        rank=3,
                                        shape=("?", "?", "?"),
                                        struct_name="t_tv"), )),
            "t_tv":
            DerivedType(name="t_tv",
                        module="mo_edges",
                        members=(Member(name="v1", fortran_type="real(c_double)",
                                        rank=0), Member(name="v2", fortran_type="real(c_double)", rank=0))),
        },
        used_modules={"mo_edges": ("t_edges", "t_tv")},
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    # Per-field extents: each field gets its own ``_d<i>`` block.
    for f in ("v1", "v2"):
        for d in range(3):
            assert f"integer(c_int), value :: s_pnc_{f}_d{d}" in text, f"missing per-field extent s_pnc_{f}_d{d}"
        assert f"type(c_ptr), value :: s_pnc_{f}_p" in text
        assert f"call c_f_pointer(s_pnc_{f}_p, s_pnc_{f}, [s_pnc_{f}_d0, s_pnc_{f}_d1, s_pnc_{f}_d2])" in text
    # extents ride immediately before their pointer (v1_d0..d2, v1_p, v2_d0..d2,
    # v2_p) -- emit_library's per_member_soa interleave.
    sig = re.search(r"subroutine\s+kern_c\(([^)]*)\)", text, re.S).group(1).replace("&", " ")
    order = [a.strip() for a in sig.split(",") if a.strip()]
    assert order.index("s_pnc_v1_d0") < order.index("s_pnc_v1_p") < order.index("s_pnc_v2_d0") \
        < order.index("s_pnc_v2_p"), f"per-field extent/pointer interleave wrong: {order}"
    # Single shared allocate from the first field's extents.
    assert "allocate(s%pnc(s_pnc_v1_d0, s_pnc_v1_d1, s_pnc_v1_d2))" in text


def test_emit_shim_pointer_array_record_indexed_and_scalar_extent_by_value(tmp_path: Path):
    """Struct dummy with a rank-1 pointer-array-of-record member (``p1d(:)``)
    allocates to size 1, descended at ``(1)`` (ICON single-patch idiom). A
    scalar member that's ALSO a flat array's extent rides ONCE by value --
    not duplicated as a length-1 pointer alias."""
    iface = OriginalInterface(
        entry="kern",
        args=(
            OriginalArg(name="patch", fortran_type="type(t_patch3d)", rank=0, intent="in", struct_type="t_patch3d"),
            OriginalArg(name="fld", fortran_type="real(c_double)", rank=1, intent="inout", shape=("patch_p1d_nblk", )),
        ),
        struct_types={
            "t_patch3d":
            DerivedType(name="t_patch3d",
                        module="mo_dom",
                        members=(Member(name="p1d",
                                        fortran_type="type(t_pv)",
                                        rank=1,
                                        shape=("?", ),
                                        struct_name="t_pv"), )),
            "t_pv":
            DerivedType(name="t_pv",
                        module="mo_dom",
                        members=(Member(name="nblk", fortran_type="integer(c_int)", rank=0),
                                 Member(name="dolic", fortran_type="integer(c_int)", rank=2, shape=("?", "?")))),
        },
        used_modules={"mo_dom": ("t_patch3d", "t_pv")},
    )
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    assert "allocate(patch%p1d(1))" in text
    assert "patch%p1d(1)%nblk = patch_p1d_nblk" in text
    assert ("allocate(patch%p1d(1)%dolic(patch_p1d_dolic_lb0 : patch_p1d_dolic_lb0 + patch_p1d_dolic_d0 - 1, "
            "patch_p1d_dolic_lb1 : patch_p1d_dolic_lb1 + patch_p1d_dolic_d1 - 1))") in text
    assert "patch%p1d(1)%dolic = patch_p1d_dolic" in text
    assert "integer(c_int), value :: patch_p1d_nblk" in text
    assert text.count("patch_p1d_nblk") == 4  # header arg + decl + struct copy + array shape
    assert "integer(c_int), pointer :: patch_p1d_nblk(:)" not in text
    assert "call c_f_pointer(fld_p, fld, [patch_p1d_nblk])" in text


# ---------------------------------------------------------------------------
#  End-to-end: auto-shim's ``.so`` numerically matches a gfortran reference.
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

# Plain Fortran bind(c) reference driver: same flat C ABI as the auto-shim,
# so both .so's are compared apples-to-apples through the same ctypes call.
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
    """``bind_c_shim=True`` produces an ``axpy_c`` symbol invoked via ``ctypes``;
    compared bit-for-bit against a gfortran reference through the same C ABI."""
    name = "axpy"

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

    # Reference: gfortran-compile the kernel + a hand-written axpy_c driver on the same C ABI.
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

# Plain Fortran bind(c) reference, same flat C ABI as the auto-shim (one
# c_ptr per member): loads pointers into a local struct, calls the
# un-transformed kernel, copies values back out.
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
# member-layout snapshot is the preferred path; this is the fallback for
# shapes it can't carry yet (nested struct, allocatable member, ...).
_STRUCT_IFACE = OriginalInterface(
    entry="shim_kern",
    args=(OriginalArg(name="fld",
                      fortran_type="type(t_shim_fields)",
                      rank=0,
                      intent="inout",
                      struct_type="t_shim_fields"), ),
    struct_types={
        "t_shim_fields":
        DerivedType(name="t_shim_fields",
                    module="mo_shim_fields",
                    members=(
                        Member(name="a", fortran_type="real(c_double)", rank=2, shape=("NX", "NY")),
                        Member(name="b", fortran_type="real(c_double)", rank=2, shape=("NX", "NY")),
                    ))
    },
    used_modules={"mo_shim_fields": ("t_shim_fields", "NX", "NY")},
)


def test_bind_c_shim_e2e_struct_two_real_array(tmp_path: Path):
    """``bind_c_shim=True`` on a struct dummy with two static array members
    produces a ``shim_kern_c`` symbol matching a gfortran reference via ``ctypes``."""
    name = "shim_kern"

    # Types module must precede the binding on gfortran's cmdline; rides as a prelude_sources entry.
    types_path = tmp_path / "kernel_types.f90"
    types_path.write_text(_STRUCT_KERNEL_SRC)

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(_STRUCT_KERNEL_SRC, sdfg_dir, name=name, entry=f"_QP{name}")
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


# ---------------------------------------------------------------------------
#  Pointer-to-record HANDLE members (ICON's ``POINTER :: comm_pat_c``) have no
#  SoA image; marshal + FlattenStructs both skip them via one shared
#  ``pointerToRecordMember`` predicate. The bridge's struct-layout snapshot
#  feeding ``emit_bind_c_shim`` must skip it too, or the shim's per-member-SoA
#  ABI desyncs from the marshaller. A value-record ARRAY member is not a
#  handle and stays.
# ---------------------------------------------------------------------------
_HANDLE_SNAPSHOT_SRC = """
module m_handle_snap
  use iso_c_binding
  implicit none
  type :: t_cpat
    integer(c_int) :: n_recv
    integer(c_int), allocatable :: recv_limits(:)
  end type
  type :: t_patch
    integer(c_int) :: nblks_e
    real(c_double), allocatable :: area(:, :)
    type(t_cpat), pointer :: comm_pat_c
  end type
contains
  subroutine kern(p, je, jb, out)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: out
    out = p % area(je, jb) + real(p % nblks_e, c_double)
  end subroutine
end module
"""


def test_snapshot_skips_pointer_to_record_handle_member(tmp_path):
    """Struct-layout snapshot omits a pointer-to-record handle member; shim
    reconstructs real data members (``area``, ``nblks_e``) but none of the handle's pointee record."""
    from dace_fortran.bindings.fortran_interface import build_auto_interface
    sdfg_dir = tmp_path / "sdfg"
    sdfg = build_sdfg(_HANDLE_SNAPSHOT_SRC, sdfg_dir, name="kern", entry="m_handle_snap::kern").build()
    iface = build_auto_interface(sdfg._fortran_interface_raw, "kern")
    # The handle member carries no layout the shim can reconstruct.
    tp = iface.struct_types["t_patch"]
    member_names = {m.name for m in tp.members}
    assert "comm_pat_c" not in member_names, \
        f"pointer-to-record handle leaked into the struct snapshot: {sorted(member_names)}"
    assert {"nblks_e", "area"} <= member_names, f"real data members missing: {sorted(member_names)}"
    text = emit_bind_c_shim(iface, str(tmp_path / "kern_c.f90")).read_text()
    assert "comm_pat_c" not in text, "shim reconstructed the pointer-to-record handle"
    assert "area" in text and "nblks_e" in text
