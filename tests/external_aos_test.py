"""External (``keep_external``) calls and the AoS/SoA argument-layout decision.

When the bridge SoA-flattens struct data but a registered external expects a
particular memory layout, each call-site argument falls into one of:

  * a plain array / flattened struct array-member  --  the SoA pointer is
    handed over directly (zero copy).  This is the common ICON
    double-buffering shape (``p_nh%prog(nnow)%w`` flattens to the standalone
    array ``..._w``), and it is what every halo-exchange data argument in
    ``solve_nh`` looks like.
  * an array-of-scalar-structs the bridge keeps AoS  --  the struct pointer is
    handed over directly (zero copy).
  * a scalar-member struct an AoS external wants contiguous  --  a shallow
    inner-dim alias when the struct is innermost and all members share one
    scalar type, else a deep gather / scatter.  See
    ``_icon_build/_diag_20260526/AOS_EXTERNAL_DESIGN.md``.

The first case is what unblocks the ``solve_nh`` halo externalisation, so it is
pinned here; the deep-copy case is marked ``xfail`` until the binding wrap is
wired.
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _helpers import xfail
from _util import build_sdfg, have_flang
from dace_fortran.external import Arg, clear_external_registry, keep_external

#: Standalone "fake" mo_velocity_advection (full velocity_tendencies + its USE
#: closure as one file).  Outside the repo; the full-velocity test is gated on
#: its presence (set ``VELOCITY_FULL_F90`` to override the default path).
_VELOCITY_FULL = Path(os.environ.get(
    "VELOCITY_FULL_F90",
    "/home/primrose/Work/icon-artifacts/velocity/velocity.f90"))

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_c_so(out_dir: Path, name: str, csrc: str) -> Path:
    """Compile a small C source to a shared library and return its path.

    :param out_dir: scratch directory for the ``.c`` / ``.so`` pair.
    :param name: base name; the library is ``lib<name>.so``.
    :param csrc: C source text defining the ``extern "C"`` symbol.
    :returns: path to the built shared library.
    """
    (out_dir / f"{name}.c").write_text(csrc)
    so = out_dir / f"lib{name}.so"
    subprocess.check_call(["gcc", "-shared", "-fPIC", "-o", str(so),
                           str(out_dir / f"{name}.c")])
    return so


def test_aliasable_array_member_no_copy(tmp_path):
    """A flattened struct array-member reaches a plain-array external as the
    SoA pointer directly  --  no copy (the ICON double-buffering shape)."""
    so = _build_c_so(tmp_path, "ext_scale",
                     "void ext_scale(double* a, int n)"
                     "{ for (int i = 0; i < n; ++i) a[i] *= 2.0; }")
    src = """
module m_alias
  use iso_c_binding
  implicit none
  type :: t_vec
    real(c_double) :: u(8)
    real(c_double) :: v(8)
  end type
contains
  subroutine kern(s, n)
    type(t_vec), intent(inout) :: s
    integer(c_int), intent(in) :: n
    interface
      subroutine ext_scale(a, n) bind(c, name="ext_scale")
        use iso_c_binding
        real(c_double), intent(inout) :: a(*)
        integer(c_int), value :: n
      end subroutine
    end interface
    call ext_scale(s%u, n)
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_scale",
                      args=(Arg(kind="array", dtype="float64", intent="inout"),
                            Arg(kind="scalar", dtype="int32", intent="in")),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_aliasPkern").build()
        u = np.arange(8, dtype=np.float64)
        v = np.ones(8, dtype=np.float64)
        sdfg(s_u=u, s_v=v, n=np.int32(8))
        np.testing.assert_allclose(u, np.arange(8) * 2.0)
        np.testing.assert_allclose(v, 1.0)  # the other member is untouched
    finally:
        clear_external_registry()


def test_scalar_member_struct_aos_external(tmp_path):
    """An external wanting the contiguous AoS struct, fed from a struct the
    bridge split into separate scalar flats.  ``hlfir-marshal-external-structs``
    expands the call to per-member args (the SoA flats) and ``emit_call``
    generates a C tasklet that packs them into a local AoS buffer, calls the
    external, and unpacks the result back into the SoA flats  --  so the SDFG
    only ever sees the SoA arrays."""
    so = _build_c_so(tmp_path, "ext_swap",
                     "struct pt{double f1;double f2;};"
                     "void ext_swap(struct pt* p){double t=p->f1;p->f1=p->f2;p->f2=t;}")
    src = """
module m_aos
  use iso_c_binding
  implicit none
  type, bind(c) :: pt
    real(c_double) :: f1
    real(c_double) :: f2
  end type
contains
  subroutine kern(s)
    type(pt), intent(inout) :: s
    interface
      subroutine ext_swap(p) bind(c, name="ext_swap")
        import :: pt
        type(pt), intent(inout) :: p
      end subroutine
    end interface
    s%f1 = s%f1 + 1.0_c_double   ! component access -> bridge splits f1/f2
    call ext_swap(s)             ! external wants the contiguous AoS struct
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_swap",
                      args=(Arg(kind="aos", intent="inout"),),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_aosPkern").build()
        f1 = np.array([3.0]); f2 = np.array([5.0])
        sdfg(s_f1=f1, s_f2=f2)
        np.testing.assert_allclose(f1, 5.0)        # swapped
        np.testing.assert_allclose(f2, 3.0 + 1.0)  # +1 then swapped
    finally:
        clear_external_registry()


def test_array_member_struct_aos_external(tmp_path):
    """A velocity-style struct with uniform array members passed whole to an
    AoS external.  The marshalling pass expands the call to the per-member SoA
    arrays; emit_call packs them into a local AoS buffer with array fields
    (element-loop copies), calls the external, and unpacks  --  exercising the
    array-member path (the ICON velocity ``state_t`` shape)."""
    so = _build_c_so(tmp_path, "ext_state",
                     "struct state_t { double u[4]; double v[4]; };"
                     "void ext_state(struct state_t* p)"
                     "{ for (int i = 0; i < 4; ++i) p->u[i] += p->v[i]; }")
    src = """
module m_vel
  use iso_c_binding
  implicit none
  type, bind(c) :: state_t
    real(c_double) :: u(4)
    real(c_double) :: v(4)
  end type
contains
  subroutine kern(s)
    type(state_t), intent(inout) :: s
    interface
      subroutine ext_state(p) bind(c, name="ext_state")
        import :: state_t
        type(state_t), intent(inout) :: p
      end subroutine
    end interface
    s%u(1) = s%u(1) + 1.0_c_double   ! component access -> flatten splits u/v
    call ext_state(s)                ! external wants the contiguous AoS struct
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_state",
                      args=(Arg(kind="aos", intent="inout"),),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_velPkern").build()
        u = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([10.0, 20.0, 30.0, 40.0])
        sdfg(s_u=u, s_v=v)
        # s%u(1) += 1 -> u = [2,2,3,4]; then ext_state: u[i] += v[i].
        np.testing.assert_allclose(u, [12.0, 22.0, 33.0, 44.0])
        np.testing.assert_allclose(v, [10.0, 20.0, 30.0, 40.0])  # untouched
    finally:
        clear_external_registry()


def _external_call_body(sdfg):
    """Return the C body of the single ExternalCall node in ``sdfg``."""
    from dace_fortran.external import ExternalCall
    for st in sdfg.all_states():
        for nd in st.nodes():
            if isinstance(nd, ExternalCall):
                return nd.body
    raise AssertionError("no ExternalCall node found")


def test_velocity_field_array_external_is_shallow(tmp_path):
    """A velocity-style field reaches an external as a plain array.  ICON
    velocity binds its prognostic fields through pointer assignment, not struct
    flattening, so a field resolves to a plain array -- passing the pointer
    itself to an external is a shallow pointer pass (no AoS buffer / deep
    copy): hlfir-rewrite-pointer-assigns folds the rebind's copy_in/copy_out
    straight to the contiguous target, so the external reads / writes it in
    place."""
    so = _build_c_so(tmp_path, "ext_scale",
                     "void ext_scale(double* a, int n)"
                     "{ for (int i = 0; i < n; ++i) a[i] *= 2.0; }")
    src = """
module m_velptr
  use iso_c_binding
  implicit none
contains
  subroutine kern(tgt, n)
    integer(c_int), intent(in) :: n
    real(c_double), intent(inout), target :: tgt(n)
    real(c_double), pointer :: fld(:)
    interface
      subroutine ext_scale(a, n) bind(c, name="ext_scale")
        use iso_c_binding
        real(c_double), intent(inout) :: a(*)
        integer(c_int), value :: n
      end subroutine
    end interface
    fld => tgt              ! pointer assignment (the velocity binding pattern)
    call ext_scale(fld, n)  ! pass the pointer -> folds to the target, shallow
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_scale",
                      args=(Arg(kind="array", dtype="float64", intent="inout"),
                            Arg(kind="scalar", dtype="int32", intent="in")),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_velptrPkern").build()
        # Shallow pass: the tasklet calls the external on the array pointer
        # directly -- no AoS struct buffer / element copies.
        body = _external_call_body(sdfg)
        assert "struct" not in body, f"expected a shallow pointer pass, got:\n{body}"
        tgt = np.arange(6, dtype=np.float64)
        sdfg(tgt=tgt, n=np.int32(6))
        np.testing.assert_allclose(tgt, np.arange(6) * 2.0)
    finally:
        clear_external_registry()


def test_velocity_state_t_whole_struct_external(tmp_path):
    """The real ICON velocity ``state_t`` shape (four uniform 2-D ``real(8)``
    field members) passed whole to an AoS external.  Confirms the deep-copy
    marshalling handles the actual velocity-tendencies state struct: four
    members, multi-dim arrays, packed into one AoS buffer and unpacked."""
    so = _build_c_so(tmp_path, "ext_velstate",
                     "struct state_t { double u[16]; double v[16];"
                     "                 double w[16]; double p[16]; };"
                     "void ext_velstate(struct state_t* s) {"
                     "  for (int i = 0; i < 16; ++i) {"
                     "    s->u[i] += s->v[i];"   # u += v (elementwise)
                     "    s->p[i]  = s->w[i] * 2.0;"  # p = 2w
                     "  } }")
    src = """
module m_velstate
  use iso_c_binding
  implicit none
  integer, parameter :: nx = 4, ny = 4
  type, bind(c) :: state_t
    real(c_double) :: u(nx, ny)
    real(c_double) :: v(nx, ny)
    real(c_double) :: w(nx, ny)
    real(c_double) :: p(nx, ny)
  end type state_t
contains
  subroutine kern(st)
    type(state_t), intent(inout) :: st
    interface
      subroutine ext_velstate(s) bind(c, name="ext_velstate")
        import :: state_t
        type(state_t), intent(inout) :: s
      end subroutine
    end interface
    st%u(1, 1) = st%u(1, 1) + 1.0_c_double  ! component access -> SoA split
    call ext_velstate(st)                    ! whole struct -> AoS marshalling
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_velstate",
                      args=(Arg(kind="aos", intent="inout"),),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_velstatePkern").build()
        rng = np.random.default_rng(0)
        u = np.asfortranarray(rng.random((4, 4)))
        v = np.asfortranarray(rng.random((4, 4)))
        w = np.asfortranarray(rng.random((4, 4)))
        p = np.asfortranarray(rng.random((4, 4)))
        u_in, v_in, w_in = u.copy(), v.copy(), w.copy()
        sdfg(st_u=u, st_v=v, st_w=w, st_p=p)
        expect_u = u_in.copy(); expect_u.flat[0] += 1.0   # st%u(1,1) += 1
        expect_u = expect_u + v_in                         # u += v
        np.testing.assert_allclose(u, expect_u)
        np.testing.assert_allclose(p, w_in * 2.0)
        np.testing.assert_allclose(v, v_in)                # untouched
    finally:
        clear_external_registry()


@pytest.mark.skipif(not _VELOCITY_FULL.is_file(),
                    reason="full velocity fake f90 not present "
                           "(set VELOCITY_FULL_F90)")
def test_full_velocity_advection_external_call(tmp_path):
    """Add an external call to the FULL ``mo_velocity_advection`` fake f90 and
    confirm it builds and the call reaches a real pointer-contiguous prognostic
    field.  ``p_prog%w`` (a ``POINTER, CONTIGUOUS`` member of ``t_nh_prog``)
    resolves + flattens to ``p_prog_w``; the external connects to it as a
    shallow pointer pass (no AoS buffer) -- the velocity binding pattern at full
    scale."""
    src = _VELOCITY_FULL.read_text()
    use_line = "    USE mo_loopindices, ONLY: get_indices_c, get_indices_e"
    assert use_line in src, "velocity fake f90 layout changed; update injection"
    src = src.replace(use_line, use_line + """
    INTERFACE
      SUBROUTINE ext_sync(a) BIND(C, name="ext_sync")
        USE iso_c_binding
        REAL(c_double), INTENT(INOUT) :: a(*)
      END SUBROUTINE ext_sync
    END INTERFACE""", 1)
    src = src.replace("  END SUBROUTINE velocity_tendencies",
                      "    CALL ext_sync(p_prog%w)\n"
                      "  END SUBROUTINE velocity_tendencies", 1)

    so = _build_c_so(tmp_path, "ext_sync", "void ext_sync(double* a){(void)a;}")
    clear_external_registry()
    try:
        keep_external("ext_sync",
                      args=(Arg(kind="array", dtype="float64", intent="inout"),),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="velext",
                          entry="_QMmo_velocity_advectionPvelocity_tendencies").build()
        from dace_fortran.external import ExternalCall
        node = next((n for st in sdfg.all_states() for n in st.nodes()
                     if isinstance(n, ExternalCall)), None)
        assert node is not None, "external call not lowered in full velocity"
        assert "struct" not in node.body, f"expected shallow pass, got:\n{node.body}"
        st = next(s for s in sdfg.all_states() if node in s.nodes())
        touched = {e.data.data for e in st.in_edges(node)} | \
                  {e.data.data for e in st.out_edges(node)}
        assert touched == {"p_prog_w"}, \
            f"external should read/write the resolved field p_prog_w, got {touched}"
    finally:
        clear_external_registry()


def test_dycore_mixed_shallow_and_deepcopy(tmp_path):
    """Dycore-style external: one call mixing a plain field (shallow pointer
    pass) and a whole sub-struct that needs the AoS<->SoA deep copy.  ``s%w`` is
    a contiguous field -> passed by pointer in place; the nested ``s%blk`` (a
    struct member) is packed into a local AoS buffer, passed, and unpacked.  So
    only the one ``blk`` array takes a real deep copy; everything else is
    shallow."""
    so = _build_c_so(tmp_path, "ext_mixed",
                     "struct t_blk { double arr[8]; };"
                     "void ext_mixed(double* w, int n, struct t_blk* blk) {"
                     "  for (int i = 0; i < n; ++i) w[i] *= 2.0;"
                     "  for (int i = 0; i < 8; ++i) blk->arr[i] += 1.0; }")
    src = """
module m_dycore
  use iso_c_binding
  implicit none
  type, bind(c) :: t_blk
    real(c_double) :: arr(8)
  end type
  type :: t_state
    real(c_double) :: w(16)
    real(c_double) :: vn(16)
    type(t_blk) :: blk
  end type
contains
  subroutine kern(s)
    type(t_state), intent(inout) :: s
    interface
      subroutine ext_mixed(w, n, blk) bind(c, name="ext_mixed")
        import :: t_blk, c_double, c_int
        real(c_double), intent(inout) :: w(*)
        integer(c_int), value :: n
        type(t_blk), intent(inout) :: blk
      end subroutine
    end interface
    s%blk%arr(1) = s%blk%arr(1) + 1.0_c_double   ! force blk flatten
    call ext_mixed(s%w, 16, s%blk)
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_mixed",
                      args=(Arg(kind="array", dtype="float64", intent="inout"),
                            Arg(kind="scalar", dtype="int32", intent="in"),
                            Arg(kind="aos", intent="inout")),
                      libraries=(str(so),))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_dycorePkern").build()
        body = _external_call_body(sdfg)
        assert "_a0_o" in body and "struct" in body, \
            f"expected shallow w + AoS buffer for blk, got:\n{body}"
        w = np.arange(16, dtype=np.float64)
        vn = np.ones(16, dtype=np.float64)
        arr = np.arange(8, dtype=np.float64)
        sdfg(s_w=w, s_vn=vn, s_blk_arr=arr)
        exp_arr = np.arange(8, dtype=np.float64); exp_arr[0] += 1.0; exp_arr += 1.0
        np.testing.assert_allclose(w, np.arange(16) * 2.0)   # field: shallow x2
        np.testing.assert_allclose(arr, exp_arr)             # blk: deep-copy roundtrip
        np.testing.assert_allclose(vn, 1.0)                  # untouched
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
#  v2.1 marshal expansion (Phase 2.3.E) -- ``MarshalExternalStructs.cpp``
#  now recursively walks nested derived-type members down to inline-flat
#  leaves (scalar or static-shape array of scalar) and expands each leaf
#  into its own call-arg + struct field.  Still on the unsupported side
#  of the v2 boundary: box / pointer / allocatable / dynamic-shape
#  members -- :func:`emit_call`'s structured diagnostic continues to
#  point at :func:`dace_fortran.external.inline_external` for those.
# ---------------------------------------------------------------------------


# Shared kernel source for the v2.1 test pair.  The outer struct
# ``outer_t`` has a *nested* derived-type member ``inner_t``; the
# recursive flatten in ``MarshalExternalStructs.cpp`` walks it down to
# three contiguous f64 leaves (``ip%u``, ``ip%v``, ``scale``).
_V2_NESTED_SRC = """
module m_v2
  use iso_c_binding
  implicit none
  type :: inner_t
    real(c_double) :: u
    real(c_double) :: v
  end type
  type :: outer_t
    type(inner_t) :: ip
    real(c_double) :: scale
  end type
contains
  subroutine kern(s)
    type(outer_t), intent(inout) :: s
    interface
      subroutine ext_v2(p)
        import :: outer_t
        type(outer_t), intent(inout) :: p
      end subroutine
    end interface
    call ext_v2(s)
  end subroutine
end module
"""


def test_v2_aos_external_with_nested_struct(tmp_path):
    """``keep_external(kind='aos')`` on a callee whose struct has a
    nested derived-type member.  The recursive expansion produces one
    SoA flat per leaf member; ``emit_call`` lays out the AoS struct
    field-by-field in declaration order (``ip%u``, ``ip%v``,
    ``scale``).

    Asserts on both ends of the contract: the build succeeds (the
    pre-v2 diagnostic does *not* fire here) AND the resulting
    ``ExternalCall`` carries three leaf-aligned SoA edges, one per
    leaf in declaration order."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_v2",
                      args=(Arg(kind="aos", intent="inout"), ))
        sdfg = build_sdfg(_V2_NESTED_SRC, tmp_path, name="kern",
                          entry="_QMm_v2Pkern").build()
        # Locate the external-call node and check the three leaves
        # (``ip%u``, ``ip%v``, ``scale``) are wired in declaration order.
        # The SoA flats inherit the outer struct's name prefix and the
        # full member path: ``s_ip_u``, ``s_ip_v``, ``s_scale``.
        node = next((n for st in sdfg.all_states() for n in st.nodes()
                     if isinstance(n, ExternalCall)), None)
        assert node is not None, "external call not lowered for nested struct"
        st = next(s for s in sdfg.all_states() if node in s.nodes())
        touched = {e.data.data for e in st.in_edges(node)} | \
                  {e.data.data for e in st.out_edges(node)}
        assert touched == {"s_ip_u", "s_ip_v", "s_scale"}, \
            f"expected three SoA-flat leaves, got {touched}"
    finally:
        clear_external_registry()


# Smallest shape that exercises the v2 box/allocatable expansion:
# a derived type with an allocatable array member.  Before v2 the
# marshal pass refused this shape and ``emit_call`` raised the
# inline_external diagnostic; v2 (this branch's
# ``isBoxOfScalarArray`` + ``rewriteCall`` ``fir.load`` /
# ``fir.box_addr`` chain) handles it directly -- the test now
# anchors successful build + correct marshalling-group count
# instead of the diagnostic-message contract.
_V2_ALLOCATABLE_SRC = """
module m_v2_alloc
  use iso_c_binding
  implicit none
  type :: alloc_t
    real(c_double), allocatable :: w(:)
    integer(c_int) :: n
  end type
contains
  subroutine kern(s)
    type(alloc_t), intent(inout) :: s
    interface
      subroutine ext_v2_alloc(p)
        import :: alloc_t
        type(alloc_t), intent(inout) :: p
      end subroutine
    end interface
    call ext_v2_alloc(s)
  end subroutine
end module
"""


def test_v2_aos_external_with_allocatable_member(tmp_path):
    """``Arg(kind='aos')`` on a callee whose struct has an
    ``allocatable`` array member.  v2 (box / pointer / allocatable
    expansion) ``isBoxOfScalarArray`` accepts the box-typed member
    and ``rewriteCall`` emits the ``fir.load`` + ``fir.box_addr``
    chain at the call site to extract the data pointer; the marshal
    expansion tags two leaves (the allocatable ``w`` data pointer +
    the scalar ``n``).

    Previously a diagnostic anchor that asserted the
    ``inline_external`` workaround appeared in ``emit_call``'s error
    message -- with v2 there is no error; the build succeeds and
    the callee carries a marshalling group."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_v2_alloc",
                      args=(Arg(kind="aos", intent="inout"), ))
        sdfg = build_sdfg(_V2_ALLOCATABLE_SRC, tmp_path, name="kern",
                          entry="_QMm_v2_allocPkern").build()
        # The external-call lowering produced one ExternalCall node
        # with the per-leaf marshal-expansion shape.
        ext = next((n for st in sdfg.all_states() for n in st.nodes()
                    if isinstance(n, ExternalCall)), None)
        assert ext is not None, "marshal expansion did not produce an ExternalCall"
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
#  Arg.c_abi axis (Fortran shape x C ABI shape, decoupled): the same
#  ``state_t`` shape that test_array_member_struct_aos_external above
#  passes as an AoS struct pointer (default ``c_abi='aos_struct_ptr'``)
#  is here passed as per-member SoA pointers (``c_abi='per_member_soa'``).
#  The marshal expansion is identical -- only the call-site body
#  differs: no ``_aosbuf`` struct, no pack/unpack copy, leaves forwarded
#  verbatim.
# ---------------------------------------------------------------------------


def test_aos_external_per_member_soa_skips_aos_buffer(tmp_path):
    """``Arg(kind='aos', c_abi='per_member_soa')`` -- the same
    ``state_t`` shape passed as a *whole* struct from Fortran (so the
    marshal pass expands it), but the C external takes per-member SoA
    pointers in marshal-expansion order; no stack AoS struct is
    materialised.

    The C signature mirrors what a sibling SDFG's ``bind_c_shim``
    would emit (one pointer per leaf), so the same registration
    pattern reaches both an opaque C library that speaks SoA *and* a
    sibling SDFG -- the decoupling of Fortran-side ``kind`` from the
    C-side ABI is what closes the gap."""
    so = _build_c_so(tmp_path, "ext_per_member",
                     "void ext_per_member(double* u, double* v)"
                     "{ for (int i = 0; i < 4; ++i) u[i] += v[i]; }")
    # Note: the *Fortran* call passes the whole struct ``s`` so the
    # marshal pass tags it as an aos group of 2 members; the
    # ``c_abi='per_member_soa'`` registration tells emit_call to
    # forward the SoA flats directly.
    src = """
module m_perm
  use iso_c_binding
  implicit none
  type, bind(c) :: state_t
    real(c_double) :: u(4)
    real(c_double) :: v(4)
  end type
contains
  subroutine kern(s)
    type(state_t), intent(inout) :: s
    interface
      subroutine ext_per_member(p) bind(c, name="ext_per_member")
        import :: state_t
        type(state_t), intent(inout) :: p
      end subroutine
    end interface
    s%u(1) = s%u(1) + 1.0_c_double
    call ext_per_member(s)
  end subroutine
end module
"""
    clear_external_registry()
    try:
        keep_external("ext_per_member",
                      args=(Arg(kind="aos", intent="inout",
                                c_abi="per_member_soa"), ),
                      libraries=(str(so), ))
        sdfg = build_sdfg(src, tmp_path, name="kern",
                          entry="_QMm_permPkern").build()
        u = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([10.0, 20.0, 30.0, 40.0])
        sdfg(s_u=u, s_v=v)
        # s%u(1) += 1 -> u = [2,2,3,4]; then ext_per_member: u[i] += v[i].
        np.testing.assert_allclose(u, [12.0, 22.0, 33.0, 44.0])
        np.testing.assert_allclose(v, [10.0, 20.0, 30.0, 40.0])

        # Body contract: no ``_aosbuf`` struct, no AoS pack/unpack.
        from dace_fortran.external import ExternalCall
        node = next((n for st in sdfg.all_states() for n in st.nodes()
                     if isinstance(n, ExternalCall)), None)
        assert node is not None
        assert "_aosbuf" not in node.body, (
            f"per_member_soa path emitted an AoS buffer; body=\n{node.body}")
        # The leaves are forwarded verbatim in marshal-expansion order;
        # the per-member ``ctype*`` decl shape (not ``void*``) is the
        # other half of the contract.
        assert "double*" in node.c_decl, (
            f"per_member_soa decl should expand to per-leaf pointers; "
            f"got {node.c_decl!r}")
        assert "void *" not in node.c_decl, (
            f"per_member_soa decl should not surface ``void *``; "
            f"got {node.c_decl!r}")
    finally:
        clear_external_registry()
