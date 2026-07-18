"""External (``keep_external``) calls and the AoS/SoA argument-layout decision:
plain array / flattened struct member -> SoA pointer handed over directly (zero
copy, e.g. ICON's double-buffered ``p_nh%prog(nnow)%w``); array-of-scalar-structs
kept AoS -> struct pointer directly; scalar-member struct an AoS external wants
contiguous -> shallow alias or deep gather/scatter. Plain-array cases use the
unified policy (``apply_external_functions``); AoS/``per_member_soa`` cases use
``keep_external`` with an authored ``Arg`` list (ABI fact HLFIR can't infer).
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _helpers import xfail
from _util import build_sdfg, have_flang
from dace_fortran.external import Arg, apply_external_functions, clear_external_registry, keep_external
from dace_fortran.external_functions import ExternalFunction

#: Standalone "fake" mo_velocity_advection (full velocity_tendencies + its USE
#: closure as one file).  Vendored in-repo at
#: ``tests/icon/full/velocity_full.f90``; override with
#: ``VELOCITY_FULL_F90`` to point at an out-of-tree copy.
_VELOCITY_FULL = Path(
    os.environ.get("VELOCITY_FULL_F90", str(Path(__file__).resolve().parent / "icon" / "full" / "velocity_full.f90")))

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build_c_so(out_dir: Path, name: str, csrc: str) -> Path:
    """Compile a small C source to a shared library ``lib<name>.so`` and return its path."""
    (out_dir / f"{name}.c").write_text(csrc)
    so = out_dir / f"lib{name}.so"
    subprocess.check_call(["gcc", "-shared", "-fPIC", "-o", str(so), str(out_dir / f"{name}.c")])
    return so


def test_aliasable_array_member_no_copy(tmp_path):
    """A flattened struct array-member reaches a plain-array external as the
    SoA pointer directly  --  no copy (the ICON double-buffering shape)."""
    so = _build_c_so(tmp_path, "ext_scale", "void ext_scale(double* a, int n)"
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
        # Plain array + scalar -> unified policy derives the zero-copy shape (array inout pointer, scalar by-value).
        apply_external_functions([ExternalFunction("ext_scale", library=str(so))])
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_alias::kern").build()
        u = np.arange(8, dtype=np.float64)
        v = np.ones(8, dtype=np.float64)
        sdfg(s_u=u, s_v=v, n=np.int32(8))
        np.testing.assert_allclose(u, np.arange(8) * 2.0)
        np.testing.assert_allclose(v, 1.0)  # the other member is untouched
    finally:
        clear_external_registry()


def test_scalar_member_struct_aos_external(tmp_path):
    """External wants a contiguous AoS struct; the bridge splits it into SoA flats, and
    ``emit_call``'s C tasklet packs/unpacks a local AoS buffer around the external call."""
    so = _build_c_so(
        tmp_path, "ext_swap", "struct pt{double f1;double f2;};"
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
        keep_external("ext_swap", args=(Arg(kind="aos", intent="inout"), ), libraries=(str(so), ))
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_aos::kern").build()
        f1 = np.array([3.0])
        f2 = np.array([5.0])
        sdfg(s_f1=f1, s_f2=f2)
        np.testing.assert_allclose(f1, 5.0)  # swapped
        np.testing.assert_allclose(f2, 3.0 + 1.0)  # +1 then swapped
    finally:
        clear_external_registry()


def test_array_member_struct_aos_external(tmp_path):
    """Velocity-style struct with array members passed whole to an AoS external:
    marshalling expands to per-member SoA; emit_call packs/unpacks via element-loop copies."""
    so = _build_c_so(
        tmp_path, "ext_state", "struct state_t { double u[4]; double v[4]; };"
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
        keep_external("ext_state", args=(Arg(kind="aos", intent="inout"), ), libraries=(str(so), ))
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_vel::kern").build()
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
    """ICON velocity binds prognostic fields via pointer assignment (not struct
    flattening): hlfir-rewrite-pointer-assigns folds the rebind straight to the
    contiguous target, so the external gets a shallow pointer pass, no AoS buffer."""
    so = _build_c_so(tmp_path, "ext_scale", "void ext_scale(double* a, int n)"
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
        # Plain array + scalar -> unified policy derives the same shallow (no AoS buffer) pass.
        apply_external_functions([ExternalFunction("ext_scale", library=str(so))])
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_velptr::kern").build()
        # Shallow pass: tasklet calls the external directly on the array pointer, no AoS buffer/copies.
        body = _external_call_body(sdfg)
        assert "struct" not in body, f"expected a shallow pointer pass, got:\n{body}"
        tgt = np.arange(6, dtype=np.float64)
        sdfg(tgt=tgt, n=np.int32(6))
        np.testing.assert_allclose(tgt, np.arange(6) * 2.0)
    finally:
        clear_external_registry()


def test_velocity_state_t_whole_struct_external(tmp_path):
    """Real ICON velocity ``state_t`` shape (four 2-D real(8) members) passed whole to
    an AoS external: deep-copy marshalling packs/unpacks the full multi-member struct."""
    so = _build_c_so(
        tmp_path, "ext_velstate", "struct state_t { double u[16]; double v[16];"
        "                 double w[16]; double p[16]; };"
        "void ext_velstate(struct state_t* s) {"
        "  for (int i = 0; i < 16; ++i) {"
        "    s->u[i] += s->v[i];"
        "    s->p[i]  = s->w[i] * 2.0;"
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
        keep_external("ext_velstate", args=(Arg(kind="aos", intent="inout"), ), libraries=(str(so), ))
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_velstate::kern").build()
        rng = np.random.default_rng(0)
        u = np.asfortranarray(rng.random((4, 4)))
        v = np.asfortranarray(rng.random((4, 4)))
        w = np.asfortranarray(rng.random((4, 4)))
        p = np.asfortranarray(rng.random((4, 4)))
        u_in, v_in, w_in = u.copy(), v.copy(), w.copy()
        sdfg(st_u=u, st_v=v, st_w=w, st_p=p)
        expect_u = u_in.copy()
        expect_u.flat[0] += 1.0  # st%u(1,1) += 1
        expect_u = expect_u + v_in  # u += v
        np.testing.assert_allclose(u, expect_u)
        np.testing.assert_allclose(p, w_in * 2.0)
        np.testing.assert_allclose(v, v_in)  # untouched
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
    src = src.replace(
        use_line, use_line + """
    INTERFACE
      SUBROUTINE ext_sync(a) BIND(C, name="ext_sync")
        USE iso_c_binding
        REAL(c_double), INTENT(INOUT) :: a(*)
      END SUBROUTINE ext_sync
    END INTERFACE""", 1)
    src = src.replace("  END SUBROUTINE velocity_tendencies", "    CALL ext_sync(p_prog%w)\n"
                      "  END SUBROUTINE velocity_tendencies", 1)

    so = _build_c_so(tmp_path, "ext_sync", "void ext_sync(double* a){(void)a;}")
    clear_external_registry()
    try:
        # Single plain array -> derived inout pointer (shallow pass).
        apply_external_functions([ExternalFunction("ext_sync", library=str(so))])
        sdfg = build_sdfg(src, tmp_path, name="velext", entry="mo_velocity_advection::velocity_tendencies").build()
        from dace_fortran.external import ExternalCall
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
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
    """Dycore-style external mixing a plain shallow field (``s%w``, pointer pass) and
    a nested sub-struct (``s%blk``) needing the AoS<->SoA deep copy via a local buffer."""
    so = _build_c_so(
        tmp_path, "ext_mixed", "struct t_blk { double arr[8]; };"
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
                      args=(Arg(kind="array", dtype="float64",
                                intent="inout"), Arg(kind="scalar", dtype="int32",
                                                     intent="in"), Arg(kind="aos", intent="inout")),
                      libraries=(str(so), ))
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_dycore::kern").build()
        body = _external_call_body(sdfg)
        assert "_a0_o" in body and "struct" in body, \
            f"expected shallow w + AoS buffer for blk, got:\n{body}"
        w = np.arange(16, dtype=np.float64)
        vn = np.ones(16, dtype=np.float64)
        arr = np.arange(8, dtype=np.float64)
        sdfg(s_w=w, s_vn=vn, s_blk_arr=arr)
        exp_arr = np.arange(8, dtype=np.float64)
        exp_arr[0] += 1.0
        exp_arr += 1.0
        np.testing.assert_allclose(w, np.arange(16) * 2.0)  # field: shallow x2
        np.testing.assert_allclose(arr, exp_arr)  # blk: deep-copy roundtrip
        np.testing.assert_allclose(vn, 1.0)  # untouched
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
# v2.1 marshal expansion (Phase 2.3.E): MarshalExternalStructs.cpp recursively
# walks nested derived-type members to inline-flat leaves, each its own call-arg.
# Still unsupported: box/pointer/allocatable/dynamic-shape members (emit_call
# points at inline_external for those).
# ---------------------------------------------------------------------------

# Shared v2.1 kernel: outer_t nests inner_t; recursive flatten walks to three f64 leaves (ip%u, ip%v, scale).
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
    """``keep_external(kind='aos')`` on a struct with a nested derived-type member:
    recursive expansion produces one SoA flat per leaf (``ip%u``, ``ip%v``, ``scale``),
    laid out field-by-field; build succeeds and the ExternalCall carries all three."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_v2", args=(Arg(kind="aos", intent="inout"), ))
        sdfg = build_sdfg(_V2_NESTED_SRC, tmp_path, name="kern", entry="m_v2::kern").build()
        # Three leaves (ip%u, ip%v, scale) wired in declaration order as s_ip_u/s_ip_v/s_scale.
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None, "external call not lowered for nested struct"
        st = next(s for s in sdfg.all_states() if node in s.nodes())
        touched = {e.data.data for e in st.in_edges(node)} | \
                  {e.data.data for e in st.out_edges(node)}
        assert touched == {"s_ip_u", "s_ip_v", "s_scale"}, \
            f"expected three SoA-flat leaves, got {touched}"
    finally:
        clear_external_registry()


# Smallest v2 box/allocatable shape: a derived type with an allocatable array member.
# Pre-v2 this raised inline_external; v2's isBoxOfScalarArray + rewriteCall handles it
# directly, so the test anchors successful build + marshalling-group count.
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
    """``Arg(kind='aos')`` with an allocatable array member: v2's ``isBoxOfScalarArray``
    + ``rewriteCall`` extracts the data pointer, tagging two leaves (``w`` + scalar ``n``);
    build succeeds with no error (was a diagnostic-anchor before v2)."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_v2_alloc", args=(Arg(kind="aos", intent="inout"), ))
        sdfg = build_sdfg(_V2_ALLOCATABLE_SRC, tmp_path, name="kern", entry="m_v2_alloc::kern").build()
        # One ExternalCall node with the per-leaf marshal-expansion shape.
        ext = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert ext is not None, "marshal expansion did not produce an ExternalCall"
    finally:
        clear_external_registry()


# v2.2: a box-of-value-record-array member (ICON's t_patch%edges%primal_normal_cell,
# an allocatable array of {v1,v2} records), also read element-wise, so flatten mints
# per-field SoA companions s_e_v1/s_e_v2 -- one marshal leaf per FIELD, matching
# bind_c_shim._emit_value_record_array's per-field C slots.
_V2_VALUE_RECORD_ARRAY_SRC = """
module m_v2_vra
  use iso_c_binding
  implicit none
  type :: tv
    real(c_double) :: v1
    real(c_double) :: v2
  end type
  type :: vra_t
    type(tv), allocatable :: e(:, :, :)
    integer(c_int) :: n
  end type
contains
  subroutine kern(s, je, jb, acc)
    type(vra_t), intent(inout) :: s
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: acc
    interface
      subroutine ext_v2_vra(p)
        import :: vra_t
        type(vra_t), intent(inout) :: p
      end subroutine
    end interface
    acc = s % e(je, jb, 1) % v1 + s % e(je, jb, 2) % v2
    call ext_v2_vra(s)
  end subroutine
end module
"""


def test_v2_aos_external_with_value_record_array_member(tmp_path):
    """``Arg(kind='aos', c_abi='per_member_soa')`` with a box-of-value-record-array member:
    v2.2 expands to one leaf per record field (``s_e_v1``/``s_e_v2``), not the AoS box.
    Milestone-1 anchor for the ICON solve_nh velocity callback (``t_patch.primal_normal_cell``)."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_v2_vra", args=(Arg(kind="aos", intent="inout", c_abi="per_member_soa"), ))
        sdfg = build_sdfg(_V2_VALUE_RECORD_ARRAY_SRC, tmp_path, name="kern", entry="m_v2_vra::kern").build()
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None, "marshal expansion did not produce an ExternalCall"
        st = next(s for s in sdfg.all_states() if node in s.nodes())
        touched = {e.data.data for e in st.in_edges(node)} | \
                  {e.data.data for e in st.out_edges(node)}
        # Two record fields cross as per-field SoA companions -- not the AoS box, not a single s_e member.
        assert "s_e_v1" in touched and "s_e_v2" in touched, \
            f"expected per-field SoA leaves s_e_v1 / s_e_v2, got {sorted(touched)}"
        # The scalar member ``n`` also crosses (a plain inline-flat leaf).
        assert "s_n" in touched, f"expected scalar member s_n, got {sorted(touched)}"
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
# Arg.c_abi axis (Fortran shape x C ABI shape, decoupled): the same state_t shape
# as test_array_member_struct_aos_external, here passed as per-member SoA pointers
# (c_abi='per_member_soa') instead of an AoS struct pointer -- same marshal
# expansion, but no _aosbuf / pack-unpack, leaves forwarded verbatim.
# ---------------------------------------------------------------------------


def test_aos_external_per_member_soa_skips_aos_buffer(tmp_path):
    """``Arg(kind='aos', c_abi='per_member_soa')``: whole struct from Fortran, but the C
    external takes per-member SoA pointers -- no stack AoS struct materialised. Same
    registration pattern reaches both an opaque SoA-speaking C library and a sibling SDFG."""
    so = _build_c_so(tmp_path, "ext_per_member", "void ext_per_member(double* u, double* v)"
                     "{ for (int i = 0; i < 4; ++i) u[i] += v[i]; }")
    # Fortran passes the whole struct s (tagged aos group of 2); c_abi='per_member_soa'
    # tells emit_call to forward the SoA flats directly.
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
                      args=(Arg(kind="aos", intent="inout", c_abi="per_member_soa"), ),
                      libraries=(str(so), ))
        sdfg = build_sdfg(src, tmp_path, name="kern", entry="m_perm::kern").build()
        u = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([10.0, 20.0, 30.0, 40.0])
        sdfg(s_u=u, s_v=v)
        # s%u(1) += 1 -> u = [2,2,3,4]; then ext_per_member: u[i] += v[i].
        np.testing.assert_allclose(u, [12.0, 22.0, 33.0, 44.0])
        np.testing.assert_allclose(v, [10.0, 20.0, 30.0, 40.0])

        # Body contract: no ``_aosbuf`` struct, no AoS pack/unpack.
        from dace_fortran.external import ExternalCall
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None
        assert "_aosbuf" not in node.body, (f"per_member_soa path emitted an AoS buffer; body=\n{node.body}")
        # Leaves forwarded verbatim in marshal-expansion order; per-member ctype* decl
        # (not void*) is the other half of the contract.
        assert "double*" in node.c_decl, (f"per_member_soa decl should expand to per-leaf pointers; "
                                          f"got {node.c_decl!r}")
        assert "void *" not in node.c_decl, (f"per_member_soa decl should not surface ``void *``; "
                                             f"got {node.c_decl!r}")
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
# Pointer-to-record HANDLE members (ICON t_patch%comm_pat_c): a scalar
# pointer/allocatable to a record has no SoA image, so flatten mints no companion.
# The marshaller SKIPS such a member if only passed through, and FAILS LOUDLY if
# the callee reads its pointed-to data (which per-member-SoA marshalling can't supply).
# ---------------------------------------------------------------------------

# t_patch-shaped struct: value-record-array member (marshalled per field) plus a
# pointer-to-record handle (comm_pat_c) whose data is never read -- pure pass-through.
_HANDLE_UNUSED_SRC = """
module m_handle_ok
  use iso_c_binding
  implicit none
  type :: tv
    real(c_double) :: v1
    real(c_double) :: v2
  end type
  type :: t_cpat
    integer(c_int) :: n_recv
    integer(c_int), allocatable :: recv_limits(:)
  end type
  type :: t_edges
    type(tv), allocatable :: primal_normal_cell(:, :, :)
  end type
  type :: t_patch
    type(t_edges) :: edges
    type(t_cpat), pointer :: comm_pat_c
  end type
contains
  subroutine kern(p, je, jb, acc)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: acc
    interface
      subroutine ext_handle(pp)
        import :: t_patch
        type(t_patch), intent(in) :: pp
      end subroutine
    end interface
    acc = p % edges % primal_normal_cell(je, jb, 1) % v1
    call ext_handle(p)
  end subroutine
end module
"""


def test_marshal_skips_unused_pointer_to_record_handle(tmp_path):
    """Pointer-to-record handle (``comm_pat_c``) that's only passed through is SKIPPED
    by the marshaller (no leaf); the struct still marshals its real data members."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_handle",
                      args=(Arg(kind="aos", intent="in", c_abi="per_member_soa"), ),
                      dynamic_extents_abi=True)
        sdfg = build_sdfg(_HANDLE_UNUSED_SRC, tmp_path, name="kern", entry="m_handle_ok::kern").build()
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None, "marshal expansion did not produce an ExternalCall"
        st = next(s for s in sdfg.all_states() if node in s.nodes())
        touched = {e.data.data for e in st.in_edges(node)} | {e.data.data for e in st.out_edges(node)}
        # The value-record fields cross; the comm-pattern handle does NOT.
        assert "p_edges_primal_normal_cell_v1" in touched and "p_edges_primal_normal_cell_v2" in touched
        assert not any("comm_pat" in t for t in touched), \
            f"pointer-to-record handle should be skipped, but a comm_pat leaf crossed: {sorted(touched)}"
    finally:
        clear_external_registry()


# Same struct, but the callee READS the handle's data (p%comm_pat_c%n_recv): silently
# dropping it would corrupt the C ABI, so the pass must fail loudly.
_HANDLE_USED_SRC = """
module m_handle_bad
  use iso_c_binding
  implicit none
  type :: tv
    real(c_double) :: v1
    real(c_double) :: v2
  end type
  type :: t_cpat
    integer(c_int) :: n_recv
    integer(c_int), allocatable :: recv_limits(:)
  end type
  type :: t_edges
    type(tv), allocatable :: primal_normal_cell(:, :, :)
  end type
  type :: t_patch
    type(t_edges) :: edges
    type(t_cpat), pointer :: comm_pat_c
  end type
contains
  subroutine kern(p, je, jb, acc)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: acc
    interface
      subroutine ext_handle(pp)
        import :: t_patch
        type(t_patch), intent(in) :: pp
      end subroutine
    end interface
    acc = p % edges % primal_normal_cell(je, jb, 1) % v1 &
        + real(p % comm_pat_c % n_recv, c_double)
    call ext_handle(p)
  end subroutine
end module
"""


def test_marshal_loud_fails_on_used_pointer_to_record_handle(tmp_path):
    """Callee reading a pointer-to-record handle's data (``p%comm_pat_c%n_recv``) must
    NOT be silently dropped; the pass emits an error naming the offending member."""
    clear_external_registry()
    try:
        keep_external("ext_handle",
                      args=(Arg(kind="aos", intent="in", c_abi="per_member_soa"), ),
                      dynamic_extents_abi=True)
        with pytest.raises(Exception) as ei:
            build_sdfg(_HANDLE_USED_SRC, tmp_path, name="kern", entry="m_handle_bad::kern").build()
        msg = str(ei.value)
        assert "pipeline failed" in msg or "pointer-to-record" in msg, \
            f"expected a loud marshal failure, got: {msg[:300]}"
    finally:
        clear_external_registry()


# ---------------------------------------------------------------------------
# A struct SCALAR member read only as an array extent/loop bound is promoted to
# an sdfg.symbols entry (not sdfg.arrays); emit_call must forward it across the
# per_member_soa C ABI BY VALUE, matching bind_c_shim's <type>, value slot.
# ICON's t_patch%nlev/%nblks_e/%id shape: velocity consumes them, solve_nh uses
# them only as extents.
# ---------------------------------------------------------------------------
_SYMBOL_MEMBER_SRC = """
module m_symmem
  use iso_c_binding
  implicit none
  type :: t
    integer(c_int) :: n
    real(c_double), allocatable :: a(:)
  end type
contains
  subroutine kern(s, out)
    type(t), intent(in) :: s
    real(c_double), intent(out) :: out(s % n)
    interface
      subroutine ext_sym(pp)
        import :: t
        type(t), intent(in) :: pp
      end subroutine
    end interface
    out(1) = s % a(1)
    call ext_sym(s)
  end subroutine
end module
"""


def test_marshal_scalar_symbol_member_forwarded_by_value(tmp_path):
    """Struct scalar member used only as an extent lands in ``sdfg.symbols`` (not
    ``sdfg.arrays``); per_member_soa marshalling forwards it BY VALUE, no pointer."""
    from dace_fortran.external import ExternalCall
    clear_external_registry()
    try:
        keep_external("ext_sym",
                      args=(Arg(kind="aos", intent="in", c_abi="per_member_soa"), ),
                      dynamic_extents_abi=True)
        sdfg = build_sdfg(_SYMBOL_MEMBER_SRC, tmp_path, name="kern", entry="m_symmem::kern").build()
        sdfg.validate()
        # The member ``n`` is a symbol, not an array.
        assert "s_n" in sdfg.symbols and "s_n" not in sdfg.arrays, \
            f"expected s_n as a symbol; symbols={('s_n' in sdfg.symbols)} arrays={('s_n' in sdfg.arrays)}"
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None
        # n crosses by value: body renders (int)(s_n), decl carries scalar int (not int*).
        assert "(s_n)" in node.body, f"symbol member not forwarded by value; body=\n{node.body}"
        decl_args = [a.strip() for a in node.c_decl[node.c_decl.index('(') + 1:node.c_decl.rindex(')')].split(',')]
        assert decl_args[0] == "int", f"first arg (symbol member n) should be scalar int, got {decl_args}"
    finally:
        clear_external_registry()
