"""Milestone-1 ABI-alignment contract for the ICON velocity-callback shape: an OUTER kernel
calling an INNER kernel via ``keep_external(c_abi='per_member_soa')`` must produce a
marshalled per-member-SoA leaf sequence that EQUALS the INNER's ``bind_c_shim`` slot
sequence, member-for-member, when both compile against the SAME union struct type. Both
sides walk the struct in Fortran declaration order, so the sequences coincide iff every
member CLASS is handled the same way on both sides -- pinned here for: scalar-symbol
(by-value forward), box-of-scalar-array pass-through (both directions), value-record-array
(one leaf per field), and pointer-to-record handle (skipped on both sides, no leaf).
Build-only, no gfortran link/run."""
import re

import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings.bind_c_shim import emit_bind_c_shim
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, ExternalCall, clear_external_registry, keep_external

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# The shared UNION type text, identical on both sides.  ``t_patch`` carries one
# member of every class the real ICON type mixes.
_UNION_TYPES = """
  type :: tv
    real(c_double) :: v1
    real(c_double) :: v2
  end type
  type :: t_cpat
    integer(c_int) :: n_recv
    integer(c_int), allocatable :: recv_limits(:)
  end type
  type :: t_edges
    real(c_double), allocatable :: solve_only_a(:, :)
    real(c_double), allocatable :: velo_only_b(:, :)
    type(tv), allocatable :: pnc(:, :, :)
  end type
  type :: t_patch
    integer(c_int) :: nblks_e
    type(t_edges) :: edges
    type(t_cpat), pointer :: comm_pat_c
  end type
"""

_INNER_SRC = f"""
module m_align_inner
  use iso_c_binding
  implicit none
{_UNION_TYPES}
contains
  subroutine velo(p, je, jb, out)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: out(p % nblks_e)
    out(1) = p % edges % velo_only_b(je, jb) + p % edges % pnc(je, jb, 1) % v1
  end subroutine
end module
"""

_OUTER_SRC = f"""
module m_align_outer
  use iso_c_binding
  implicit none
{_UNION_TYPES}
contains
  subroutine driver(p, je, jb, acc)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: acc(p % nblks_e)
    interface
      subroutine velo(pp, je, jb, o)
        import :: t_patch, c_int, c_double
        type(t_patch), intent(in) :: pp
        integer(c_int), intent(in) :: je, jb
        real(c_double), intent(out) :: o(*)
      end subroutine
    end interface
    acc(1) = p % edges % solve_only_a(je, jb)
    call velo(p, je, jb, acc)
  end subroutine
end module
"""

# The p_patch member leaves, in declaration order, that BOTH sides must emit
# (the pointer handle ``comm_pat_c`` is deliberately absent -- skipped).
_EXPECTED_PATCH_LEAVES = [
    "nblks_e",
    "edges_solve_only_a",
    "edges_velo_only_b",
    "edges_pnc_v1",
    "edges_pnc_v2",
]


def _outer_patch_leaf_order(tmp_path):
    """The OUTER external call's p_patch leaf sequence, in ABI order. Recovered from the
    ExternalCall body's arg list: a data leaf appears as a connector that memlets
    ``p_<leaf>``; a by-value symbol member appears as ``(int)(p_<leaf>)``. Mapped back to
    bare leaf names, first-appearance order, restricted to ``p_`` leaves."""
    clear_external_registry()
    keep_external("velo",
                  args=(Arg(kind="aos", intent="in",
                            c_abi="per_member_soa"), Arg(kind="scalar", dtype="int32",
                                                         intent="in"), Arg(kind="scalar", dtype="int32", intent="in"),
                        Arg(kind="array", dtype="float64", intent="inout")),
                  dynamic_extents_abi=True)
    try:
        sdfg = build_sdfg(_OUTER_SRC, tmp_path / "outer", name="driver", entry="m_align_outer::driver").build()
        sdfg.validate()
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None, "no ExternalCall lowered"
        st = next(s for s in sdfg.all_states() if node in s.nodes())
        # connector -> the SDFG array it memlets (data leaves).
        conn_to_arr = {}
        for e in st.in_edges(node):
            if e.dst_conn:
                conn_to_arr[e.dst_conn] = e.data.data
        for e in st.out_edges(node):
            if e.src_conn:
                conn_to_arr[e.src_conn] = e.data.data
        order = []
        seen = set()

        def _add(name):
            # name is a p_<leaf> flat name, possibly a dynamic-extent companion p_<leaf>_d<i>
            # -- strip the extent suffix so it maps to its owning leaf (not a distinct leaf).
            if not name.startswith("p_"):
                return
            leaf = re.sub(r"_d\d+$", "", name[len("p_"):])
            if leaf and leaf not in seen:
                seen.add(leaf)
                order.append(leaf)

        # Walk the call body's argument tokens left-to-right.  The call is the
        # last ``velo(...)`` statement; parse its arg list.
        call = re.search(r"\bvelo\(([^;]*)\)\s*;", node.body)
        assert call, f"velocity call not found in body:\n{node.body}"
        for tok in call.group(1).split(","):
            tok = tok.strip()
            m = re.fullmatch(r"\(\w+\)\((\w+)\)", tok)  # by-value member / extent cast
            if m:
                _add(m.group(1))
                continue
            if tok in conn_to_arr:  # data-pointer connector
                _add(conn_to_arr[tok])
        return order
    finally:
        clear_external_registry()


def _inner_patch_slot_order(tmp_path):
    """The INNER bind_c_shim's p_patch slot sequence, in ABI order. A member contributes a
    value/pointer slot named ``p_<leaf>``/``p_<leaf>_p`` (extents ``p_<leaf>_d<i>`` map to
    the same leaf). First-appearance order of ``p_`` leaves."""
    clear_external_registry()
    try:
        sdfg = build_sdfg(_INNER_SRC, tmp_path / "inner", name="velo", entry="m_align_inner::velo").build()
        sdfg.validate()
        iface = build_auto_interface(sdfg._fortran_interface_raw, "velo")
        text = emit_bind_c_shim(iface, str(tmp_path / "velo_c.f90")).read_text()
        sig = re.search(r"subroutine\s+velo_c\(([^)]*)\)", text, re.S)
        args = [a.strip() for a in sig.group(1).replace("&", " ").split(",") if a.strip()]
        order = []
        seen = set()
        for a in args:
            if not a.startswith("p_"):
                continue
            leaf = a[len("p_"):]
            # A dynamic member rides a lower-bound + extent scalar per dim ahead of its
            # pointer (<flat>_lb<i> / <flat>_d<i>); both collapse to the owning leaf (the
            # OUTER's matching offset_<flat>_d<i> token is filtered out above).
            leaf = re.sub(r"_lb\d+$", "", leaf)  # strip lower-bound suffix
            leaf = re.sub(r"_d\d+$", "", leaf)  # strip extent suffix
            leaf = re.sub(r"_p$", "", leaf)  # strip pointer suffix
            if leaf and leaf not in seen:
                seen.add(leaf)
                order.append(leaf)
        return order
    finally:
        clear_external_registry()


def test_outer_marshal_leaf_order_equals_inner_shim_slot_order(tmp_path):
    """OUTER per-member-SoA marshalled p_patch leaf sequence EQUALS INNER bind_c_shim slot
    sequence, member-for-member: scalar-symbol (by value), box-of-scalar-array pass-through
    (both directions), value-record-array (per field), skipped pointer-to-record handle."""
    outer = _outer_patch_leaf_order(tmp_path)
    inner = _inner_patch_slot_order(tmp_path)
    assert outer == _EXPECTED_PATCH_LEAVES, \
        f"OUTER p_patch leaf order drifted:\n  got      {outer}\n  expected {_EXPECTED_PATCH_LEAVES}"
    assert inner == _EXPECTED_PATCH_LEAVES, \
        f"INNER shim slot order drifted:\n  got      {inner}\n  expected {_EXPECTED_PATCH_LEAVES}"
    assert outer == inner, f"outer/inner ABI desync:\n  outer {outer}\n  inner {inner}"
    # The pointer-to-record handle is absent from BOTH (skipped on both sides).
    assert not any("comm_pat" in leaf for leaf in outer + inner), \
        "pointer-to-record handle leaked into a leaf sequence"


# ---------------------------------------------------------------------------
# Regression: concrete-NEGATIVE folded _lb slot for a callback member
# (commit 9bf289a -- ICON end_block(-10) refinement-control pattern).
# ---------------------------------------------------------------------------
# A deferred-shape ALLOCATABLE member accessed at a literal negative index has its lower
# bound STATICALLY INFERRED to that literal (inferLowerBoundsFromLiteralAccesses), folding
# offset_<member>_d0 to -10. The inner shim still mints one <flat>_lb<i> slot per dim
# regardless, so the marshal MUST forward the folded literal -- else the outer emits one
# fewer slot than the inner reads (925 vs 921 desync). Build + string-inspect, no run.

_NEG_LB_TYPES = """
  type :: t_edges
    integer(c_int), allocatable :: end_block(:, :)
  end type
  type :: t_patch
    integer(c_int) :: nblks_e
    type(t_edges) :: edges
  end type
"""

_NEG_LB_INNER = f"""
module m_neglb_inner
  use iso_c_binding
  implicit none
{_NEG_LB_TYPES}
contains
  subroutine velo(p, je, jb, out)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: out(4)
    out(1) = real(p % edges % end_block(-10, jb), c_double) + real(p % nblks_e, c_double)
  end subroutine
end module
"""

_NEG_LB_OUTER = f"""
module m_neglb_outer
  use iso_c_binding
  implicit none
{_NEG_LB_TYPES}
contains
  subroutine driver(p, je, jb, acc)
    type(t_patch), intent(in) :: p
    integer(c_int), intent(in) :: je, jb
    real(c_double), intent(out) :: acc(4)
    interface
      subroutine velo(pp, je, jb, o)
        import :: t_patch, c_int, c_double
        type(t_patch), intent(in) :: pp
        integer(c_int), intent(in) :: je, jb
        real(c_double), intent(out) :: o(*)
      end subroutine
    end interface
    ! The OUTER's own literal-negative access folds end_block's dim-1 lower
    ! bound to -10 in the outer builder, so the marshal sees offset<1.
    acc(1) = real(p % edges % end_block(-10, jb), c_double) + real(p % nblks_e, c_double)
    call velo(p, je, jb, acc)
  end subroutine
end module
"""

_NEG_LB_CALLBACK_ARGS = (
    Arg(kind="aos", intent="in", c_abi="per_member_soa"),  # p
    Arg(kind="scalar", dtype="int32", intent="in"),  # je
    Arg(kind="scalar", dtype="int32", intent="in"),  # jb
    Arg(kind="array", dtype="float64", intent="inout"),  # out / acc
)


def _neg_lb_inner_shim_slots(tmp_path):
    """The INNER velo ``bind_c_shim`` header slot list, in order."""
    clear_external_registry()
    try:
        builder = build_sdfg(_NEG_LB_INNER, tmp_path / "in", name="velo", entry="m_neglb_inner::velo")
        builder.build()
        iface = build_auto_interface(builder._fortran_interface_raw, "velo")
        shim = emit_bind_c_shim(iface, str(tmp_path / "velo_c.f90")).read_text()
        hdr = re.search(r"subroutine\s+velo_c\s*\(([^)]*)\)", shim, re.S)
        return [a.strip() for a in hdr.group(1).replace("&", " ").split(",") if a.strip()]
    finally:
        clear_external_registry()


def _neg_lb_outer_marshal_args(tmp_path):
    """The OUTER driver's marshalled ``velo(...)`` ExternalCall argument list."""
    clear_external_registry()
    keep_external("velo", args=_NEG_LB_CALLBACK_ARGS, dynamic_extents_abi=True)
    try:
        sdfg = build_sdfg(_NEG_LB_OUTER, tmp_path / "out", name="driver", entry="m_neglb_outer::driver").build()
        sdfg.validate()
        node = next((n for st in sdfg.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None, "outer driver SDFG emitted no velo ExternalCall"
        call = re.search(r"\bvelo\(([^;]*)\)\s*;", node.body)
        assert call, f"velo call not found in body:\n{node.body}"
        return [a.strip() for a in call.group(1).split(",")]
    finally:
        clear_external_registry()


def test_neg_lbound_member_lb_slot_marshalled_with_folded_literal(tmp_path):
    """A neg-folded member lower bound is marshalled as its ``_lb`` slot. ``end_block(-10,
    jb)`` folds ``offset_p_edges_end_block_d0`` to literal ``-10`` (commit 9bf289a). Two
    facets: LITERAL -- the marshal forwards the folded lb as ``(int)(-10)`` (pre-fix skipped
    non-free offsets); COUNT PARITY -- with it forwarded, outer arg count == inner slot
    count (a dropped ``_lb`` reproduces the capstone's 925-vs-921 desync at toy scale)."""
    inner_slots = _neg_lb_inner_shim_slots(tmp_path)
    outer_args = _neg_lb_outer_marshal_args(tmp_path)

    # Precondition: inner shim mints a dim-0 _lb slot for the dynamic end_block member.
    assert "p_edges_end_block_lb0" in inner_slots, \
        f"inner shim did not mint end_block dim-0 lb slot:\n{inner_slots}"

    # LITERAL: the folded -10 lower bound rides as the member's dim-0 _lb slot.
    assert "(int)(-10)" in outer_args, \
        f"neg-folded end_block lower bound not marshalled as an _lb slot:\n{outer_args}"

    # COUNT PARITY: outer marshal lines up slot-for-slot with the inner shim.
    assert len(outer_args) == len(inner_slots), (
        f"neg-lbound _lb slot desync: inner shim {len(inner_slots)} slots vs "
        f"outer marshal {len(outer_args)} args\n  inner={inner_slots}\n  outer={outer_args}")
