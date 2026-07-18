"""Verify the ``hlfir-flatten-structs`` pass rewrites derived-type data into
flat per-member companions (uniform case) or into a single ELLPACK-style
combined array (jagged case) before SDFG generation sees it."""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang, run_passes_dump

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not available")

_HERE = Path(__file__).resolve().parent
_SRC = (_HERE / "complex_struct.f90").read_text()
_VELOCITY_SRC = (_HERE / "velocity_struct.f90").read_text()
_JAGGED_SRC = (_HERE / "jagged_struct.f90").read_text()
#: PRIVATE components: flang spells a bare-PRIVATE member as
#: ``_QM<module>T<type>.<base>`` (a '.', illegal in Fortran identifiers, is what
#: the flatten gate keys on). ``t_hidden`` exercises both gate halves:
#:   never_used -- private AND never designated -> pruned (ICON's t_patch %
#:                 comm_pat_gather_c shape).
#:   seen       -- private BUT designated in-module -> kept (an in-module
#:                 procedure may read its own type's privates).
_PRIVATE_COMPONENT_SRC = """
module m_private_components
  implicit none
  type :: t_hidden
    private
    integer, allocatable :: never_used(:)
    integer :: seen(4)
  end type t_hidden
  type :: t_outer
    type(t_hidden) :: book
    real(8) :: data(8)
  end type t_outer
contains
  subroutine private_component_kernel(o, res)
    type(t_outer), intent(in) :: o
    real(8), intent(out) :: res(8)
    integer :: i
    do i = 1, 8
      res(i) = o%data(i) + real(o%book%seen(1), 8)
    end do
  end subroutine private_component_kernel
end module m_private_components
"""
_FLATTEN_ONLY = "hlfir-flatten-structs"


def _names(builder):
    return set(builder.arrays) | set(builder.scalars)


def test_flatten_structs_splits_members(tmp_path):
    """Array-of-struct ``z(8)`` of ``complex_t``: pass synthesises flat companions z_re/z_im."""
    b = build_sdfg(_SRC, tmp_path, name="complex_struct")

    names = _names(b)
    assert any(n.endswith("_re") for n in names), (f"missing re companion array in {sorted(names)}")
    assert any(n.endswith("_im") for n in names), (f"missing im companion array in {sorted(names)}")


# ----------------------------------------------------------------------------
# Struct-typed dummy: four uniform 2-D members -> four 2-D args; function renamed.
# ----------------------------------------------------------------------------


def test_velocity_struct_arg_flattens_to_four_args(tmp_path):
    ir = run_passes_dump(_VELOCITY_SRC, tmp_path, name="velocity", pipeline=_FLATTEN_ONLY)

    assert "_soa" in ir, f"function should be renamed to *_soa:\n{ir[:600]}"

    # member companions appear as hlfir.declare uniq_names
    for mem in ("_u", "_v", "_w", "_p"):
        needle = f"Est{mem}"
        assert needle in ir, (f"expected declare with uniq_name ending in {needle!r}; IR excerpt:"
                              f"\n{ir[:800]}")

    assert "!fir.type<" not in ir.split("func.func")[1].splitlines()[0], (
        f"function signature should no longer reference a struct type:\n{ir[:400]}")


# ----------------------------------------------------------------------------
# Jagged struct: four differently-sized 1-D members pack into one [4 x max(extents)] ELLPACK companion.
# ----------------------------------------------------------------------------


def test_jagged_struct_arg_packs_into_2d(tmp_path):
    ir = run_passes_dump(_JAGGED_SRC, tmp_path, name="jagged", pipeline=_FLATTEN_ONLY)

    assert "_soa" in ir, f"function should be renamed to *_soa:\n{ir[:600]}"

    # The combined 2-D array: 4 rows (one per member), 20 cols (= max(10,20,15,5)).
    assert "!fir.array<4x20xf64>" in ir, (f"expected packed 4x20xf64 array in post-pass IR:\n{ir[:1500]}")

    # four coordinate_of/convert pairs alias each member into a row of the combined array
    assert ir.count("fir.coordinate_of") >= 4, (
        f"expected four fir.coordinate_of ops (one per jagged member):\n{ir[:1500]}")


# ----------------------------------------------------------------------------
# Negative case: a POINTER/ALLOCATABLE struct LOCAL is a rebindable runtime
# descriptor, not owned storage -- hlfir-flatten-structs must NOT split it into
# per-member companions. Regression for a verifier crash where splitLocal
# synthesised a companion with declare-result type ref<box<ptr<i32>>> against a
# plain ref<i32> alloca ("first result type is inconsistent with variable
# properties"). Mirrors ICON-O coriolis' ``verts_in_domain => patch%verts%in_domain``.
# Asserted at the flatten-structs boundary: OWNED locals still split; descriptor
# locals stay intact for the pointer-rewrite/view path.
# ----------------------------------------------------------------------------

_FLATTEN_ONLY = "hlfir-flatten-structs"


def test_pointer_scalar_struct_local_not_flattened(tmp_path):
    src = """
module lib
  implicit none
  type subset
    integer :: start_index
    integer :: end_index
  end type subset
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, intent(out) :: out
  type(subset), target :: s
  type(subset), pointer :: p
  s%start_index = 3
  s%end_index = 9
  p => s
  out = p%end_index - p%start_index
end subroutine main
"""
    # throws "inconsistent with variable properties" before the fix
    ir = run_passes_dump(src, tmp_path, name="main", pipeline=_FLATTEN_ONLY)

    assert "Es_start_index" in ir and "Es_end_index" in ir, (
        f"owned struct local should still flatten into companions:\n{ir[:1200]}")
    # descriptor local p stays intact -- pointer declare survives for the downstream pointer-rewrite path
    assert "Ep_start_index" not in ir and "Ep_end_index" not in ir, (
        f"POINTER struct local must NOT be locally flattened:\n{ir[:1200]}")
    assert '"_QFmainEp"' in ir, f"pointer local declare should survive:\n{ir[:1200]}"


def test_allocatable_scalar_struct_local_not_flattened(tmp_path):
    src = """
module lib
  implicit none
  type subset
    integer :: lo
    integer :: hi
  end type subset
end module lib

subroutine main(out)
  use lib
  implicit none
  integer, intent(out) :: out
  type(subset), allocatable :: a
  type(subset), target :: s
  s%lo = 4
  s%hi = 11
  allocate(a)
  a%lo = s%lo
  a%hi = s%hi
  out = a%hi - a%lo
  deallocate(a)
end subroutine main
"""
    ir = run_passes_dump(src, tmp_path, name="main", pipeline=_FLATTEN_ONLY)
    # OWNED ``s`` flattens; ALLOCATABLE descriptor ``a`` is left intact.
    assert "Es_lo" in ir and "Es_hi" in ir, (f"owned struct local should still flatten:\n{ir[:1200]}")
    assert "Ea_lo" not in ir and "Ea_hi" not in ir, (
        f"ALLOCATABLE struct local must NOT be locally flattened:\n{ir[:1200]}")


# ----------------------------------------------------------------------------
# Multi-dim AoS with a static-ARRAY leaf (ICON-O ``t_cartesian_coordinates::x(3)``),
# accessed WHOLE through an inlined callee, must flatten to SoA ``field_x(i,j,k,3)``
# -- not leak as an unregistered libcall arg (coriolis' ``p_vn_dual_x`` KeyError).
# Exercises: (a) AoS gate on both inlined-callee alias declare results; (b) AoS
# concat designate-rewrite preserving element indices (else the libcall gets a
# WHOLE-array memlet and fails "dot product only supported on 1-dimensional arrays").
# ----------------------------------------------------------------------------

_AOS_ARR_MEMBER_SRC = """
module lib
  implicit none
  type cc
    real(8) :: x(3)
  end type cc
contains
  subroutine inner(field, out)
    type(cc), intent(in) :: field(4, 2, 5)
    real(8), intent(out) :: out
    out = dot_product(field(2, 1, 3) % x, field(2, 1, 3) % x)
  end subroutine inner

  subroutine driver(field, out)
    type(cc), intent(in) :: field(4, 2, 5)
    real(8), intent(out) :: out
    call inner(field, out)
  end subroutine driver
end module lib
"""


def test_inlined_multidim_aos_array_member_flattens_to_soa(tmp_path):
    sdfg = build_sdfg(_AOS_ARR_MEMBER_SRC, tmp_path / "sdfg", name="driver", entry="lib::driver").build()
    # Companion is the AoS dims followed by the member extent: (4, 2, 5, 3).
    assert "field_x" in sdfg.arrays, f"missing SoA companion; arrays={sorted(sdfg.arrays)}"
    assert tuple(sdfg.arrays["field_x"].shape) == (4, 2, 5, 3), sdfg.arrays["field_x"].shape

    # field(2,1,3) -> 0-based (1,0,2); its 3-vector is x (struct dummy became field_x)
    field_x = np.zeros((4, 2, 5, 3), order="F", dtype=np.float64)
    field_x[1, 0, 2, :] = [3.0, 4.0, 12.0]
    out = np.zeros(1, dtype=np.float64)
    sdfg(field_x=field_x, out=out)
    # dot_product(v, v) = 9 + 16 + 144 = 169.
    assert abs(out[0] - 169.0) < 1e-12, out[0]


# ----------------------------------------------------------------------------
# Gate #7: struct dummy with a POINTER-array-of-records member (``e(:,:)``) whose
# inner static leaf ``x(3)`` is accessed WHOLE through an inlined callee.
# ``splitMultiDimAoRScalarMembers`` flattens it to dynamic SoA ``c_e_x(d0,d1,3)``
# (pointer extents leading, member extent trailing). Regression for coriolis'
# ``operators_coefficients_edge2vert_coeff_cc_t_x`` KeyError; needs array-leaf
# support in the scalar-member splitter + walk-back alias-following.
# ----------------------------------------------------------------------------

_PTR_AOR_ARR_MEMBER_SRC = """
module lib
  implicit none
  type cc
    real(8) :: x(3)
  end type cc
  type coeff
    type(cc), pointer :: e(:, :)
  end type coeff
contains
  subroutine inner(c, out)
    type(coeff), intent(in) :: c
    real(8), intent(out) :: out
    out = dot_product(c % e(2, 1) % x, c % e(2, 1) % x)
  end subroutine inner

  subroutine driver(c, out)
    type(coeff), intent(in) :: c
    real(8), intent(out) :: out
    call inner(c, out)
  end subroutine driver
end module lib
"""


def test_inlined_pointer_aor_array_member_flattens_to_dynamic_soa(tmp_path):
    sdfg = build_sdfg(_PTR_AOR_ARR_MEMBER_SRC, tmp_path / "sdfg", name="driver", entry="lib::driver").build()
    assert "c_e_x" in sdfg.arrays, f"missing pointer-AoR SoA companion; arrays={sorted(sdfg.arrays)}"
    # Two runtime pointer-array dims + the static member extent (3).
    assert len(sdfg.arrays["c_e_x"].shape) == 3
    assert str(sdfg.arrays["c_e_x"].shape[-1]) == "3"

    d0, d1 = 4, 2
    c_e_x = np.zeros((d0, d1, 3), order="F", dtype=np.float64)
    c_e_x[1, 0, :] = [3.0, 4.0, 12.0]  # c%e(2,1)%x -> 0-based (1,0,:)
    out = np.zeros(1, dtype=np.float64)
    sdfg(c_e_x=c_e_x, out=out, c_e_x_d0=d0, c_e_x_d1=d1)
    assert abs(out[0] - 169.0) < 1e-12, out[0]


# ----------------------------------------------------------------------------
# Gate #8 (A/C): a SCALAR member nested under a POINTER-ARRAY-OF-RECORDS
# intermediate, used only symbolically (a DO loop bound), must register as a
# caller-bound SDFG free symbol, not surface as "unresolved free symbol". ICON-O
# shape: ``patch_3d%p_patch_2d(jb)%edges%in_domain%start_block``, a runtime
# pointer array the flatten pass can't statically split, so the scalar stays a
# live designate chain. ``traceToDecl`` renders the flat name but needs
# ``extract_vars`` to synthesise a ``role=symbol`` VarInfo backing it -- the
# scalar-as-symbol counterpart to the whole-array DATA-member tests above.
# ----------------------------------------------------------------------------

_NESTED_AOR_SCALAR_SYMBOL_SRC = """
module lib
  implicit none
  type subset
    integer :: start_block
    integer :: end_block
  end type subset
  type grid
    type(subset) :: in_domain
  end type grid
  type patch3d
    type(grid), pointer :: p2d(:)
  end type patch3d
contains
  subroutine inner(patch, out)
    type(patch3d), intent(in) :: patch
    real(8), intent(out) :: out
    integer :: jb
    out = 0.0d0
    do jb = patch % p2d(1) % in_domain % start_block, patch % p2d(1) % in_domain % end_block
      out = out + 1.0d0
    end do
  end subroutine inner
end module lib
"""


def test_nested_aor_scalar_member_loop_bound_registers_as_symbol(tmp_path):
    sdfg = build_sdfg(_NESTED_AOR_SCALAR_SYMBOL_SRC, tmp_path / "sdfg", name="inner", entry="lib::inner").build()
    syms = set(sdfg.symbols)
    # Both nested-struct scalar bounds resolve as caller-bound free symbols.
    for m in ("start_block", "end_block"):
        name = f"patch_p2d_in_domain_{m}"
        assert name in syms, (f"nested struct-member loop bound {name!r} should register as an SDFG "
                              f"symbol; symbols={sorted(syms)}")
    # struct dummy wasn't statically flattened (pointer-array intermediate) -- bounds are pure symbols, no data companion
    assert "patch_p2d_in_domain_start_block" not in sdfg.arrays


_NESTED_AOR_ARRAY_MEMBER_SRC = """
module lib
  implicit none
  type vert
    integer, pointer :: dolic(:, :)
  end type vert
  type patch3d
    type(vert), pointer :: p1d(:)
  end type patch3d
contains
  subroutine inner(patch, i, j, out)
    type(patch3d), intent(in) :: patch
    integer, intent(in) :: i, j
    integer, intent(out) :: out
    out = patch % p1d(1) % dolic(i, j)
  end subroutine inner
end module lib
"""


def test_nested_aor_array_member_registers_with_record_dim(tmp_path):
    # ``patch % p1d(1) % dolic(i,j)`` -- a runtime pointer-array-of-records array
    # member left as a live designate chain. ``expandDesignateChain`` PREPENDS the
    # record index, so the companion must register at record_dim+member_rank (1+2=3)
    # or the top offset symbol is never registered and the build raises
    # "unresolved free symbol". Build succeeding IS the binding assertion.
    sdfg = build_sdfg(_NESTED_AOR_ARRAY_MEMBER_SRC, tmp_path / "sdfg", name="inner", entry="lib::inner").build()
    assert "patch_p1d_dolic" in sdfg.arrays, (f"nested pointer-AoR array member companion missing; "
                                              f"arrays={sorted(sdfg.arrays)}")
    shape = sdfg.arrays["patch_p1d_dolic"].shape
    assert len(shape) == 3, (f"companion must carry the prepended record dim (rank 1+2=3), got rank "
                             f"{len(shape)}: {shape}")


# ----------------------------------------------------------------------------
# Type-level PRIVATE components: flatten gate is a CONJUNCTION of "flang
# mangled it <type>.<base>" AND "function never designates it" -- a test per half.
# ----------------------------------------------------------------------------


def _private_component_names(tmp_path):
    b = build_sdfg(_PRIVATE_COMPONENT_SRC,
                   tmp_path,
                   name="private_component_struct",
                   entry="m_private_components::private_component_kernel")
    return _names(b)


def test_private_component_never_designated_is_pruned(tmp_path):
    """Gate half 1: a PRIVATE component the kernel never designates must NOT get a
    flat companion. flang mangles it with an illegal '.' in the name; ICON's real
    ``t_patch % comm_pat_gather_c`` shape (27 gfortran syntax errors before the gate)."""
    names = _private_component_names(tmp_path)
    assert not any(
        "never_used" in n
        for n in names), (f"private, never-designated component must be pruned, not flattened; got {sorted(names)}")


def test_private_component_designated_in_module_survives(tmp_path):
    """Gate half 2: a PRIVATE component the in-module kernel DOES designate must
    still flatten -- keying on "is private" alone would drop a live member (the
    ocean solver's ``act__tag`` dispatch tag); the kernel reads ``o%book%seen(1)``."""
    names = _private_component_names(tmp_path)
    assert any("seen" in n for n in names), (
        f"private component designated by the in-module kernel must survive the gate; got {sorted(names)}")


def test_public_sibling_of_private_component_still_flattens(tmp_path):
    """Ordinary public members (``data``, unmangled) are never a gate candidate, unaffected."""
    names = _private_component_names(tmp_path)
    assert any("data" in n for n in names), (f"public member must be unaffected by the gate; got {sorted(names)}")
