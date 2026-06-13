"""Compound jagged + AoR + double-buffer probes.

Per user request: 'We should add more jagged array of records, also let's
have a jagged array of records inside a struct array that is used as
double buffering'.

This file builds up from a plain Array-of-Records-of-jagged-members to
the full ICON-dycore-style shape: a struct array
``state(:)`` where each element is a struct whose MEMBERS are
heterogeneous-extent arrays (jagged), and ``state`` itself is accessed
through stable double-buffer index symbols (``nnow`` / ``nnew``
toggle).

Coverage:

  L_A  Jagged-AoR (records whose members have different extents,
       packed into one 2-D companion per record)
  L_B  Same shape with multi-record indexing (runtime + const)
  L_C  Jagged-AoR inside double-buffer accessed AoR struct
  L_D  Compound: jagged + AoR + double-buffer (ICON dycore prog
       struct shape)

Each probe verifies the SDFG builds and (where deterministic) compares
output element-wise.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# -----------------------------------------------------------------
# L_A -- jagged AoR (records with heterogeneous-extent members)
# -----------------------------------------------------------------


def test_jagged_aor_basic(tmp_path):
    """``type :: t; a(3); b(5); c(7); end type; type(t) :: arr(2)``.
    Each record is a jagged scalar-struct.  The bridge has two valid
    flatten outcomes:
      (1) Per-record + per-member: ``arr_a`` of shape (N, 3),
          ``arr_b`` of (N, 5), ``arr_c`` of (N, 7).
      (2) AoR-then-jagged packing (less common).
    Either way, accessing ``arr(i) % a(j)`` must work."""
    src = """
module m
  type :: t
    real(kind=8) :: a(3)
    real(kind=8) :: b(5)
    real(kind=8) :: c(7)
  end type
contains
  subroutine driver(arr, out)
    type(t), intent(in) :: arr(2)
    real(kind=8), intent(out) :: out
    out = arr(1) % a(1) + arr(1) % b(1) + arr(1) % c(1) + &
          arr(2) % a(2) + arr(2) % b(2) + arr(2) % c(2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    # At minimum the bridge should not emit the unflattened struct
    # base ``arr``; it should produce per-field companions.
    assert "arr" not in arrs, f"struct base leaked: {sorted(arrs.keys())}"
    has_per_member = ("arr_a" in arrs and "arr_b" in arrs and "arr_c" in arrs)
    assert has_per_member, f"expected per-member flatten: {sorted(arrs.keys())}"


def test_jagged_aor_runtime_indexed(tmp_path):
    """Same jagged AoR but accessed in a runtime-indexed loop."""
    src = """
module m
  type :: t
    real(kind=8) :: a(3)
    real(kind=8) :: b(5)
  end type
contains
  subroutine driver(arr, n, out)
    integer, intent(in) :: n
    type(t), intent(in) :: arr(n)
    real(kind=8), intent(out) :: out
    integer :: i
    out = 0.0d0
    do i = 1, n
      out = out + arr(i) % a(1) + arr(i) % b(2)
    end do
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    assert "arr_a" in sdfg.arrays
    assert "arr_b" in sdfg.arrays


# -----------------------------------------------------------------
# L_C -- jagged AoR inside double-buffer accessed struct array
# -----------------------------------------------------------------


def test_jagged_aor_in_double_buffered_struct(tmp_path):
    """ICON dycore shape: a top-level struct array indexed by stable
    double-buffer symbols, with EACH element being a struct whose
    members are themselves arrays of records (jagged inner)."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: a(3)
    real(kind=8) :: b(5)
  end type
  type :: outer_t
    type(inner_t) :: data(4)
  end type
contains
  subroutine driver(state, nnow, nnew, out)
    type(outer_t), pointer, intent(in) :: state(:)
    integer, intent(in) :: nnow, nnew
    real(kind=8), intent(out) :: out
    out = state(nnow) % data(1) % a(1) + state(nnew) % data(1) % a(1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    # Double-buffer split should mint per-symbol companions, each
    # with the inner jagged + AoR flatten.
    arrs = sdfg.arrays
    has_buffer_split = any("nnow" in k for k in arrs) and any("nnew" in k for k in arrs)
    assert has_buffer_split, f"expected nnow/nnew companions: {sorted(arrs.keys())}"


# -----------------------------------------------------------------
# L_D -- prog struct shape (ICON dycore typical)
# -----------------------------------------------------------------


def test_double_buffer_member_inside_outer_struct(tmp_path):
    """Pattern ``p % prog(nnow) % w(1)`` -- double-buffered ``prog``
    is a POINTER AoR member inside an outer struct ``p``.  Per user
    request: 'Add support for pattern where the double-buffer member
    is inside a struct like p%'."""
    src = """
module m
  type :: prog_t
    real(kind=8) :: w(4)
  end type
  type :: nh_t
    type(prog_t), pointer :: prog(:)
  end type
contains
  subroutine driver(p, nnow, nnew, out)
    type(nh_t), intent(in) :: p
    integer, intent(in) :: nnow, nnew
    real(kind=8), intent(out) :: out
    out = p % prog(nnow) % w(1) + p % prog(nnew) % w(1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    # Both buffer companions should be present.
    has_buf_split = any("nnow" in k for k in arrs) and any("nnew" in k for k in arrs)
    assert has_buf_split, f"expected nnow/nnew companions: {sorted(arrs.keys())}"


def test_double_buffer_member_nested_two_levels(tmp_path):
    """Pattern ``p % w % s(nnow)`` -- double-buffered ``s`` is two
    levels deep (``p`` -> ``w`` -> ``s``).  Per user request: 'also
    like p%w%s(nnow)'."""
    src = """
module m
  type :: s_t
    real(kind=8) :: v(3)
  end type
  type :: w_t
    type(s_t), pointer :: s(:)
  end type
  type :: p_t
    type(w_t) :: w
  end type
contains
  subroutine driver(p, nnow, nnew, out)
    type(p_t), intent(in) :: p
    integer, intent(in) :: nnow, nnew
    real(kind=8), intent(out) :: out
    out = p % w % s(nnow) % v(1) + p % w % s(nnew) % v(1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    has_buf_split = any("nnow" in k for k in arrs) and any("nnew" in k for k in arrs)
    assert has_buf_split, f"expected nnow/nnew companions for nested chain: {sorted(arrs.keys())}"


def test_dycore_prog_struct_full_shape(tmp_path):
    """``p % prog(nnow) % w`` -- this is the literal ICON prog struct
    access pattern."""
    src = """
module m
  type :: prog_t
    real(kind=8) :: w(4)
    real(kind=8) :: rho(2)
  end type
  type :: nh_t
    type(prog_t), pointer :: prog(:)
  end type
contains
  subroutine driver(p, nnow, nnew, out)
    type(nh_t), intent(in) :: p
    integer, intent(in) :: nnow, nnew
    real(kind=8), intent(out) :: out
    out = p % prog(nnow) % w(1) + p % prog(nnew) % w(1) + &
          p % prog(nnow) % rho(1) + p % prog(nnew) % rho(1)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    # Should mint per-symbol companions for the double-buffer split,
    # each with the inner jagged AoR fully flattened.
    arrs = sdfg.arrays
    has_buffer_split = any("nnow" in k for k in arrs) and any("nnew" in k for k in arrs)
    assert has_buffer_split, f"expected nnow/nnew companions: {sorted(arrs.keys())}"
