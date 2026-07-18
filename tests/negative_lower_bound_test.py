"""Negative-lower-bound Fortran arrays.

ICON uses refined-cell-tag indexing (``start_block(min_rlcell_int:max_rlcell_int)``,
spanning roughly [-10, 7]). The bridge lowers accesses as
``arr[(fortran_index) - offset_<arr>_d<i>]``, so the lower bound must land as
the offset symbol's value -- from the declare for explicit-shape arrays, or
inferred (literal accesses / runtime ALLOCATE) for deferred-shape allocatables.
"""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_explicit_negative_lower_bound(tmp_path: Path):
    """Explicit-shape ``arr(-5:5)`` -- the bridge SHOULD see the
    bound on the declare and specialise the offset symbol."""
    src = """
module read_arr_mod
contains
subroutine read_arr(arr, idx, out)
  implicit none
  integer, intent(in) :: arr(-5:5)
  integer, intent(in) :: idx
  integer, intent(out) :: out
  out = arr(idx)
end subroutine read_arr
end module read_arr_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_arr", entry="read_arr_mod::read_arr").build()
    sdfg.validate()

    # arr[i] = (i-5)*10 -- distinguishes offset-by-1 from offset-by-(-5).
    arr = np.asfortranarray(np.array([(i - 5) * 10 for i in range(11)], dtype=np.int32))  # values -50 ... 50
    out = np.zeros(1, dtype=np.int32, order='F')

    # arr(-3) should be -30 (arr(-5)=-50, arr(-4)=-40, ...); a wrong offset would segfault or misread.
    sdfg(arr=arr, idx=np.int32(-3), out=out)
    assert out[0] == -30, f"arr(-3) should be -30; got {out[0]}"

    sdfg(arr=arr, idx=np.int32(5), out=out)
    assert out[0] == 50, f"arr(5) should be 50; got {out[0]}"

    sdfg(arr=arr, idx=np.int32(0), out=out)
    assert out[0] == 0, f"arr(0) should be 0; got {out[0]}"


def test_deferred_shape_allocatable_offset_is_one_by_default(tmp_path: Path):
    """Deferred-shape ``ALLOCATABLE :: arr(:)`` with a positive-only-indexed
    runtime ``ALLOCATE`` -- offset defaults to 1 and works correctly here.
    Documents the gap for negative lower bounds (see tests below)."""
    src = """
module read_alloc_mod
contains
subroutine read_alloc(idx, out)
  implicit none
  integer, intent(in) :: idx
  integer, intent(out) :: out
  integer, allocatable :: arr(:)
  integer :: i
  allocate(arr(1:11))
  do i = 1, 11
    arr(i) = i * 10
  end do
  out = arr(idx)
  deallocate(arr)
end subroutine read_alloc
end module read_alloc_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_alloc", entry="read_alloc_mod::read_alloc").build()
    sdfg.validate()

    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(idx=np.int32(3), out=out)
    # 1-based: arr(3) = 30 -- confirms the default offset=1 path for positive-only indices.
    assert out[0] == 30, f"arr(3) should be 30; got {out[0]}"


def test_deferred_shape_allocatable_negative_lower_bound(tmp_path: Path):
    """ICON refined-cell-tag pattern: ``ALLOCATABLE :: arr(:)`` + runtime
    ``ALLOCATE(arr(-5:5))``, body accesses at literal negative indices.
    Static-inference scans ``hlfir.designate`` ops for the per-dim min literal
    (-5 here) as the lower bound. Pre-fix: offset defaulted to 1, so
    ``arr(-3)`` lowered to ``arr[-4]`` -> segfault."""
    src = """
module read_alloc_mod
contains
subroutine read_alloc(out)
  implicit none
  integer, intent(out) :: out(4)
  integer, allocatable :: arr(:)
  allocate(arr(-5:5))
  arr(-5) = -50
  arr(-3) = -30
  arr( 0) =   0
  arr( 5) =  50
  out(1) = arr(-5)
  out(2) = arr(-3)
  out(3) = arr( 0)
  out(4) = arr( 5)
  deallocate(arr)
end subroutine read_alloc
end module read_alloc_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_alloc", entry="read_alloc_mod::read_alloc").build()
    sdfg.validate()

    # Inference picks -5 (most-negative literal designate index in the body).
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0') == -5, (
        f"expected offset_arr_d0 inferred to -5; got "
        f"{dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0')}")

    out = np.zeros(4, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [-50, -30, 0, 50])


def test_zero_based_allocatable(tmp_path: Path):
    """0-based deferred-shape array (common in ICON, e.g. ``sfc%tsoil(nproma,
    0:nlev_soil, nblks_c)``): inference should drop offset_d0 to 0 after seeing the literal ``arr(0)`` write."""
    src = """
module zero_based_mod
contains
subroutine zero_based(out)
  implicit none
  integer, intent(out) :: out(3)
  integer, allocatable :: arr(:)
  allocate(arr(0:2))
  arr(0) = 100
  arr(1) = 101
  arr(2) = 102
  out(1) = arr(0)
  out(2) = arr(1)
  out(3) = arr(2)
  deallocate(arr)
end subroutine zero_based
end module zero_based_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="zero_based", entry="zero_based_mod::zero_based").build()
    sdfg.validate()
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0') == 0

    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [100, 101, 102])


def test_icon_min_rledge_pattern(tmp_path: Path):
    """ICON's deepest negative bound: ``min_rledge = -13`` (from
    ``mo_impl_constants``); ``mo_alloc_patches.f90`` allocates
    ``start_block(min_rledge:max_rledge)``, read later at literal negative indices."""
    src = """
module icon_edge_blocks_mod
contains
subroutine icon_edge_blocks(out)
  implicit none
  integer, parameter :: min_rl = -13
  integer, parameter :: max_rl =   8
  integer, intent(out) :: out(3)
  integer, allocatable :: end_block(:)
  allocate(end_block(min_rl:max_rl))
  end_block(-13) = -1300
  end_block( -8) =  -800
  end_block(  5) =   500
  out(1) = end_block(-13)
  out(2) = end_block( -8)
  out(3) = end_block(  5)
  deallocate(end_block)
end subroutine icon_edge_blocks
end module icon_edge_blocks_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="icon_edge_blocks", entry="icon_edge_blocks_mod::icon_edge_blocks").build()
    sdfg.validate()
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_end_block_d0') == -13

    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [-1300, -800, 500])


def test_multidim_mixed_negative_bounds(tmp_path: Path):
    """ICON's ``start_idx(min_rlcell:max_rlcell, max_childdom)``: rank-2,
    dim-0 negative lower bound, dim-1 default 1-based; inference adjusts only dim-0."""
    src = """
module mixed_bounds_mod
contains
subroutine mixed_bounds(out)
  implicit none
  integer, intent(out) :: out(4)
  integer, allocatable :: arr(:, :)
  allocate(arr(-4:4, 1:3))
  arr(-4, 1) = -41
  arr( 0, 2) =   2
  arr( 4, 3) =  43
  arr(-2, 2) = -22
  out(1) = arr(-4, 1)
  out(2) = arr( 0, 2)
  out(3) = arr( 4, 3)
  out(4) = arr(-2, 2)
  deallocate(arr)
end subroutine mixed_bounds
end module mixed_bounds_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="mixed_bounds", entry="mixed_bounds_mod::mixed_bounds").build()
    sdfg.validate()
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0') == -4
    # dim 1 has no literal index below 1, so stays at default 1
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d1') == 1

    out = np.zeros(4, dtype=np.int32, order='F')
    sdfg(out=out)
    np.testing.assert_array_equal(out, [-41, 2, 43, -22])


def test_symbolic_index_local_allocatable(tmp_path: Path):
    """Loop-iterator access into a negative-bound *local* allocatable: no
    literal indices, so ``ALLOCATE(arr(-5:5))``'s ``fir.shape_shift -5, 11``
    operand is read directly into ``v.lower_bounds[0]`` -- no literal-index hint needed."""
    src = """
module sym_idx_mod
contains
subroutine sym_idx(out)
  implicit none
  integer, intent(out) :: out(11)
  integer, allocatable :: arr(:)
  integer :: i
  allocate(arr(-5:5))
  do i = -5, 5
    arr(i) = i * 100
  end do
  do i = -5, 5
    out(i + 6) = arr(i)
  end do
  deallocate(arr)
end subroutine sym_idx
end module sym_idx_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="sym_idx", entry="sym_idx_mod::sym_idx").build()
    sdfg.validate()
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0') == -5, (
        f"shape_shift inference should pick -5 from ALLOCATE(arr(-5:5)); "
        f"got {dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0')}")
    out = np.zeros(11, dtype=np.int32, order='F')
    sdfg(out=out)
    expected = np.array([(i - 5) * 100 for i in range(11)], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


def test_dummy_arg_allocatable_literal_negative_index(tmp_path: Path):
    """Dummy-arg deferred-shape allocatable with literal negative indices,
    mirroring ICON velocity_tendencies' flattened ``start_block(-10)`` reads;
    literal-index inference sets ``offset_arr_d0`` without bindings-layer involvement."""
    src = """
module read_dummy_mod
contains
subroutine read_dummy(arr, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(out) :: out(3)
  out(1) = arr(-3)
  out(2) = arr( 0)
  out(3) = arr( 5)
end subroutine read_dummy
end module read_dummy_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="read_dummy", entry="read_dummy_mod::read_dummy").build()
    sdfg.validate()
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0') == -3, (
        f"expected offset_arr_d0 == -3 (most-negative literal); "
        f"got {dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0')}")

    # Buffer is 1-based numpy; SDFG reads at arr[N-(-3)]=arr[N+3], so for
    # N in [-3,5] buf[0]=arr(-3)=-30, buf[3]=arr(0)=0, buf[8]=arr(5)=50.
    arr = np.asfortranarray(np.array([(i - 3) * 10 for i in range(9)], dtype=np.int32))
    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, [-30, 0, 50])


def test_dummy_arg_allocatable_symbolic_loop(tmp_path: Path):
    """Dummy-arg deferred-shape allocatable + loop-iterator access, no literal
    indices: bridge leaves ``offset_arr_d0`` free on the signature; the
    direct-call test passes it explicitly."""
    src = """
module sum_arr_mod
contains
subroutine sum_arr(arr, n, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:)
  integer, intent(in) :: n
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -5, 5
    total = total + arr(i)
  end do
  out = total
end subroutine sum_arr
end module sum_arr_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="sum_arr", entry="sum_arr_mod::sum_arr").build()
    sdfg.validate()
    # Loop bounds are literal but the designate index is symbolic; offset stays free.
    assert 'offset_arr_d0' in sdfg.arglist(), (f"expected offset_arr_d0 to be free on SDFG signature; "
                                               f"arglist: {list(sdfg.arglist().keys())}")

    # Buffer holds arr(-5)..arr(5) at buf[0..10]; sum is 0.
    arr = np.asfortranarray(np.array(range(-5, 6), dtype=np.int32))
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(arr=arr, n=np.int32(11), out=out, offset_arr_d0=np.int64(-5), arr_d0=np.int64(11))
    assert out[0] == 0, f"sum(arr(-5..5)) should be 0; got {out[0]}"


def test_dummy_arg_allocatable_multidim_mixed(tmp_path: Path):
    """Dummy-arg rank-2 allocatable, mixed bounds: dim 0 negative (symbolic
    index), dim 1 default 1-based (literal index). offset_arr_d0 stays free
    (caller binds), offset_arr_d1 defaults to 1."""
    src = """
module sum_col_mod
contains
subroutine sum_col(arr, n, out)
  implicit none
  integer, allocatable, intent(in) :: arr(:, :)
  integer, intent(in) :: n
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -3, 3
    total = total + arr(i, 1)
  end do
  out = total
end subroutine sum_col
end module sum_col_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="sum_col", entry="sum_col_mod::sum_col").build()
    sdfg.validate()
    arglist = sdfg.arglist()
    assert 'offset_arr_d0' in arglist, "offset_arr_d0 should be free"
    # dim 1's literal index (1) is positive, so inference doesn't fire (min>=1);
    # behavior: free if no literal seen, baked at 1 if a positive literal was seen.

    # Buffer: column 0 holds arr(-3..3, 1) = values -3..3 -> sum=0.
    arr = np.asfortranarray(np.array([[i] for i in range(-3, 4)], dtype=np.int32))  # 7 rows, 1 col
    out = np.zeros(1, dtype=np.int32, order='F')
    kw = dict(arr=arr, n=np.int32(7), out=out, offset_arr_d0=np.int64(-3), arr_d0=np.int64(7), arr_d1=np.int64(1))
    if 'offset_arr_d1' in arglist:
        kw['offset_arr_d1'] = np.int64(1)
    sdfg(**kw)
    assert out[0] == 0, f"sum(arr(-3..3, 1)) should be 0; got {out[0]}"


def test_dummy_arg_allocatable_inout_symbolic_write(tmp_path: Path):
    """Dummy-arg ALLOCATABLE, INTENT(INOUT), symbolic-index loop write:
    verifies the runtime offset works for writebacks too, not just reads."""
    src = """
module write_arr_mod
contains
subroutine write_arr(arr)
  implicit none
  integer, allocatable, intent(inout) :: arr(:)
  integer :: i
  do i = -4, 4
    arr(i) = i * 10
  end do
end subroutine write_arr
end module write_arr_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="write_arr", entry="write_arr_mod::write_arr").build()
    sdfg.validate()
    assert 'offset_arr_d0' in sdfg.arglist()

    arr = np.zeros(9, dtype=np.int32, order='F')
    sdfg(arr=arr, offset_arr_d0=np.int64(-4), arr_d0=np.int64(9))
    expected = np.array([i * 10 for i in range(-4, 5)], dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


def test_dummy_arg_allocatable_two_arrays_independent_offsets(tmp_path: Path):
    """Two dummy-arg allocatables with DIFFERENT negative bounds: each gets
    its own free offset symbol, caller passes the right one for each."""
    src = """
module pair_sum_mod
contains
subroutine pair_sum(a, b, out)
  implicit none
  integer, allocatable, intent(in) :: a(:), b(:)
  integer, intent(out) :: out
  integer :: i, total
  total = 0
  do i = -3, 3
    total = total + a(i)
  end do
  do i = -7, 7
    total = total + b(i)
  end do
  out = total
end subroutine pair_sum
end module pair_sum_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="pair_sum", entry="pair_sum_mod::pair_sum").build()
    sdfg.validate()
    arglist = sdfg.arglist()
    assert 'offset_a_d0' in arglist, "a should have a free offset symbol"
    assert 'offset_b_d0' in arglist, "b should have a free offset symbol"

    a = np.asfortranarray(np.array(range(-3, 4), dtype=np.int32))  # sum = 0
    b = np.asfortranarray(np.array(range(-7, 8), dtype=np.int32))  # sum = 0
    out = np.zeros(1, dtype=np.int32, order='F')
    sdfg(a=a, b=b, out=out, offset_a_d0=np.int64(-3), offset_b_d0=np.int64(-7), a_d0=np.int64(7), b_d0=np.int64(15))
    assert out[0] == 0, f"sum(a)+sum(b) should be 0; got {out[0]}"


def test_explicit_shape_parameter_negative_bound(tmp_path: Path):
    """Explicit-shape declare with a PARAMETER lower bound (``arr(lb:5)``,
    ``lb=-8``): ``resolveLowerBounds`` reads the ShapeShiftOp and traces the
    parameter via ``traceConstInt``, alongside the literal-access inference."""
    src = """
module param_bound_mod
contains
subroutine param_bound(arr, out)
  implicit none
  integer, parameter :: lb = -8
  integer, parameter :: ub =  5
  integer, intent(in) :: arr(lb:ub)
  integer, intent(out) :: out(3)
  out(1) = arr(-8)
  out(2) = arr( 0)
  out(3) = arr( 5)
end subroutine param_bound
end module param_bound_mod
"""
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name="param_bound", entry="param_bound_mod::param_bound").build()
    sdfg.validate()
    assert dict(getattr(sdfg, '_fortran_offset_values', sdfg.constants)).get('offset_arr_d0') == -8

    arr = np.asfortranarray(np.array([(i - 8) * 100 for i in range(14)], dtype=np.int32))  # arr(-8) = -800, etc
    out = np.zeros(3, dtype=np.int32, order='F')
    sdfg(arr=arr, out=out)
    np.testing.assert_array_equal(out, [-800, 0, 500])
