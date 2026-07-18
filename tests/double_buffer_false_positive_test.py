"""False-positive avoidance for the double-buffer split pass.

The bridge's ``splitDoubleBufferMembers`` (FlattenStructs.cpp:2634) splits an
allocatable struct-array dummy accessed via >=2 stable index symbols (ICON's
nnow/nnew toggle) into per-symbol scalar-struct dummies. Must fire ONLY for
genuinely stable double-buffer symbols -- not loop iterators, single-use
runtime args, computed expressions, or constants. Positive cases live in
``tests/double_buffer_test.py``; the dycore probe in
``tests/icon/dycore/test_solve_nonhydro_parse.py`` must keep passing.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_no_split_for_runtime_arg_index(tmp_path):
    """Single runtime arg ``ia`` indexing AoR is NOT a double-buffer pattern;
    regular AoR flatten should produce ``arr_box`` (no ``ia`` in the name).
    QE's ``tabxx(ia) % box(ir)`` shape surfaced this as ``arr_ia_box``."""
    src = """
module m
  type :: t
    integer, allocatable :: box(:)
  end type
contains
  subroutine driver(arr, ia, ir, out)
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(in) :: ia, ir
    integer, intent(out) :: out
    out = arr(ia) % box(ir)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    # Should NOT have ``arr_ia_box`` (the false-positive split shape).
    # Should have ``arr_box`` (the proper AoR flatten).
    bad_names = [k for k in sdfg.arrays if "_ia_" in k or "_ir_" in k]
    assert not bad_names, f"false-positive index-baked names: {bad_names}"


def test_no_split_for_loop_iter_index(tmp_path):
    """``do i = 1, n; arr(i) % w(:) ...`` -- loop iter ``i`` varies
    per iteration; NOT a double-buffer.  Split must not fire."""
    src = """
module m
  type :: t
    real(kind=8) :: w(4)
  end type
contains
  subroutine driver(arr, n, out)
    integer, intent(in) :: n
    type(t), intent(in) :: arr(n)
    real(kind=8), intent(out) :: out
    integer :: i
    out = 0.0d0
    do i = 1, n
      out = out + arr(i) % w(1)
    end do
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    bad_names = [k for k in sdfg.arrays if "_i_" in k]
    assert not bad_names, f"false-positive loop-iter-baked names: {bad_names}"


def test_no_split_for_constant_index(tmp_path):
    """``arr(1) % w(2)`` -- constant indices fold to direct array access,
    NOT per-constant companion names."""
    src = """
module m
  type :: t
    real(kind=8) :: w(4)
  end type
contains
  subroutine driver(arr, out)
    type(t), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = arr(1) % w(2)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    bad_names = [k for k in sdfg.arrays if "_1_" in k or "_2_" in k]
    assert not bad_names, f"false-positive const-baked names: {bad_names}"


def test_no_split_for_single_index_runtime(tmp_path):
    """``arr(idx) % w`` with ONE runtime index: double-buffer needs TWO
    distinct toggling symbols, so split should not fire."""
    src = """
module m
  type :: t
    real(kind=8) :: w
  end type
contains
  subroutine driver(arr, idx, out)
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(in) :: idx
    real(kind=8), intent(out) :: out
    out = arr(idx) % w
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    bad_names = [k for k in sdfg.arrays if "_idx_" in k]
    assert not bad_names, f"false-positive single-index-baked names: {bad_names}"


def test_triple_buffer_split_fires_for_three_distinct_symbols(tmp_path):
    """Three distinct stable index symbols on the same (root, member): the
    >=2 gate fires, minting per-symbol companions arr_nn1_w/arr_nn2_w/arr_nn3_w."""
    src = """
module m
  type :: t
    real(kind=8) :: w(4)
  end type
contains
  subroutine driver(arr, nn1, nn2, nn3, out)
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(in) :: nn1, nn2, nn3
    real(kind=8), intent(out) :: out
    out = arr(nn1) % w(1) + arr(nn2) % w(2) + arr(nn3) % w(3)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    arrs = sdfg.arrays
    # Each per-symbol companion should be there.
    assert "arr_nn1_w" in arrs, f"missing arr_nn1_w: {sorted(arrs.keys())}"
    assert "arr_nn2_w" in arrs, f"missing arr_nn2_w: {sorted(arrs.keys())}"
    assert "arr_nn3_w" in arrs, f"missing arr_nn3_w: {sorted(arrs.keys())}"


def test_quad_buffer_split_fires_for_four_distinct_symbols(tmp_path):
    """Four distinct stable index symbols (quad-buffer) mint per-symbol
    companions arr_a_w..arr_d_w. Split is reserved for records with an ARRAY
    member; an all-scalar record instead collapses to a single arr_x via the
    regular flatten (scalar AoR is already one contiguous array)."""
    src = """
module m
  type :: t
    real(kind=8) :: w(4)
  end type
contains
  subroutine driver(arr, a, b, c, d, out)
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(in) :: a, b, c, d
    real(kind=8), intent(out) :: out
    out = arr(a) % w(1) + arr(b) % w(2) + arr(c) % w(3) + arr(d) % w(4)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    arrs = sdfg.arrays
    for sym in ("a", "b", "c", "d"):
        flat = f"arr_{sym}_w"
        assert flat in arrs, f"missing {flat}: {sorted(arrs.keys())}"


def test_single_distinct_symbol_does_not_split(tmp_path):
    """Same symbol used multiple times on (root, member) is still ONE distinct
    symbol, not a buffer-toggle; count is on DISTINCT symbols, not access sites."""
    src = """
module m
  type :: t
    real(kind=8) :: w
  end type
contains
  subroutine driver(arr, idx, out)
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(in) :: idx
    real(kind=8), intent(out) :: out
    ! Two USES of the SAME symbol ``idx`` -- still one distinct.
    out = arr(idx) % w + arr(idx) % w
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    bad_names = [k for k in sdfg.arrays if "_idx_" in k]
    assert not bad_names, f"false-positive on same-symbol repeated use: {bad_names}"


def test_no_split_for_computed_index_expression(tmp_path):
    """``arr(mod(i, 2) + 1) % w`` -- computed expression, not a stable symbol;
    split must not fire and MOD must render instead of bottoming out at ``?``."""
    src = """
module m
  type :: t
    real(kind=8) :: w
  end type
contains
  subroutine driver(arr, i, out)
    type(t), intent(in) :: arr(2)
    integer, intent(in) :: i
    real(kind=8), intent(out) :: out
    out = arr(mod(i, 2) + 1) % w
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    bad_names = [k for k in sdfg.arrays if "_mod_" in k or "_i_" in k]
    assert not bad_names, f"false-positive computed-expr-baked names: {bad_names}"
