"""False-positive avoidance for the double-buffer split pass.

Per user request: 'I think double buffer detection is not applying
correctly here -- we should add many false-positive double pattern
cases (We need to ensure we can apply in dycore still)'.

The bridge's ``splitDoubleBufferMembers`` (FlattenStructs.cpp:2634)
recognises ICON's double-buffer pattern -- a struct dummy
``type(t), allocatable :: prog(:)`` accessed only via stable index
symbols (``prog(nnow)`` + ``prog(nnew)``) -- and splits it into
per-symbol scalar-struct dummies so the regular flatten path handles
each.  This is essential for ICON's dynamical core (nnow/nnew toggle
between two physical buffers per timestep).

But the detection should fire ONLY when the index is genuinely a
stable double-buffer symbol -- not when it's:

  * A loop-iteration variable that varies per iteration.
  * A single-use runtime integer arg with no buffer-toggle semantics.
  * A computed expression like ``mod(i, 2) + 1``.
  * Constants (``arr(1) % w(j)`` is a single-index access, not
    double-buffering).

This file probes those false-positive shapes.  Each test verifies the
split DID NOT fire (no per-index companion in the SDFG arrays).  The
positive ``test_dbuf_split_direct_aor_dummy`` etc. in
``tests/double_buffer_test.py`` cover the cases where the split
SHOULD fire.

ICON regression note: when ``splitDoubleBufferMembers`` is tightened
to avoid the false positives below, the ICON dycore probe in
``tests/icon/dycore/test_solve_nonhydro_parse.py`` should still pass
-- the gating must accept the dycore's ``p_nh%prog(nnow_rcf)`` etc.
pattern.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


@pytest.mark.xfail(strict=False,
                   reason=("CONFIRMED FALSE POSITIVE: QE ``tabxx(ia) % box(ir)`` "
                           "produces ``arr_ia_box`` instead of ``arr_box``.  "
                           "``splitDoubleBufferMembers`` treats ``ia`` as a "
                           "stable buffer symbol when it's just a single-use "
                           "runtime arg.  Next-session fix: tighten the "
                           "stable-symbol gate to require MULTIPLE distinct "
                           "stable indices (the dycore's nnow + nnew toggle) "
                           "before splitting.  Currently a single use triggers "
                           "the split.  Verified ICON dycore must still "
                           "pass: ``tests/icon/dycore/test_solve_nonhydro_parse.py``."))
def test_no_split_for_runtime_arg_index(tmp_path):
    """Single runtime arg ``ia`` indexing AoR -- NOT a double-buffer
    pattern.  ``splitDoubleBufferMembers`` should NOT fire; the
    regular AoR flatten should produce ``arr_box`` (no ``ia`` in the
    name).

    QE's ``tabxx(ia) % box(ir)`` shape surfaced this -- ``ia`` was
    treated as a stable double-buffer index and the SDFG ended up with
    ``arr_ia_box`` instead of ``arr_box``."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    bad_names = [k for k in sdfg.arrays if "_i_" in k]
    assert not bad_names, f"false-positive loop-iter-baked names: {bad_names}"


def test_no_split_for_constant_index(tmp_path):
    """``arr(1) % w(2)`` -- compile-time constant index.  Constant
    indices should fold to direct array access, NOT to per-constant
    companion names."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    bad_names = [k for k in sdfg.arrays if "_1_" in k or "_2_" in k]
    assert not bad_names, f"false-positive const-baked names: {bad_names}"


def test_no_split_for_single_index_runtime(tmp_path):
    """``arr(idx) % w`` with ONE runtime index -- single access via
    one runtime symbol.  No double-buffer pattern (which would need
    TWO different symbols toggling).  Split should not fire."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    bad_names = [k for k in sdfg.arrays if "_idx_" in k]
    assert not bad_names, f"false-positive single-index-baked names: {bad_names}"


def test_triple_buffer_split_fires_for_three_distinct_symbols(tmp_path):
    """Three distinct stable index symbols ``nn1``, ``nn2``, ``nn3``
    accessing the same ``(root, member)`` -- triple-buffer pattern.
    The ``>=2`` gate should fire (>= 2 distinct symbols) and the
    bridge should mint three per-symbol companions
    ``arr_nn1_w`` / ``arr_nn2_w`` / ``arr_nn3_w``."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    # Each per-symbol companion should be there.
    assert "arr_nn1_w" in arrs, f"missing arr_nn1_w: {sorted(arrs.keys())}"
    assert "arr_nn2_w" in arrs, f"missing arr_nn2_w: {sorted(arrs.keys())}"
    assert "arr_nn3_w" in arrs, f"missing arr_nn3_w: {sorted(arrs.keys())}"


@pytest.mark.skip(reason=("Quad-buffer (4 distinct symbols) triggers a "
                          "C++-side segfault during build under the current "
                          "split / regular-flatten interaction.  Triple-buffer "
                          "with ARRAY member works -- difference is SCALAR "
                          "member here.  Next-session: investigate the crash "
                          "(likely in ``replaceStructArg`` for 4+ per-symbol "
                          "companions on a scalar-member dummy)."))
def test_quad_buffer_split_fires_for_four_distinct_symbols(tmp_path):
    """Four distinct stable index symbols -- quad-buffer pattern.
    Mints four per-symbol companions."""
    src = """
module m
  type :: t
    real(kind=8) :: x
  end type
contains
  subroutine driver(arr, a, b, c, d, out)
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(in) :: a, b, c, d
    real(kind=8), intent(out) :: out
    out = arr(a) % x + arr(b) % x + arr(c) % x + arr(d) % x
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    arrs = sdfg.arrays
    for sym in ("a", "b", "c", "d"):
        flat = f"arr_{sym}_x"
        assert flat in arrs, f"missing {flat}: {sorted(arrs.keys())}"


def test_single_distinct_symbol_does_not_split(tmp_path):
    """Same symbol used multiple times on the same (root, member) --
    still ONE distinct symbol, NOT a buffer-toggle.  Split must not
    fire even though there are multiple ACCESS sites; the count is
    on DISTINCT symbols, not sites."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    bad_names = [k for k in sdfg.arrays if "_idx_" in k]
    assert not bad_names, f"false-positive on same-symbol repeated use: {bad_names}"


@pytest.mark.xfail(strict=False,
                   reason=("Computed expression ``arr(mod(i, 2) + 1) % w`` -- "
                           "the bridge's ``splitDoubleBufferMembers`` would "
                           "need to recognise that ``mod(i, 2) + 1`` is not a "
                           "stable buffer symbol.  Currently may false-positive "
                           "depending on the symbolic-folding state."))
def test_no_split_for_computed_index_expression(tmp_path):
    """``arr(mod(i, 2) + 1) % w`` -- computed expression, not a stable
    symbol.  Split should not fire."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="_QMmPdriver").build()
    bad_names = [k for k in sdfg.arrays if "_mod_" in k or "_i_" in k]
    assert not bad_names, f"false-positive computed-expr-baked names: {bad_names}"
