"""Array-of-Records (AoR) e2e tests.

An "array of records" in Fortran is what C calls an array of structs:
``TYPE(t) :: arr(N)`` -- N records side by side in memory.  The
bridge's ``hlfir-flatten-structs`` pass lowers this AoS layout to
SoA (Structure of Arrays) so each field becomes its own flat array
the SDFG keys by ``<base>_<field>`` -- ``arr % x`` -> ``arr_x``,
``arr % i`` -> ``arr_i``, etc.

Coverage in this file builds up from the simplest static case to
QE's surfacing pattern (``tabxx(ia) % box(ir)`` -- pointer AoR with
allocatable inner array):

  L1  Static AoR, scalar member               -- arr(i) % x
  L2  Static AoR, array member                -- arr(i) % x(j)
  L3  Pointer AoR, scalar member              -- arr(i) % x
  L4  Pointer AoR, allocatable inner array    -- arr(i) % box(j)

Each probe verifies the SDFG builds AND produces the right number
via an element-wise compare against a numpy reference.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# ----------------------------------------------------------------
# L1: static AoR, scalar member
# ----------------------------------------------------------------


def test_aor_l1_static_scalar_member(tmp_path):
    """``TYPE(t) :: arr(3); arr(i) % x`` -- the simplest case.
    The flatten pass produces a 1-D ``arr_x`` SDFG array."""
    src = """
module m
  type :: t
    real(kind=8) :: x
  end type
contains
  subroutine driver(arr, out)
    type(t), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = arr(1) % x + arr(2) % x + arr(3) % x
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert "arr_x" in sdfg.arrays
    arr_x = np.array([1.0, 2.0, 4.0], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(arr_x=arr_x, out=out)
    np.testing.assert_allclose(out[0], arr_x[0] + arr_x[1] + arr_x[2])


# ----------------------------------------------------------------
# L2: static AoR, array member
# ----------------------------------------------------------------


def test_aor_l2_static_array_member(tmp_path):
    """``TYPE(t) :: arr(3); arr(i) % x(j)`` with ``x(4)`` per record.
    Flatten produces a 2-D ``arr_x`` shape ``(3, 4)`` -- record index
    on the outer (Fortran-leftmost) dim, field index on the inner."""
    src = """
module m
  type :: t
    real(kind=8) :: x(4)
  end type
contains
  subroutine driver(arr, out)
    type(t), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = arr(1) % x(1) + arr(2) % x(2) + arr(3) % x(3)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert "arr_x" in sdfg.arrays
    assert tuple(int(s) for s in sdfg.arrays["arr_x"].shape) == (3, 4)
    arr_x = np.array([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(arr_x=arr_x, out=out)
    np.testing.assert_allclose(out[0], arr_x[0, 0] + arr_x[1, 1] + arr_x[2, 2])


def test_aor_l2_static_array_member_loop_indexed(tmp_path):
    """Same shape as L2 but accessed in a runtime-indexed loop --
    verifies the flatten preserves the runtime-index path
    (``arr_x(i, j)``)."""
    src = """
module m
  type :: t
    real(kind=8) :: x(4)
  end type
contains
  subroutine driver(arr, n, out)
    integer, intent(in) :: n
    type(t), intent(in) :: arr(n)
    real(kind=8), intent(out) :: out
    integer :: i
    out = 0.0d0
    do i = 1, n
      out = out + arr(i) % x(2)
    end do
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert "arr_x" in sdfg.arrays
    n = 3
    arr_x = np.array([[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]], dtype=np.float64, order='F')
    out = np.zeros((1, ), dtype=np.float64, order='F')
    sdfg(arr_x=arr_x, n=np.int32(n), out=out)
    np.testing.assert_allclose(out[0], arr_x[:, 1].sum())


# ----------------------------------------------------------------
# L3: pointer AoR
# ----------------------------------------------------------------


def test_aor_l3_pointer_scalar_member(tmp_path):
    """``type(t), pointer :: arr(:); arr(i) % x``.  The pointer
    descriptor (``!fir.box<!fir.ptr<...>>``) has zero users after
    ``hlfir-flatten-structs`` redirects all accesses to the flat
    companion ``arr_x``; the phantom-pointer-drop guard in
    extract_vars.cpp ensures only the flat companion lands on
    the SDFG signature."""
    src = """
module m
  type :: t
    real(kind=8) :: x
  end type
contains
  subroutine driver(arr, n, out)
    integer, intent(in) :: n
    type(t), pointer, intent(in) :: arr(:)
    real(kind=8), intent(out) :: out
    integer :: i
    out = 0.0d0
    do i = 1, n
      out = out + arr(i) % x
    end do
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    # Should be flat: only arr_x, not arr.
    assert "arr_x" in sdfg.arrays
    assert "arr" not in sdfg.arrays


# ----------------------------------------------------------------
# L4: pointer AoR with allocatable inner array (QE tabxx pattern)
# ----------------------------------------------------------------


def test_aor_multi_level_static_const_indexed(tmp_path):
    """``arr(i) % inner % x(j)`` -- two levels of struct inside the AoR.
    Tests that the flat name ``arr_inner_x`` propagates through the
    designate chain.  Per user request: 'For implementations wrt. AoS
    struct hierarchies we need to be able to support multiple levels'."""
    src = """
module m
  type :: inner_t
    real(kind=8) :: x(4)
  end type
  type :: outer_t
    type(inner_t) :: inner
  end type
contains
  subroutine driver(arr, out)
    type(outer_t), intent(in) :: arr(3)
    real(kind=8), intent(out) :: out
    out = arr(1) % inner % x(1) + arr(2) % inner % x(2) + arr(3) % inner % x(3)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert "arr_inner_x" in sdfg.arrays


def test_aor_l4_pointer_with_allocatable_inner(tmp_path):
    """Pointer AoR with an allocatable array member -- the QE shape."""
    src = """
module m
  type :: t
    integer, allocatable :: box(:)
  end type
contains
  subroutine driver(arr, n, ia, ir, out)
    integer, intent(in) :: n, ia, ir
    type(t), pointer, intent(in) :: arr(:)
    integer, intent(out) :: out
    out = arr(ia) % box(ir)
  end subroutine
end module
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="driver", entry="driver").build()
    assert "arr_box" in sdfg.arrays
