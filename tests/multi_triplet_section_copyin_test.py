"""Regression: a copy_in/copy_out section with MORE THAN ONE triplet must fold.

``hlfir-fold-copy-in-out`` used to handle only a single trailing triplet (``arr(i, lo:hi)``),
so a two-triplet section of a non-contiguous parent -- ``st%p_diag%v3(:, :, b)``, the shape 12
of the 18 surviving pairs in ICON-O ``solve_free_sfc`` have -- was left unfolded.  The bridge
models neither copy_in nor copy_out, so the temp surfaced as a phantom SDFG argument that the
binding shim allocated and zero-filled: every write through it was dropped silently.

``arr(:, 1, :)`` is the ordering case: the scalar sits BETWEEN two triplets, so the rebuilt
index list cannot be "scalar prefix then triplet" -- the fold has to walk source dims in order.
A per-element pattern (``i + 100*j``) catches a dropped or transposed index.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Parent is a POINTER member so contiguity is unknown at the call site and flang guards it with
# copy_in/copy_out; a plain contiguous local would be passed by reference with no copy at all.
_SRC_TRAILING = """
module mo_mt
  implicit none
  type diag_t
    real(8), pointer :: v3(:, :, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine fill2d(v, n1, n2, tag)
    integer, intent(in) :: n1, n2
    real(8), intent(in) :: tag
    real(8), intent(inout) :: v(n1, n2)
    integer :: i, j
    do j = 1, n2
      do i = 1, n1
        v(i, j) = tag + real(i, 8) + 100.0d0 * real(j, 8)
      end do
    end do
  end subroutine fill2d
  subroutine run(st, n1, n2, nb)
    type(state_t), intent(inout) :: st
    integer, intent(in) :: n1, n2, nb
    integer :: b
    do b = 1, nb
      call fill2d(st % p_diag % v3(:, :, b), n1, n2, 1000.0d0 * real(b, 8))
    end do
  end subroutine run
end module mo_mt
"""

# Scalar BETWEEN two triplets: parent(i, 1, j), not parent(1, i, j).
_SRC_MIDDLE = """
module mo_mt2
  implicit none
  type diag_t
    real(8), pointer :: v3(:, :, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine fill2d(v, n1, nb)
    integer, intent(in) :: n1, nb
    real(8), intent(inout) :: v(n1, nb)
    integer :: i, b
    do b = 1, nb
      do i = 1, n1
        v(i, b) = real(i, 8) + 100.0d0 * real(b, 8)
      end do
    end do
  end subroutine fill2d
  subroutine run(st, n1, nb)
    type(state_t), intent(inout) :: st
    integer, intent(in) :: n1, nb
    call fill2d(st % p_diag % v3(:, 1, :), n1, nb)
  end subroutine run
end module mo_mt2
"""


def test_two_triplet_trailing_scalar_section(tmp_path: Path):
    """``v3(:, :, b)`` -- two triplets plus a trailing scalar. Writes must reach the member."""
    sdfg = build_sdfg(_SRC_TRAILING, tmp_path / "sdfg", name="run", entry="mo_mt::run").build()
    sdfg.validate()
    assert "v" not in sdfg.arrays, f"phantom copy-in temp leaked as an SDFG array: {sorted(sdfg.arrays)}"

    n1, n2, nb = 3, 4, 2
    v3 = np.zeros((n1, n2, nb), dtype=np.float64, order="F")
    sdfg(st_p_diag_v3=v3, n1=np.int32(n1), n2=np.int32(n2), nb=np.int32(nb))

    i = np.arange(1, n1 + 1, dtype=np.float64)[:, None, None]
    j = np.arange(1, n2 + 1, dtype=np.float64)[None, :, None]
    b = np.arange(1, nb + 1, dtype=np.float64)[None, None, :]
    np.testing.assert_array_equal(v3, 1000.0 * b + i + 100.0 * j)


def test_two_triplet_middle_scalar_section(tmp_path: Path):
    """``v3(:, 1, :)`` -- scalar BETWEEN the triplets. A prefix-then-triplet rebuild would write
    plane ``v3(1, :, :)`` instead, so the untouched planes double as the transposition check."""
    sdfg = build_sdfg(_SRC_MIDDLE, tmp_path / "sdfg", name="run", entry="mo_mt2::run").build()
    sdfg.validate()
    assert "v" not in sdfg.arrays, f"phantom copy-in temp leaked as an SDFG array: {sorted(sdfg.arrays)}"

    n1, n2, nb = 3, 4, 2
    v3 = np.zeros((n1, n2, nb), dtype=np.float64, order="F")
    sdfg(st_p_diag_v3=v3, n1=np.int32(n1), nb=np.int32(nb))

    expected = np.zeros((n1, n2, nb), dtype=np.float64, order="F")
    i = np.arange(1, n1 + 1, dtype=np.float64)[:, None]
    b = np.arange(1, nb + 1, dtype=np.float64)[None, :]
    expected[:, 0, :] = i + 100.0 * b
    np.testing.assert_array_equal(v3, expected)
