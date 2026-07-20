"""Regression: writes through an inlined-callee dummy bound to a WHOLE POINTER struct member
must reach the member.

The ICON ``solve_free_sfc`` idiom: ``st%p_diag%vort`` (a ``REAL(8), POINTER :: (:,:)`` member)
is passed whole to an inlined worker with a contiguous explicit-shape dummy ``v(n1,n2)`` that
WRITES it. Flang guards contiguity with an ``hlfir.copy_in`` / ``hlfir.copy_out`` pair; the
bridge models neither, so the temp surfaced as a phantom argument ``v`` and the write-back was
dropped -- the member never changed (and calling with the member name raised ``KeyError: 'v'``).

``hlfir-fold-copy-in-out`` now reparents the alias accesses onto the member box directly
(``hlfir.designate %memberBox (i, j)``), which honours the component's strides and erases the
phantom ``v``. A per-element pattern (``i + 100*j``) catches a dropped/transposed index.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_wb
  implicit none
  type diag_t
    real(8), pointer :: vort(:, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine compute_vort(v, n1, n2)
    integer, intent(in) :: n1, n2
    real(8), intent(inout) :: v(n1, n2)
    integer :: i, j
    do j = 1, n2
      do i = 1, n1
        v(i, j) = real(i, 8) + 100.0d0 * real(j, 8)
      end do
    end do
  end subroutine compute_vort
  subroutine run(st, n1, n2)
    type(state_t), intent(inout) :: st
    integer, intent(in) :: n1, n2
    call compute_vort(st % p_diag % vort, n1, n2)
  end subroutine run
end module mo_wb
"""


def test_writeback_through_whole_pointer_member_inlined(tmp_path: Path):
    """The inlined worker's writes to ``st%p_diag%vort`` must land in the member. Without the
    reparent the member is a phantom ``v`` arg and the writes vanish (call raises KeyError)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="run", entry="mo_wb::run").build()
    sdfg.validate()

    # The member is the SDFG arg -- no phantom ``v``.
    assert "v" not in sdfg.arrays, f"phantom copy-in temp leaked as an SDFG array: {sorted(sdfg.arrays)}"

    n1, n2 = 3, 4
    vort = np.zeros((n1, n2), dtype=np.float64, order="F")
    sdfg(st_p_diag_vort=vort, n1=np.int32(n1), n2=np.int32(n2))

    i = np.arange(1, n1 + 1, dtype=np.float64)[:, None]
    j = np.arange(1, n2 + 1, dtype=np.float64)[None, :]
    expected = i + 100.0 * j
    np.testing.assert_array_equal(vort, expected)


# The veloc_adv_vert -> _mimetic -> _rot idiom: the pointer member is FORWARDED through two
# callee levels before it is written, plus a whole-plane section zero-init on the dummy.
_SRC_CHAIN = """
module mo_wb2
  implicit none
  type diag_t
    real(8), pointer :: vort(:, :, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine inner(x, n1, n2, nb, sb, eb)
    integer, intent(in) :: n1, n2, nb, sb, eb
    real(8), intent(inout) :: x(n1, n2, nb)
    integer :: b, i, j
    do b = sb, eb
      x(:, :, b) = 0.0d0
      do j = 1, n2
        do i = 1, n1
          x(i, j, b) = real(i, 8) + 100.0d0 * real(j, 8)
        end do
      end do
    end do
  end subroutine inner
  subroutine outer(y, n1, n2, nb, sb, eb)
    integer, intent(in) :: n1, n2, nb, sb, eb
    real(8), intent(inout) :: y(n1, n2, nb)
    call inner(y, n1, n2, nb, sb, eb)
  end subroutine outer
  subroutine run(st, n1, n2, nb, sb, eb)
    type(state_t), intent(inout) :: st
    integer, intent(in) :: n1, n2, nb, sb, eb
    call outer(st % p_diag % vort, n1, n2, nb, sb, eb)
  end subroutine run
end module mo_wb2
"""


def test_writeback_through_forwarding_chain_inlined(tmp_path: Path):
    """Member forwarded through TWO inlined levels (outer -> inner). Each level re-declares the
    dummy, so the copy_in temp reaches the stores through a ladder of ``hlfir.declare`` re-views.
    reparentMemberCopy must follow the whole ladder; before the fix it saw the inner re-declare as
    a foreign use, bailed, and the writes landed in a disconnected phantom ``y`` (member untouched)."""
    sdfg = build_sdfg(_SRC_CHAIN, tmp_path / "sdfg", name="run", entry="mo_wb2::run").build()
    sdfg.validate()

    # No phantom re-declare temp (``y`` from outer, ``x`` from inner) leaks as a top-level array.
    for phantom in ("x", "y"):
        assert phantom not in sdfg.arrays, f"phantom re-declare temp leaked: {sorted(sdfg.arrays)}"

    n1, n2, nb = 3, 4, 2
    vort = np.zeros((n1, n2, nb), dtype=np.float64, order="F")
    sdfg(st_p_diag_vort=vort, n1=np.int32(n1), n2=np.int32(n2), nb=np.int32(nb), sb=np.int32(1), eb=np.int32(nb))

    i = np.arange(1, n1 + 1, dtype=np.float64)[:, None]
    j = np.arange(1, n2 + 1, dtype=np.float64)[None, :]
    expected = i + 100.0 * j
    for b in range(nb):
        np.testing.assert_array_equal(vort[:, :, b], expected)


# The veloc_adv_vert_mimetic idiom: the pointer member reaches the explicit-shape writer through an
# ASSUMED-shape dummy. Flang reboxes the pointer member into an assumed-shape box and copy_in's THAT
# for contiguity, so the copy_in source is a declare(rebox(load(designate))), not a plain load.
_SRC_ASSUMED = """
module mo_wb3
  implicit none
  type diag_t
    real(8), pointer :: e(:, :, :)
  end type
  type state_t
    type(diag_t) :: p_diag
  end type
contains
  subroutine writer(e, n1, n2, nb, sb, eb)
    integer, intent(in) :: n1, n2, nb, sb, eb
    real(8), intent(inout) :: e(n1, n2, nb)        ! explicit shape
    integer :: b, i, j
    do b = sb, eb
      e(:, :, b) = 0.0d0
      do j = 1, n2
        do i = 1, n1
          e(i, j, b) = real(i, 8) + 100.0d0 * real(j, 8)
        end do
      end do
    end do
  end subroutine writer
  subroutine dispatch(e, n1, n2, nb, sb, eb)
    integer, intent(in) :: n1, n2, nb, sb, eb
    real(8), intent(inout) :: e(:, :, :)            ! assumed shape
    call writer(e, n1, n2, nb, sb, eb)
  end subroutine dispatch
  subroutine run(st, n1, n2, nb, sb, eb)
    type(state_t), intent(inout) :: st
    integer, intent(in) :: n1, n2, nb, sb, eb
    call dispatch(st % p_diag % e, n1, n2, nb, sb, eb)
  end subroutine run
end module mo_wb3
"""


def test_writeback_through_assumed_shape_forward_inlined(tmp_path: Path):
    """Member forwarded through an ASSUMED-shape dummy then an explicit-shape writer. The copy_in
    source is declare(rebox(load(designate))); dispatchNonSectionSource must trace past the rebox to
    the pointer member. Before the fix that copy_in matched neither fold path -> phantom ``e``, member
    left untouched (the residual solve_free_sfc veloc_adv_vert plane drop)."""
    sdfg = build_sdfg(_SRC_ASSUMED, tmp_path / "sdfg", name="run", entry="mo_wb3::run").build()
    sdfg.validate()

    assert "e" not in sdfg.arrays, f"phantom assumed-shape copy-in temp leaked: {sorted(sdfg.arrays)}"

    n1, n2, nb = 3, 4, 2
    e = np.zeros((n1, n2, nb), dtype=np.float64, order="F")
    sdfg(st_p_diag_e=e, n1=np.int32(n1), n2=np.int32(n2), nb=np.int32(nb), sb=np.int32(1), eb=np.int32(nb))

    i = np.arange(1, n1 + 1, dtype=np.float64)[:, None]
    j = np.arange(1, n2 + 1, dtype=np.float64)[None, :]
    expected = i + 100.0 * j
    for b in range(nb):
        np.testing.assert_array_equal(e[:, :, b], expected)
