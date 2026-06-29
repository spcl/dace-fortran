"""Regression: a pointer rebound in BOTH arms of an ``IF (PRESENT(optional))``
guard could not be lowered (``interleaved rebind`` rejection).

This is ICON's pervasive ``_onBlock`` subset idiom and the second blocker that
stopped the ocean dynamical core (``solve_free_sfc_ab_mimetic``) from lowering:

  ``TYPE(t_subset_range), POINTER :: cells_subset``
  ``IF (PRESENT(subset_range)) THEN``
  ``  cells_subset => subset_range``
  ``ELSE``
  ``  cells_subset => patch_3d%p_patch_2d(1)%cells%in_domain``
  ``END IF``
  ``... start_block = cells_subset%start_block ...``

``hlfir-rewrite-pointer-assigns`` models a rebind as a View of ONE source, so a
pointer rebound to two different targets in mutually-exclusive branches (a
runtime selection) was rejected outright.  But after ``hlfir-inline-all`` each
inlined copy's ``PRESENT(subset_range)`` is a compile-time constant -- the call
either passed the optional or it didn't -- so exactly one branch is live.

``foldPresenceGuardedIfs`` now resolves the optional's presence post-inline
(omitted -> ``fir.absent`` -> false; passed -> concrete box -> true) and folds
the guard to its live branch, hoisting that branch's body in front of the
``if``.  The rebind becomes straight-line (one store whose target dominates the
reads), which the per-pointer rewrite then handles.  Plain ``canonicalize``
can't do this (it doesn't fold ``fir.is_present`` behind an ``hlfir.declare``),
and a general canonicalize at that pipeline slot would break
``hlfir-preserve-mutable-globals`` ordering -- hence the targeted fold.

Both directions are covered: ``driver_absent`` omits the optional (else branch:
``cs => grid%in_domain``) and ``driver_present`` passes ``grid%in_domain`` (then
branch: ``cs => subset``).  Both must build (a broken fold leaves the
unlowerable conditional rebind) and run to the closed form -- the loop writes
``out(jb)=jb`` over the selected subset's ``[sblk, eblk]`` range.  The struct
dummy flattens to scalar companions, so the built SDFG is called directly.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module mo_cr
  implicit none
  type t_sub
    integer :: sblk
    integer :: eblk
  end type
  type t_grid
    type(t_sub) :: in_domain
  end type
contains
  subroutine compute(grid, n, out, subset)
    type(t_grid), intent(in), target :: grid
    integer, intent(in) :: n
    real(8), intent(inout) :: out(:)
    type(t_sub), intent(in), target, optional :: subset
    type(t_sub), pointer :: cs
    integer :: jb
    if (present(subset)) then
      cs => subset
    else
      cs => grid%in_domain
    end if
    do jb = cs%sblk, cs%eblk
      out(jb) = real(jb, 8)
    end do
  end subroutine compute
  subroutine driver_absent(grid, n, out)
    type(t_grid), intent(in) :: grid
    integer, intent(in) :: n
    real(8), intent(inout) :: out(:)
    call compute(grid, n, out)               ! optional OMITTED -> else branch
  end subroutine driver_absent
  subroutine driver_present(grid, n, out)
    type(t_grid), intent(in), target :: grid
    integer, intent(in) :: n
    real(8), intent(inout) :: out(:)
    call compute(grid, n, out, grid%in_domain)  ! optional PASSED -> then branch
  end subroutine driver_present
end module mo_cr
"""


@pytest.mark.parametrize("entry", ["driver_absent", "driver_present"])
def test_present_guarded_conditional_pointer_rebind(tmp_path: Path, entry: str):
    """Both fold directions: the conditional rebind collapses to the live
    branch and the kernel runs to the closed form ``out(jb)=jb`` over the
    selected subset range.  Without the fold the build fails outright with the
    ``interleaved rebind`` rejection."""
    sdfg = build_sdfg(_SRC, tmp_path / entry, name=entry, entry=f"mo_cr::{entry}").build()
    sdfg.validate()

    n, sblk, eblk = 8, 2, 5
    out = np.asfortranarray(np.zeros(n))
    sdfg(out=out, n=np.int32(n), grid_in_domain_sblk=np.int32(sblk), grid_in_domain_eblk=np.int32(eblk), out_d0=n)

    expected = np.zeros(n)
    expected[sblk - 1:eblk] = np.arange(sblk, eblk + 1, dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
