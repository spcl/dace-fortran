"""Regression: a pointer rebound in BOTH arms of an ``IF (PRESENT(optional))`` guard
could not be lowered (``interleaved rebind`` rejection) -- ICON's ``_onBlock`` subset
idiom, second blocker for ``solve_free_sfc_ab_mimetic``.

``hlfir-rewrite-pointer-assigns`` models a rebind as a View of ONE source, so a rebind
to two mutually-exclusive targets was rejected.  Post ``hlfir-inline-all``,
``PRESENT(...)`` is compile-time constant per inlined copy; ``foldPresenceGuardedIfs``
resolves it and hoists the live branch, making the rebind straight-line (plain
``canonicalize`` can't fold ``fir.is_present`` behind ``hlfir.declare``).

Both fold directions covered: optional omitted (else branch) and passed (then branch)."""
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
    """Conditional rebind collapses to the live branch; kernel runs to the closed form
    ``out(jb)=jb`` over the selected range.  Without the fold, build fails outright."""
    sdfg = build_sdfg(_SRC, tmp_path / entry, name=entry, entry=f"mo_cr::{entry}").build()
    sdfg.validate()

    n, sblk, eblk = 8, 2, 5
    out = np.asfortranarray(np.zeros(n))
    sdfg(out=out, n=np.int32(n), grid_in_domain_sblk=np.int32(sblk), grid_in_domain_eblk=np.int32(eblk), out_d0=n)

    expected = np.zeros(n)
    expected[sblk - 1:eblk] = np.arange(sblk, eblk + 1, dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
