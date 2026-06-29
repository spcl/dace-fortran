"""Regression: a 2-D SECTION of a 3-D POINTER struct member, passed to an
inlined worker that indexes it element-wise, lost its fixed block index.

This is the ubiquitous ICON ``_onBlock`` idiom and the concrete blocker that
stopped the ocean dynamical core (``solve_free_sfc_ab_mimetic``) from lowering:

  ``op_coeffs%grad_coeff`` is a ``REAL(8), POINTER :: (:,:,:)`` member; an outer
  routine loops over blocks and calls a worker with the per-block 2-D slice
  ``grad_coeff(:, :, blockno)``; the worker indexes it ``grad_coeff(i, k)``.

``hlfir-flatten-structs`` splits the struct dummy into a flat rank-3 companion
``op_coeffs_grad_coeff``.  After ``hlfir-inline-all`` the worker's ``(i, k)``
designate reads through the section alias, but both the chain walker and the
alias-prefix walk bail on the section's ``(:, :)`` triplets -- which used to
drop the fixed ``blockno`` entirely and leave a rank-mismatched
``companion(i, k)`` designate (2 indices over a rank-3 array), crashing the
``hlfir.designate`` verifier with ``indices number must match memref rank``.

``rewriteSectionedAliasLeaf`` now composes the section positionally: each
full-range unit-stride triplet dim is filled by the next worker index, each
scalar dim (``blockno``) is kept -- yielding ``companion(i, k, blockno)``.

f2py's crackfortran can't wrap the derived-type dummy, so the reference is the
exact closed form.  The flattened struct dummy lowers to a plain rank-3 array
arg (the companion), so the built SDFG is called directly with that array; a
dropped/wrong block index either crashes the build (rank mismatch) or reads the
wrong block (caught by per-block-distinct data).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Two inlining levels are essential to reproduce the ocean blocker: ``mid``
# takes the WHOLE 3-D member and passes a per-block 2-D SECTION to ``worker``.
# After inlining, the section's base is ``mid``'s dummy -- an inlined ALIAS of
# the flat companion, not the companion directly -- which is the chain shape
# (``leaf -> section -> alias-declare -> companion``) that defeats the chain
# walker.  A single level (struct -> worker) is handled by the generic path and
# does NOT reproduce the bug.  This mirrors ICON's
# ``solve_free_sfc -> ... -> grad_fd_norm_oce_3d -> grad_fd_norm_oce_3d_onBlock``.
_SRC = """
module mo_sec
  implicit none
  type t_coeff
    real(8), pointer :: gc(:,:,:)        ! (n, 2, nblk) pointer member
  end type
contains
  subroutine worker(gc2d, n, o)
    real(8), intent(in) :: gc2d(:,:)     ! the per-block 2-D slice
    integer, intent(in) :: n
    real(8), intent(inout) :: o(:)
    integer :: i
    do i = 1, n
      o(i) = gc2d(i, 1) * 10.0d0 + gc2d(i, 2)
    end do
  end subroutine worker
  subroutine mid(gc3d, n, nblk, out)
    real(8), intent(in) :: gc3d(:,:,:)   ! the WHOLE 3-D member
    integer, intent(in) :: n, nblk
    real(8), intent(inout) :: out(:,:)
    integer :: jb
    do jb = 1, nblk
      call worker(gc3d(:, :, jb), n, out(:, jb))
    end do
  end subroutine mid
  subroutine driver(coeff, n, nblk, out)
    type(t_coeff), intent(in) :: coeff
    integer, intent(in) :: n, nblk
    real(8), intent(inout) :: out(:,:)   ! (n, nblk)
    call mid(coeff%gc, n, nblk, out)
  end subroutine driver
end module mo_sec
"""


def test_section_of_3d_pointer_member_inlined(tmp_path: Path):
    """Build the ``op_coeffs%grad_coeff(:,:,blk)`` shape and run it: a broken
    section composition crashes the build (rank mismatch) or reads the wrong
    block; the correct ``companion(i, k, blk)`` matches the closed form."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="driver", entry="mo_sec::driver").build()
    sdfg.validate()

    n, nblk = 5, 3
    rng = np.random.default_rng(0)
    gc = np.asfortranarray(rng.random((n, 2, nblk)))  # the flat companion
    out = np.asfortranarray(np.zeros((n, nblk)))

    sdfg(coeff_gc=gc,
         out=out,
         n=np.int32(n),
         nblk=np.int32(nblk),
         coeff_gc_d0=n,
         coeff_gc_d1=2,
         coeff_gc_d2=nblk,
         out_d0=n,
         out_d1=nblk)

    expected = gc[:, 0, :] * 10.0 + gc[:, 1, :]
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
