"""Regression (impl PENDING): a per-block 2-D SECTION of a 3-D POINTER
array-of-``t_cartesian_coordinates`` member, passed to an inlined worker that
reads it component-wise (``vec_in(jc,jk) % x(k)``), loses its fixed block index.

This is the ``vec_in_x`` blocker -- the last hard residual free symbol of the
Mode-1 ocean ``solve_free_sfc_ab_mimetic`` dynamical core.  It is the AoR twin
of ``section_of_pointer_member_inlined_test`` (a PLAIN-real 3-D pointer member),
here on an ARRAY-OF-RECORDS member whose element carries ``x(3)``:

  ``fld % p_vn`` is a ``TYPE(t_cartesian_coordinates), POINTER :: (:,:,:)``
  member; an outer routine loops over blocks and calls a worker with the
  per-block 2-D slice ``p_vn(:, :, blockno)``; the worker reads
  ``vec_in(jc, jk) % x(k)``.

After ``hlfir-inline-all`` the worker's ``(jc,jk)%x(k)`` designate reads through
``vec_in`` -- an inlined ``dummy_scope`` alias over a ``hlfir.copy_in`` of the
2-D SECTION.  The section's memref peels (``box_addr`` -> ``copy_in``) to a PURE
section designate (indices, NO component), so ``leadsToComponentDesignate`` is
false and all three back-walkers (``rootedAtStructDummy`` / ``walkMemberChain`` /
``traceToDecl``) plus ``expandDesignateChain`` STOP at the ``vec_in`` declare.
The read then renders as a bare, unbacked ``vec_in_x`` free symbol (no AoS->SoA
marshalling minted) and the ``blockno`` section dim is dropped from the access.

The fix (two parts; see the design note in the session handoff):

  (1) MINT side -- extend ``walkMemberChain`` (extract_vars.cpp) with a
      section-peel hop: when the inlined dummy's memref peels through
      ``box_addr``/``copy_in`` to a PURE section designate that itself leads to a
      component designate, walk THROUGH the section into the ``p_vn`` component
      and root at the struct dummy, counting the section's fixed SCALAR dims
      (``blockno``) as record subscripts.  Yields
      ``aos_origin_struct = fld % p_vn``, ``aos_member_path = "x"``,
      ``aos_outer_rank = 3``; the EXISTING ``_render_aos_copy_in`` static-value-
      member path (the ``p_vn_dual(:,:,:)%x`` gate-#12c machinery) then gathers a
      rank-4 ``[d0, d1, nblocks, 3]`` SoA companion ``fld_p_vn_x`` -- reused, no
      new marshaller.

  (2) ACCESS side (the genuinely new capability) -- ``expandDesignateChain``
      (bridge/ast/elementals.cpp) currently yields ``[jc, jk, k]`` (rank-3,
      ``blockno`` dropped) because its parent-walk breaks at the ``vec_in``
      declare.  The same section-peel hop must run there, and the triplet
      composition must INSERT the section's fixed scalar dim into the record-
      index position AND append the trailing member (``%x``) dim ->
      ``[jc, jk, blockno, k]`` over the rank-4 companion.

The single narrow shared gate (inlined dummy -> box peel -> pure section ->
component designate) keys on structure, not on any member name.

Reference: f2py's crackfortran cannot wrap the derived-type dummy, so the SDFG
is called directly with the pre-gathered SoA buffer (exactly what the host-side
AoS->SoA copy-in would hand the kernel) and compared against the closed form.
Distinct random data per (jc, jk, blockno, k) makes a dropped/permuted
``blockno`` or ``k`` index a value mismatch, not just a rank crash.

The test mirrors ``section_of_pointer_member_inlined_test`` but is marked
``xfail(strict=True)``: it FAILS today (bare ``vec_in_x`` -> unresolved arg /
rank mismatch / wrong block) and flips to XPASS once the two-part fix lands.

NOTE: authored read-only, WITHOUT a bridge build.  It is UNVALIDATED pending a
build -- the exact companion arg name (``fld_p_vn_x``) and its shape symbols
(``fld_p_vn_x_d0/_d1/_d2``) follow the proven ``diag_pvd_x`` / ``coeff_gc``
conventions of the sibling AoS-cartesian and section-alias tests, but must be
re-confirmed against the built SDFG's signature when the fix is implemented.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Two inlining levels are essential (as in the plain-real analog): ``mid`` takes
# the WHOLE 3-D member and passes a per-block 2-D SECTION to ``worker``.  After
# inlining, the section's base is ``mid``'s dummy -- an inlined ALIAS of the
# member, reached through ``copy_in`` -- which is the chain shape that defeats
# the walkers.  This mirrors ICON-O's
# ``solve_free_sfc -> ... -> map_edges2cell / _onBlock`` nest reading
# ``p_diag % p_vn(:, :, blockno)`` element-wise as ``vec_in(jc,jk) % x(k)``.
_SRC = """
module m
  implicit none
  type t_cc
    real(8) :: x(3)                                   ! cartesian coord element
  end type t_cc
  type t_field
    type(t_cc), pointer :: p_vn(:,:,:)                ! (nc, nl, nblk) AoR member
  end type t_field
contains
  subroutine worker(vec_in, nc, nl, o)
    type(t_cc), intent(in) :: vec_in(:,:)             ! per-block 2-D AoR slice
    integer, intent(in) :: nc, nl
    real(8), intent(inout) :: o(:,:,:)                ! (nc, nl, 3)
    integer :: jc, jk, k
    do jk = 1, nl
      do jc = 1, nc
        do k = 1, 3
          o(jc, jk, k) = vec_in(jc, jk) % x(k) * 2.0d0 + real(k, 8)
        end do
      end do
    end do
  end subroutine worker
  subroutine mid(pvn3d, nc, nl, nblk, out)
    type(t_cc), intent(in) :: pvn3d(:,:,:)            ! the WHOLE 3-D member
    integer, intent(in) :: nc, nl, nblk
    real(8), intent(inout) :: out(:,:,:,:)            ! (nc, nl, 3, nblk)
    integer :: jb
    do jb = 1, nblk
      call worker(pvn3d(:, :, jb), nc, nl, out(:, :, :, jb))
    end do
  end subroutine mid
  subroutine driver(fld, nc, nl, nblk, out)
    type(t_field), intent(in) :: fld
    integer, intent(in) :: nc, nl, nblk
    real(8), intent(inout) :: out(:,:,:,:)            ! (nc, nl, 3, nblk)
    call mid(fld % p_vn, nc, nl, nblk, out)
  end subroutine driver
end module m
"""


def test_aor_cartesian_section_blockloop_threads_block_index(tmp_path: Path):
    """Build the ``p_diag % p_vn(:,:,blockno)`` per-block AoR-section read and run
    it: today the walkers stop at the section and the block index is dropped
    (bare ``vec_in_x``); the correct lowering mints the rank-4 SoA companion
    ``fld_p_vn_x[jc, jk, blockno, k]`` and matches the closed form bit-exactly."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    sdfg.validate()

    nc, nl, nblk = 4, 3, 2
    rng = np.random.default_rng(0)

    # The pre-gathered SoA companion: what the host-side AoS->SoA copy-in of
    # ``fld % p_vn(:)%x`` hands the kernel -- shape [nc, nl, nblk, 3].
    pvn_x = np.asfortranarray(rng.random((nc, nl, nblk, 3)))
    out = np.asfortranarray(np.zeros((nc, nl, 3, nblk)))

    sdfg(fld_p_vn_x=pvn_x,
         out=out,
         nc=np.int32(nc),
         nl=np.int32(nl),
         nblk=np.int32(nblk),
         fld_p_vn_x_d0=nc,
         fld_p_vn_x_d1=nl,
         fld_p_vn_x_d2=nblk,
         out_d0=nc,
         out_d1=nl,
         out_d2=3,
         out_d3=nblk)

    # out(jc, jk, k, jb) = p_vn(jc, jk, jb) % x(k) * 2 + k   (k = 1..3, Fortran)
    # SoA layout pvn_x[jc, jk, jb, k]; output moves k ahead of the block dim.
    k_off = np.arange(1, 4, dtype=np.float64)[None, None, :, None]
    expected = np.transpose(pvn_x, (0, 1, 3, 2)) * 2.0 + k_off

    max_diff = float(np.abs(out - expected).max())
    assert max_diff == 0.0, f"AoR-section block-loop read not bit-exact: max_diff={max_diff}"
