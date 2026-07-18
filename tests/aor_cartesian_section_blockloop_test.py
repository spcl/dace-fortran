"""xfail(strict, unvalidated): per-block 2-D AoR section read (``fld %
p_vn(:,:,blockno)`` -> inlined worker's ``vec_in(jc,jk)%x(k)``) drops the
blockno index -- the section peel (box_addr->copy_in) stops walkMemberChain /
expandDesignateChain before the record component, so ``vec_in_x`` renders as a
bare free symbol. Blocks Mode-1 ocean solve_free_sfc_ab_mimetic; fix needs a
section-peel hop in both walkers (mint side: extract_vars.cpp; access side:
elementals.cpp). Companion arg name/shape unconfirmed pending a bridge build.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

# Two inlining levels needed: ``mid`` passes a per-block 2-D SECTION of the WHOLE
# 3-D member to ``worker``, so after inlining the section's base is an inlined
# alias reached through ``copy_in`` -- the chain shape that defeats the walkers.
# Mirrors ICON-O's solve_free_sfc -> map_edges2cell/_onBlock reading
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
    """Correct lowering mints the rank-4 SoA companion ``fld_p_vn_x[jc,jk,blockno,k]``
    for the per-block AoR section read and matches the closed form bit-exactly."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="driver", entry="m::driver").build()
    sdfg.validate()

    nc, nl, nblk = 4, 3, 2
    rng = np.random.default_rng(0)

    # pre-gathered SoA companion the host-side AoS->SoA copy-in would hand the kernel.
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

    # out(jc,jk,k,jb) = p_vn(jc,jk,jb)%x(k)*2+k; SoA pvn_x[jc,jk,jb,k] moves k ahead of block.
    k_off = np.arange(1, 4, dtype=np.float64)[None, None, :, None]
    expected = np.transpose(pvn_x, (0, 1, 3, 2)) * 2.0 + k_off

    max_diff = float(np.abs(out - expected).max())
    assert max_diff == 0.0, f"AoR-section block-loop read not bit-exact: max_diff={max_diff}"
