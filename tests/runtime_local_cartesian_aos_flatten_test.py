"""Regression: a LOCAL, RUNTIME-SIZED array-of-struct with a static-array leaf
member (the ICON-O ``t_cartesian_coordinates :: x(3)`` shape) must flatten to a
multi-dim SoA companion ``p_x(n, nb, 3)`` -- even though the outer extents are
only known at runtime.

This is the third blocker that stopped the ocean dynamical core
(``solve_free_sfc_ab_mimetic``) from lowering.  ``veloc_diff_biharmonic_curl_curl``
declares a LOCAL ``TYPE(t_cartesian_coordinates) :: p_nabla2_dual(nblks, ...)``
and does the manual AoS<->SoA copy ``ox(:,jb) = cc(:,jb)%x(1)``.  The struct is
sized from runtime block counts, so ``FlattenStructs::isLocallyFlattenable``
bailed at its "static shape only" gate (a companion ``fir.alloca`` over a
dynamic pointee, or a static-literal ``fir.shape`` baking ``-1`` for the unknown
dims, is verifier-invalid).  With the local AoS never flattened, ``cc(:,jb)%x(1)``
had no flat companion to read and the copy surfaced downstream as a phantom
section-to-section assign (``p_nabla2_dual_x = p_nabla2_dual``).

The fix threads the declare's live outer extents through both the companion
alloca and a fresh ``fir.shape`` (``outerExtentValues`` + ``makeCompanionAlloca``
in ``FlattenStructs.cpp``); the fully-static path is unchanged.  Once flattened,
``p(i,jb)%x(k)`` resolves to ``p_x(i,jb,k)`` and the copy lowers as ordinary
array indexing.

Closed form: each element writes ``x = [s, 2s, 3s]`` for ``s = src(i,jb)`` and
reads back ``sum(x) = 6*s``, so ``out = 6*src``.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module lib
  implicit none
  type cc
    real(8) :: x(3)
  end type cc
contains
  subroutine kern(n, nb, src, out)
    integer, intent(in) :: n, nb
    real(8), intent(in) :: src(n, nb)
    real(8), intent(out) :: out(n, nb)
    type(cc) :: p(n, nb)          ! LOCAL, runtime-sized AoS
    integer :: i, jb
    do jb = 1, nb
      do i = 1, n
        p(i, jb) % x(1) = src(i, jb)
        p(i, jb) % x(2) = 2.0d0 * src(i, jb)
        p(i, jb) % x(3) = 3.0d0 * src(i, jb)
      end do
    end do
    do jb = 1, nb
      do i = 1, n
        out(i, jb) = p(i, jb) % x(1) + p(i, jb) % x(2) + p(i, jb) % x(3)
      end do
    end do
  end subroutine kern
end module lib
"""


def test_runtime_local_cartesian_aos_flattens_to_soa(tmp_path: Path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="kern", entry="lib::kern").build()
    sdfg.validate()

    # The local AoS flattened to a SoA companion; trailing dim is the member
    # extent (3), the leading dims are the runtime outer extents.
    assert "p_x" in sdfg.arrays, f"missing SoA companion; arrays={sorted(sdfg.arrays)}"
    shp = sdfg.arrays["p_x"].shape
    assert len(shp) == 3, shp
    assert int(shp[-1]) == 3, shp

    n, nb = 4, 3
    rng = np.random.default_rng(0)
    src = np.asfortranarray(rng.standard_normal((n, nb)))
    out = np.asfortranarray(np.zeros((n, nb)))
    sdfg(src=src, out=out, n=np.int32(n), nb=np.int32(nb), src_d0=n, src_d1=nb, out_d0=n, out_d1=nb)

    np.testing.assert_allclose(out, 6.0 * src, rtol=1e-12, atol=1e-12)
