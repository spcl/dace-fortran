"""End-to-end numerical correctness for the assumed-rank dummy +
``SELECT RANK`` pattern (Fortran 2018, ``DIMENSION(..)``).

The bridge handles this via the ``hlfir-fold-assumed-rank-queries``
pass.  When the caller's actual reaches the callee through a
``fir.convert : box<array<?x?xT>> -> box<array<*:T>>`` rank-erase,
the pass folds:

  * ``fir.box_rank %erased`` -> ``arith.constant <N>`` (the static rank)
  * ``fir.is_assumed_size %erased`` -> ``arith.constant false``

Canonicalize then reduces the ``SELECT RANK`` dispatch chain (modelled
as ``arith.cmpi`` + nested ``scf.if`` + ``fir.select_case``) to a
single live branch -- the one matching the actual's static rank.  The
non-matching branches' ``hlfir.declare`` ops disappear with the dead
code, so the bridge sees exactly one declare per assumed-rank dummy.

This test asserts the round-trip writes through the rank-2 branch
land at the right offsets in the actual.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  integer, parameter :: N = 4, M = 3
  double precision :: arr2d(N, M)
contains
  subroutine inner(a)
    double precision, intent(inout) :: a(..)
    integer :: i, j
    select rank (a)
    rank (1)
      do i = 1, size(a)
        a(i) = real(i, kind=8)
      end do
    rank (2)
      ! ``size(a, 1)`` / ``size(a, 2)`` lowers to ``fir.box_dims``;
      ! the bridge's fold pass replaces it with the concrete extents
      ! when the box traces back to a static-shape actual.
      do j = 1, size(a, 2)
        do i = 1, size(a, 1)
          a(i, j) = real(i + 10 * j, kind=8)
        end do
      end do
    rank default
    end select
  end subroutine inner

  subroutine outer()
    call inner(arr2d)
  end subroutine outer
end module m
"""


def test_assumed_rank_dispatches_to_rank2_branch(tmp_path):
    """The rank-2 branch of ``SELECT RANK`` must run when the caller
    passes a 2D array.  Verify by checking each ``a(i, j) = i + 10*j``
    write landed at the right slot.  The ``hlfir-fold-assumed-rank-
    queries`` pass eliminates the rank-1 and default branches, leaving
    only the rank-2 dispatch live for the canonicalizer + bridge AST
    extractor."""
    N, M = 4, 3
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="outer", entry="_QMmPouter").build()

    arr_sdfg = np.full((N, M), -1.0, dtype=np.float64, order='F')
    sdfg(arr2d=arr_sdfg)

    expected = np.empty((N, M), dtype=np.float64, order='F')
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            expected[i - 1, j - 1] = i + 10 * j

    np.testing.assert_array_equal(arr_sdfg, expected)
