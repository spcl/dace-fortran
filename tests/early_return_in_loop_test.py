"""End-to-end test for the early-return-inside-loop pattern.

NPB LU's ``ssor`` has an early ``return`` inside a counted loop. After
``lift-cf-to-scf`` structurises it into an ``scf.while``, every nested
``fir.do_loop`` write in the BEFORE region must survive -- the bridge's
``walkSCFBeforeRegion`` (``bridge/ast/dispatch.cpp``) used to drop such loops,
silently discarding their assigns (LU symptom: SSOR sweep a no-op, ~6 orders
of magnitude divergence). Reproducer: four ``d(M1,M2,i,j)`` assigns guarded
by an early return in one outer loop.
"""
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  double precision :: dt, residual
  integer :: nx, ny
  double precision :: d(5,5,33,33)
contains
  subroutine foo(niter)
    integer, intent(in) :: niter
    integer :: i, j, it
    do it = 1, niter
      do j = 1, ny
        do i = 1, nx
          d(1,1,i,j) = 1.0 + dt * 2.0
          d(1,2,i,j) = 0.0
          d(2,1,i,j) = -dt * 2.0
          d(2,2,i,j) = 1.0 + dt * 4.0
        end do
      end do
      if (residual < 1.0d-8) then
        return
      end if
    end do
  end subroutine foo

  subroutine bar()
    call foo(50)
  end subroutine bar
end module m
"""


def _count_assigns_to(builder, target_name: str) -> int:
    """Walk the bridge's AST and count assign nodes with the given target."""
    ast = builder.module.get_ast()
    total = 0

    def walk(nodes):
        nonlocal total
        for n in nodes:
            if n.kind == 'assign' and str(getattr(n, 'target', '')) == target_name:
                total += 1
            walk(list(getattr(n, 'children', [])))

    walk(ast)
    return total


def test_early_return_preserves_loop_body_assigns(tmp_path):
    """AST extractor must walk into ``fir.do_loop`` ops inside an ``scf.while``
    BEFORE region -- otherwise assigns in a loop with an early ``return``
    silently vanish. Asserts all four ``d(M1,M2,i,j)`` writes survive."""
    builder = build_sdfg(_SRC, tmp_path / "sdfg", name="bar", entry="m::bar")
    n_d = _count_assigns_to(builder, 'd')
    assert n_d >= 4, (f"expected at least 4 ``d`` assigns to reach the AST; got {n_d}. "
                      f"The bridge's ``walkSCFBeforeRegion`` likely skipped a "
                      f"``fir.do_loop`` op inside the lifted ``scf.while`` before "
                      f"region (see ``bridge/ast/dispatch.cpp``).")
