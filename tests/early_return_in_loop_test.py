"""End-to-end test for the early-return-inside-loop pattern.

NPB LU's ``ssor`` runs a fixed-point iteration with an early ``return``
on residual convergence:

    do istep = 1, niter
        ! ... many writes to ``d``, ``a``, ``b``, ``c`` arrays ...
        if (rsdnm(1) < tolrsd(1) .and. ...) then
            return
        end if
    end do

When ``lift-cf-to-scf`` structurises the early return, the
``do istep`` loop becomes an ``scf.while``.  All four levels of
nested ``fir.do_loop`` writes -- including the ``d(M1,M2,i,j) = ...``
assignments that drive the SSOR solver -- end up inside the
``scf.while``'s BEFORE region.

The bridge's ``walkSCFBeforeRegion`` (``bridge/ast/dispatch.cpp``)
recognises ``scf.if`` / ``scf.condition`` / ``scf.index_switch`` /
``hlfir.assign`` / ``fir.store`` ops in the before region, but
nothing else falls through as "pure-value ops, no AST node, their
values flow inline."  A ``fir.do_loop`` parked inside the before
region is therefore dropped on the floor, taking every assign in
its body with it.

LU surfaces the bug as: ``dt`` (and the entire jacld d-build) is
absent from the SDFG, the SSOR sweep is a no-op, and the
numerical-correctness comparison against the gfortran reference
diverges by ~6 orders of magnitude on every residual component.
The minimal reproducer below distils the same shape down to four
``d(M1,M2,i,j) = expr`` assigns guarded by a single early
``return`` inside one outer ``do it`` loop.
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
    """The bridge's AST extractor must walk into ``fir.do_loop`` ops
    inside an ``scf.while`` BEFORE region -- otherwise every assign
    in a Fortran ``do`` whose containing loop has an early ``return``
    silently disappears from the SDFG.

    The reproducer's ``do it`` body writes ``d(M1,M2,i,j)`` four
    times per (i,j) pair.  After ``lift-cf-to-scf`` lowers the
    ``if (...) return`` into a structured break, the inner i / j
    loops live inside the scf.while BEFORE region.  We assert all
    four ``d`` writes survive into the AST.
    """
    builder = build_sdfg(_SRC, tmp_path / "sdfg", name="bar", entry="m::bar")
    n_d = _count_assigns_to(builder, 'd')
    assert n_d >= 4, (f"expected at least 4 ``d`` assigns to reach the AST; got {n_d}. "
                      f"The bridge's ``walkSCFBeforeRegion`` likely skipped a "
                      f"``fir.do_loop`` op inside the lifted ``scf.while`` before "
                      f"region (see ``bridge/ast/dispatch.cpp``).")
