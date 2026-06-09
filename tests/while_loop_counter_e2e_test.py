"""End-to-end test for the Fortran ``do counter = 1, n`` loop with an
early ``return`` -- exactly the shape of NPB LU's ``ssor`` istep loop.

The bridge processes this as ``lift-cf-to-scf`` lowers the early
return into an ``scf.while`` whose BEFORE region holds the body +
condition check + counter increment.  At AST extraction the
``walkSCFBeforeRegion`` correctly emits the counter assigns (init +
increment) and the SDFG-builder routes them through interstate edges
since the counter is classified as a SYMBOL.

This test pins the contract: after ``n`` calls to the subroutine the
counter must reach ``n`` (not stay at ``1``), proving the increment
actually runs.  The LU SSOR sweep being a no-op at the time this
test was written suggests the counter doesn't iterate correctly.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_do_loop_with_early_return_counts_iterations(tmp_path):
    """``do counter = 1, n; arr(counter) = real(counter); end do`` -- the
    counter must increment all the way to ``n`` so every slot of ``arr``
    gets filled.  If the increment store doesn't reach the SDFG, only
    ``arr(1)`` would be written and the rest stay at the sentinel."""
    src = """
module m
  implicit none
contains
  subroutine fill(arr, n)
    integer, intent(in) :: n
    double precision, intent(inout) :: arr(n)
    integer :: counter
    do counter = 1, n
      arr(counter) = real(counter, kind=8)
      if (counter == n + 1) return    ! never fires, but forces scf.while lowering
    end do
  end subroutine fill
end module m
"""
    N = 8
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="_QMmPfill").build()
    arr = np.full((N, ), -1.0, dtype=np.float64, order='F')
    sdfg(arr=arr, n=np.int32(N))
    expected = np.arange(1, N + 1, dtype=np.float64)
    np.testing.assert_array_equal(arr, expected)


def test_do_while_with_break_on_convergence(tmp_path):
    """``do; if (cond) exit; counter = counter + 1; end do`` -- the
    bridge models this as ``scf.while True`` with a ``break`` child
    inside the body and an explicit counter increment.  The test pins
    that the counter reaches the loop limit before the break fires."""
    src = """
module m
  implicit none
contains
  subroutine count_to_n(out_arr, n)
    integer, intent(in) :: n
    integer, intent(inout) :: out_arr(n)
    integer :: counter
    counter = 1
    do
      out_arr(counter) = counter * 7
      if (counter == n) exit
      counter = counter + 1
    end do
  end subroutine count_to_n
end module m
"""
    N = 6
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="count_to_n", entry="_QMmPcount_to_n").build()
    out = np.full((N, ), -1, dtype=np.int32, order='F')
    sdfg(out_arr=out, n=np.int32(N))
    expected = np.arange(1, N + 1, dtype=np.int32) * 7
    np.testing.assert_array_equal(out, expected)
