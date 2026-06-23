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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="fill", entry="m::fill").build()
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="count_to_n", entry="m::count_to_n").build()
    out = np.full((N, ), -1, dtype=np.int32, order='F')
    sdfg(out_arr=out, n=np.int32(N))
    expected = np.arange(1, N + 1, dtype=np.int32) * 7
    np.testing.assert_array_equal(out, expected)


def test_do_while_whole_array_elementwise_in_body(tmp_path):
    """``a = a * 2`` (a whole-array ELEMENTWISE op) inside a ``do while``
    body must build + run.  Regression for the do-while miscompile:

    flang lowers ``do while`` to an UNSTRUCTURED CFG, which the bridge
    reconstructs via the ``scf.while`` body walker.  That walker's
    ``hlfir.assign`` handler routed every array<-array assign straight
    to ``buildCopyNode`` -- but a whole-array elementwise RHS is an
    ``hlfir.elemental`` (result type ``!hlfir.expr<?>``, so ``is_array``
    is true) and NOT a plain copy.  ``buildCopyNode`` then traced the
    elemental result to an empty name and emitted ``kind="copy"`` with
    ``reduce_src=""`` -> ``KeyError: ''`` at ``emit_copy``
    (``sdfg.arrays[""]``).  The structured ``do`` / ``if`` paths built
    it correctly as ``[loop][assign expr='(a*2)']``; only ``do while``
    miscompiled.  The walker now mirrors the structured dispatch and
    expands the elemental to the same map+assign.

    Counts i = 0,1,2 -> ``a`` doubled three times -> ``a * 8``."""
    src = """
module m
  implicit none
contains
  subroutine scale_loop(a, n)
    integer, intent(in) :: n
    real(kind=8), intent(inout) :: a(n)
    integer :: i
    i = 0
    do while (i < 3)
      a = a * 2.0d0
      i = i + 1
    end do
  end subroutine scale_loop
end module m
"""
    N = 4
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="scale_loop", entry="m::scale_loop").build()
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64, order='F')
    sdfg(a=a, n=np.int32(N))
    np.testing.assert_array_equal(a, np.array([1.0, 2.0, 3.0, 4.0]) * 8.0)


def test_do_while_whole_array_decrement_in_body(tmp_path):
    """``a = a - 1`` whole-array elementwise op in a ``do while`` whose
    condition reads the array (``sum(a) > 0``).  Second shape of the
    same miscompile -- the convergence-style loop where each iteration
    mutates the whole array the condition tests."""
    src = """
module m
  implicit none
contains
  subroutine drain(a, n)
    integer, intent(in) :: n
    real(kind=8), intent(inout) :: a(n)
    do while (sum(a) > 0.0d0)
      a = a - 1.0d0
    end do
  end subroutine drain
end module m
"""
    N = 3
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="drain", entry="m::drain").build()
    # start all 2.0 -> after iter1 all 1.0 (sum 3>0), iter2 all 0.0 (sum 0, stop)
    a = np.full((N, ), 2.0, dtype=np.float64, order='F')
    sdfg(a=a, n=np.int32(N))
    np.testing.assert_array_equal(a, np.zeros(N, dtype=np.float64))
