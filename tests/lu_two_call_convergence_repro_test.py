"""Minimal repro of NPB LU's "SSOR sweep no-op" bug isolated.

The full LU SDFG produces residuals ~6.79e4 regardless of itmax (1, 5,
or 50).  Reference progression is 15159 -> 2087 -> 0.016 for itmax=1,
5, 50 -- so the body should accumulate across iterations but doesn't.

This repro distils the trigger down to ~30 lines of Fortran:

  * Two sequential subroutine calls (``ssor(1)`` then ``ssor(itmax)``);
  * Inside each: ``do istep = 1, niter`` with body + early-return
    on a MULTI-ELEMENT AND convergence check that never fires;
  * Module-level arrays for the convergence operands.

With this shape, ``itmax`` EVEN values produce results as if only ONE
iteration ran (the literal ``ssor(1)``'s iteration), while ODD values
produce the correct total iteration count.  Specifically:

  * itmax=1: u[0]=2 (=ssor(1) + ssor(1) =2 iters)  -- CORRECT
  * itmax=2: u[0]=2 (expected 3, only 1 iter of ssor(itmax) ran)
  * itmax=3: u[0]=4 (=2+ssor(itmax=3) =4 iters)    -- CORRECT
  * itmax=4: u[0]=2 (expected 5)                   -- WRONG
  * itmax=5: u[0]=6                                 -- CORRECT

The bug requires ALL three factors -- removing any reproduces correct
behaviour:

  1. Two sequential calls (single call works for all niter).
  2. Multi-element AND convergence check (1-element / no convergence
     check works for all itmax).
  3. The first call uses a LITERAL constant (``ssor(1)``).  Module-level
     scalars + early-return alone don't trigger it.

The pattern matches LU's ``dolu``::

    call ssor(1)        ! ALWAYS 1 iteration
    call ssor(itmax)    ! ITMAX iterations

with the multi-element convergence inside the ssor body matching LU's
``if (rsdnm(1) < tolrsd(1) .and. ... .and. rsdnm(5) < tolrsd(5))
return`` shape.

The minimal repro lets us iterate on the bridge fix without the full
LU SDFG's 1041-state complexity slowing down each diagnostic cycle.
"""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

import dace.data

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  integer, parameter :: N = 4
  double precision :: u(N), rsdnm(2), tolrsd(2)
contains
  subroutine ssor(niter)
    integer, intent(in) :: niter
    integer :: istep, i, m
    do istep = 1, niter
      do i = 1, N
        u(i) = u(i) + 1.0d0
      end do
      do m = 1, 2
        rsdnm(m) = 1.0d0
      end do
      if (rsdnm(1) < tolrsd(1) .and. rsdnm(2) < tolrsd(2)) return
    end do
  end subroutine ssor

  subroutine dolu(itmax)
    integer, intent(in) :: itmax
    integer :: i, m
    do i = 1, N
      u(i) = 0.0d0
    end do
    do m = 1, 2
      tolrsd(m) = 1.0d-16
    end do
    call ssor(1)
    call ssor(itmax)
  end subroutine dolu
end module m
"""


def _run(sdfg, itmax_v: int) -> float:
    """Build a fresh kw dict, invoke, return ``u[0]``."""
    kw = {}
    for nm, desc in sdfg.arglist().items():
        is_scalar = isinstance(desc, dace.data.Scalar)
        is_int = 'int' in str(desc.dtype).lower()
        if is_scalar:
            kw[nm] = np.int32(itmax_v) if nm == 'itmax' else (np.int32(0) if is_int else np.float64(0))
        else:
            shape = tuple(int(s) for s in desc.shape)
            kw[nm] = np.zeros(shape, dtype=(np.int32 if is_int else np.float64), order='F')
    sdfg(**kw)
    return float(kw['u'][0])


def test_lu_two_call_convergence_repro_itmax_2(tmp_path):
    """itmax=2 must give u[0] = 1 + 2 = 3 (2 iterations of ssor(itmax)
    on top of ssor(1)'s 1 iteration)."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="dolu", entry="_QMmPdolu").build()
    assert _run(sdfg, 2) == 3.0, "itmax=2 should give u[0]=3, not 2 (one iter only)"
