"""Minimal repro of NPB LU's "SSOR sweep no-op" bug: the full LU SDFG produces
residuals ~6.79e4 regardless of itmax, instead of the reference's 15159->2087->0.016
progression for itmax=1,5,50.

Trigger (all three required, removing any fixes it): (1) two sequential calls
(``ssor(1)`` then ``ssor(itmax)``); (2) a multi-element AND convergence check inside
the loop that never fires; (3) the first call uses a LITERAL constant (``ssor(1)``).
With this shape, EVEN itmax values produce results as if only ``ssor(1)``'s single
iteration ran; ODD values give the correct total.

Matches LU's ``dolu``: ``call ssor(1)`` (always 1 iter) then ``call ssor(itmax)``, with
the multi-element convergence matching LU's ``rsdnm(1)<tolrsd(1) .and. ...`` shape.
Isolates the bug from LU's full 1041-state SDFG for fast iteration."""
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
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="dolu", entry="m::dolu").build()
    assert _run(sdfg, 2) == 3.0, "itmax=2 should give u[0]=3, not 2 (one iter only)"
