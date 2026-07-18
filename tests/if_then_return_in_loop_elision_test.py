"""Repro for an IF-block elision bug (bisected from NPB LU's ``ssor``): inside a do-loop, an
``if (<cond>) then ... end if`` followed by ``if (<other-cond>) return`` silently drops the IF body's writes."""
import numpy as np
import pytest

from _util import build_sdfg, have_flang

import dace.data

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """\
module m
  implicit none
  integer, parameter :: N = 4
  double precision :: u(N), rsdnm(5), tolrsd(5)
contains
  subroutine ssor(niter, itmax)
    integer, intent(in) :: niter, itmax
    integer :: istep, i, mm
    do istep = 1, niter
      do i = 1, N
        u(i) = u(i) + 1.0d0
      end do
      if (istep == itmax) then
        do mm = 1, 5
          rsdnm(mm) = u(1) * mm
        end do
      end if
      if (istep > niter) return
    end do
  end subroutine ssor

  subroutine dolu(itmax)
    integer, intent(in) :: itmax
    integer :: i, mm
    do i = 1, N
      u(i) = 0.0d0
    end do
    do mm = 1, 5
      tolrsd(mm) = 1.0d-16
    end do
    call ssor(itmax, itmax)
  end subroutine dolu
end module m
"""


def _run(sdfg, itmax_v: int):
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
    return float(kw['rsdnm'][0])


def test_if_then_return_in_loop_does_not_elide_if_body(tmp_path):
    """itmax=3: at istep==itmax the IF body writes rsdnm(1)=u(1)*1=3.0 (u(1) increments
    by 1/iter from 0).  If the IF block is elided, rsdnm stays 0."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="dolu", entry="m::dolu").build()
    assert _run(sdfg, 3) == 3.0, "IF body elided: rsdnm(1) should be u(1)=3.0 at istep==itmax"
