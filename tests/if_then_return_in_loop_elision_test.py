"""Minimal repro of the IF-block elision bug surfaced by NPB LU.

When a Fortran ``do`` loop body contains the pattern::

    do istep = 1, niter
      ... unconditional body ...
      if (<cond>) then
        ... writes to arr ...
      end if
      if (<other-cond>) return
    end do

the IF block wrapping the writes is SILENTLY ELIDED from the SDFG, so
the writes never run.  The unconditional body BEFORE the IF runs
correctly; the return AFTER the IF runs correctly; ONLY the IF body's
writes are dropped.

Discovered while bisecting NPB LU's residuals-don't-accumulate bug.
LU's ``ssor`` subroutine contains::

    do istep = 1, niter
      ... ssor body ...
      if (mod(istep, inorm) == 0 .or. istep == itmax) then
        call l2norm(... rsdnm)         ! this WRITES rsdnm per iter
      end if
      if (rsdnm(1) < tolrsd(1) .and. ...) return
    end do

The ``call l2norm`` was elided, so ``rsdnm`` was never updated per
iteration -- only the pre-loop initial ``l2norm`` value persisted.
This produced rsdnm[0] = 67877 vs reference 15158 (4.5x off) at
itmax=1 because rsdnm reflected the PRE-SSOR-sweep state.

Bisection showed the trigger is the MINIMAL pattern below -- any
two-IF combination (writes-in-IF + IF-return) inside a do-loop
triggers the elision.  None of these need to be true to trigger:

  * inorm/itmax sourced from module globals (LU's actual setup)
  * ``mod()`` intrinsic in the condition
  * ``.or.`` in the condition
  * multi-element AND in the return-condition
  * the IF body being a ``call`` (writes via a loop do work)

The return-condition's predicate can be ANYTHING (``istep > niter``,
``u(1) < 0.0``, ``rsdnm(1) < 0.0`` -- all reproduce).

Pipeline candidate suspects (from project_lu_dt_bug_session_handoff
documentation): the second ``lift-cf-to-scf`` pass, ``hlfir-inline-
all``, or the bridge AST emitter folding the IF during structurise.
``mlir::createCanonicalizerPass`` and ``mlir::createCSEPass`` are
NOT in DEFAULT_PIPELINE; no DaCe-level transformations applied.
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
    """For itmax=3: ssor runs 3 iterations.  At istep==itmax (=3), the IF
    body writes rsdnm(1) = u(1) * 1 = 3.0 (u(1) increments by 1 each
    iteration starting from 0).  If the IF block is elided, rsdnm stays
    at 0."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="dolu", entry="dolu").build()
    assert _run(sdfg, 3) == 3.0, "IF body elided: rsdnm(1) should be u(1)=3.0 at istep==itmax"
