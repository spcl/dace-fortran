"""Minimal reproducer for a module-level scalar INPUT being dropped from
the SDFG signature.

Pattern: a module declares a scalar (``DOUBLE PRECISION :: x``), a
subroutine reads it, the caller pre-sets it before invocation.  The
bridge's classifier sees ``x`` as ``role='scalar', intent='inout'``,
so ``descriptors.add_descriptors`` registers it via
``sdfg.add_array(shape=(1,), transient=False)`` -- which would put it
on the kernel argument list.  By the time ``build`` returns however,
``x`` is gone from ``sdfg.arrays`` / ``sdfg.symbols`` / ``arglist()``.

The suspected mechanism is the upstream SCCP + symbol-dce sweep in
``DEFAULT_PIPELINE``: ``fir.global @_QMmEx`` has a BSS-zero
initializer, SCCP folds every load of it to ``0.0``, ``symbol-dce``
then drops the now-unused ``fir.address_of`` chain -- and the
declare goes with it.  The kernel runs with the constant-folded
zero, so the caller's pre-set value never reaches the body.

LU surfaces this as ``dt`` (the SSOR time step): the gfortran
reference behaviour is "user sets ``dt`` before calling ``dolu()``",
but the SDFG behaves as if ``dt = 0`` and the SSOR sweep is a
no-op.  This tiny reproducer exhibits the same gap on a 5-line
kernel.  Once the bridge stops constant-folding non-parameter
module globals (or, equivalently, promotes them into the arglist as
length-1 arrays before SCCP can see the BSS-zero init), the test
flips to PASS and the LU numerical correctness test follows.
"""
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

_SRC = """\
module m
  implicit none
  double precision :: x
contains
  subroutine doublex(y)
    double precision, intent(out) :: y
    y = x * 2.0d0
  end subroutine doublex
end module m

module mdriver
  use m, only: doublex
  implicit none
contains
  subroutine call_doublex(y)
    double precision, intent(out) :: y
    call doublex(y)
  end subroutine call_doublex
end module mdriver
"""

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_scalar_input_survives_sccp(tmp_path):
    """A module-level scalar pre-set by the caller must reach the
    kernel body intact -- not constant-folded to its BSS-zero init."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="call_doublex", entry="mdriver::call_doublex").build()
    assert 'x' in sdfg.arglist(), (f"module scalar ``x`` missing from SDFG arglist; "
                                   f"got {sorted(sdfg.arglist())}")
    y_buf = np.zeros(1, dtype=np.float64, order='F')
    kw = {'y': y_buf}
    # The bridge surfaces module scalars as either (1,)-Array (intent
    # inout) or true Scalar (intent in).  Accept either binding form.
    if 'x' in sdfg.arrays:
        kw['x'] = np.array([5.0], dtype=np.float64, order='F')
    else:
        kw['x'] = np.float64(5.0)
    sdfg(**kw)
    np.testing.assert_allclose(y_buf[0], 10.0, err_msg=f"y={y_buf[0]}; expected 2*x=10 "
                               "(x was pre-set to 5)")
