"""Reproducer: a module-level scalar pre-set by the caller (``x``) gets dropped from the SDFG
arglist. Suspected cause: upstream SCCP + symbol-dce in ``DEFAULT_PIPELINE`` folds
``fir.global @_QMmEx``'s BSS-zero init to a constant and DCEs the declare, so the kernel
always sees ``x = 0`` regardless of the caller's value. Same root cause as LU's ``dt``
(SSOR time step) bug -- fixing this flips both to PASS.
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
    """Module-level scalar pre-set by the caller must reach the kernel body intact, not constant-folded to its BSS-zero init."""
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="call_doublex", entry="mdriver::call_doublex").build()
    assert 'x' in sdfg.arglist(), (f"module scalar ``x`` missing from SDFG arglist; "
                                   f"got {sorted(sdfg.arglist())}")
    y_buf = np.zeros(1, dtype=np.float64, order='F')
    kw = {'y': y_buf}
    # module scalars surface as either (1,)-Array (intent inout) or Scalar (intent in) -- accept either.
    if 'x' in sdfg.arrays:
        kw['x'] = np.array([5.0], dtype=np.float64, order='F')
    else:
        kw['x'] = np.float64(5.0)
    sdfg(**kw)
    np.testing.assert_allclose(y_buf[0], 10.0, err_msg=f"y={y_buf[0]}; expected 2*x=10 "
                               "(x was pre-set to 5)")
