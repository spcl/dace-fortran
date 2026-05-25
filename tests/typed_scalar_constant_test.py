"""The builder generates typed scalar constants.

A read-only Fortran scalar that reaches the SDFG as a constant carries its
true precision: a ``real(4)`` parameter becomes a ``np.float32`` constant (not
a Python ``float`` widened to double), a ``real(8)`` becomes ``np.float64``,
and integers stay Python ``int``.  This keeps fp32 constants single precision
through code generation, building on DaCe's typed-constant support.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_SRC = """
module m
  implicit none
  real(4) :: cscale = 0.1
  real(8) :: dscale = 0.2d0
  integer :: icount = 7
contains
  subroutine apply(x, y)
    real(4), intent(in) :: x(2)
    real(4), intent(out) :: y(2)
    integer :: i
    do i = 1, 2
      y(i) = x(i) * cscale + real(dscale) + real(icount)
    end do
  end subroutine apply
end module m
"""


def _constants(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path, name="apply", entry="_QMmPapply").build()
    return {name: val[1] for name, val in sdfg.constants_prop.items()}


def test_fp32_scalar_constant_is_float32(tmp_path):
    consts = _constants(tmp_path)
    assert isinstance(consts["cscale"], np.float32)
    np.testing.assert_allclose(consts["cscale"], np.float32(0.1), rtol=0)


def test_fp64_and_int_scalar_constants_keep_their_types(tmp_path):
    consts = _constants(tmp_path)
    assert isinstance(consts["dscale"], np.float64)
    np.testing.assert_allclose(consts["dscale"], 0.2)
    # Integers stay Python ``int`` (round-trip- and isinstance(int)-friendly).
    assert type(consts["icount"]) is int and consts["icount"] == 7
