"""The builder surfaces read-only module-level scalars with the right
dtype on the SDFG arglist.

After ``hlfir-preserve-mutable-globals`` runs, every non-PARAMETER,
non-written module-level global is a caller kwarg on the SDFG -- a
non-transient length-1 ``Array`` whose dtype reflects the source-
declared Fortran KIND.  A ``real(4)`` global lands as ``np.float32``,
a ``real(8)`` as ``np.float64``, an ``integer`` as ``np.int32``.
PARAMETER constants stay baked in the constant pool with the same
dtype rule; that path is exercised separately by
``module_global_vs_constant_test.py::test_parameter_is_baked_constant``.
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


def _arg_dtype(tmp_path, name: str):
    """Resolve the SDFG arglist entry for ``name`` and return its
    numpy dtype.  Verifies the symbol actually surfaced as a kwarg."""
    sdfg = build_sdfg(_SRC, tmp_path, name="apply", entry="_QMmPapply").build()
    assert name in sdfg.arglist(), f"{name!r} not in arglist; got {sorted(sdfg.arglist())}"
    return sdfg.arglist()[name].dtype.as_numpy_dtype()


def test_fp32_scalar_constant_is_float32(tmp_path):
    """A ``real(4)`` module-level global surfaces as a float32 kwarg."""
    assert _arg_dtype(tmp_path, "cscale") == np.float32


def test_fp64_and_int_scalar_constants_keep_their_types(tmp_path):
    """``real(8)`` -> float64; ``integer`` -> int32 (Fortran default
    integer KIND = int32 on every platform we target)."""
    assert _arg_dtype(tmp_path, "dscale") == np.float64
    assert _arg_dtype(tmp_path, "icount") == np.int32
