"""Port of f2dace/dev:tests/fortran/non-interactive/pointers_test.py: a TARGET derived-type's rank-3
member s%w is rebound through a contiguous pointer p_area => s%w and read. Unlike the original
build-only port, this DRIVES the compiled SDFG and compares BIT-EXACT against gfortran/f2py plus the
closed form (s%w(1,1,1)=5.5; lout(1)=p_area(1,1,1)+lon(1); rest 0)."""

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_pointer_test(tmp_path):
    src = """
subroutine main(lon, lout)
  real, intent(in) :: lon(10)
  real, intent(out) :: lout(10)
  type simple_type
    real:: w(5, 5, 5), z(5)
    integer:: a
  end type simple_type
  type(simple_type), target :: s
  real :: area
  real, pointer, contiguous :: p_area(:, :, :)
  integer :: i, j
  s%w(1, 1, 1) = 5.5
  lout(:) = 0.0
  p_area => s%w
  lout(1) = p_area(1, 1, 1) + lon(1)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()

    rng = np.random.default_rng(5)
    lon = np.asfortranarray(rng.standard_normal(10).astype(np.float32))
    lout = np.zeros(10, dtype=np.float32, order='F')
    sdfg(lon=lon, lout=lout)

    # reference: same source via gfortran/f2py (lout is intent(out), f2py returns it). Pointer rebind
    # must read s%w(1,1,1)=5.5 back through p_area -- a dropped/miscompiled rebind leaves lout(1) at 0.
    ref = f2py_compile(src, tmp_path / 'ref', 'pointer_ref')
    lout_ref = ref.main(lon.copy(order='F'))
    np.testing.assert_array_equal(lout, lout_ref)

    # closed form: single 5.5 + lon(1) add (bit-exact in float32); only element 1 written, rest zero.
    expected = np.zeros(10, dtype=np.float32)
    expected[0] = np.float32(5.5) + lon[0]
    np.testing.assert_array_equal(lout, expected)
