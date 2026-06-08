"""Verbatim port of f2dace/dev:tests/fortran/view_reshape_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_view_reshape(tmp_path):
    src = """
module lib1
  implicit none
  real :: outside_init = 1
end module lib1

module lib2
contains
  subroutine view_reshape_test_function(dd)
    use lib1, only: outside_init
    double precision dd(16)
    real:: bob = epsilon(1.0)

    dd(2) = 5.5 + bob

  end subroutine view_reshape_test_function
end module lib2

subroutine main(d)
  use lib2, only: view_reshape_test_function
  implicit none
  integer :: i
  integer :: j
  double precision d(4,4,2)

  i=2
  j=1
  call view_reshape_test_function(d(:,:,1))
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='_QPmain').build()
    a = np.full([4, 4, 2], 42, order="F", dtype=np.float64)
    # ``outside_init`` is a non-PARAMETER scalar global -- surfaces as a
    # caller kwarg after the write-based classifier.  Pass a length-1
    # numpy array (the bridge surfaces module scalars as (1,)-Arrays).
    sdfg(d=a, outside_init=np.array([0.0], dtype=np.float32, order='F'))
    assert (a[0, 0, 0] == 42)
    assert (a[1, 0, 0] == 5.5)
    assert (a[2, 0, 0] == 42)
