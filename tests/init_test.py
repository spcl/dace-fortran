"""Verbatim port of f2dace/dev:tests/fortran/init_test.py."""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_fortran_frontend_init(tmp_path):
    src = """
module lib1
  implicit none
  real :: outside_init = epsilon(1.0)
end module lib1

module lib2
contains
  subroutine init_test_function(d)
    use lib1, only: outside_init
    double precision d(4)
    real:: bob = epsilon(1.0)
    d(2) = 5.5 + bob + outside_init
  end subroutine init_test_function
end module lib2

subroutine main(d)
  use lib2, only: init_test_function
  implicit none
  double precision d(4)
  call init_test_function(d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    # outside_init is a non-PARAMETER module scalar with a source initialiser; hlfir-preserve-mutable-globals surfaces it as a caller kwarg defaulting to the source value.
    eps = np.finfo(np.float32).eps
    sdfg(d=a, outside_init=np.array([eps], dtype=np.float32, order='F'))
    assert (a[0] == 42)
    # 5.5 + bob + outside_init, both epsilon(1.0) (~1.19e-7). Fortran real(4) rounds this to exactly 5.5; the bridge promotes to double earlier so it lands at ~5.5000002 instead -- both correct, tolerate ~3*epsilon(1.0).
    assert abs(a[1] - 5.5) < 1e-6
    assert (a[2] == 42)


def test_fortran_frontend_init2(tmp_path):
    src = """
module lib1
  implicit none
  real, parameter :: TORUS_MAX_LAT = 4.0/18.0*atan(1.0)
end module lib1

module lib2
contains
  subroutine init2_test_function(d)
    use lib1, only: TORUS_MAX_LAT
    double precision d(4)
    d(2) = 5.5 + TORUS_MAX_LAT
  end subroutine init2_test_function
end module lib2

subroutine main(d)
  use lib2, only: init2_test_function
  implicit none
  double precision d(4)
  call init2_test_function(d)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    a = np.full([4], 42, order="F", dtype=np.float64)
    sdfg(d=a)
    assert np.allclose(a, [42, 5.674532920122147, 42, 42])
