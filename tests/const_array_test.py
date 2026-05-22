"""Array-of-constants support (coefficient / extrapolation tables common in
scientific Fortran).

Two shapes, decided by whether the kernel WRITES the array:

  * **immutable** -- a ``parameter`` array or a read-only DATA-statement
    array (``real :: c(3); data c /.../``) bakes into a DaCe ``constexpr``
    array (``sdfg.add_constant``); it never surfaces as a kwarg.
  * **mutable** -- a DATA-statement (or initialised) array the kernel also
    assigns to becomes a writable transient whose initial values are
    unfolded into per-element init tasklets at SDFG entry (a ``constexpr``
    would be read-only and the store could not compile).

flang lowers a DATA-statement array to a ``fir.global`` carrying a dense
initialiser but NOT marked ``constant`` (DATA variables are mutable); the
bridge extracts that dense data the same as a ``parameter`` array's.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_readonly_data_array_is_constexpr(tmp_path):
    """A read-only DATA-statement array bakes into a constexpr array -- not
    a kwarg the caller must supply."""
    src = """
module m
contains
  subroutine k(x, y)
    real, intent(in) :: x(3)
    real, intent(out) :: y
    real :: c(3)
    data c /1.0, 10.0, 100.0/
    integer :: i
    y = 0.0
    do i = 1, 3
      y = y + c(i) * x(i)
    end do
  end subroutine k
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name='k', entry='_QMmPk').build()
    assert 'c' not in sdfg.arglist(), "a read-only DATA array must bake, not be a kwarg"
    assert 'c' in getattr(sdfg, 'constants', {}), "expected c in the constant pool"
    x = np.ones(3, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y[0], 111.0, rtol=1e-6)


def test_mutable_data_array_is_seeded_transient(tmp_path):
    """A DATA-statement array the kernel also writes becomes a writable
    transient seeded by per-element init tasklets (not a constexpr, not a
    kwarg)."""
    src = """
module m
contains
  subroutine k(x, y)
    real, intent(in) :: x(3)
    real, intent(out) :: y
    real :: c(3)
    data c /1.0, 10.0, 100.0/
    integer :: i
    c(2) = c(2) + 5.0
    y = 0.0
    do i = 1, 3
      y = y + c(i) * x(i)
    end do
  end subroutine k
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name='k', entry='_QMmPk').build()
    assert 'c' not in sdfg.arglist(), "a kernel-written const array is an internal transient"
    assert 'c' not in getattr(sdfg, 'constants', {}), "a written array must not be a constexpr"
    x = np.ones(3, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y[0], 116.0, rtol=1e-6)  # 1 + (10+5) + 100


def test_module_parameter_array_is_constexpr(tmp_path):
    """A module-scope ``parameter`` coefficient array (the extrapolation /
    weight-table shape) bakes into a constexpr array, indexed in a loop."""
    src = """
module extrap_mod
  implicit none
  real, parameter :: w(4) = (/ 0.5, 1.5, -0.5, 0.25 /)
contains
  subroutine extrap(x, y)
    real, intent(in)  :: x(4)
    real, intent(out) :: y
    integer :: i
    y = 0.0
    do i = 1, 4
      y = y + w(i) * x(i)
    end do
  end subroutine extrap
end module extrap_mod
"""
    sdfg = build_sdfg(src, tmp_path, name='extrap', entry='_QMextrap_modPextrap').build()
    assert 'w' not in sdfg.arglist()
    x = np.arange(1, 5, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y[0], float((np.array([0.5, 1.5, -0.5, 0.25], dtype=np.float32) * x).sum()), rtol=1e-6)


def test_mutable_multidim_const_array_is_fortran_major(tmp_path):
    """A mutable 2-D DATA array: the per-element init tasklets must place
    each value at the Fortran (column-major) position the kernel later
    reads, so reads of ``a(i, j)`` see the original column-major layout
    (not a row-major transpose).  Also pins that the transient is laid out
    Fortran-major (unit leading stride)."""
    src = """
module m
contains
  subroutine k(ii, jj, y)
    integer, intent(in) :: ii, jj
    real, intent(out) :: y
    real :: a(2, 3)
    data a /1., 2., 3., 4., 5., 6./
    a(1, 2) = a(1, 2) + 100.0
    y = a(ii, jj)
  end subroutine k
end module m
"""
    sdfg = build_sdfg(src, tmp_path, name='k', entry='_QMmPk').build()
    assert 'a' not in sdfg.arglist() and 'a' not in getattr(sdfg, 'constants', {})
    # Fortran column-major: a(1,1)=1 a(2,1)=2 a(1,2)=3 a(2,2)=4 a(1,3)=5 a(2,3)=6.
    # Column-major storage -> the leading (row) stride is 1.
    assert int(sdfg.arrays['a'].strides[0]) == 1, "const array transient must be Fortran (column-major) laid out"
    for ii, jj, exp in [(1, 2, 103.0), (2, 1, 2.0), (1, 1, 1.0), (2, 3, 6.0), (1, 3, 5.0)]:
        y = np.zeros(1, dtype=np.float32)
        sdfg(ii=np.int32(ii), jj=np.int32(jj), y=y)
        np.testing.assert_allclose(y[0], exp, rtol=1e-6, err_msg=f"a({ii},{jj})")
