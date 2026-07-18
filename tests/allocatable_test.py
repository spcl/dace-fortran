"""ALLOCATABLE arrays, single alloc/dealloc scope (no reallocation): reads/writes through
the box descriptor. ``deallocate`` is a no-op -- bridge skips the ``fir.if`` cleanup guard,
transient dies at end-of-scope. Reallocation and COMMON/module allocatables not covered."""

from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_alloc_then_whole_array_copy(tmp_path: Path):
    """``allocate(x(n)); x = src; out = x; deallocate(x)`` round-trips end to end."""
    src = """
subroutine main(n, src, out)
  integer, intent(in) :: n
  double precision, intent(in)  :: src(n)
  double precision, intent(out) :: out(n)
  double precision, allocatable :: x(:)
  allocate(x(n))
  x = src
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 8
    src_arr = np.empty(n, dtype=np.float64, order='F')
    src_arr[:] = np.arange(1.0, n + 1.0)
    out = np.zeros(n, order='F', dtype=np.float64)
    sdfg(n=n, src=src_arr, out=out)
    np.testing.assert_array_equal(out, src_arr)


def test_alloc_then_section_copy(tmp_path: Path):
    """Section assigns into/out of an allocatable land on the right elements."""
    src = """
subroutine main(n, src, out)
  integer, intent(in) :: n
  double precision, intent(in)  :: src(n)
  double precision, intent(out) :: out(n)
  double precision, allocatable :: x(:)
  allocate(x(n))
  x(:) = src(:)
  out(:) = x(:)
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 6
    src_arr = np.array([10., 20., 30., 40., 50., 60.], dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    sdfg(n=n, src=src_arr, out=out)
    np.testing.assert_array_equal(out, src_arr)


def test_alloc_2d(tmp_path: Path):
    """2-D allocatable: extents traced from the multi-dim ``fir.allocmem``."""
    src = """
subroutine main(n, m, src, out)
  integer, intent(in) :: n, m
  double precision, intent(in)  :: src(n, m)
  double precision, intent(out) :: out(n, m)
  double precision, allocatable :: x(:, :)
  allocate(x(n, m))
  x = src
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n, m = 3, 4
    src_arr = np.asfortranarray(np.arange(n * m, dtype=np.float64).reshape(n, m))
    out = np.zeros((n, m), order='F', dtype=np.float64)
    sdfg(n=n, m=m, src=src_arr, out=out)
    np.testing.assert_array_equal(out, src_arr)
