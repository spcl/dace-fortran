"""Whole-array copies between ALLOCATABLEs and normal arrays.

An ALLOCATABLE's SDFG transient carries its own ALLOCATE extent symbol, distinct
from a normal dummy's extent -- a Fortran whole-array assignment between the two
conforms but the names differ, so ``CopyLibraryNode``'s same-rank expansion can't
prove the per-dim shapes equal.  ``emit_copy`` drives the destination memlet off
the SOURCE descriptor's shape so subsets align; these tests pin correct values
(no truncation/overflow) both directions, 1-D/2-D, and allocatable-to-allocatable.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_copy_normal_into_allocatable_1d(tmp_path: Path):
    """``x = src`` -- destination is the allocatable (extent symbol
    ``x_d0``), source a normal dummy (extent ``n``)."""
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
    n = 7
    src_a = np.arange(1, n + 1, dtype=np.float64).copy(order='F')
    out = np.zeros(n, dtype=np.float64, order='F')
    sdfg(n=n, src=src_a, out=out)
    np.testing.assert_array_equal(out, src_a)


def test_copy_allocatable_into_normal_1d(tmp_path: Path):
    """``out = x`` -- source is the allocatable, destination a normal
    dummy (the mirror direction)."""
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
    # Non-contiguous-valued input so a wrong subset would show up as garbage.
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 5
    src_a = (10.0 * np.sin(np.arange(n))).copy(order='F')
    out = np.zeros(n, dtype=np.float64, order='F')
    sdfg(n=n, src=src_a, out=out)
    np.testing.assert_allclose(out, src_a, rtol=1e-13)


def test_copy_allocatable_to_allocatable_1d(tmp_path: Path):
    """``x = y`` -- both sides are allocatables, each with its own
    distinct ALLOCATE extent symbol."""
    src = """
subroutine main(n, src, out)
  integer, intent(in) :: n
  double precision, intent(in)  :: src(n)
  double precision, intent(out) :: out(n)
  double precision, allocatable :: x(:), y(:)
  allocate(x(n))
  allocate(y(n))
  y = src
  x = y
  out = x
  deallocate(x)
  deallocate(y)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    n = 6
    src_a = np.arange(100, 100 + n, dtype=np.float64).copy(order='F')
    out = np.zeros(n, dtype=np.float64, order='F')
    sdfg(n=n, src=src_a, out=out)
    np.testing.assert_array_equal(out, src_a)


def test_copy_normal_into_allocatable_2d(tmp_path: Path):
    """Same-rank multi-dim copy: ``x(m,n) = src(m,n)`` where ``x`` is a
    rank-2 allocatable (per-dim symbols ``x_d0`` / ``x_d1``)."""
    src = """
subroutine main(m, n, src, out)
  integer, intent(in) :: m, n
  double precision, intent(in)  :: src(m, n)
  double precision, intent(out) :: out(m, n)
  double precision, allocatable :: x(:, :)
  allocate(x(m, n))
  x = src
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    m, n = 3, 4
    src_a = np.arange(1, m * n + 1, dtype=np.float64).reshape(m, n, order='F').copy(order='F')
    out = np.zeros((m, n), dtype=np.float64, order='F')
    sdfg(m=m, n=n, src=src_a, out=out)
    np.testing.assert_array_equal(out, src_a)


def test_copy_allocatable_into_normal_2d(tmp_path: Path):
    """Rank-2 mirror direction: ``out(m,n) = x(m,n)`` with ``x`` the
    allocatable source."""
    src = """
subroutine main(m, n, src, out)
  integer, intent(in) :: m, n
  double precision, intent(in)  :: src(m, n)
  double precision, intent(out) :: out(m, n)
  double precision, allocatable :: x(:, :)
  allocate(x(m, n))
  x = src
  out = x
  deallocate(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main').build()
    m, n = 4, 2
    src_a = (np.arange(m * n, dtype=np.float64).reshape(m, n, order='F') * 0.5).copy(order='F')
    out = np.zeros((m, n), dtype=np.float64, order='F')
    sdfg(m=m, n=n, src=src_a, out=out)
    np.testing.assert_array_equal(out, src_a)
