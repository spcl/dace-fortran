"""Tests covering multiple Fortran modules, ``use`` chains, and module-level constants.

Flang folds module imports during -emit-hlfir, so by the time the bridge sees
the IR every cross-module reference is a global symbol or inlined constant.
Pins: single-module use; chained use (a -> b uses a -> main uses b); parameter
constants land as inlined literals, not runtime args; multiple modules each
contributing a procedure to main (inline-all collapses all into one HLFIR).
"""

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_single_module_use(tmp_path):
    """One module exporting a constant and a procedure, ``use``d by main."""
    src = """
module phys
  implicit none
  real(8), parameter :: g = 9.81d0
contains
  subroutine apply_gravity(v, dt)
    real(8), intent(inout) :: v
    real(8), intent(in)    :: dt
    v = v - g * dt
  end subroutine
end module phys

subroutine main(v, dt)
  use phys, only: apply_gravity
  real(8), intent(inout) :: v
  real(8), intent(in)    :: dt
  call apply_gravity(v, dt)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    v = np.array([10.0], dtype=np.float64)
    sdfg(v=v, dt=0.1)
    np.testing.assert_allclose(v[0], 10.0 - 9.81 * 0.1, rtol=1e-12)


def test_chained_module_use(tmp_path):
    """Three modules chained a->b->main; a's constant folds into a literal at
    compile-time, bridge sees only the inlined body in main."""
    src = """
module consts
  implicit none
  real(8), parameter :: factor = 2.5d0
end module consts

module ops
  use consts, only: factor
  implicit none
contains
  subroutine scale(x, y)
    real(8), intent(in)  :: x
    real(8), intent(out) :: y
    y = x * factor
  end subroutine
end module ops

subroutine main(x, y)
  use ops, only: scale
  real(8), intent(in)  :: x
  real(8), intent(out) :: y
  call scale(x, y)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    y = np.zeros(1, dtype=np.float64)
    sdfg(x=4.0, y=y)
    np.testing.assert_allclose(y[0], 4.0 * 2.5, rtol=1e-12)


def test_two_modules_combined_in_main(tmp_path):
    """Two independent modules threaded in sequence from main -- inline-all walks
    both callees and merges them."""
    src = """
module add_mod
  implicit none
contains
  subroutine add_one(x)
    real(8), intent(inout) :: x
    x = x + 1.0d0
  end subroutine
end module add_mod

module mul_mod
  implicit none
contains
  subroutine times_two(x)
    real(8), intent(inout) :: x
    x = x * 2.0d0
  end subroutine
end module mul_mod

subroutine main(x)
  use add_mod, only: add_one
  use mul_mod, only: times_two
  real(8), intent(inout) :: x
  call add_one(x)
  call times_two(x)
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    x = np.array([3.0], dtype=np.float64)
    sdfg(x=x)
    np.testing.assert_allclose(x[0], (3.0 + 1.0) * 2.0, rtol=1e-12)


def test_module_parameter_in_loop_bound(tmp_path):
    """Module parameter as a DO loop upper bound: flang folds it to a literal so the
    LoopRegion condition is a constant expression. RHS uses indexed array reads
    (via iter_map) rather than the loop iter as a scalar -- a known bridge gap
    where scalar RHS uses of the loop iter don't remap to LoopRegion's uniquified i_0."""
    src = """
module sizes
  implicit none
  integer, parameter :: nlev = 8
end module sizes

subroutine main(src_arr, a)
  use sizes, only: nlev
  real(8), intent(in)  :: src_arr(nlev)
  real(8), intent(out) :: a(nlev)
  integer :: i
  do i = 1, nlev
    a(i) = src_arr(i) * 2.0d0
  end do
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    src_arr = np.arange(1, 9, dtype=np.float64)
    a = np.zeros(8, dtype=np.float64)
    sdfg(src_arr=src_arr, a=a, i=0)
    np.testing.assert_allclose(a, src_arr * 2.0, rtol=1e-12)


def test_use_only_renamed_symbol(tmp_path):
    """``use mod, only: local => mod_name`` renames the import -- resolved entirely
    in Flang's symbol resolution, the bridge sees the canonical name."""
    src = """
module trig_consts
  implicit none
  real(8), parameter :: pi_value = 3.14159265358979323846d0
end module trig_consts

subroutine main(out)
  use trig_consts, only: my_pi => pi_value
  real(8), intent(out) :: out
  out = my_pi * 2.0d0
end subroutine main
"""
    sdfg = build_sdfg(src, tmp_path, name='main', entry='main').build()
    out = np.zeros(1, dtype=np.float64)
    sdfg(out=out)
    np.testing.assert_allclose(out[0], 3.14159265358979323846 * 2.0, rtol=1e-10)
