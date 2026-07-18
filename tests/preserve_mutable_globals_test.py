"""Coverage for ``hlfir-preserve-mutable-globals``: classifies ``fir.global`` ops as INPUT vs
MUTABLE by whether the IR writes them, with a function-scope (``_QF...``) carve-out for
routine-local SAVE-semantic initialisers. Clears the init body of every non-written,
non-PARAMETER ``_QM<mod>E<var>`` global so ``sccp`` can't fold loads to the BSS init;
PARAMETER and function-scope globals stay untouched (caller can't bind the latter).
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_global_read_only_surfaces_as_kwarg(tmp_path):
    """``_QM<mod>E<var>`` read-only init becomes a caller kwarg (INPUT bucket); the caller's runtime value reaches the kernel, not the source initialiser."""
    src = """
module m
  implicit none
  double precision :: scale = 99.0d0
contains
  subroutine apply(x, y)
    double precision, intent(in) :: x(2)
    double precision, intent(out) :: y(2)
    integer :: i
    do i = 1, 2
      y(i) = x(i) * scale
    end do
  end subroutine apply
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="m::apply").build()
    assert 'scale' in sdfg.arglist(), \
        f"module read-only init must surface as kwarg; got {sorted(sdfg.arglist())}"
    x = np.array([2.0, 3.0], dtype=np.float64, order='F')
    y = np.zeros(2, dtype=np.float64, order='F')
    # caller supplies a value DIFFERENT from the source default 99.0, to prove the kernel uses the runtime value
    sdfg(x=x, y=y, scale=np.array([10.0], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y, [20.0, 30.0])


def test_function_scope_global_stays_baked(tmp_path):
    """``_QF<func>E<var>`` routine-local SAVE-initialiser (``real :: bob = 1`` inside a subroutine)
    stays baked -- NOT a caller-bindable input."""
    src = """
module s_mod
  implicit none
contains
  subroutine s(out)
    implicit none
    double precision, intent(out) :: out(2)
    real :: bob = 7.5
    out(1) = bob
    out(2) = bob + 1.0
  end subroutine s
end module s_mod
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s_mod::s").build()
    # bob is NOT a kwarg -- function-scope globals are caller-invisible (no symbol to bind)
    assert 'bob' not in sdfg.arglist(), \
        f"function-scope global must stay baked; arglist={sorted(sdfg.arglist())}"
    out = np.zeros(2, dtype=np.float64, order='F')
    sdfg(out=out)
    # source default 7.5 must reach the kernel -- if the pass cleared bob's body, both reads would see uninitialised (~0) memory
    np.testing.assert_allclose(out, [7.5, 8.5])


def test_parameter_module_constant_stays_baked(tmp_path):
    """``parameter`` module constants stay baked regardless of the write-based classifier -- ``constant`` attribute is the first filter."""
    src = """
module m
  implicit none
  double precision, parameter :: kappa = 0.5d0
contains
  subroutine apply(x, y)
    double precision, intent(in) :: x(2)
    double precision, intent(out) :: y(2)
    integer :: i
    do i = 1, 2
      y(i) = x(i) * kappa
    end do
  end subroutine apply
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="m::apply").build()
    assert 'kappa' not in sdfg.arglist(), \
        f"PARAMETER must stay baked; arglist={sorted(sdfg.arglist())}"
    x = np.array([4.0, 6.0], dtype=np.float64, order='F')
    y = np.zeros(2, dtype=np.float64, order='F')
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y, [2.0, 3.0])  # 4*0.5, 6*0.5


def test_cross_module_use_propagates_global(tmp_path):
    """Global defined in one module and USE-imported into another surfaces as a kwarg of the
    importer's entry kernel -- the canonical LTO/config-propagation shape.
    """
    src = """
module mod_cfg
  implicit none
  double precision :: scale = 7.0d0
end module mod_cfg

module mod_use
  use mod_cfg, only: scale
  implicit none
contains
  subroutine apply(x, y)
    double precision, intent(in) :: x(2)
    double precision, intent(out) :: y(2)
    integer :: i
    do i = 1, 2
      y(i) = x(i) * scale
    end do
  end subroutine apply
end module mod_use
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="mod_use::apply").build()
    assert 'scale' in sdfg.arglist()
    x = np.array([2.0, 3.0], dtype=np.float64, order='F')
    y = np.zeros(2, dtype=np.float64, order='F')
    sdfg(x=x, y=y, scale=np.array([10.0], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y, [20.0, 30.0])


def test_module_array_global_surfaces_as_kwarg(tmp_path):
    """Module-level ARRAY global (not just scalar), read-only, also surfaces as a kwarg -- same
    contract as the scalar case (config blobs / lookup tables).
    """
    src = """
module mod_table
  implicit none
  double precision :: lut(4) = (/1.0d0, 2.0d0, 3.0d0, 4.0d0/)
contains
  subroutine apply(x, y)
    double precision, intent(in) :: x(4)
    double precision, intent(out) :: y(4)
    integer :: i
    do i = 1, 4
      y(i) = x(i) * lut(i)
    end do
  end subroutine apply
end module mod_table
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="mod_table::apply").build()
    assert 'lut' in sdfg.arglist()
    x = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64, order='F')
    y = np.zeros(4, dtype=np.float64, order='F')
    # Caller supplies a DIFFERENT lut than the source default.
    sdfg(x=x, y=y, lut=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y, [5.0, 15.0, 25.0, 35.0])


def test_module_global_scalar_writeback(tmp_path):
    """Companion to ``test_module_global_read_only_surfaces_as_kwarg``: when the kernel WRITES
    the module-scalar global, the caller's buffer reflects the post-call value (writeback)."""
    src = """
module m
  implicit none
  double precision :: state
contains
  subroutine update(x)
    double precision, intent(in) :: x
    state = state + x
  end subroutine update
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="update", entry="m::update").build()
    assert 'state' in sdfg.arglist()
    state = np.array([4.0], dtype=np.float64, order='F')
    sdfg(state=state, x=np.float64(3.5))
    np.testing.assert_allclose(state[0], 7.5)


def test_cross_module_global_writeback(tmp_path):
    """Companion to ``test_cross_module_use_propagates_global``: a USE'd global the kernel WRITES is updated on the caller's buffer after the call."""
    src = """
module mod_cfg
  implicit none
  double precision :: scale = 7.0d0
end module mod_cfg

module mod_use
  use mod_cfg, only: scale
  implicit none
contains
  subroutine bump_scale(amount)
    double precision, intent(in) :: amount
    scale = scale * amount
  end subroutine bump_scale
end module mod_use
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bump_scale", entry="mod_use::bump_scale").build()
    assert 'scale' in sdfg.arglist()
    scale = np.array([10.0], dtype=np.float64, order='F')
    sdfg(scale=scale, amount=np.float64(2.5))
    np.testing.assert_allclose(scale[0], 25.0)


def test_module_array_global_writeback(tmp_path):
    """Companion to ``test_module_array_global_surfaces_as_kwarg``: a written array global updates
    the caller's buffer after the call; bindings route inbound value and outbound writeback through the same length-N buffer.
    """
    src = """
module mod_table
  implicit none
  double precision :: lut(4) = (/1.0d0, 2.0d0, 3.0d0, 4.0d0/)
contains
  subroutine scale_lut(factor)
    double precision, intent(in) :: factor
    integer :: i
    do i = 1, 4
      lut(i) = lut(i) * factor
    end do
  end subroutine scale_lut
end module mod_table
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="scale_lut", entry="mod_table::scale_lut").build()
    assert 'lut' in sdfg.arglist()
    lut = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64, order='F')
    sdfg(lut=lut, factor=np.float64(2.0))
    np.testing.assert_allclose(lut, [1.0, 3.0, 5.0, 7.0])


def test_module_global_written_in_body_is_mutable(tmp_path):
    """Module-level global the kernel WRITES is mutable: the pass leaves its body alone (runtime
    may legitimately observe the initial value before the first write); still a (1,)-Array kwarg."""
    src = """
module m
  implicit none
  integer :: counter = 5
contains
  subroutine bump(n_calls)
    integer, intent(in) :: n_calls
    integer :: i
    do i = 1, n_calls
      counter = counter + 1
    end do
  end subroutine bump
end module m
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bump", entry="m::bump").build()
    assert 'counter' in sdfg.arglist(), \
        f"mutable module global must be a kwarg; arglist={sorted(sdfg.arglist())}"
    # counter pre-set to 10, n_calls=3 -> counter=13; intent(in) scalars surface as true Scalar (pass-by-value)
    counter = np.array([10], dtype=np.int32, order='F')
    sdfg(counter=counter, n_calls=np.int32(3))
    assert counter[0] == 13
