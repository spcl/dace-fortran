"""Coverage for ``hlfir-preserve-mutable-globals``: classification of
``fir.global`` ops as INPUT vs MUTABLE based on whether the IR writes
them, with the function-scope (``_QF...``) carve-out for routine-local
``SAVE``-semantic initialisers.

The pass clears the init body of every non-written, non-PARAMETER
``_QM<mod>E<var>``-style global so ``sccp`` cannot fold its loads
to the BSS initializer.  PARAMETER constants stay baked, MUTABLE
globals stay alone, and function-scope ``_QF<func>E<var>`` globals
stay alone too -- the caller has no way to bind them, so their
source-level initialiser must reach the kernel intact.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_module_global_read_only_surfaces_as_kwarg(tmp_path):
    """``_QM<mod>E<var>`` read-only init becomes a caller kwarg
    (INPUT bucket), and the caller's runtime value reaches the
    kernel body instead of the source-level initialiser."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="apply").build()
    assert 'scale' in sdfg.arglist(), \
        f"module read-only init must surface as kwarg; got {sorted(sdfg.arglist())}"
    x = np.array([2.0, 3.0], dtype=np.float64, order='F')
    y = np.zeros(2, dtype=np.float64, order='F')
    # Caller supplies a value DIFFERENT from the source default 99.0
    # to prove the kernel uses the runtime value.
    sdfg(x=x, y=y, scale=np.array([10.0], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y, [20.0, 30.0])


def test_function_scope_global_stays_baked(tmp_path):
    """``_QF<func>E<var>`` routine-local initialiser (Fortran SAVE
    semantics for a subroutine-local declared with ``= literal``)
    stays baked -- it is NOT a caller-bindable input.  This is the
    ``real :: bob = 1`` pattern inside a subroutine body."""
    src = """
subroutine s(out)
  implicit none
  double precision, intent(out) :: out(2)
  real :: bob = 7.5
  out(1) = bob
  out(2) = bob + 1.0
end subroutine s
"""
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="s", entry="s").build()
    # ``bob`` is NOT a kwarg -- function-scope globals are caller-
    # invisible (the caller has no symbol to bind).
    assert 'bob' not in sdfg.arglist(), \
        f"function-scope global must stay baked; arglist={sorted(sdfg.arglist())}"
    out = np.zeros(2, dtype=np.float64, order='F')
    sdfg(out=out)
    # Source default 7.5 must reach the kernel.  If the pass had
    # cleared bob's body, both reads would see uninitialised memory
    # (typically 0) and the assertion below would catch it.
    np.testing.assert_allclose(out, [7.5, 8.5])


def test_parameter_module_constant_stays_baked(tmp_path):
    """``parameter`` module constants stay baked regardless of the
    write-based classifier -- ``constant`` attribute is the first
    filter in the pass."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="apply").build()
    # PARAMETER constants stay baked; no kwarg expected.
    assert 'kappa' not in sdfg.arglist(), \
        f"PARAMETER must stay baked; arglist={sorted(sdfg.arglist())}"
    x = np.array([4.0, 6.0], dtype=np.float64, order='F')
    y = np.zeros(2, dtype=np.float64, order='F')
    sdfg(x=x, y=y)
    np.testing.assert_allclose(y, [2.0, 3.0])  # 4*0.5, 6*0.5


def test_cross_module_use_propagates_global(tmp_path):
    """A global defined in one module and USE-imported into another
    must surface as a kwarg of the importer's entry kernel.  This is
    the canonical LTO / config-propagation shape: config lives in a
    shared module, every kernel that ``use``s it gets the same kwarg
    contract, the bindings emitter marshals one value into every
    call site.
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="apply").build()
    assert 'scale' in sdfg.arglist()
    x = np.array([2.0, 3.0], dtype=np.float64, order='F')
    y = np.zeros(2, dtype=np.float64, order='F')
    sdfg(x=x, y=y, scale=np.array([10.0], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y, [20.0, 30.0])


def test_module_array_global_surfaces_as_kwarg(tmp_path):
    """A module-level ARRAY global (not just a scalar) the kernel only
    reads also surfaces as a kwarg.  Config blobs (lookup tables,
    coefficient arrays) are a common LTO / config-propagation shape
    and must behave the same as the scalar case.
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="apply", entry="apply").build()
    assert 'lut' in sdfg.arglist()
    x = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64, order='F')
    y = np.zeros(4, dtype=np.float64, order='F')
    # Caller supplies a DIFFERENT lut than the source default.
    sdfg(x=x, y=y, lut=np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y, [5.0, 15.0, 25.0, 35.0])


def test_module_global_scalar_writeback(tmp_path):
    """Companion to ``test_module_global_read_only_surfaces_as_kwarg``:
    when the kernel WRITES the module-scalar global, the caller's
    runtime buffer reflects the post-call value.  Same arglist
    contract (non-transient (1,)-Array), but with writeback semantics."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="update", entry="update").build()
    assert 'state' in sdfg.arglist()
    state = np.array([4.0], dtype=np.float64, order='F')
    sdfg(state=state, x=np.float64(3.5))
    np.testing.assert_allclose(state[0], 7.5)


def test_cross_module_global_writeback(tmp_path):
    """Companion to ``test_cross_module_use_propagates_global``: a
    global the kernel WRITES via a USE statement is updated on the
    caller's buffer after the call."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bump_scale", entry="bump_scale").build()
    assert 'scale' in sdfg.arglist()
    scale = np.array([10.0], dtype=np.float64, order='F')
    sdfg(scale=scale, amount=np.float64(2.5))
    np.testing.assert_allclose(scale[0], 25.0)


def test_module_array_global_writeback(tmp_path):
    """Companion to ``test_module_array_global_surfaces_as_kwarg``:
    a non-PARAMETER initialised array global the kernel WRITES is
    updated on the caller's buffer after the call.  The bindings
    layer routes both the inbound value and the outbound writeback
    through the same length-N caller buffer.
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="scale_lut", entry="scale_lut").build()
    assert 'lut' in sdfg.arglist()
    lut = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64, order='F')
    sdfg(lut=lut, factor=np.float64(2.0))
    np.testing.assert_allclose(lut, [1.0, 3.0, 5.0, 7.0])


def test_module_global_written_in_body_is_mutable(tmp_path):
    """A module-level global the kernel WRITES is mutable; the pass
    leaves its body alone (the runtime may legitimately observe the
    BSS / source-level initial value before the first write).  Still
    surfaces as a (1,)-Array kwarg per the prune-non-transient
    contract."""
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
    sdfg = build_sdfg(src, tmp_path / "sdfg", name="bump", entry="bump").build()
    assert 'counter' in sdfg.arglist(), \
        f"mutable module global must be a kwarg; arglist={sorted(sdfg.arglist())}"
    # Caller pre-sets counter = 10, calls with n_calls = 3 -> counter = 13.
    # ``intent(in)`` scalars surface as true ``Scalar`` (pass-by-value).
    counter = np.array([10], dtype=np.int32, order='F')
    sdfg(counter=counter, n_calls=np.int32(3))
    assert counter[0] == 13
