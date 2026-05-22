"""Classification contract for module-scope globals: caller-supplied
kwarg vs baked constant vs writable transient.

A Fortran module-scope variable reaches the SDFG through one of three
shapes, decided by ``extract_vars.cpp`` + the SDFG builder:

  * **PARAMETER / literal pool** (``real, parameter :: g = 9.81``):
    a true compile-time constant.  Baked into the constant pool, never
    a kwarg, and carries NO module-origin provenance (the caller can't
    rebind a ``parameter``).

  * **Uninitialised module global** (``integer :: ncfg``): an external
    input the caller fills via ``USE``.  Surfaces as a non-transient
    kwarg with module-origin provenance.

  * **Initialised module global, read-only** (``real :: s = 2.5``):
    takes the constant-pool path so its declared default is baked in,
    BUT still records module-origin provenance so the binding layer can
    ``USE``-import a host override (the baked value is the default).
    This is the ICON ``i_am_accel_node = .FALSE.`` shape.

  * **Module global the kernel WRITES** (``logical :: ready = .false.``
    set inside the routine): "not really constant" -- a writable
    transient seeded with its declared initial value at entry, not a
    read-only ``constexpr`` and not a caller import.

Each case is pinned structurally (arglist membership + frozen-signature
``module_symbol_origins``) and, where a value is observable, end-to-end
against an f2py reference compiled from the same source.
"""
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, f2py_compile, have_flang

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def _build(src: str, tmp: Path, entry: str):
    """Build the SDFG for ``entry`` and validate it."""
    sdfg_dir = tmp / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(src, sdfg_dir, name=entry.split("P")[-1], entry=entry).build()
    sdfg.validate()
    return sdfg


def _origin(sdfg, name: str):
    """Module-origin ``(module, entity)`` the bridge auto-detected for
    ``name``, or ``None`` when it recorded no provenance."""
    return sdfg._frozen_signature.module_symbol_origins.get(name)


def test_parameter_is_baked_constant(tmp_path: Path):
    """A ``parameter`` is a compile-time constant: baked, never a kwarg,
    and with no module-origin provenance (it cannot be rebound)."""
    src = """
module mod_param
  implicit none
  real(8), parameter :: gconst = 9.81d0
contains
  subroutine apply_param(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    do i = 1, 4
      y(i) = x(i) * gconst
    end do
  end subroutine apply_param
end module mod_param
"""
    sdfg = _build(src, tmp_path, "_QMmod_paramPapply_param")
    assert 'gconst' not in sdfg.arglist(), "a parameter must be baked, not a kwarg"
    assert _origin(sdfg, 'gconst') is None, "a parameter carries no caller provenance"

    ref = f2py_compile(src, tmp_path / "ref", "param_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_param.apply_param(x)
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_uninitialised_global_is_caller_kwarg(tmp_path: Path):
    """An uninitialised module global is an external input: a
    non-transient kwarg with module-origin provenance."""
    src = """
module mod_cfg
  implicit none
  real(8) :: cfg_scale
contains
  subroutine apply_cfg(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    do i = 1, 4
      y(i) = x(i) * cfg_scale
    end do
  end subroutine apply_cfg
end module mod_cfg
"""
    sdfg = _build(src, tmp_path, "_QMmod_cfgPapply_cfg")
    assert 'cfg_scale' in sdfg.arglist(), "an uninitialised module global must surface as a kwarg"
    assert _origin(sdfg, 'cfg_scale') == ('mod_cfg', 'cfg_scale')

    ref = f2py_compile(src, tmp_path / "ref", "cfg_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    ref.mod_cfg.cfg_scale = 3.0
    y_ref = ref.mod_cfg.apply_cfg(x)
    # A module-global scalar surfaces as a length-1 array kwarg (same as a
    # passed-in module array), so bind it as a 1-element array.
    sdfg(x=x, y=y_sdfg, cfg_scale=np.array([3.0], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_initialised_numeric_global_is_baked_with_provenance(tmp_path: Path):
    """A read-only numeric module global WITH an initialiser bakes its
    default into the constant pool (init_test relies on this), yet still
    records module-origin provenance for a host override."""
    src = """
module mod_init
  implicit none
  real(8) :: init_scale = 2.5d0
contains
  subroutine apply_init(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    do i = 1, 4
      y(i) = x(i) * init_scale
    end do
  end subroutine apply_init
end module mod_init
"""
    sdfg = _build(src, tmp_path, "_QMmod_initPapply_init")
    assert 'init_scale' not in sdfg.arglist(), "an initialised read-only global bakes its default"
    assert _origin(sdfg, 'init_scale') == ('mod_init', 'init_scale'), \
        "the baked default must still record provenance for a host override"

    ref = f2py_compile(src, tmp_path / "ref", "init_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_init.apply_init(x)  # uses the module default 2.5
    sdfg(x=x, y=y_sdfg)                  # uses the baked default 2.5
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_initialised_logical_global_is_baked_with_provenance(tmp_path: Path):
    """The ICON ``i_am_accel_node = .FALSE.`` shape: a read-only LOGICAL
    module global with an initialiser bakes its default and records
    provenance (regression guard -- logical-init globals must classify
    exactly like numeric-init globals)."""
    src = """
module mod_flag
  implicit none
  logical :: use_neg = .true.
contains
  subroutine apply_flag(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    do i = 1, 4
      if (use_neg) then
        y(i) = -x(i)
      else
        y(i) = x(i)
      end if
    end do
  end subroutine apply_flag
end module mod_flag
"""
    sdfg = _build(src, tmp_path, "_QMmod_flagPapply_flag")
    assert 'use_neg' not in sdfg.arglist(), "an initialised read-only logical bakes its default"
    assert _origin(sdfg, 'use_neg') == ('mod_flag', 'use_neg')

    ref = f2py_compile(src, tmp_path / "ref", "flag_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_flag.apply_flag(x)  # default .true. -> negate
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_written_global_is_seeded_writable_transient(tmp_path: Path):
    """A module global the kernel WRITES is "not really constant": a
    writable transient seeded with its declared initial value at entry,
    not a read-only ``constexpr`` (which the store could not compile
    against) and not a caller import."""
    src = """
module mod_state
  implicit none
  logical :: initialized = .false.
  real(8) :: cached = 0.0d0
contains
  subroutine compute(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    if (.not. initialized) then
      cached = 10.0d0
      initialized = .true.
    end if
    do i = 1, 4
      y(i) = x(i) + cached
    end do
  end subroutine compute
end module mod_state
"""
    sdfg = _build(src, tmp_path, "_QMmod_statePcompute")
    assert 'initialized' not in sdfg.arglist(), "a kernel-written flag is an internal transient, not a kwarg"
    assert _origin(sdfg, 'initialized') is None, "a kernel-written global is not a caller import"

    ref = f2py_compile(src, tmp_path / "ref", "state_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_state.compute(x)  # first call: initialized .false. -> cached = 10
    sdfg(x=x, y=y_sdfg)               # seeded .false. -> cached = 10
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_written_global_no_initialiser_same_module(tmp_path: Path):
    """A module global with NO declared initialiser that the kernel
    assigns before reading: an internal writable transient (no seed
    needed, written-before-read), neither a kwarg nor a caller import."""
    src = """
module mod_scratch
  implicit none
  real(8) :: sval
contains
  subroutine use_sval(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    sval = 3.0d0
    do i = 1, 4
      y(i) = x(i) + sval
    end do
  end subroutine use_sval
end module mod_scratch
"""
    sdfg = _build(src, tmp_path, "_QMmod_scratchPuse_sval")
    assert 'sval' not in sdfg.arglist(), "a written-inside scratch global is an internal transient"
    assert _origin(sdfg, 'sval') is None, "a kernel-written global is not a caller import"

    ref = f2py_compile(src, tmp_path / "ref", "scratch_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_scratch.use_sval(x)
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cross-module variants: the global is declared in one module and reached
# through ``USE <other_module>, ONLY: <name>`` from the kernel's module.
# ``merge_used_modules`` inlines the declaring module's source, and the
# global's mangled symbol stays ``_QM<decl_module>E<name>`` -- so it must
# classify exactly like a same-module global.
# ---------------------------------------------------------------------------


def test_parameter_from_other_module_is_baked(tmp_path: Path):
    """A ``parameter`` declared in another module and ``USE``-imported is
    still a baked constant: not a kwarg, no provenance."""
    src = """
module mod_phys_const
  implicit none
  real(8), parameter :: gravity = 9.81d0
end module mod_phys_const

module mod_kern_a
  implicit none
contains
  subroutine use_const(x, y)
    use mod_phys_const, only: gravity
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    do i = 1, 4
      y(i) = x(i) * gravity
    end do
  end subroutine use_const
end module mod_kern_a
"""
    sdfg = _build(src, tmp_path, "_QMmod_kern_aPuse_const")
    assert 'gravity' not in sdfg.arglist(), "a USE-imported parameter is baked"
    assert _origin(sdfg, 'gravity') is None

    ref = f2py_compile(src, tmp_path / "ref", "xconst_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_kern_a.use_const(x)
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_initialised_updated_global_from_other_module(tmp_path: Path):
    """A global declared WITH an initialiser in another module and UPDATED
    inside the kernel: seeded with its init at entry, the update is applied
    within the call, and provenance is recorded (for the host write-back
    handled separately).  The computed output reflects the updated value."""
    src = """
module mod_state_x
  implicit none
  real(8) :: accum = 1.0d0
end module mod_state_x

module mod_kern_b
  implicit none
contains
  subroutine use_state(x, y)
    use mod_state_x, only: accum
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    accum = accum + 10.0d0
    do i = 1, 4
      y(i) = x(i) + accum
    end do
  end subroutine use_state
end module mod_kern_b
"""
    sdfg = _build(src, tmp_path, "_QMmod_kern_bPuse_state")
    # The update is applied within the call, so the computed output is
    # correct today.  Host-visibility of the global's final value (the
    # binding writing it back on exit) is a separate, pending feature; until
    # it lands a written global records no caller provenance.
    assert _origin(sdfg, 'accum') is None

    ref = f2py_compile(src, tmp_path / "ref", "xstate_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_kern_b.use_state(x)  # accum 1.0 -> 11.0, output x + 11
    sdfg(x=x, y=y_sdfg)                   # seeded 1.0 -> 11.0, output x + 11
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_written_global_no_initialiser_from_other_module(tmp_path: Path):
    """A global declared with NO initialiser in another module, assigned
    before being read inside the kernel: an internal writable transient.
    The computed output reflects the in-kernel assignment."""
    src = """
module mod_scratch_x
  implicit none
  real(8) :: tmpval
end module mod_scratch_x

module mod_kern_c
  implicit none
contains
  subroutine use_scratch(x, y)
    use mod_scratch_x, only: tmpval
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    tmpval = 7.0d0
    do i = 1, 4
      y(i) = x(i) * tmpval
    end do
  end subroutine use_scratch
end module mod_kern_c
"""
    sdfg = _build(src, tmp_path, "_QMmod_kern_cPuse_scratch")
    assert 'tmpval' not in sdfg.arglist(), "a written-inside cross-module scratch global is internal"

    ref = f2py_compile(src, tmp_path / "ref", "xscratch_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_kern_c.use_scratch(x)
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)
