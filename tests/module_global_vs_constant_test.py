"""Classification contract for module-scope globals: caller kwarg vs baked constant vs
writable transient. Four shapes (decided by ``extract_vars.cpp`` + the SDFG builder):

  * PARAMETER: baked constant, never a kwarg, no provenance (cannot be rebound).
  * Uninitialised global: external input, kwarg with module-origin provenance.
  * Initialised global, read-only: baked as default but still records provenance so the
    binding can ``USE``-import a host override (ICON's ``i_am_accel_node`` shape).
  * Global the kernel WRITES: inout kwarg with provenance; copy-in host value, copy-out the
    kernel's final value (a SAVE-local the kernel writes is instead a private transient).

Each case pinned structurally (arglist + ``module_symbol_origins``) and, where observable, against an f2py reference.
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
    """Module-origin ``(module, entity)`` the bridge auto-detected for ``name``, or None."""
    return sdfg._frozen_signature.module_symbol_origins.get(name)


def test_parameter_is_baked_constant(tmp_path: Path):
    """``parameter`` is a compile-time constant: baked, never a kwarg, no module-origin provenance."""
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
    sdfg = _build(src, tmp_path, "apply_param")
    assert 'gconst' not in sdfg.arglist(), "a parameter must be baked, not a kwarg"
    assert _origin(sdfg, 'gconst') is None, "a parameter carries no caller provenance"

    ref = f2py_compile(src, tmp_path / "ref", "param_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_param.apply_param(x)
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_uninitialised_global_is_caller_kwarg(tmp_path: Path):
    """Uninitialised module global is an external input: non-transient kwarg with module-origin provenance."""
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
    sdfg = _build(src, tmp_path, "apply_cfg")
    assert 'cfg_scale' in sdfg.arglist(), "an uninitialised module global must surface as a kwarg"
    assert _origin(sdfg, 'cfg_scale') == ('mod_cfg', 'cfg_scale')

    ref = f2py_compile(src, tmp_path / "ref", "cfg_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    ref.mod_cfg.cfg_scale = 3.0
    y_ref = ref.mod_cfg.apply_cfg(x)
    # module-global scalar surfaces as a length-1 array kwarg, same as a passed-in module array
    sdfg(x=x, y=y_sdfg, cfg_scale=np.array([3.0], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_initialised_numeric_global_is_caller_kwarg(tmp_path: Path):
    """Read-only numeric global WITH a source initialiser still surfaces as a caller kwarg (the
    source default becomes the value the caller supplies); provenance is still recorded.
    Pre-``hlfir-preserve-mutable-globals`` this was baked and never reached the arglist --
    the bug that broke LU's ``dt`` and every similar caller-pre-set module scalar.
    """
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
    sdfg = _build(src, tmp_path, "apply_init")
    assert 'init_scale' in sdfg.arglist(), \
        "an initialised read-only global surfaces as a caller kwarg"
    assert _origin(sdfg, 'init_scale') == ('mod_init', 'init_scale'), \
        "the kwarg must record provenance so the host can spot its source"

    ref = f2py_compile(src, tmp_path / "ref", "init_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_init.apply_init(x)  # f2py uses the module's source default 2.5
    # SDFG: caller supplies same value as source default
    sdfg(x=x, y=y_sdfg, init_scale=np.array([2.5], dtype=np.float64, order='F'))
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_initialised_logical_global_is_caller_kwarg(tmp_path: Path):
    """ICON ``i_am_accel_node = .FALSE.`` shape: read-only LOGICAL global with an initialiser
    surfaces as a caller kwarg, same as the numeric case. Pre-``hlfir-preserve-mutable-globals``
    this was baked and hidden from the caller.
    """
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
    sdfg = _build(src, tmp_path, "apply_flag")
    assert 'use_neg' in sdfg.arglist(), \
        "an initialised read-only logical surfaces as a caller kwarg"
    assert _origin(sdfg, 'use_neg') == ('mod_flag', 'use_neg')

    ref = f2py_compile(src, tmp_path / "ref", "flag_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_flag.apply_flag(x)  # default .true. -> negate
    sdfg(x=x, y=y_sdfg, use_neg=np.array([True], dtype=np.bool_, order='F'))
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def _written_arg(sdfg, name: str):
    """Frozen-signature arg for ``name`` (asserts it is present)."""
    fa = next((a for a in sdfg._frozen_signature.args if a.sdfg_name == name), None)
    assert fa is not None, f"{name} is not an SDFG arg"
    return fa


def test_written_global_is_inout_with_writeback(tmp_path: Path):
    """Module global the kernel WRITES is host-shared inout state: inout arg with provenance;
    the kernel's update is written back to the host module on exit, visible to the caller."""
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
    sdfg = _build(src, tmp_path, "compute")
    for name in ('initialized', 'cached'):
        assert name in sdfg.arglist(), f"{name}: a kernel-written global must be an inout arg"
        fa = _written_arg(sdfg, name)
        assert fa.intent == 'inout' and fa.is_written, f"{name}: expected written inout arg"
    assert _origin(sdfg, 'initialized') == ('mod_state', 'initialized')
    assert _origin(sdfg, 'cached') == ('mod_state', 'cached')

    ref = f2py_compile(src, tmp_path / "ref", "state_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_state.compute(x)  # initialized .false. -> cached 10, y = x + 10
    # pass the inout globals' host defaults; the kernel writes final values back in place
    initialized = np.array([False])
    cached = np.array([0.0], dtype=np.float64, order='F')
    sdfg(x=x, y=y_sdfg, initialized=initialized, cached=cached)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)
    assert bool(initialized[0]) is True, "the kernel's flag update is visible to the caller"
    np.testing.assert_allclose(cached, [10.0], rtol=1e-12)


def test_written_global_no_initialiser_same_module(tmp_path: Path):
    """Module global with no declared initialiser, assigned before read, is still host-shared
    inout state: an inout arg with provenance, updated in place."""
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
    sdfg = _build(src, tmp_path, "use_sval")
    assert 'sval' in sdfg.arglist(), "a kernel-written global is an inout arg"
    assert _written_arg(sdfg, 'sval').is_written
    assert _origin(sdfg, 'sval') == ('mod_scratch', 'sval')

    ref = f2py_compile(src, tmp_path / "ref", "scratch_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_scratch.use_sval(x)  # sval = 3.0, y = x + 3
    sval = np.array([0.0], dtype=np.float64, order='F')  # written before read; init irrelevant
    sdfg(x=x, y=y_sdfg, sval=sval)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)
    np.testing.assert_allclose(sval, [3.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Cross-module variants: global declared in one module, reached via ``USE <mod>, ONLY: <name>``.
# ``merge_used_modules`` inlines the source; the mangled symbol stays ``_QM<decl_module>E<name>``,
# so it must classify exactly like a same-module global.
# ---------------------------------------------------------------------------


def test_parameter_from_other_module_is_baked(tmp_path: Path):
    """``parameter`` declared in another module and ``USE``-imported is still baked: not a kwarg, no provenance."""
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
    sdfg = _build(src, tmp_path, "use_const")
    assert 'gravity' not in sdfg.arglist(), "a USE-imported parameter is baked"
    assert _origin(sdfg, 'gravity') is None

    ref = f2py_compile(src, tmp_path / "ref", "xconst_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_kern_a.use_const(x)
    sdfg(x=x, y=y_sdfg)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)


def test_initialised_updated_global_from_other_module(tmp_path: Path):
    """Global declared WITH an initialiser in another module and UPDATED inside the kernel:
    seeded at entry, updated in-call, provenance recorded for host write-back."""
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
    sdfg = _build(src, tmp_path, "use_state")
    assert 'accum' in sdfg.arglist(), "a written cross-module global is an inout arg"
    assert _written_arg(sdfg, 'accum').is_written
    assert _origin(sdfg, 'accum') == ('mod_state_x', 'accum'), \
        "a written cross-module global records provenance for host write-back"

    ref = f2py_compile(src, tmp_path / "ref", "xstate_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_kern_b.use_state(x)  # accum 1.0 -> 11.0, output x + 11
    accum = np.array([1.0], dtype=np.float64, order='F')  # host default
    sdfg(x=x, y=y_sdfg, accum=accum)  # 1.0 -> 11.0, output x + 11
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)
    np.testing.assert_allclose(accum, [11.0], rtol=1e-12)


def test_written_global_no_initialiser_from_other_module(tmp_path: Path):
    """Global declared with NO initialiser in another module, assigned before being read inside
    the kernel: an internal writable transient reflecting the in-kernel assignment."""
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
    sdfg = _build(src, tmp_path, "use_scratch")
    assert 'tmpval' in sdfg.arglist(), "a written cross-module global is an inout arg"
    assert _written_arg(sdfg, 'tmpval').is_written
    assert _origin(sdfg, 'tmpval') == ('mod_scratch_x', 'tmpval')

    ref = f2py_compile(src, tmp_path / "ref", "xscratch_ref")
    x = np.asfortranarray(np.arange(1, 5, dtype=np.float64))
    y_sdfg = np.zeros(4, dtype=np.float64, order='F')
    y_ref = ref.mod_kern_c.use_scratch(x)  # tmpval = 7.0, y = x * 7
    tmpval = np.array([0.0], dtype=np.float64, order='F')  # written before read
    sdfg(x=x, y=y_sdfg, tmpval=tmpval)
    np.testing.assert_allclose(y_sdfg, y_ref, rtol=1e-12)
    np.testing.assert_allclose(tmpval, [7.0], rtol=1e-12)
