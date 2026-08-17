"""Reproducer for bug 1 (dace-fortran-fixes-needed-6c99810.md):
bindings emitter hard-sets ``<dummy>_present = 0`` for ARRAY optionals.

These tests encode the CORRECT behavior.  At dace-fortran 6c99810 the two
``_binding``-suffixed tests FAIL (that is the bug firing); they must PASS once
``_optional_outer_dummies`` in dace_fortran/bindings/block_builders.py stops
restricting presence forwarding to scalar optionals.  The ``_sdfg_side`` test
is a CONTROL: it passes already, proving the defect is confined to the
bindings layer, not the SDFG.

Companion doc: bug1_optional_array_present.md
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.fortran_interface import build_auto_interface

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="no LLVM flang on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

# Minimal QE ``addusxx_g(becphi_c, ...)`` shape: an OPTIONAL ARRAY dummy whose
# presence gates the whole computation.
_KERNEL = """
module opt_addv_mod
  implicit none
contains
subroutine opt_addv(n, y, a)
  implicit none
  integer, intent(in) :: n
  real(8), intent(inout) :: y(n)
  real(8), intent(in), optional :: a(n)
  integer :: i
  if (present(a)) then
    do i = 1, n
      y(i) = y(i) + a(i)
    end do
  end if
end subroutine opt_addv
end module opt_addv_mod
"""

# bind(c) driver exercising BOTH the present and the absent call of the
# generated wrapper (mirror of tests/bindings/optional_forward_present_e2e_test.py).
_SDFG_DRIVER = """
subroutine run_opt_addv(n, y, a, has) bind(c, name='run_opt_addv')
  use iso_c_binding
  use opt_addv_dace_bindings
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: y(n)
  real(c_double), intent(in) :: a(n)
  integer(c_int), value :: has
  if (has /= 0) then
    call opt_addv_dace(n, y, a)
  else
    call opt_addv_dace(n, y)
  end if
  call opt_addv_dace_finalize()
end subroutine run_opt_addv
"""

_REF_DRIVER = """
subroutine run_opt_addv_ref(n, y, a, has) bind(c, name='run_opt_addv_ref')
  use iso_c_binding
  use opt_addv_mod, only: opt_addv
  implicit none
  integer(c_int), value :: n
  real(c_double), intent(inout) :: y(n)
  real(c_double), intent(in) :: a(n)
  integer(c_int), value :: has
  if (has /= 0) then
    call opt_addv(n, y, a)
  else
    call opt_addv(n, y)
  end if
end subroutine run_opt_addv_ref
"""


def _build_sdfg_and_iface(tmp_path: Path):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_KERNEL, sdfg_dir, name="opt_addv", entry="opt_addv_mod::opt_addv").build()
    sdfg.name = "opt_addv"
    iface = build_auto_interface(sdfg._fortran_interface_raw, "opt_addv")
    return sdfg, iface


def _emit(tmp_path: Path, sdfg, iface) -> str:
    bindings_path = tmp_path / "opt_addv_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    return bindings_path.read_text()


def test_sdfg_side_models_array_optional_presence(tmp_path: Path):
    """CONTROL (passes at 6c99810): the SDFG itself branches correctly on
    ``a_present`` -- the defect is only in the emitted wrapper."""
    sdfg, _iface = _build_sdfg_and_iface(tmp_path)
    assert "a_present" in {str(s) for s in sdfg.free_symbols}

    y = np.zeros(4)
    a = np.array([1.0, 2.0, 3.0, 4.0])
    sdfg(n=np.int32(4), y=y, a=a, a_present=1)
    np.testing.assert_allclose(y, a)

    y2 = np.zeros(4)
    sdfg(n=np.int32(4), y=y2, a=a, a_present=0)
    np.testing.assert_allclose(y2, np.zeros(4))


def test_array_optional_presence_forwarded_in_binding(tmp_path: Path):
    """FAILS at 6c99810: the wrapper declares ``a`` optional and forwards its
    DATA via ``c_loc(a)``, but hard-sets ``a_present = 0``."""
    sdfg, iface = _build_sdfg_and_iface(tmp_path)
    text = _emit(tmp_path, sdfg, iface)
    assert "optional :: a(" in text, text
    # Presence must be sourced from the actual argument, exactly like the
    # (already-fixed) scalar-optional case:
    assert "a_present = int(merge(1, 0, present(a)), c_int)" in text, \
        "array-optional presence not forwarded; found hardwired default instead"
    assert "a_present = 0" not in text, \
        "wrapper still hard-sets the array optional's presence to absent"


def test_array_optional_present_call_matches_reference(tmp_path: Path):
    """FAILS at 6c99810: calling the wrapper WITH ``a`` present must add it
    (currently the kernel's PRESENT-gated region is dead and y is returned
    unchanged, byte-identical to the input)."""
    sdfg, iface = _build_sdfg_and_iface(tmp_path)
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    bindings_path = tmp_path / "opt_addv_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    driver_path = tmp_path / "opt_addv_driver.f90"
    driver_path.write_text(_SDFG_DRIVER)

    build_dir = tmp_path / "bind_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    drv_so = build_dir / "opt_addv_drv.so"
    gfortran_compile_so(drv_so, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)

    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "k.f90").write_text(_KERNEL)
    (ref_dir / "d.f90").write_text(_REF_DRIVER)
    ref_so = ref_dir / "opt_addv_ref.so"
    gfortran_compile_so(ref_so, ref_dir / "k.f90", ref_dir / "d.f90", mod_dir=ref_dir)

    lib = ctypes.CDLL(str(drv_so))
    ref = ctypes.CDLL(str(ref_so))

    def call(fn, has):
        dp = ctypes.POINTER(ctypes.c_double)
        y = np.zeros(4)
        a = np.array([1.0, 2.0, 3.0, 4.0])
        fn.restype = None
        fn.argtypes = [ctypes.c_int, dp, dp, ctypes.c_int]
        fn(4, y.ctypes.data_as(dp), a.ctypes.data_as(dp), has)
        return y

    # Present: y must become a (reference does; wrapper must match).
    y_bind = call(lib.run_opt_addv, 1)
    y_ref = call(ref.run_opt_addv_ref, 1)
    np.testing.assert_allclose(y_ref, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y_bind, y_ref, err_msg="present-call no-op: a_present hardwired to 0 in the wrapper")

    # Absent: y stays zero on both sides.
    np.testing.assert_allclose(call(lib.run_opt_addv, 0), np.zeros(4))
    np.testing.assert_allclose(call(ref.run_opt_addv_ref, 0), np.zeros(4))
