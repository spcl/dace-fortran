"""M1: an OPTIONAL dummy's presence must forward through the binding.
Fix: wrapper sources <dummy>_present from present(x) (was hardwired to 0, so callers passing the
optional still got the absent branch). Pins both the emitted .f90 plumbing and numeric match vs an
untransformed gfortran reference."""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, gfortran_compile_so, have_flang
from dace_fortran.bindings import FlattenPlan, emit_bindings
from dace_fortran.bindings.fortran_interface import build_auto_interface

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_KERNEL = """
module opt_scale_mod
  implicit none
contains
subroutine opt_scale(a, out, scale)
  implicit none
  real(8), intent(in)  :: a(3)
  real(8), intent(out) :: out
  real(8), intent(in), optional :: scale
  if (present(scale)) then
    out = sum(a) * scale
  else
    out = sum(a)
  end if
end subroutine opt_scale
end module opt_scale_mod
"""

# bind(c) driver exercising BOTH present (has /= 0) and absent calls of the same generated wrapper.
_SDFG_DRIVER = """
subroutine run_opt(a, out, scale, has) bind(c, name='run_opt')
  use iso_c_binding
  use opt_scale_dace_bindings
  implicit none
  real(c_double), intent(in)  :: a(3)
  real(c_double), intent(out) :: out
  real(c_double), value :: scale
  integer(c_int), value :: has
  if (has /= 0) then
    call opt_scale_dace(a, out, scale)
  else
    call opt_scale_dace(a, out)
  end if
  call opt_scale_dace_finalize()
end subroutine run_opt
"""

_REF_DRIVER = """
subroutine run_opt_ref(a, out, scale, has) bind(c, name='run_opt_ref')
  use iso_c_binding
  use opt_scale_mod, only: opt_scale
  implicit none
  real(c_double), intent(in)  :: a(3)
  real(c_double), intent(out) :: out
  real(c_double), value :: scale
  integer(c_int), value :: has
  if (has /= 0) then
    call opt_scale(a, out, scale)
  else
    call opt_scale(a, out)
  end if
end subroutine run_opt_ref
"""


def _build_sdfg_and_iface(tmp_path: Path):
    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    sdfg = build_sdfg(_KERNEL, sdfg_dir, name="opt_scale", entry="opt_scale_mod::opt_scale").build()
    sdfg.name = "opt_scale"
    iface = build_auto_interface(sdfg._fortran_interface_raw, "opt_scale")
    return sdfg, iface


def test_optional_flag_in_auto_interface(tmp_path: Path):
    """Bridge exposes OPTIONAL attribute; build_auto_interface carries it onto the matching OriginalArg."""
    _sdfg, iface = _build_sdfg_and_iface(tmp_path)
    by_name = {a.name: a for a in iface.args}
    assert by_name["scale"].optional, "scale dummy must be flagged optional"
    assert not by_name["a"].optional
    assert not by_name["out"].optional


def test_present_forwarded_in_emitted_binding(tmp_path: Path):
    """Emitted wrapper declares the dummy optional, forwards present(scale) into scale_present, and
    routes the value through a guarded local."""
    sdfg, iface = _build_sdfg_and_iface(tmp_path)
    bindings_path = tmp_path / "opt_scale_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    text = bindings_path.read_text()
    # Outer dummy declared optional.
    assert "optional :: scale" in text, text
    # Presence forwarded from the actual, not hardwired to 0.
    assert "scale_present = int(merge(1, 0, present(scale)), c_int)" in text, text
    assert "scale_present = 0" not in text, text
    # Guarded local: declared, filled under present(scale), passed to the call.
    assert "scale__opt" in text, text
    assert "if (present(scale)) then" in text, text


def _compile(tmp_path: Path):
    sdfg, iface = _build_sdfg_and_iface(tmp_path)
    compiled = sdfg.compile()
    so_path = Path(compiled._lib._library_filename)

    bindings_path = tmp_path / "opt_scale_bindings.f90"
    emit_bindings(sdfg._frozen_signature, iface, FlattenPlan(entries=()), str(bindings_path))
    driver_path = tmp_path / "opt_scale_driver.f90"
    driver_path.write_text(_SDFG_DRIVER)

    build_dir = tmp_path / "bind_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    drv_so = build_dir / "opt_scale_drv.so"
    gfortran_compile_so(drv_so, bindings_path, driver_path, mod_dir=build_dir, link_so=so_path)

    ref_dir = tmp_path / "ref_build"
    ref_dir.mkdir(parents=True, exist_ok=True)
    k = ref_dir / "opt_scale_k.f90"
    k.write_text(_KERNEL)
    d = ref_dir / "opt_scale_d.f90"
    d.write_text(_REF_DRIVER)
    ref_so = ref_dir / "opt_scale_ref.so"
    gfortran_compile_so(ref_so, k, d, mod_dir=ref_dir)
    return ctypes.CDLL(str(drv_so)), ctypes.CDLL(str(ref_so)), sdfg


def _call(fn, a, scale, has):
    dp = ctypes.POINTER(ctypes.c_double)
    out = ctypes.c_double(0.0)
    fn.restype = None
    fn.argtypes = [dp, dp, ctypes.c_double, ctypes.c_int]
    fn(a.ctypes.data_as(dp), ctypes.byref(out), ctypes.c_double(scale), has)
    return out.value


def test_optional_present_and_absent_match_reference(tmp_path: Path):
    """Binding == gfortran reference for both present and absent calls; present branch actually
    multiplies by scale (scale_present no longer hardwired absent)."""
    lib, ref, sdfg = _compile(tmp_path)
    a = np.asfortranarray(np.array([1.0, 2.0, 4.0], dtype=np.float64))
    scale = 3.0

    # Present: out = sum(a) * scale = 7 * 3 = 21.
    bind_present = _call(lib.run_opt, a, scale, 1)
    ref_present = _call(ref.run_opt_ref, a, scale, 1)
    np.testing.assert_allclose(bind_present, ref_present, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bind_present, 21.0, rtol=1e-12, atol=1e-12)

    # Absent: out = sum(a) = 7 (scale ignored even though a value is passed).
    bind_absent = _call(lib.run_opt, a, scale, 0)
    ref_absent = _call(ref.run_opt_ref, a, scale, 0)
    np.testing.assert_allclose(bind_absent, ref_absent, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(bind_absent, 7.0, rtol=1e-12, atol=1e-12)

    # Direct SDFG ABI leg: the kernel branches on the ``scale_present`` symbol.
    out_p = np.zeros(1, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), out=out_p, scale=scale, scale_present=1)
    np.testing.assert_allclose(out_p[0], 21.0, rtol=1e-12, atol=1e-12)

    out_a = np.zeros(1, dtype=np.float64)
    sdfg(a=np.ascontiguousarray(a), out=out_a, scale=scale, scale_present=0)
    np.testing.assert_allclose(out_a[0], 7.0, rtol=1e-12, atol=1e-12)
