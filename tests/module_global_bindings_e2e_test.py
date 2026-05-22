"""End-to-end binding tests for module-global classification.

Companion to ``module_global_vs_constant_test.py``, which exercises the
SDFG directly.  Here every case goes through the *generated Fortran
binding*: the SDFG is compiled, ``build_fortran_library`` emits + links
the ``<entry>_dace`` wrapper, and a C-bound shim calls that wrapper
exactly as a host would.  This is the only path that exercises the
binding's module-global copy-in (host module variable -> SDFG arg) and,
for a kernel-written global, copy-out / write-back (SDFG arg -> host
module variable on exit).

Each case is compared against a gfortran reference: the SAME shim, but
calling the original (un-transformed) subroutine.  The reference's host
module variable is read the same way, so a correct binding reproduces
both the computed output and the post-call module-global value.
"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings.build_fortran_library import build_fortran_library
from dace_fortran.bindings.fortran_interface import OriginalArg, OriginalInterface
from dace_fortran.bindings.flatten_plan import FlattenPlan

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")

_N = 4


def _iface(sub: str) -> OriginalInterface:
    """Caller-facing surface of ``<sub>(x(4), y(4))`` -- the two real
    dummies; module globals are reached via ``USE`` inside the wrapper,
    not as dummies."""
    return OriginalInterface(
        entry=sub,
        args=(
            OriginalArg(name='x', fortran_type='real(c_double)', rank=1, shape=(str(_N), ), intent='in'),
            OriginalArg(name='y', fortran_type='real(c_double)', rank=1, shape=(str(_N), ), intent='out'),
        ),
    )


def _shim(call_target: str, target_use, uses, sets, reads) -> str:
    """Render a ``bind(c)`` shim ``run_kern(xin, yout, <set...>, <read...>)``
    that assigns each input module global before the call and reads each
    written module global after it.

    :param call_target: ``<sub>_dace`` (generated wrapper) or ``<sub>``
        (original subroutine).
    :param target_use: ``(module, name)`` to ``USE`` for ``call_target``
        (the binding module for the wrapper, or the kernel module for the
        original subroutine).
    :param uses: ``[(module, name), ...]`` of module globals to ``USE``.
    :param sets: ``[(name, dummy), ...]`` -- ``name = dummy`` before call.
    :param reads: ``[(name, dummy), ...]`` -- ``dummy = name`` after call.
    """
    use_lines = "\n".join(f"  use {mod}, only: {nm}" for mod, nm in (target_use, *uses))
    set_decls = "\n".join(f"  real(c_double), intent(in) :: {d}" for _, d in sets)
    read_decls = "\n".join(f"  real(c_double), intent(out) :: {d}" for _, d in reads)
    set_lines = "\n".join(f"  {nm} = {d}" for nm, d in sets)
    read_lines = "\n".join(f"  {d} = {nm}" for nm, d in reads)
    extra = "".join(f", {d}" for _, d in (*sets, *reads))
    return f"""
subroutine run_kern(xin, yout{extra}) bind(c, name="run_kern")
  use iso_c_binding
{use_lines}
  real(c_double), intent(in) :: xin({_N})
  real(c_double), intent(out) :: yout({_N})
{set_decls}
{read_decls}
{set_lines}
  call {call_target}(xin, yout)
{read_lines}
end subroutine run_kern
"""


def _gfortran(out_so: Path, *sources, mod_dir: Path):
    subprocess.check_call([
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
        f"-J{mod_dir}", *[str(s) for s in sources], "-o",
        str(out_so)
    ],
                          cwd=mod_dir)


def _invoke(lib, x, set_vals, n_read):
    """Call ``run_kern`` through ``lib`` and return ``(y, [read values])``."""
    y = np.zeros(_N, dtype=np.float64, order='F')
    set_bufs = [np.array([v], dtype=np.float64, order='F') for v in set_vals]
    read_bufs = [np.zeros(1, dtype=np.float64, order='F') for _ in range(n_read)]
    fn = lib.run_kern
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p] * (2 + len(set_bufs) + len(read_bufs))
    fn(x.ctypes.data, y.ctypes.data, *[b.ctypes.data for b in set_bufs], *[b.ctypes.data for b in read_bufs])
    return y, [float(b[0]) for b in read_bufs]


def _e2e(tmp_path, name, src, *, kern_mod, sub, uses=(), sets=(), reads=()):
    """Build ``<kern_mod>::<sub>`` through the generated binding and a
    gfortran reference, run both, and assert the computed output and every
    read-back module-global value agree.

    :param uses: ``[(module, name), ...]`` the shim ``USE``s.
    :param sets: ``[(name, value), ...]`` input globals set before the call.
    :param reads: ``[name, ...]`` written globals read back after the call.
    """
    src_path = tmp_path / f"{name}.f90"
    src_path.write_text(src)
    set_pairs = [(nm, f"gset{i}") for i, (nm, _) in enumerate(sets)]
    read_pairs = [(nm, f"gread{i}") for i, nm in enumerate(reads)]
    set_vals = [v for _, v in sets]

    sdfg_dir = tmp_path / "sdfg"
    sdfg_dir.mkdir(parents=True, exist_ok=True)
    builder = build_sdfg(src, sdfg_dir, name=sub, entry=f"_QM{kern_mod}P{sub}")
    plan = FlattenPlan.from_dict(builder.module.get_flatten_plan())
    sdfg = builder.build()
    sdfg.validate()
    # Reset the per-test name suffix the test helper added: the binding's C
    # entry symbols (``__dace_init_<name>`` etc.) must match ``iface.entry``.
    # A per-tmp_path build_folder keeps the (now-shared-name) builds isolated.
    sdfg.name = sub
    sdfg.build_folder = str(tmp_path / "dacecache")

    dace_shim = tmp_path / "dace_shim.f90"
    dace_shim.write_text(_shim(f"{sub}_dace", (f"{sub}_dace_bindings", f"{sub}_dace"), uses, set_pairs, read_pairs))
    lib = build_fortran_library(sdfg,
                                _iface(sub),
                                plan,
                                str(tmp_path / "lib"),
                                name=f"{sub}_lib",
                                prelude_sources=[src_path],
                                extra_sources=[dace_shim])
    dace_lib = lib.load()

    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_shim = ref_dir / "ref_shim.f90"
    ref_shim.write_text(_shim(sub, (kern_mod, sub), uses, set_pairs, read_pairs))
    ref_so = ref_dir / f"lib{sub}_ref.so"
    _gfortran(ref_so, src_path, ref_shim, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    x = np.asfortranarray(np.arange(1, _N + 1, dtype=np.float64))
    y_dace, reads_dace = _invoke(dace_lib, x, set_vals, len(read_pairs))
    y_ref, reads_ref = _invoke(ref_lib, x.copy(order='F'), set_vals, len(read_pairs))

    np.testing.assert_allclose(y_dace, y_ref, rtol=1e-12, err_msg="binding output disagrees with reference")
    for nm, dval, rval in zip(reads, reads_dace, reads_ref):
        np.testing.assert_allclose(dval,
                                   rval,
                                   rtol=1e-12,
                                   err_msg=f"module global {nm!r} write-back disagrees with reference")


def test_e2e_parameter_baked(tmp_path: Path):
    """A ``parameter`` baked into the SDFG reproduces the reference output
    through the binding (no kwarg, no copy-in/out)."""
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
    _e2e(tmp_path, "param", src, kern_mod="mod_param", sub="apply_param")


def test_e2e_uninitialised_global_copy_in(tmp_path: Path):
    """An uninitialised module global is copied IN from the host module
    variable through the binding (the shim sets it before the call)."""
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
    _e2e(tmp_path,
         "cfg",
         src,
         kern_mod="mod_cfg",
         sub="apply_cfg",
         uses=[("mod_cfg", "cfg_scale")],
         sets=[("cfg_scale", 3.0)])


def test_e2e_initialised_readonly_global_baked(tmp_path: Path):
    """An initialised read-only global bakes its default; the binding
    reproduces the reference output."""
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
    _e2e(tmp_path, "init", src, kern_mod="mod_init", sub="apply_init")


def test_e2e_written_global_writeback(tmp_path: Path):
    """A kernel-written module global is written BACK to the host module
    variable on exit: after the binding call, the host's ``counter`` holds
    the kernel's updated value, matching the reference."""
    src = """
module mod_acc
  implicit none
  real(8) :: counter = 100.0d0
contains
  subroutine bump(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    counter = counter + 1.0d0
    do i = 1, 4
      y(i) = x(i) + counter
    end do
  end subroutine bump
end module mod_acc
"""
    _e2e(tmp_path,
         "acc",
         src,
         kern_mod="mod_acc",
         sub="bump",
         uses=[("mod_acc", "counter")],
         sets=[("counter", 100.0)],
         reads=["counter"])


def test_e2e_written_global_no_init_writeback(tmp_path: Path):
    """A kernel-written global with NO initialiser is still written back to
    the host module variable through the binding."""
    src = """
module mod_scr
  implicit none
  real(8) :: tmpval
contains
  subroutine use_tmp(x, y)
    real(8), intent(in) :: x(4)
    real(8), intent(out) :: y(4)
    integer :: i
    tmpval = 7.0d0
    do i = 1, 4
      y(i) = x(i) * tmpval
    end do
  end subroutine use_tmp
end module mod_scr
"""
    _e2e(tmp_path,
         "scr",
         src,
         kern_mod="mod_scr",
         sub="use_tmp",
         uses=[("mod_scr", "tmpval")],
         sets=[("tmpval", 0.0)],
         reads=["tmpval"])


def test_e2e_cross_module_written_writeback(tmp_path: Path):
    """A global declared in one module, ``USE``-imported and written by a
    kernel in another, is written back to its declaring module variable
    through the binding."""
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
    _e2e(tmp_path,
         "xstate",
         src,
         kern_mod="mod_kern_b",
         sub="use_state",
         uses=[("mod_state_x", "accum")],
         sets=[("accum", 1.0)],
         reads=["accum"])
