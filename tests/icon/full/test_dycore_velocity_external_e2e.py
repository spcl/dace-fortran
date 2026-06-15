"""Dycore + external velocity_tendencies E2E (xfail anchor pre-v2).

This is the velocity-scale, struct-shaped E2E of the dycore +
external-SDFG pattern.  The architecture proof at small scale
(``test_dycore_struct_ext_e2e.py`` -- ``state_t{u, v}`` + per-member
SoA) is generalised here to the actual ICON ``velocity_tendencies``
signature (five derived types -- ``t_nh_prog`` / ``t_patch`` /
``t_int_state`` / ``t_nh_metrics`` / ``t_nh_diag`` -- plus three
naked rank-3 arrays plus scalars).

Architecture under test:

  1. **Inner SDFG** (the velocity stand-in): built from
     ``velocity_full.f90`` with
     ``build_fortran_library(..., bind_c_shim=True)`` --
     ``libvelocity_inner_wrap.so`` exposes ``velocity_tendencies_c``
     with one ``c_ptr`` per leaf of the marshal expansion (the
     ``bind_c_shim`` emits the per-member C ABI for each derived
     type, matching what the outer's ``emit_call`` will forward).

  2. **Outer SDFG** (the dycore stand-in): a thin
     ``dycore_wrapper`` subroutine that takes the same arg list as
     ``velocity_tendencies`` and just calls it via an ``interface``
     block.  ``velocity_tendencies`` registers as
     ``keep_external(c_name='velocity_tendencies_c',
     libraries=[inner.so_path])`` with each derived-type arg
     declared ``Arg(kind='aos', c_abi='per_member_soa')`` -- the
     per-member SoA pointers the marshal expansion produces forward
     directly into ``velocity_tendencies_c``, no AoS struct buffer.

  3. **Caller**: drives the outer via standard
     ``build_fortran_library`` bindings (``dycore_wrapper_dace``
     entry); a flat-C-ABI shim derived from the proven
     ``run_velocity_flat_c`` swaps the target call from
     ``velocity_tendencies`` to ``dycore_wrapper_dace`` so the same
     ``ctypes`` driver runs both paths.

  4. **Reference**: the existing gfortran reference of the
     un-transformed ``velocity_tendencies`` + ``run_velocity_flat_c``.

**Why xfail today**: ``hlfir-marshal-external-structs`` v2.1
(``4de6c7a``) covers nested-record members but explicitly does not
cover *box / pointer / allocatable / dynamic-shape* members --
exactly what ``t_nh_prog`` / ``t_patch`` / etc. carry.  The bridge
build of the outer SDFG therefore raises in ``emit_call`` with the
diagnostic "``'aos' arg #0 has no marshalling group``" -- the
same boundary anchored by
``test_v2_aos_external_with_nested_struct``'s xfail-flipped peer
``test_v2_aos_external_diagnostic_mentions_inline_external``.  When
v2 box / allocatable expansion lands this test flips green; the
``xfail(strict=True)`` makes the flip visible.
"""
import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dace
from _util import build_sdfg, have_flang
from icon.full._harness import _INIT_ARRAY_ORDER, _OUTPUT_NAMES, _allocate

from dace_fortran.bindings import (
    FlattenPlan,
    OriginalArg,
    OriginalInterface,
    build_fortran_library,
)

# ``-O0 -fno-fast-math -ffp-contract=off`` matched across every build
# layer so the SDFG path's arithmetic order matches the gfortran
# reference exactly.  ``_O0_FFLAGS`` overrides
# ``build_fortran_library`` 's ``-O3 -frounding-math`` default for
# both the inner velocity wrapper and the outer dycore wrapper;
# ``_O0_CXX_FLAGS`` overrides DaCe's ``compiler.cpu.args`` default
# of ``-O3 -march=native -ffast-math`` which would otherwise
# contract ``a*b + c`` into FMA and add ~1 ULP per element to the
# diff.  Pinned to expose any genuine numerical error in the
# external-call routing -- the e2e is a regression gate, not a
# tolerance-shopping target.
_O0_FFLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off",
              "-ffree-line-length-none")
_O0_CXX_FLAGS = ("-O0", "-fno-fast-math", "-ffp-contract=off",
                 "-fPIC", "-Wno-unused-parameter", "-Wno-unused-label")
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_HERE = Path(__file__).resolve().parent
_VELOCITY_PATH = _HERE / "velocity_full.f90"
_CALLER_PATH = _HERE / "velocity_full_caller.f90"


# Module globals the inner ``velocity_tendencies_dace`` kernel
# reads.  Each tuple is ``(module, member, dtype, rank)``: the
# inner ``bind_c_shim`` exposes them as additional C ABI args, and
# the outer ``emit_call`` reads ``__<mod>_MOD_<member>`` from the
# OUTER library's BSS (which its wrapper has already populated
# from ``run_velocity_flat_sdfg`` 's caller args via the existing
# ``use ...`` import path).  Without this, gfortran's per-library
# module BSS leaves the inner copy at zero -- the diagnostic ASan
# report for the velocity dycore + external e2e xfail traced the
# kernel's ``new double[expr=nproma*..]`` to a 1-byte sentinel for
# exactly this reason.  Order = the
# ``_velocity_iface.module_symbol_sources`` keys; the dtype +
# rank columns match each module member's declaration in
# ``velocity_full.f90``.
_VELOCITY_MODULE_FORWARD = (
    ("mo_parallel_config", "nproma", "int32", 0),
    ("mo_run_config", "timers_level", "int32", 0),
    ("mo_vertical_grid", "nrdmax", "int32", 1),
    ("mo_init_vgrid", "nflatlev", "int32", 1),
    ("mo_mpi", "i_am_accel_node", "bool", 0),
    ("mo_nonhydrostatic_config", "lextra_diffu", "bool", 0),
    ("mo_run_config", "lvert_nest", "bool", 0),
    ("mo_timer", "timer_intp", "int32", 0),
    ("mo_timer", "timer_solve_nh_veltend", "int32", 0),
)


# Two side-effecting "sync" externals exercise the rest of the
# external-call surface: (a) a plain Fortran subroutine that the
# dycore wrapper CALLs directly (no ``bind(c)``), bridged into the
# SDFG via a small Fortran ``bind(c)`` shim that the registration
# points at; (b) a C++ implementation registered the same way.
#
# The bodies print to stderr with a unique marker so the test can
# assert -- by tailing the subprocess child log -- that BOTH the
# Fortran-path and the C++-path externals actually fired (the bridge
# is not allowed to optimise ``keep_external`` calls away; the prints
# make a missed routing visible immediately).  Both syncs leave the
# array unchanged so the numerical comparison against the gfortran
# reference is unaffected (the reference path doesn't go through the
# dycore wrapper and thus doesn't run the syncs -- the SDFG path's
# extra "no-op" external calls are observed via the stderr markers,
# not the numerical output).
_SYNC_FORTRAN_SRC = """
module mo_sync_helper
  use iso_c_binding
  implicit none
contains
  ! Original Fortran ``sync_patch_array`` -- NO ``bind(c)``.  The
  ! dycore wrapper CALLs this through the Fortran interface.  The
  ! bridge sees a regular Fortran call site and, with the
  ! ``keep_external`` registration below, emits an SDFG library
  ! node routed to the ``sync_patch_array_c`` ``bind(c)`` wrapper.
  subroutine sync_patch_array(tag, field)
    integer(c_int), intent(in) :: tag
    real(c_double), intent(inout) :: field(:, :, :)
    write(0, '(A,I0,A,3I6,A,ES13.6)') '[sync_patch_array Fortran] tag=', tag, &
                                       ' shape=', shape(field), &
                                       ' first=', field(1, 1, 1)
    flush(0)
  end subroutine sync_patch_array

  ! ``bind(c)`` wrapper the SDFG actually invokes (the c_name the
  ! ``keep_external`` registration points at).  Receives flat
  ! pointer + extents at the C ABI, rebuilds the assumed-shape
  ! descriptor with ``c_f_pointer``, calls the original
  ! ``sync_patch_array`` so the side effect (the stderr print)
  ! still runs from the original-Fortran code path.
  subroutine sync_patch_array_c(tag, d0, d1, d2, field_p) &
    bind(c, name='sync_patch_array_c')
    integer(c_int), value :: tag, d0, d1, d2
    type(c_ptr), value :: field_p
    real(c_double), pointer :: field_local(:, :, :)
    call c_f_pointer(field_p, field_local, [d0, d1, d2])
    call sync_patch_array(tag, field_local)
  end subroutine sync_patch_array_c

  ! Fortran-side companion the dycore wrapper CALLs; its body is
  ! a Fortran interface block + a forwarded call to the C++
  ! ``sync_patch_cpp`` impl.  When the SDFG path routes through
  ! ``sync_patch_cpp_via_c`` below the body is NOT run (the bridge
  ! externalises the symbol); when the gfortran reference link
  ! runs it (which the velocity-e2e doesn't, but kept here for
  ! symmetry with the Fortran sync above), the C++ impl still
  ! fires because the wrapper goes through the same interface
  ! block.
  subroutine sync_patch_cpp_via(tag, field)
    integer(c_int), intent(in) :: tag
    real(c_double), target, intent(inout) :: field(:, :, :)
    interface
      subroutine sync_patch_cpp_impl(tag, d0, d1, d2, field_p) &
        bind(c, name='sync_patch_cpp')
        use iso_c_binding
        integer(c_int), value :: tag, d0, d1, d2
        type(c_ptr), value :: field_p
      end subroutine
    end interface
    call sync_patch_cpp_impl(tag, &
                             int(size(field, 1), c_int), &
                             int(size(field, 2), c_int), &
                             int(size(field, 3), c_int), &
                             c_loc(field))
  end subroutine sync_patch_cpp_via

  ! ``bind(c)`` wrapper for the C++ sync -- mirrors the Fortran
  ! sync's pattern.  The SDFG calls this; this calls the C++
  ! ``sync_patch_cpp`` directly.
  subroutine sync_patch_cpp_via_c(tag, d0, d1, d2, field_p) &
    bind(c, name='sync_patch_cpp_via_c')
    integer(c_int), value :: tag, d0, d1, d2
    type(c_ptr), value :: field_p
    interface
      subroutine sync_patch_cpp_impl(tag, d0, d1, d2, field_p) &
        bind(c, name='sync_patch_cpp')
        use iso_c_binding
        integer(c_int), value :: tag, d0, d1, d2
        type(c_ptr), value :: field_p
      end subroutine
    end interface
    call sync_patch_cpp_impl(tag, d0, d1, d2, field_p)
  end subroutine sync_patch_cpp_via_c
end module mo_sync_helper
"""


# C++ side of the sync external.  Reads + writes nothing through the
# pointer (so it can't perturb the numerical comparison) but does
# print a unique stderr marker the test can grep for.  Kept tiny so
# the build cost is negligible.
_SYNC_CPP_SRC = """
#include <cstdio>

extern "C" void sync_patch_cpp(int tag, int d0, int d1, int d2,
                               double *field) {
    double first = (field != nullptr) ? field[0] : 0.0;
    std::fprintf(stderr,
                 "[sync_patch_cpp C++] tag=%d shape=%d %d %d first=%g\\n",
                 tag, d0, d1, d2, first);
    std::fflush(stderr);
}
"""


# Dycore stand-in: a passthrough wrapper with the exact
# velocity_tendencies signature.  The bridge sees ``call
# velocity_tendencies(...)``; with the callee registered as
# ``keep_external``, ``hlfir-marshal-external-structs`` is asked to
# expand each derived-type arg into its per-member leaves, then
# ``emit_call`` emits the C call directly into
# ``velocity_tendencies_c`` exported by the inner SDFG.  The wrapper
# also CALLs ``sync_patch_array`` (Fortran-no-bind-c) and
# ``sync_patch_cpp_via`` (Fortran wrapper forwarding to a C++ impl)
# so the SDFG exercises three distinct external-call shapes in one
# kernel.
_DYCORE_WRAPPER_SRC = """
module mo_dycore_wrapper
  use iso_c_binding
  use mo_sync_helper, only: sync_patch_array, sync_patch_cpp_via
contains
  subroutine dycore_wrapper(p_prog, p_patch, p_int, p_metrics, p_diag, &
                            z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                            ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
    use mo_model_domain,   only: t_patch
    use mo_intp_data_strc, only: t_int_state
    use mo_nonhydro_types, only: t_nh_prog, t_nh_diag, t_nh_metrics
    use mo_velocity_advection, only: velocity_tendencies
    type(t_patch),     target, intent(in)    :: p_patch
    type(t_int_state), target, intent(in)    :: p_int
    type(t_nh_prog),           intent(inout) :: p_prog
    type(t_nh_metrics),        intent(inout) :: p_metrics
    type(t_nh_diag),           intent(inout) :: p_diag
    real(kind=8), dimension(:, :, :), target, intent(inout) :: &
                       z_w_concorr_me, z_kin_hor_e, z_vt_ie
    integer, intent(in) :: ntnd, istep
    logical, intent(in) :: lvn_only
    real(kind=8), intent(in) :: dtime, dt_linintp_ubc
    logical, intent(in) :: ldeepatmo
    ! Pre-velocity Fortran sync external -- prints to stderr; the
    ! bridge externalises it via ``keep_external`` and routes to
    ! ``sync_patch_array_c``.  Tag = 1 so the test can pin the call
    ! site in the stderr log.
    call sync_patch_array(1_c_int, z_kin_hor_e)
    call velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, &
                             z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                             ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
    ! Post-velocity C++ sync external -- same shape, routes to
    ! ``sync_patch_cpp_via_c`` which calls the ``sync_patch_cpp``
    ! C++ impl.  Tag = 2.
    call sync_patch_cpp_via(2_c_int, z_kin_hor_e)
  end subroutine dycore_wrapper
end module mo_dycore_wrapper
"""


def _scalar(name, ftype, intent, stype=None):
    return OriginalArg(name=name, fortran_type=ftype, rank=0, shape=(), intent=intent,
                       struct_type=stype)


def _arr3(name, intent):
    return OriginalArg(name=name, fortran_type="real(8)", rank=3,
                       shape=(":", ":", ":"), intent=intent, struct_type=None)


# Same ``OriginalInterface`` shape ``test_velocity_full_bindings_e2e``
# uses for the inner velocity binding; reused here for both the inner
# (entry = ``velocity_tendencies``) and the outer (entry =
# ``dycore_wrapper`` -- same arg list, the passthrough wraps it).
def _velocity_iface(entry: str) -> OriginalInterface:
    return OriginalInterface(
        entry=entry,
        args=(
            _scalar("p_prog", "type(t_nh_prog)", "inout", "t_nh_prog"),
            _scalar("p_patch", "type(t_patch)", "in", "t_patch"),
            _scalar("p_int", "type(t_int_state)", "in", "t_int_state"),
            _scalar("p_metrics", "type(t_nh_metrics)", "inout", "t_nh_metrics"),
            _scalar("p_diag", "type(t_nh_diag)", "inout", "t_nh_diag"),
            _arr3("z_w_concorr_me", "inout"),
            _arr3("z_kin_hor_e", "inout"),
            _arr3("z_vt_ie", "inout"),
            _scalar("ntnd", "integer", "in"),
            _scalar("istep", "integer", "in"),
            _scalar("lvn_only", "logical", "in"),
            _scalar("dtime", "real(8)", "in"),
            _scalar("dt_linintp_ubc", "real(8)", "in"),
            _scalar("ldeepatmo", "logical", "in"),
        ),
        struct_types={},
        used_modules={
            "mo_model_domain": ("t_patch", ),
            "mo_intp_data_strc": ("t_int_state", ),
            "mo_nonhydro_types": ("t_nh_prog", "t_nh_metrics", "t_nh_diag"),
        },
        module_symbol_sources={
            "nproma": ("mo_parallel_config", "nproma"),
            "timers_level": ("mo_run_config", "timers_level"),
            "nrdmax": ("mo_vertical_grid", "nrdmax"),
            "nflatlev": ("mo_init_vgrid", "nflatlev"),
            "i_am_accel_node": ("mo_mpi", "i_am_accel_node"),
            "lextra_diffu": ("mo_nonhydrostatic_config", "lextra_diffu"),
            "lvert_nest": ("mo_run_config", "lvert_nest"),
            "timer_intp": ("mo_timer", "timer_intp"),
            "timer_solve_nh_veltend": ("mo_timer", "timer_solve_nh_veltend"),
        },
    )


def _make_sdfg_shim_for_outer(caller_src: str) -> str:
    """Derive the SDFG-side shim from the proven flat caller: rename
    ``run_velocity_flat_c`` -> ``run_velocity_flat_sdfg``, retarget the
    kernel call from ``velocity_tendencies`` to ``dycore_wrapper_dace``
    (the outer SDFG's binding entry), and add the finalize call.
    Mirrors ``_make_sdfg_driver`` in ``test_velocity_full_bindings_e2e``
    but targeting the dycore wrapper instead of the velocity binding."""
    m = re.search(r"(?is)(SUBROUTINE\s+run_velocity_flat_c\b.*?END\s+SUBROUTINE\s+run_velocity_flat_c)",
                  caller_src)
    if not m:
        raise RuntimeError("run_velocity_flat_c not found in caller source")
    shim = m.group(1)
    shim = shim.replace("run_velocity_flat_c", "run_velocity_flat_sdfg")
    shim = shim.replace(
        "USE mo_velocity_advection,  ONLY: velocity_tendencies",
        "USE dycore_wrapper_dace_bindings, ONLY: dycore_wrapper_dace, "
        "dycore_wrapper_dace_finalize",
    )
    shim = shim.replace("CALL velocity_tendencies(p_prog, p_patch",
                        "CALL dycore_wrapper_dace(p_prog, p_patch")
    shim = re.sub(
        r"(?i)\bEND\s+SUBROUTINE\s+run_velocity_flat_sdfg",
        "  CALL dycore_wrapper_dace_finalize()\nEND SUBROUTINE run_velocity_flat_sdfg",
        shim,
    )
    return shim


def _gfortran(out_so: Path, *sources, mod_dir: Path, link_so: Path | None = None):
    cmd = [
        "gfortran", "-shared", "-fPIC", "-O0", "-fno-fast-math", "-ffp-contract=off",
        "-ffree-line-length-none", f"-J{mod_dir}",
    ]
    cmd += [str(s) for s in sources]
    cmd += ["-o", str(out_so)]
    if link_so is not None:
        cmd += [f"-L{link_so.parent}", f"-Wl,-rpath,{link_so.parent}", f"-l:{link_so.name}"]
    subprocess.check_call(cmd, cwd=mod_dir)


def _run(lib, fn, dims, bufs, z_arrays):
    """Invoke ``fn`` in ``lib`` with the standard velocity-flat C ABI
    (``dims``, scalars, then every input/output buffer pointer).

    Raises :class:`RuntimeError` if the call C-aborts -- runs in a
    subprocess and re-raises on non-zero exit so xfail can catch it
    instead of having SIGABRT terminate pytest.
    """
    # The call may SIGABRT inside the SDFG's external chain; isolate
    # it in a subprocess so xfail can observe a Python-side failure
    # rather than a process-level abort.
    import multiprocessing as mp

    nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v = dims
    buf_views = {k: (np.asarray(v).tobytes(), v.shape, v.dtype.str)
                 for k, v in bufs.items()}
    z_views = [(z.tobytes(), z.shape, z.dtype.str) for z in z_arrays]
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    # Redirect child stdout/stderr to a file so the debug prints
    # written by the bind_c_shim survive the SIGABRT path (the parent
    # tails the file on failure to surface where the child died).
    import tempfile
    log_path = tempfile.mktemp(prefix=f"{fn}_", suffix=".log")
    p = ctx.Process(target=_run_child,
                    args=(str(lib._name), fn, dims, buf_views, z_views,
                          q, log_path))
    p.start()
    p.join()
    if p.exitcode != 0:
        try:
            with open(log_path) as f:
                log_tail = "\n".join(f.read().splitlines()[-30:])
        except OSError:
            log_tail = "<no log captured>"
        raise RuntimeError(
            f"{fn} aborted (exitcode={p.exitcode}).  Last child output:\n"
            f"{log_tail}")
    out_bufs, out_z = q.get(timeout=5)
    for k, (raw, shape, dtype) in out_bufs.items():
        bufs[k][:] = np.frombuffer(raw, dtype=dtype).reshape(shape, order='F')
    for i, (raw, shape, dtype) in enumerate(out_z):
        z_arrays[i][:] = np.frombuffer(raw, dtype=dtype).reshape(shape, order='F')
    # Return the child's captured stderr so the caller can assert
    # the sync-external prints actually fired (the bridge is not
    # allowed to optimise ``keep_external`` calls away; the markers
    # in the log make a missed routing visible).
    try:
        with open(log_path) as f:
            return f.read()
    except OSError:
        return ""


def _run_child(lib_path, fn, dims, buf_views, z_views, q, log_path):
    """Subprocess body for :func:`_run`: load the library, reconstruct
    the buffers, invoke ``fn``, and ship the post-call buffers back
    over ``q``.  Redirects stdout/stderr to ``log_path`` so any
    ``write(0, *) ...`` debug output from the bind_c_shim survives
    the SIGABRT path."""
    import os
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    lib = ctypes.CDLL(lib_path)
    bufs = {k: np.frombuffer(raw, dtype=dtype).copy().reshape(shape, order='F')
            for k, (raw, shape, dtype) in buf_views.items()}
    z_arrays = [np.frombuffer(raw, dtype=dtype).copy().reshape(shape, order='F')
                for raw, shape, dtype in z_views]
    f = getattr(lib, fn)
    f.restype = None
    f.argtypes = ([ctypes.c_int] * 6 + [ctypes.c_int, ctypes.c_int] +
                  [ctypes.c_int8, ctypes.c_int8] +
                  [ctypes.c_double, ctypes.c_double] +
                  [ctypes.c_void_p, ctypes.c_void_p] +
                  [ctypes.c_int8, ctypes.c_int8, ctypes.c_int] +
                  [ctypes.c_void_p] * (len(_INIT_ARRAY_ORDER) + 3))
    nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v = dims
    nrdmax_in = np.full(10, nlev, dtype=np.int32, order='F')
    nflatlev_in = np.ones(10, dtype=np.int32, order='F')
    f(nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v, 1, 1, 0, 0, 60.0, 0.0,
      nrdmax_in.ctypes.data, nflatlev_in.ctypes.data, 0, 0, 0,
      *[bufs[k].ctypes.data for k in _INIT_ARRAY_ORDER],
      *[z.ctypes.data for z in z_arrays])
    out_bufs = {k: (v.tobytes(), v.shape, v.dtype.str)
                for k, v in bufs.items()}
    out_z = [(z.tobytes(), z.shape, z.dtype.str) for z in z_arrays]
    q.put((out_bufs, out_z))


def _build_sync_helpers(tmp_path: Path) -> Path:
    """Pre-compile the sync side-effect library.  Produces
    ``libsync_helpers.so`` containing the ``mo_sync_helper`` Fortran
    module + the ``sync_patch_cpp`` C++ impl.  The SDFG kernel's
    link command picks the .so up via the ``keep_external`` library
    list so calls to ``sync_patch_array_c`` / ``sync_patch_cpp_via_c``
    / ``sync_patch_cpp`` resolve at SDFG load time."""
    build_dir = tmp_path / "sync_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    fortran_src = build_dir / "mo_sync_helper.f90"
    fortran_src.write_text(_SYNC_FORTRAN_SRC)
    cpp_src = build_dir / "sync_patch_cpp.cpp"
    cpp_src.write_text(_SYNC_CPP_SRC)
    fortran_obj = build_dir / "mo_sync_helper.o"
    cpp_obj = build_dir / "sync_patch_cpp.o"
    so_path = build_dir / "libsync_helpers.so"
    subprocess.check_call(
        ["gfortran", "-fPIC", "-O0", "-g", f"-J{build_dir}",
         "-c", str(fortran_src), "-o", str(fortran_obj)],
        cwd=build_dir)
    subprocess.check_call(
        ["g++", "-fPIC", "-O0", "-g", "-c", str(cpp_src), "-o", str(cpp_obj)],
        cwd=build_dir)
    subprocess.check_call(
        ["gfortran", "-shared", "-fPIC", "-O0", "-g",
         str(fortran_obj), str(cpp_obj), "-o", str(so_path)],
        cwd=build_dir)
    return so_path


def test_dycore_outer_calls_velocity_sdfg_via_c_abi(tmp_path: Path):
    """The dycore SDFG calls the standalone velocity_tendencies SDFG
    over the C ABI, with each derived-type arg crossing via
    per-member SoA pointers (``Arg(kind='aos',
    c_abi='per_member_soa')``).  The dycore wrapper ALSO calls a
    Fortran-side ``sync_patch_array`` and a C++-side
    ``sync_patch_cpp`` via Fortran wrappers, exercising the
    no-bind-c-Fortran + bind-c-wrapper pattern and the C++ external
    pattern side-by-side with the AoS-marshalled velocity call.
    Random inputs from the existing velocity harness; reference is
    the gfortran-compiled velocity_tendencies + run_velocity_flat_c
    driver.  Numerical comparison element-by-element on every
    output array; the sync prints are asserted via the subprocess
    child log."""
    # ---- 0. Pre-build the sync-helpers side library ----
    sync_lib_so = _build_sync_helpers(tmp_path)
    # ---- 0b. Pin DaCe's C++ codegen to ``-O0 -fno-fast-math
    #          -ffp-contract=off`` so the inner + outer SDFG
    #          kernels' arithmetic order matches the gfortran
    #          reference exactly.  Save + restore around the build
    #          so unrelated tests aren't perturbed. ----
    _orig_cxx_args = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args",
                    value=" ".join(_O0_CXX_FLAGS))
    # ---- 1. Inner velocity SDFG with bind_c_shim ----
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir(parents=True, exist_ok=True)
    inner_sdfg_dir = inner_dir / "sdfg"
    inner_sdfg_dir.mkdir(parents=True, exist_ok=True)
    clear_external_registry()
    velocity_src = _VELOCITY_PATH.read_text()
    inner_sdfg = build_sdfg(velocity_src, inner_sdfg_dir,
                            name="velocity_tendencies",
                            entry="velocity_tendencies").build()
    inner_sdfg.name = "velocity_tendencies"
    inner_sdfg.build_folder = str(inner_dir / "dacecache")
    # Bridge-derived ``OriginalInterface`` -- carries the
    # ``struct_types`` member layouts the bind_c_shim emitter needs
    # (populated since c7b1f41).  The hand-authored ``_velocity_iface``
    # below is for the outer-side ``build_fortran_library`` call where
    # ``module_symbol_sources`` matters.
    inner_iface = build_auto_interface(inner_sdfg._fortran_interface_raw,
                                       "velocity_tendencies")
    inner_plan = FlattenPlan.from_dict(inner_sdfg._flatten_plan_raw or {})
    inner_lib = build_fortran_library(
        inner_sdfg,
        iface=inner_iface,
        plan=inner_plan,
        out_dir=str(inner_dir / "lib"),
        name="velocity_inner_wrap",
        # The bind_c_shim ``use``s the ICON type modules
        # (``mo_nonhydro_types`` etc.).  Build velocity_full.f90 as a
        # prelude so its ``.mod`` files land in the bind dir before
        # the shim source needs them.
        prelude_sources=[_VELOCITY_PATH],
        bind_c_shim=True,
        bind_c_shim_module_symbol_forward=_VELOCITY_MODULE_FORWARD,
        # Match the reference's ``-O0 -fno-fast-math
        # -ffp-contract=off`` so the inner SDFG's arithmetic order
        # is identical at the binding-wrapper layer.
        flags=_O0_FFLAGS,
    )
    assert inner_lib.bind_c_shim_f90 is not None

    # ---- 2. Register velocity_tendencies as a per_member_soa external ----
    # ``c_name='velocity_tendencies_c'`` is the bind_c_shim entry on
    # the inner; each derived-type arg crosses as per-member SoA so
    # the outer's emit_call forwards the marshal-expanded leaves
    # verbatim into ``velocity_tendencies_c(...)``.
    keep_external(
        "velocity_tendencies",
        c_name="velocity_tendencies_c",
        args=(
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),  # p_prog
            Arg(kind="aos", intent="in", c_abi="per_member_soa"),     # p_patch
            Arg(kind="aos", intent="in", c_abi="per_member_soa"),     # p_int
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),  # p_metrics
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),  # p_diag
            Arg(kind="array", dtype="float64", intent="inout"),        # z_w_concorr_me
            Arg(kind="array", dtype="float64", intent="inout"),        # z_kin_hor_e
            Arg(kind="array", dtype="float64", intent="inout"),        # z_vt_ie
            Arg(kind="scalar", dtype="int32", intent="in"),            # ntnd
            Arg(kind="scalar", dtype="int32", intent="in"),            # istep
            Arg(kind="scalar", dtype="bool", intent="in"),             # lvn_only
            Arg(kind="scalar", dtype="float64", intent="in"),          # dtime
            Arg(kind="scalar", dtype="float64", intent="in"),          # dt_linintp_ubc
            Arg(kind="scalar", dtype="bool", intent="in"),             # ldeepatmo
        ),
        libraries=(str(inner_lib.so_path), ),
        # The inner library was built with ``bind_c_shim=True``; its
        # C ABI takes one ``int`` extent per dim ahead of every
        # dynamic-shape leaf pointer.  Tell emit_call to emit those.
        dynamic_extents_abi=True,
        # Forward every Fortran module global the inner kernel
        # reads.  Required: under default ELF+gfortran linking each
        # library has its own BSS copy, so the outer's
        # ``mo_parallel_config::nproma = N`` never reaches the
        # inner's copy without this explicit C-ABI bridge.  See
        # ``ExternalSignature.module_symbol_forward``.
        module_symbol_forward=_VELOCITY_MODULE_FORWARD,
    )
    # ---- 2b. Register the Fortran + C++ sync externals ----
    # Both have an identical Fortran-side signature
    # ``(tag: int, field: real(8)(:,:,:))`` and an identical
    # ``bind(c)`` wrapper signature
    # ``(tag, d0, d1, d2, field_p)``.  Set ``dynamic_extents_abi``
    # so emit_call prepends the per-dim ``int`` extents the wrapper
    # needs for its ``c_f_pointer`` reconstruction.  ``libraries``
    # points at the pre-built sync .so so the SDFG link resolves
    # both ``sync_patch_array_c`` and ``sync_patch_cpp_via_c``.
    _sync_args = (
        Arg(kind="scalar", dtype="int32", intent="in"),       # tag
        Arg(kind="array", dtype="float64", intent="inout"),    # field
    )
    keep_external(
        "sync_patch_array",
        c_name="sync_patch_array_c",
        args=_sync_args,
        libraries=(str(sync_lib_so), ),
        dynamic_extents_abi=True,
    )
    keep_external(
        "sync_patch_cpp_via",
        c_name="sync_patch_cpp_via_c",
        args=_sync_args,
        libraries=(str(sync_lib_so), ),
        dynamic_extents_abi=True,
    )
    try:
        # ---- 3. Outer dycore wrapper SDFG ----
        outer_dir = tmp_path / "outer"
        outer_dir.mkdir(parents=True, exist_ok=True)
        outer_sdfg_dir = outer_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        # Concat the sync helpers' Fortran source so flang sees
        # ``mo_sync_helper`` when it processes the dycore wrapper's
        # ``USE`` statement.  The ``sync_patch_*`` bodies are
        # externalised by the ``keep_external`` registrations above
        # so the bridge does NOT lower them into the dycore SDFG
        # kernel -- they survive as library nodes.
        outer_src = velocity_src + _SYNC_FORTRAN_SRC + _DYCORE_WRAPPER_SRC
        outer_sdfg = build_sdfg(outer_src, outer_sdfg_dir,
                                name="dycore_wrapper",
                                entry="dycore_wrapper").build()
        outer_sdfg.name = "dycore_wrapper"
        outer_sdfg.build_folder = str(outer_dir / "dacecache")
        outer_iface = _velocity_iface("dycore_wrapper")
        outer_plan = FlattenPlan.from_dict(outer_sdfg._flatten_plan_raw or {})
        # ---- 4. Write the SDFG shim retargeting run_velocity_flat_c
        #         -> run_velocity_flat_sdfg + dycore_wrapper_dace
        #         BEFORE the build consumes it as an extra_source. ----
        sdfg_shim = outer_dir / "sdfg_shim.f90"
        sdfg_shim.write_text(_make_sdfg_shim_for_outer(_CALLER_PATH.read_text()))
        outer_lib = build_fortran_library(
            outer_sdfg,
            iface=outer_iface,
            plan=outer_plan,
            out_dir=str(outer_dir / "lib"),
            name="dycore_wrapper",
            prelude_sources=[_VELOCITY_PATH, _CALLER_PATH],
            extra_sources=[sdfg_shim],
            bind_c_shim=False,
            flags=_O0_FFLAGS,
        )
    finally:
        clear_external_registry()
        dace.Config.set("compiler", "cpu", "args", value=_orig_cxx_args)

    sdfg_so = ctypes.CDLL(str(outer_lib.so_path))

    # ---- 5. Reference: gfortran-link velocity_full + caller ----
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_so = ref_dir / "libvelocity_ref.so"
    _gfortran(ref_so, _VELOCITY_PATH, _CALLER_PATH, mod_dir=ref_dir)
    ref_lib = ctypes.CDLL(str(ref_so))

    # ---- 6. Random inputs via the existing harness allocator ----
    nproma, nlev, nblks_c, nblks_e, nblks_v = 8, 6, 4, 4, 4
    nlevp1 = nlev + 1
    dims = (nproma, nlev, nlevp1, nblks_c, nblks_e, nblks_v)
    bufs_ref = _allocate(*dims)
    init = ref_lib.init_inputs_random_c
    init.restype = None
    init.argtypes = [ctypes.c_int] * 7 + [ctypes.c_void_p] * len(_INIT_ARRAY_ORDER)
    init(42, *dims, *[bufs_ref[k].ctypes.data for k in _INIT_ARRAY_ORDER])
    bufs_sdfg = {k: v.copy(order='F') for k, v in bufs_ref.items()}

    zshape = ((nproma, nlev, nblks_e), (nproma, nlev, nblks_e),
              (nproma, nlevp1, nblks_e))
    z_ref = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]
    z_sdfg = [np.zeros(s, dtype=np.float64, order='F') for s in zshape]

    # ---- 7. Run both paths and compare on every output ----
    _run(ref_lib, "run_velocity_flat_c", dims, bufs_ref, z_ref)
    sdfg_stderr = _run(sdfg_so, "run_velocity_flat_sdfg",
                       dims, bufs_sdfg, z_sdfg)

    # The dycore wrapper CALLs two side-effect externals that the
    # bridge routed through ``keep_external``: a Fortran-no-bind-c
    # ``sync_patch_array`` (via its ``sync_patch_array_c`` ``bind(c)``
    # wrapper) and a C++ ``sync_patch_cpp`` (via its
    # ``sync_patch_cpp_via`` Fortran wrapper -> ``sync_patch_cpp_via_c``
    # ``bind(c)`` wrapper).  Both bodies print a unique stderr marker
    # so a missed routing surfaces here.  The reference path (which
    # doesn't go through dycore_wrapper) doesn't print these, so the
    # assertions cover only the SDFG run.
    assert "[sync_patch_array Fortran] tag=1" in sdfg_stderr, \
        f"Fortran sync external did not fire.  SDFG stderr:\n{sdfg_stderr}"
    assert "[sync_patch_cpp C++] tag=2" in sdfg_stderr, \
        f"C++ sync external did not fire.  SDFG stderr:\n{sdfg_stderr}"

    # Numerical correctness with ``-O0 -fno-fast-math
    # -ffp-contract=off`` pinned across all three build layers
    # (DaCe C++ codegen, SDFG-binding gfortran link, reference
    # gfortran link).  Measured: every output is BIT-EXACT against
    # the gfortran reference today (worst rel = 0 ULP).  The
    # assertion below uses a 1-ULP envelope (``rtol = 2**-52``,
    # ``atol = 0``) as a safety buffer -- a real codegen regression
    # (FMA leak through ``-ffp-contract``, swapped operand order,
    # dropped parenthesisation) exceeds 1 ULP immediately on a
    # kernel as large as velocity_tendencies and trips here.  A
    # parallel ``assert_array_equal`` pin guarantees byte-for-byte
    # agreement on this exact source today; relax to
    # ``assert_allclose`` (above) if a future flang version
    # reorders a reduction.
    one_ulp_rtol = 2 ** -52  # ~2.22e-16
    extras = dict(zip(('z_w_concorr_me', 'z_kin_hor_e', 'z_vt_ie'),
                      zip(z_sdfg, z_ref)))
    per_output_max_rel = {}
    for nm in _OUTPUT_NAMES:
        sd, rf = extras[nm] if nm in extras else (bufs_sdfg[nm], bufs_ref[nm])
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.maximum(np.abs(rf), np.finfo(np.float64).tiny)
            rel = np.abs(sd - rf) / denom
            rel = np.where(np.isfinite(rel), rel, 0.0)
        per_output_max_rel[nm] = float(rel.max())
    worst_nm = max(per_output_max_rel, key=per_output_max_rel.get)
    worst_rel = per_output_max_rel[worst_nm]
    # Print so ``pytest -s`` users can read the envelope live.
    print(f"\n[velocity e2e] worst output {worst_nm!r}: "
          f"rel = {worst_rel:.3e} ({worst_rel / (2 ** -52):.1f} ULP)")
    for nm in _OUTPUT_NAMES:
        sd, rf = extras[nm] if nm in extras else (bufs_sdfg[nm], bufs_ref[nm])
        np.testing.assert_allclose(
            sd, rf, rtol=one_ulp_rtol, atol=0.0, equal_nan=True,
            err_msg=(f"output {nm!r} diverged beyond 1 ULP "
                     f"(rtol={one_ulp_rtol:.3e}).  Per-output "
                     f"rel-max: {per_output_max_rel}"))
        np.testing.assert_array_equal(
            sd, rf,
            err_msg=(f"output {nm!r} not bit-exact against the "
                     f"gfortran reference.  Per-output rel-max: "
                     f"{per_output_max_rel}.  If this fires on a "
                     f"new flang or new -O level, relax the "
                     f"``assert_array_equal`` to leave the 1-ULP "
                     f"``assert_allclose`` above as the gate."))
