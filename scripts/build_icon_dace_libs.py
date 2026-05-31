"""Standalone driver: build the DaCe-backed ``libvelocity_inner_wrap.so``
(and ICON-side wrapper + CPU-mode stubs) that ICON will link against.

Lifts the build steps out of the ``test_dycore_velocity_external_e2e``
fixture so they can be run outside ``pytest`` -- typically once per
deployment, then ICON's normal configure + make picks the artifacts
up via ``$FCFLAGS`` + ``$LDFLAGS``.

See ``docs/ICON_INTEGRATION.md`` for the surrounding workflow.

Usage::

    python -m scripts.build_icon_dace_libs --out-dir $WORK/dace-icon-libs

The default ``--velocity-source`` is
``tests/icon_full/velocity_full.f90`` -- the pre-merged
self-contained ICON ``mo_velocity_advection`` source the e2e test
drives bit-exact.  Override only if you have your own merged
single-TU source; the bridge's ``merge_used_modules`` pass searches
the SDFG build dir, NOT the ICON source tree, so pointing at the
in-tree ``mo_velocity_advection.f90`` would fail to resolve
``USE mo_kind``, ``USE mo_nonhydro_types`` etc.

The script pins ``-O0 -fno-fast-math -ffp-contract=off`` on every
build layer (DaCe C++ codegen, the gfortran link of the
``bind_c_shim``, the gfortran link of the bindings wrapper) so the
ICON-vs-DaCe comparison stays bit-exact.  Switch to ``-O3
-fno-fast-math -ffp-contract=off`` with ``--release`` for production
timings (numerical envelope drops to 1 ULP).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import dace

# Re-use the iface + module-symbol-forward constants already pinned by
# the velocity e2e test so the standalone build produces *exactly* the
# artifacts the e2e is known to drive bit-exact.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from icon_full.test_dycore_velocity_external_e2e import (
    _VELOCITY_MODULE_FORWARD,
    _O0_FFLAGS,
    _O0_CXX_FLAGS,
    _velocity_iface,
    _CALLER_PATH,
    _SYNC_FORTRAN_SRC,
    _SYNC_CPP_SRC,
    _DYCORE_WRAPPER_SRC,
    _make_sdfg_shim_for_outer,
    _build_sync_helpers,
)

from _util import build_sdfg
from dace_fortran.bindings import build_fortran_library, FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external


_RELEASE_FFLAGS = (
    "-O3", "-fno-fast-math", "-ffp-contract=off", "-ffree-line-length-none",
)
_RELEASE_CXX_FLAGS = (
    "-O3", "-fno-fast-math", "-ffp-contract=off",
    "-fPIC", "-Wno-unused-parameter", "-Wno-unused-label",
)


# Free-standing wrapper subroutine ICON's patched ``mo_velocity_advection``
# calls into.  Exports the link-time symbol ``velocity_tendencies_dace_icon_``
# (trailing underscore, NO module prefix) so the call site can declare it
# via an ``INTERFACE`` block instead of ``USE``-ing the bindings module
# (which would re-import stub copies of ICON's ``t_patch`` / ``t_nh_prog``
# and conflict with ``mo_model_domain`` / ``mo_nonhydro_types``).
#
# The ICON module variables the ``bind_c_shim``'s
# ``module_symbol_forward`` references (``i_am_accel_node``,
# ``nproma``, ``nrdmax``, ``timer_*``, ...) are NOT stubbed here:
# ``velocity_full.f90`` (the canonical ``prelude_sources`` input)
# already defines each as a ``PUBLIC`` module variable, so the
# emitted ``__mo_mpi_MOD_i_am_accel_node`` symbol etc. land in
# ``libvelocity_inner_wrap.so`` as a side effect of compiling the
# prelude.  ICON's CPU build keeps its own copies in its own
# objects; under the default ``--as-needed`` linker the .so's copies
# stay live without colliding.
_ICON_WRAPPER_F90 = """\
! ICON-side wrapper: forwards to the SDFG-generated
! ``velocity_tendencies_dace`` (module-scoped name
! ``__velocity_tendencies_dace_bindings_MOD_velocity_tendencies_dace``)
! under the FREE-STANDING name ``velocity_tendencies_dace_icon`` so
! ICON's call site can declare it via an INTERFACE block (no USE of
! the bindings module, which would re-import stub copies of ICON's
! derived types and conflict with ``mo_model_domain`` /
! ``mo_nonhydro_types``).
SUBROUTINE velocity_tendencies_dace_icon(p_prog, p_patch, p_int, p_metrics, p_diag, &
                                         z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                                         ntnd, istep, lvn_only, &
                                         dtime, dt_linintp_ubc, ldeepatmo)
  USE iso_c_binding, ONLY: c_int, c_double, c_bool
  USE velocity_tendencies_dace_bindings, ONLY: velocity_tendencies_dace
  USE mo_model_domain,      ONLY: t_patch
  USE mo_intp_data_strc,    ONLY: t_int_state
  USE mo_nonhydro_types,    ONLY: t_nh_prog, t_nh_metrics, t_nh_diag
  TYPE(t_nh_prog),    INTENT(INOUT), TARGET :: p_prog
  TYPE(t_patch),      INTENT(IN),    TARGET :: p_patch
  TYPE(t_int_state),  INTENT(IN),    TARGET :: p_int
  TYPE(t_nh_metrics), INTENT(INOUT), TARGET :: p_metrics
  TYPE(t_nh_diag),    INTENT(INOUT), TARGET :: p_diag
  REAL(c_double),     INTENT(INOUT), TARGET :: z_w_concorr_me(:,:,:)
  REAL(c_double),     INTENT(INOUT), TARGET :: z_kin_hor_e(:,:,:)
  REAL(c_double),     INTENT(INOUT), TARGET :: z_vt_ie(:,:,:)
  INTEGER(c_int),     INTENT(IN),    TARGET :: ntnd
  INTEGER(c_int),     INTENT(IN),    TARGET :: istep
  LOGICAL(c_bool),    INTENT(IN),    TARGET :: lvn_only
  REAL(c_double),     INTENT(IN),    TARGET :: dtime
  REAL(c_double),     INTENT(IN),    TARGET :: dt_linintp_ubc
  LOGICAL(c_bool),    INTENT(IN),    TARGET :: ldeepatmo
  CALL velocity_tendencies_dace(p_prog, p_patch, p_int, p_metrics, p_diag, &
                                z_w_concorr_me, z_kin_hor_e, z_vt_ie, &
                                ntnd, istep, lvn_only, &
                                dtime, dt_linintp_ubc, ldeepatmo)
END SUBROUTINE velocity_tendencies_dace_icon
"""


def build_velocity_inner_wrap(velocity_source: Path, out_dir: Path,
                              release: bool):
    """Build ``libvelocity_inner_wrap.so`` from
    ``mo_velocity_advection.f90``.  The output dir gets the .so + the
    .mod (``velocity_tendencies_dace_bindings.mod``) ICON needs at
    Fortran-compile time + the .f90 sources kept around for
    inspection."""
    velocity_source = velocity_source.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fflags = _RELEASE_FFLAGS if release else _O0_FFLAGS
    cxx_flags = _RELEASE_CXX_FLAGS if release else _O0_CXX_FLAGS

    print(f"[build_icon_dace_libs] velocity source: {velocity_source}",
          flush=True)
    print(f"[build_icon_dace_libs] output dir:      {out_dir}", flush=True)
    print(f"[build_icon_dace_libs] FP flags ({'release' if release else 'debug'}): "
          f"{' '.join(fflags)}", flush=True)

    sdfg_dir = out_dir / "_sdfg_build"
    sdfg_dir.mkdir(parents=True, exist_ok=True)

    velocity_src = velocity_source.read_text()

    # Drop the ICON-side wrapper next to the binding source.
    # ``extra_sources`` is compiled AFTER the binding module so the
    # wrapper's ``USE velocity_tendencies_dace_bindings`` and
    # ``USE mo_model_domain`` (resolved against the stub .mod files
    # the binding emits as a side effect of ``prelude_sources``)
    # both succeed.
    icon_wrapper_f90 = out_dir / "icon_wrapper.f90"
    icon_wrapper_f90.write_text(_ICON_WRAPPER_F90)

    # Pin DaCe's C++ codegen flags before any SDFG build.
    orig_cxx_args = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args", value=" ".join(cxx_flags))

    try:
        sdfg = build_sdfg(
            velocity_src, sdfg_dir,
            name="velocity_tendencies",
            entry="_QMmo_velocity_advectionPvelocity_tendencies",
        ).build()
        sdfg.name = "velocity_tendencies"
        sdfg.build_folder = str(sdfg_dir / "dacecache")
        iface = build_auto_interface(
            sdfg._fortran_interface_raw, "velocity_tendencies")
        plan = FlattenPlan.from_dict(sdfg._flatten_plan_raw or {})
        lib = build_fortran_library(
            sdfg,
            iface=iface,
            plan=plan,
            out_dir=str(out_dir),
            name="velocity_inner_wrap",
            prelude_sources=[velocity_source],
            extra_sources=[icon_wrapper_f90],
            bind_c_shim=True,
            bind_c_shim_module_symbol_forward=_VELOCITY_MODULE_FORWARD,
            flags=fflags,
        )
    finally:
        dace.Config.set("compiler", "cpu", "args", value=orig_cxx_args)

    print(f"[build_icon_dace_libs] artifact: {lib.so_path}", flush=True)
    # Surface the .mod path explicitly so the user can ``-I`` it.
    mod_path = out_dir / "velocity_tendencies_dace_bindings.mod"
    if mod_path.exists():
        print(f"[build_icon_dace_libs] .mod:     {mod_path}", flush=True)
        print(f"\nICON build flags:\n"
              f"  export FCFLAGS=\"-I{out_dir} ${{FCFLAGS-}}\"\n"
              f"  export LDFLAGS=\"-L{out_dir} -Wl,-rpath,{out_dir} "
              f"-l:{lib.so_path.name} ${{LDFLAGS-}}\"\n",
              flush=True)
    return lib


def build_dycore_wrapper(velocity_source: Path, inner_lib_so: Path,
                         out_dir: Path, release: bool):
    """Build ``libdycore_wrapper.so`` -- the outer SDFG that calls
    ``velocity_tendencies`` (resolved at runtime from
    ``libvelocity_inner_wrap.so``) and a Fortran/C++ pair of
    ``sync_patch_*`` helpers.  Mirrors the
    ``test_dycore_outer_calls_velocity_sdfg_via_c_abi`` flow.

    :param velocity_source: pre-merged ``velocity_full.f90`` driving
        both inner and outer kernels.
    :param inner_lib_so: path to the already-built
        ``libvelocity_inner_wrap.so`` (its ``velocity_tendencies_c``
        symbol is what the outer SDFG dispatches to).
    :param out_dir: scratch directory for the dycore lib + its sync
        helpers.
    :param release: ``True`` for ``-O3``, ``False`` for ``-O0``.
    """
    velocity_source = velocity_source.resolve()
    inner_lib_so = inner_lib_so.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fflags = _RELEASE_FFLAGS if release else _O0_FFLAGS
    cxx_flags = _RELEASE_CXX_FLAGS if release else _O0_CXX_FLAGS

    print(f"[build_icon_dace_libs] dycore wrapper out: {out_dir}", flush=True)

    # 1. sync_patch_array (Fortran) + sync_patch_cpp (C++) helpers ->
    #    libsync_helpers.so the outer SDFG resolves at load time.
    sync_lib_so = _build_sync_helpers(out_dir)
    print(f"[build_icon_dace_libs] sync helpers:       {sync_lib_so}",
          flush=True)

    velocity_src = velocity_source.read_text()
    sync_args = (
        Arg(kind="scalar", dtype="int32", intent="in"),
        Arg(kind="array", dtype="float64", intent="inout"),
    )
    clear_external_registry()
    keep_external(
        "velocity_tendencies",
        c_name="velocity_tendencies_c",
        args=(
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),
            Arg(kind="aos", intent="in", c_abi="per_member_soa"),
            Arg(kind="aos", intent="in", c_abi="per_member_soa"),
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),
            Arg(kind="aos", intent="inout", c_abi="per_member_soa"),
            Arg(kind="array", dtype="float64", intent="inout"),
            Arg(kind="array", dtype="float64", intent="inout"),
            Arg(kind="array", dtype="float64", intent="inout"),
            Arg(kind="scalar", dtype="int32", intent="in"),
            Arg(kind="scalar", dtype="int32", intent="in"),
            Arg(kind="scalar", dtype="bool", intent="in"),
            Arg(kind="scalar", dtype="float64", intent="in"),
            Arg(kind="scalar", dtype="float64", intent="in"),
            Arg(kind="scalar", dtype="bool", intent="in"),
        ),
        libraries=(str(inner_lib_so), ),
        dynamic_extents_abi=True,
        module_symbol_forward=_VELOCITY_MODULE_FORWARD,
    )
    keep_external(
        "sync_patch_array",
        c_name="sync_patch_array_c",
        args=sync_args,
        libraries=(str(sync_lib_so), ),
        dynamic_extents_abi=True,
    )
    keep_external(
        "sync_patch_cpp_via",
        c_name="sync_patch_cpp_via_c",
        args=sync_args,
        libraries=(str(sync_lib_so), ),
        dynamic_extents_abi=True,
    )

    orig_cxx_args = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args", value=" ".join(cxx_flags))
    try:
        outer_sdfg_dir = out_dir / "sdfg"
        outer_sdfg_dir.mkdir(parents=True, exist_ok=True)
        outer_src = velocity_src + _SYNC_FORTRAN_SRC + _DYCORE_WRAPPER_SRC
        outer_sdfg = build_sdfg(
            outer_src, outer_sdfg_dir,
            name="dycore_wrapper",
            entry="_QMmo_dycore_wrapperPdycore_wrapper").build()
        outer_sdfg.name = "dycore_wrapper"
        outer_sdfg.build_folder = str(out_dir / "dacecache")
        outer_iface = _velocity_iface("dycore_wrapper")
        outer_plan = FlattenPlan.from_dict(outer_sdfg._flatten_plan_raw or {})

        sdfg_shim = out_dir / "sdfg_shim.f90"
        sdfg_shim.write_text(_make_sdfg_shim_for_outer(_CALLER_PATH.read_text()))

        outer_lib = build_fortran_library(
            outer_sdfg,
            iface=outer_iface,
            plan=outer_plan,
            out_dir=str(out_dir),
            name="dycore_wrapper",
            prelude_sources=[velocity_source, _CALLER_PATH],
            extra_sources=[sdfg_shim],
            bind_c_shim=False,
            flags=fflags,
        )
    finally:
        clear_external_registry()
        dace.Config.set("compiler", "cpu", "args", value=orig_cxx_args)

    print(f"[build_icon_dace_libs] dycore artifact:    {outer_lib.so_path}",
          flush=True)
    return outer_lib


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--velocity-source", type=Path,
        default=(Path(__file__).resolve().parents[1]
                 / "tests" / "icon_full" / "velocity_full.f90"),
        help="Pre-merged self-contained ICON ``mo_velocity_advection`` "
             "source.  Defaults to the e2e test's ``velocity_full.f90``.")
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Where libvelocity_inner_wrap.so + .mod files go.")
    ap.add_argument(
        "--release", action="store_true",
        help="Use -O3 -fno-fast-math -ffp-contract=off instead of -O0.  "
             "Numerical envelope drops to 1 ULP but performance matches a "
             "stock ICON production build.")
    ap.add_argument(
        "--with-dycore", action="store_true",
        help="Also build ``libdycore_wrapper.so`` -- the outer SDFG "
             "that dispatches to the inner velocity .so over the C "
             "ABI (``velocity_tendencies_c``) AND calls a Fortran/C++ "
             "sync helper pair.  ICON itself does NOT link this "
             "artifact (its real ``mo_solve_nonhydro`` signature does "
             "not match the wrapper), but building it validates the "
             "SDFG-to-SDFG C-ABI chain.")
    args = ap.parse_args()

    if not args.velocity_source.exists():
        print(f"error: velocity source not found: {args.velocity_source}",
              file=sys.stderr)
        return 1
    if not shutil.which("gfortran") or not shutil.which("flang-new-21"):
        print("error: need gfortran + flang-new-21 on PATH", file=sys.stderr)
        return 1

    inner_lib = build_velocity_inner_wrap(
        args.velocity_source, args.out_dir, release=args.release)
    if args.with_dycore:
        build_dycore_wrapper(
            args.velocity_source, inner_lib.so_path,
            args.out_dir / "_dycore_build", release=args.release)
    return 0


if __name__ == "__main__":
    sys.exit(main())
