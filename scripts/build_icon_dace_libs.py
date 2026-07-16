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
``tests/icon/full/velocity_full.f90`` -- the pre-merged
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
from icon.full.test_dycore_velocity_external_e2e import (
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

import dace_fortran

from _util import build_sdfg
from dace_fortran.bindings import build_fortran_library, FlattenPlan
from dace_fortran.bindings.bind_c_shim import scalar_pointer_members
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.external import Arg, clear_external_registry, keep_external

_RELEASE_FFLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-ffree-line-length-none",
)
_RELEASE_CXX_FLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fPIC",
    "-Wno-unused-parameter",
    "-Wno-unused-label",
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

#: The ICON object whose ``make -n`` line supplies the real ``-D`` / ``-I`` set.
_VELOCITY_TARGET = "src/atm_dyn_iconam/mo_velocity_advection.o"
#: ``module::procedure`` entry form ``build_sdfg_from_hlfir`` takes (the legacy
#: pre-merged route uses the flang-mangled ``_QM...P...`` name instead).
_VELOCITY_ENTRY_QUALIFIED = "mo_velocity_advection::velocity_tendencies"

#: ICON utility procedures ``velocity_tendencies`` calls structurally (error
#: reporting, timer hooks) but whose bodies the bridge does not need to lower.
#: Dropped BEFORE ``hlfir-inline-all`` so their unlowerable internals
#: (``fir.iterate_while`` LEN_TRIM scans, ``CLASS(*)`` polymorphism) never reach
#: the bridge.  Mirrors ``tests/icon/full/test_velocity_from_icon_source.py``.
_ICON_EXTERNAL_STUBS = (
    "finish",
    "message",
    "message_text",
    "warning",
    "print_status",
    "print_value",
    "init_logger",
    "timer_start",
    "timer_stop",
    "new_timer",
    "delete_timer",
)


def _icon_search_dirs(icon_src: Path) -> list:
    """The USE-graph closure of ``mo_velocity_advection``: ICON's ``src`` plus
    the external library trees ICON bundles.  Same set the real-source build
    test bisected down to."""
    return [
        icon_src / "src",
        icon_src / "externals/fortran-support/src",
        icon_src / "externals/mtime/src",
        icon_src / "externals/iconmath/src",
        icon_src / "externals/cdi/src",
        icon_src / "externals/memman/src/bindings/fortran",
        icon_src / "support",
    ]


def _icon_mod_dirs(icon_build: Path) -> list:
    """Every directory under the ICON build holding a compiled ``.mod``.  ICON
    scatters them across its own ``mod/`` plus each bundled external's build tree
    (fortran-support, mtime, iconmath, cdi, memman), so a single ``-I .../mod``
    misses e.g. ``mo_fortran_tools.mod``.  Absolute -- the gfortran link runs with
    ``cwd`` set to the output dir, so a relative ``-I`` would not resolve."""
    return sorted({p.parent.resolve() for p in icon_build.rglob("*.mod")})


def build_velocity_sdfg_from_icon_source(icon_src: Path, icon_build: Path, sdfg_dir: Path):
    """Lower ICON's REAL ``mo_velocity_advection.f90`` to an SDFG.

    The pre-merged ``velocity_full.f90`` route below builds a STUB-TYPED kernel:
    its ``t_patch`` / ``t_nh_prog`` are self-contained stand-ins, NOT ICON's real
    layout, so a lib built from it SEGVs inside the first ``velocity_tendencies``
    call in a real ICON run.  This route instead resolves the real source's USE
    closure across the ICON tree and lowers THAT, so the marshalling matches the
    struct layout ICON actually passes.

    ``icon_build`` must be the build of the SAME ICON CONFIGURATION the lib will
    be linked into: its ``-D`` set decides conditional type layouts
    (``__NO_ICON_OCEAN__``, ``__NO_JSBACH__``, ...), so defines from a
    differently-configured build would reintroduce the very layout mismatch this
    route exists to remove."""
    args = dace_fortran.extract_make_compile_args(makefile_dir=icon_build, target=_VELOCITY_TARGET)
    print(
        f"[build_icon_dace_libs] ICON defines from {icon_build}: {len(args['defines'])} -D, "
        f"{len(args['include_dirs'])} -I",
        flush=True)
    velocity_real = icon_src / "src" / "atm_dyn_iconam" / "mo_velocity_advection.f90"
    # Prefer a pristine ``.bak`` when an ICON workflow has patched the live file
    # (run_icon_e2e.sh's DaCe dispatch patch keeps the original there).
    velocity_bak = velocity_real.with_suffix(".f90.bak")
    entry_src = velocity_bak if velocity_bak.is_file() else velocity_real
    print(f"[build_icon_dace_libs] real ICON velocity source: {entry_src}", flush=True)

    hlfir = dace_fortran.emit_hlfir_from_codebase(
        entry_source=entry_src.read_text(),
        out_path=sdfg_dir / "velocity.hlfir",
        search_dirs=_icon_search_dirs(icon_src),
        library_stubs=["mpi", "netcdf"],
        defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
        include_dirs=args["include_dirs"],
        cache_dir=Path(os.environ.get("DACE_FORTRAN_CACHE", str(Path.home() / ".cache" / "dace-fortran"))),
    )
    dace_fortran.clear_external_registry()
    dace_fortran.apply_external_functions(do_not_emit=list(_ICON_EXTERNAL_STUBS))
    try:
        return dace_fortran.build_sdfg_from_hlfir(hlfir, entry=_VELOCITY_ENTRY_QUALIFIED)
    finally:
        dace_fortran.clear_external_registry()


def build_velocity_inner_wrap(velocity_source: Path,
                              out_dir: Path,
                              release: bool,
                              icon_src: Path = None,
                              icon_build: Path = None):
    """Build ``libvelocity_inner_wrap.so`` from
    ``mo_velocity_advection.f90``.  The output dir gets the .so + the
    .mod (``velocity_tendencies_dace_bindings.mod``) ICON needs at
    Fortran-compile time + the .f90 sources kept around for
    inspection.

    With ``icon_src`` + ``icon_build`` the SDFG is lowered from ICON's REAL
    velocity source (see :func:`build_velocity_sdfg_from_icon_source`) and the
    bind_c shim's ``USE``s resolve against that build's real ``.mod`` via ``-I``
    instead of against a self-contained prelude -- ICON's modules ARE the
    prelude.  Without them, the legacy pre-merged (stub-typed) route is used."""
    velocity_source = velocity_source.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fflags = _RELEASE_FFLAGS if release else _O0_FFLAGS
    cxx_flags = _RELEASE_CXX_FLAGS if release else _O0_CXX_FLAGS

    print(f"[build_icon_dace_libs] velocity source: {velocity_source}", flush=True)
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

    real_source = icon_src is not None and icon_build is not None
    try:
        if real_source:
            sdfg = build_velocity_sdfg_from_icon_source(icon_src.resolve(), icon_build.resolve(), sdfg_dir)
        else:
            sdfg = build_sdfg(
                velocity_src,
                sdfg_dir,
                name="velocity_tendencies",
                entry="_QMmo_velocity_advectionPvelocity_tendencies",
            ).build()
        sdfg.name = "velocity_tendencies"
        sdfg.build_folder = str(sdfg_dir / "dacecache")
        iface = build_auto_interface(sdfg._fortran_interface_raw, "velocity_tendencies")
        # Grid-dim scalar members the shim takes as ``type(c_ptr), value`` -- the
        # OUTER dycore SDFG's velocity ``keep_external`` must pass their ADDRESS
        # (``callee_ptr_scalar_members``), else its by-value marshal segfaults the
        # inner ``c_f_pointer``.  Derived from the real callee iface here (the
        # hand-authored ``_velocity_iface`` carries no ``struct_types``).
        callee_ptr_members = scalar_pointer_members(iface)
        plan = FlattenPlan.from_dict(sdfg._flatten_plan_raw or {})
        # Real-source route: ICON's own compiled modules ARE the prelude, so the
        # Fortran-ABI bindings' ``USE mo_model_domain`` / ``mo_nonhydro_types``
        # resolve via -I against ICON's real .mod (scattered across its own
        # ``mod/`` plus each bundled external's build tree), not a stub prelude.
        # NO C-ABI shim on this route: ICON calls the Fortran-ABI
        # ``velocity_tendencies_dace`` (whole structs, via icon_wrapper), never
        # the flat ``velocity_tendencies_c``.  The bindings truncate-and-hash deep
        # member names to Fortran's 63-char limit; the C shim does not, so the
        # real t_patch's ``..._decomp_info_glb2loc_index_..._lb0`` (66 chars) would
        # only break a shim ICON does not link.  Dropping it also moots the
        # module-symbol-forward table (a C-shim-only detail).
        icon_incs = tuple(f"-I{d}" for d in _icon_mod_dirs(icon_build)) if real_source else ()
        lib = build_fortran_library(
            sdfg,
            iface=iface,
            plan=plan,
            out_dir=str(out_dir),
            name="velocity_inner_wrap",
            prelude_sources=[] if real_source else [velocity_source],
            extra_sources=[icon_wrapper_f90],
            bind_c_shim=not real_source,
            bind_c_shim_module_symbol_forward=_VELOCITY_MODULE_FORWARD,
            flags=(*fflags, *icon_incs),
        )
    finally:
        dace.Config.set("compiler", "cpu", "args", value=orig_cxx_args)

    print(f"[build_icon_dace_libs] artifact: {lib.so_path}", flush=True)
    # Surface the .mod path explicitly so the user can ``-I`` it.
    mod_path = out_dir / "velocity_tendencies_dace_bindings.mod"
    if mod_path.exists():
        print(f"[build_icon_dace_libs] .mod:     {mod_path}", flush=True)
        print(
            f"\nICON build flags:\n"
            f"  export FCFLAGS=\"-I{out_dir} ${{FCFLAGS-}}\"\n"
            f"  export LDFLAGS=\"-L{out_dir} -Wl,-rpath,{out_dir} "
            f"-l:{lib.so_path.name} ${{LDFLAGS-}}\"\n",
            flush=True)
    return lib, callee_ptr_members


def build_dycore_wrapper(velocity_source: Path,
                         inner_lib_so: Path,
                         out_dir: Path,
                         release: bool,
                         callee_ptr_scalar_members: frozenset = frozenset()):
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
    print(f"[build_icon_dace_libs] sync helpers:       {sync_lib_so}", flush=True)

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
        callee_ptr_scalar_members=callee_ptr_scalar_members,
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
        outer_sdfg = build_sdfg(outer_src,
                                outer_sdfg_dir,
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

    print(f"[build_icon_dace_libs] dycore artifact:    {outer_lib.so_path}", flush=True)
    return outer_lib


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--velocity-source",
                    type=Path,
                    default=(Path(__file__).resolve().parents[1] / "tests" / "icon" / "full" / "velocity_full.f90"),
                    help="Pre-merged self-contained ICON ``mo_velocity_advection`` "
                    "source.  Defaults to the e2e test's ``velocity_full.f90``.")
    ap.add_argument("--icon-src",
                    type=Path,
                    default=None,
                    help="ICON source tree (e.g. tests/icon/full/icon-model).  With --icon-build, "
                    "builds the SDFG from ICON's REAL mo_velocity_advection.f90 instead of the "
                    "pre-merged stub-typed --velocity-source.  Required for an in-ICON run: a lib "
                    "built from the stub types SEGVs in the first velocity_tendencies call.")
    ap.add_argument("--icon-build",
                    type=Path,
                    default=None,
                    help="Build dir of the ICON CONFIGURATION this lib will be linked into.  Supplies "
                    "the real -D defines (which decide conditional type layouts) and the real .mod "
                    "the bind_c shim compiles against.  MUST match the target ICON's configure "
                    "options -- defines from a differently-configured build reintroduce the layout "
                    "mismatch this route removes.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where libvelocity_inner_wrap.so + .mod files go.")
    ap.add_argument("--release",
                    action="store_true",
                    help="Use -O3 -fno-fast-math -ffp-contract=off instead of -O0.  "
                    "Numerical envelope drops to 1 ULP but performance matches a "
                    "stock ICON production build.")
    ap.add_argument("--with-dycore",
                    action="store_true",
                    help="Also build ``libdycore_wrapper.so`` -- the outer SDFG "
                    "that dispatches to the inner velocity .so over the C "
                    "ABI (``velocity_tendencies_c``) AND calls a Fortran/C++ "
                    "sync helper pair.  ICON itself does NOT link this "
                    "artifact (its real ``mo_solve_nonhydro`` signature does "
                    "not match the wrapper), but building it validates the "
                    "SDFG-to-SDFG C-ABI chain.")
    args = ap.parse_args()

    if not args.velocity_source.exists():
        print(f"error: velocity source not found: {args.velocity_source}", file=sys.stderr)
        return 1
    if not shutil.which("gfortran") or not shutil.which("flang-new-21"):
        print("error: need gfortran + flang-new-21 on PATH", file=sys.stderr)
        return 1

    if (args.icon_src is None) != (args.icon_build is None):
        print("error: --icon-src and --icon-build must be given together", file=sys.stderr)
        return 1
    if args.icon_src is not None and not (args.icon_build / "Makefile").is_file():
        print(f"error: no ICON build at {args.icon_build} (expected a configured Makefile)", file=sys.stderr)
        return 1

    inner_lib, callee_ptr_members = build_velocity_inner_wrap(args.velocity_source,
                                                              args.out_dir,
                                                              release=args.release,
                                                              icon_src=args.icon_src,
                                                              icon_build=args.icon_build)
    if args.with_dycore:
        build_dycore_wrapper(args.velocity_source,
                             inner_lib.so_path,
                             args.out_dir / "_dycore_build",
                             release=args.release,
                             callee_ptr_scalar_members=callee_ptr_members)
    return 0


if __name__ == "__main__":
    sys.exit(main())
