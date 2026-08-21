#!/usr/bin/env python3
"""Standalone driver: build the DaCe-backed ``libsolve_nh.so``
(and ICON-side wrapper) that a patched ICON atmosphere model links against.

Lifts the build steps out of ``tests/icon/full/test_dycore_from_icon_source.py``
and ``test_icon_solve_nh_swap.py`` so they can be run outside ``pytest``.

Two SDFG sources are supported:

* ``--single-tu`` (default): the pre-merged single-file checkpoint
  ``tests/icon/atmosphere/solve_nonhydro_inlined_single_tu.f90``.  It carries
  stub modules whose type layouts match ICON's real types, so the generated
  binding can be linked into the real model.  This path is fast and stable
  enough for CI.

* ``--real-source``: lower ICON's real ``mo_solve_nonhydro.f90`` so the
  binding's flattened struct layout is derived from the actual
  ``t_nh_state`` / ``t_patch`` / ``t_int_state`` / ``t_prepare_adv`` types.
  ``velocity_tendencies`` is kept as an external and resolves at ICON link time
  to the stock Fortran implementation.  This path is more faithful but slower
  and currently fragile in the local environment.

Usage::

    python -m scripts.build_icon_solve_nh_libs \
        --icon-src tests/icon/full/icon-model \
        --icon-build tests/icon/full/icon-model/build/stock_cpu \
        --out-dir $WORK/solve-nh-dace-libs

See ``docs/ICON_INTEGRATION.md`` for the surrounding workflow.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import dace

_DACE_FORTRAN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DACE_FORTRAN / "tests"))
from _util import build_sdfg, have_flang
from icon._halo_modes import _MPI_STUB
from icon.full._icon_solve_nh_patch import SOLVE_NH_WRAPPER_NAME
from icon.full.test_dycore_from_icon_source import (
    _ICON_DEFINES_FALLBACK,
    _ICON_EXTERNAL_STUBS,
)

import dace_fortran
from dace_fortran.bindings import build_fortran_library, FlattenPlan
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.flang_codebase import _DEFINE_RE, _INCLUDE_RE

_RELEASE_FFLAGS = (
    "-O3",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-ffree-line-length-none",
)

_DEBUG_FFLAGS = (
    "-O0",
    "-g",
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

_DEBUG_CXX_FLAGS = (
    "-O0",
    "-g",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fPIC",
    "-Wno-unused-parameter",
    "-Wno-unused-label",
)

#: Fortran object target used to extract ICON's real compile flags.
_SOLVE_NH_TARGET = "src/atm_dyn_iconam/mo_solve_nonhydro.o"

#: ``module::procedure`` entry passed to ``build_sdfg_from_hlfir``.
_SOLVE_NH_ENTRY = "mo_solve_nonhydro::solve_nh"


def _real_source(icon_src: Path) -> Path:
    """Pristine ``mo_solve_nonhydro.f90``; prefer ``.bak`` when the live file
    has been patched."""
    src = icon_src / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
    bak = src.with_suffix(".f90.bak")
    return bak if bak.is_file() else src


def _icon_search_dirs(icon_src: Path) -> list:
    """Roots for the ``merge_used_modules`` search closure."""
    return [
        icon_src / "src",
        icon_src / "externals/fortran-support/src",
        icon_src / "externals/mtime/src",
        icon_src / "externals/iconmath/src",
        icon_src / "externals/cdi/src",
        icon_src / "externals/memman/src/bindings/fortran",
        icon_src / "support",
    ]


def _icon_compile_args(icon_src: Path, icon_build: Path) -> dict:
    """ICON's real ``-D`` / ``-I`` set when a configured build exists.

    Falls back to the canonical fallback defines + ``src/include`` when the
    build is missing or ``make -n`` does not produce a recognizable command.
    """
    fallback_incs = [icon_src / "src/include", icon_src / "externals" / "iconmath" / "include"]
    fallback_incs = [p for p in fallback_incs if p.is_dir()]
    if not (icon_build / "Makefile").is_file():
        return {
            "defines": list(_ICON_DEFINES_FALLBACK),
            "include_dirs": fallback_incs,
        }

    try:
        extracted = dace_fortran.extract_make_compile_args(makefile_dir=icon_build, target=_SOLVE_NH_TARGET)
    except (RuntimeError, subprocess.CalledProcessError):
        return {
            "defines": list(_ICON_DEFINES_FALLBACK),
            "include_dirs": fallback_incs,
        }

    # ``make -n`` for an ICON target emits prerequisite builds first; the helper
    # can grab a dependency's compile line instead of ``mo_solve_nonhydro.f90``.
    # Re-scan the full dry-run output for the actual source file and merge.
    real_src = _real_source(icon_src).name
    if extracted.get("source") is None or extracted["source"].name != real_src:
        artefact = icon_build / _SOLVE_NH_TARGET
        if artefact.is_file():
            artefact.unlink()
        out = subprocess.check_output(["make", "-n", _SOLVE_NH_TARGET],
                                      cwd=str(icon_build),
                                      stderr=subprocess.STDOUT,
                                      text=True)
        for ln in out.splitlines():
            if real_src in ln and (" -c " in ln or ln.lstrip().startswith("mpifort") or "gfortran" in ln):
                extracted = {
                    "defines": sorted(set(_DEFINE_RE.findall(ln))),
                    "include_dirs": [Path(p) for p in _INCLUDE_RE.findall(ln)],
                    "source": _real_source(icon_src),
                    "command": ln.strip(),
                }
                break

    # Always include the fallback defines and the key include directories; ICON's
    # Makefile may express some of these through wrapper defaults or environment
    # variables that ``extract_make_compile_args`` does not capture.
    defines = sorted(set(list(_ICON_DEFINES_FALLBACK) + extracted.get("defines", [])))
    include_dirs = list(dict.fromkeys([*fallback_incs, *extracted.get("include_dirs", [])]))
    return {
        "defines": defines,
        "include_dirs": include_dirs,
        "source": extracted.get("source", _real_source(icon_src)),
        "command": extracted.get("command", ""),
    }


def _icon_mod_dirs(icon_build: Path) -> list:
    """Every directory under the ICON build holding compiled ``.mod`` files."""
    return sorted({p.parent.resolve() for p in icon_build.rglob("*.mod")})


def _render_icon_wrapper() -> str:
    """Render a Fortran wrapper ``solve_nh_dace_icon`` that forwards to the
    SDFG-generated ``solve_nh_dace`` inside ``solve_nh_dace_bindings``.

    The wrapper signature matches ``tests/icon/full/_icon_solve_nh_patch.py``
    exactly so the patched ``mo_solve_nonhydro.f90`` INTERFACE block resolves.
    """
    name = SOLVE_NH_WRAPPER_NAME
    lines = [
        "! ICON-side wrapper: forwards to the SDFG-generated",
        "! ``solve_nh_dace`` under the free-standing name ``solve_nh_dace_icon``.",
        "! Generated by scripts/build_icon_solve_nh_libs.py -- do not edit.",
        f"SUBROUTINE {name}(p_nh, p_patch, p_int, prep_adv, &",
        "                                  nnow, nnew, &",
        "                                  l_init, l_recompute, lsave_mflx, &",
        "                                  lprep_adv, lclean_mflx, &",
        "                                  idyn_timestep, jstep, dtime, lacc)",
        "  USE iso_c_binding,     ONLY: c_int, c_double, c_bool",
        "  USE solve_nh_dace_bindings, ONLY: solve_nh_dace",
        "  USE mo_model_domain,   ONLY: t_patch",
        "  USE mo_intp_data_strc, ONLY: t_int_state",
        "  USE mo_nonhydro_types, ONLY: t_nh_state",
        "  USE mo_prepadv_types,  ONLY: t_prepare_adv",
        "  TYPE(t_nh_state),    TARGET, INTENT(INOUT) :: p_nh",
        "  TYPE(t_int_state),   TARGET, INTENT(IN)    :: p_int",
        "  TYPE(t_patch),       TARGET, INTENT(INOUT) :: p_patch",
        "  TYPE(t_prepare_adv), TARGET, INTENT(INOUT) :: prep_adv",
        "  INTEGER(c_int),              INTENT(IN)    :: nnow, nnew",
        "  LOGICAL(c_bool),             INTENT(IN)    :: l_init",
        "  LOGICAL(c_bool),             INTENT(IN)    :: l_recompute",
        "  LOGICAL(c_bool),             INTENT(IN)    :: lsave_mflx",
        "  LOGICAL(c_bool),             INTENT(IN)    :: lprep_adv",
        "  LOGICAL(c_bool),             INTENT(IN)    :: lclean_mflx",
        "  INTEGER(c_int),              INTENT(IN)    :: idyn_timestep",
        "  INTEGER(c_int),              INTENT(IN)    :: jstep",
        "  REAL(c_double),              INTENT(IN)    :: dtime",
        "  LOGICAL(c_bool),             INTENT(IN)    :: lacc",
        "  CALL solve_nh_dace(p_nh, p_patch, p_int, prep_adv, &",
        "                   nnow, nnew, &",
        "                   l_init, l_recompute, lsave_mflx, &",
        "                   lprep_adv, lclean_mflx, &",
        "                   idyn_timestep, jstep, dtime, lacc)",
        f"END SUBROUTINE {name}",
        "",
    ]
    return "\n".join(lines)


def build_solve_nh_sdfg(icon_src: Path, icon_build: Path, sdfg_dir: Path):
    """Lower ICON's REAL ``mo_solve_nonhydro.f90`` to an SDFG."""
    args = _icon_compile_args(icon_src, icon_build)
    print(
        f"[build_icon_solve_nh_libs] ICON defines from {icon_build}: {len(args['defines'])} -D, "
        f"{len(args['include_dirs'])} -I",
        flush=True,
    )
    entry_src = _real_source(icon_src)
    print(f"[build_icon_solve_nh_libs] real ICON solve_nh source: {entry_src}", flush=True)

    hlfir = dace_fortran.emit_hlfir_from_codebase(
        entry_source=entry_src.read_text(),
        out_path=sdfg_dir / "solve_nh.hlfir",
        search_dirs=_icon_search_dirs(icon_src),
        library_stubs=["mpi", "netcdf"],
        defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
        include_dirs=args["include_dirs"],
        cache_dir=Path(os.environ.get("DACE_FORTRAN_CACHE", str(Path.home() / ".cache" / "dace-fortran"))),
    )
    dace_fortran.clear_external_registry()
    dace_fortran.apply_external_functions(do_not_emit=_ICON_EXTERNAL_STUBS)
    try:
        return dace_fortran.build_sdfg_from_hlfir(hlfir, entry=_SOLVE_NH_ENTRY)
    finally:
        dace_fortran.clear_external_registry()


_SINGLE_TU = _DACE_FORTRAN / "tests" / "icon" / "atmosphere" / "solve_nonhydro_inlined_single_tu.f90"


def _stage_single_tu_sources(sdfg_dir: Path) -> tuple[Path, Path]:
    """Stage the single-TU source plus the MPI stub module it needs."""
    print(f"[build_icon_solve_nh_libs] single-TU source: {_SINGLE_TU}", flush=True)
    tu_src = _SINGLE_TU.read_text()
    if "MODULE mo_mpi\n" not in tu_src:
        raise ValueError("single-TU anchor MODULE mo_mpi not found; cannot inject USE mpi")
    # The MPI stub module provides a single TYPE(*) interface for point-to-point
    # calls, avoiding gfortran's dual-typed implicit-interface error.
    use_mpi_tu = sdfg_dir / "solve_nonhydro_single_tu_usempi.f90"
    use_mpi_tu.write_text(tu_src.replace("MODULE mo_mpi\n", "MODULE mo_mpi\n  use mpi\n", 1))
    mpi_stub = sdfg_dir / "_mpi_stub.f90"
    mpi_stub.write_text(_MPI_STUB)
    return use_mpi_tu, mpi_stub


def _compile_single_tu_objects(sdfg_dir: Path, fflags: Sequence[str]) -> tuple[Path, Path, Path]:
    """Compile the single-TU source + MPI stub to objects in a private dir.

    Keeping the objects out of the binding's ``out_dir`` prevents their stub
    ``.mod`` files from shadowing ICON's real module files while the wrapper is
    compiled.  The resulting objects are linked into ``libsolve_nh.so`` so the
    wrapper's references to module scalar symbols (e.g.
    ``__mo_mpi_MOD_process_is_mpi_parallel``) are satisfied from within the DSO
    instead of relying on hidden symbols in ``libicon.a``.

    Only ``mo_mpi`` is exposed through ``singletu_mods``; ICON's real
    ``mo_mpi.mod`` keeps these scalar variables private, so the binding cannot
    compile against it.  ``mo_sync`` and all other single-TU modules intentionally
    shadow nothing: the binding compiles against the real ICON ``.mod`` files and
    gets the correct ``t_patch``/``t_nh_state`` layouts, while the single-TU object
    still provides the link-time symbols that the SDFG references.
    """
    use_mpi_tu, mpi_stub = _stage_single_tu_sources(sdfg_dir)
    obj_dir = sdfg_dir / "singletu_objs"
    mod_dir = sdfg_dir / "singletu_mods"
    obj_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    def _compile(src: Path, obj: Path) -> Path:
        cmd = ["gfortran", "-c", "-fPIC", *fflags, "-fopenmp", f"-J{obj_dir}", str(src), "-o", str(obj)]
        print(f"[build_icon_solve_nh_libs] {' '.join(cmd)}", flush=True)
        subprocess.check_call(cmd, cwd=obj_dir)
        return obj

    mpi_stub_o = _compile(mpi_stub, obj_dir / "_mpi_stub.o")
    tu_o = _compile(use_mpi_tu, obj_dir / "solve_nonhydro_single_tu.o")

    # Only ``mo_mpi`` needs the stub interface for compilation; everything else
    # must resolve to ICON's real module files so the struct layouts match.
    for mod_name in ("mo_mpi.mod", ):
        src_mod = obj_dir / mod_name
        dst_mod = mod_dir / mod_name
        if src_mod.is_file():
            shutil.copy2(src_mod, dst_mod)
        else:
            raise FileNotFoundError(f"single-TU build did not produce {src_mod}")

    return mpi_stub_o, tu_o, mod_dir


def build_solve_nh_sdfg_single_tu(sdfg_dir: Path):
    """Lower the pre-merged single-file ``solve_nonhydro`` checkpoint to an SDFG.

    The checkpoint contains stub modules used only during DaCe lowering.  The
    generated binding is compiled against ICON's real module files, and
    ``velocity_tendencies`` / halo-exchange procedures are kept as externals that
    resolve at ICON link time.
    """
    # HLFIR emission with flang uses the *original* single-TU; the stub modules
    # are not compiled into the binding library because their type definitions
    # would shadow the real ICON modules.
    sdfg = build_sdfg(_SINGLE_TU.read_text(), out_dir=sdfg_dir / "sdfg", name="solve_nh", entry=_SOLVE_NH_ENTRY).build()
    sdfg.name = "solve_nh"
    return sdfg


def build_solve_nh_binding(icon_src: Path,
                           icon_build: Path,
                           out_dir: Path,
                           release: bool = False,
                           single_tu: bool = True):
    """Build ``libsolve_nh.so`` from either the single-TU checkpoint or real source.

    The output directory receives the .so, the .mod
    (``solve_nh_dace_bindings.mod``), and the wrapper source.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fflags = _RELEASE_FFLAGS if release else _DEBUG_FFLAGS
    cxx_flags = _RELEASE_CXX_FLAGS if release else _DEBUG_CXX_FLAGS

    print(f"[build_icon_solve_nh_libs] ICON source: {icon_src}", flush=True)
    print(f"[build_icon_solve_nh_libs] ICON build:  {icon_build}", flush=True)
    print(f"[build_icon_solve_nh_libs] output dir:  {out_dir}", flush=True)
    print(f"[build_icon_solve_nh_libs] mode:        {'release' if release else 'debug'}", flush=True)
    print(f"[build_icon_solve_nh_libs] SDFG source: {'single-TU' if single_tu else 'real ICON'}", flush=True)

    sdfg_dir = out_dir / "_sdfg_build"
    sdfg_dir.mkdir(parents=True, exist_ok=True)

    wrapper_f90 = out_dir / "solve_nh_icon_wrapper.f90"
    diff_f90 = _DACE_FORTRAN / "tests" / "icon" / "full" / "mo_solve_nh_diff.f90"

    orig_cxx_args = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args", value=" ".join(cxx_flags))
    try:
        if single_tu:
            sdfg = build_solve_nh_sdfg_single_tu(sdfg_dir)
            sdfg.build_folder = str(sdfg_dir / "dacecache")
            # Compile the single-TU stub modules to objects in a private directory
            # so their .mod files do NOT shadow ICON's real modules while the
            # binding is compiled.  The objects are then linked into
            # libsolve_nh.so to satisfy the wrapper's references to module scalar
            # symbols that are hidden inside libicon.a.
            mpi_stub_o, tu_o, singletu_mod_dir = _compile_single_tu_objects(sdfg_dir, fflags)
            prelude_sources = [diff_f90]
            extra_sources = [wrapper_f90, mpi_stub_o, tu_o]
            bind_c_shim = True
        else:
            sdfg = build_solve_nh_sdfg(icon_src.resolve(), icon_build.resolve(), sdfg_dir)
            sdfg.name = "solve_nh"
            sdfg.build_folder = str(sdfg_dir / "dacecache")
            singletu_mod_dir = None
            prelude_sources = [diff_f90]
            extra_sources = [wrapper_f90]
            bind_c_shim = False

        iface = build_auto_interface(sdfg._fortran_interface_raw, "solve_nh")
        plan = FlattenPlan.from_dict(sdfg._flatten_plan_raw or {})

        wrapper_f90.write_text(_render_icon_wrapper())
        icon_incs = tuple(f"-I{d}" for d in _icon_mod_dirs(icon_build))
        # Single-TU mo_mpi/mo_sync expose symbols the real modules keep private;
        # put that include dir first so the wrapper sees them, while every other
        # module resolves to ICON's real .mod files.
        if singletu_mod_dir is not None:
            icon_incs = (f"-I{singletu_mod_dir}", *icon_incs)

        lib = build_fortran_library(
            sdfg,
            iface=iface,
            plan=plan,
            out_dir=str(out_dir),
            name="solve_nh",
            prelude_sources=prelude_sources,
            extra_sources=extra_sources,
            bind_c_shim=bind_c_shim,
            flags=(*fflags, *icon_incs),
        )
    finally:
        dace.Config.set("compiler", "cpu", "args", value=orig_cxx_args)

    print(f"[build_icon_solve_nh_libs] artifact: {lib.so_path}", flush=True)
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icon-src", type=Path, required=True, help="ICON source tree (e.g. tests/icon/full/icon-model).")
    ap.add_argument("--icon-build",
                    type=Path,
                    required=True,
                    help="Build dir of the ICON CONFIGURATION this lib will be linked into.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where libsolve_nh.so + .mod files go.")
    ap.add_argument("--release", action="store_true", help="Use -O3 -fno-fast-math -ffp-contract=off instead of -O0.")
    ap.add_argument("--single-tu",
                    action="store_true",
                    default=True,
                    help="Build the SDFG from the pre-merged single-TU checkpoint (default).")
    ap.add_argument("--real-source", action="store_true", help="Build the SDFG from ICON's real mo_solve_nonhydro.f90.")
    args = ap.parse_args()

    if not shutil.which("gfortran") or not have_flang():
        print("error: need gfortran + an LLVM flang on PATH", file=sys.stderr)
        return 1
    if not (args.icon_build / "Makefile").is_file():
        print(f"error: no ICON build at {args.icon_build} (expected a configured Makefile)", file=sys.stderr)
        return 1

    single_tu = args.single_tu and not args.real_source
    build_solve_nh_binding(args.icon_src, args.icon_build, args.out_dir, release=args.release, single_tu=single_tu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
