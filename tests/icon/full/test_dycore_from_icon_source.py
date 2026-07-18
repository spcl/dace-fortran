"""Build the dycore SDFG end-to-end from ICON's REAL ``mo_solve_nonhydro.f90``
(3166 LoC), with ``velocity_tendencies`` kept as a per-member-SoA external
(resolved at link time to the inner velocity SDFG's bind-C entry) and
``sync_patch_array`` kept as an opaque external (real binding is step 3).

Step 2 of the source-derived-bindings plan: the dycore SDFG carries member
offsets for ICON's REAL ``t_nh_state``/``t_int_state``/``t_patch`` layouts, so
ICON's own call site's flattened SoA leaves land at the right addresses.

Composes a TU from the icon-model submodule, registers externals, drives
``build_sdfg_from_hlfir``.  Skipped if the submodule/flang/OpenMPI is absent.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import dace_fortran
from dace_fortran.flang_codebase import find_openmpi_include

from ._fc import (
    FLANG_RT_HINT,
    FORTRAN_COMPILERS,
    env_with_flang_runtime,
    find_flang_runtime_dir,
    fortran_compiler_flags,
    syntax_check_argv,
)
from ._icon_sync_iso_c_build import (
    _WRAPPER_SRC as _SYNC_WRAPPER_SRC,
    build_icon_sync_iso_c_so,
)

#: Per-test xfail/skip register for flang-new-21 bugs without a workaround yet.
#: Keys are the ``fc`` param; values are the flang issue summary.  Empty for now.
_FLANG_KNOWN_BUGS: dict = {}

#: Minimal stubs for the modules ``icon_sync_iso_c.f90`` USEs -- a per-compiler
#: ``.mod`` to compile against without a real ICON build; concrete bodies are no-ops.
_SYNC_WRAPPER_STUBS = """\
MODULE mo_kind
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(15)
  INTEGER, PARAMETER :: sp = SELECTED_REAL_KIND(6)
END MODULE mo_kind

MODULE mo_model_domain
  IMPLICIT NONE
  TYPE :: t_patch
    INTEGER :: id
  END TYPE t_patch
  TYPE(t_patch), TARGET :: p_patch(8)
END MODULE mo_model_domain

MODULE mo_sync
  USE mo_kind, ONLY: dp, sp
  USE mo_model_domain, ONLY: t_patch
  IMPLICIT NONE
  INTERFACE sync_patch_array
    MODULE PROCEDURE sync_patch_array_3d_dp
  END INTERFACE
  INTERFACE sync_patch_array_mult
    MODULE PROCEDURE sync_patch_array_mult_dp
  END INTERFACE
  INTERFACE sync_patch_array_mult_mixprec
    MODULE PROCEDURE sync_patch_array_mult_mixprec_impl
  END INTERFACE
CONTAINS
  SUBROUTINE sync_patch_array_3d_dp(typ, patch, arr, lacc, opt_varname)
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), TARGET, INTENT(IN) :: patch
    REAL(dp), INTENT(INOUT) :: arr(:,:,:)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN=*), TARGET, INTENT(IN), OPTIONAL :: opt_varname
  END SUBROUTINE
  SUBROUTINE sync_patch_array_mult_dp(typ, patch, nfields, lacc, &
                                       f3din1, f3din2, f3din3)
    INTEGER, INTENT(IN) :: typ, nfields
    TYPE(t_patch), TARGET, INTENT(IN) :: patch
    LOGICAL, INTENT(IN) :: lacc
    REAL(dp), INTENT(INOUT), OPTIONAL :: f3din1(:,:,:), f3din2(:,:,:), f3din3(:,:,:)
  END SUBROUTINE
  SUBROUTINE sync_patch_array_mult_mixprec_impl(typ, patch, n_sp, n_dp, &
                                                 f3din1_sp, f3din1_dp, lacc)
    INTEGER, INTENT(IN) :: typ, n_sp, n_dp
    TYPE(t_patch), TARGET, INTENT(IN) :: patch
    REAL(sp), INTENT(INOUT), OPTIONAL :: f3din1_sp(:,:,:)
    REAL(dp), INTENT(INOUT), OPTIONAL :: f3din1_dp(:,:,:)
    LOGICAL, INTENT(IN) :: lacc
  END SUBROUTINE
END MODULE mo_sync
"""


def _fc_mod_flag(fc_name: str, mod_dir: str) -> list:
    """Per-compiler ``-J``/``-module`` flag for writing emitted ``.mod`` files."""
    if fc_name == "nvfortran":
        return ["-module", str(mod_dir)]
    if fc_name == "gfortran":
        return ["-J", str(mod_dir)]
    # flang-new accepts -J too; -module-dir is canonical but -J is the documented
    # alias since LLVM 17.
    return ["-J", str(mod_dir)]


def _fc_pic_flag(fc_name: str) -> str:
    """PIC flag; nvfortran's spelling is lowercase."""
    return "-fpic" if fc_name == "nvfortran" else "-fPIC"


_HERE = Path(__file__).resolve().parent
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))
_ICON_BUILD = Path(os.environ.get("ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_SOLVE_NH_SRC = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
_SOLVE_NH_BAK = _SOLVE_NH_SRC.with_suffix(".f90.bak")

_CACHE_DIR = Path(os.environ.get("DACE_FORTRAN_CACHE", str(Path.home() / ".cache" / "dace-fortran")))

_SOLVE_NH_TARGET = "src/atm_dyn_iconam/mo_solve_nonhydro.o"
_SOLVE_NH_ENTRY = "mo_solve_nonhydro::solve_nh"

_HAVE_FLANG = shutil.which("flang-new-21") is not None
_HAVE_OPENMPI = find_openmpi_include() is not None


def _real_source() -> Path:
    """Pristine ICON solve_nonhydro source; prefers the ``.bak`` so a developer-patched
    live file doesn't perturb the test."""
    return _SOLVE_NH_BAK if _SOLVE_NH_BAK.is_file() else _SOLVE_NH_SRC


def _have_icon() -> bool:
    """Submodule checked out (build dir is optional)."""
    return _real_source().is_file()


def _icon_search_dirs() -> list:
    """Same closure roots as the velocity test (mo_solve_nonhydro transitively USEs the
    same library trees)."""
    return [
        _ICON_SRC / "src",
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: ICON's release-frozen ``-D`` defines for the standard CPU build of solve_nonhydro.o.
#: Same set the velocity test uses.
_ICON_DEFINES_FALLBACK = (
    "HAVE_CDI_GRIB2",
    "HAVE_FC_ATTRIBUTE_CONTIGUOUS",
    "ICON_MPI_SUBVERSION=1",
    "ICON_MPI_VERSION=3",
    "__HAVE_QUAD_PRECISION",
    "__ICON__",
    "__LOOP_EXCHANGE",
    "__NO_ICON_COMIN__",
    "__NO_ICON_OCEAN__",
    "__NO_ICON_TESTBED__",
    "__NO_ICON_WAVES__",
    "__NO_JSBACH_HD__",
    "__NO_JSBACH__",
    "__NO_QUINCY__",
    "__NO_RAGNAROK__",
)


def _icon_compile_args() -> dict:
    if (_ICON_BUILD / "Makefile").is_file():
        return dace_fortran.extract_make_compile_args(makefile_dir=_ICON_BUILD, target=_SOLVE_NH_TARGET)
    return {"defines": list(_ICON_DEFINES_FALLBACK), "include_dirs": [_ICON_SRC / "src/include"]}


#: ICON utility procedures whose bodies the bridge can't lower (string scans,
#: polymorphic dispatch, MPI infra) but ``solve_nh`` only calls structurally.
#: Stripped BEFORE ``hlfir-inline-all`` to keep them external.  Same list as the
#: velocity test, plus ``mo_sync``'s halo-exchange procedures and MPI barrier/timers.
_ICON_EXTERNAL_STUBS = (
    # mo_exception: diagnostic/error termination, no numerical effect; LEN_TRIM scans
    # the bridge can't lower.
    "finish",
    "message",
    "message_text",
    "warning",
    "print_status",
    "print_value",
    "init_logger",
    # mo_real_timer / mo_timer -- profiling hooks.
    "timer_start",
    "timer_stop",
    "new_timer",
    "delete_timer",
    # mo_mpi -- MPI sync barrier.
    "work_mpi_barrier",
    # mo_sync's halo-exchange entry points solve_nh calls directly (INTERFACE
    # sync_patch_array resolves at compile time to type-rank specifics like
    # sync_patch_array_2d_dp, so we externalize those).  Stripping removes
    # everything they transitively reach, so no need to enumerate downstream
    # callees.  Step 3 wires these to an iso_c wrapper; until then the bridge
    # emits an opaque ExternalCall.
    "sync_patch_array_2d_sp",
    "sync_patch_array_2d_dp",
    "sync_patch_array_2d_int",
    "sync_patch_array_2d_bool",
    "sync_patch_array_3d_sp",
    "sync_patch_array_3d_dp",
    "sync_patch_array_3d_int",
    "sync_patch_array_3d_bool",
    "sync_patch_array_4de1_sp",
    "sync_patch_array_4de1_dp",
    "sync_patch_array_mult_f3din_sp",
    "sync_patch_array_mult_f3din_f4din_sp",
    "sync_patch_array_mult_f3din_f3din_arr_sp",
    "sync_patch_array_mult_f3din_dp",
    "sync_patch_array_mult_f3din_f4din_dp",
    "sync_patch_array_mult_f3din_f3din_arr_dp",
    "sync_patch_array_mult_mixprec",
    "check_patch_array_2d_sp",
    "check_patch_array_2d_dp",
    "check_patch_array_3d_sp",
    "check_patch_array_3d_dp",
    "check_patch_array_4d_sp",
    "check_patch_array_4d_dp",
    # mo_velocity_advection: headline external, resolves at link time to the inner
    # velocity binding's bind-C entry.
    "velocity_tendencies",
)

# Reads ICON's real mo_solve_nonhydro through the icon-model submodule; only the
# heavy CI lane checks it out -> `long`.
pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not (_HAVE_FLANG and _HAVE_OPENMPI), reason="needs flang-new-21 + OpenMPI"),
]


@pytest.mark.skipif(not _have_icon(),
                    reason="icon-model submodule not checked out; run "
                    "`git submodule update --init --recursive` to pull it")
def test_emit_hlfir_for_icon_solve_nh(tmp_path: Path):
    """``emit_hlfir_from_codebase`` produces non-trivial HLFIR for ICON's real
    ``mo_solve_nonhydro.f90``."""
    args = _icon_compile_args()
    out = dace_fortran.emit_hlfir_from_codebase(
        entry_source=_real_source().read_text(),
        out_path=tmp_path / "solve_nh.hlfir",
        search_dirs=_icon_search_dirs(),
        library_stubs=["mpi", "netcdf"],
        defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
        include_dirs=args["include_dirs"],
        cache_dir=_CACHE_DIR,
    )
    assert out.is_file()
    # solve_nonhydro (3166 LoC) > velocity_tendencies (1130 LoC); both lower to
    # hundreds of MB of HLFIR -- a small file would mean truncation.
    assert out.stat().st_size > 100 * 1024 * 1024


@pytest.mark.skipif(not _have_icon(), reason="icon-model submodule not checked out")
def test_build_sdfg_for_icon_solve_nh(tmp_path: Path):
    """Build a DaCe SDFG from ICON's real ``mo_solve_nonhydro``: ``velocity_tendencies``
    kept as a per-member-SoA external, ``sync_patch_array``/``_mult`` as opaque stub
    externals (real binding wired in step 3).  Asserts the SDFG builds and carries the
    entry's name."""
    args = _icon_compile_args()
    hlfir = dace_fortran.emit_hlfir_from_codebase(
        entry_source=_real_source().read_text(),
        out_path=tmp_path / "solve_nh.hlfir",
        search_dirs=_icon_search_dirs(),
        library_stubs=["mpi", "netcdf"],
        defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
        include_dirs=args["include_dirs"],
        cache_dir=_CACHE_DIR,
    )
    dace_fortran.clear_external_registry()
    # Don't-inline + DON'T-emit every infrastructure procedure: ignore list drops
    # their calls.
    dace_fortran.apply_external_functions(do_not_emit=_ICON_EXTERNAL_STUBS)
    try:
        sdfg = dace_fortran.build_sdfg_from_hlfir(hlfir, entry=_SOLVE_NH_ENTRY)
    finally:
        dace_fortran.clear_external_registry()

    assert sdfg is not None
    assert sdfg.name
    name_lc = sdfg.name.lower()
    assert "solve_nh" in name_lc or "solve_nonhydro" in name_lc, \
        f"SDFG name doesn't carry the expected entry: {sdfg.name!r}"
    # API-level structural validation: dangling memlets/orphan connectors/missing
    # access nodes.
    sdfg.validate()


# ---------------------------------------------------------------------------
# Step 3: iso_c wrapper for sync_patch_array.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fc", FORTRAN_COMPILERS)
def test_sync_iso_c_wrapper_source_parses(fc, tmp_path: Path):
    """iso_c wrapper source is well-formed under every Fortran compiler we ship
    binding-shim recipes for: gfortran, flang-new-21, nvfortran.  Regression gate for
    the ``bind(c, name=...)`` signatures, independent of whether icon-model is built."""
    fc_name, fc_path = fc
    # Minimal stubs for the modules the wrapper USEs; concrete bodies are no-ops.
    stubs = tmp_path / "stubs.f90"
    stubs.write_text("""\
MODULE mo_kind
  IMPLICIT NONE
  INTEGER, PARAMETER :: dp = SELECTED_REAL_KIND(15)
  INTEGER, PARAMETER :: sp = SELECTED_REAL_KIND(6)
END MODULE mo_kind

MODULE mo_model_domain
  IMPLICIT NONE
  TYPE :: t_patch
    INTEGER :: id
  END TYPE t_patch
  TYPE(t_patch), TARGET :: p_patch(8)
END MODULE mo_model_domain

MODULE mo_sync
  USE mo_kind, ONLY: dp, sp
  USE mo_model_domain, ONLY: t_patch
  IMPLICIT NONE
  INTERFACE sync_patch_array
    MODULE PROCEDURE sync_patch_array_3d_dp
  END INTERFACE
  INTERFACE sync_patch_array_mult
    MODULE PROCEDURE sync_patch_array_mult_dp
  END INTERFACE
  INTERFACE sync_patch_array_mult_mixprec
    MODULE PROCEDURE sync_patch_array_mult_mixprec_impl
  END INTERFACE
CONTAINS
  SUBROUTINE sync_patch_array_3d_dp(typ, patch, arr, lacc, opt_varname)
    INTEGER, INTENT(IN) :: typ
    TYPE(t_patch), TARGET, INTENT(IN) :: patch
    REAL(dp), INTENT(INOUT) :: arr(:,:,:)
    LOGICAL, INTENT(IN) :: lacc
    CHARACTER(LEN=*), TARGET, INTENT(IN), OPTIONAL :: opt_varname
  END SUBROUTINE
  SUBROUTINE sync_patch_array_mult_dp(typ, patch, nfields, lacc, &
                                       f3din1, f3din2, f3din3)
    INTEGER, INTENT(IN) :: typ, nfields
    TYPE(t_patch), TARGET, INTENT(IN) :: patch
    LOGICAL, INTENT(IN) :: lacc
    REAL(dp), INTENT(INOUT), OPTIONAL :: f3din1(:,:,:), f3din2(:,:,:), f3din3(:,:,:)
  END SUBROUTINE
  SUBROUTINE sync_patch_array_mult_mixprec_impl(typ, patch, n_sp, n_dp, &
                                                 f3din1_sp, f3din1_dp, lacc)
    INTEGER, INTENT(IN) :: typ, n_sp, n_dp
    TYPE(t_patch), TARGET, INTENT(IN) :: patch
    REAL(sp), INTENT(INOUT), OPTIONAL :: f3din1_sp(:,:,:)
    REAL(dp), INTENT(INOUT), OPTIONAL :: f3din1_dp(:,:,:)
    LOGICAL, INTENT(IN) :: lacc
  END SUBROUTINE
END MODULE mo_sync
""")
    # -fsyntax-only (gfortran/flang-new) vs -c -o <obj> (nvfortran has no syntax-only knob).
    syntax_argv = syntax_check_argv(fc_name, tmp_path)
    if fc_name == "nvfortran":
        mod_out_flag = ["-module", str(tmp_path)]
    elif fc_name == "flang-new-21":
        # flang-new -fsyntax-only skips module emission; stubs parsed in-line with the
        # wrapper, no -J/-module needed.
        mod_out_flag = []
    else:  # gfortran
        mod_out_flag = ["-J", str(tmp_path)]
    subprocess.check_call(
        [fc_path, *syntax_argv, *fortran_compiler_flags(fc_name), *mod_out_flag,
         str(stubs),
         str(_SYNC_WRAPPER_SRC)],
        cwd=str(tmp_path))


@pytest.mark.parametrize("fc", FORTRAN_COMPILERS)
def test_sync_iso_c_wrapper_full_compile_link(fc, tmp_path: Path):
    """Full per-compiler compile + link: emit ``.o`` from stubs + wrapper, link into a
    ``.so``, verify the bind-C symbols are exported.  Catches issues ``-fsyntax-only``
    misses: kind-mismatch truncation, ``c_f_pointer`` shape-array drift, ``LOGICAL``
    ABI bugs, ``USE``-merge handling -- anything surfacing only at link time.

    flang needs ``LIBRARY_PATH`` pointed at a local ``libflang_rt.runtime.a`` (see
    ``_fc.FLANG_RT_HINT``); skips cleanly when unreachable.  Known flang-21 bugs
    register in ``_FLANG_KNOWN_BUGS`` and xfail without poisoning other compilers."""
    fc_name, fc_path = fc

    # flang full-link needs libflang_rt.runtime.a; skip cleanly if unreachable -- the
    # syntax-only test above still pins the interface even on hosts that can't link.
    if fc_name.startswith("flang") and find_flang_runtime_dir() is None:
        pytest.skip(FLANG_RT_HINT)

    # Each compiler emits its OWN .mod format, so stubs compile with the SAME compiler
    # used for the wrapper.
    stubs = tmp_path / "stubs.f90"
    stubs.write_text(_SYNC_WRAPPER_STUBS)

    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    pic = _fc_pic_flag(fc_name)

    # Compile stubs -> stubs.o + per-compiler .mod files.
    subprocess.check_call([
        fc_path, "-c", pic, *fortran_compiler_flags(fc_name), *_fc_mod_flag(fc_name, mod_dir),
        str(stubs), "-o",
        str(tmp_path / "stubs.o")
    ],
                          cwd=str(tmp_path),
                          env=env_with_flang_runtime(fc_name))

    # Compile wrapper against those .mods.
    wrapper_obj = tmp_path / "icon_sync_iso_c.o"
    subprocess.check_call([
        fc_path, "-c", pic, *fortran_compiler_flags(fc_name), f"-I{mod_dir}", *_fc_mod_flag(fc_name, mod_dir),
        str(_SYNC_WRAPPER_SRC), "-o",
        str(wrapper_obj)
    ],
                          cwd=str(tmp_path),
                          env=env_with_flang_runtime(fc_name))

    # Link wrapper + stubs into a .so (no main entry needed).
    so_path = tmp_path / "libicon_sync_iso_c_test.so"
    subprocess.check_call(
        [fc_path, "-shared", pic,
         str(wrapper_obj), str(tmp_path / "stubs.o"), "-o",
         str(so_path)],
        cwd=str(tmp_path),
        env=env_with_flang_runtime(fc_name))

    # The four bind-C names must appear in the .so's dynamic symbol table (``nm -D``
    # gives a portable export list, works for nvfortran too).
    sym_out = subprocess.check_output(["nm", "-D", str(so_path)], text=True)
    for sym in (
            "sync_patch_array_3d_dp_c",
            "sync_patch_array_mult_2_dp_c",
            "sync_patch_array_mult_3_dp_c",
            "sync_patch_array_mult_mixprec_1sp_1dp_c",
    ):
        # Strict "T" (text, defined) marker so an UNDEFINED symbol fails the test.
        assert f" T {sym}" in sym_out, (f"{fc_name} produced a .so MISSING the bind-C symbol "
                                        f"{sym!r}; nm -D output:\n{sym_out}")


@pytest.mark.skipif(not _have_icon(), reason="icon-model submodule not checked out")
def test_sync_iso_c_wrapper_builds_against_icon_mods(tmp_path: Path, icon_build):
    """Build the wrapper into ``libicon_sync_iso_c.so`` against ICON's own ``.mod``
    files.  ``icon_build`` fixture configures + builds ICON on demand."""
    so_path = build_icon_sync_iso_c_so(icon_build, out_dir=tmp_path)
    assert so_path is not None
    assert so_path.is_file()
    # Sanity: ELF, has our bind(c) symbols
    output = subprocess.check_output(["nm", "-D", str(so_path)], text=True)
    for sym in (
            "sync_patch_array_3d_dp_c",
            "sync_patch_array_mult_2_dp_c",
            "sync_patch_array_mult_3_dp_c",
            "sync_patch_array_mult_mixprec_1sp_1dp_c",
    ):
        assert f" T {sym}" in output, \
            f"wrapper .so missing bind-C symbol {sym!r}; got:\n{output}"


def test_sync_iso_c_wrapper_pins_bind_c_signatures():
    """Pin the wrapper's ``bind(c, name=...)`` symbol set -- regression gate for a typo
    in any bind-C name, independent of whether the SDFG build path runs."""
    src = _SYNC_WRAPPER_SRC.read_text()
    for sym in (
            "sync_patch_array_3d_dp_c",
            "sync_patch_array_mult_2_dp_c",
            "sync_patch_array_mult_3_dp_c",
            "sync_patch_array_mult_mixprec_1sp_1dp_c",
    ):
        assert f"name='{sym}'" in src, \
            f"wrapper missing bind-C entry {sym!r}"
    # ICON-side imports the wrapper depends on.
    for use in ("USE mo_sync", "USE mo_model_domain", "USE mo_kind"):
        assert use in src, f"wrapper missing {use!r}"
