"""Build the dycore SDFG end-to-end from ICON's REAL
``mo_solve_nonhydro.f90`` (3166 LoC), with ``velocity_tendencies``
kept as a per-member-SoA external (resolved at link time to the
inner velocity SDFG's bind-C entry) and ``sync_patch_array`` kept
as an opaque external (Fortran halo exchange the bridge can't
lower; an iso_c wrapper for it is step 3 of the integration plan).

This is step 2 of the source-derived-bindings plan: the dycore
SDFG carries member offsets for ICON's REAL ``t_nh_state`` /
``t_int_state`` / ``t_patch`` layouts, so when ICON's own
``mo_solve_nonhydro`` call site dispatches into our binding, the
flattened SoA leaves it hands the SDFG land at the right addresses.

The test composes a translation unit from the icon-model
submodule via :mod:`dace_fortran.flang_codebase`, registers the
externals, then drives ``build_sdfg_from_hlfir``.  Skipped if the
submodule isn't checked out, or if flang / OpenMPI is absent.
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


#: Per-test xfail/skip register for flang-new-21 bugs that don't have
#: a workaround yet.  Keys are the test ID's ``fc`` parameter; values
#: are the flang issue summary the test would otherwise expose.  Move
#: a test out of the register once flang ships a fix.  Empty for now
#: -- our wrapper sources don't trip any known flang-21 ICE on the
#: full-compile path.
_FLANG_KNOWN_BUGS: dict = {}


#: Minimal stubs for the modules ``icon_sync_iso_c.f90`` USEs.  Used
#: when we want a per-compiler ``.mod`` to compile against without
#: needing a real ICON build (whose ``.mod`` files are compiler-
#: specific).  Every symbol the wrapper references appears here as
#: a thin shell; concrete bodies are no-ops.
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
    """Per-compiler ``-J`` / ``-module`` flag for writing emitted
    ``.mod`` files to ``mod_dir``.  Each compiler spells it
    differently."""
    if fc_name == "nvfortran":
        return ["-module", str(mod_dir)]
    if fc_name == "gfortran":
        return ["-J", str(mod_dir)]
    # flang-new accepts -J as well; ``-module-dir`` is the canonical
    # spelling but -J is the documented alias since LLVM 17.
    return ["-J", str(mod_dir)]


def _fc_pic_flag(fc_name: str) -> str:
    """PIC flag; nvfortran's spelling is lowercase."""
    return "-fpic" if fc_name == "nvfortran" else "-fPIC"


_HERE = Path(__file__).resolve().parent
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))
_ICON_BUILD = Path(os.environ.get(
    "ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_SOLVE_NH_SRC = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_solve_nonhydro.f90"
_SOLVE_NH_BAK = _SOLVE_NH_SRC.with_suffix(".f90.bak")

_CACHE_DIR = Path(os.environ.get(
    "DACE_FORTRAN_CACHE", str(Path.home() / ".cache" / "dace-fortran")))

_SOLVE_NH_TARGET = "src/atm_dyn_iconam/mo_solve_nonhydro.o"
_SOLVE_NH_ENTRY = "mo_solve_nonhydro::solve_nh"

_HAVE_FLANG = shutil.which("flang-new-21") is not None
_HAVE_OPENMPI = find_openmpi_include() is not None


def _real_source() -> Path:
    """Pristine ICON solve_nonhydro source.  Prefers
    ``mo_solve_nonhydro.f90.bak`` when present so a developer-patched
    live file doesn't perturb the test."""
    return _SOLVE_NH_BAK if _SOLVE_NH_BAK.is_file() else _SOLVE_NH_SRC


def _have_icon() -> bool:
    """Submodule checked out (build dir is optional)."""
    return _real_source().is_file()


def _icon_search_dirs() -> list:
    """Same closure roots as the velocity test -- ``mo_solve_nonhydro``
    transitively USEs the same library trees ICON itself bundles."""
    return [
        _ICON_SRC / "src",
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: ICON's release-frozen ``-D`` defines for the standard CPU build of
#: solve_nonhydro.o.  Same set the velocity test uses; both objects
#: compile under the same configure.
_ICON_DEFINES_FALLBACK = (
    "HAVE_CDI_GRIB2", "HAVE_FC_ATTRIBUTE_CONTIGUOUS",
    "ICON_MPI_SUBVERSION=1", "ICON_MPI_VERSION=3",
    "__HAVE_QUAD_PRECISION", "__ICON__", "__LOOP_EXCHANGE",
    "__NO_ICON_COMIN__", "__NO_ICON_OCEAN__", "__NO_ICON_TESTBED__",
    "__NO_ICON_WAVES__", "__NO_JSBACH_HD__", "__NO_JSBACH__",
    "__NO_QUINCY__", "__NO_RAGNAROK__",
)


def _icon_compile_args() -> dict:
    if (_ICON_BUILD / "Makefile").is_file():
        return dace_fortran.extract_make_compile_args(
            makefile_dir=_ICON_BUILD, target=_SOLVE_NH_TARGET)
    return {"defines": list(_ICON_DEFINES_FALLBACK),
            "include_dirs": [_ICON_SRC / "src/include"]}


#: ICON utility procedures whose bodies the bridge can't lower
#: (string-LEN_TRIM scans, polymorphic dispatch, MPI infrastructure)
#: but ``solve_nh`` only calls structurally.  Stripping them BEFORE
#: ``hlfir-inline-all`` keeps them external.  Same list as the
#: velocity test, plus ``mo_sync``'s halo-exchange procedures
#: (real binding goes through an iso_c wrapper in step 3) and
#: MPI barrier / timer ops.
_ICON_EXTERNAL_STUBS = (
    # mo_exception -- diagnostic / error termination; no numerical
    # effect.  Body has LEN_TRIM string scans the bridge can't lower.
    "finish", "message", "message_text", "warning",
    "print_status", "print_value", "init_logger",
    # mo_real_timer / mo_timer -- profiling hooks.
    "timer_start", "timer_stop", "new_timer", "delete_timer",
    # mo_mpi -- MPI sync barrier.
    "work_mpi_barrier",
    # mo_sync's halo-exchange entry points -- the procedures
    # solve_nh directly calls.  ``INTERFACE sync_patch_array`` is a
    # generic that resolves at compile time to one of the type-rank
    # specific ``sync_patch_array_2d_dp`` / ``_3d_int`` / ... names,
    # so we externalize the SPECIFIC procedures (flang emits direct
    # calls to those after generic resolution).  Stripping their
    # bodies removes everything they transitively reach
    # (``mo_communication``'s polymorphic ``exchange_data_*`` family,
    # the ``CLASS(*)`` communication-pattern dispatch), so we DO NOT
    # need to enumerate the downstream callees.  Step 3 wires these
    # to a thin iso_c wrapper that forwards to Fortran
    # ``sync_patch_array`` at runtime; until then, the bridge emits
    # an opaque ExternalCall library node.
    "sync_patch_array_2d_sp", "sync_patch_array_2d_dp",
    "sync_patch_array_2d_int", "sync_patch_array_2d_bool",
    "sync_patch_array_3d_sp", "sync_patch_array_3d_dp",
    "sync_patch_array_3d_int", "sync_patch_array_3d_bool",
    "sync_patch_array_4de1_sp", "sync_patch_array_4de1_dp",
    "sync_patch_array_mult_f3din_sp",
    "sync_patch_array_mult_f3din_f4din_sp",
    "sync_patch_array_mult_f3din_f3din_arr_sp",
    "sync_patch_array_mult_f3din_dp",
    "sync_patch_array_mult_f3din_f4din_dp",
    "sync_patch_array_mult_f3din_f3din_arr_dp",
    "sync_patch_array_mult_mixprec",
    "check_patch_array_2d_sp", "check_patch_array_2d_dp",
    "check_patch_array_3d_sp", "check_patch_array_3d_dp",
    "check_patch_array_4d_sp", "check_patch_array_4d_dp",
    # mo_velocity_advection -- the headline external.  Resolves at
    # link time to the inner velocity binding's bind-C entry point.
    "velocity_tendencies",
)


pytestmark = pytest.mark.skipif(
    not (_HAVE_FLANG and _HAVE_OPENMPI),
    reason="needs flang-new-21 + OpenMPI")


@pytest.mark.skipif(
    not _have_icon(),
    reason="icon-model submodule not checked out; run "
           "`git submodule update --init --recursive` to pull it")
def test_emit_hlfir_for_icon_solve_nh(tmp_path: Path):
    """``emit_hlfir_from_codebase`` produces a non-trivial HLFIR for
    ICON's real ``mo_solve_nonhydro.f90``.  Sanity check before
    lowering to SDFG."""
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
    # solve_nonhydro is bigger than velocity_tendencies (3166 LoC
    # entry source vs 1130 LoC for velocity).  Both lower to hundreds
    # of MB of HLFIR.  A small file would mean truncation.
    assert out.stat().st_size > 100 * 1024 * 1024


@pytest.mark.skipif(
    not _have_icon(),
    reason="icon-model submodule not checked out")
def test_build_sdfg_for_icon_solve_nh(tmp_path: Path):
    """Build a DaCe SDFG from ICON's real ``mo_solve_nonhydro``.

    ``velocity_tendencies`` is kept as a per-member-SoA external (the
    dycore SDFG calls into the inner velocity binding via its bind-C
    entry point ``velocity_tendencies_c``); ``sync_patch_array`` /
    ``sync_patch_array_mult`` stay as opaque stub externals (their
    real binding is wired in step 3 of the integration plan).

    Asserts the SDFG builds and carries the entry's name -- the
    same pin the velocity-from-real-icon test uses, scaled to the
    dycore."""
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
    for sym in _ICON_EXTERNAL_STUBS:
        dace_fortran.keep_external(sym, stub=True)
    try:
        sdfg = dace_fortran.build_sdfg_from_hlfir(
            hlfir, entry=_SOLVE_NH_ENTRY)
    finally:
        dace_fortran.clear_external_registry()

    assert sdfg is not None
    assert sdfg.name
    name_lc = sdfg.name.lower()
    assert "solve_nh" in name_lc or "solve_nonhydro" in name_lc, \
        f"SDFG name doesn't carry the expected entry: {sdfg.name!r}"
    # API-level structural validation: dangling memlets / orphan
    # connectors / missing access nodes etc. that the bridge could in
    # principle emit but mustn't.
    sdfg.validate()


# ---------------------------------------------------------------------------
# Step 3: iso_c wrapper for sync_patch_array.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fc", FORTRAN_COMPILERS)
def test_sync_iso_c_wrapper_source_parses(fc, tmp_path: Path):
    """The iso_c wrapper Fortran source is well-formed under every
    Fortran compiler we ship binding-shim recipes for: gfortran (ICON
    CPU default), flang-new-21 (the bridge's own frontend), nvfortran
    (NVHPC, ICON GPU build).  We compile minimal stubs of the
    USE'd modules side-by-side so the parse succeeds standalone, no
    ICON build required.

    Pins the wrapper's iso_c interface as a regression gate across
    the toolchain matrix: a typo in any of the ``bind(c, name=...)``
    signatures, or a compiler-specific feature drift, fails the
    parse independently of whether the icon-model submodule is
    built."""
    fc_name, fc_path = fc
    # Minimal stubs for the modules the wrapper USEs.  Only the symbols
    # the wrapper references need to exist; concrete bodies are no-ops.
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
    # ``-fsyntax-only`` (gfortran / flang-new) vs ``-c -o <obj>``
    # (nvfortran, which has no syntax-only knob).
    syntax_argv = syntax_check_argv(fc_name, tmp_path)
    if fc_name == "nvfortran":
        mod_out_flag = ["-module", str(tmp_path)]
    elif fc_name == "flang-new-21":
        # ``flang-new -fsyntax-only`` skips module emission; the
        # stubs file is parsed in-line with the wrapper, so no -J/
        # -module needed (all symbols resolve within the same
        # compile invocation).
        mod_out_flag = []
    else:  # gfortran
        mod_out_flag = ["-J", str(tmp_path)]
    subprocess.check_call(
        [fc_path, *syntax_argv,
         *fortran_compiler_flags(fc_name),
         *mod_out_flag,
         str(stubs), str(_SYNC_WRAPPER_SRC)],
        cwd=str(tmp_path))


@pytest.mark.parametrize("fc", FORTRAN_COMPILERS)
def test_sync_iso_c_wrapper_full_compile_link(fc, tmp_path: Path):
    """Full per-compiler compile + link path: emit ``.o`` from the
    stubs, then from the wrapper, link both into a ``.so``, and
    verify the bind-C symbols flang/gfortran/nvfortran each declare
    are actually exported.

    This catches issues a ``-fsyntax-only`` check misses: kind-mismatch
    truncation, ``c_f_pointer`` shape-array signature drift, ``LOGICAL``
    bit-width ABI bugs, ``c_bool`` truth-value layout, the ``USE``-merge
    handling between modules.  Anything the runtime would surface only
    at link time.

    For flang specifically the test injects ``LIBRARY_PATH`` to point
    at a user-local ``libflang_rt.runtime.a`` (see ``_fc.FLANG_RT_HINT``
    for build instructions); it skips cleanly when no runtime is
    reachable.  Known flang-21 bugs that block specific binding shapes
    register their case ID in ``_FLANG_KNOWN_BUGS``; this test
    ``xfail``s those without poisoning the gfortran / nvfortran
    variants."""
    fc_name, fc_path = fc

    # flang full-link needs ``libflang_rt.runtime.a``.  Find it once
    # (via ROCm's copy or a locally-built one) and skip cleanly if
    # neither is reachable -- the syntax-only test above still pins
    # the wrapper interface even on hosts that can't fully link.
    if fc_name.startswith("flang") and find_flang_runtime_dir() is None:
        pytest.skip(FLANG_RT_HINT)

    # Each compiler emits its OWN ``.mod`` format, so we compile the
    # stubs with the SAME compiler used for the wrapper -- both
    # invocations within one test agree on the mod format.
    stubs = tmp_path / "stubs.f90"
    stubs.write_text(_SYNC_WRAPPER_STUBS)

    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    pic = _fc_pic_flag(fc_name)

    # Compile stubs -> stubs.o + per-compiler .mod files.
    subprocess.check_call(
        [fc_path, "-c", pic,
         *fortran_compiler_flags(fc_name),
         *_fc_mod_flag(fc_name, mod_dir),
         str(stubs), "-o", str(tmp_path / "stubs.o")],
        cwd=str(tmp_path),
        env=env_with_flang_runtime(fc_name))

    # Compile wrapper against those .mods.
    wrapper_obj = tmp_path / "icon_sync_iso_c.o"
    subprocess.check_call(
        [fc_path, "-c", pic,
         *fortran_compiler_flags(fc_name),
         f"-I{mod_dir}",
         *_fc_mod_flag(fc_name, mod_dir),
         str(_SYNC_WRAPPER_SRC), "-o", str(wrapper_obj)],
        cwd=str(tmp_path),
        env=env_with_flang_runtime(fc_name))

    # Link wrapper + stubs into a .so (no main entry needed -- the
    # shared object exports the bind(c) symbols for runtime
    # resolution).
    so_path = tmp_path / "libicon_sync_iso_c_test.so"
    subprocess.check_call(
        [fc_path, "-shared", pic,
         str(wrapper_obj), str(tmp_path / "stubs.o"),
         "-o", str(so_path)],
        cwd=str(tmp_path),
        env=env_with_flang_runtime(fc_name))

    # The four bind-C names must appear in the .so's dynamic symbol
    # table.  ``nm -D`` for the shared object gives us the export list
    # in a portable form (works for nvfortran's output too).
    sym_out = subprocess.check_output(["nm", "-D", str(so_path)], text=True)
    for sym in (
        "sync_patch_array_3d_dp_c",
        "sync_patch_array_mult_2_dp_c",
        "sync_patch_array_mult_3_dp_c",
        "sync_patch_array_mult_mixprec_1sp_1dp_c",
    ):
        # Strict ``T`` (text, defined) marker so an UNDEFINED symbol
        # (which would mean the wrapper's bind(c) didn't actually
        # materialise) fails the test.
        assert f" T {sym}" in sym_out, (
            f"{fc_name} produced a .so MISSING the bind-C symbol "
            f"{sym!r}; nm -D output:\n{sym_out}")


@pytest.mark.skipif(not _have_icon(),
                    reason="icon-model submodule not checked out")
def test_sync_iso_c_wrapper_builds_against_icon_mods(tmp_path: Path, icon_build):
    """Build the wrapper into ``libicon_sync_iso_c.so`` against ICON's
    own ``.mod`` files.  The ``icon_build`` fixture configures + builds
    ICON on demand so the ``.mod`` ``-I`` set exists."""
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
    """Pin the wrapper's ``bind(c, name=...)`` symbol set.  Step 4's
    integration test will register each of these as the ``c_name``
    for the corresponding ``mo_sync`` generic-interface specialisation,
    so we want a regression gate that catches a typo in any of the
    bind-C names independently of whether the SDFG build path runs."""
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
