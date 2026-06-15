"""Build the velocity_tendencies SDFG end-to-end from TWO source paths:

1. The existing self-contained ``velocity_full.f90`` stub kernel
   (already known to bridge cleanly -- this run pins backwards
   compatibility for the recipe).
2. ICON's REAL ``mo_velocity_advection.f90``, pulled via the
   ``tests/icon/full/icon-model`` git submodule (pinned to release tag
   ``icon-2026.04-public``), with its USE closure resolved by
   :func:`dace_fortran.prepare_flang_translation_unit`.

For the real-source path the test composes a translation unit via
the helpers in :mod:`dace_fortran.flang_codebase` (merge + library
stubs + ICON's own ``-D`` defines extracted from its build), runs
flang to emit HLFIR, and lowers that to a DaCe SDFG.  Both paths are
parametrized so a regression on either route surfaces immediately.

Skipped if the icon-model submodule isn't checked out (run
``git submodule update --init --recursive tests/icon/full/icon-model``
to pull it) or if flang-new-21 / OpenMPI is absent.
"""
import shutil
from pathlib import Path

import pytest

import dace_fortran
from dace_fortran.flang_codebase import find_openmpi_include

_HERE = Path(__file__).resolve().parent
_STUB_SOURCE = _HERE / "velocity_full.f90"

#: The ICON checkout pinned by the ``icon-model`` git submodule.
#: ``ICON_SRC`` env var lets a developer point at a separate
#: checkout (e.g. the existing ``/home/primrose/Work/icon-model-public``
#: tree) without re-cloning into the submodule.
import os

_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))

#: ICON's own build dir (Makefile + .o per source); supplies the
#: ``-D`` defines and ``-I`` paths the velocity object compiles with.
#: Override with ``ICON_BUILD`` to point at a non-default build tree.
_ICON_BUILD = Path(os.environ.get("ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_VELOCITY_REAL = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_velocity_advection.f90"
# Some ICON workflows keep a pristine backup at ``.bak`` while
# patching the live file; prefer it when present so the test is
# stable against the patch state.
_VELOCITY_REAL_BAK = _VELOCITY_REAL.with_suffix(".f90.bak")

_CACHE_DIR = Path(os.environ.get("DACE_FORTRAN_CACHE", str(Path.home() / ".cache" / "dace-fortran")))

_VELOCITY_TARGET = "src/atm_dyn_iconam/mo_velocity_advection.o"
_VELOCITY_ENTRY = "velocity_tendencies"

_HAVE_FLANG = shutil.which("flang-new-21") is not None
_HAVE_OPENMPI = find_openmpi_include() is not None


def _real_velocity_source() -> Path:
    """The pristine ICON velocity source.  Prefers ``mo_velocity_advection.f90.bak``
    when present so a developer-patched live file doesn't perturb the test."""
    return _VELOCITY_REAL_BAK if _VELOCITY_REAL_BAK.is_file() else _VELOCITY_REAL


def _have_icon() -> bool:
    """Submodule checked out (build dir is optional -- when present
    its ``-D`` / ``-I`` set is used verbatim, otherwise the
    release-frozen fallback in :data:`_ICON_DEFINES_FALLBACK`)."""
    return _real_velocity_source().is_file()


def _icon_search_dirs() -> list:
    """The USE-graph closure of ``mo_velocity_advection`` lives under
    ICON's ``src/`` plus a handful of external library trees that
    ICON itself bundles (fortran-support, mtime, iconmath, cdi,
    memman, support).  Bisected down to exactly this set."""
    return [
        _ICON_SRC / "src",
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: ICON utility procedures that velocity_tendencies calls structurally
#: (error reporting, timer hooks, version stamps) but whose bodies the
#: bridge doesn't need to lower.  Stripping them BEFORE
#: ``hlfir-inline-all`` keeps their unlowerable internals
#: (``fir.iterate_while`` LEN_TRIM scans, ``CLASS(*)`` polymorphism)
#: from being inlined into the entry.
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

#: ICON's standard CPU build defines for release icon-2026.04-public,
#: lifted verbatim from a ``make -n src/atm_dyn_iconam/mo_velocity_advection.o``
#: capture.  Used as a fallback when the submodule has no build dir
#: (a fresh clone needs no ICON configure pass to drive this test).
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
    """Return ``{defines, include_dirs}`` for ICON's velocity build.
    Prefers extracting from a real build (``$ICON_BUILD/Makefile`` ->
    ``make -n``) so per-config tweaks land; falls back to the
    release-frozen :data:`_ICON_DEFINES_FALLBACK` set when no build
    dir is present (the common case for a fresh submodule clone)."""
    if (_ICON_BUILD / "Makefile").is_file():
        return dace_fortran.extract_make_compile_args(makefile_dir=_ICON_BUILD, target=_VELOCITY_TARGET)
    return {"defines": list(_ICON_DEFINES_FALLBACK), "include_dirs": [_ICON_SRC / "src/include"]}


pytestmark = pytest.mark.skipif(not (_HAVE_FLANG and _HAVE_OPENMPI), reason="needs flang-new-21 + OpenMPI")

# ---------------------------------------------------------------------------
# Headline test: build the velocity SDFG from BOTH a stub source and
# ICON's REAL source, in one parametrization so a bridge-side
# regression on either route is caught.
# ---------------------------------------------------------------------------


def _build_sdfg_from_real_icon(tmp_path: Path):
    """Compose a translation unit for ICON's real
    ``mo_velocity_advection.f90`` via the codebase helpers and lower
    it to a DaCe SDFG."""
    args = _icon_compile_args()
    hlfir = dace_fortran.emit_hlfir_from_codebase(
        entry_source=_real_velocity_source().read_text(),
        out_path=tmp_path / "velocity.hlfir",
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
        return dace_fortran.build_sdfg_from_hlfir(hlfir, entry=_VELOCITY_ENTRY)
    finally:
        dace_fortran.clear_external_registry()


def _build_sdfg_from_stub(tmp_path: Path):
    """Build the SDFG directly from ``velocity_full.f90`` -- the
    self-contained stub kernel the bridge has been driving for a
    while.  No flang flag plumbing needed; the existing
    ``build_sdfg`` entry handles the closure-merge internally."""
    # Inline path so this file doesn't take a dependency on the
    # private test harness in ``tests/_util.py``.
    # ``build_sdfg`` returns the built SDFG directly.
    return dace_fortran.build_sdfg(
        _STUB_SOURCE.read_text(),
        out_dir=str(tmp_path / "sdfg"),
        entry="mo_velocity_advection::velocity_tendencies",
        name="velocity_stub",
    )


_PATHS = [
    pytest.param("stub", marks=[], id="velocity_full_stub"),
    pytest.param(
        "real_icon",
        marks=[
            pytest.mark.skipif(not _have_icon(),
                               reason="icon-model submodule not checked out + built; "
                               "run `git submodule update --init --recursive "
                               "tests/icon/full/icon-model` and configure a "
                               "stock CPU build before re-running"),
        ],
        id="icon_real_source",
    ),
]


@pytest.mark.parametrize("source", _PATHS)
def test_build_velocity_sdfg(tmp_path: Path, source: str):
    """Build the velocity_tendencies SDFG and check the SDFG names
    the procedure.  Parametrized over the two source paths so a
    regression on either route fails its own variant clearly."""
    if source == "stub":
        sdfg = _build_sdfg_from_stub(tmp_path)
    elif source == "real_icon":
        sdfg = _build_sdfg_from_real_icon(tmp_path)
    else:  # defensive
        raise AssertionError(f"unknown source variant {source!r}")

    assert sdfg is not None
    assert sdfg.name
    name_lc = sdfg.name.lower()
    assert "velocity" in name_lc, \
        f"SDFG name doesn't carry the expected entry: {sdfg.name!r}"
    # API-level structural validation: catches dangling memlets,
    # orphan connectors, missing access nodes, schedule mismatches
    # the bridge could in principle emit but mustn't.
    sdfg.validate()


# ---------------------------------------------------------------------------
# Helper-level test: assert ``emit_hlfir_from_codebase`` produces a
# non-trivial ``.hlfir`` for ICON's real velocity TU.  Kept as a
# separate test so its failure mode is distinguishable from the SDFG
# build's (the latter exercises the bridge end-to-end).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_icon(), reason="icon-model submodule not checked out")
def test_emit_hlfir_for_icon_velocity(tmp_path: Path):
    """``emit_hlfir_from_codebase`` produces an ``.hlfir`` flang
    actually wrote (sanity check before we lower it to SDFG)."""
    args = _icon_compile_args()
    out = dace_fortran.emit_hlfir_from_codebase(
        entry_source=_real_velocity_source().read_text(),
        out_path=tmp_path / "velocity.hlfir",
        search_dirs=_icon_search_dirs(),
        library_stubs=["mpi", "netcdf"],
        defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
        include_dirs=args["include_dirs"],
        cache_dir=_CACHE_DIR,
    )
    assert out.is_file()
    # The full closure lowering hits hundreds of MB -- a small file
    # would mean flang silently truncated.
    assert out.stat().st_size > 100 * 1024 * 1024
