"""Build the velocity_tendencies SDFG from TWO source paths, parametrized so a
regression on either route surfaces immediately: (1) the self-contained
``velocity_full.f90`` stub (backwards-compat pin), and (2) ICON's REAL
``mo_velocity_advection.f90`` via the ``icon-model`` submodule, composed through
:mod:`dace_fortran.flang_codebase` (merge + library stubs + ICON's ``-D`` defines),
emitted to HLFIR and lowered to an SDFG.

Skipped without the icon-model submodule or flang-new-21/OpenMPI.
"""
import shutil
from pathlib import Path

import pytest

import dace_fortran
from dace_fortran.flang_codebase import find_openmpi_include

_HERE = Path(__file__).resolve().parent
_STUB_SOURCE = _HERE / "velocity_full.f90"

#: ICON checkout pinned by the ``icon-model`` submodule; override via ``ICON_SRC``.
import os

_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE / "icon-model")))

#: ICON's build dir; supplies -D/-I for the velocity object. Override via ``ICON_BUILD``.
_ICON_BUILD = Path(os.environ.get("ICON_BUILD", str(_ICON_SRC / "build" / "stock_cpu")))

_VELOCITY_REAL = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_velocity_advection.f90"
# Prefer the .bak pristine backup when present so the test is stable against live-file patching.
_VELOCITY_REAL_BAK = _VELOCITY_REAL.with_suffix(".f90.bak")

_CACHE_DIR = Path(os.environ.get("DACE_FORTRAN_CACHE", str(Path.home() / ".cache" / "dace-fortran")))

_VELOCITY_TARGET = "src/atm_dyn_iconam/mo_velocity_advection.o"
_VELOCITY_ENTRY = "mo_velocity_advection::velocity_tendencies"

_HAVE_FLANG = shutil.which("flang-new-21") is not None
_HAVE_OPENMPI = find_openmpi_include() is not None


def _real_velocity_source() -> Path:
    """Pristine ICON velocity source -- prefers the ``.bak`` so a patched live file doesn't perturb the test."""
    return _VELOCITY_REAL_BAK if _VELOCITY_REAL_BAK.is_file() else _VELOCITY_REAL


def _have_icon() -> bool:
    """Submodule checked out (build dir optional -- falls back to :data:`_ICON_DEFINES_FALLBACK`)."""
    return _real_velocity_source().is_file()


def _icon_search_dirs() -> list:
    """USE-graph closure of ``mo_velocity_advection``: ICON's ``src/`` plus the
    external trees it bundles. Bisected to exactly this set."""
    return [
        _ICON_SRC / "src",
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: Utility procedures (error reporting, timer hooks) velocity_tendencies calls
#: structurally but whose bodies the bridge can't lower (fir.iterate_while LEN_TRIM
#: scans, CLASS(*) polymorphism) -- stripped before hlfir-inline-all.
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

#: ICON's CPU build defines for icon-2026.04-public (from a ``make -n`` capture);
#: fallback so a fresh submodule clone needs no ICON configure pass.
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
    """``{defines, include_dirs}`` for ICON's velocity build -- extracted from a real
    build via ``make -n`` when present, else :data:`_ICON_DEFINES_FALLBACK`."""
    if (_ICON_BUILD / "Makefile").is_file():
        return dace_fortran.extract_make_compile_args(makefile_dir=_ICON_BUILD, target=_VELOCITY_TARGET)
    return {"defines": list(_ICON_DEFINES_FALLBACK), "include_dirs": [_ICON_SRC / "src/include"]}


# Reads ICON's real source via the icon-model submodule (heavy CI lane only) -> long.
# The self-contained velocity_full.f90 e2e tests stay in the fast lane.
pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not (_HAVE_FLANG and _HAVE_OPENMPI), reason="needs flang-new-21 + OpenMPI"),
]

# ---------------------------------------------------------------------------
# Headline test: build from BOTH the stub and ICON's real source in one
# parametrization so a regression on either route is caught.
# ---------------------------------------------------------------------------


def _build_sdfg_from_real_icon(tmp_path: Path):
    """Compose a TU for ICON's real ``mo_velocity_advection.f90`` and lower it to an SDFG."""
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
    # Ignore-list drops calls to the infrastructure stubs so their unlowerable bodies never reach the bridge.
    dace_fortran.apply_external_functions(do_not_emit=_ICON_EXTERNAL_STUBS)
    try:
        return dace_fortran.build_sdfg_from_hlfir(hlfir, entry=_VELOCITY_ENTRY)
    finally:
        dace_fortran.clear_external_registry()


def _build_sdfg_from_stub(tmp_path: Path):
    """Build the SDFG directly from ``velocity_full.f90`` -- no flang flag plumbing
    needed; ``build_sdfg`` handles the closure-merge internally."""
    # avoids a dependency on the private harness in tests/_util.py
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
    """Build velocity_tendencies and check the SDFG names the procedure;
    parametrized over both source paths so either route's regression is isolated."""
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
    # Structural validation: dangling memlets, orphan connectors, missing access nodes, schedule mismatches.
    sdfg.validate()
    # Load-bearing: an orphaned view_alias passes validate() but raises KeyError in
    # framecode's get_view_edge at compile -- only compile() catches that class.
    sdfg.compile()


# ---------------------------------------------------------------------------
# Helper-level test: emit_hlfir_from_codebase produces a non-trivial .hlfir for
# ICON's real TU -- separate from the SDFG build so failure modes are distinguishable.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_icon(), reason="icon-model submodule not checked out")
def test_emit_hlfir_for_icon_velocity(tmp_path: Path):
    """``emit_hlfir_from_codebase`` produces an ``.hlfir`` flang actually wrote (sanity check before SDFG lowering)."""
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
    # Full closure lowering hits hundreds of MB; a small file means flang silently truncated.
    assert out.stat().st_size > 100 * 1024 * 1024
