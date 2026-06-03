"""Unit tests for :mod:`dace_fortran.flang_codebase`.

The headline test exercises ``extract_make_compile_args`` +
``prepare_flang_translation_unit`` end-to-end against ICON's real
``mo_velocity_advection.f90``: configure-time ``-D`` defines come
from ``make -n``, source merging from ICON's own ``src/`` +
``externals/*/src``, mpi/netcdf stubs from
:data:`dace_fortran.LIBRARY_STUBS`.  Asserts the resulting TU lowers
to HLFIR cleanly under flang-21.  Skipped if no ICON checkout, no
flang, or no OpenMPI.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

import dace_fortran
from dace_fortran.flang_codebase import (
    FLANG_BUG_PATCHES,
    LIBRARY_STUBS,
    extract_make_compile_args,
    find_openmpi_include,
    mpi_stub_source,
    patch_mpi_sizeof,
    prepare_flang_translation_unit,
)

_ICON_SRC = Path("/home/primrose/Work/icon-model-public")
_ICON_BUILD = _ICON_SRC / "build" / "stock_cpu"
_VELOCITY_BAK = _ICON_SRC / "src" / "atm_dyn_iconam" / "mo_velocity_advection.f90.bak"

_CACHE_DIR = Path("/home/primrose/.cache/dace-fortran")

_HAVE_FLANG = shutil.which("flang-new-21") is not None
_HAVE_ICON = _ICON_BUILD.is_dir() and _VELOCITY_BAK.is_file()
_HAVE_OPENMPI = find_openmpi_include() is not None

# ---------------------------------------------------------------------------
# Stand-alone tests for the individual helpers.
# ---------------------------------------------------------------------------


def test_library_stubs_registry_keys():
    """Every built-in library stub name should resolve to a
    :class:`LibraryStub` with ``source`` and ``flags`` callables."""
    assert set(LIBRARY_STUBS) >= {"mpi", "netcdf"}
    for stub in LIBRARY_STUBS.values():
        assert callable(stub.source)
        assert callable(stub.flags)


def test_flang_bug_patches_registry_keys():
    """The patch registry exposes at least the safe default."""
    assert "mpi_sizeof" in FLANG_BUG_PATCHES


@pytest.mark.skipif(not _HAVE_OPENMPI, reason="OpenMPI include not on this system")
def test_mpi_stub_source_and_flags():
    """The MPI stub source is plain Fortran and the include flag
    points at a directory containing ``mpif-config.h``."""
    src = mpi_stub_source()
    assert "MODULE mpi" in src
    assert "INCLUDE 'mpif-config.h'" in src
    inc_flag = LIBRARY_STUBS["mpi"].flags()[0]
    assert inc_flag.startswith("-I")
    assert (Path(inc_flag[2:]) / "mpif-config.h").is_file()


def test_patch_mpi_sizeof_substitutes_static_byte_count():
    """``CALL MPI_SIZEOF(x, sz, err)`` becomes a static assignment.
    Pure text transform -- no upstream dependencies."""
    before = ("    CALL MPI_SIZEOF(rrg, p_real_byte, p_error)\n"
              "    CALL MPI_SIZEOF(ii4, p_int_i4_byte, p_error)\n")
    after = patch_mpi_sizeof(before)
    assert "CALL MPI_SIZEOF" not in after
    # rrg -> double (8); ii4 -> single (4)
    assert "p_real_byte = 8" in after
    assert "p_int_i4_byte = 4" in after


def test_patch_mpi_sizeof_noop_on_unrelated_source():
    """The patch leaves any non-matching code untouched."""
    src = "    CALL some_other_proc(a, b, c)\n    x = y + 1\n"
    assert patch_mpi_sizeof(src) == src


def test_prepare_translation_unit_rejects_unknown_stub():
    """Asking for an unregistered library stub is a programming error."""
    with pytest.raises(KeyError, match="not_a_real_stub"):
        prepare_flang_translation_unit("module m; end module", library_stubs=["not_a_real_stub"])


def test_prepare_translation_unit_rejects_unknown_patch():
    """Asking for an unregistered patch is also a programming error."""
    with pytest.raises(KeyError, match="bogus_patch"):
        prepare_flang_translation_unit("module m; end module", patches=["bogus_patch"])


# ---------------------------------------------------------------------------
# End-to-end ICON integration -- the real proof.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_ICON, reason="ICON checkout not present at the standard path")
def test_extract_make_compile_args_for_icon_velocity():
    """``make -n`` against an ICON build dir yields the same ``-D`` /
    ``-I`` set that ICON's own gfortran invocation uses for the
    velocity object file."""
    args = extract_make_compile_args(makefile_dir=_ICON_BUILD, target="src/atm_dyn_iconam/mo_velocity_advection.o")
    # ICON's recognisable feature-disable defines.
    assert "__ICON__" in args["defines"]
    assert "__NO_JSBACH__" in args["defines"]
    assert "__NO_RAGNAROK__" in args["defines"]
    # The expected source file.
    assert args["source"].name == "mo_velocity_advection.f90"
    # The build's own include layout includes ``src/include``.
    src_include = _ICON_SRC / "src/include"
    assert src_include in args["include_dirs"]


@pytest.mark.skipif(not (_HAVE_ICON and _HAVE_FLANG and _HAVE_OPENMPI),
                    reason="needs ICON checkout + flang-new-21 + OpenMPI")
def test_prepare_translation_unit_flang_clean_on_icon_velocity(tmp_path: Path):
    """Compose a TU for ICON's real ``mo_velocity_advection.f90`` via
    the helpers and verify flang-21 lowers it to HLFIR with zero
    errors.  This is the load-bearing assertion: it pins the entire
    recipe (merge + stubs + patches + extracted defines) as a
    regression gate."""
    args = extract_make_compile_args(makefile_dir=_ICON_BUILD, target="src/atm_dyn_iconam/mo_velocity_advection.o")
    entry = _VELOCITY_BAK.read_text()
    tu, flang_flags = prepare_flang_translation_unit(
        entry,
        search_dirs=[
            _ICON_SRC / "src",
            _ICON_SRC / "externals/fortran-support/src",
            _ICON_SRC / "externals/mtime/src",
            _ICON_SRC / "externals/iconmath/src",
            _ICON_SRC / "externals/cdi/src",
            _ICON_SRC / "externals/memman/src/bindings/fortran",
            _ICON_SRC / "support",
        ],
        library_stubs=["mpi", "netcdf"],
        patches=["mpi_sizeof"],
        defines=args["defines"] + ["NO_MPI_CHOICE_ARG"],
        include_dirs=[_ICON_SRC / "src/include"],
        cache_dir=_CACHE_DIR,
    )
    tu_path = tmp_path / "velocity_merged.F90"
    tu_path.write_text(tu)
    hlfir_path = tmp_path / "velocity.hlfir"

    # cwd=tmp_path keeps flang from picking up any stale ``.mod``
    # in the repo root (a prior flang run can leak a flang-format
    # ``iso_c_binding.mod`` there which flang then rejects as stale).
    result = subprocess.run(
        [
            "flang-new-21",
            "-fc1",
            "-cpp",
            "-U_OPENMP",
            "-U_OPENACC",
            *flang_flags,
            "-emit-hlfir",
            str(tu_path),
            "-o",
            str(hlfir_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    err_lines = [ln for ln in result.stderr.splitlines() if "error:" in ln]
    assert not err_lines, (f"flang reported {len(err_lines)} error(s); "
                           f"first 5:\n" + "\n".join(err_lines[:5]))
    assert hlfir_path.is_file()
    # 753 MB is the empirical size we got; >100 MB just confirms the
    # full closure was lowered (no premature truncation).
    assert hlfir_path.stat().st_size > 100 * 1024 * 1024


def test_lazy_import_surface():
    """The headline helpers are re-exported from the package root for
    the documented ``import dace_fortran; dace_fortran.foo`` use."""
    assert callable(dace_fortran.prepare_flang_translation_unit)
    assert callable(dace_fortran.extract_make_compile_args)
    assert callable(dace_fortran.vendor_netcdf_fortran)
    assert callable(dace_fortran.mpi_stub_source)
    assert "mpi" in dace_fortran.LIBRARY_STUBS
    assert "mpi_sizeof" in dace_fortran.FLANG_BUG_PATCHES
