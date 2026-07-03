"""Tier-3 prebuilt-HLFIR end-to-end across two mock projects built
with two different build systems, to confirm the helper is generic.

This shape -- the bridge as a clean consumer of whatever the
project's build system produces -- is the canonical path for
codebases too large or dep-tangled for the bridge to compile itself
(ICON / ECRAD / ...).  The tier-3 API is :func:`build_sdfg_from_hlfir`
(WIP, see README).

The user adds one DB-capture step to their normal build, then runs
the helper once:

    python -m dace_fortran.emit_hlfir <build>/compile_commands.json \\
        --out <build>/hlfir [--stub <stub.f90>]...

The helper reads the artefact for build order + ``-I`` / ``-D``
flags.  Two ways to get the ``compile_commands.json`` artefact are
exercised here:

* ``jacobi/`` -- **autotools** project (autoconf + automake), built
  the way ICON is (ICON itself is autoconf + a hand-written
  ``Makefile.in``; ``bear`` is build-system-agnostic so it captures
  either identically).  ``bear -- make`` writes the DB by
  intercepting compiler ``exec()`` calls.  4 ``.f90`` files, hard
  deps on real MPI + netCDF; the entry ``jacobi2d_update`` has an
  inlinable ``stencil_5pt`` helper and a sibling ``halo_exchange``
  that uses MPI -- the bridge lowers only the entry, MPI stays out
  of the SDFG.  Two flang stubs (``stubs/mpi_stub.f90``,
  ``stubs/netcdf_stub.f90``) stand in for the modules flang has no
  shipped ``.mod`` for.
* ``csr_spmv/`` -- **cmake** project, DB via
  ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``.  2 ``.f90`` files, no
  external deps, no stubs.  Entry ``csr_spmv`` has an inlinable
  ``dot_row`` helper.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_project

_HERE = Path(__file__).resolve().parent
_JACOBI_DIR = _HERE / "jacobi"
_CSR_DIR = _HERE / "csr_spmv"
_JACOBI_STUBS = [_JACOBI_DIR / "stubs" / "mpi_stub.f90", _JACOBI_DIR / "stubs" / "netcdf_stub.f90"]


def _have(*tools: str) -> bool:
    return all(shutil.which(t) is not None for t in tools)


def _has_netcdf_fortran() -> bool:
    pkg = shutil.which("pkg-config")
    return pkg is not None and \
        subprocess.run([pkg, "--exists", "netcdf-fortran"]).returncode == 0


def _assert_inlined(sdfg, helper: str):
    arr_names = " ".join(sdfg.arrays.keys()).lower()
    assert helper not in arr_names, (f"{helper} should have been inlined; appeared in SDFG arrays: "
                                     f"{sorted(sdfg.arrays.keys())}")


@pytest.mark.skipif(
    not (_have("flang-new-21", "bear", "autoreconf", "automake", "mpif90", "nf-config") and _has_netcdf_fortran()),
    reason="flang-new-21 / bear / autotools / MPI / netcdf-fortran missing",
)
def test_jacobi_autotools_bear(tmp_path: Path):
    """Autotools + ``bear -- make`` -> compile_commands.json (the ICON
    build shape).  4 files, MPI + netCDF, two flang stubs.  Drives the
    one-call ``build_sdfg_from_project`` tier-3 entry point."""
    build = tmp_path / "build"
    shutil.copytree(_JACOBI_DIR, build)

    subprocess.check_call(["autoreconf", "--install"], cwd=build)
    subprocess.check_call(["./configure"], cwd=build)
    # Serial make so the Fortran module .mod files land in USE-dep
    # order; bear records each compiler exec, writing the DB even on
    # a partial build.
    subprocess.check_call(["bear", "--", "make"], cwd=build)

    sdfg = build_sdfg_from_project(build / "compile_commands.json",
                                   entry="mod_jacobi::jacobi2d_update",
                                   stubs=_JACOBI_STUBS,
                                   out_dir=tmp_path / "hlfir")
    sdfg.validate()
    _assert_inlined(sdfg, "stencil_5pt")
    for node, _ in sdfg.all_nodes_recursive():
        label = (getattr(node, "label", "") or "").lower()
        assert "mpi_" not in label, f"MPI reference leaked into SDFG: {node}"


@pytest.mark.skipif(
    not _have("cmake", "flang-new-21"),
    reason="cmake / flang-new-21 missing",
)
def test_csr_spmv_cmake(tmp_path: Path):
    """cmake ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`` -> compile_commands.json.
    2 files, no external deps, no stubs -- a structurally-different
    project with no per-project plumbing."""
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cmake", "-S", str(_CSR_DIR), "-B", str(build), "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"])
    subprocess.check_call(["cmake", "--build", str(build), "--target", "csr_demo"])

    sdfg = build_sdfg_from_project(build / "compile_commands.json",
                                   entry="mod_csr::csr_spmv",
                                   out_dir=tmp_path / "hlfir")
    sdfg.validate()
    _assert_inlined(sdfg, "dot_row")
