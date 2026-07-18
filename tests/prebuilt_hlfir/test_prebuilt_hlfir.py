"""Tier-3 prebuilt-HLFIR end-to-end across two mock projects built with two different build systems, confirming :func:`build_sdfg_from_hlfir` is generic -- the canonical path for codebases too large/dep-tangled to compile directly (ICON/ECRAD/...).

Usage: ``python -m dace_fortran.emit_hlfir <build>/compile_commands.json --out <build>/hlfir [--stub <stub.f90>]...`` reads the DB for build order + -I/-D flags.

* jacobi/ -- autotools (like ICON), DB via ``bear -- make``.  4 files, real MPI+netCDF deps; entry jacobi2d_update inlines stencil_5pt, sibling halo_exchange uses MPI (stays out of the SDFG).  Two flang stubs stand in for modules flang ships no .mod for.
* csr_spmv/ -- cmake, DB via -DCMAKE_EXPORT_COMPILE_COMMANDS=ON.  2 files, no deps, no stubs; entry csr_spmv inlines dot_row.
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
    """Autotools + bear -- make -> compile_commands.json (ICON build shape); drives the one-call build_sdfg_from_project tier-3 entry point."""
    build = tmp_path / "build"
    shutil.copytree(_JACOBI_DIR, build)

    subprocess.check_call(["autoreconf", "--install"], cwd=build)
    subprocess.check_call(["./configure"], cwd=build)
    # serial make so .mod files land in USE-dep order; bear records each compiler exec, writing the DB even on a partial build.
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
    """cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -> compile_commands.json; structurally different project, no per-project plumbing needed."""
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cmake", "-S", str(_CSR_DIR), "-B", str(build), "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"])
    subprocess.check_call(["cmake", "--build", str(build), "--target", "csr_demo"])

    sdfg = build_sdfg_from_project(build / "compile_commands.json",
                                   entry="mod_csr::csr_spmv",
                                   out_dir=tmp_path / "hlfir")
    sdfg.validate()
    _assert_inlined(sdfg, "dot_row")
