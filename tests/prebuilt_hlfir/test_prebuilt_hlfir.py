"""Tier-3 prebuilt-HLFIR end-to-end across two mock projects.

This shape -- the bridge as a clean consumer of whatever the
project's build system produces -- is the canonical path for
codebases too large or dep-tangled for the bridge to compile itself
(ICON / ECRAD / ...).  The tier-3 API is :func:`build_sdfg_from_hlfir`
(WIP, see README).

What the user does (one extra cmake flag + one extra command):

1. add ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`` to the cmake config
2. after the regular build, run::

      python -m dace_fortran.emit_hlfir \\
          <build_dir>/compile_commands.json \\
          --out <build_dir>/hlfir \\
          [--stub <stub.f90>]...

The helper reads the artefact for build order + ``-I`` / ``-D``
flags, so it works on any project that emits a
``compile_commands.json`` (cmake / ninja / fpm).  ``--stub`` injects
flang-buildable stand-ins for modules flang has no shipped ``.mod``
for (``mpi`` / ``netcdf`` / ``hdf5`` / ...).

Two projects are exercised here to confirm the helper is generic:

* ``jacobi/`` -- 4 ``.f90`` files, hard deps on MPI + netCDF, two
  stubs.  Entry ``jacobi2d_update`` has an inlinable
  ``stencil_5pt`` helper and a sibling ``halo_exchange`` that uses
  MPI -- the bridge lowers only the entry, MPI references stay out
  of the SDFG.
* ``csr_spmv/`` -- 2 ``.f90`` files, no external deps, no stubs.
  Entry ``csr_spmv`` has an inlinable ``dot_row`` helper.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_hlfir
from dace_fortran.emit_hlfir import emit as emit_hlfir

_HERE = Path(__file__).resolve().parent
_JACOBI_DIR = _HERE / "jacobi"
_CSR_DIR = _HERE / "csr_spmv"
_JACOBI_STUBS = [_JACOBI_DIR / "mpi_stub.f90", _JACOBI_DIR / "netcdf_stub.f90"]


def _has_cmake_and_flang() -> bool:
    return shutil.which("cmake") is not None and have_flang()


def _has_mpi_netcdf() -> bool:
    """jacobi needs system MPI + netcdf-fortran; csr_spmv does not."""
    if shutil.which("mpif90") is None and shutil.which("mpifort") is None:
        return False
    pkg = shutil.which("pkg-config")
    if pkg is None:
        return False
    return subprocess.run([pkg, "--exists", "netcdf-fortran"]).returncode == 0


def _build_and_emit(src_dir: Path, build_dir: Path, hlfir_dir: Path,
                    target: str, stubs: Sequence[Path] = ()):
    """Run the canonical user flow for one project: configure cmake
    with ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``, build the project's
    own executable target, then emit HLFIR from the
    ``compile_commands.json`` artefact."""
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "cmake", "-S", str(src_dir), "-B", str(build_dir),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ])
    subprocess.check_call(["cmake", "--build", str(build_dir), "--target", target])
    emit_hlfir(compile_commands=build_dir / "compile_commands.json",
               stubs=list(stubs),
               out_dir=hlfir_dir)


@pytest.mark.skipif(
    not (_has_cmake_and_flang() and _has_mpi_netcdf()),
    reason="cmake / flang-new-21 / MPI / netcdf-fortran missing",
)
def test_jacobi_update_from_prebuilt_hlfir(tmp_path: Path):
    """Project 1 -- 4 files, MPI + netCDF, stubs needed."""
    _build_and_emit(_JACOBI_DIR, tmp_path / "build", tmp_path / "hlfir",
                    target="jacobi", stubs=_JACOBI_STUBS)

    entry = "_QMmod_jacobiPjacobi2d_update"
    sdfg = build_sdfg_from_hlfir(tmp_path / "hlfir", entry=entry)
    sdfg.validate()

    arr_names = " ".join(sdfg.arrays.keys()).lower()
    assert "stencil_5pt" not in arr_names, (
        f"stencil_5pt should have been inlined; appeared in SDFG arrays: "
        f"{sorted(sdfg.arrays.keys())}")
    for node, _ in sdfg.all_nodes_recursive():
        label = (getattr(node, "label", "") or "").lower()
        assert "mpi_" not in label, f"MPI reference leaked into SDFG: {node}"


@pytest.mark.skipif(
    not _has_cmake_and_flang(),
    reason="cmake / flang-new-21 missing",
)
def test_csr_spmv_from_prebuilt_hlfir(tmp_path: Path):
    """Project 2 -- 2 files, no external deps, no stubs.  Confirms
    the helper handles a structurally-different project (different
    file count, no MPI/netCDF, no stubs) with no per-project plumbing."""
    _build_and_emit(_CSR_DIR, tmp_path / "build", tmp_path / "hlfir",
                    target="csr_demo")

    entry = "_QMmod_csrPcsr_spmv"
    sdfg = build_sdfg_from_hlfir(tmp_path / "hlfir", entry=entry)
    sdfg.validate()

    arr_names = " ".join(sdfg.arrays.keys()).lower()
    assert "dot_row" not in arr_names, (
        f"dot_row should have been inlined; appeared in SDFG arrays: "
        f"{sorted(sdfg.arrays.keys())}")
