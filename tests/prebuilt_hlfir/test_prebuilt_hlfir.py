"""Tier-3 prebuilt-HLFIR end-to-end: a small distributed-Jacobi
project (4 ``.f90`` files, hard deps on MPI + netCDF) is built by
its own cmake, then HLFIR-emitted alongside, and the bridge consumes
only the relevant ``.hlfir`` to lower ``jacobi2d_update`` into an
SDFG without touching MPI or netCDF.

This shape -- the bridge as a clean consumer of whatever the
project's build system produces -- is the canonical path for
codebases too large or dep-tangled for the bridge to compile itself
(ICON / ECRAD / ...).  The tier-3 API is :func:`build_sdfg_from_hlfir`
(WIP, see README).

The test:

1. cmake configures + builds the real executable -- this is what
   verifies the codebase genuinely compiles + links against the
   system MPI / netCDF.  Test skips if any of cmake / flang-new-21 /
   mpi / netcdf-fortran are missing on the host.
2. cmake runs the ``emit_hlfir`` custom target, which drives
   ``flang-new-21 -fc1 -emit-hlfir`` over the same sources (against
   flang-compatible ``mpi`` / ``netcdf`` stubs the project's
   ``CMakeLists.txt`` builds first -- the system Fortran modules
   are in gfortran's binary format and flang cannot consume them).
3. ``build_sdfg_from_hlfir(<build_dir>, entry=...)`` walks the
   build dir, finds the ``.hlfir`` that defines the entry, and
   lowers it.

The same ``.hlfir`` ``mod_jacobi`` also contains ``halo_exchange``
(uses MPI) and ``stencil_5pt`` (the inlinable helper).  The bridge
lowers only ``jacobi2d_update``; the test asserts:

* the produced SDFG validates,
* the helper ``stencil_5pt`` was inlined (the SDFG arglist does
  not surface any ``stencil_5pt`` artefact -- the body is fused
  into the jacobi loop),
* the MPI-using sibling ``halo_exchange`` is absent (no
  ``MPI_Sendrecv`` reference anywhere in the SDFG arrays /
  tasklets).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from _util import have_flang
from dace_fortran import build_sdfg_from_hlfir

_HERE = Path(__file__).resolve().parent


def _tools_available() -> bool:
    """Hard-skip when any of the prerequisites is missing.  The test
    requires cmake, flang-new-21, a system MPI compiler wrapper, and
    netcdf-fortran (which the cmake ``find_package`` resolves via
    pkg-config) -- not point reporting failures when the env is
    incomplete."""
    if shutil.which("cmake") is None:
        return False
    if not have_flang():
        return False
    if shutil.which("mpif90") is None and shutil.which("mpifort") is None:
        return False
    pkg = shutil.which("pkg-config")
    if pkg is None:
        return False
    return subprocess.run([pkg, "--exists", "netcdf-fortran"]).returncode == 0


pytestmark = pytest.mark.skipif(
    not _tools_available(),
    reason="cmake / flang-new-21 / MPI / netcdf-fortran missing",
)


def _cmake_build(build_dir: Path):
    """Configure + build the real exe + the ``emit_hlfir`` target."""
    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["cmake", "-S", str(_HERE), "-B", str(build_dir)])
    subprocess.check_call(["cmake", "--build", str(build_dir), "--target", "jacobi"])
    subprocess.check_call(["cmake", "--build", str(build_dir), "--target", "emit_hlfir"])


def test_jacobi_update_from_prebuilt_hlfir(tmp_path: Path):
    build_dir = tmp_path / "build"
    _cmake_build(build_dir)

    # The bridge walks ``build_dir`` for the .hlfir whose MLIR holds
    # ``func.func @_QMmod_jacobiPjacobi2d_update``.  ``mod_jacobi.hlfir``
    # also contains ``halo_exchange`` (MPI-using) and ``stencil_5pt``
    # (inlinable helper) -- the bridge lowers only the entry.
    entry = "_QMmod_jacobiPjacobi2d_update"
    sdfg = build_sdfg_from_hlfir(build_dir, entry=entry)
    sdfg.validate()

    # Inlining: ``stencil_5pt`` should have been fused into the
    # jacobi loop, not left as a separate access / tasklet ``.label``.
    arr_names = " ".join(sdfg.arrays.keys()).lower()
    assert "stencil_5pt" not in arr_names, (
        f"stencil_5pt should have been inlined; appeared in SDFG arrays: "
        f"{sorted(sdfg.arrays.keys())}")
    # No MPI calls survive (the sibling halo_exchange was not lowered).
    for node, _ in sdfg.all_nodes_recursive():
        label = (getattr(node, "label", "") or "").lower()
        assert "mpi_" not in label, f"MPI reference leaked into SDFG: {node}"
