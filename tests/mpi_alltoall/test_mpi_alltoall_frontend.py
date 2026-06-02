"""Frontend-recognition test for MPI_Alltoall.

Drives :file:`mpi_alltoall_probe.f90`'s ``run_alltoall`` entry through
the bridge and asserts the resulting SDFG contains a single
:class:`dace.libraries.mpi.nodes.alltoall.Alltoall` lib node.  Numerical
correctness of the lib node itself is covered separately in d-face's
MPI test suite.
"""
from pathlib import Path

import pytest

import dace_fortran
from _util import have_flang

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "mpi_alltoall_probe.f90"

pytestmark = pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH")


def test_mpi_alltoall_recognised(tmp_path):
    src = _SRC.read_text()
    sdfg = dace_fortran.build_sdfg(src, out_dir=str(tmp_path / "sdfg"),
                                   entry="mpi_alltoall_probe::run_alltoall",
                                   name="run_alltoall")
    sdfg.validate()
    classes = {type(n).__name__ for s in sdfg.states() for n in s.nodes()}
    assert "Alltoall" in classes, \
        f"expected an Alltoall lib node, got {sorted(classes)!r}"
