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
    from dace.libraries.mpi.nodes.alltoall import Alltoall

    src = _SRC.read_text()
    sdfg = dace_fortran.build_sdfg(src,
                                   out_dir=str(tmp_path / "sdfg"),
                                   entry="run_alltoall_mod::run_alltoall",
                                   name="run_alltoall")
    sdfg.validate()
    alltoall = [n for s in sdfg.states() for n in s.nodes() if isinstance(n, Alltoall)]
    assert len(alltoall) == 1, \
        f"expected one Alltoall lib node, got {[type(n).__name__ for s in sdfg.states() for n in s.nodes()]!r}"

    # The probe passes ``MPI_COMM_WORLD`` (a local ``parameter`` flang lowers to
    # a synthetic scalar the bridge treats as a runtime communicator), so the
    # Alltoall must thread it via a ``_comm`` connector -- exactly like the other
    # collectives.  Previously the Alltoall path dropped the communicator
    # entirely and always ran on ``MPI_COMM_WORLD`` regardless of the Fortran
    # ``comm`` argument.
    assert set(alltoall[0].in_connectors) == {"_inbuffer", "_comm"}, \
        f"Alltoall must thread the user communicator, got {sorted(alltoall[0].in_connectors)!r}"
