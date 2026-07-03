"""The full ICON ``solve_nh`` dycore lowers to an SDFG and its Fortran binding
COMPILES -- the prerequisite for the standalone e2e and the ICON binding-swap
integration.

Two blockers had to clear for the ~700-argument ``bind(c)`` binding to be valid
Fortran:

  * **Synthetic ``__``-names.** ``ExpandVectorSubscriptGather`` materialises
    gather temps named ``__assoc_scalar_N``; a couple reach the SDFG signature as
    free symbols.  ``integer(c_int), value :: __assoc_scalar_198`` is a syntax
    error (a Fortran dummy can't start with ``_``).  ``extractName`` now prefixes
    a letter onto ``__``-names so every emitted identifier is valid in both the
    C++ codegen and the Fortran binding.

  * **Dual-typed MPI.** The inlined ``mo_mpi`` wrappers call one ``mpi_recv`` /
    ``mpi_isend`` / ... with both real*8 (``p_*_dp``) and real*4 (``p_*_sp``)
    buffers.  gfortran rejects that without an interface (``-fallow`` -- which we
    do NOT use).  The sound fix is a ``TYPE(*)`` assumed-type interface: prepend
    the ``_MPI_STUB`` ``module mpi`` and give ``mo_mpi`` a ``use mpi`` so one
    interface accepts any buffer type.  (The committed single-TU predates the
    ``use mpi`` re-emission, so we inject it here at build time.)

Marked ``long``: it builds the 3166-LoC dycore to an SDFG (minutes).
"""
import shutil
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang
from icon._halo_modes import HALO_INLINED_EXTRA_SOURCES

from dace_fortran.bindings import build_fortran_library

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_HERE = Path(__file__).resolve().parent
_TU = _HERE / "solve_nonhydro_inlined_single_tu.f90"
_ENTRY = "mo_solve_nonhydro::solve_nh"


def test_solve_nh_binding_compiles(tmp_path: Path):
    """solve_nh -> SDFG -> generated bind(c) binding compiles with gfortran."""
    sdfg = build_sdfg(_TU.read_text(), tmp_path / "sdfg", name="solve_nh", entry=_ENTRY).build()
    sdfg.name = "solve_nh"

    # No leaked ``__``-prefixed args (each would be an invalid Fortran dummy).
    invalid = [a for a in sdfg.arglist() if a.startswith("_")]
    assert not invalid, f"SDFG signature carries Fortran-invalid names: {invalid[:5]}"

    # Prepend the TYPE(*) MPI stub + give the inlined ``mo_mpi`` a ``use mpi`` so
    # its dual-typed real*8/real*4 point-to-point calls resolve through one
    # assumed-type interface (no -fallow).
    tu_src = _TU.read_text()
    assert "MODULE mo_mpi\n" in tu_src, "inlined mo_mpi module anchor missing"
    use_mpi_tu = tmp_path / "solve_nh_usempi.f90"
    use_mpi_tu.write_text(tu_src.replace("MODULE mo_mpi\n", "MODULE mo_mpi\n  use mpi\n", 1))
    stub_name, stub_content = next(iter(HALO_INLINED_EXTRA_SOURCES.items()))
    stub = tmp_path / stub_name
    stub.write_text(stub_content)

    # ``prelude_sources`` compile left-to-right before the binding, so the stub
    # (module mpi) comes before the TU that ``use``s it.
    lib = build_fortran_library(sdfg,
                                out_dir=str(tmp_path / "lib"),
                                prelude_sources=[stub, use_mpi_tu],
                                bind_c_shim=True)
    assert Path(lib.so_path).exists(), "binding .so was not produced"
