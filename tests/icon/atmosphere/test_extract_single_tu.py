"""``input -> single TU`` extraction gate for the ICON atmosphere dynamical
core (``mo_solve_nonhydro::solve_nh``).

Unlike the ocean harness (which black-boxes the halo), this extraction *inlines*
``sync_patch_array`` / ``exchange_data`` and lets the inliner's default
monomorphisation pass devirtualise the (single-arm, post-cpp) ``t_comm_pattern``
dispatch: ``t_comm_pattern_yaxt`` is cpp'd out, leaving ``t_comm_pattern_orig``,
which the pass retypes so ``p_pat%exchange_data_*`` becomes a static call the
inliner inlines.  The pack/gather loops land inline and only the MPI
point-to-point (``p_isend`` / ``p_irecv`` / ``p_wait`` / ``p_send`` / ``p_recv``)
remains external -- "only MPI calls remain" (mapped to ``dace.libraries.mpi``
libnodes when the TU is lowered to an SDFG; see
``tests/sync_devirt_mpi_libnode_test.py``).

The test regenerates the single TU via the fparser inliner (merge closure ->
cpp -> force-include the comm-pattern arm -> monomorphise -> prune -> ``gfortran
-fsyntax-only``) and asserts it (a) compiles and (b) is byte-identical to the
committed artifact -- so the SDFG-lowering stage never silently drifts.

Slow (the merged closure is ~140k lines) and memory-heavy, so ``long`` and
serialised onto one xdist worker in a memory-capped subprocess.  Gated on
flang-new-21 + OpenMPI + the icon-model submodule.
"""
from pathlib import Path

import pytest

from icon.atmosphere._atmo_harness import (HAVE_FLANG, HAVE_OPENMPI, KERNELS, SINGLE_TU_ARTIFACTS, extract_single_tu,
                                           have_icon_atmo)

_HERE = Path(__file__).resolve().parent
_SOURCE = {k[0]: k[1] for k in KERNELS}

#: One case per ``(key, halo_mode, filename, entry)`` -- the solver in BOTH halo
#: modes (inlined = MPI-only, external = halo black-boxed / callback boundary).
_CASES = SINGLE_TU_ARTIFACTS

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not (HAVE_FLANG and HAVE_OPENMPI), reason="needs flang-new-21 + OpenMPI"),
    pytest.mark.skipif(not have_icon_atmo(),
                       reason="icon-model atmosphere source not checked out; run "
                       "`git submodule update --init --recursive tests/icon/full/icon-model`"),
]


@pytest.mark.xdist_group("atmo_fparser")
@pytest.mark.parametrize("key,halo_mode,filename,entry", _CASES, ids=[f"{c[0]}-{c[1]}" for c in _CASES])
def test_extract_compiles_and_matches_committed(tmp_path, key, halo_mode, filename, entry):
    """Extract ``solve_nonhydro`` in one halo mode into a compiling single TU and
    check it against the committed artifact -- both the inlined (MPI-only) and
    external (callback boundary) halo modes must always be correct."""
    res = extract_single_tu(_SOURCE[key], entry, tmp_path / f"{key}_{halo_mode}", halo_mode=halo_mode)
    assert res["passed"], \
        f"{key}[{halo_mode}]: extraction did not produce a compiling single TU.\n{res['output'][-4000:]}"
    # the ~140k-line closure must prune to the kernel by orders of magnitude.
    assert res["tu_lines"] is not None and res["tu_lines"] < 50_000, \
        f"{key}[{halo_mode}]: pruned TU is {res['tu_lines']} lines -- pruning did not converge"
    committed = _HERE / filename
    assert committed.is_file(), \
        f"{key}[{halo_mode}]: no committed artifact {committed.name}; save the extracted TU into this folder"
    assert Path(res["tu_path"]).read_text() == committed.read_text(), \
        f"{key}[{halo_mode}]: extracted TU drifted from committed {committed.name}; regenerate it"
