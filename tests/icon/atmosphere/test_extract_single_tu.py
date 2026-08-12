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
import re
from pathlib import Path

import pytest

from icon.atmosphere._atmo_harness import (HAVE_FLANG, HAVE_OPENMPI, KERNELS, SINGLE_TU_ARTIFACTS, extract_single_tu,
                                           have_icon_atmo)

_HERE = Path(__file__).resolve().parent
_SOURCE = {k[0]: k[1] for k in KERNELS}

#: One case per ``(key, halo_mode, filename, entry, loop_exchange)`` -- the solver in BOTH halo
#: modes (inlined = MPI-only, external = halo black-boxed / callback boundary), and the velocity
#: kernel in both ``__LOOP_EXCHANGE`` variants.
_CASES = SINGLE_TU_ARTIFACTS

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not (HAVE_FLANG and HAVE_OPENMPI), reason="needs an LLVM flang on PATH + OpenMPI"),
    pytest.mark.skipif(not have_icon_atmo(),
                       reason="icon-model atmosphere source not checked out; run "
                       "`git submodule update --init --recursive tests/icon/full/icon-model`"),
]


@pytest.mark.xdist_group("atmo_fparser")
@pytest.mark.parametrize("key,halo_mode,filename,entry,loop_exchange",
                         _CASES,
                         ids=[f"{c[0]}-{c[1]}-{'loopexch' if c[4] else 'noloopexch'}" for c in _CASES])
def test_extract_compiles_and_matches_committed(tmp_path, key, halo_mode, filename, entry, loop_exchange):
    """Extract one kernel, in one halo mode and one ``__LOOP_EXCHANGE`` variant, into a compiling
    single TU and check it against the committed artifact -- both halo modes (inlined = MPI-only,
    external = callback boundary) and both layout variants must always be correct."""
    variant = f"{key}_{halo_mode}_{'le' if loop_exchange else 'nole'}"
    res = extract_single_tu(_SOURCE[key], entry, tmp_path / variant, halo_mode=halo_mode, loop_exchange=loop_exchange)
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


_ACC_LINE = re.compile(r"\s*!\s*\$\s*acc", re.IGNORECASE)


@pytest.mark.xdist_group("atmo_fparser")
def test_keep_acc_directives_survive_into_the_velocity_tu(tmp_path):
    """``keep_acc_directives=True`` must carry the ~94 ``!$ACC`` source
    directives of ``mo_velocity_advection`` into the emitted TU, attached to
    their statements, WITHOUT changing one byte of the code around them
    (stripping the directive lines must reproduce the committed artifact)."""
    res = extract_single_tu(_SOURCE["velocity_advection"],
                            "mo_velocity_advection::velocity_tendencies",
                            tmp_path / "keep_acc",
                            loop_exchange=False,
                            keep_acc_directives=True)
    assert res["tu_path"] is not None and Path(res["tu_path"]).is_file(), res["output"][-4000:]
    tu_lines = Path(res["tu_path"]).read_text().splitlines(keepends=True)
    n_acc = sum(1 for line in tu_lines if _ACC_LINE.match(line))
    assert 60 <= n_acc <= 150, f"expected ~90 surviving !$ACC directives, got {n_acc}"
    committed = (_HERE / "velocity_advection_inlined_no_loop_exchange_single_tu.f90").read_text()
    stripped = "".join(line for line in tu_lines if not _ACC_LINE.match(line))
    assert stripped == committed, "keep_acc_directives perturbed the TU beyond the inserted directive lines"
