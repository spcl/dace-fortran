"""Bit-exact single-rank differential for ICON's solve_nh dycore vs stock gfortran (numerical
counterpart to build-only test_solve_nh_binding.py), using the same run_kernel_e2e engine as
test_velocity_numerical_e2e, scaled to the ~900-arg dycore.

Milestone 2a-i: velocity_tendencies and halo/sync/timers are neutralised on BOTH sides (dropped from
the DUT, no-op'd in the reference) rather than compared -- sound since velocity's outputs are flat ABI
slots initialised identically both sides, and single-rank halo exchange is a no-op. Real velocity
wiring (per-member-SoA callback) is 2a-ii (see test_solve_nh_velocity_callback_e2e).

inject_use_mpi gives the inlined mo_mpi a `use mpi` so its dual-typed point-to-point calls resolve
through the stub's `type(*)` interface (avoids -fallow-argument-mismatch); only the 7 raw mpi_* leaves
are genuinely undefined and need the no-op prelude -- the bridge's devirtualised call names are
USE-renames of in-TU bodies, not undefined.

@pytest.mark.long: builds the 3166-LoC dycore to an SDFG (minutes).
"""
import shutil
from pathlib import Path

import pytest

from _util import have_flang
from icon._halo_modes import _MPI_NOOP_IMPL, _MPI_STUB
from icon.ocean._ocean_e2e import run_kernel_e2e

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
]

_HERE = Path(__file__).resolve().parent
_TU = _HERE / "solve_nonhydro_inlined_single_tu.f90"
_ENTRY = "mo_solve_nonhydro::solve_nh"

# halo/sync/diag/timers dropped from the DUT SDFG so no MPI survives single-rank and both sides
# read identical state. velocity_tendencies is NOT dropped -- it's a no-op stub in this TU, so both
# DUT and reference call the same empty stub. Dropping it instead would leave a per-member-SoA
# marshal boundary (MarshalExternalStructs on p_patch's value-record members) whose leaves collide
# with the ordinary FlattenStructs split, double-declaring p_patch_edges_primal_normal_cell_v1 etc.
# Milestone 2a-i keeps velocity a no-op; 2a-ii wires it.
_DO_NOT_EMIT = [
    "sync_patch_array", "sync_patch_array_mult", "exchange_data", "p_barrier", "p_max", "p_min", "p_sum", "global_max",
    "global_min", "global_sum", "setup_comm_pattern", "finish", "message", "message_text", "warning", "print_status",
    "print_value", "init_logger", "dbg_print", "work_mpi_barrier", "timer_start", "timer_stop", "new_timer",
    "delete_timer", "check_patch_array_3d_dp"
]


@pytest.mark.xdist_group("atmo_solve_nh_numerical")
def test_solve_nh_numerical_e2e(tmp_path: Path):
    """solve_nh -> SDFG on a degenerate valid mesh is BIT-EXACT vs stock gfortran solve_nh (velocity
    + halo neutralised on both sides)."""
    stub = tmp_path / "_mpi_stub.f90"
    stub.write_text(_MPI_STUB)
    noop = tmp_path / "_mpi_noop_impl.f90"
    noop.write_text(_MPI_NOOP_IMPL)

    res = run_kernel_e2e(
        _TU,
        _ENTRY,
        int_fill=1,
        # isolated kernel reads config globals BSS-0; bit-exactness only needs IDENTICAL values
        # (seeded both sides), so unseeded globals stay 0 -- valid unless 0 OOBs. Seed block size +
        # level-start/substep bounds that index or divide (0 -> jk=0 OOB or dtime/0 -> NaN).
        module_seeds={
            "nproma": 8,
            "nflatlev": 1,
            "nflat_gradp": 1,
            "nrdmax": 1,
            "kstart_moist": 1,
            "ndyn_substeps_var": 1,
        },
        array_overrides={
            "p_patch_nlev": 7,
            "p_patch_nlevp1": 8
        },
        # vct_a (vertical coord table) is an ALLOCATABLE module global the kernel indexes directly
        # (vct_a(jk)); unallocated in an isolated run, so the stock reference would SEGV. Harness
        # allocates + fills it identically on both sides (value is inert, only cross-side identity
        # matters); sized past nlevp1 (8) so every vct_a(jk+1) stays in bounds.
        module_array_seeds={"vct_a": 16},
        do_not_emit=_DO_NOT_EMIT,
        prelude_paths=[stub, noop],
        inject_use_mpi=True,
    )
    assert res["passed"], f"solve_nh e2e did not run:\n{res['output'][-6000:]}"
    assert res["n_changed"] > 0, "solve_nh produced no output changes (kernel ran as a no-op?)"
    assert res["max_diff"] == 0.0, f"solve_nh not bit-exact: max|dut-ref| = {res['max_diff']}"
