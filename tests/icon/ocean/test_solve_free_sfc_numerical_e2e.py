"""Bit-exact single-rank differential for ICON-O ``solve_free_sfc_ab_mimetic``
(Mode-1 external halo), the numerical counterpart to ``test_extract_single_tu``.

Driver lowers to an SDFG, runs on a degenerate single-rank mesh, and every
mutable output must match stock gfortran EXACTLY (max diff 0.0) via the same
``run_kernel_e2e`` engine as the ocean block-kernel tests. Halo/collectives/
IO/timers/debug reporters are dropped from the DUT (``do_not_emit``) and
no-op'd in the reference (single-rank halo exchange has no neighbours anyway,
so both sides read identical state). ``inject_use_mpi`` lets the inlined
``mo_mpi``'s dual-typed real*8/real*4 calls resolve through the stub's
``type(*)`` assumed-type interface.

``@pytest.mark.long``: builds the full driver to an SDFG (minutes).
"""
import shutil
from pathlib import Path

import pytest

from _util import have_flang
from icon._halo_modes import _MPI_NOOP_IMPL, _MPI_STUB
from icon.ocean._ocean_harness import have_icon_ocean
from icon.ocean._ocean_e2e import run_kernel_e2e

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
    pytest.mark.skipif(not have_icon_ocean(), reason="icon-model ocean source not checked out"),
]

_HERE = Path(__file__).resolve().parent
_TU = _HERE / "solve_free_sfc_single_tu.f90"
_ENTRY = "mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic"

# Halo/sync/collectives/comm-pattern-init/IO+timers/debug reporters dropped so
# no MPI survives single-rank. Mirrors atmosphere solve_nh's set + ocean's dbg_print_2d/3d/minmaxmean.
_DO_NOT_EMIT = [
    "sync_patch_array",
    "sync_patch_array_mult",
    "exchange_data",
    "p_barrier",
    "p_max",
    "p_min",
    "p_sum",
    "global_max",
    "global_min",
    "global_sum",
    "setup_comm_pattern",
    "subset_transfer_construct",
    "finish",
    "message",
    "message_text",
    "warning",
    "print_status",
    "print_value",
    "init_logger",
    "dbg_print",
    "dbg_print_2d",
    "dbg_print_3d",
    "debug_print_maxminmean",
    "print_2dvalue_location",
    "work_mpi_barrier",
    "timer_start",
    "timer_stop",
    "new_timer",
    "delete_timer",
    "check_patch_array_3d_dp",
]


@pytest.mark.xfail(strict=True,
                   reason="solve_free_sfc SDFG builds and starts, but the DUT run ABORTS (SIGABRT, glibc "
                   "`free(): invalid size`; ref runs clean). Before the forwarded-optional PRESENT fix the "
                   "extracted TU mis-ran the div_oce vertical loop as level 0..0 (both opt levels read the "
                   "degenerate 0 seed), so the DUT barely wrote and survived to diverge on 7 of 171 arrays; "
                   "the corrected TU runs the true 1..n_zlev range and the extra writes now corrupt the heap. "
                   "Prime suspect: the bindings layer declares SDFG transient-dim symbols as uninitialized "
                   "locals and passes them to __dace_init (block_builders.py declares every frozen free "
                   "symbol; nothing sources transients), so a transient allocated with a garbage/zero extent "
                   "is overflowed by the full-depth loop and glibc aborts at teardown. Fixing that "
                   "transient-extent sourcing is its own focused session; strict=True flags the day this "
                   "starts passing so the marker is removed.")
@pytest.mark.xdist_group("ocean_fparser")
def test_solve_free_sfc_numerical_e2e(tmp_path: Path):
    """solve_free_sfc -> SDFG, driven on a degenerate valid mesh, BIT-EXACT against
    stock gfortran (halo neutralised both sides). NOTE: seeds are a first attempt --
    tune module_seeds/array_overrides during bit-exact bring-up."""
    stub = tmp_path / "_mpi_stub.f90"
    stub.write_text(_MPI_STUB)
    noop = tmp_path / "_mpi_noop_impl.f90"
    noop.write_text(_MPI_NOOP_IMPL)

    res = run_kernel_e2e(
        _TU,
        _ENTRY,
        int_fill=1,
        module_seeds={
            "nproma": 8,
            "n_zlev": 7,
            # nold/nnew (time-level indices) + n_dom (domain index) drive
            # p_prog(nold(1))/p_patch_2d(n_dom) to valid elements -- without them
            # p_prog(0)/p_patch_2d(0) OOB-crash the run.
            "nold": 1,
            "nnew": 1,
            "n_dom": 1,
        },
        do_not_emit=_DO_NOT_EMIT,
        prelude_paths=[stub, noop],
        inject_use_mpi=True,
        # single-TU extraction stubs ocean_solve_construct -> empty body, so the
        # stock reference leaves free_sfc_solver%x_loc_wp/res_loc_wp unallocated and
        # SEGVs; the DUT gets this scratch from SDFG marshalling, so the reference
        # shim must build it here (shape mirrors ICON's x_loc_wp(nproma,alloc_cell_blocks)/res_loc_wp(2)).
        ref_solver_allocs=[
            ["free_sfc_solver", "x_loc_wp", "nproma__refmod, patch_3d % p_patch_2d(1) % alloc_cell_blocks"],
            ["free_sfc_solver", "res_loc_wp", "2"],
            ["free_sfc_solver_comp", "x_loc_wp", "nproma__refmod, patch_3d % p_patch_2d(1) % alloc_cell_blocks"],
            ["free_sfc_solver_comp", "res_loc_wp", "2"],
        ],
    )
    assert res["passed"], f"solve_free_sfc e2e did not run:\n{res['output'][-6000:]}"
    assert res["n_changed"] > 0, "solve_free_sfc produced no output changes (kernel ran as a no-op?)"
    assert res["max_diff"] == 0.0, f"solve_free_sfc not bit-exact: max|dut-ref| = {res['max_diff']}"
