"""Bit-exact standalone single-rank differential for the ICON-O free-surface
solver ``solve_free_sfc_ab_mimetic`` (Mode-1 external halo).

The numerical counterpart to the ``test_extract_single_tu`` extraction gate: the
free-surface surface-pressure solver driver lowers to an SDFG, is driven on a
degenerate valid single-rank mesh, and its every mutable output is compared
BIT-EXACT (max diff exactly 0.0, no tolerance) against the stock gfortran
``solve_free_sfc_ab_mimetic`` on the same inputs -- the same ``run_kernel_e2e``
engine that pins the ocean block kernels (``test_ocean_kernel_numerical_e2e``),
scaled to the full dynamical-core driver.

The external halo policy (``sync_patch_array`` / ``exchange_data`` / the
``p_*`` collectives / MPI comm-pattern init / terminal IO+timers / the debug
``dbg_print`` / ``minmaxmean`` reporters) is dropped from the DUT SDFG
(``do_not_emit``) and made a no-op in the reference (the halo's ``mpi_*`` leaves
resolve to :data:`icon._halo_modes._MPI_NOOP_IMPL`); single-rank halo exchange
is a no-op (no neighbours), so both runs read identical state.  ``inject_use_mpi``
gives the inlined ``mo_mpi`` a ``use mpi`` so its dual-typed real*8/real*4
point-to-point calls resolve through the stub's one ``type(*)`` assumed-type
interface (no ``-fallow-argument-mismatch``).

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

# Halo / sync / collectives / comm-pattern init / IO+timers / debug reporters
# dropped from the DUT SDFG so no MPI survives single-rank and both sides read
# identical state.  Mirrors the atmosphere ``solve_nh`` set plus the ocean
# ``dbg_print_2d`` / ``dbg_print_3d`` / ``minmaxmean`` debug reporters.
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
                   reason="solve_free_sfc SDFG builds, runs and mutates output, but is not yet bit-exact vs stock "
                   "gfortran: 7 of 171 output arrays diverge -- vort / press_hyd / veloc_adv_vert (56/512 elems) "
                   "and g_n / g_nimd / press_grad / veloc_adv_horz (1/512, |d|~1-2). The DUT computes halo/boundary "
                   "cells (index [0,0,*]) the reference leaves 0.0 on the degenerate single-rank mesh -- a "
                   "subset-range/halo loop-bound difference plus mesh-seed tuning. Bit-exact bring-up is its own "
                   "focused session; strict=True flags the day it starts passing so this marker is removed.")
@pytest.mark.xdist_group("ocean_fparser")
def test_solve_free_sfc_numerical_e2e(tmp_path: Path):
    """solve_free_sfc -> SDFG, driven on a degenerate valid mesh, BIT-EXACT
    against stock gfortran (halo neutralised on both sides).

    NOTE: seeds below are a first attempt -- the free-surface driver reads patch
    geometry (nproma / n_zlev / subset-range block bounds) as module globals /
    struct members; tune ``module_seeds`` + ``array_overrides`` during bit-exact
    bring-up from the harness's reported binding signature.
    """
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
            # Time-level indices (mo_dynamics_config, static INTEGER(10), BSS 0 in
            # isolation) + single-domain index (mo_grid_config::n_dom): drive
            # p_prog(nold(1))/p_prog(nnew(1)) and p_patch_2d(n_dom) to valid
            # elements. Without these p_prog(0)/p_patch_2d(0) OOB-crash the run.
            "nold": 1,
            "nnew": 1,
            "n_dom": 1,
        },
        do_not_emit=_DO_NOT_EMIT,
        prelude_paths=[stub, noop],
        inject_use_mpi=True,
        # The single-TU extraction stubs the ocean-solver allocator
        # (``ocean_solve_construct`` -> empty body), so the stock reference leaves
        # ``free_sfc_solver % x_loc_wp`` (+ its ``_comp`` twin, ``res_loc_wp``)
        # unallocated and SEGVs on the firstguess write.  The DUT gets this scratch
        # from the SDFG marshalling layer; the reference shim must build it.  Shape
        # mirrors ICON's ``x_loc_wp(nidx=nproma, nblk_a=alloc_cell_blocks)`` /
        # ``res_loc_wp(2)``; it is fully overwritten (or zero-init here) before read.
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
