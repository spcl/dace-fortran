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
        # dolic_{c,e} = wet-level count. int_fill would leave it 1, a single-level
        # ocean -- but veloc_adv_vert_mimetic_rot only initialises the surface
        # value z_adv_u_i(jc,1) inside ``IF (fin_level >= 2)`` (fin_level =
        # dolic_c), so a 1-level cell reads z_adv_u_i(jc,1) uninitialised and
        # map_vec_prismtop2center feeds the garbage into veloc_adv_vert (then g_n,
        # g_nimd).  Real wet cells have dolic >= 2; pin both to 2 (valid two-level
        # ocean, in-bounds for n_zlev=7) so the surface init path runs.
        array_overrides={
            "patch_3d_p_patch_1d_dolic_c": 2,
            "patch_3d_p_patch_1d_dolic_e": 2,
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
        # icon_pp_edge_vnpredict_scheme reads the MODULE GLOBAL
        # mo_ocean_physics_types::v_params, not the p_phys_param dummy the entry point is
        # handed.  ICON's init_ho_params is outside the single TU, so v_params's POINTER
        # members are unassociated: the reference NULL-derefs a_veloc_v once dolic_e=2
        # makes the ``DO jk=2,dolic_e`` body run, and the DUT's binding, sizing the SoA
        # companion from size(v_params%a_veloc_v), takes the degenerate (1,1,1) fallback
        # and the same loop smashes the heap.  Both shims build them from p_phys_param, so
        # both sides start from byte-identical values.  Only the deferred-shape members
        # need this: v_params's scalars (bottom_drag_coeff, a_veloc_v_back) lower to an
        # SDFG transient zero-initialised by a ``zinit_v_params_*`` tasklet, which already
        # agrees with the reference's uninitialised module BSS.
        ref_global_binds=[
            ["mo_ocean_physics_types", "v_params", "a_veloc_v", "p_phys_param % a_veloc_v"],
            ["mo_ocean_physics_types", "v_params", "velocity_windmixing", "p_phys_param % velocity_windmixing"],
        ],
    )
    assert res["passed"], f"solve_free_sfc e2e did not run:\n{res['output'][-6000:]}"
    assert res["n_changed"] > 0, "solve_free_sfc produced no output changes (kernel ran as a no-op?)"
    assert res["max_diff"] == 0.0, f"solve_free_sfc not bit-exact: max|dut-ref| = {res['max_diff']}"
