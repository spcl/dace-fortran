"""2-rank real-MPI differential for the ICON-O ocean dycore
``solve_free_sfc_ab_mimetic`` -- the multi-rank counterpart to
``test_solve_free_sfc_numerical_e2e.py``.

Same degenerate single-TU mesh and the same seeds as the single-rank run, with one
difference: the TU's ``sync_patch_array`` stubs are no longer DROPPED, they carry a real
halo exchange.  So the DUT SDFG holds genuine ``dace.libraries.mpi`` nodes and each rank's
outputs are compared bit-exact against stock gfortran running the same exchange.

Why the stubs and not a ``keep_external``: ICON's sync takes a ``t_patch`` struct, which is
not marshalable across the flat C ABI.  Filling the stub body instead keeps the exchange
INSIDE the SDFG, where the bridge lowers the MPI calls natively and the unused ``p_patch``
dummy just falls away.  The body packs into contiguous local buffers before sending, which
is what ICON itself does -- and it keeps assumed-shape arrays away from the
implicit-interface MPI calls that ``hlfir-fold-copy-in-out`` cannot model.

Block 1 is owned, block 2 is halo, and the swap lands the neighbour's owned block in it --
the same convention as ``test_ocean_veloc_mpi_sync_e2e.py`` and the dummy 2-rank dycore.
The exchange pairs rank r with rank 1-r, so this runs at exactly 2 ranks.

``@pytest.mark.long``: builds the full dycore to an SDFG (minutes).
"""
import ctypes
import shutil
from pathlib import Path

import numpy as np
import pytest

from _util import build_on_root, have_flang
from icon._halo_modes import _MPI_NOOP_IMPL, _MPI_STUB
from icon.ocean._ocean_harness import have_icon_ocean
from icon.ocean._ocean_e2e import build_dut_and_ref, synth_call_inputs, _invoke

pytestmark = [
    pytest.mark.long,
    pytest.mark.mpi,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("mpifort") is None, reason="mpifort not on PATH (need an MPI Fortran wrapper)"),
    pytest.mark.skipif(not have_icon_ocean(), reason="icon-model ocean source not checked out"),
]

_HERE = Path(__file__).resolve().parent
_TU = _HERE / "solve_free_sfc_single_tu.f90"
_ENTRY = "mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic"
_N = 8

# Same drop-list as the single-rank e2e MINUS the two sync entries: those now carry a real
# exchange and must reach the SDFG.  Everything else (collectives, comm-pattern init, IO,
# timers, debug reporters) stays dropped on the DUT and no-op'd in the reference, exactly as
# single-rank, so the two runs remain comparable.
_DO_NOT_EMIT = [
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

# ``_MPI_NOOP_IMPL`` defines mpi_send / mpi_recv as real symbols; linked against libmpi they
# would SHADOW the real ones and the halo exchange would silently do nothing -- a vacuous
# pass.  Drop exactly those two so they resolve to libmpi, and keep the rest (mo_mpi's
# wrappers still have to link, they are just never reached).
_NOOP_KEEP = _MPI_NOOP_IMPL
for _dead in ("mpi_send", "mpi_recv"):
    _start = _NOOP_KEEP.index(f"subroutine {_dead}(")
    _end = _NOOP_KEEP.index("end subroutine\n", _start) + len("end subroutine\n")
    _NOOP_KEEP = _NOOP_KEEP[:_start] + _NOOP_KEEP[_end:]


def _sync_body(field: str, rank: int) -> str:
    """Halo-exchange body for a ``field`` of rank 2 or 3: pack block 1, swap with the
    partner, land the result in block 2.

    Packed through contiguous local buffers -- an assumed-shape actual argument at an
    implicit-interface MPI call forces a copy_in/copy_out pair the bridge cannot model, and
    ICON packs its send buffers the same way.  MPI constants are hard-coded to their OpenMPI
    values so the bridge build needs no ``mpi.mod``.  Even/odd ordering avoids the deadlock a
    symmetric blocking send would risk.
    """
    buf_decl = ":" if rank == 2 else ":, :"
    buf_dims = "sync_n1" if rank == 2 else "sync_n1, sync_n2"
    owned = "(:, 1)" if rank == 2 else "(:, :, 1)"
    halo = "(:, 2)" if rank == 2 else "(:, :, 2)"
    count = "sync_n1" if rank == 2 else "sync_n1 * sync_n2"
    blk = "2" if rank == 2 else "3"
    n2 = "" if rank == 2 else "\n    sync_n2 = SIZE(%s, 2)" % field
    return f"""
    INTEGER :: sync_rank, sync_np, sync_neigh, sync_ierr, sync_cnt, sync_n1, sync_n2, sync_nb
    REAL(KIND = 8), ALLOCATABLE :: sync_sbuf({buf_decl}), sync_rbuf({buf_decl})
    INTEGER, PARAMETER :: SYNC_COMM = 0
    INTEGER, PARAMETER :: SYNC_DP = 17
    INTEGER, PARAMETER :: SYNC_STAT_IGNORE = -1
    EXTERNAL :: MPI_Comm_rank, MPI_Comm_size, MPI_Send, MPI_Recv
    CALL MPI_Comm_rank(SYNC_COMM, sync_rank, sync_ierr)
    CALL MPI_Comm_size(SYNC_COMM, sync_np, sync_ierr)
    IF (sync_np <= 1) RETURN
    sync_n1 = SIZE({field}, 1){n2}
    sync_nb = SIZE({field}, {blk})
    IF (sync_nb < 2) RETURN
    sync_cnt = {count}
    ALLOCATE(sync_sbuf({buf_dims}))
    ALLOCATE(sync_rbuf({buf_dims}))
    sync_neigh = 1 - sync_rank
    sync_sbuf = {field}{owned}
    IF (MOD(sync_rank, 2) == 0) THEN
      CALL MPI_Send(sync_sbuf, sync_cnt, SYNC_DP, sync_neigh, typ, SYNC_COMM, sync_ierr)
      CALL MPI_Recv(sync_rbuf, sync_cnt, SYNC_DP, sync_neigh, typ, SYNC_COMM, SYNC_STAT_IGNORE, sync_ierr)
    ELSE
      CALL MPI_Recv(sync_rbuf, sync_cnt, SYNC_DP, sync_neigh, typ, SYNC_COMM, SYNC_STAT_IGNORE, sync_ierr)
      CALL MPI_Send(sync_sbuf, sync_cnt, SYNC_DP, sync_neigh, typ, SYNC_COMM, sync_ierr)
    END IF
    {field}{halo} = sync_rbuf
    DEALLOCATE(sync_sbuf, sync_rbuf)
"""


#: ``(stub name, declaration line that ends its body, field, rank)``.  The declaration is the
#: last line of each empty stub, so the exchange is spliced straight after it.
_STUBS = [
    ("sync_patch_array_3d_dp", "    CHARACTER(LEN = *), TARGET, INTENT(IN), OPTIONAL :: opt_varname", "arr", 3),
    ("sync_patch_array_2d_dp", "    CHARACTER*(*), INTENT(IN), OPTIONAL :: opt_varname", "arr", 2),
]


def _fill_sync_stubs(src: str) -> str:
    """Splice a real halo exchange into the TU's empty ``sync_patch_array`` stub bodies."""
    for name, decl, field, rank in _STUBS:
        end = f"  END SUBROUTINE {name}"
        anchor = f"{decl}\n{end}"
        if src.count(anchor) != 1:
            raise RuntimeError(f"{name} stub anchor not unique (found {src.count(anchor)})")
        src = src.replace(anchor, decl + _sync_body(field, rank) + end)
    return src


def _build_artifacts(tmp_path: Path) -> dict:
    """Build (rank 0 only) the DUT + REF against the sync-filled TU."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    tu = tmp_path / "solve_free_sfc_mpi_tu.f90"
    tu.write_text(_fill_sync_stubs(_TU.read_text()))
    stub = tmp_path / "_mpi_stub.f90"
    stub.write_text(_MPI_STUB)
    noop = tmp_path / "_mpi_noop_impl.f90"
    noop.write_text(_NOOP_KEEP)
    return build_dut_and_ref(
        tu,
        _ENTRY,
        n=_N,
        out=tmp_path,
        module_seeds={
            "nproma": 8,
            "n_zlev": 7,
            "nold": 1,
            "nnew": 1,
            "n_dom": 1,
        },
        do_not_emit=_DO_NOT_EMIT,
        prelude_paths=[stub, noop],
        inject_use_mpi=True,
        ref_solver_allocs=[
            ["free_sfc_solver", "x_loc_wp", "nproma__refmod, patch_3d % p_patch_2d(1) % alloc_cell_blocks"],
            ["free_sfc_solver", "res_loc_wp", "2"],
            ["free_sfc_solver_comp", "x_loc_wp", "nproma__refmod, patch_3d % p_patch_2d(1) % alloc_cell_blocks"],
            ["free_sfc_solver_comp", "res_loc_wp", "2"],
        ],
        ref_global_binds=[
            ["mo_ocean_physics_types", "v_params", "a_veloc_v", "p_phys_param % a_veloc_v"],
            ["mo_ocean_physics_types", "v_params", "velocity_windmixing", "p_phys_param % velocity_windmixing"],
        ],
        # The halo now resolves to real MPI on both sides.
        fc="mpifort",
    )


@pytest.mark.xfail(strict=True,
                   reason="Blocked in the bridge, not here: with the sync live, "
                   "``hlfir-fold-copy-in-out`` cannot fold four ``hlfir.copy_in`` pairs at the dycore's own "
                   "top-level calls into veloc_adv_horz_mimetic / veloc_adv_vert_mimetic / velocity_diffusion "
                   "(assumed-shape actuals -- veloc_adv_vert_e(:,:,:), laplacian_vn_out(:,:,:) -- forwarded to "
                   "explicit-shape dummies).  The single-rank e2e never hits this because ``do_not_emit`` drops "
                   "the sync and ``hlfir-drop-stub-calls`` erases the copy-in temps hanging off it before the "
                   "fold pass runs.  Same defect fails the five sync_devirt_mpi_libnode_test cases: a whole-array "
                   "box copy_in whose fir.box_addr feeds a raw call argument instead of an inlined-callee "
                   "hlfir.declare, so reparentMemberCopy finds no alias to reparent onto.  Remove this marker "
                   "once FoldCopyInOut.cpp handles that shape.")
@pytest.mark.xdist_group("ocean_fparser")
def test_solve_free_sfc_2rank_bit_exact(tmp_path: Path):
    """solve_free_sfc on 2 ranks with a real MPI halo: SDFG vs stock gfortran, bit-exact per rank."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    if size != 2:
        pytest.skip("needs exactly 2 ranks (mpirun --oversubscribe -n 2 ...)")

    # Every rank builds against rank 0's tmp_path so the .so artefacts are shared.
    tmp_path = Path(world.bcast(str(tmp_path) if rank == 0 else None, root=0))
    art = build_on_root(world, lambda: _build_artifacts(tmp_path))
    world.Barrier()

    dace_name = art["dace_name"]
    # Distinct per-rank seeds so the exchange has something to move: with identical inputs a
    # dropped halo would be indistinguishable from a working one.
    call_plan, inputs, ptr_args, _ptr_local = synth_call_inputs(
        art["shim"],
        n=_N,
        seed=rank,
        int_fill=1,
        array_overrides={
            "patch_3d_p_patch_1d_dolic_c": 2,
            "patch_3d_p_patch_1d_dolic_e": 2,
        },
    )

    dut = {k: v.copy() for k, v in inputs.items()}
    _invoke(art["dut_so"], call_plan, dut, f"{dace_name}_c", sdfg_so=art["sdfg_so"], module_seeds=art["seed_specs"])
    ref = {k: v.copy() for k, v in inputs.items()}
    _invoke(art["ref_so"], call_plan, ref, f"{dace_name}_ref_c", module_seeds=art["seed_specs"])

    n_changed = 0
    for k in ptr_args:
        d, r = dut[k].astype(np.float64), ref[k].astype(np.float64)
        equal = (d == r) | (np.isnan(d) & np.isnan(r))
        assert equal.all(), f"rank {rank}: {k} diverged from the reference at {(~equal).sum()} position(s)"
        if not np.array_equal(dut[k], inputs[k]):
            n_changed += 1
    assert n_changed > 0, f"rank {rank}: no output changed -- the dycore ran as a no-op"

    # The differential above passes just as well if the halo never fired, so prove the
    # exchange moved real data: re-run the DUT on COMM_SELF, where the sync takes its
    # size<=1 no-op path.  A changed result is only possible if the 2-rank swap ran.
    self_bufs = {k: v.copy() for k, v in inputs.items()}
    ctypes.CDLL(art["dut_so"], mode=ctypes.RTLD_GLOBAL)
    _invoke(art["dut_so"],
            call_plan,
            self_bufs,
            f"{dace_name}_c",
            sdfg_so=art["sdfg_so"],
            module_seeds=art["seed_specs"])
    assert any(
        not np.array_equal(dut[k], self_bufs[k])
        for k in ptr_args), (f"rank {rank}: the 2-rank run is identical to the single-rank one -- the halo exchange "
                             "moved no neighbour data")
