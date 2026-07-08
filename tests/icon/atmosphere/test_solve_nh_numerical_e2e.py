"""Bit-exact standalone single-rank differential for the ICON ``solve_nh`` dycore.

The numerical counterpart to the build-only ``test_solve_nh_binding.py``: the
full nonhydrostatic solver lowers to an SDFG, is driven on a degenerate valid
single-rank mesh, and its every mutable output is compared BIT-EXACT (max diff
exactly 0.0, no tolerance) against the stock gfortran ``solve_nh`` on the same
inputs -- the same ``run_kernel_e2e`` engine that pins ``velocity_tendencies``
(``test_velocity_numerical_e2e``), scaled to the ~900-argument dycore.

**Milestone 2a-i -- velocity + halo neutralised on BOTH sides.**  ``solve_nh``
calls ``velocity_tendencies`` (a stub in this TU) and the halo exchange /
sync / timers; all are dropped from the DUT SDFG (``do_not_emit``) and made
no-ops in the reference (the halo's ``mpi_*`` leaves resolve to
:data:`icon._halo_modes._MPI_NOOP_IMPL`).  This is sound because everything
velocity would write -- ``ddt_vn_apc_pc`` / ``ddt_vn_cor_pc`` / ``ddt_w_adv_pc``
-- is a ``t_nh_diag`` struct member, i.e. a flat ABI slot initialised
identically on both sides, so dropping velocity leaves both runs reading the
same state; and single-rank halo exchange is a no-op (no neighbours).  It gates
the BULK of the dycore (the pressure + vertical-implicit solve) plus the
~900-arg binding and the differential harness.  The real velocity contribution
is a follow-up (2a-ii) that wires velocity back as a per-member-SoA callback to
the inner velocity SDFG (the callback ABI is aligned -- see
``test_solve_nh_velocity_callback_e2e``).

One reference fix-up makes the extracted single-TU gfortran-EXECUTABLE (the SDFG
build reads the raw TU; the bridge resolves things its own way): ``inject_use_mpi``
gives the inlined ``mo_mpi`` a ``use mpi`` so its dual-typed real*8 / real*4
point-to-point calls resolve through the stub's one ``type(*)`` assumed-type
interface (no ``-fallow-argument-mismatch``).  The bridge's devirtualised call
names (``cells2verts_scalar_dp_deconiface_127`` / ``p_irecv_dp_deconiface_10``)
are NOT undefined -- each is a ``USE <mod>, ONLY: <name> => <base>`` rename of an
in-TU body -- so the TU compiles as-is; only the 7 raw ``mpi_*`` leaves are
genuinely undefined and resolve to the no-op prelude.

``@pytest.mark.long``: builds the 3166-LoC dycore to an SDFG (minutes).
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

# Halo / sync / diagnostics / timers dropped from the DUT SDFG so no MPI survives
# single-rank and both sides read identical state.  ``velocity_tendencies`` is
# NOT dropped: it is a no-op STUB in this TU, so the bridge inlines it to nothing
# on the DUT and the reference calls the same empty stub -- both neutralise
# velocity identically.  Dropping it instead would leave a per-member-SoA marshal
# BOUNDARY (``MarshalExternalStructs`` gathers ``p_patch``'s value-record members
# ``primal_normal_cell`` / ``dual_normal_cell`` for the callback) whose leaves
# then COLLIDE with the ordinary FlattenStructs split -> the binding double-
# declares ``p_patch_edges_primal_normal_cell_v1`` etc.  Milestone 2a-i keeps
# velocity a no-op; wiring the real velocity callback is 2a-ii.
_DO_NOT_EMIT = [
    "sync_patch_array", "sync_patch_array_mult", "exchange_data", "p_barrier", "p_max", "p_min", "p_sum", "global_max",
    "global_min", "global_sum", "setup_comm_pattern", "finish", "message", "message_text", "warning", "print_status",
    "print_value", "init_logger", "dbg_print", "work_mpi_barrier", "timer_start", "timer_stop", "new_timer",
    "delete_timer", "check_patch_array_3d_dp"
]


@pytest.mark.xdist_group("atmo_solve_nh_numerical")
def test_solve_nh_numerical_e2e(tmp_path: Path):
    """solve_nh -> SDFG, driven on a degenerate valid mesh, is BIT-EXACT against
    the stock gfortran solve_nh (velocity + halo neutralised on both sides)."""
    stub = tmp_path / "_mpi_stub.f90"
    stub.write_text(_MPI_STUB)
    noop = tmp_path / "_mpi_noop_impl.f90"
    noop.write_text(_MPI_NOOP_IMPL)

    res = run_kernel_e2e(
        _TU,
        _ENTRY,
        int_fill=1,
        # ICON config globals: an isolated kernel reads them BSS-0.  For
        # bit-exactness only IDENTICAL values matter (seeded on both sides), so
        # unseeded globals stay 0 on both -- valid unless 0 OOBs.  Seed the block
        # size + the vertical level-start / substep bounds that index or divide
        # (0 -> jk=0 OOB, or dtime/0 -> NaN): everything else takes its 0-branch
        # equally on both sides.
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
        do_not_emit=_DO_NOT_EMIT,
        prelude_paths=[stub, noop],
        inject_use_mpi=True,
    )
    assert res["passed"], f"solve_nh e2e did not run:\n{res['output'][-6000:]}"
    assert res["n_changed"] > 0, "solve_nh produced no output changes (kernel ran as a no-op?)"
    assert res["max_diff"] == 0.0, f"solve_nh not bit-exact: max|dut-ref| = {res['max_diff']}"
