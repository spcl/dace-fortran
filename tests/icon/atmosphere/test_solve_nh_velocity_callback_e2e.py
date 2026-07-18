"""Bit-exact differential e2e for ICON ``solve_nh`` with ``velocity_tendencies`` kept as a
CALLBACK to our velocity SDFG (SDFG-to-SDFG composition over the C ABI).

Architecture: inner velocity SDFG -> ``libvelocity_inner_wrap.so`` (``velocity_tendencies_c``
per-member-SoA shim); outer ``solve_nh`` SDFG registers it ``keep_external`` and drops
halo/sync/timers. DUT = outer SDFG (dispatches velocity to the inner ``.so``); REF = stock
Fortran solve_nh calling stock velocity_tendencies. Both run on the same degenerate mesh with
every mutable struct deep-copied, compared prognostic + full-diag + prep_adv BIT-EXACT.

``@pytest.mark.long``, single-node (2-rank halo variant is a separate ``@pytest.mark.mpi``
follow-up, not built here).

STATUS: the outer-SDFG-builds gate below PASSES (marshal v2 landed: ``t_patch``'s
``primal_normal_cell``/``dual_normal_cell``, an allocatable array of the value record
``t_tangent_vectors``, now expands to one SoA leaf per record field). Remaining: wire the
inner velocity ``.so`` + the full deep-copy-all differential run.
"""
import re
import shutil
from pathlib import Path

import pytest

from _util import build_sdfg, have_flang
from dace_fortran.bindings.bind_c_shim import emit_bind_c_shim, scalar_pointer_members
from dace_fortran.bindings.fortran_interface import build_auto_interface
from dace_fortran.build import make_builder
from dace_fortran.external import Arg, ExternalCall, apply_external_functions, clear_external_registry, keep_external
from dace_fortran.flang_codebase import find_openmpi_include

pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not on PATH"),
    pytest.mark.skipif(find_openmpi_include() is None, reason="OpenMPI headers not found"),
]

_HERE = Path(__file__).resolve().parent
_TU = _HERE / "solve_nonhydro_inlined_single_tu.f90"
_ENTRY = "mo_solve_nonhydro::solve_nh"
_VELOCITY_TU = _HERE / "velocity_advection_inlined_single_tu.f90"
_VELOCITY_ENTRY = "mo_velocity_advection::velocity_tendencies"

# five derived types cross per-member SoA (matches the inner bind_c_shim), then three rank-3
# arrays + scalars; same shape as scripts/build_icon_dace_libs.py's dycore wrapper.
_VELOCITY_CALLBACK_ARGS = tuple(
    [Arg(kind="aos", intent="inout", c_abi="per_member_soa")]  # p_prog
    + [Arg(kind="aos", intent="in", c_abi="per_member_soa")]  # p_patch
    + [Arg(kind="aos", intent="in", c_abi="per_member_soa")]  # p_int
    + [Arg(kind="aos", intent="inout", c_abi="per_member_soa")]  # p_metrics
    + [Arg(kind="aos", intent="inout", c_abi="per_member_soa")]  # p_diag
    + [Arg(kind="array", dtype="float64", intent="inout")] * 3  # z_w_concorr_me / z_kin_hor_e / z_vt_ie
    + [Arg(kind="scalar", dtype="int32", intent="in")] * 2  # ntnd / istep
    + [Arg(kind="scalar", dtype="bool", intent="in")]  # lvn_only
    + [Arg(kind="scalar", dtype="float64", intent="in")] * 2  # dtime / dt_linintp_ubc
    + [Arg(kind="scalar", dtype="bool", intent="in")])  # ldeepatmo

# halo/sync/diagnostics dropped so no MPI survives and velocity is the only external;
# same set as the atmo "external" harness config plus inlined-mode side effects.
_DO_NOT_EMIT = [
    "sync_patch_array", "sync_patch_array_mult", "exchange_data", "p_barrier", "p_max", "p_min", "p_sum", "global_max",
    "global_min", "global_sum", "setup_comm_pattern", "finish", "message", "message_text", "warning", "print_status",
    "print_value", "init_logger", "dbg_print", "work_mpi_barrier", "timer_start", "timer_stop", "new_timer",
    "delete_timer", "check_patch_array_3d_dp"
]


@pytest.mark.xdist_group("atmo_solve_nh_callback")
def test_solve_nh_velocity_callback_outer_sdfg_builds(tmp_path: Path):
    """The OUTER solve_nh SDFG with velocity as a per-member-SoA callback builds.

    Enabling gate for the velocity-callback differential e2e; passed once marshal v2 (value-
    record-array members) let ``t_patch``'s ``primal_normal_cell``/``dual_normal_cell`` expand
    to one SoA leaf per record field. Full differential (DUT vs REF, bit-exact) not built here.
    """
    clear_external_registry()
    keep_external(
        "velocity_tendencies",
        c_name="velocity_tendencies_c",
        args=_VELOCITY_CALLBACK_ARGS,
        dynamic_extents_abi=True,
    )
    apply_external_functions(do_not_emit=_DO_NOT_EMIT)
    try:
        sdfg = make_builder(_TU.read_text(), entry=_ENTRY, name="solve_nh_callback", out_dir=tmp_path / "sdfg").build()
        sdfg.validate()
        # velocity callback must emit as an external library node, not inlined/dropped
        from dace.sdfg import nodes as dnodes
        ext = [
            n for n, _ in sdfg.all_nodes_recursive()
            if isinstance(n, dnodes.LibraryNode) and "velocity" in (n.label or "").lower()
        ]
        assert ext, "outer SDFG built but emitted no external velocity_tendencies call"
    finally:
        clear_external_registry()


@pytest.mark.xdist_group("atmo_solve_nh_callback")
def test_callback_abi_aligns_slot_for_slot_with_inner_shim(tmp_path: Path):
    """The OUTER solve_nh marshalled ``velocity_tendencies_c(...)`` call lines up with the
    INNER velocity ``bind_c_shim`` slot-for-slot in COUNT and TYPE (real-scale version of the
    toy ``marshal_shim_abi_alignment_test``).

    * COUNT -- the shim emits one ``<flat>_lb<i>`` slot per dim of every dynamic member; guards
      ``extract_vars.cpp`` giving deferred-shape ALLOCATABLE/POINTER dummies uniformly-free
      bounds, while value-record AoR companions stay 1-based extent-only (no spurious lb).
    * TYPE -- a grid-dim scalar the shim takes ``type(c_ptr), value`` must marshal as a
      materialised ``&_pv_<sym>`` pointer, not a raw by-value int -- else the inner
      ``c_f_pointer`` dereferences the integer as an address.
    """
    clear_external_registry()
    inner_builder = build_sdfg(_VELOCITY_TU.read_text(),
                               tmp_path / "vel_sdfg",
                               name="velocity_tendencies",
                               entry=_VELOCITY_ENTRY)
    inner_builder.build()
    iface = build_auto_interface(inner_builder._fortran_interface_raw, "velocity_tendencies")
    shim = emit_bind_c_shim(iface, str(tmp_path / "vel_c.f90")).read_text()
    header = re.search(r"subroutine\s+velocity_tendencies_c\s*\(([^)]*)\)", shim, re.S)
    inner_slots = [a.strip() for a in header.group(1).replace("&", " ").split(",") if a.strip()]
    ptr_members = scalar_pointer_members(iface)

    clear_external_registry()
    keep_external(
        "velocity_tendencies",
        c_name="velocity_tendencies_c",
        args=_VELOCITY_CALLBACK_ARGS,
        dynamic_extents_abi=True,
        callee_ptr_scalar_members=ptr_members,
    )
    apply_external_functions(do_not_emit=_DO_NOT_EMIT)
    try:
        outer = make_builder(_TU.read_text(), entry=_ENTRY, name="solve_nh_abi", out_dir=tmp_path / "sdfg").build()
        node = next((n for st in outer.all_states() for n in st.nodes() if isinstance(n, ExternalCall)), None)
        assert node is not None, "outer solve_nh SDFG emitted no velocity ExternalCall"
        call = re.search(r"\bvelocity_tendencies_c\(([^;]*)\)\s*;", node.body)
        outer_args = [a.strip() for a in call.group(1).split(",")]
    finally:
        clear_external_registry()

    # COUNT: the marshalled call matches the inner shim slot-for-slot.  A drift
    # here is a dropped ``_lb`` slot on a deferred-shape member (count too low)
    # or a spurious lb on a value-record companion (count too high).
    assert len(outer_args) == len(inner_slots), (
        f"velocity callback ABI slot-count desync: inner shim {len(inner_slots)} vs "
        f"outer marshal {len(outer_args)}")

    # TYPE: each grid-dim scalar member the shim takes as a pointer marshals as a
    # materialised ``&_pv_<sym>`` -- the member slot, distinct from that same
    # symbol's legitimate by-value ``(int)(<sym>)`` use as ANOTHER array's extent.
    for member in ("p_patch_id", "p_patch_n_childdom", "p_patch_nblks_c", "p_patch_nblks_e", "p_patch_nblks_v",
                   "p_patch_nlev", "p_patch_nlevp1"):
        assert member in ptr_members, f"{member} should be a callee pointer member"
        assert f"&_pv_{member}" in outer_args, (
            f"grid-dim member {member} must marshal as a materialised pointer (&_pv_{member}); the "
            f"inner shim declares it type(c_ptr), value and dereferences via c_f_pointer")
