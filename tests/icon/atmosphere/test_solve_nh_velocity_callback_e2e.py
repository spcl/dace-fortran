"""Bit-exact differential e2e for the ICON ``solve_nh`` dycore with
``velocity_tendencies`` kept as a CALLBACK to our velocity SDFG (SDFG-to-SDFG
composition over the C ABI).

Intended architecture (the same ``keep_external`` per-member-SoA composition the
velocity e2e ``test_dycore_outer_calls_velocity_sdfg_via_c_abi`` and
``scripts/build_icon_dace_libs.py`` build, scaled to the REAL ``solve_nh``):

  * **Inner** velocity SDFG -> ``libvelocity_inner_wrap.so`` exporting
    ``velocity_tendencies_c`` (its per-member-SoA ``bind_c_shim`` entry).
  * **Outer** ``solve_nh`` SDFG built from the extracted single-TU with
    ``velocity_tendencies`` registered ``keep_external(c_name=
    'velocity_tendencies_c', args=(<5 per-member-SoA structs> + arrays +
    scalars), dynamic_extents_abi=True)`` and the halo / sync / timers dropped
    (``apply_external_functions(do_not_emit=...)``) -- so the outer SDFG
    dispatches the velocity sub-call to the inner ``.so`` at run time and no MPI
    survives.
  * **DUT** = the outer ``solve_nh`` SDFG (dispatching velocity into the inner
    SDFG).  **REF** = stock Fortran ``solve_nh`` calling stock
    ``velocity_tendencies`` (full body).  Both run on the SAME degenerate valid
    mesh, with EVERY mutable struct deep-copied (:file:`mo_solve_nh_diff.f90`:
    ``clone_state_indep_prog`` re-points every ``prog`` AND every ``diag``
    pointer to a fresh target -- the velocity callback writes
    ``ddt_vn_apc_pc`` / ``ddt_vn_cor_pc`` / ``ddt_w_adv_pc``, which
    ``compare_diag`` checks) so the two runs are fully independent, and the
    prognostic + full-diag + prep_adv output is compared BIT-EXACT (max diff
    exactly 0.0, no tolerance).

``@pytest.mark.long`` (builds two SDFGs + gfortran-links) and SINGLE-NODE (REF
vs DUT in one process; the 2-rank halo-exchange variant is a separate
``@pytest.mark.mpi`` follow-up, NOT built here).

------------------------------------------------------------------------------
STATUS: xfail (strict) -- blocked on ``hlfir-marshal-external-structs`` v2.

Building the OUTER ``solve_nh`` SDFG with ``velocity_tendencies`` as a real
``keep_external`` callback raises in ``builder.emit_library.emit_call``:

    ValueError: external 'velocity_tendencies': 'aos' arg #4 has no marshalling
                group.  ... its struct has non-inline-flat members ...

ROOT CAUSE (root-caused precisely, not the generic v1/v2 blurb): the outer's
velocity call marshals FIVE derived types.  Four (``t_nh_prog`` / ``t_int_state``
/ ``t_nh_metrics`` / ``t_nh_diag``) marshal -- every member is a scalar or a
box-of-scalar-array (v2.1 handles both, incl. the rank-4 ``ddt_vn_apc_pc``).
``t_patch`` does NOT: its ``edges`` sub-record carries

    TYPE(t_tangent_vectors), ALLOCATABLE :: primal_normal_cell(:, :, :)
    TYPE(t_tangent_vectors), ALLOCATABLE :: dual_normal_cell(:, :, :)

i.e. ``box<heap<array<record<v1:f64, v2:f64>>>>`` -- an allocatable ARRAY OF A
VALUE RECORD.  ``MarshalExternalStructs.cpp``'s ``scalarStructPointee`` rejects
``t_patch`` because ``isRecursiveInlineFlatMember`` accepts only
box-of-scalar-*array* (``isBoxOfScalarArray``), not box-of-record-array, so
``t_patch`` is left un-tagged: the pass emits 4 groups for 5 aos args and
``emit_call`` fails at the 5th.  (Confirmed: with velocity DROPPED via
``do_not_emit`` the outer ``solve_nh`` SDFG builds and validates -- FlattenStructs
and the ``bind_c_shim`` value-record-array scatter both handle
``primal_normal_cell`` already; ONLY the marshal pass rejects it.)

WHAT FLIPS THIS GREEN -- two stacked pieces of ``marshal v2``:

  1. ``MarshalExternalStructs.cpp``: accept ``box<ptr|heap<array<value-record>>>``
     as a marshalable member, expanding it into ONE per-record-FIELD SoA leaf
     each (``primal_normal_cell_v1`` / ``_v2``) so the call-site leaves line up
     with the ``bind_c_shim``'s existing ``_emit_value_record_array`` per-field
     scatter (which emits one C slot per record field, NOT one per member).

  2. The inner velocity SDFG must declare the SAME ``t_patch`` as the outer
     (identical leaf set) or the per-member-SoA ABI won't align:
     ``velocity_full.f90`` uses a PRUNED ``t_patch`` (no ``primal_normal_cell``),
     so pairing it with the full-type ``solve_nh`` mismatches.  Build the inner
     velocity from a matching full-``t_patch`` source (e.g. extract velocity via
     the atmo harness) once (1) lands.

When both land, delete the ``xfail`` and the assertion below runs for real.  The
bit-exact assertion is NOT weakened here.
"""
import shutil
from pathlib import Path

import pytest

from _util import have_flang
from dace_fortran.build import make_builder
from dace_fortran.external import Arg, apply_external_functions, clear_external_registry, keep_external
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

# The velocity callback registration: five derived types cross per-member SoA
# (matching the inner SDFG's bind_c_shim), then three rank-3 arrays and the
# scalars.  Same shape as scripts/build_icon_dace_libs.py's dycore wrapper.
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

# Halo / sync / diagnostics dropped so no MPI survives in the outer SDFG and the
# velocity callback is the only external.  Same do_not_emit set as the atmo
# "external" harness config plus the inlined-mode side effects.
_DO_NOT_EMIT = [
    "sync_patch_array", "sync_patch_array_mult", "exchange_data", "p_barrier", "p_max", "p_min", "p_sum", "global_max",
    "global_min", "global_sum", "setup_comm_pattern", "finish", "message", "message_text", "warning", "print_status",
    "print_value", "init_logger", "dbg_print", "work_mpi_barrier", "timer_start", "timer_stop", "new_timer",
    "delete_timer", "check_patch_array_3d_dp"
]


@pytest.mark.xdist_group("atmo_solve_nh_callback")
@pytest.mark.xfail(strict=True,
                   raises=ValueError,
                   reason="hlfir-marshal-external-structs v2 not yet landed: "
                   "t_patch's primal_normal_cell/dual_normal_cell (allocatable array of the value record "
                   "t_tangent_vectors) is not marshalable, so the velocity callback's 5th aos arg has no "
                   "group (see module docstring).")
def test_solve_nh_velocity_callback_outer_sdfg_builds(tmp_path: Path):
    """The OUTER solve_nh SDFG with velocity as a per-member-SoA callback builds.

    This is the immediate gate for the velocity-callback differential e2e: it
    fails today in ``emit_call`` (the marshalling boundary above).  When
    ``hlfir-marshal-external-structs`` v2 lands (value-record-array members), this
    XPASSes -- the strict xfail then flags that the deep-copy-all differential
    (DUT outer SDFG vs REF stock solve_nh, ``mo_solve_nh_diff`` bit-exact compare
    over prog(nnew) + full diag + prep_adv) should be wired up to run for real.
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
        # Reached only when marshal v2 lands.  Pin the external velocity call is
        # emitted (a library node), not inlined / dropped.
        from dace.sdfg import nodes as dnodes
        ext = [
            n for n, _ in sdfg.all_nodes_recursive()
            if isinstance(n, dnodes.LibraryNode) and "velocity" in getattr(n, "label", "").lower()
        ]
        assert ext, "outer SDFG built but emitted no external velocity_tendencies call"
    finally:
        clear_external_registry()
