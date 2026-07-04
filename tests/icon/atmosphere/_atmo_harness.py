"""ICON atmosphere ``solve_nonhydro`` ``input -> single TU`` extraction config.

The solver extracts in two halo modes (see :mod:`icon._halo_modes`):
``external`` black-boxes ``sync_patch_array`` / ``exchange_data``; ``inlined``
inlines the halo and devirtualises the ``t_comm_pattern`` dispatch, leaving only
the MPI point-to-point.  The single concrete arm comes for free under the
standard CPU defines: ``t_comm_pattern_yaxt`` lives behind ``#ifdef HAVE_YAXT``
(undefined), so the cpp pre-pass leaves only ``t_comm_pattern_orig`` for the
default monomorphisation pass to retype to.

Non-halo externals (both modes): the inner ``velocity_tendencies`` kernel
(separately bound), the MPI collectives, the comm-pattern construction, and
terminal I/O / timers.  Slow (~140k-line closure) and memory-heavy, so the
extraction runs in a memory-capped subprocess.
"""
import os
import shutil
from pathlib import Path

from dace_fortran.external_functions import ExternalFunction
from dace_fortran.flang_codebase import find_openmpi_include

from icon._halo_modes import halo_config

_HERE = Path(__file__).resolve().parent
#: ``tests/icon/full/icon-model`` holds the pinned ICON checkout (shared with the
#: ocean + velocity tests).  ``ICON_SRC`` overrides it.
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE.parent / "full" / "icon-model")))
SRC = _ICON_SRC / "src"

HAVE_FLANG = shutil.which("flang-new-21") is not None
HAVE_OPENMPI = find_openmpi_include() is not None


def have_icon_atmo() -> bool:
    """True when every atmosphere kernel source referenced by :data:`KERNELS`
    is checked out."""
    return all((SRC / source).is_file() for _, source, *_ in KERNELS)


def atmo_search_dirs() -> list:
    """USE-graph closure roots for the atmosphere solver -- the same set the
    velocity / ocean tests bisected to (ICON's ``src`` plus the bundled
    external library trees)."""
    return [
        SRC,
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: ICON's standard CPU build defines for the atmosphere solver object.  The
#: ocean / waves / testbed components are compiled out (``__NO_ICON_OCEAN__`` &c.)
#: and -- crucially for the single-arm halo -- ``HAVE_YAXT`` is NOT defined, so
#: the cpp pre-pass strips ``t_comm_pattern_yaxt`` and leaves a single concrete
#: comm-pattern arm.  Same set as ``tests/icon/full/test_dycore_from_icon_source``.
ATMO_DEFINES = [
    "HAVE_CDI_GRIB2",
    "HAVE_FC_ATTRIBUTE_CONTIGUOUS",
    "ICON_MPI_SUBVERSION=1",
    "ICON_MPI_VERSION=3",
    "__HAVE_QUAD_PRECISION",
    "__ICON__",
    "__LOOP_EXCHANGE",
    "__NO_ICON_COMIN__",
    "__NO_ICON_OCEAN__",
    "__NO_ICON_TESTBED__",
    "__NO_ICON_WAVES__",
    "__NO_JSBACH_HD__",
    "__NO_JSBACH__",
    "__NO_QUINCY__",
    "__NO_RAGNAROK__",
    "NO_MPI_CHOICE_ARG",
]

#: NON-HALO external leaves of the standalone dycore kernel: the inner velocity
#: kernel, the MPI collectives / reductions, and the comm-pattern construction.
#: The HALO externals are added per mode (:func:`atmo_externals`) -- "external"
#: black-boxes ``sync_patch_array`` / ``exchange_data``, "inlined" leaves only the
#: MPI point-to-point.  Both modes must extract to a compiling TU.
ATMO_BASE_EXTERNAL_FUNCTIONS = [
    ExternalFunction("velocity_tendencies"),  # the inner kernel; separately bound at link time
    ExternalFunction("setup_comm_pattern"),  # comm-pattern INIT (the construction boundary, a marshalled input)
]

#: DON'T-EMIT (externalised + the bridge DROPs the call): pure side-effects with
#: no numerics -- terminal I/O (debug / error / log) and profiling timers.
ATMO_DO_NOT_EMIT = [
    "finish",
    "message",
    "message_text",
    "warning",
    "print_status",
    "print_value",
    "init_logger",
    "dbg_print",
    "work_mpi_barrier",
    "timer_start",
    "timer_stop",
    "new_timer",
    "delete_timer",
    # The halo DEBUG sync check: a redundant reference exchange + comparison gated
    # by ``IF (p_test_run .AND. do_sync_checks)``.  Pure debug (no numerics; its
    # body is multi-block with ``finish`` + MPI test-mode collectives), and it is
    # called with absent optional fields in the multi-field sync -- stub it so the
    # ``sync_patch_array_mult`` specialization does not drag it in.
    "check_patch_array_3d_dp",
]

#: Non-halo LOGICAL config queries stubbed to ``.FALSE.`` (none for atmosphere;
#: the halo's ``my_process_is_mpi_seq`` is added per mode by :func:`atmo_config`).
ATMO_BASE_RETURN_FALSE: list = []

#: ``typename -> [component names]`` derived-type members that the INNER
#: ``velocity_tendencies`` kernel reads but ``solve_nh`` does NOT, so the closure
#: pruning would drop them from ``solve_nh``'s single-TU struct types.  Keeping
#: them (``inline_to_single_tu(keep_type_components=...)``) makes the extracted
#: ``t_patch`` / ``t_int_state`` / ``t_nh_diag`` / ``t_nh_metrics`` carry the
#: UNION of members both kernels touch -- so when ``solve_nh`` calls
#: ``velocity_tendencies`` as a per-member-SoA ``keep_external`` callback, the
#: outer marshal-expansion leaf set equals the inner ``bind_c_shim`` slot set
#: member-for-member (both in source declaration order).  The kept members are
#: pass-through inputs on the ``solve_nh`` side (received from its caller and
#: forwarded to velocity untouched).  Curated data (not code): the members
#: ``velocity_tendencies`` reads minus the members ``solve_nh`` reads.
ATMO_VELOCITY_UNION_COMPONENTS = {
    # ``t_patch`` top level: velocity reads ``p_patch % nshift`` (the singular
    # field), which ``solve_nh`` does NOT (it uses ``nshift_total`` /
    # ``nshift_child``).  All three co-exist in the real ``mo_model_domain``
    # ``t_patch``; keep ``nshift`` so the union ``t_patch`` carries all three and
    # matches the velocity-side TU member-for-member.
    "t_patch": ["nshift"],
    "t_grid_edges": ["area_edge", "f_e", "fn_e", "ft_e"],
    "t_grid_cells": ["area", "decomp_info"],
    # ``decomp_info`` is itself a ``t_grid_domain_decomp_info`` record; velocity
    # reads its ``owner_mask``.  Keep that nested member too -- otherwise the
    # type prunes to an EMPTY record, which the marshaller rejects (an empty
    # nested record is not inline-flat), breaking the whole ``t_patch`` group.
    "t_grid_domain_decomp_info": ["owner_mask"],
    "t_grid_vertices": ["edge_idx", "edge_blk"],
    "t_int_state": ["geofac_rot", "geofac_n2s"],
    "t_nh_diag": ["max_vcfl_dyn"],
    "t_nh_metrics":
    ["coeff_gradekin", "coeff1_dwdz", "coeff2_dwdz", "deepatmo_gradh_ifc", "deepatmo_invr_mc", "deepatmo_invr_ifc"],
}

#: The MIRROR of :data:`ATMO_VELOCITY_UNION_COMPONENTS`: derived-type members
#: ``solve_nh`` reads but ``velocity_tendencies`` does NOT.  Passed as
#: ``keep_type_components`` when extracting the INNER ``velocity_tendencies``
#: single-TU (same ``mo_model_domain`` source), so velocity's ``t_patch`` /
#: ``t_int_state`` / ``t_nh_diag`` / ``t_nh_metrics`` / ``t_nh_prog`` /
#: ``t_grid_edges`` carry the IDENTICAL union of members as the ``solve_nh`` TU
#: -- byte-for-byte member set + declaration order.  That makes the outer
#: (``solve_nh``) marshal-expansion leaf sequence equal the inner
#: (``velocity_tendencies``) ``bind_c_shim`` slot sequence member-for-member, so
#: the per-member-SoA callback C ABI lines up.  ``comm_pat_c`` / ``comm_pat_e``
#: are OMITTED: they are pointer-to-record HANDLES the marshaller + shim both
#: skip (no SoA leaf), so keeping them on only one side does not desync the ABI,
#: and dropping them avoids dragging the polymorphic comm-pattern arm into
#: velocity's closure.
ATMO_SOLVE_NH_UNION_COMPONENTS = {
    "t_patch": ["geometry_info", "n_childdom", "nshift_total", "nshift_child"],
    "t_grid_edges": ["primal_normal_cell", "dual_normal_cell", "refin_ctrl"],
    # Nested record FIELDS that produce marshal leaves and so must exist on BOTH
    # sides.  ``solve_nh`` reads ``edges % primal_normal_cell(..) % v1`` and
    # ``patch % geometry_info % mean_cell_area``; velocity never touches those
    # scalar fields, so pruning would empty ``t_tangent_vectors`` /
    # ``t_grid_geometry_info`` on the velocity side -- leaving ``primal_normal_cell``
    # an array of a ZERO-field record (0 leaves) against ``solve_nh``'s ``_v1`` /
    # ``_v2`` (2 leaves), a per-member-SoA desync.  Keeping the fields makes both
    # record types structurally identical, so both walks emit the same leaves.
    "t_tangent_vectors": ["v1", "v2"],
    "t_grid_geometry_info": ["mean_cell_area"],
    "t_int_state": ["e_flx_avg", "geofac_div", "geofac_grg", "pos_on_tplane_e", "nudgecoeff_e"],
    "t_nh_prog": ["rho", "exner", "theta_v"],
    # The tail ``ddt_vn_{dmp,adv,cor,pgr,phd,iau,ray,grf}`` (+ ``_is_associated``
    # flags) are the per-process vn tendency contributions solve_nh SUMS;
    # velocity_tendencies writes only the ``_pc`` predictor/corrector arrays, so
    # pruning drops them on the velocity side -- they are kept so the marshalled
    # t_nh_diag leaf set matches solve_nh's member-for-member.
    "t_nh_diag": [
        "exner_pr", "mass_fl_e", "rho_ic", "theta_v_ic", "grf_tend_vn", "grf_tend_w", "grf_tend_rho", "grf_tend_mflx",
        "grf_bdy_mflx", "grf_tend_thv", "vn_ie_int", "w_int", "w_ubc", "theta_v_ic_int", "theta_v_ic_ubc", "rho_ic_int",
        "rho_ic_ubc", "mflx_ic_int", "mflx_ic_ubc", "vn_incr", "exner_incr", "rho_incr", "ddt_exner_phy", "ddt_vn_phy",
        "exner_dyn_incr", "mass_fl_e_sv", "ddt_vn_dyn", "ddt_vn_dyn_is_associated", "ddt_vn_dmp", "ddt_vn_adv",
        "ddt_vn_cor", "ddt_vn_pgr", "ddt_vn_phd", "ddt_vn_iau", "ddt_vn_ray", "ddt_vn_grf", "ddt_vn_dmp_is_associated",
        "ddt_vn_pgr_is_associated", "ddt_vn_phd_is_associated", "ddt_vn_iau_is_associated", "ddt_vn_ray_is_associated",
        "ddt_vn_grf_is_associated"
    ],
    "t_nh_metrics": [
        "rayleigh_w", "rayleigh_vn", "scalfac_dd3d", "hmask_dd3d", "vwind_expl_wgt", "vwind_impl_wgt",
        "inv_ddqz_z_full", "wgtfacq_c", "wgtfacq1_c", "zdiff_gradp", "coeff_gradp", "exner_exfac", "theta_ref_mc",
        "theta_ref_me", "theta_ref_ic", "exner_ref_mc", "rho_ref_mc", "rho_ref_me", "d_exner_dz_ref_ic",
        "d2dexdz2_fac1_mc", "d2dexdz2_fac2_mc", "pg_exdist", "vertidx_gradp", "pg_edgeidx", "pg_edgeblk", "pg_vertidx",
        "bdy_halo_c_idx", "bdy_halo_c_blk", "bdy_mflx_e_idx", "bdy_mflx_e_blk", "deepatmo_divh_mc", "deepatmo_divzu_mc",
        "deepatmo_divzl_mc", "pg_listdim", "bdy_halo_c_dim", "bdy_mflx_e_dim", "mask_prog_halo_c"
    ],
}


def atmo_config(halo_mode: str, entry: str = "") -> dict:
    """Full atmosphere extraction config for the given halo mode (see
    :mod:`icon._halo_modes`): the non-halo base externals merged with the
    mode-specific halo pieces.

    ``keep_type_components`` (the shared-union machinery) is applied only in
    ``inlined`` mode -- the mode the ``solve_nh`` + ``velocity_tendencies``
    per-member-SoA callback e2e drives; the ``external`` TU black-boxes the halo
    and hosts no callback, so it needs no pass-through members.  Which union is
    kept depends on the ENTRY being extracted:

      * the OUTER ``solve_nh`` keeps :data:`ATMO_VELOCITY_UNION_COMPONENTS` (the
        members velocity reads but solve_nh doesn't),
      * the INNER ``velocity_tendencies`` keeps the MIRROR
        :data:`ATMO_SOLVE_NH_UNION_COMPONENTS`,

    so both single-TUs carry the IDENTICAL union of struct members from the same
    ``mo_model_domain`` -- the precondition for the callback ABI to align
    member-for-member."""
    h = halo_config(halo_mode)
    keep = None
    if halo_mode == "inlined":
        keep = (ATMO_SOLVE_NH_UNION_COMPONENTS if "velocity_tendencies" in entry else ATMO_VELOCITY_UNION_COMPONENTS)
    return dict(
        external_functions=ATMO_BASE_EXTERNAL_FUNCTIONS + h["external_functions"],
        force_include=h["force_include"],
        rename_specifics=h["rename_specifics"],
        make_return_false=ATMO_BASE_RETURN_FALSE + h["return_false"],
        do_not_emit=ATMO_DO_NOT_EMIT,
        defines=ATMO_DEFINES,
        extra_sources=h["extra_sources"],
        specialize_at_source=h["specialize_at_source"],
        keep_type_components=keep,
    )


#: The atmosphere kernels extracted.  Each entry is
#: ``(key, source-relative-to-src, module::procedure, body-line-count)``.
#: ``velocity_advection`` is the INNER ``velocity_tendencies`` kernel the
#: ``solve_nh`` dycore calls as a per-member-SoA ``keep_external`` callback;
#: extracted from the SAME ``mo_model_domain`` closure so its struct types match
#: ``solve_nh``'s union (see :data:`ATMO_SOLVE_NH_UNION_COMPONENTS`).
KERNELS = [
    ("solve_nonhydro", "atm_dyn_iconam/mo_solve_nonhydro.f90", "mo_solve_nonhydro::solve_nh", 0),
    ("velocity_advection", "atm_dyn_iconam/mo_velocity_advection.f90", "mo_velocity_advection::velocity_tendencies", 0),
]

#: Checked-in single-TU artifacts, one per (kernel, halo mode):
#: ``(key, halo_mode, filename, module::procedure)``.  ``velocity_advection`` is
#: extracted in ``inlined`` mode only (the callback-inner shape).
SINGLE_TU_ARTIFACTS = [
    ("solve_nonhydro", "inlined", "solve_nonhydro_inlined_single_tu.f90", "mo_solve_nonhydro::solve_nh"),
    ("solve_nonhydro", "external", "solve_nonhydro_external_single_tu.f90", "mo_solve_nonhydro::solve_nh"),
    ("velocity_advection", "inlined", "velocity_advection_inlined_single_tu.f90",
     "mo_velocity_advection::velocity_tendencies"),
]

_EXTRACT_SCRIPT = _HERE / "_extract_single_tu.py"


def extract_single_tu(source_relpath: str,
                      entry: str,
                      out_dir: Path,
                      halo_mode: str = "inlined",
                      mem_gb: float = 12.0) -> dict:
    """Extract one atmosphere kernel into a single, gfortran-compiling ``.f90`` in
    a memory-capped subprocess (the fparser parse of the merged closure peaks near
    9 GB, so it must not OOM the host) and return a result dict with keys
    ``passed`` (bool), ``tu_path`` (str|None), ``tu_lines`` (int|None) and
    ``output`` (str).

    The subprocess writes all artifacts under ``out_dir`` and uses it as
    ``TMPDIR`` too, keeping the large merged file off the RAM-backed ``/tmp``
    tmpfs.  ``PYTHONHASHSEED`` is pinned so the inliner's regeneration is
    byte-reproducible (the drift guard asserts the extracted TU is identical to
    the committed one)."""
    import subprocess
    import sys

    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    tests_root = str(_HERE.parent.parent)
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([tests_root, prev_pp]) if prev_pp else tests_root
    env["TMPDIR"] = str(out_dir)
    env.setdefault("UCX_VFS_ENABLE", "n")
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable,
         str(_EXTRACT_SCRIPT), source_relpath, entry,
         str(out_dir),
         str(mem_gb), halo_mode],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(out_dir))
    tu_path, tu_lines = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("TU_PATH:"):
            tu_path = line.split(":", 1)[1].strip()
        elif line.startswith("TU_LINES:"):
            tu_lines = int(line.split(":", 1)[1])
    passed = any(line.startswith("RESULT: PASS") for line in proc.stdout.splitlines())
    return {"passed": passed, "tu_path": tu_path, "tu_lines": tu_lines, "output": proc.stdout + "\n" + proc.stderr}
