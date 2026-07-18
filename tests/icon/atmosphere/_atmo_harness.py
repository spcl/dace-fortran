"""ICON atmosphere ``solve_nonhydro`` ``input -> single TU`` extraction config.

Two halo modes (:mod:`icon._halo_modes`): ``external`` black-boxes
``sync_patch_array``/``exchange_data``; ``inlined`` inlines the halo and
devirtualises ``t_comm_pattern``, leaving only MPI point-to-point.  Single
concrete arm comes free under standard CPU defines (``HAVE_YAXT`` undefined ->
cpp strips ``t_comm_pattern_yaxt``, leaving only ``t_comm_pattern_orig``).

Non-halo externals (both modes): the inner ``velocity_tendencies`` kernel, MPI
collectives, comm-pattern construction, terminal I/O/timers.  Slow (~140k-line
closure) and memory-heavy -- extraction runs in a memory-capped subprocess.
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
    """True when every atmosphere kernel source in :data:`KERNELS` is checked out."""
    return all((SRC / source).is_file() for _, source, *_ in KERNELS)


def atmo_search_dirs() -> list:
    """USE-graph closure roots for the atmosphere solver (ICON's ``src`` + bundled
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


#: ICON's standard CPU build defines for the atmosphere solver.  Ocean/waves/testbed
#: compiled out; ``HAVE_YAXT`` NOT defined so cpp strips ``t_comm_pattern_yaxt``,
#: leaving a single concrete comm-pattern arm.  Same set as
#: ``tests/icon/full/test_dycore_from_icon_source``.
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

#: NON-HALO external leaves: the inner velocity kernel, MPI collectives/reductions,
#: comm-pattern construction.  HALO externals added per mode (:func:`atmo_externals`);
#: both modes must extract to a compiling TU.
ATMO_BASE_EXTERNAL_FUNCTIONS = [
    ExternalFunction("velocity_tendencies"),  # the inner kernel; separately bound at link time
    ExternalFunction("setup_comm_pattern"),  # comm-pattern INIT (the construction boundary, a marshalled input)
]

#: DON'T-EMIT (externalised + dropped): pure side-effects, no numerics -- I/O + timers.
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
    # Halo DEBUG sync check (gated by p_test_run .AND. do_sync_checks); pure debug, no
    # numerics.  Stubbed so sync_patch_array_mult specialization doesn't drag it in.
    "check_patch_array_3d_dp",
]

#: Non-halo LOGICAL queries stubbed ``.FALSE.`` (none for atmosphere; halo adds its
#: own per mode via :func:`atmo_config`).
ATMO_BASE_RETURN_FALSE: list = []

#: ``typename -> [component names]`` derived-type members ``velocity_tendencies``
#: reads but ``solve_nh`` does NOT -- kept (``keep_type_components``) so both
#: kernels' extracted structs carry the UNION of members, keeping the outer
#: marshal-expansion leaf set member-for-member aligned with the inner
#: ``bind_c_shim`` slot set (both in source declaration order).  Curated data:
#: members velocity_tendencies reads minus members solve_nh reads.
ATMO_VELOCITY_UNION_COMPONENTS = {
    # velocity reads p_patch%nshift; solve_nh uses nshift_total/nshift_child instead --
    # keep nshift so the union t_patch matches velocity-side member-for-member.
    "t_patch": ["nshift"],
    "t_grid_edges": ["area_edge", "f_e", "fn_e", "ft_e"],
    "t_grid_cells": ["area", "decomp_info"],
    # decomp_info's nested owner_mask must be kept too, else it prunes to an EMPTY
    # record, which the marshaller rejects (breaking the whole t_patch group).
    "t_grid_domain_decomp_info": ["owner_mask"],
    "t_grid_vertices": ["edge_idx", "edge_blk"],
    "t_int_state": ["geofac_rot", "geofac_n2s"],
    "t_nh_diag": ["max_vcfl_dyn"],
    "t_nh_metrics":
    ["coeff_gradekin", "coeff1_dwdz", "coeff2_dwdz", "deepatmo_gradh_ifc", "deepatmo_invr_mc", "deepatmo_invr_ifc"],
}

#: MIRROR of :data:`ATMO_VELOCITY_UNION_COMPONENTS`: members ``solve_nh`` reads but
#: ``velocity_tendencies`` does NOT.  Kept when extracting the INNER TU so both
#: sides carry the IDENTICAL member union, aligning the outer marshal-expansion
#: leaf sequence with the inner ``bind_c_shim`` slot sequence member-for-member.
#: ``comm_pat_c``/``comm_pat_e`` OMITTED: pointer-to-record HANDLES with no SoA
#: leaf on either side, so keeping them would only drag in the polymorphic arm.
ATMO_SOLVE_NH_UNION_COMPONENTS = {
    "t_patch": ["geometry_info", "n_childdom", "nshift_total", "nshift_child"],
    "t_grid_edges": ["primal_normal_cell", "dual_normal_cell", "refin_ctrl"],
    # Nested record FIELDS must exist on BOTH sides: velocity never touches these
    # scalar fields, so pruning would leave a ZERO-field record on the velocity side
    # vs solve_nh's 2-leaf record -- a per-member-SoA desync.
    "t_tangent_vectors": ["v1", "v2"],
    "t_grid_geometry_info": ["mean_cell_area"],
    "t_int_state": ["e_flx_avg", "geofac_div", "geofac_grg", "pos_on_tplane_e", "nudgecoeff_e"],
    "t_nh_prog": ["rho", "exner", "theta_v"],
    # ddt_vn_{dmp,adv,cor,pgr,phd,iau,ray,grf} are per-process contributions solve_nh
    # SUMS; velocity writes only the _pc arrays -- kept for member-for-member match.
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
    """Full atmosphere extraction config for ``halo_mode`` (see :mod:`icon._halo_modes`):
    non-halo base externals merged with the mode-specific halo pieces.

    ``keep_type_components`` applies only in ``inlined`` mode (the callback e2e);
    OUTER ``solve_nh`` keeps :data:`ATMO_VELOCITY_UNION_COMPONENTS`, INNER
    ``velocity_tendencies`` keeps the MIRROR :data:`ATMO_SOLVE_NH_UNION_COMPONENTS`,
    so both single-TUs carry the identical struct-member union the callback ABI needs."""
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


#: Atmosphere kernels extracted: ``(key, source-relative-to-src, module::procedure,
#: body-line-count)``.  ``velocity_advection`` is the INNER kernel ``solve_nh`` calls
#: as a per-member-SoA callback; matches its struct union
#: (:data:`ATMO_SOLVE_NH_UNION_COMPONENTS`).
KERNELS = [
    ("solve_nonhydro", "atm_dyn_iconam/mo_solve_nonhydro.f90", "mo_solve_nonhydro::solve_nh", 0),
    ("velocity_advection", "atm_dyn_iconam/mo_velocity_advection.f90", "mo_velocity_advection::velocity_tendencies", 0),
]

#: Checked-in single-TU artifacts: ``(key, halo_mode, filename, module::procedure)``.
#: ``velocity_advection`` extracted in ``inlined`` mode only (callback-inner shape).
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
    """Extract one atmosphere kernel into a single, gfortran-compiling ``.f90`` in a
    memory-capped subprocess (fparser parse peaks near 9 GB).  Returns a dict with
    ``passed``/``tu_path``/``tu_lines``/``output``.

    Subprocess uses ``out_dir`` as ``TMPDIR`` too (keeps the merged file off the
    RAM-backed ``/tmp`` tmpfs); ``PYTHONHASHSEED`` pinned for byte-reproducible
    regeneration (the drift guard compares against the committed TU)."""
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
