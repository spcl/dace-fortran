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


def atmo_config(halo_mode: str) -> dict:
    """Full atmosphere extraction config for the given halo mode (see
    :mod:`icon._halo_modes`): the non-halo base externals merged with the
    mode-specific halo pieces.

    The velocity-callback union (:data:`ATMO_VELOCITY_UNION_COMPONENTS`) is
    applied only in ``inlined`` mode -- the mode the ``solve_nh`` +
    ``velocity_tendencies`` per-member-SoA callback e2e drives.  The
    ``external`` TU black-boxes the halo and is not a callback host, so it
    needs no velocity-only pass-through members (keeping it lean also avoids a
    needless re-pin of the external artifact)."""
    h = halo_config(halo_mode)
    return dict(
        external_functions=ATMO_BASE_EXTERNAL_FUNCTIONS + h["external_functions"],
        force_include=h["force_include"],
        rename_specifics=h["rename_specifics"],
        make_return_false=ATMO_BASE_RETURN_FALSE + h["return_false"],
        do_not_emit=ATMO_DO_NOT_EMIT,
        defines=ATMO_DEFINES,
        extra_sources=h["extra_sources"],
        specialize_at_source=h["specialize_at_source"],
        keep_type_components=(ATMO_VELOCITY_UNION_COMPONENTS if halo_mode == "inlined" else None),
    )


#: The atmosphere kernels extracted.  Each entry is
#: ``(key, source-relative-to-src, module::procedure, body-line-count)``.
KERNELS = [
    ("solve_nonhydro", "atm_dyn_iconam/mo_solve_nonhydro.f90", "mo_solve_nonhydro::solve_nh", 0),
]

#: Checked-in single-TU artifacts, one per (kernel, halo mode):
#: ``(key, halo_mode, filename, module::procedure)``.
SINGLE_TU_ARTIFACTS = [
    ("solve_nonhydro", "inlined", "solve_nonhydro_inlined_single_tu.f90", "mo_solve_nonhydro::solve_nh"),
    ("solve_nonhydro", "external", "solve_nonhydro_external_single_tu.f90", "mo_solve_nonhydro::solve_nh"),
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
