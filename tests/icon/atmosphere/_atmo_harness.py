"""Shared configuration for the ICON atmosphere dynamical-core
``input -> single TU`` extraction (``mo_solve_nonhydro::solve_nh``).

This mirrors the ocean harness (:mod:`icon.ocean._ocean_harness`) but for the
atmosphere solver, with one deliberate difference: the halo exchange
(``sync_patch_array`` / ``exchange_data``) is **inlined**, not externalised.

ICON's halo dispatches through the abstract ``t_comm_pattern`` over two arms --
``t_comm_pattern_orig`` and ``t_comm_pattern_yaxt`` -- but ``t_comm_pattern_yaxt``
lives entirely behind ``#ifdef HAVE_YAXT`` (module body, both factory
``ALLOCATE`` arms, the ``TYPE IS`` selector).  The standard CPU build does not
define ``HAVE_YAXT``, so after the cpp pre-pass only ``t_comm_pattern_orig``
remains -- a single concrete arm.  The inliner's default monomorphisation pass
(:func:`dace_fortran.inliner.ast_desugaring.monomorphize_rewrite.monomorphize_auto`)
then retypes ``CLASS(t_comm_pattern)`` to ``TYPE(t_comm_pattern_orig)``, turning
``p_pat%exchange_data_*`` into a static call the inliner inlines -- so the pack
loop lands inline and the only thing left at the halo boundary is the raw MPI
point-to-point (mapped to ``dace.libraries.mpi`` libnodes when the TU is lowered
to an SDFG, exactly as ``tests/sync_devirt_mpi_libnode_test.py`` proves in
miniature).

What stays external is only the genuine leaves: terminal I/O + timers
(``do_not_emit``), the MPI *collectives* (``p_max`` / ``p_min`` / ``p_sum`` /
``p_barrier`` -- global reductions / barriers, no per-cell numerics), the
comm-pattern *construction* (``setup_comm_pattern`` -- pure comm-topology setup
done once at model init, its result a marshalled input), and
``velocity_tendencies`` (the inner kernel, separately bound at link time).

This chat owns the ``input -> single TU`` stage only; lowering the TU to an SDFG
is a separate concern.  The extraction is slow (the merged closure is ~140k
lines) and memory-heavy, so it runs in a memory-capped subprocess.
"""
import os
import shutil
from pathlib import Path

from dace_fortran.external_functions import ExternalFunction
from dace_fortran.flang_codebase import find_openmpi_include

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

#: EXTERNAL (don't-inline, bridge EMITs an external call): the genuine leaves of
#: a standalone dycore kernel.  The halo exchange is deliberately ABSENT -- it is
#: inlined and devirtualised (see the module docstring), unlike the ocean harness
#: which black-boxes it.  What remains is the inner velocity kernel, the MPI
#: collectives, and the comm-pattern construction.
ATMO_EXTERNAL_FUNCTIONS = [
    ExternalFunction("velocity_tendencies"),  # the inner kernel; separately bound at link time
    # MPI point-to-point: the halo-exchange leaves left after the comm pattern is
    # devirtualised and the pack/gather inlined -- "only MPI calls remain".  These
    # are mo_mpi wrappers over mpi_isend/irecv/wait/send/recv.
    ExternalFunction("p_isend"),
    ExternalFunction("p_irecv"),
    ExternalFunction("p_wait"),
    ExternalFunction("p_send"),
    ExternalFunction("p_recv"),
    # MPI collectives / reductions (global, no per-cell numerics).
    ExternalFunction("p_max"),  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_MAX)
    ExternalFunction("p_min"),  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_MIN)
    ExternalFunction("p_sum"),  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_SUM)
    ExternalFunction("p_barrier"),  # MPI collective barrier (mo_mpi wrapper, timer-gated)
    ExternalFunction("global_max"),  # MPI global reduction wrapper
    ExternalFunction("global_min"),  # MPI global reduction wrapper
    ExternalFunction("global_sum"),  # MPI global reduction wrapper
    ExternalFunction("setup_comm_pattern"),  # comm-pattern INIT (pure comm-topology setup, no numerics)
]

#: Concrete comm-pattern arm module to FORCE-INCLUDE in the merge closure.  The
#: abstract ``t_comm_pattern`` is reached from ``solve_nh`` (via the halo
#: wrappers), but its single concrete arm ``t_comm_pattern_orig`` lives in
#: ``mo_communication_orig`` -- reached only through the comm-pattern *factory*,
#: which runs at model init and is externalised.  Without the arm in the
#: closure the monomorphisation pass has nothing to retype to, so we splice the
#: arm module in explicitly; ``t_comm_pattern_yaxt`` stays cpp'd out (no
#: ``HAVE_YAXT``), keeping a single arm.
ATMO_FORCE_INCLUDE_MODULES = [
    "parallel_infrastructure/mo_communication_orig.f90",
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
]

#: Specific-procedure renames to break generic/specific name collisions before
#: externalisation.  ICON's ``mo_mpi`` declares ``INTERFACE p_wait`` whose
#: ``MODULE PROCEDURE`` list includes a specific *also* named ``p_wait`` (the
#: no-argument wait); externalising the generic leaves a dangling ``USE ... =>
#: p_wait``.  Renaming the specific disambiguates it (the generic + call sites
#: stay, dispatching to the renamed specific).
ATMO_RENAME_SPECIFICS = {
    "p_wait": "p_wait_noarg",
}

#: LOGICAL config queries stubbed to ``.FALSE.`` (NOT inlined).  ICON's halo
#: ``exchange_data_*`` bodies branch ``IF (my_process_is_mpi_seq()) THEN <local
#: copy> ELSE <MPI isend/irecv/wait>``; pinning ``my_process_is_mpi_seq`` to
#: ``.FALSE.`` selects the real MPI halo path (the point we want -- "only MPI
#: calls remain"), so the seq local-copy arm folds away.
ATMO_RETURN_FALSE: list = [
    "my_process_is_mpi_seq",
]

#: The atmosphere kernels extracted.  Each entry is
#: ``(key, source-relative-to-src, module::procedure, body-line-count)``.
KERNELS = [
    ("solve_nonhydro", "atm_dyn_iconam/mo_solve_nonhydro.f90", "mo_solve_nonhydro::solve_nh", 0),
]

#: Checked-in single-TU artifacts: ``(key, filename, module::procedure)``.
SINGLE_TU_ARTIFACTS = [
    ("solve_nonhydro", "solve_nonhydro_single_tu.f90", "mo_solve_nonhydro::solve_nh"),
]

_EXTRACT_SCRIPT = _HERE / "_extract_single_tu.py"


def extract_single_tu(source_relpath: str, entry: str, out_dir: Path, mem_gb: float = 12.0) -> dict:
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
        [sys.executable, str(_EXTRACT_SCRIPT), source_relpath, entry,
         str(out_dir), str(mem_gb)],
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
