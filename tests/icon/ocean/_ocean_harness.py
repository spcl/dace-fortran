"""Shared configuration for the ICON-O (ocean) ``input -> single TU`` extraction.

Extracts a numerically critical ocean kernel into a self-contained, compiling
single translation unit (lowering the TU to an SDFG is handled elsewhere).

Route: merge the USE closure (regex, no mpi/netcdf stubs) -> fparser
``inline_to_single_tu`` (cpp pre-pass, CONTIGUOUS strip, external-USE tolerance,
namelist pruning) -> gfortran ``-fsyntax-only``.

Ocean is NOT compiled out (atmosphere's ``__NO_ICON_OCEAN__`` intentionally
dropped) and slow (~137k-line closure), so gated on flang + icon-model and run
in a memory-capped subprocess.
"""
import os
import subprocess
import sys
from pathlib import Path

from dace_fortran.external_functions import ExternalFunction
from dace_fortran.flang_codebase import find_openmpi_include

from icon._halo_modes import halo_config

import shutil

_HERE = Path(__file__).resolve().parent
#: ``tests/icon/full/icon-model`` holds the pinned ICON checkout (shared with
#: the atmosphere velocity test).  ``ICON_SRC`` overrides it.
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE.parent / "full" / "icon-model")))
SRC = _ICON_SRC / "src"

HAVE_FLANG = shutil.which("flang-new-21") is not None
HAVE_OPENMPI = find_openmpi_include() is not None


def have_icon_ocean() -> bool:
    """True when EVERY ocean kernel source in :data:`KERNELS` is checked out (a partial
    checkout must skip, not fail, the missing kernel's extraction)."""
    return all((SRC / source).is_file() for _, source, *_ in KERNELS)


def ocean_search_dirs() -> list:
    """USE-graph closure roots for the ocean kernels: ICON's ``src`` (covers
    ``src/ocean``) plus the bundled external library trees."""
    return [
        SRC,
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: ICON's standard CPU build defines with the ocean component ENABLED
#: (``__NO_ICON_OCEAN__`` intentionally dropped); select the ``#ifdef`` arms during cpp.
OCEAN_DEFINES = [
    "HAVE_CDI_GRIB2",
    "HAVE_FC_ATTRIBUTE_CONTIGUOUS",
    "ICON_MPI_SUBVERSION=1",
    "ICON_MPI_VERSION=3",
    "__HAVE_QUAD_PRECISION",
    "__ICON__",
    "__LOOP_EXCHANGE",
    "__NO_ICON_COMIN__",
    "__NO_ICON_TESTBED__",
    "__NO_ICON_WAVES__",
    "__NO_JSBACH_HD__",
    "__NO_JSBACH__",
    "__NO_QUINCY__",
    "__NO_RAGNAROK__",
    "NO_MPI_CHOICE_ARG",
]

#: NON-HALO ocean externals (don't-inline; bridge emits an external call).  HALO
#: subset added per mode by :func:`ocean_config` -- "external" black-boxes it,
#: "inlined" devirtualises leaving only MPI point-to-point.  The solver subsystem
#: (``ocean_solve_*``/``*_construct``) uses virtual dispatch neither the inliner
#: nor flang can statically lower, so it stays one opaque black box (pinned by
#: ``tests/hlfir_devirtualization_test.py``).
OCEAN_BASE_EXTERNAL_FUNCTIONS = [
    ExternalFunction("setup_comm_pattern"),  # MPI comm-pattern INIT: pure comm-topology setup, no numerics
    ExternalFunction("ocean_solve_construct"),  # runtime factory (ALLOCATE+dispatch); ONCE (is_init guard)
    ExternalFunction("trivial_transfer_construct"),  # transfer-object construct; ONCE (is_init guard)
    ExternalFunction("subset_transfer_construct"),  # raw MPI comm-topology INIT; pure comm init, no numerics
    ExternalFunction("lhs_primal_flip_flop_construct"),  # LHS re-init; PER LEVEL (static bind)
    ExternalFunction("ocean_solve_solve"),  # linear solve; PER LEVEL (dispatches act/lhs/trans on abstract bases)
]
#: DON'T-EMIT = externalised + dropped: pure side-effects, no numerics -- I/O + timers.
OCEAN_DO_NOT_EMIT = [
    "work_mpi_barrier",  # timer-gated MPI barrier -- pure profiling sync, no numerics
    "dbg_print",  # terminal write (debug print)
    "finish",
    "message",
    "warning",  # terminal write (error / log)
    "timer_start",
    "timer_stop",
    "new_timer",
    "delete_timer",  # timers
]
#: LOGICAL config queries stubbed ``.FALSE.`` (not inlined).  The dycore's
#: first-timestep guard is `timestep==1 .AND. .NOT.(isRestart() .OR.
#: isInitFromRestart())`; for the standalone (no-restart) extraction both are
#: false, so the guard reduces to `timestep==1` without dragging in restart
#: bookkeeping (mo_master_config/mo_impl_constants).
OCEAN_BASE_RETURN_FALSE = [
    "isrestart",
    "isinitfromrestart",
]


def ocean_config(halo_mode: str) -> dict:
    """Full ocean extraction config for ``halo_mode`` (:mod:`icon._halo_modes`): non-halo
    base externals merged with the mode-specific halo pieces."""
    h = halo_config(halo_mode)
    return dict(
        external_functions=OCEAN_BASE_EXTERNAL_FUNCTIONS + h["external_functions"],
        force_include=h["force_include"],
        rename_specifics=h["rename_specifics"],
        make_return_false=OCEAN_BASE_RETURN_FALSE + h["return_false"],
        do_not_emit=OCEAN_DO_NOT_EMIT,
        defines=OCEAN_DEFINES,
        extra_sources=h["extra_sources"],
    )


# NOTE: rot_vertex_ocean_3d is INLINED (pure vorticity compute, no MPI in its body).
# Pulls in the host module's USE closure (mo_mpi reductions, t_comm_pattern CLASS(*)),
# but the inliner's external-USE tolerance prunes that unreachable MPI baggage.

#: The ICON-O kernels currently extracted.  Each entry is
#: ``(key, source-relative-to-src, module::procedure, body-line-count)``.
KERNELS = [
    ("ppm_vflux", "ocean/tracer_transport/mo_ocean_tracer_transport_vert.f90",
     "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock", 339),
    ("coriolis_pv", "ocean/math/mo_scalar_product.f90", "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar", 273),
    # Ocean horizontal velocity advection (distinct from ICON atmosphere's
    # velocity_tendencies).  Rotational form: vorticity flux + kinetic-energy grad.
    ("ocean_veloc_adv", "ocean/dynamics/mo_ocean_velocity_advection.f90",
     "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", 102),
    # The free-surface surface-pressure solver driver: dynamical-core keystone.
    # Its ~137k-line closure exercises the full external policy (MPI halo, comm-pattern
    # INIT, terminal IO/timers, restart queries) and inlines everything else.
    ("solve_free_sfc", "ocean/dynamics/mo_ocean_ab_timestepping_mimetic.f90",
     "mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic", 191),
]

#: Checked-in single-TU artifacts: ``(key, halo_mode, filename, module::procedure)``.
#: Generated by extraction and committed for a stable SDFG-lowering input; the
#: extraction test regenerates and checks for drift.  Non-solver kernels extract in
#: ``external`` mode only; the free-surface SOLVER is tested in BOTH modes.
SINGLE_TU_ARTIFACTS = [
    ("ppm_vflux", "external", "ppm_vflux_single_tu.f90", "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock"),
    ("coriolis_pv", "external", "coriolis_pv_single_tu.f90", "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar"),
    ("ocean_veloc_adv", "external", "ocean_veloc_adv_single_tu.f90",
     "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot"),
    ("solve_free_sfc", "external", "solve_free_sfc_single_tu.f90",
     "mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic"),
    ("solve_free_sfc", "inlined", "solve_free_sfc_inlined_single_tu.f90",
     "mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic"),
]

_EXTRACT_SCRIPT = _HERE / "_extract_single_tu.py"


def extract_single_tu(source_relpath: str,
                      entry: str,
                      out_dir: Path,
                      halo_mode: str = "external",
                      mem_gb: float = 10.0) -> dict:
    """Extract one ocean kernel into a single, gfortran-compiling ``.f90`` in a
    memory-capped subprocess (fparser parse peaks near 9 GB).  Returns a dict with
    ``passed``/``tu_path``/``tu_lines``/``output``.

    Subprocess uses ``out_dir`` as ``TMPDIR`` too (keeps the merged file off the
    RAM-backed ``/tmp`` tmpfs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    tests_root = str(_HERE.parent.parent)
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([tests_root, prev_pp]) if prev_pp else tests_root
    env["TMPDIR"] = str(out_dir)
    env.setdefault("UCX_VFS_ENABLE", "n")
    # Pin the hash seed for byte-reproducible regeneration -- the drift guard compares
    # against the committed TU, and set/dict-iteration order leaking into names would flake it.
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
    out = proc.stdout + "\n" + proc.stderr
    tu_path, tu_lines = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("TU_PATH:"):
            tu_path = line.split(":", 1)[1].strip()
        elif line.startswith("TU_LINES:"):
            tu_lines = int(line.split(":", 1)[1])
    passed = any(line.startswith("RESULT: PASS") for line in proc.stdout.splitlines())
    return {"passed": passed, "tu_path": tu_path, "tu_lines": tu_lines, "output": out}
