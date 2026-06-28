"""Shared configuration for the ICON-O (ocean) ``input -> single TU``
extraction.

ICON-O's dynamical core does not run on GPU; the goal is to extract a
numerically critical ocean kernel into a self-contained, compiling single
translation unit that can then (separately) be lowered to a DaCe SDFG.
This chat owns only the *first* stage -- real ICON source to a valid,
gfortran-compiling ``.f90`` checked into this folder.  Lowering the TU to
an SDFG is handled elsewhere.

The extraction route is:

  merge the USE closure (regex, no mpi/netcdf library stubs)
  -> fparser ``inline_to_single_tu`` with the C-preprocessor pre-pass
     (``expand_cpp``), the CONTIGUOUS-attribute strip, external-USE
     tolerance (netcdf / mpi / cdi dropped by pruning), the function-call
     external tolerance, and consistent namelist pruning
  -> gfortran ``-fsyntax-only``

It is enabled (ocean is NOT compiled out: the atmosphere recipe's
``__NO_ICON_OCEAN__`` is intentionally dropped) and is slow (the merged
closure is ~137k lines), so it is gated on flang + the icon-model submodule
and each extraction runs in a memory-capped subprocess.
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
    """True when EVERY ocean kernel source referenced by :data:`KERNELS` is
    checked out (a partial checkout that has one kernel but not another must
    skip, not fail the missing kernel's extraction)."""
    return all((SRC / source).is_file() for _, source, *_ in KERNELS)


def ocean_search_dirs() -> list:
    """USE-graph closure roots for the ocean kernels: ICON's ``src`` (which
    recursively covers ``src/ocean``) plus the external library trees ICON
    bundles -- the same set the atmosphere velocity test bisected to."""
    return [
        SRC,
        _ICON_SRC / "externals/fortran-support/src",
        _ICON_SRC / "externals/mtime/src",
        _ICON_SRC / "externals/iconmath/src",
        _ICON_SRC / "externals/cdi/src",
        _ICON_SRC / "externals/memman/src/bindings/fortran",
        _ICON_SRC / "support",
    ]


#: ICON's standard CPU build defines, with the ocean component ENABLED (the
#: atmosphere recipe's ``__NO_ICON_OCEAN__`` is intentionally dropped so the
#: ocean modules are not preprocessed away).  These select the ``#ifdef`` arms
#: during the cpp pre-pass.
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

#: NON-HALO ocean externals (don't-inline; the bridge emits an external call).
#: The HALO subset (``sync_patch_array`` / ``sync_patch_array_mult`` /
#: ``exchange_data``) is added per mode by :func:`ocean_config` -- "external"
#: black-boxes it (the callback boundary), "inlined" devirtualises it leaving
#: only the MPI point-to-point.  The solver subsystem (``ocean_solve_*`` /
#: ``*_construct``) is built on Fortran virtual dispatch neither the inliner nor
#: flang can statically lower, so it stays one opaque black box (pinned by
#: ``tests/hlfir_devirtualization_test.py``).
OCEAN_BASE_EXTERNAL_FUNCTIONS = [
    ExternalFunction("work_mpi_barrier"),  # MPI collective barrier (mo_mpi: MPI_Barrier)
    ExternalFunction("p_barrier"),  # MPI collective barrier (mo_mpi wrapper, timer-gated)
    ExternalFunction("p_max"),  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_MAX)
    ExternalFunction("p_min"),  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_MIN)
    ExternalFunction("p_sum"),  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_SUM)
    ExternalFunction("setup_comm_pattern"),  # MPI comm-pattern INIT (deferred p_pat%setup on abstract t_comm_pattern,
    #                                          mo_communication_factory): pure comm-topology setup, no numerics
    ExternalFunction("ocean_solve_construct"),  # runtime factory (ALLOCATE+dispatch); ONCE (is_init guard)
    ExternalFunction("trivial_transfer_construct"),  # transfer-object construct; ONCE (is_init guard)
    ExternalFunction("subset_transfer_construct"),  # subset transfer construct = raw MPI comm-topology INIT
    #                                                 (mpi_isend/recv/waitall/comm_split to exchange index ownership);
    #                                                 pure comm init, no numerics -- sibling of trivial_transfer_construct
    ExternalFunction("lhs_primal_flip_flop_construct"
                     ),  # LHS re-init; PER LEVEL (static bind, kept external for a clean boundary)
    ExternalFunction("ocean_solve_solve"),  # the linear solve; PER LEVEL (dispatches act/lhs/trans on abstract bases)
]
#: DON'T-EMIT = externalised (NOT inlined) and the bridge DROPs the call: pure
#: side-effects with no numerics -- terminal I/O (debug / error / log) and timers.
OCEAN_DO_NOT_EMIT = [
    "dbg_print",  # terminal write (debug print)
    "finish",
    "message",
    "warning",  # terminal write (error / log)
    "timer_start",
    "timer_stop",
    "new_timer",
    "delete_timer",  # timers
]
#: LOGICAL config-query functions stubbed to return ``.FALSE.`` (NOT inlined,
#: body replaced with ``<result> = .FALSE.``).  The dycore's first-timestep
#: init guard is `IF (timestep==1 .AND. .NOT.(isRestart() .OR.
#: isInitFromRestart()))`; for the standalone (no-restart) extraction both
#: queries are false, so the guard faithfully reduces to `timestep==1`.
#: Inlining their real bodies instead would drag in mo_master_config's
#: `my_model_do_restart` plus the mo_impl_constants NORMAL_RESTART /
#: INIT_FROM_RESTART parameters (restart bookkeeping, no dycore numerics).
OCEAN_BASE_RETURN_FALSE = [
    "isrestart",
    "isinitfromrestart",
]


def ocean_config(halo_mode: str) -> dict:
    """Full ocean extraction config for the given halo mode (see
    :mod:`icon._halo_modes`): the non-halo base externals merged with the
    mode-specific halo pieces."""
    h = halo_config(halo_mode)
    return dict(
        external_functions=OCEAN_BASE_EXTERNAL_FUNCTIONS + h["external_functions"],
        force_include=h["force_include"],
        rename_specifics=h["rename_specifics"],
        make_return_false=OCEAN_BASE_RETURN_FALSE + h["return_false"],
        do_not_emit=OCEAN_DO_NOT_EMIT,
        defines=OCEAN_DEFINES,
    )


# NOTE: rot_vertex_ocean_3d is INLINED (it is pure vorticity compute, no MPI in
# its body).  Inlining it pulls in its host module's USE closure (mo_mpi
# reductions, t_comm_pattern CLASS(*)), but the inliner's external-USE tolerance
# processes-then-prunes that unreachable MPI baggage, so the kernel still
# extracts to a compiling single TU with the vorticity computed in-line.

#: The ICON-O kernels currently extracted.  Each entry is
#: ``(key, source-relative-to-src, module::procedure, body-line-count)``.
KERNELS = [
    ("ppm_vflux", "ocean/tracer_transport/mo_ocean_tracer_transport_vert.f90",
     "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock", 339),
    ("coriolis_pv", "ocean/math/mo_scalar_product.f90", "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar", 273),
    # Ocean horizontal velocity advection (NOT ICON's atmosphere
    # mo_velocity_advection::velocity_tendencies -- a distinct kernel covered by
    # tests/icon/full).  The rotational ("inUse") form: vorticity flux (inlines
    # nonlinear_coriolis_3d, same compute as coriolis_pv) + kinetic-energy grad.
    ("ocean_veloc_adv", "ocean/dynamics/mo_ocean_velocity_advection.f90",
     "mo_ocean_velocity_advection::veloc_adv_horz_mimetic_rot", 102),
    # The free-surface surface-pressure solver driver: the dynamical-core
    # keystone.  Its ~137k-line closure exercises the full external policy --
    # the MPI halo (sync_patch_array / exchange_data / p_*), MPI comm-pattern
    # INIT (setup_comm_pattern / subset_transfer_construct: pure comm-topology
    # setup, no numerics), terminal IO / timers (do_not_emit), and the restart
    # config queries (isRestart / isInitFromRestart -> .FALSE.) -- and inlines
    # everything else (the Krylov backend ladder + the mimetic operator apply).
    ("solve_free_sfc", "ocean/dynamics/mo_ocean_ab_timestepping_mimetic.f90",
     "mo_ocean_ab_timestepping_mimetic::solve_free_sfc_ab_mimetic", 191),
]

#: Checked-in single-TU artifacts: ``(key, filename, module::procedure)``.
#: Generated by the extraction above and committed here so the SDFG-lowering
#: stage (handled elsewhere) has a stable input; the extraction test
#: regenerates them and checks for drift.
#: ``(key, halo_mode, filename, module::procedure)``.  The non-solver kernels
#: extract in the ``external`` halo mode only; the free-surface SOLVER is the one
#: tested in BOTH modes (external = halo black-boxed / callback boundary; inlined
#: = halo devirtualised, MPI-only), per the both-modes contract.
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
    """Extract one ocean kernel into a single, gfortran-compiling ``.f90`` in
    a memory-capped subprocess (the fparser parse of the merged closure peaks
    near 9 GB, so it must not OOM the host) and return a result dict with keys
    ``passed`` (bool), ``tu_path`` (str|None), ``tu_lines`` (int|None) and
    ``output`` (str).

    The subprocess writes all artifacts under ``out_dir`` and uses it as
    ``TMPDIR`` too, keeping the large merged file off the RAM-backed ``/tmp``
    tmpfs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    tests_root = str(_HERE.parent.parent)
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([tests_root, prev_pp]) if prev_pp else tests_root
    env["TMPDIR"] = str(out_dir)
    env.setdefault("UCX_VFS_ENABLE", "n")
    # Pin the hash seed so the inliner's regeneration is byte-reproducible: the
    # drift guard asserts the extracted TU is identical to the committed one, and
    # any set/dict-iteration order leaking into emitted names would flake it.
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
