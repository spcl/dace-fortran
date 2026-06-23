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

import shutil

_HERE = Path(__file__).resolve().parent
#: ``tests/icon/full/icon-model`` holds the pinned ICON checkout (shared with
#: the atmosphere velocity test).  ``ICON_SRC`` overrides it.
_ICON_SRC = Path(os.environ.get("ICON_SRC", str(_HERE.parent / "full" / "icon-model")))
SRC = _ICON_SRC / "src"

HAVE_FLANG = shutil.which("flang-new-21") is not None
HAVE_OPENMPI = find_openmpi_include() is not None


def have_icon_ocean() -> bool:
    """True when the ocean source tree is checked out."""
    return (SRC / "ocean" / "tracer_transport" / "mo_ocean_tracer_transport_vert.f90").is_file()


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

#: ICON service procedures that are genuinely external to a compute kernel and
#: must NOT be inlined.  The unified external-function policy (see
#: :mod:`dace_fortran.external_functions`) splits them into two collections,
#: declared ONCE here and consumed by both inliner engines and the bridge:
#:
#: * ``OCEAN_EXTERNAL_FUNCTIONS`` -- don't-inline + the bridge EMITs an external
#:   call (the MPI halo exchange: the real boundary of a standalone kernel).
#:   ``library``/``c_function`` are left at their defaults (symbol == name) until
#:   the SDFG-lowering stage binds them to a concrete ``libicon_halo`` ABI.
#: * ``OCEAN_DO_NOT_EMIT`` -- don't-inline + the bridge DROPs the call (terminal
#:   read/write error/log I/O and timers: pure side-effects, no numerics).
#:
#: At the inliner both lists are stubbed identically (opener+spec+END kept, body
#: emptied via ``make_noop``) so their MPI / type-bound-procedure / I/O internals
#: never enter the TU.  Everything else -- including the real operators a kernel
#: calls such as ``get_index_range`` and ``rot_vertex_ocean_3d`` -- is INLINED.
#: Declared MANUALLY (the inliner hardcodes nothing); ocean kernels that don't
#: call any of these (e.g. the PPM block kernel) are simply unaffected.
#: EMIT externals = every genuine MPI operation the kernel performs.  All are
#: real, semantically-meaningful calls the bridge emits as external and the
#: SDFG-lowering stage binds to dace's distributed primitives (dace grid == MPI
#: communicator) -- adding ``LibNode``s for any collective dace lacks (e.g. an
#: AllReduce for the global reductions).  Two families:
#:   * data communication -- halo exchange (``sync_patch_array*`` / ``exchange_data``);
#:   * collectives -- barriers (``work_mpi_barrier`` / ``p_barrier``) and global
#:     reductions (``p_max`` / ``p_min`` / ``p_sum``, the ``mo_mpi`` MPI_Allreduce
#:     wrappers).  These MUST be emitted: a barrier is a real synchronisation and
#:     a distributed reduction changes the numerics, so neither may be dropped.
#: Stubbing them at the inliner is also what lets the TU shed ``USE mpi`` entirely.
OCEAN_EXTERNAL_FUNCTIONS = [
    ExternalFunction("sync_patch_array"),       # MPI halo exchange (generic: sync_patch_array_3d_dp, ...)
    ExternalFunction("sync_patch_array_mult"),  # MPI multi-field halo exchange
    ExternalFunction("exchange_data"),          # MPI halo primitive underneath the syncs (exchange_data_r3d, ...)
    ExternalFunction("work_mpi_barrier"),       # MPI collective barrier (mo_mpi: MPI_Barrier)
    ExternalFunction("p_barrier"),              # MPI collective barrier (mo_mpi wrapper, timer-gated)
    ExternalFunction("p_max"),                  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_MAX)
    ExternalFunction("p_min"),                  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_MIN)
    ExternalFunction("p_sum"),                  # MPI global reduction (mo_mpi: MPI_Allreduce, MPI_SUM)
    # The iterative-solver subsystem used by invert_mass_matrix for the inverse
    # mass-matrix / flip-flop solve.  The WHOLE subsystem is externalised so the
    # solver is one opaque black box at the SDFG boundary, because its core is
    # built on Fortran VIRTUAL DISPATCH that neither our inliner nor flang can
    # turn into static dataflow (an SDFG has no node for a runtime indirect call
    # through a type descriptor's binding table):
    #
    #   * ``ocean_solve_construct`` is a runtime FACTORY:
    #     ``ALLOCATE(t_ocean_solve_gmres :: this%act)`` picks the concrete backend
    #     from a ``SELECT CASE`` and then dispatches ``CALL this%act%construct``.
    #   * the SOLVE (``ocean_solve_solve`` -> backend ``%solve``) dispatches
    #     ``this%act%solve`` / ``this%lhs%apply`` / ``this%trans%into`` on the
    #     ABSTRACT bases ``t_ocean_solve_backend`` / ``t_lhs_agen`` / ``t_transfer``
    #     (``TYPE, ABSTRACT`` with ``DEFERRED`` bindings).
    #
    # flang lowers every such call to ``fir.dispatch`` (a runtime vtable lookup),
    # NOT a direct call -- even for the factory shape where the concrete type was
    # ``ALLOCATE``d one line above, and even after ``-O2`` / ``--inline-all`` /
    # ``--fir-polymorphic-op`` (that pass only *lowers* dispatch to the explicit
    # vtable indirect; it never devirtualises).  Pinned by
    # ``tests/hlfir_devirtualization_test.py``.  The only shape that binds
    # statically is a call on a concrete ``TYPE(..)`` entity -- reaching it would
    # need source-level MONOMORPHISATION in our inliner (commit each abstract
    # dummy to its single runtime type), a feature we do not have.
    #
    # Run frequency differs (these are NOT all one-time -- see
    # invert_mass_matrix lines 1499-1532):
    #   ONCE, guarded by ``IF (.NOT. inv_mm_solver % is_init)``:
    #     - ``ocean_solve_construct``     (the factory; the externalised construct
    #       allocates/binds the solver object and sets ``is_init``)
    #     - ``trivial_transfer_construct``
    #   PER VERTICAL LEVEL, inside ``DO jk = 1, n_zlev``:
    #     - ``ocean_solve_solve``         (the actual solve, line 1521; real
    #       numerics the bridge EMITs as an external call the SDFG-lowering binds
    #       to a solver LibNode -- invoked once per level)
    #     - ``lhs_primal_flip_flop_construct`` (re-inits the LHS each level,
    #       line 1514).  NB this one does NOT itself dispatch -- it is a static
    #       bind on a concrete ``TYPE(t_primal_flip_flop_lhs)`` with a trivial
    #       pointer/scalar-setter body -- so it COULD be inlined; it is kept
    #       external only so solver-internal LHS state stays inside the black box
    #       the external solve consumes, rather than leaking into the SDFG.
    # (``ocean_solve_parm_init`` == par%init is a pure, MPI-free, non-dispatching
    # setter and is INLINED, per policy.)
    ExternalFunction("ocean_solve_construct"),          # runtime factory (ALLOCATE+dispatch); ONCE (is_init guard)
    ExternalFunction("trivial_transfer_construct"),     # transfer-object construct; ONCE (is_init guard)
    ExternalFunction("lhs_primal_flip_flop_construct"), # LHS re-init; PER LEVEL (static bind, kept external for a clean boundary)
    ExternalFunction("ocean_solve_solve"),              # the linear solve; PER LEVEL (dispatches act/lhs/trans on abstract bases)
]
#: DON'T-EMIT = externalised (NOT inlined) and the bridge DROPs the call: pure
#: side-effects with no numerics -- terminal I/O (debug / error / log) and timers.
OCEAN_DO_NOT_EMIT = [
    "dbg_print",                      # terminal write (debug print)
    "finish", "message", "warning",   # terminal write (error / log)
    "timer_start", "timer_stop", "new_timer", "delete_timer",  # timers
]
# NOTE: rot_vertex_ocean_3d is INLINED (it is pure vorticity compute, no MPI in
# its body).  Inlining it pulls in its host module's USE closure (mo_mpi
# reductions, t_comm_pattern CLASS(*)), but the inliner's external-USE tolerance
# processes-then-prunes that unreachable MPI baggage, so the kernel still
# extracts to a compiling single TU with the vorticity computed in-line.

#: The ICON-O kernels currently extracted.  Each entry is
#: ``(key, source-relative-to-src, module::procedure, body-line-count)``.
KERNELS = [
    ("ppm_vflux",
     "ocean/tracer_transport/mo_ocean_tracer_transport_vert.f90",
     "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock", 339),
    ("coriolis_pv",
     "ocean/math/mo_scalar_product.f90",
     "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar", 273),
    ("veloc_adv_horz",
     "ocean/dynamics/mo_ocean_velocity_advection.f90",
     "mo_ocean_velocity_advection::veloc_adv_horz_mimetic", 85),
]

#: Checked-in single-TU artifacts: ``(key, filename, module::procedure)``.
#: Generated by the extraction above and committed here so the SDFG-lowering
#: stage (handled elsewhere) has a stable input; the extraction test
#: regenerates them and checks for drift.
SINGLE_TU_ARTIFACTS = [
    ("ppm_vflux", "ppm_vflux_single_tu.f90",
     "mo_ocean_tracer_transport_vert::upwind_vflux_ppm_onBlock"),
    ("coriolis_pv", "coriolis_pv_single_tu.f90",
     "mo_scalar_product::nonlinear_coriolis_3d_fast_scalar"),
    ("veloc_adv_horz", "veloc_adv_horz_single_tu.f90",
     "mo_ocean_velocity_advection::veloc_adv_horz_mimetic"),
]

_EXTRACT_SCRIPT = _HERE / "_extract_single_tu.py"


def extract_single_tu(source_relpath: str, entry: str, out_dir: Path, mem_gb: float = 10.0) -> dict:
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
    env["PYTHONPATH"] = os.pathsep.join([str(_HERE.parent.parent), env.get("PYTHONPATH", "")])
    env["TMPDIR"] = str(out_dir)
    env.setdefault("UCX_VFS_ENABLE", "n")
    proc = subprocess.run(
        [sys.executable, str(_EXTRACT_SCRIPT), source_relpath, entry, str(out_dir), str(mem_gb)],
        capture_output=True, text=True, env=env, cwd=str(out_dir))
    out = proc.stdout + "\n" + proc.stderr
    tu_path, tu_lines = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("TU_PATH:"):
            tu_path = line.split(":", 1)[1].strip()
        elif line.startswith("TU_LINES:"):
            tu_lines = int(line.split(":", 1)[1])
    passed = any(line.startswith("RESULT: PASS") for line in proc.stdout.splitlines())
    return {"passed": passed, "tu_path": tu_path, "tu_lines": tu_lines, "output": out}
