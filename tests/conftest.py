"""Put the HLFIR test directory on sys.path so ``from _util import ...`` works
when pytest collects tests from this folder.

Also isolates DaCe's build cache directory per pytest-xdist worker so
parallel runs don't race on the shared ``.dacecache/<sdfg_name>/build``
directory  --  most HLFIR tests reuse the SDFG name ``main`` and would
otherwise clobber each other's CMake state under ``-n N``.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Disable hwloc's GL/X11 topology backend BEFORE any ``from mpi4py import MPI``
# (DaCe pulls MPI in transitively; ``pytest_unconfigure`` below imports it
# explicitly to ``MPI_Finalize`` cleanly).  ``MPI_Init`` runs hwloc topology
# detection, and hwloc's GL backend probes every local X11 display
# (``:0``, ``:1``, ...) to enumerate NVIDIA GPUs via the NV-CONTROL X
# extension.  On a GNOME-on-Wayland desktop, gnome-shell squats on the
# abstract ``@/tmp/.X11-unix/X1`` socket without speaking the X protocol, so
# hwloc's probe connects, gets accepted, then blocks forever waiting for the
# X handshake -- hanging the whole pytest process at teardown (and any
# ``import mpi4py`` in a dev shell).  ``HWLOC_COMPONENTS=-gl`` disables only
# the GL/X11 GPU probe; CPU / memory / PCI topology detection (all MPI needs
# for binding) is unaffected.  ``setdefault`` so an explicit user override
# (e.g. someone debugging hwloc) still wins.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

# DaCe's frontend lazily ``from mpi4py import MPI`` during ``to_sdfg`` (and the
# ``pytest_unconfigure`` hook below imports it to ``MPI_Finalize`` cleanly).  On
# hosts where MPI_Init stalls on UCX -- or, worse, where UCX/PMIx teardown
# aborts in ``PMIx_Finalize`` (SIGABRT) and kills the process -- steer Open MPI
# onto the in-node ``ob1``/``self,vader`` transports BEFORE that import runs.
# Without this, the finalize-on-exit abort takes down xdist workers
# ("node down: Not properly terminated"), cascading to sibling workers and
# deadlocking the controller.  ``setdefault`` keeps any externally-provided MPI
# configuration.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
# ``pml=ob1`` + ``btl=self,vader`` steer point-to-point onto shared memory, but
# two more components still probe hardware at ``MPI_Init`` and block forever on a
# node without the matching fabric (the symptom: pytest sits at 0% CPU on a
# socket read while importing a kernel that pulls in mpi4py): the UCX one-sided
# OSC component, and the OFI/libfabric matching layer (MTL) enumerating network
# providers.  Pin both off, disable vader's CMA/xpmem single-copy probe (a
# kernel-module capability check), and steer PMIx onto its in-process ``hash``
# data store so the shared-memory ``gds`` never blocks.  ``setdefault`` so a
# real launcher's configuration still wins.
os.environ.setdefault("OMPI_MCA_osc", "^ucx")
os.environ.setdefault("OMPI_MCA_mtl", "^ofi")
os.environ.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
os.environ.setdefault("PMIX_MCA_gds", "hash")

# Raise the stack size to the hard limit (typically ``unlimited`` on Linux)
# for every test process.  Deeply-nested fully-inlined kernels (cloudsc,
# ICON dycore, QE microkernels) drive MLIR's recursive ``Region::cloneInto``
# / verifier / printer far past the default 8 MB stack.  The bridge already
# runs its pass pipeline on a 2 GB-stack worker thread, but any pre-pipeline
# IR walk (parse, ``set_entry_symbol``, ``dump``, ``get_ast``) runs on the
# Python main thread, and on systems whose soft limit ``RLIMIT_STACK`` is
# 8 MB those walks can overflow even before the pipeline starts.  Bumping
# soft to hard at session start gives every kernel the same ample stack
# without per-test boilerplate, and is a no-op when the user has already
# raised the limit themselves.
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
    if _hard != resource.RLIM_INFINITY and (_soft == resource.RLIM_INFINITY or _soft < _hard):
        resource.setrlimit(resource.RLIMIT_STACK, (_hard, _hard))
    elif _hard == resource.RLIM_INFINITY and _soft != resource.RLIM_INFINITY:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, _hard))
except (ImportError, ValueError, OSError):
    # ``resource`` is POSIX-only; ``setrlimit`` fails when the new soft
    # limit exceeds the hard limit -- fall through with the inherited
    # limit and let any kernel that overflows surface as a stack
    # overflow the user can raise their own shell limit for.
    pass

# Strict numerical correctness compile flags for DaCe's CPU codegen.
#
# DaCe's default ``compiler.cpu.args`` is
# ``-O3 -march=native -ffast-math`` which permits FMA contractions,
# reciprocal approximations, and associativity-based rewrites -- all of
# which produce a quietly different bit pattern from a strict Fortran
# reference compiled at ``-O0 -fno-fast-math -ffp-contract=off``.  For
# every numerical-correctness test against an f2py / gfortran reference,
# this drift surfaces as a small percentage residual gap (NPB LU's ssor
# residual sat ~1.7% off the reference at itmax=50 with the default
# flags) that is REAL but is a flag mismatch, not a bridge bug.
#
# Force strict IEEE FP semantics on the SDFG's compile here so every
# test starts from the same baseline as the reference.  Tests that
# specifically want optimised performance set ``compiler.cpu.args``
# themselves; this baseline gives them a known reference point.
from dace.config import Config

Config.set("compiler",
           "cpu",
           "args",
           value=("-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off "
                  "-Wno-unused-parameter -Wno-unused-label"))

# Per-worker DaCe build folder.  ``PYTEST_XDIST_WORKER`` is set by
# pytest-xdist to ``gw0``, ``gw1``, ... on each worker process; absent on
# serial runs (we keep the default ``.dacecache`` so existing tooling
# behaves the same).
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config
    Config.set("default_build_folder", value=f".dacecache_{_worker}")
else:
    # Master-only: build the C++ ``hlfir_bridge`` extension BEFORE
    # pytest-xdist spawns workers.  Every worker imports
    # ``dace_fortran.build_bridge`` (transitively, via SDFGBuilder),
    # whose module-level ``ensure_fresh()`` rebuilds the .so when
    # sources are newer or the .so is missing.  Without this gate,
    # multiple workers can hit ``needs_build() == True`` concurrently,
    # each launch a CMake build into the shared output path, and
    # produce a partial .so that ImportErrors when any worker tries
    # to load it.  Eagerly importing here forces the staleness check
    # + (re)build to complete in the master where it is single-
    # threaded; workers then inherit a fresh .so and the import
    # in their conftest is a no-op fast path.
    import dace_fortran.build_bridge  # noqa: F401

import pytest


# --- ICON build fixture --------------------------------------------------
# The ICON-integration tests need a configured + ``make``-d ICON tree
# (compiled ``.mod`` files under ``build/stock_cpu/mod`` + a Makefile
# ``make -n`` can introspect).  Rather than gate them on an externally-
# provisioned build (which made them SKIP on CI), they build ICON
# themselves via this session-scoped fixture: ``ensure_icon_built`` is
# idempotent + cached, so the configure + make runs at most once per
# session and is a no-op when a build (developer tree or CI cache
# restore) is already present.
def pytest_collection_modifyitems(config, items):
    """Mark every ICON-build test ``long`` and pin them onto ONE xdist worker.

    The ``icon_build`` fixture configures + ``make``s ICON from source
    (session-scoped) -- minutes of wall time.  Two things follow from
    any test that requests it, applied here in one place so future
    ``icon_build`` consumers inherit both automatically:

    * ``long`` -- the full ICON-from-source build is too slow for a
      routine local "run all" sweep, so it is tagged ``long`` and
      excluded with ``-m "not long"``.  CI does NOT filter ``long``
      (it runs ``-m "not mpi"``), so CI still builds + runs these.
      Marking the *test* (not the dir) means deselecting ``long``
      never instantiates the session fixture, so no ICON build fires.
    * ``xdist_group("icon_build")`` -- under ``pytest -n auto`` an
      unpinned ICON test could land on any worker, building ICON once
      per such worker (wasteful; a cross-worker race on the shared tmp
      build dir, serialised by ``ensure_icon_built``'s file lock).  One
      shared ``xdist_group`` makes ``--dist loadgroup`` schedule them
      onto ONE worker, so ICON builds exactly once.  No-op without
      xdist / loadgroup.

    Tests that read ICON's real source through the icon-model submodule but
    do NOT build it (HLFIR emit / SDFG build / compile-commands parse) carry
    an explicit module-level ``pytest.mark.long`` instead -- they still need
    the submodule (checked out only by the heavy CI lane), so they belong in
    the same lane.  The self-contained single-TU velocity e2e correctness
    tests (in-tree ``velocity_full.f90``, no submodule) deliberately stay in
    the fast lane.
    """
    for item in items:
        if "icon_build" in getattr(item, "fixturenames", ()):  # uses the fixture
            item.add_marker(pytest.mark.long)
            item.add_marker(pytest.mark.xdist_group("icon_build"))


@pytest.fixture(scope="session")
def icon_build():
    """Configure + build the ICON submodule on demand; yield the build dir.

    Resolves ICON source from ``ICON_SRC`` (default: the in-repo
    submodule ``tests/icon/full/icon-model``) and the build location
    from ``ICON_BUILD`` (default ``<src>/build/stock_cpu``).  Only
    SKIPS when the submodule itself is not checked out -- a genuine
    "nothing to build" state; with the submodule present it builds
    (never silently skips).
    """
    here = Path(__file__).resolve().parent
    icon_src = Path(os.environ.get(
        "ICON_SRC", str(here / "icon" / "full" / "icon-model")))
    if not (icon_src / "configure").is_file():
        pytest.skip("icon-model submodule not checked out (run "
                    "`git submodule update --init tests/icon/full/icon-model`)")
    from icon.full._icon_build import ensure_icon_built, default_build_dir
    # Build into TMP storage by default (no repo-tree pollution);
    # ``ICON_BUILD`` overrides to a persistent / cached location.
    icon_build_dir = default_build_dir()
    build = ensure_icon_built(icon_src, icon_build_dir)
    if build is None:
        pytest.skip("icon-model submodule not checked out")
    return build


# --- f2py-reference teardown-crash guard ---------------------------------
# The e2e tests import f2py-compiled reference extension modules and never
# unload them.  At CPython finalisation numpy's teardown races those
# modules' deallocators, double-freeing a heap block -> SIGABRT (exit
# 134).  It happens strictly after every test ran AND pytest wrote its
# summary, so the verdict is already correct; the crash only corrupts the
# exit path and, under -q, ate buffered output that made results
# unreadable.  We record the real exit status at sessionfinish, then
# os._exit at pytest_unconfigure -- which runs LAST, after the terminal
# summary, exactly where the double-free otherwise fires -- skipping the
# crashing finaliser while preserving pytest's verdict and exit code.
_pytest_exitstatus = [0]

#: Repo root -- ``tests/conftest.py`` -> parent -> parent.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _clean_stray_mods():
    """Remove stray Fortran ``*.mod`` files that flang / gfortran drop into the
    repo-root CWD during SDFG builds and reference compiles.

    They are gitignored build artifacts, but a leftover silently poisons a
    later gfortran compile that ``USE``s the same module: a flang-format
    ``iso_c_binding.mod`` (or ``constants.mod`` &c.) sitting in the module
    search path makes gfortran abort with "is not a GNU Fortran module file".
    Only the repo-root TOP LEVEL is swept (``glob`` not ``rglob``) so
    legitimate module trees under ``.dacecache`` / ``_session_scratch`` /
    ``icon-model`` build dirs are never touched.  Run both at conftest import
    (clear a previous crashed run's residue before this run builds) and at
    session finish (leave a clean tree)."""
    for mod in _REPO_ROOT.glob("*.mod"):
        try:
            mod.unlink()
        except OSError:
            pass


# Defensive pre-clean: a prior crashed / ``os._exit``-ed run may have left
# poison ``.mod`` files in the root that would break this run's gfortran e2e
# compiles before its own teardown ever runs.
_clean_stray_mods()


def pytest_sessionfinish(session, exitstatus):
    _pytest_exitstatus[0] = int(exitstatus)
    _clean_stray_mods()


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    # ``os._exit`` skips CPython finalisation -- including mpi4py's
    # atexit ``MPI_Finalize``.  DaCe's MPI environment deliberately
    # never finalises (process-global, driver's job), so under
    # ``mpirun`` the ranks would exit MPI-initialised-but-not-finalised:
    # newer OpenMPI/PRTE only warns, but the runner's older OpenMPI/ORTE
    # treats it as abnormal termination and makes ``mpirun`` return
    # non-zero, failing CI even though every test passed.  Finalise
    # explicitly here (guarded; a no-op when MPI was never initialised,
    # i.e. the normal non-mpi sweep) so termination is clean everywhere.
    # Only finalise under a real ``mpirun``/``mpiexec`` launch (where ORTE
    # needs every rank finalised for a zero exit code).  In the normal,
    # non-MPI sweep there is no launcher and finalising is unnecessary --
    # and on this host OpenMPI's teardown corrupts the heap ("corrupted
    # double-linked list" -> SIGABRT) inside ``MPI_Finalize`` itself, which
    # is uncatchable and fires *before* the ``os._exit`` below, killing the
    # process (and, under xdist, the worker -> "node down" cascade).  When
    # not under a launcher we skip the finalise entirely: ``os._exit`` skips
    # CPython finalisation (and mpi4py's atexit ``MPI_Finalize``), so the
    # process leaves MPI initialised-but-not-finalised -- harmless without a
    # launcher -- and exits cleanly with the recorded verdict.
    if any(k in os.environ for k in ("OMPI_COMM_WORLD_SIZE", "PMIX_RANK")):
        try:
            from mpi4py import MPI
            if MPI.Is_initialized() and not MPI.Is_finalized():
                MPI.Finalize()
        except Exception:
            pass

    os._exit(_pytest_exitstatus[0])
