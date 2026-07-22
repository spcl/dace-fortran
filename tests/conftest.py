"""Puts the HLFIR test dir on sys.path (``from _util import ...``) and isolates DaCe's build cache per pytest-xdist worker so parallel runs don't race on shared ``.dacecache/<sdfg_name>/build`` (most tests reuse SDFG name ``main``)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# hwloc's GL/X11 probe (run during MPI_Init, before any mpi4py import) hangs forever on
# GNOME-on-Wayland: gnome-shell's abstract X11 socket accepts the connection but never
# completes the handshake. -gl disables only the GL/X11 GPU probe; setdefault keeps user overrides.
os.environ.setdefault("HWLOC_COMPONENTS", "-gl")

# Steer Open MPI onto in-node ob1/self,vader transports before mpi4py import: UCX/PMIx
# teardown can SIGABRT on MPI_Finalize, taking down xdist workers ("node down") and
# cascading to a controller deadlock. setdefault keeps any externally-provided config.
os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")
# Two more components still probe hardware at MPI_Init and block forever without a matching
# fabric (symptom: pytest stuck at 0% CPU on a socket read): UCX one-sided OSC and the
# OFI/libfabric MTL. Pin both off, disable vader's CMA/xpmem probe, and force PMIx's
# in-process hash store. setdefault so a real launcher's config still wins.
os.environ.setdefault("OMPI_MCA_osc", "^ucx")
os.environ.setdefault("OMPI_MCA_mtl", "^ofi")
os.environ.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
os.environ.setdefault("PMIX_MCA_gds", "hash")

# Raise stack soft limit to hard limit: deeply-nested inlined kernels (cloudsc, ICON, QE)
# drive MLIR's recursive Region::cloneInto/verifier/printer past the default 8MB stack, and
# pre-pipeline IR walks run on the main thread (not the bridge's 2GB-stack worker thread).
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_STACK)
    if _hard != resource.RLIM_INFINITY and (_soft == resource.RLIM_INFINITY or _soft < _hard):
        resource.setrlimit(resource.RLIMIT_STACK, (_hard, _hard))
    elif _hard == resource.RLIM_INFINITY and _soft != resource.RLIM_INFINITY:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, _hard))
except (ImportError, ValueError, OSError):
    # resource is POSIX-only; setrlimit can fail -- fall through with the inherited limit
    pass

# DaCe's default compiler.cpu.args (-O3 -march=native -ffast-math) permits FMA/reassociation
# that quietly diverges from a strict Fortran reference (-O0 -fno-fast-math -ffp-contract=off);
# NPB LU's ssor residual sat ~1.7% off at itmax=50 from this alone, a flag mismatch not a bridge
# bug. Force strict IEEE FP as the baseline here; tests wanting optimised perf override it.
from dace.config import Config

Config.set("compiler",
           "cpu",
           "args",
           value=("-fPIC -Wall -Wextra -O0 -fno-fast-math -ffp-contract=off "
                  "-Wno-unused-parameter -Wno-unused-label"))

# Per-worker DaCe build folder: PYTEST_XDIST_WORKER is gw0/gw1/... per worker, absent on
# serial runs (keeps the default .dacecache there).
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config
    Config.set("default_build_folder", value=f".dacecache_{_worker}")
else:
    # Master-only: force the hlfir_bridge .so build/staleness-check here, single-threaded,
    # before xdist spawns workers -- otherwise concurrent workers race ensure_fresh()'s
    # CMake build into the shared output path and one loads a partial .so (ImportError).
    import dace_fortran.build_bridge  # noqa: F401

import pytest

from dace_fortran.external import clear_external_registry

# --- generated-C++ sanity check ------------------------------------------
# Every SDFG a test compiles gets its generated C++ scanned for the UB-class warnings in
# CRITICAL_WARNINGS.  Wrapping compile() (rather than checking in one dedicated test) is what makes it
# a sanity check: any test that generates code exercises it, including the ones whose kernels nobody
# thought to analyse.  The uninitialised-extent miscompile that motivated this shipped through a suite
# that compiled the offending TU hundreds of times without ever reading a warning.
from dace.sdfg import SDFG

from dace_fortran.codegen_check import analyze

compile_without_check = SDFG.compile


def compile_and_check_generated_code(self, *args, **kwargs):
    result = compile_without_check(self, *args, **kwargs)
    found = analyze(self, "warnings")
    if found:
        raise AssertionError(f"generated C++ for SDFG '{self.name}' emits critical warnings:\n" + "\n".join(found))
    return result


SDFG.compile = compile_and_check_generated_code


# --- external-registry isolation -----------------------------------------
@pytest.fixture(autouse=True)
def isolate_external_registry():
    """Drop external-function registrations after every test.

    ``apply_external_functions``/``keep_external`` write a process-global registry; an
    unclean test leaks a registration into later tests -- a name collision rebinds an
    inlined call to the wrong ``.so``, and an arity mismatch reads garbage args (segfault
    or silently wrong values), surfacing only under ``-n auto`` depending on worker sharing.
    """
    yield
    clear_external_registry()


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
    """Mark every ``icon_build``-requesting test ``long`` and pin them to ONE xdist worker.

    * ``long`` -- the from-source ICON build is minutes of wall time; excluded via ``-m "not
      long"`` locally, but CI (``-m "not mpi"``) still runs it. Marking the test (not the dir)
      means deselecting ``long`` never instantiates the session fixture.
    * ``xdist_group("icon_build")`` -- under ``-n auto`` an unpinned test could build ICON once
      per worker (wasteful, racy); one shared group makes ``--dist loadgroup`` schedule them
      onto ONE worker so ICON builds exactly once.

    Tests that read ICON source but don't build it carry an explicit module-level ``long``
    instead (still need the submodule, same CI lane).
    """
    for item in items:
        if "icon_build" in getattr(item, "fixturenames", ()):  # uses the fixture
            item.add_marker(pytest.mark.long)
            item.add_marker(pytest.mark.xdist_group("icon_build"))


@pytest.fixture(scope="session")
def icon_build():
    """Configure + build the ICON submodule on demand; yield the build dir.

    ``ICON_SRC``/``ICON_BUILD`` override source/build location. Only SKIPS when the submodule
    isn't checked out; otherwise it always builds (never silently skips).
    """
    here = Path(__file__).resolve().parent
    icon_src = Path(os.environ.get("ICON_SRC", str(here / "icon" / "full" / "icon-model")))
    if not (icon_src / "configure").is_file():
        pytest.skip("icon-model submodule not checked out (run "
                    "`git submodule update --init tests/icon/full/icon-model`)")
    from icon.full._icon_build import ensure_icon_built, default_build_dir
    # builds into TMP by default (no repo-tree pollution); ICON_BUILD overrides to a persistent location
    icon_build_dir = default_build_dir()
    build = ensure_icon_built(icon_src, icon_build_dir)
    if build is None:
        pytest.skip("icon-model submodule not checked out")
    return build


# --- f2py-reference teardown-crash guard ---------------------------------
# f2py reference modules are never unloaded; at CPython finalisation numpy's teardown races
# their deallocators -> double-free -> SIGABRT (exit 134), always AFTER pytest's summary is
# already written. Record the real exit status here, then os._exit at pytest_unconfigure
# (runs last) to skip the crashing finaliser while preserving pytest's verdict.
_pytest_exitstatus = [0]

#: Repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _clean_stray_mods():
    """Remove stray ``*.mod`` files flang/gfortran drop into the repo-root CWD.

    Gitignored, but a leftover silently poisons a later gfortran compile of the same module
    name ("is not a GNU Fortran module file"). Top-level only (``glob`` not ``rglob``) so
    ``.dacecache``/``icon-model`` build trees are untouched. Run at conftest import (clear a
    crashed run's residue) and at session finish.
    """
    for mod in _REPO_ROOT.glob("*.mod"):
        try:
            mod.unlink()
        except OSError:
            pass


# defensive pre-clean: a prior crashed/os._exit-ed run may have left poison .mod files
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

    # os._exit skips CPython finalisation (incl. mpi4py's atexit MPI_Finalize). Under mpirun,
    # older OpenMPI/ORTE treats an unfinalised rank as abnormal termination -> nonzero exit even
    # though tests passed, so finalise explicitly when a launcher is detected. Without a
    # launcher, skip finalising entirely: on this host OpenMPI's own MPI_Finalize corrupts the
    # heap (SIGABRT, uncatchable, fires before os._exit) -- leaving MPI unfinalised is harmless
    # there and avoids the crash.
    if any(k in os.environ for k in ("OMPI_COMM_WORLD_SIZE", "PMIX_RANK")):
        try:
            from mpi4py import MPI
            if MPI.Is_initialized() and not MPI.Is_finalized():
                MPI.Finalize()
        except Exception:
            pass

    os._exit(_pytest_exitstatus[0])
