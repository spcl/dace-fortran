"""Isolates DaCe's build cache per pytest-xdist worker so parallel runs don't race on shared ``.dacecache/<sdfg_name>/build`` (most tests reuse SDFG name ``main``)."""

import os
from pathlib import Path

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

# Build the way production does: DaCe's own defaults (Release -O3, -march=native and the
# targeted FP relaxations) so what the suite exercises is what users get. A test comparing
# against a strict Fortran reference requests the ``e2e_cpu_args`` fixture (below) rather than
# rely on the whole suite being compiled strict: -O3 contracts a multiply-add and
# -freciprocal-math turns a/b into a*(1/b), each a different result by an ULP, not a wrong one.

# Per-worker DaCe build folder: PYTEST_XDIST_WORKER is gw0/gw1/... per worker, absent on
# serial runs (keeps the configured folder there). Workers sharing one folder race each other's
# links (`collect2: ld returned 1`, "Could not load library"), so each suffixes the configured one
# (``DACE_default_build_folder`` or the ``.dacecache`` default).
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    from dace.config import Config

    Config.set("default_build_folder", value=f"{Config.get('default_build_folder')}_{_worker}")
else:
    # Master-only: force the hlfir_bridge .so build/staleness-check here, single-threaded,
    # before xdist spawns workers -- otherwise concurrent workers race ensure_fresh()'s
    # CMake build into the shared output path and one loads a partial .so (ImportError).
    import dace_fortran.build_bridge  # noqa: F401

import contextlib
import functools

# --- generated-C++ sanity check ------------------------------------------
# Every SDFG a test compiles gets its generated C++ scanned for the UB-class warnings in
# CRITICAL_WARNINGS.  Wrapping compile() (rather than checking in one dedicated test) is what makes it
# a sanity check: any test that generates code exercises it, including the ones whose kernels nobody
# thought to analyse.  The uninitialised-extent miscompile that motivated this shipped through a suite
# that compiled the offending TU hundreds of times without ever reading a warning.
import dace
import pytest
from dace.sdfg import SDFG

from dace_fortran.external import clear_external_registry
from tests import _cache
from tests._util import BITEXACT_CPU_ARGS
from tests.codegen_check import (
    NONCRITICAL_NOISE,
    analyze,
    build_flags,
    compiler_is_clang,
    critical_tags,
    generated_source,
)

compile_without_check = SDFG.compile


@functools.cache
def runtime_headers_digest() -> str:
    """Content of every DaCe runtime header: an upgraded runtime can change what the compiler warns about in a TU."""
    include = Path(dace.__file__).parent / "runtime" / "include"
    return _cache.digest(*(p.read_bytes() for p in sorted(include.rglob("*")) if p.is_file()))


def warning_diagnostics(sdfg: SDFG) -> list[str]:
    """Every ``-W`` diagnostic of the generated C++, memoised on the TU text, its flags and the compiler.

    The analysis is a second full compile of the TU, so an identical TU (another run, another parametrisation) reuses the
    earlier verdict. The build folder is masked out of the key: it differs per worker and per run.
    """
    folder = str(Path(sdfg.build_folder).resolve())
    compiler = "clang++" if compiler_is_clang() else "g++"
    key = _cache.digest(
        generated_source(sdfg).read_bytes().replace(folder.encode(), b"<build>"),
        "\0".join(f.replace(folder, "<build>") for f in build_flags(sdfg)),
        _cache.tool_version(compiler),
        runtime_headers_digest(),
    )
    hit = _cache.entry("warnings", key)
    if hit is not None:
        return hit.read_text().splitlines()
    diagnostics = analyze(sdfg, "warnings", critical_only=False)
    verdict = Path(sdfg.build_folder) / "warnings.txt"
    verdict.write_text("\n".join(diagnostics))
    _cache.store("warnings", key, verdict)
    return diagnostics


def compile_and_check_generated_code(self, *args, **kwargs):
    result = compile_without_check(self, *args, **kwargs)
    # One compile-flag pass gives every -W diagnostic; split it: critical (UB-class) fails the build, the rest is
    # surfaced as information (minus the hundreds-strong style noise) so a non-fatal warning is at least visible.
    diagnostics = warning_diagnostics(self)
    tags = critical_tags()
    critical = [ln for ln in diagnostics if any(t in ln for t in tags)]
    informational = [
        ln for ln in diagnostics if ln not in critical and not any(f"[-W{n}]" in ln for n in NONCRITICAL_NOISE)
    ]
    if informational:
        print(
            f"[codegen-check] '{self.name}' non-critical warnings ({len(informational)}):\n" + "\n".join(informational)
        )
    if critical:
        raise AssertionError(f"generated C++ for SDFG '{self.name}' emits critical warnings:\n" + "\n".join(critical))
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
    """Configure + build the ICON submodule on demand; return the build dir.

    ``ICON_SRC``/``ICON_BUILD`` override source/build location. A missing submodule fails the test.
    """
    from tests.icon.full._icon_build import default_build_dir, ensure_icon_built

    icon_src = Path(os.environ.get("ICON_SRC", str(Path(__file__).resolve().parent / "icon" / "full" / "icon-model")))
    # builds into TMP by default (no repo-tree pollution); ICON_BUILD overrides to a persistent location
    return ensure_icon_built(icon_src, default_build_dir())


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
        with contextlib.suppress(OSError):
            mod.unlink()


# defensive pre-clean: a prior crashed/os._exit-ed run may have left poison .mod files
_clean_stray_mods()


def pytest_sessionfinish(session, exitstatus):
    _pytest_exitstatus[0] = int(exitstatus)
    _clean_stray_mods()


#: Environment variables an MPI launcher (Open MPI's prterun/mpirun, PMIx) sets in every rank.
_MPI_LAUNCHER_ENV = ("OMPI_COMM_WORLD_SIZE", "PMIX_RANK")


def _under_mpi_launcher() -> bool:
    return any(k in os.environ for k in _MPI_LAUNCHER_ENV)


def pytest_configure(config):
    """Under an MPI launcher, initialise MPI if mpi4py did not.

    The multi-rank tests call ``COMM_WORLD`` straight away and rely on mpi4py's import-time ``MPI_Init``. An
    environment exporting ``MPI4PY_RC_INITIALIZE=0`` (the node suite does, for its single-process lanes) turns that
    off, and the first ``Get_rank`` then aborts the rank ("MPI_Comm_rank() was called before MPI_INIT", exit 14).
    """
    if _under_mpi_launcher():
        from mpi4py import MPI

        if not MPI.Is_initialized():
            MPI.Init()


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
    if _under_mpi_launcher():
        try:
            from mpi4py import MPI

            if MPI.Is_initialized() and not MPI.Is_finalized():
                MPI.Finalize()
        except Exception:
            pass

    os._exit(_pytest_exitstatus[0])


@pytest.fixture
def e2e_cpu_args():
    """Build the SDFG with the paper's flags; restore whatever was set before.

    Request it in every test that asserts bit-exact equality with a strict Fortran/NumPy reference: DaCe's
    default args carry -freciprocal-math (a/b -> a*(1/b)) and leave -ffp-contract at the compiler default (FMA
    under -march=native on x86 with FMA), either of which moves the last ULP.
    """
    prev = dace.Config.get("compiler", "cpu", "args")
    dace.Config.set("compiler", "cpu", "args", value=BITEXACT_CPU_ARGS)
    yield BITEXACT_CPU_ARGS
    dace.Config.set("compiler", "cpu", "args", value=prev)
