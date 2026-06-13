"""Self-contained ICON build helper for the ICON-source/built tests.

The ICON-integration tests (``test_velocity_from_icon_source``,
``test_dycore_from_icon_source``, ``test_icon_solve_nh_swap``,
``flang_codebase_test``) need a CONFIGURED + ``make``-d ICON tree so
the compiler ``.mod`` files exist under ``build/stock_cpu/mod`` and
``make -n`` can report the per-object ``-D`` / ``-I`` flags.

Rather than depend on an externally-provisioned build, these tests
build ICON themselves (once) via :func:`ensure_icon_built`.  The build
is idempotent and cached: if ``<build>/mod`` already holds ``.mod``
files (a developer's out-of-tree build, or a CI cache restore), the
configure + make is skipped.

The configure recipe mirrors the stock CPU build (extracted from a
known-good ``config.log``): gfortran/OpenMPI wrappers, serial HDF5 +
netCDF-Fortran, eccodes/grib2, OpenMP, bundled mtime, and the
ocean/jsbach/coupling/waves components disabled (the dycore tests
don't touch them and disabling cuts the build roughly in half).

System packages the build needs (Ubuntu names):
    libopenmpi-dev openmpi-bin
    libnetcdff-dev libnetcdf-dev
    libhdf5-dev
    libeccodes-dev
    libfyaml-dev
    libxml2-dev
    liblapack-dev libblas-dev
    python3 perl   (configure + bundled mtime codegen)
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

# Module-level memo so a session that touches several ICON tests builds
# at most once even before any external cache is warm.
_BUILT: dict = {}


def default_build_dir() -> Path:
    """Where to build ICON when no explicit dir is given.

    Defaults to TMP storage (``$TMPDIR/dace_fortran_icon_build/stock_cpu``)
    so the build artifacts -- hundreds of MB of ``.o`` / ``.mod`` --
    never land in the repo working tree (the submodule) and get reaped
    with the rest of tmp.  Override with ``ICON_BUILD`` to point at a
    persistent cache (e.g. a GitHub-Actions-cached path) when a warm
    rebuild matters more than ephemerality.
    """
    env = os.environ.get("ICON_BUILD")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "dace_fortran_icon_build" / "stock_cpu"


def _have_mods(build_dir: Path) -> bool:
    """True when the build dir already carries compiled ``.mod`` files."""
    mod_dir = build_dir / "mod"
    return mod_dir.is_dir() and any(mod_dir.glob("*.mod"))


def _missing_build_tools() -> list:
    """Return the subset of required executables not on PATH."""
    need = ["mpicc", "mpicxx", "mpifort", "make", "perl", "python3"]
    return [t for t in need if shutil.which(t) is None]


def ensure_icon_built(icon_src: Path,
                      build_dir: Optional[Path] = None,
                      jobs: Optional[int] = None) -> Optional[Path]:
    """Configure + ``make`` ICON under ``build_dir`` (default
    ``<icon_src>/build/stock_cpu``).  Idempotent: returns immediately
    when ``<build_dir>/mod`` already holds ``.mod`` files.

    :param icon_src: the ICON checkout (the submodule root).
    :param build_dir: out-of-tree build location; created if absent.
    :param jobs: ``make -j`` parallelism (default: all CPUs).
    :returns: ``build_dir`` on success, ``None`` if the ICON source
        isn't checked out (submodule absent).
    :raises RuntimeError: build tools missing, or configure/make fails
        -- surfaced (not swallowed) so the test FAILS loudly rather
        than silently skipping.
    """
    # Resolve to absolute paths: the configure + make subprocesses run
    # with ``cwd=build_dir``, so a relative ``icon_src`` would break.
    icon_src = Path(icon_src).resolve()
    if build_dir is None:
        # TMP storage by default -- never pollute the submodule tree.
        build_dir = default_build_dir()
    build_dir = Path(build_dir).resolve()

    key = str(build_dir.resolve())
    if key in _BUILT:
        return _BUILT[key]

    # Submodule not checked out -> nothing we can build.
    if not (icon_src / "configure").is_file():
        return None

    # Already built (developer tree or restored CI cache).
    if _have_mods(build_dir):
        _BUILT[key] = build_dir
        return build_dir

    missing = _missing_build_tools()
    if missing:
        raise RuntimeError(
            "ICON build needs these tools on PATH: " + ", ".join(missing) +
            ".  Install libopenmpi-dev openmpi-bin + perl + python3.")

    build_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs or (os.cpu_count() or 4)

    # xdist-safe: under ``pytest -n auto`` several workers may reach an
    # ICON test at once.  Serialise the configure + make behind a file
    # lock so exactly ONE worker builds while the others block, then
    # all reuse the result.  (``flock`` is advisory + released on fd
    # close / process exit, so a crashed builder doesn't deadlock.)
    import fcntl
    lock_path = build_dir.parent / ".icon_build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Re-check under the lock: a peer worker may have built while
        # we waited.
        if _have_mods(build_dir):
            _BUILT[key] = build_dir
            return build_dir
        return _configure_and_make(icon_src, build_dir, jobs, key)


def _configure_and_make(icon_src: Path, build_dir: Path, jobs: int,
                        key: str) -> Path:
    """Run the stock-CPU configure + ``make`` (called under the build
    lock).  Factored out so the lock-held critical section is explicit."""

    # --- configure (stock CPU recipe) --------------------------------
    common_cflags = "-O0 -g -fno-fast-math -ffp-contract=off -fPIC"
    fcflags = (common_cflags +
               " -fbacktrace -ffree-line-length-none"
               " -I/usr/include -I/usr/include/hdf5/serial")
    cppflags = "-I/usr/include -I/usr/include/hdf5/serial -I/usr/include/libxml2"
    ldflags = "-L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial"
    # GRIB2 (eccodes) is DISABLED: the dycore tests need only compiled
    # ``.mod`` files (no GRIB I/O), and the eccodes Fortran binding
    # (``libeccodes_f90``) is not shipped by stock Ubuntu
    # ``libeccodes-dev`` -- requiring it made the build non-portable.
    # Dropping ``--enable-grib2`` removes the eccodes link entirely.
    libs = ("-lxml2 -lfyaml -llapack -lblas"
            " -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 -lstdc++")
    configure_cmd = [
        str(icon_src / "configure"),
        "CC=mpicc", "CXX=mpicxx", "FC=mpifort",
        f"CFLAGS={common_cflags}",
        f"CXXFLAGS={common_cflags}",
        f"FCFLAGS={fcflags}",
        f"CPPFLAGS={cppflags}",
        f"LDFLAGS={ldflags}",
        f"LIBS={libs}",
        "MPI_LAUNCH=mpiexec",
        "--disable-grib2", "--enable-loop-exchange", "--enable-openmp",
        "--enable-bundled-python=mtime",
        "--disable-jsbach", "--disable-ocean",
        "--disable-coupling", "--disable-waves",
    ]
    subprocess.run(configure_cmd, cwd=build_dir, check=True)

    # --- make --------------------------------------------------------
    subprocess.run(["make", f"-j{jobs}"], cwd=build_dir, check=True)

    if not _have_mods(build_dir):
        raise RuntimeError(
            f"ICON make completed but no .mod files appeared under "
            f"{build_dir / 'mod'} -- the build recipe may need updating.")

    _BUILT[key] = build_dir
    return build_dir
