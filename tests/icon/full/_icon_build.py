"""Self-contained ICON build helper for the ICON-source/built tests.

The ICON-integration tests need a CONFIGURED + made ICON tree (.mod files
under build/stock_cpu/mod). :func:`ensure_icon_built` builds it once,
idempotently -- skips configure+make if <build>/mod already has .mod files
(dev out-of-tree build or CI cache restore). Recipe: stock-CPU config
(gfortran/OpenMPI, serial HDF5+netCDF-Fortran, eccodes/grib2, OpenMP, bundled
mtime), ocean/jsbach/coupling/waves disabled.

Needs (Ubuntu): libopenmpi-dev openmpi-bin libnetcdff-dev libnetcdf-dev
libhdf5-dev libeccodes-dev libfyaml-dev libxml2-dev liblapack-dev libblas-dev
python3 perl.
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
    """Default ICON build dir: ``$TMPDIR/dace_fortran_icon_build/stock_cpu`` (keeps
    hundreds of MB of .o/.mod out of the repo tree); override via ``ICON_BUILD``
    for a persistent cache."""
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


def ensure_icon_built(icon_src: Path, build_dir: Optional[Path] = None, jobs: Optional[int] = None) -> Optional[Path]:
    """Configure + ``make`` ICON under ``build_dir`` (default ``<icon_src>/build/stock_cpu``).
    Idempotent: no-ops if ``<build_dir>/mod`` already has .mod files. Returns None if
    the ICON submodule isn't checked out; raises RuntimeError (not swallowed --
    tests must FAIL loudly, not silently skip) if tools are missing or the build fails."""
    # absolute paths: configure/make run with cwd=build_dir, relative icon_src would break
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
        raise RuntimeError("ICON build needs these tools on PATH: " + ", ".join(missing) +
                           ".  Install libopenmpi-dev openmpi-bin + perl + python3.")

    build_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs or (os.cpu_count() or 4)

    # xdist-safe: serialise configure+make behind a flock so only one worker builds;
    # flock is advisory, released on fd close/process exit, so a crashed builder doesn't deadlock
    import fcntl
    lock_path = build_dir.parent / ".icon_build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # re-check under the lock: a peer worker may have built while we waited
        if _have_mods(build_dir):
            _BUILT[key] = build_dir
            return build_dir
        return _configure_and_make(icon_src, build_dir, jobs, key)


def _configure_and_make(icon_src: Path, build_dir: Path, jobs: int, key: str) -> Path:
    """Stock-CPU configure + make, called under the build lock (keeps the critical section explicit)."""

    # --- configure (stock CPU recipe) --------------------------------
    common_cflags = "-O0 -g -fno-fast-math -ffp-contract=off -fPIC"
    fcflags = (common_cflags + " -fbacktrace -ffree-line-length-none"
               " -I/usr/include -I/usr/include/hdf5/serial")
    cppflags = "-I/usr/include -I/usr/include/hdf5/serial -I/usr/include/libxml2"
    ldflags = "-L/usr/lib/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu/hdf5/serial"
    # GRIB2/eccodes disabled: dycore tests only need .mod files (no GRIB I/O), and
    # libeccodes_f90 isn't shipped by stock Ubuntu libeccodes-dev -- non-portable otherwise.
    libs = ("-lxml2 -lfyaml -llapack -lblas"
            " -lnetcdff -lnetcdf -lhdf5_hl -lhdf5 -lstdc++")
    configure_cmd = [
        str(icon_src / "configure"),
        "CC=mpicc",
        "CXX=mpicxx",
        "FC=mpifort",
        f"CFLAGS={common_cflags}",
        f"CXXFLAGS={common_cflags}",
        f"FCFLAGS={fcflags}",
        f"CPPFLAGS={cppflags}",
        f"LDFLAGS={ldflags}",
        f"LIBS={libs}",
        "MPI_LAUNCH=mpiexec",
        "--disable-grib2",
        "--enable-loop-exchange",
        "--enable-openmp",
        "--enable-bundled-python=mtime",
        "--disable-jsbach",
        "--disable-ocean",
        "--disable-coupling",
        "--disable-waves",
    ]
    subprocess.run(configure_cmd, cwd=build_dir, check=True)

    # --- make --------------------------------------------------------
    subprocess.run(["make", f"-j{jobs}"], cwd=build_dir, check=True)

    if not _have_mods(build_dir):
        raise RuntimeError(f"ICON make completed but no .mod files appeared under "
                           f"{build_dir / 'mod'} -- the build recipe may need updating.")

    _BUILT[key] = build_dir
    return build_dir
