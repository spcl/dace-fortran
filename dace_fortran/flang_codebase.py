"""Feed flang-new an entry point from a real Fortran codebase (ICON, ECRAD,
IFS, ...) without rebuilding it against flang's own ``.mod`` files: upstream
libs ship gfortran-format ``.mod``s only, so :func:`prepare_flang_translation_unit`
resolves the USE graph, stubs those libraries, patches flang-21 false positives,
and replays the project's own ``-D``/``-I`` flags.
"""
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .preprocess import merge_used_modules

# ---------------------------------------------------------------------------
# 1. Library stubs.
# ---------------------------------------------------------------------------

# MODULE mpi wrapping OpenMPI's mpif-*.h. Skips mpif-sizeof.h (flang-21
# rejects its COMPLEX(KIND=-1)). Adds an interface for mpi_get_library_version
# (only in mpi.h, not mpif.h). Pair with -I<openmpi-include-dir>.
_MPI_STUB_SOURCE = """\
! Stub ``MODULE mpi`` for the flang frontend.  Wraps OpenMPI's
! ``mpif-*.h`` parameter / handle / sentinel headers (a flang-21
! compatible subset of ``mpif.h``) and adds explicit interfaces for
! procedures the F77 header omits.  Runtime symbols are resolved at
! link time against ``libmpi`` -- this module only has to satisfy
! the flang semantic checker.
MODULE mpi
  IMPLICIT NONE
  PUBLIC
  INCLUDE 'mpif-config.h'
  INCLUDE 'mpif-constants.h'
  INCLUDE 'mpif-handles.h'
  INCLUDE 'mpif-io-constants.h'
  INCLUDE 'mpif-io-handles.h'
  INCLUDE 'mpif-externals.h'
  INCLUDE 'mpif-sentinels.h'
  ! Skipped: mpif-sizeof.h (COMPLEX(KIND=-1) tripped flang).
  INTERFACE
    SUBROUTINE mpi_get_library_version(version, resultlen, ierror)
      CHARACTER(LEN=*), INTENT(OUT) :: version
      INTEGER, INTENT(OUT) :: resultlen
      INTEGER, INTENT(OUT) :: ierror
    END SUBROUTINE mpi_get_library_version
  END INTERFACE
END MODULE mpi
"""

# Probe locations for an OpenMPI install; first one with mpif-config.h wins.
# Callers can override via prepare_flang_translation_unit.
_OPENMPI_INCLUDE_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/openmpi/include",
    "/usr/include/openmpi",
    "/usr/local/include/openmpi",
    "/opt/openmpi/include",
)


def find_openmpi_include() -> Optional[str]:
    """Returns the first directory in :data:`_OPENMPI_INCLUDE_CANDIDATES`
    that contains ``mpif-config.h``, or ``None`` if no install probes."""
    for d in _OPENMPI_INCLUDE_CANDIDATES:
        if (Path(d) / "mpif-config.h").is_file():
            return d
    return None


def mpi_stub_source() -> str:
    """Fortran source for ``MODULE mpi`` wrapping OpenMPI's headers.
    Pair with the ``-I`` dir from :func:`find_openmpi_include`."""
    return _MPI_STUB_SOURCE


# Not vendored (4 MB, updates yearly): fetched from upstream on first use
# and cached, same layout as the bridge's own build (~/.cache/dace-fortran).
_NETCDF_FORTRAN_URL = ("https://github.com/Unidata/netcdf-fortran/archive/refs/tags/v{version}.tar.gz")
_NETCDF_FORTRAN_DEFAULT_VERSION = "4.6.2"


def vendor_netcdf_fortran(cache_dir: Path, version: str = _NETCDF_FORTRAN_DEFAULT_VERSION) -> Path:
    """Ensure netcdf-fortran source is unpacked under ``cache_dir``.
    Returns the path that should be added to flang's ``-I``.

    Downloads from upstream's github release on first call; idempotent
    after.  Flang ingests ``netcdf4.F90`` plus its ``#include``-d
    helper files cleanly with ``-cpp``, no extra patches needed.

    :param cache_dir: where to extract the tarball.
    :param version: upstream release tag (default 4.6.2).
    :returns: the ``fortran/`` subdirectory of the unpacked source.
    """
    cache_dir = Path(cache_dir)
    target = cache_dir / f"netcdf-fortran-{version}"
    fortran_dir = target / "fortran"
    if fortran_dir.is_dir():
        return fortran_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _NETCDF_FORTRAN_URL.format(version=version)
    tarball = cache_dir / f"netcdf-fortran-{version}.tar.gz"
    if not tarball.is_file():
        with urllib.request.urlopen(url) as r, tarball.open("wb") as f:
            shutil.copyfileobj(r, f)
    with tarfile.open(tarball) as t:
        t.extractall(cache_dir)
    if not fortran_dir.is_dir():
        raise RuntimeError(f"netcdf-fortran tarball did not extract a fortran/ "
                           f"subdirectory under {target}")
    return fortran_dir


def netcdf_stub_source(fortran_dir: Path) -> str:
    """Returns Fortran source that defines ``MODULE typesizes`` and
    ``MODULE netcdf`` by concatenating netcdf-fortran's own
    ``module_typesizes.F90`` and ``netcdf4.F90``.  Use together with
    ``-I <fortran_dir>`` so the ``#include`` directives in
    ``netcdf4.F90`` resolve.

    :param fortran_dir: directory returned by
        :func:`vendor_netcdf_fortran` (or an equivalent local copy).
    """
    fortran_dir = Path(fortran_dir)
    return ((fortran_dir / "module_typesizes.F90").read_text() + "\n" + (fortran_dir / "netcdf4.F90").read_text())


# Registry: name -> (stub source provider, include-path provider).  The
# stub source goes into the merged TU; the include-path provider
# returns a list of ``-I<dir>`` flags to append to flang's command line.
@dataclass(frozen=True)
class LibraryStub:
    """A pluggable wrapper for an upstream Fortran library that ships
    only binary ``.mod`` files.  See :data:`LIBRARY_STUBS` for the
    built-in set."""
    name: str
    #: Returns Fortran source defining the library's module(s).
    source: Callable[..., str]
    #: Returns ``-I<dir>`` flags flang needs alongside the source.
    flags: Callable[..., List[str]]


def _mpi_flags(openmpi_include: Optional[str] = None, **_) -> List[str]:
    """``-I`` flags for the MPI stub.  Auto-probes a standard OpenMPI
    install when ``openmpi_include`` isn't passed."""
    inc = openmpi_include or find_openmpi_include()
    if inc is None:
        raise RuntimeError("MPI stub needs OpenMPI's include directory (with "
                           "mpif-config.h).  Pass openmpi_include=... or install "
                           "libopenmpi-dev.")
    return [f"-I{inc}"]


def _netcdf_source(cache_dir: Optional[Path] = None, **_) -> str:
    """Source for the netcdf stub.  Requires ``cache_dir`` to vendor
    the upstream tarball."""
    if cache_dir is None:
        raise ValueError("netcdf stub needs cache_dir for the vendored "
                         "netcdf-fortran source.")
    return netcdf_stub_source(vendor_netcdf_fortran(cache_dir))


def _netcdf_flags(cache_dir: Optional[Path] = None, **_) -> List[str]:
    """``-I`` flags for the netcdf stub.  Same ``cache_dir`` as
    :func:`_netcdf_source`."""
    if cache_dir is None:
        raise ValueError("netcdf stub needs cache_dir.")
    return [f"-I{vendor_netcdf_fortran(cache_dir)}"]


LIBRARY_STUBS: Dict[str, LibraryStub] = {
    "mpi": LibraryStub(name="mpi", source=lambda **_: mpi_stub_source(), flags=_mpi_flags),
    "netcdf": LibraryStub(name="netcdf", source=_netcdf_source, flags=_netcdf_flags),
}

# ---------------------------------------------------------------------------
# 2. Flang-21 bug patches (Fortran-text rewrites).
# ---------------------------------------------------------------------------

# ``CALL MPI_SIZEOF(arg, sz, err)`` -- flang-21's generic resolution
# can't disambiguate the overload set in OpenMPI's ``mpif-sizeof.h``.
# Real callers use this only to learn the byte size of a built-in
# type, which is statically known, so we substitute the constant.
_MPI_SIZEOF_RE = re.compile(
    r"(\s*)CALL\s+MPI_SIZEOF\s*\(\s*([A-Za-z_]\w*)\s*,\s*"
    r"([A-Za-z_]\w*)\s*,\s*([A-Za-z_]\w*)\s*\)", re.IGNORECASE)


def patch_mpi_sizeof(source: str) -> str:
    """Replace ``CALL MPI_SIZEOF(arg, sz, err)`` with a static
    assignment ``sz = <bytes>; err = 0``.  The byte count is inferred
    from the argument name (``*i4`` / ``*sp`` -> 4, anything else
    -> 8 since ICON / IFS / ECRAD use double precision everywhere).
    Project-independent: any code using OpenMPI's ``MPI_SIZEOF`` hits
    the same flang false positive."""

    def _replace(m: re.Match) -> str:
        indent, arg, sz, err = m.group(1), m.group(2), m.group(3), m.group(4)
        small = any(k in arg.lower() for k in ("i4", "sp"))
        sz_val = 4 if small else 8
        return (f"{indent}{sz} = {sz_val}; {err} = 0  "
                f"! flang-21 stub for MPI_SIZEOF({arg})")

    return _MPI_SIZEOF_RE.sub(_replace, source)


# Patch registry: name -> source-to-source transform.
FLANG_BUG_PATCHES: Dict[str, Callable[[str], str]] = {
    "mpi_sizeof": patch_mpi_sizeof,
}

# ---------------------------------------------------------------------------
# 3. Compile-arg extraction from a make / cmake build.
# ---------------------------------------------------------------------------

# A ``-Dfoo`` or ``-Dfoo=bar`` argument inside a compile line.
_DEFINE_RE = re.compile(r"(?<!\S)-D([A-Za-z_][\w]*(?:=\S+)?)")
# An ``-I<dir>`` or ``-I <dir>`` argument inside a compile line.
_INCLUDE_RE = re.compile(r"(?<!\S)-I(?:\s+|=?)(\S+)")
# Last non-flag token on the compile line is conventionally the source.
_FORTRAN_SOURCE_RE = re.compile(r"(\S+\.[fF]9?0)(?:\s|$)")


def extract_make_compile_args(makefile_dir: Path, target: str, make_program: str = "make") -> dict:
    """Run ``make -n -B <target>`` in ``makefile_dir`` and parse the
    compile command for the source's ``-D`` defines, ``-I`` include
    dirs, and source path.

    Project-agnostic: works for any GNU-make-driven Fortran build that
    invokes a single compiler line per source.  CMake-generated
    Makefiles are fine because cmake emits one ``mpifort ...`` /
    ``gfortran ...`` per source file.

    :param makefile_dir: directory containing the ``Makefile``.
    :param target: rule name to extract (typically the ``.o`` path,
        e.g. ``src/foo/mo_bar.o``).
    :param make_program: ``make`` binary to invoke (override for
        ``gmake`` on BSD / macOS Homebrew).
    :returns: dict with keys ``defines`` (list[str]), ``include_dirs``
        (list[Path]), ``source`` (Path), ``command`` (str, the raw
        compile line).
    :raises RuntimeError: if no Fortran compile line is found.
    """
    # ``make -n -B`` would also re-emit recipes for the target's
    # prerequisites and can fail on out-of-tree external builds.
    # Force a re-emit only for ``target`` by deleting its product
    # first; ``make -n`` then prints the recipe and exits 0.
    artefact = makefile_dir / target
    if artefact.is_file():
        artefact.unlink()
    out = subprocess.check_output([make_program, "-n", target],
                                  cwd=str(makefile_dir),
                                  stderr=subprocess.STDOUT,
                                  text=True)
    # Pick the first line that mentions a Fortran source -- ICON's
    # ``mpifort -c ... mo_velocity_advection.f90`` style.
    compile_line = None
    for ln in out.splitlines():
        if _FORTRAN_SOURCE_RE.search(ln) and (" -c " in ln or "-c " in ln.split()[0:]):
            compile_line = ln
            break
    if compile_line is None:
        # Fallback: any line with both ``-c`` and a .f90.
        for ln in out.splitlines():
            if "-c" in ln and _FORTRAN_SOURCE_RE.search(ln):
                compile_line = ln
                break
    if compile_line is None:
        raise RuntimeError(f"could not find a Fortran compile line for {target!r} in the "
                           f"output of `{make_program} -n -B`")
    defines = sorted(set(_DEFINE_RE.findall(compile_line)))
    include_dirs = [Path(p) for p in _INCLUDE_RE.findall(compile_line)]
    src_match = _FORTRAN_SOURCE_RE.search(compile_line)
    return {
        "defines": defines,
        "include_dirs": include_dirs,
        "source": Path(src_match.group(1)),
        "command": compile_line.strip(),
    }


# ---------------------------------------------------------------------------
# 4. The composer.
# ---------------------------------------------------------------------------


def prepare_flang_translation_unit(
    entry_source: str,
    *,
    search_dirs: Sequence[Path] = (),
    library_stubs: Sequence[str] = (),
    patches: Sequence[str] = ("mpi_sizeof", ),
    defines: Sequence[str] = (),
    include_dirs: Sequence[Path] = (),
    cache_dir: Optional[Path] = None,
    openmpi_include: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Stitch a flang-ready translation unit from a real-world Fortran
    codebase.  Composes four steps:

    1. Resolves the entry source's ``USE`` graph against
       ``search_dirs`` via :func:`merge_used_modules`.
    2. Prepends each opted-in :data:`LIBRARY_STUBS` entry (Fortran
       wrappers for libraries that ship only binary ``.mod`` files).
    3. Applies opted-in :data:`FLANG_BUG_PATCHES` (text rewrites
       routing around flang-21 ICEs and false-positive errors).
    4. Returns the merged source plus the ``-D`` / ``-I`` flag list
       flang needs (the caller's own defines + the stubs' includes).

    :param entry_source: Fortran source of the entry point.
    :param search_dirs: directories scanned by
        :func:`merge_used_modules` for sibling modules.
    :param library_stubs: names of upstream-library stubs to opt into;
        keys of :data:`LIBRARY_STUBS` (``'mpi'``, ``'netcdf'``).
    :param patches: names of flang-21 text patches to apply; keys of
        :data:`FLANG_BUG_PATCHES`.  Default is the single safe one
        (``'mpi_sizeof'``) -- only matches the literal call pattern
        so it's a no-op on codebases that never call ``MPI_SIZEOF``.
    :param defines: user's project-specific ``-D`` defines, e.g.
        ``['__NO_JSBACH__', 'ICON_MPI_VERSION=3']``.  Typically
        sourced from :func:`extract_make_compile_args`.
    :param include_dirs: user's project-specific ``-I`` directories
        (the project's own ``src/include`` etc.).
    :param cache_dir: scratch directory for stubs that vendor external
        source (``netcdf``).  Required only when that stub is opted
        into.
    :param openmpi_include: override the auto-detected OpenMPI include
        directory for the ``mpi`` stub.
    :returns: ``(merged_source, flang_extra_flags)``.  The flag list
        already contains ``-D`` for ``defines``, ``-I`` for
        ``include_dirs``, and the stubs' own ``-I`` paths.  Add
        ``-fc1 -cpp -emit-hlfir`` and the entry to the flang command
        yourself.
    :raises KeyError: an unknown library stub or patch name.
    """
    pieces: List[str] = []
    flags: List[str] = list(f"-D{d}" for d in defines)
    flags.extend(f"-I{p}" for p in include_dirs)

    for name in library_stubs:
        try:
            stub = LIBRARY_STUBS[name]
        except KeyError as e:
            raise KeyError(f"unknown library stub {name!r}; available: "
                           f"{sorted(LIBRARY_STUBS)}") from e
        pieces.append(stub.source(cache_dir=cache_dir, openmpi_include=openmpi_include))
        flags.extend(stub.flags(cache_dir=cache_dir, openmpi_include=openmpi_include))

    pieces.append(merge_used_modules(entry_source, search_dirs=search_dirs))
    source = "\n".join(pieces)

    for name in patches:
        try:
            source = FLANG_BUG_PATCHES[name](source)
        except KeyError as e:
            raise KeyError(f"unknown flang patch {name!r}; available: "
                           f"{sorted(FLANG_BUG_PATCHES)}") from e

    return source, flags


# ---------------------------------------------------------------------------
# 5. Driving flang directly with the composed TU.
# ---------------------------------------------------------------------------


def emit_hlfir_from_codebase(
        entry_source: str,
        out_path: Path,
        *,
        search_dirs: Sequence[Path] = (),
        library_stubs: Sequence[str] = (),
        patches: Sequence[str] = ("mpi_sizeof", ),
        defines: Sequence[str] = (),
        include_dirs: Sequence[Path] = (),
        cache_dir: Optional[Path] = None,
        openmpi_include: Optional[str] = None,
        flang_program: str = "flang-new-21",
        extra_flang_flags: Sequence[str] = (),
) -> Path:
    """Compose a translation unit for ``entry_source`` via
    :func:`prepare_flang_translation_unit`, write it next to
    ``out_path``, and run flang with the resulting flag set to emit
    an ``.hlfir`` at ``out_path``.

    The returned ``.hlfir`` can be passed to
    :func:`dace_fortran.build_sdfg_from_hlfir` to build an SDFG.

    :param entry_source: Fortran source of the entry point.
    :param out_path: where flang writes the ``.hlfir``.  The composed
        ``.F90`` is written next to it as ``out_path.with_suffix('.F90')``.
    :returns: ``out_path`` on success.
    :raises subprocess.CalledProcessError: if flang's exit code is
        non-zero (its stderr is forwarded so the failure is visible).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tu_source, flang_flags = prepare_flang_translation_unit(
        entry_source,
        search_dirs=search_dirs,
        library_stubs=library_stubs,
        patches=patches,
        defines=defines,
        include_dirs=include_dirs,
        cache_dir=cache_dir,
        openmpi_include=openmpi_include,
    )
    tu_path = out_path.with_suffix(".F90")
    tu_path.write_text(tu_source)
    # Run flang with ``cwd = out_path.parent`` so its default module
    # search (which includes ".") doesn't pick up stale gfortran-format
    # ``.mod`` files left in the caller's working directory from prior
    # test runs.  Flang's ``.mod`` format is incompatible with
    # gfortran's; a chance collision surfaces as ``Cannot use module
    # file for module X``.
    subprocess.check_call([
        flang_program,
        "-fc1",
        "-cpp",
        "-U_OPENMP",
        "-U_OPENACC",
        *flang_flags,
        *extra_flang_flags,
        "-emit-hlfir",
        str(tu_path),
        "-o",
        str(out_path),
    ],
                          cwd=str(out_path.parent))
    return out_path
