"""Parse the ICON dynamical core (``mo_solve_nonhydro::solve_nh``) through
the bridge -- the tier-3 / automake integration end to end.

Reproducible setup (a runner script ships alongside this test):
``tests/icon/dycore/setup_icon_dycore.sh`` apt-installs the deps, shallow
-clones ICON at the pinned tag, inits submodules recursively (JSBACH, yaxt,
cdi, mtime, tixi, ... -- the externals the build reads), drops in the
``config/generic/flang`` wrapper, configures, and runs ``bear -- make`` to
capture ``compile_commands.json``.  The build is allowed to fail late: we
only need flang to **cpp-preprocess + emit HLFIR** for the dycore TU and its
USE-closure, not a working ICON binary -- so the test asserts the dycore
HLFIR was emitted, not that the SDFG fully builds.

The dycore's USE-closure reaches two modules flang has no readable ``.mod``
for: ``mpi`` (kept on -- the halo-exchange / collective structure is
preserved for the MPI library nodes) and ``netcdf`` (a purely structural
dependency: ``mo_solve_nonhydro`` -> ``mo_grid_config`` -> ``mo_netcdf``).
The flang-buildable ``stubs/`` next to this test supply those ``.mod`` files;
their bodies are never lowered.

Gated on ``ICON_DYCORE_CC`` (path to the captured ``compile_commands.json``)
so a normal sweep skips it; a configured runner sets the env var.  This stops
at HLFIR emission on purpose -- lowering the 60k-line ``solve_nh`` to an SDFG
is a separate (and currently unoptimised) step, so no SDFG is built here.
"""
import os
import re
from pathlib import Path

import pytest

from _util import have_flang

_ENTRY = "mo_solve_nonhydro::solve_nh"  # friendly name; emit() resolves it
_ENTRY_SYM = "solve_nh"  # its mangled flang symbol
_STUBS_DIR = Path(__file__).parent / "stubs"


def _resolve_compile_commands() -> Path | None:
    """Find a built ICON ``compile_commands.json`` for the parse test.

    Resolution order:

    1. ``ICON_DYCORE_CC`` environment variable -- explicit override, the
       canonical CI / runner contract.
    2. The in-test build dir ``tests/icon/dycore/.icon_build/
       compile_commands.json``.  Matches what :file:`setup_icon_dycore.sh`
       produces by default (``BUILD_DIR=<this-dir>/.icon_build``); keeps
       the build state owned by the test rather than scattered at the
       workspace root.

    The first hit that names a readable regular file wins.  ``None``
    means no DB is reachable on this host -- the ``skipif`` below
    surfaces the standard ``setup_icon_dycore.sh`` pointer."""
    env = os.environ.get("ICON_DYCORE_CC")
    if env and Path(env).is_file():
        return Path(env)
    # ``__file__`` -> tests/icon/dycore/<this>; ``.icon_build`` lives
    # next to this test file as the per-test scratch dir.
    in_test = Path(__file__).resolve().parent / ".icon_build" / "compile_commands.json"
    if in_test.is_file():
        return in_test
    return None


_SETUP_HINT = ("set ICON_DYCORE_CC to a built compile_commands.json, "
               "or run tests/icon/dycore/setup_icon_dycore.sh "
               "(populates tests/icon/dycore/.icon_build/compile_commands.json)")

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(_resolve_compile_commands() is None, reason=_SETUP_HINT),
]


def test_solve_nonhydro_emits_hlfir(tmp_path):
    """flang cpp-preprocesses + emits HLFIR for the dycore, driven entirely
    by the per-TU -cpp / -I / -D flags from compile_commands.json, with the
    ``mpi`` / ``netcdf`` stubs standing in for their unreadable ``.mod``."""
    # Re-resolve at test-run time: the ``skipif`` above is evaluated at
    # collection (import) time, but a long ``-n N`` xdist sweep can sit on
    # this worker's queue for many minutes before the test runs, and an
    # interleaved disk-cleanup (very real on the dev box) can delete the
    # ``_icon_build`` directory between import and call.  A stale module-
    # level resolution would then turn a benign teardown into a hard
    # FileNotFoundError; instead, resolve again and skip cleanly.
    cc = _resolve_compile_commands()
    if cc is None:
        pytest.skip(_SETUP_HINT)
    from dace_fortran.emit_hlfir import emit
    stubs = [_STUBS_DIR / "mpi_stub.f90", _STUBS_DIR / "netcdf_stub.f90"]
    # entry= restricts the ~900-TU ICON database to solve_nh's USE-closure;
    # the plain Fortran name is resolved to the mangled symbol from the sources.
    out = emit(compile_commands=cc, out_dir=tmp_path / "hlfir", stubs=stubs, entry=_ENTRY)
    # The dycore's func must appear in one of the emitted .hlfir files --
    # i.e. flang got through the preprocessor + frontend on solve_nh.
    func_re = re.compile(rf"func\.func\s+(?!private\b)@{re.escape(_ENTRY_SYM)}\s*\(")
    assert any(func_re.search(p.read_text()) for p in out if p.suffix == ".hlfir"), \
        f"no emitted .hlfir defines {_ENTRY_SYM} ({len(out)} TUs emitted)"
