"""Parse the ICON dynamical core (mo_solve_nonhydro::solve_nh) through the bridge
-- tier-3/automake integration end to end. setup_icon_dycore.sh clones ICON,
inits submodules, configures, and runs `bear -- make` to capture
compile_commands.json; the build may fail late since we only need flang to
cpp-preprocess + emit HLFIR for the dycore TU, not a working ICON binary.
``mpi``/``netcdf`` (unreadable .mod for flang) are covered by flang-buildable
stubs/ next to this test; their bodies are never lowered. Gated on
ICON_DYCORE_CC; stops at HLFIR emission on purpose -- no SDFG is built here.
"""
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from _util import have_flang

_ENTRY = "mo_solve_nonhydro::solve_nh"  # friendly name; emit() resolves it
_ENTRY_MODULE = "mo_solve_nonhydro"  # the defining module
_ENTRY_PROC = "solve_nh"  # the plain Fortran procedure name
_STUBS_DIR = Path(__file__).parent / "stubs"

# flang mangles a module procedure to _QM<module>P<proc> (sub) or ...F<proc> (func);
# match the mangled symbol, or a bare name as a defensive fallback.
_ENTRY_DEF_RE = re.compile(rf"func\.func\s+(?!private\b)@(?:_QM{_ENTRY_MODULE}[PF])?{_ENTRY_PROC}\s*\(", re.IGNORECASE)


def _resolve_compile_commands() -> Path | None:
    """Find a built ICON compile_commands.json: ICON_DYCORE_CC env var first, else the
    in-test build dir tests/icon/dycore/.icon_build/compile_commands.json. None means
    no DB reachable -- skipif below surfaces the setup_icon_dycore.sh pointer."""
    env = os.environ.get("ICON_DYCORE_CC")
    if env and Path(env).is_file():
        return Path(env)
    # .icon_build lives next to this test file as the per-test scratch dir.
    in_test = Path(__file__).resolve().parent / ".icon_build" / "compile_commands.json"
    if in_test.is_file():
        return in_test
    return None


_SETUP_HINT = ("set ICON_DYCORE_CC to a built compile_commands.json, "
               "or run tests/icon/dycore/setup_icon_dycore.sh "
               "(populates tests/icon/dycore/.icon_build/compile_commands.json)")

# Heavy CI lane generates compile_commands.json before the long run, so this
# belongs in the `long` lane (fast lane has no such DB).
pytestmark = [
    pytest.mark.long,
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(_resolve_compile_commands() is None, reason=_SETUP_HINT),
]


def test_solve_nonhydro_emits_hlfir():
    """flang cpp-preprocesses + emits HLFIR for the dycore, driven by the per-TU
    flags from compile_commands.json, with mpi/netcdf stubs standing in for unreadable .mod."""
    # Re-resolve at run time (not collection time): a long xdist sweep can sit queued
    # while a disk-cleanup deletes _icon_build, turning a stale resolution into a hard
    # FileNotFoundError; resolve again and skip cleanly instead.
    cc = _resolve_compile_commands()
    if cc is None:
        pytest.skip(_SETUP_HINT)
    from dace_fortran.emit_hlfir import emit
    stubs = [_STUBS_DIR / "mpi_stub.f90", _STUBS_DIR / "netcdf_stub.f90"]
    # ~150-TU USE-closure (hundreds of MB of .hlfir/.mod) into a private scratch dir,
    # torn down fully after -- only the entry symbol's presence is asserted;
    # ignore_errors avoids a teardown race (concurrent disk sweep) turning a green run red.
    scratch = tempfile.mkdtemp(prefix="dycore_hlfir_")
    try:
        # entry= restricts the ~900-TU ICON DB to solve_nh's USE-closure; resolved to the mangled symbol.
        out = emit(compile_commands=cc, out_dir=Path(scratch), stubs=stubs, entry=_ENTRY)
        # dycore's func must appear in an emitted .hlfir -- i.e. flang got through preprocessor + frontend.
        assert any(_ENTRY_DEF_RE.search(p.read_text()) for p in out if p.suffix == ".hlfir"), \
            f"no emitted .hlfir defines {_ENTRY_PROC} ({len(out)} TUs emitted)"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
