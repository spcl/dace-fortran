"""Parse the ICON dynamical core (``mo_solve_nonhydro::solve_nh``) through
the bridge -- the tier-3 / automake integration end to end.

Reproducible setup (a runner script ships alongside this test):
``tests/icon_dycore/setup_icon_dycore.sh`` apt-installs the deps, shallow
-clones ICON at the pinned tag, inits submodules recursively (JSBACH, yaxt,
cdi, mtime, tixi, ... -- the externals the build reads), drops in the
``config/generic/flang`` wrapper, configures, and runs ``bear -- make`` to
capture ``compile_commands.json``.  The build is allowed to fail late: we
only need flang to **cpp-preprocess + emit HLFIR** for the dycore TU and its
USE-closure, not a working ICON binary -- so the test asserts the dycore
HLFIR was emitted, not that the SDFG fully builds.

Gated on ``ICON_DYCORE_CC`` (path to the captured ``compile_commands.json``)
so a normal sweep skips it; a configured runner sets the env var.
"""
import os
import re
from pathlib import Path

import pytest

from _util import have_flang

_CC = os.environ.get("ICON_DYCORE_CC")
_ENTRY = "_QMmo_solve_nonhydroPsolve_nh"  # mo_solve_nonhydro :: solve_nh

pytestmark = [
    pytest.mark.skipif(not have_flang(), reason="flang-new-21 not on PATH"),
    pytest.mark.skipif(not _CC or not Path(_CC).is_file(),
                       reason="set ICON_DYCORE_CC to a built compile_commands.json "
                              "(see setup_icon_dycore.sh)"),
]


def test_solve_nonhydro_emits_hlfir(tmp_path):
    """flang cpp-preprocesses + emits HLFIR for the dycore, driven entirely
    by the per-TU -cpp / -I / -D flags from compile_commands.json."""
    from dace_fortran.emit_hlfir import emit
    out = emit(compile_commands=Path(_CC), out_dir=tmp_path / "hlfir")
    # The dycore's func must appear in one of the emitted .hlfir files --
    # i.e. flang got through the preprocessor + frontend on solve_nh.
    func_re = re.compile(rf"func\.func\s+(?!private\b)@{re.escape(_ENTRY)}\s*\(")
    assert any(func_re.search(p.read_text()) for p in out if p.suffix == ".hlfir"), \
        f"no emitted .hlfir defines {_ENTRY} ({len(out)} TUs emitted)"
