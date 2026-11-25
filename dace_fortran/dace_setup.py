# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Switch a PyPI install of DaCe to the DaCe branch this frontend needs.

The PyPI package depends on the released ``dace`` (built from DaCe main), because PyPI refuses git dependencies. The
frontend needs DaCe-core pieces that live on a branch until it merges into DaCe main. On the first
``import dace_fortran``, :func:`ensure_dace` replaces a PyPI DaCe with that branch, if the branch still exists. Once
the branch has merged and been deleted, the released DaCe is the right one and nothing is replaced.

Set ``DACE_FORTRAN_NO_DACE_SETUP=1`` to turn the automatic switch off. ``dace-fortran-setup-dace`` runs it by hand.
"""

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys

DACE_REPOSITORY = "https://github.com/spcl/dace.git"
# DaCe branch v1.2: FaCe plus the symbol repository; FaCe alone lacks pieces this frontend calls.
DACE_BRANCH = os.environ.get("DACE_FORTRAN_DACE_BRANCH", "v1.2")
OPT_OUT = "DACE_FORTRAN_NO_DACE_SETUP"


def installed_from_pypi() -> bool:
    """True when the installed ``dace`` came from an index (PyPI), not from git, a path, or an editable checkout.

    Pip and uv record a ``direct_url.json`` (PEP 610) for every non-index install; an index install has none.
    """
    try:
        dist = importlib.metadata.distribution("dace")
    except importlib.metadata.PackageNotFoundError:
        return False
    return dist.read_text("direct_url.json") is None


def branch_exists(branch: str) -> bool:
    """True when ``branch`` exists in the DaCe repository. False when it does not or git cannot reach it."""
    git = shutil.which("git")
    if git is None:
        return False
    try:
        out = subprocess.run(
            [git, "ls-remote", "--heads", DACE_REPOSITORY, branch],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def install_command(branch: str) -> list[str]:
    """The command that installs DaCe ``branch`` into this interpreter's environment (pip, else uv)."""
    requirement = f"dace @ git+{DACE_REPOSITORY}@{branch}"
    # uv-created environments have no pip.
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", requirement]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(f"neither pip nor uv is available to install {requirement!r}")
    return [uv, "pip", "install", "--python", sys.executable, "--no-deps", "--reinstall", requirement]


def ensure_dace(force: bool = False) -> bool:
    """Install DaCe :data:`DACE_BRANCH` in place of a PyPI DaCe. Returns True when it installed it.

    Does nothing when the user opted out, when DaCe came from git, a path, or an editable checkout (a developer's or
    CI's choice), or when the branch no longer exists (merged: the released DaCe is right). ``force`` skips the opt-out
    and the origin check.
    """
    if not force and (os.environ.get(OPT_OUT) or not installed_from_pypi()):
        return False
    if "dace" in sys.modules:
        raise RuntimeError(
            "dace is already imported, so it cannot be replaced; import dace_fortran first or run "
            "dace-fortran-setup-dace once"
        )
    if not branch_exists(DACE_BRANCH):
        return False
    command = install_command(DACE_BRANCH)
    print(f"dace-fortran: installing DaCe branch {DACE_BRANCH} ({OPT_OUT}=1 turns this off)", file=sys.stderr)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"installing DaCe branch {DACE_BRANCH} failed; run by hand: {' '.join(command)}")
    importlib.invalidate_caches()
    return True


def main() -> int:
    """Console entry ``dace-fortran-setup-dace``: install DaCe :data:`DACE_BRANCH` now."""
    installed = ensure_dace(force=True)
    dist = importlib.metadata.distribution("dace")
    origin = json.loads(dist.read_text("direct_url.json") or "{}").get("url", "PyPI")
    state = "installed" if installed else f"left as is (branch {DACE_BRANCH} not found)"
    print(f"dace {dist.version} from {origin}: {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
