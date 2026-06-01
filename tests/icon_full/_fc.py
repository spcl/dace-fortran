"""Shared Fortran-compiler discovery + pytest parametrization helper.

The iso_c wrappers + the ``solve_nh`` patch must parse cleanly through
every Fortran compiler ICON ships configure files for: gfortran (the
default CPU build), flang-new (the HLFIR frontend's own compiler --
syntax-only checks need the same dialect), and nvfortran (NVHPC's
compiler, used by ICON's GPU build).

This module locates each one on the host (PATH + the standard NVHPC
install prefix), and exposes a ``FORTRAN_COMPILERS`` parametrize list
that tests can ``@pytest.mark.parametrize`` over.  The selected
compiler is also surfaced through ``$FC`` so build-helper scripts that
honour the standard make variable pick the right one in-process.
"""
import os
import shutil
from pathlib import Path
from typing import List, Optional

import pytest


# Standard prefixes the NVIDIA HPC SDK installs into.  ``nvhpc-25-7``
# on Ubuntu lands under ``/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/...``;
# the version segment varies by release.
_NVHPC_PREFIXES = (
    "/opt/nvidia/hpc_sdk/Linux_x86_64",
    "/opt/nvhpc/Linux_x86_64",
    "/usr/local/nvhpc/Linux_x86_64",
)


def _find_nvfortran() -> Optional[Path]:
    """Locate nvfortran on PATH or under the standard NVHPC prefixes.
    Picks the newest version dir when several are installed."""
    p = shutil.which("nvfortran")
    if p is not None:
        return Path(p)
    for prefix in _NVHPC_PREFIXES:
        prefix_path = Path(prefix)
        if not prefix_path.is_dir():
            continue
        versions = sorted(
            (d for d in prefix_path.iterdir() if d.is_dir()),
            key=lambda d: d.name, reverse=True)
        for version_dir in versions:
            candidate = version_dir / "compilers" / "bin" / "nvfortran"
            if candidate.is_file():
                return candidate
    return None


def _find_flang() -> Optional[Path]:
    """Locate flang.  Prefer ``flang-new-21`` (the bridge's pinned
    version) over ``flang-new`` over ``flang``."""
    for name in ("flang-new-21", "flang-new", "flang"):
        p = shutil.which(name)
        if p is not None:
            return Path(p)
    return None


def _find_gfortran() -> Optional[Path]:
    p = shutil.which("gfortran")
    return Path(p) if p else None


def discover_fortran_compilers() -> dict:
    """Return ``{display_name: absolute_path}`` for every Fortran
    compiler available on this host.  Display names are stable across
    versions so pytest test IDs stay diffable."""
    out: dict = {}
    gfortran = _find_gfortran()
    if gfortran:
        out["gfortran"] = gfortran
    flang = _find_flang()
    if flang:
        out["flang-new-21"] = flang
    nvfortran = _find_nvfortran()
    if nvfortran:
        out["nvfortran"] = nvfortran
    return out


def _make_params() -> List:
    """Build the ``pytest.param(...)`` list for the discovered set."""
    found = discover_fortran_compilers()
    if not found:
        return [pytest.param(
            ("gfortran", "gfortran"),
            id="gfortran",
            marks=[pytest.mark.skip(
                reason="no Fortran compiler found on the host")])]
    return [pytest.param((name, str(path)), id=name)
            for name, path in found.items()]


#: Parametrize value for tests that drive a Fortran compiler.  Each
#: pytest case receives a ``(name, executable_path)`` tuple.  Use
#: ``@pytest.mark.parametrize("fc", FORTRAN_COMPILERS)`` -- the
#: fixture handles the no-compiler-available case via per-param skip
#: marks so the test reports cleanly on any host.
FORTRAN_COMPILERS = _make_params()


def fortran_compiler_flags(fc_name: str) -> List[str]:
    """Per-compiler base flags for the wrapper-syntax checks.

    gfortran: ``-ffree-line-length-none`` lets long generated lines
    through (gfortran's default 132-col limit otherwise rejects).
    flang-new: doesn't enforce the column limit AND rejects the
    gfortran-specific flag, so pass nothing.  nvfortran: same -- its
    free-form line length is generous enough out of the box.
    """
    if fc_name == "gfortran":
        return ["-ffree-line-length-none"]
    return []


# Standard locations where a user-local ``libflang_rt.runtime.a`` may
# live.  When we find one we surface it via ``LIBRARY_PATH`` so the
# ``flang-new-21`` driver's linker invocation picks it up without the
# user having to set the env var themselves.  The build dir from
# ``runtimes/`` and a few hand-rolled install prefixes get probed.
_FLANG_RT_DIRS = (
    str(Path.home() / ".local/llvm-flang-rt-21/lib/clang/21/lib/x86_64-unknown-linux-gnu"),
    str(Path.home() / ".local/lib/clang/21/lib/x86_64-unknown-linux-gnu"),
    # ROCm's runtime is ABI-compatible with flang-21 (LLVM-22-era
    # built static archive; symbols are stable).  Probed last so a
    # user-local build wins when both are present.
    "/opt/rocm-7.2.0/lib/llvm/lib/clang/22/lib/x86_64-unknown-linux-gnu",
)


def find_flang_runtime_dir() -> Optional[str]:
    """Return the first directory on the host that has
    ``libflang_rt.runtime.a`` (the static archive flang-21's linker
    invocation references as ``-lflang_rt.runtime``), or ``None`` if
    no install is reachable.  Tests that drive a full link with
    ``flang-new-21`` use this to decide whether to skip cleanly or
    inject the path via ``LIBRARY_PATH``."""
    for d in _FLANG_RT_DIRS:
        if (Path(d) / "libflang_rt.runtime.a").is_file():
            return d
    return None


def env_with_flang_runtime(fc_name: str) -> dict:
    """A copy of ``os.environ`` with ``LIBRARY_PATH`` prepended for
    flang's freshly-built runtime when ``fc_name`` is a flang variant
    and a runtime is reachable.  Pass to ``subprocess.check_call(...,
    env=env_with_flang_runtime(name))`` so a full compile + link
    succeeds without the user having to export anything globally.
    No-op for gfortran / nvfortran (they ship their own runtimes)."""
    env = dict(os.environ)
    if "flang" in fc_name:
        rt = find_flang_runtime_dir()
        if rt is not None:
            existing = env.get("LIBRARY_PATH", "")
            env["LIBRARY_PATH"] = f"{rt}:{existing}" if existing else rt
    return env


#: Skip reason used by tests that need a full link under flang but
#: can't find the runtime.  Surfaced as the ``reason`` of a
#: ``pytest.skip`` so the test report carries the exact remediation
#: hint.
FLANG_RT_HINT = (
    "flang-new-21 needs ``libflang_rt.runtime.a`` for a full link; "
    "build it locally with the recipe at the top of the README's "
    "Fortran-compiler matrix section, or symlink ROCm's "
    "/opt/rocm-7.2.0/...x86_64-unknown-linux-gnu/libflang_rt.runtime.a "
    "into a $LIBRARY_PATH directory."
)
