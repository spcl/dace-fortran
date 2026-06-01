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
