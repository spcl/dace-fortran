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
import subprocess
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


# LLVM-flang binary names probed in order.  Ubuntu's ``flang-21``
# package ships ``flang-new-21`` and ``flang-21`` as identical
# symlinks; upstream LLVM 21 dropped the ``-new`` suffix once the
# rewritten frontend stabilised, so distributions differ on which
# name is canonical.  The bridge accepts any of these  --  what
# matters is the underlying compiler, not the path it was reached
# through.
_FLANG_NAMES = ("flang-new-21", "flang-21", "flang-new", "flang")


def _looks_like_llvm_flang(path: str) -> bool:
    """``True`` when ``path --version`` self-identifies as LLVM flang.

    Used to gate ``$FC`` override: if the user pins ``FC`` to a custom
    LLVM-flang build (Spack module, source build, ...), we honour it;
    if ``FC`` points at gfortran or nvfortran, we ignore it for the
    flang slot and the dedicated gfortran/nvfortran probes pick those
    up by their own names.
    """
    try:
        out = subprocess.check_output(
            [path, "--version"], stderr=subprocess.STDOUT,
            timeout=5).decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return False
    return "flang version" in out


def _find_flang() -> Optional[Path]:
    """Locate an LLVM-flang binary.

    Resolution order:

      1. ``$FC`` if set and the binary self-identifies as LLVM flang.
         This lets the user point at an off-PATH build (e.g.
         ``FC=/opt/llvm-21/bin/flang``) without renaming anything.
      2. The first entry in ``_FLANG_NAMES`` found on ``PATH``.
    """
    fc = os.environ.get("FC")
    if fc:
        fc_path = shutil.which(fc) or (fc if os.path.isfile(fc) else None)
        if fc_path and _looks_like_llvm_flang(fc_path):
            return Path(fc_path)
    for name in _FLANG_NAMES:
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


def syntax_check_argv(fc_name: str, scratch_dir: Path) -> List[str]:
    """Per-compiler argv tail for a parse + semantic-check pass.

    gfortran / flang-new spell it ``-fsyntax-only``: a single token
    that suppresses codegen + link.  nvfortran has no equivalent
    knob -- ``-Msyntax`` is rejected as an unknown switch, and the
    closest fit is bare ``-c``: compile to object files (full
    parse + semantic + body checks, no link).  We deliberately do
    NOT pass ``-o`` here: nvfortran refuses ``-c -o single.o``
    when more than one source is supplied ("More than one output
    file will overwrite ..."), and the caller already runs with
    ``cwd=scratch_dir`` (a per-test tmp_path under ``-n N``) so the
    per-source ``.o`` files land in an xdist-private directory and
    can't race.

    :param fc_name: stable display name from ``FORTRAN_COMPILERS``.
    :param scratch_dir: per-test scratch directory.  Reserved for a
                        future compiler whose syntax-check argv
                        legitimately needs a sandboxed output file.
    :returns: argv fragment to splat into ``subprocess.check_call``
              before the source-file arguments.
    """
    del scratch_dir  # reserved for future single-output compilers
    if fc_name == "nvfortran":
        return ["-c"]
    return ["-fsyntax-only"]


def cpp_flag(fc_name: str) -> str:
    """Per-compiler ``-cpp``-equivalent preprocessing flag.

    gfortran / flang-new accept ``-cpp``; nvfortran spells the same
    thing ``-Mpreprocess``.  Centralised here so a third compiler
    later only adds one entry, not a string-compare at every call site.
    """
    return "-Mpreprocess" if fc_name == "nvfortran" else "-cpp"


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
