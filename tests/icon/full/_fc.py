"""Shared Fortran-compiler discovery + pytest parametrization helper.

Locates gfortran / flang-new / nvfortran (PATH + standard NVHPC prefix) and exposes ``FORTRAN_COMPILERS`` for ``@pytest.mark.parametrize``; the iso_c wrappers + solve_nh patch must parse cleanly under all three ICON-shipped compilers.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest

# standard NVIDIA HPC SDK install prefixes; version segment (e.g. 25.7) varies by release.
_NVHPC_PREFIXES = (
    "/opt/nvidia/hpc_sdk/Linux_x86_64",
    "/opt/nvhpc/Linux_x86_64",
    "/usr/local/nvhpc/Linux_x86_64",
)


def _find_nvfortran() -> Optional[Path]:
    """Locate nvfortran on PATH or under the standard NVHPC prefixes; picks the newest version dir when several are installed."""
    p = shutil.which("nvfortran")
    if p is not None:
        return Path(p)
    for prefix in _NVHPC_PREFIXES:
        prefix_path = Path(prefix)
        if not prefix_path.is_dir():
            continue
        versions = sorted((d for d in prefix_path.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
        for version_dir in versions:
            candidate = version_dir / "compilers" / "bin" / "nvfortran"
            if candidate.is_file():
                return candidate
    return None


# LLVM-flang binary names probed in order; Ubuntu ships flang-new-21/flang-21 as identical symlinks, distributions differ on which is canonical.
_FLANG_NAMES = ("flang-new-21", "flang-21", "flang-new", "flang")


def _looks_like_llvm_flang(path: str) -> bool:
    """True when ``path --version`` self-identifies as LLVM flang; gates $FC override so a non-flang $FC doesn't hijack the flang slot."""
    try:
        out = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT, timeout=5).decode(errors="replace")
    except (OSError, subprocess.SubprocessError):
        return False
    return "flang version" in out


def _find_flang() -> Optional[Path]:
    """Locate an LLVM-flang binary: ``$FC`` if set and self-identifying as flang, else the first ``_FLANG_NAMES`` hit on PATH."""
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
    """{display_name: path} for gfortran and flang-new-21 (both on CI); each test compiles against an ICON .mod tree built by the SAME compiler since .mod files aren't cross-compatible.  nvfortran excluded -- multi-GB CI dependency, GPU builds out of scope."""
    out: dict = {}
    gfortran = _find_gfortran()
    if gfortran:
        out["gfortran"] = gfortran
    flang = _find_flang()
    if flang:
        out["flang-new-21"] = flang
    return out


def _make_params() -> List:
    """Build the ``pytest.param(...)`` list for the discovered set."""
    found = discover_fortran_compilers()
    if not found:
        return [
            pytest.param(("gfortran", "gfortran"),
                         id="gfortran",
                         marks=[pytest.mark.skip(reason="no Fortran compiler found on the host")])
        ]
    return [pytest.param((name, str(path)), id=name) for name, path in found.items()]


#: (name, executable_path) tuples for @pytest.mark.parametrize("fc", FORTRAN_COMPILERS); no-compiler-available is handled via per-param skip marks.
FORTRAN_COMPILERS = _make_params()


def fortran_compiler_flags(fc_name: str) -> List[str]:
    """Per-compiler base flags for wrapper-syntax checks: gfortran needs -ffree-line-length-none (default 132-col limit) for long generated lines; flang-new/nvfortran need nothing (no limit, and flang rejects the gfortran flag)."""
    if fc_name == "gfortran":
        return ["-ffree-line-length-none"]
    return []


def syntax_check_argv(fc_name: str, scratch_dir: Path) -> List[str]:
    """Per-compiler argv tail for a parse+semantic-check pass: gfortran/flang-new use -fsyntax-only; nvfortran has no equivalent, falls back to bare -c (no -o -- it refuses -c -o single.o with multiple sources; caller's cwd=scratch_dir keeps per-source .o files xdist-safe)."""
    del scratch_dir  # reserved for future single-output compilers
    if fc_name == "nvfortran":
        return ["-c"]
    return ["-fsyntax-only"]


def cpp_flag(fc_name: str) -> str:
    """Per-compiler -cpp-equivalent flag: gfortran/flang-new use -cpp, nvfortran uses -Mpreprocess."""
    return "-Mpreprocess" if fc_name == "nvfortran" else "-cpp"


# standard locations for a user-local libflang_rt.runtime.a; surfaced via LIBRARY_PATH so flang-new-21's linker finds it automatically.
_FLANG_RT_DIRS = (
    str(Path.home() / ".local/llvm-flang-rt-21/lib/clang/21/lib/x86_64-unknown-linux-gnu"),
    str(Path.home() / ".local/lib/clang/21/lib/x86_64-unknown-linux-gnu"),
    # ROCm runtime is ABI-compatible with flang-21; probed last so a user-local build wins when both are present.
    "/opt/rocm-7.2.0/lib/llvm/lib/clang/22/lib/x86_64-unknown-linux-gnu",
)


def find_flang_runtime_dir() -> Optional[str]:
    """First directory with libflang_rt.runtime.a (flang-21's -lflang_rt.runtime archive), or None.  Tests linking with flang-new-21 use this to skip cleanly or inject LIBRARY_PATH."""
    for d in _FLANG_RT_DIRS:
        if (Path(d) / "libflang_rt.runtime.a").is_file():
            return d
    # apt llvm-{21,22} layout: target-triple subdir spelling varies, so glob for it (CI's llvm-21 install keeps the archive here).
    for base in Path("/usr/lib").glob("llvm-2*/lib/clang/*/lib/*"):
        if (base / "libflang_rt.runtime.a").is_file():
            return str(base)
    return None


def env_with_flang_runtime(fc_name: str) -> dict:
    """Copy of os.environ with LIBRARY_PATH prepended for flang's runtime, when reachable and ``fc_name`` is a flang variant.  No-op for gfortran/nvfortran (they ship their own runtimes)."""
    env = dict(os.environ)
    if "flang" in fc_name:
        rt = find_flang_runtime_dir()
        if rt is not None:
            existing = env.get("LIBRARY_PATH", "")
            env["LIBRARY_PATH"] = f"{rt}:{existing}" if existing else rt
    return env


#: pytest.skip reason for tests needing a full flang link but no runtime found.
FLANG_RT_HINT = ("flang-new-21 needs ``libflang_rt.runtime.a`` for a full link; "
                 "build it locally with the recipe at the top of the README's "
                 "Fortran-compiler matrix section, or symlink ROCm's "
                 "/opt/rocm-7.2.0/...x86_64-unknown-linux-gnu/libflang_rt.runtime.a "
                 "into a $LIBRARY_PATH directory.")
