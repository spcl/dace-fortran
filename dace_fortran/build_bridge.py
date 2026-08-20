#!/usr/bin/env python3
# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Auto-build and import hlfir_bridge; import this module to get `hb`.

The .so is symlinked next to this file (as dace_fortran.hlfir_bridge) so no PYTHONPATH manipulation is needed.
"""

import fcntl
import importlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from dace_fortran.llvm_toolchain import (SUPPORTED_LLVM_VERSIONS, candidate_versions, find_flang, llvm_prefix,
                                         requested_version)

# --- Configuration: override via env vars ---

_HERE = Path(__file__).resolve().parent
_BUILD_DIR = Path(os.environ.get("DACE_FORTRAN_BUILD_DIR", _HERE / "build"))

# Pinned by the LLVM_VERSION env var; empty means "auto-detect over SUPPORTED_LLVM_VERSIONS".
_LLVM_VERSION = requested_version()

# Override with env var; only needed if cmake's LLVM auto-detect misses.
# No MLIR_DIR needed: CMakeLists.txt finds MLIR via the LLVM prefix, bypassing MLIR's broken cmake config.
_LLVM_DIR = os.environ.get("LLVM_DIR", "")

_CMAKE_MAJOR_RE = re.compile(r"set\(LLVM_VERSION_MAJOR (\d+)\)")


def _find_llvm_prefix(version: str) -> str:
    """LLVM install prefix from that version's flang (resolves its symlink); falls back to /usr/lib/llvm-<version>."""
    flang = find_flang(version)
    if flang:
        prefix = llvm_prefix(flang)
        if (prefix / "lib" / "cmake").is_dir():
            return str(prefix)

    fallback = Path(f"/usr/lib/llvm-{version}")
    if fallback.is_dir():
        return str(fallback)

    return ""


def _cmake_dir_candidates(prefix: str, version: str) -> list:
    """Directories that may hold LLVMConfig.cmake for <version>, most specific first."""
    return ([f"{prefix}/lib/cmake/llvm"] if prefix else []) + [
        f"/usr/lib/llvm-{version}/lib/cmake/llvm",
        f"/usr/lib/llvm-{version}/cmake",
        "/usr/local/lib/cmake/llvm",
        "/opt/homebrew/lib/cmake/llvm",
    ]


def _cmake_dir_major(cmake_dir: str) -> str:
    """LLVM major declared by an LLVMConfig.cmake, or '' if it cannot be read."""
    try:
        text = Path(cmake_dir, "LLVMConfig.cmake").read_text()
    except OSError:
        return ""
    match = _CMAKE_MAJOR_RE.search(text)
    return match.group(1) if match else ""


def _find_llvm_cmake_dir(prefix: str, version: str) -> str:
    """Find an LLVMConfig.cmake for <version> under a known prefix.

    Version-unspecific candidates (/usr/local, homebrew) are only accepted once the
    config file itself declares the requested major.
    """
    for c in _cmake_dir_candidates(prefix, version):
        p = Path(c)
        if p.is_dir() and (p / "LLVMConfig.cmake").exists() and _cmake_dir_major(c) in ("", version):
            return str(p)

    for pkg in [f"llvm-{version}-dev", f"libllvm-{version}-dev"]:
        try:
            out = subprocess.check_output(
                ["dpkg", "-L", pkg],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in out.splitlines():
                if line.endswith("LLVMConfig.cmake"):
                    return str(Path(line).parent)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    return ""


def _prefix_builds_the_bridge(prefix: str, version: str) -> bool:
    """Whether a prefix carries what the bridge needs, beyond an ``LLVMConfig.cmake``.

    An LLVM install can ship the LLVM cmake package and neither flang nor MLIR --
    ``llvm-21-dev`` without ``libmlir-21-dev``/``flang-21`` is exactly that.  Selecting it
    configures a build that cannot emit HLFIR and links against whatever MLIR happens to be
    reachable, which is how a mixed prefix reaches the linker and fails at ``dlopen``.
    """
    if not find_flang(version):
        return False
    root = Path(prefix)
    return (root / "lib" / "cmake" / "mlir").is_dir() or (root / "include" / "mlir").is_dir()


def _detect_dirs():
    """Populate _LLVM_DIR / _LLVM_VERSION: explicit env vars win, else probe the supported majors."""
    global _LLVM_DIR, _LLVM_VERSION
    if _LLVM_DIR:
        _LLVM_VERSION = _LLVM_VERSION or _cmake_dir_major(_LLVM_DIR) or SUPPORTED_LLVM_VERSIONS[0]
        return

    searched: list = []
    incomplete: list = []
    for version in candidate_versions():
        prefix = _find_llvm_prefix(version)
        searched += _cmake_dir_candidates(prefix, version)
        found = _find_llvm_cmake_dir(prefix, version)
        if found and not _prefix_builds_the_bridge(prefix, version):
            incomplete.append(f"{prefix} (LLVM {version}: no flang and/or no MLIR)")
            continue
        if found:
            _LLVM_VERSION, _LLVM_DIR = version, found
            return

    if incomplete:
        raise RuntimeError("No LLVM prefix can build the HLFIR bridge.  Rejected as incomplete: " +
                           "; ".join(incomplete) + ".  Install the matching flang and MLIR development "
                           "packages, or set LLVM_DIR (and optionally LLVM_VERSION) to a complete prefix.")

    majors = "/".join(candidate_versions())
    raise RuntimeError(f"Cannot find LLVMConfig.cmake for LLVM {majors}.  Looked in: "
                       f"{', '.join(dict.fromkeys(searched))}.  Set LLVM_DIR (and "
                       "optionally LLVM_VERSION) to override.")


# --- Build logic ---


def _ext_suffix() -> str:
    """Python extension suffix, e.g. '.cpython-312-x86_64-linux-gnu.so'."""
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _so_name() -> str:
    return f"hlfir_bridge{_ext_suffix()}"


def _built_so() -> Path:
    """Path to the .so inside the build directory."""
    return _BUILD_DIR / _so_name()


def _local_so() -> Path:
    """Symlink target next to this file."""
    return _HERE / _so_name()


def needs_build() -> bool:
    """True if the extension is missing or older than any source file."""
    so = _local_so()
    if not so.exists():
        return True
    so_mtime = so.stat().st_mtime
    for pat in ("**/*.cpp", "**/*.h", "CMakeLists.txt"):
        for src in _HERE.glob(pat):
            if "build" in src.parts:
                continue
            if src.stat().st_mtime > so_mtime:
                return True
    return False


def _python_cmake_hints() -> list:
    """Explicit ``Python_*`` hints for cmake's ``find_package(Python)``.

    A pyenv/venv prefix often ships no headers or libpython of its own, so cmake's venv-relative
    FindPython misses Development.Module even though the interpreter works fine; derive real paths
    from sysconfig instead.
    """
    hints = []
    include = sysconfig.get_path("include")
    if include and os.path.isdir(include):
        hints.append(f"-DPython_INCLUDE_DIR={include}")
    libdir = sysconfig.get_config_var("LIBDIR")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if libdir and ldlibrary:
        lib = os.path.join(libdir, ldlibrary)
        if os.path.exists(lib):
            hints.append(f"-DPython_LIBRARY={lib}")
    return hints


NO_REBUILD_ENV = "DACE_FORTRAN_NO_REBUILD"


def rebuild_forbidden() -> bool:
    """True if the environment bans building the bridge in this process."""
    return os.environ.get(NO_REBUILD_ENV, "") not in ("", "0")


def _cache_conflicts(expected: dict) -> list:
    """Cached cmake entries that disagree with what this build is about to pass.

    Reconfiguring an existing build directory with a different prefix or interpreter leaves the
    entries cmake does not overwrite pointing at the old one, and the module then links parts of
    two LLVM installs -- seen as ``LLVM_VERSION=21`` beside an MLIR from the 22 prefix, which
    builds and then fails at ``dlopen`` on an undefined typeinfo.
    """
    cache = _BUILD_DIR / "CMakeCache.txt"
    if not cache.is_file():
        return []
    cached = {}
    for line in cache.read_text(errors="replace").splitlines():
        name, sep, value = line.partition(":")
        if sep and "=" in value:
            cached[name.strip()] = value.split("=", 1)[1].strip()
    return [f"{k}: {cached[k]!r} -> {v!r}" for k, v in expected.items() if k in cached and cached[k] != v]


def build(clean: bool = False, verbose: bool = True):
    """Run cmake + make.  Raises on failure."""
    if rebuild_forbidden():
        raise RuntimeError(f"{NO_REBUILD_ENV} is set -- refusing to build the HLFIR bridge.\n"
                           f"  expected a prebuilt extension at: {_local_so()}\n"
                           "Something requested a build (missing or stale .so).  Doing it here would\n"
                           "compile into the shared source tree, typically from a compute node.\n"
                           "Build once on the login node with:\n"
                           "  python -c 'import dace_fortran.build_bridge as b; b.build()'\n"
                           f"then resubmit -- or unset {NO_REBUILD_ENV} to allow this process to build.")
    _detect_dirs()

    python = sys.executable
    # Pin the flang this resolution picked, so cmake does not run its own probe and land on a
    # different major than the LLVM_DIR passed beside it.
    flang = find_flang(_LLVM_VERSION) or ""
    conflicts = _cache_conflicts({
        "LLVM_DIR": _LLVM_DIR,
        "LLVM_VERSION": _LLVM_VERSION,
        "FLANG_BIN": flang,
        "Python_EXECUTABLE": python,
    })
    if conflicts and not clean:
        if verbose:
            print(
                f"[build_bridge] build dir was configured differently ({'; '.join(conflicts)}); "
                "reconfiguring from scratch",
                file=sys.stderr)
        clean = True

    if clean and _BUILD_DIR.exists():
        if verbose:
            print(f"[build_bridge] cleaning {_BUILD_DIR}", file=sys.stderr)
        shutil.rmtree(_BUILD_DIR)

    _BUILD_DIR.mkdir(exist_ok=True)

    nproc = os.cpu_count() or 4

    # --- cmake configure ---
    cmake_args = [
        "cmake",
        str(_HERE),
        f"-DLLVM_VERSION={_LLVM_VERSION}",
        f"-DLLVM_DIR={_LLVM_DIR}",
        *([f"-DFLANG_BIN={flang}"] if flang else []),
        f"-DPython_EXECUTABLE={python}",
        *_python_cmake_hints(),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if verbose:
        print(f"[build_bridge] configure: {' '.join(cmake_args)}", file=sys.stderr)
    subprocess.check_call(cmake_args, cwd=_BUILD_DIR)

    # --- cmake build ---
    build_args = ["cmake", "--build", ".", f"-j{nproc}"]
    if verbose:
        print(f"[build_bridge] build: {' '.join(build_args)}", file=sys.stderr)
    subprocess.check_call(build_args, cwd=_BUILD_DIR)

    # --- symlink .so next to this file ---
    target = _built_so()
    link = _local_so()
    if not target.exists():
        candidates = list(_BUILD_DIR.rglob(_so_name()))
        if candidates:
            target = candidates[0]
        else:
            raise RuntimeError(f"Build succeeded but cannot find {_so_name()} "
                               f"under {_BUILD_DIR}")
    link.unlink(missing_ok=True)
    link.symlink_to(target)
    if verbose:
        print(f"[build_bridge] linked {link} -> {target}", file=sys.stderr)


# --- Import-or-build ---

_BRIDGE_MODULE = "dace_fortran.hlfir_bridge"


def ensure_bridge():
    """Import the compiled bridge, building first if necessary.

    Imported as dace_fortran.hlfir_bridge (build() symlinks the .so into the package dir) -- no sys.path hacking.
    """
    try:
        return importlib.import_module(_BRIDGE_MODULE)
    except ImportError:
        pass

    print("[build_bridge] hlfir_bridge not found, building...", file=sys.stderr)
    build()
    importlib.invalidate_caches()
    return importlib.import_module(_BRIDGE_MODULE)


def ensure_fresh():
    """Import hlfir_bridge, rebuilding if any source is newer than the .so.

    flock-serialized: concurrent processes (e.g. parallel pytest runs) racing an unlocked
    ``cmake --build`` into the same build dir corrupt the link -- the resulting .so then
    segfaults on its first real call, with no Python traceback.
    """
    with open(_HERE / ".build_bridge.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if needs_build():
            print("[build_bridge] sources newer than .so, rebuilding...", file=sys.stderr)
            build()
    return ensure_bridge()


def __getattr__(name: str):
    """Resolve ``hb`` on first use, which is what the README promises and what the CLI needs.

    Binding it at import made ``python -m dace_fortran.build_bridge`` load the extension before
    rebuilding it, so the process ran cmake twice and then exited with the old .so still mapped.
    """
    if name == "hb":
        bridge = ensure_fresh()
        globals()["hb"] = bridge
        return bridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- CLI ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the hlfir_bridge nanobind extension.")
    parser.add_argument("--clean", action="store_true", help="Wipe build dir before building.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    build(clean=args.clean, verbose=not args.quiet)
    print(f"[build_bridge] OK: {_local_so()}")
