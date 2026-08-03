#!/usr/bin/env python
# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Static-analyze the DaCe-generated kernel C++ -- not the frontend/bridge.

Local-testing policy: the bugs that matter for numerical correctness live in what
we *emit* (a 1-past-end write, a bad index), not in the C++ of the bridge. So we
point ``clang-tidy`` and ``cppcheck`` at the generated ``.dacecache/<name>/src/cpu``
translation units, restricted to ``clang-analyzer-*`` (the path-sensitive overrun /
use-after-free detector) -- machine-emitted code trips every style/naming rule, and
even ``bugprone-*`` is ~all false positives on it, so both are noise here.

Both analyzers read the CMake ``compile_commands.json`` the kernel was built with,
so includes resolve without hardcoding any path. DaCe does not export it by default;
reconfigure the existing build dir once (no rebuild)::

    cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON .dacecache/<name>/build

Without it, clang-tidy is skipped and cppcheck falls back to best-effort.

The sharpest gate for heap bugs is still an ASan *run* of the kernel: pass
``--asan`` to print the copy-paste recipe (rebuild the kernel with
``-fsanitize=address`` and ``LD_PRELOAD`` the matching runtime into Python).

Usage::

    scripts/lint_generated_kernel.py --name vps            # find newest vps cache
    scripts/lint_generated_kernel.py --cache .dacecache/vps
    scripts/lint_generated_kernel.py path/to/kernel.cpp    # explicit TU(s)
    scripts/lint_generated_kernel.py --asan                # print ASan recipe

Exit status: 0 when every analyzer that ran was clean, 1 otherwise.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# gcc's default kernel args, per the bridge's local-testing note; swap -O3/fast-math
# for a debuggable, sanitizer-friendly build.
ASAN_CPU_ARGS = "-fPIC -O1 -g -fsanitize=address -fno-omit-frame-pointer -march=native"


def asan_recipe() -> str:
    """The env to run a Python test under ASan against a freshly ASan-built kernel."""
    return ("# Rebuild the kernel with ASan and preload the matching runtime (gcc kernels -> gcc libasan):\n"
            f'  export DACE_compiler_cpu_args="{ASAN_CPU_ARGS}"\n'
            "  export LD_PRELOAD=$(gcc -print-file-name=libasan.so)\n"
            "  export ASAN_OPTIONS=detect_leaks=0    # Python leaks are noise; tighten later\n"
            "  # then run the test; wipe the cache first so the kernel is recompiled with the flags:\n"
            "  rm -rf .dacecache/<name> && python -m pytest tests/<file>.py -k <case>\n")


def default_cache_root() -> Path:
    """Where DaCe drops build folders: ``$DACE_default_build_folder`` or ``./.dacecache``."""
    return Path(os.environ.get("DACE_default_build_folder", ".dacecache"))


def find_kernel_tus(cache: Path) -> list[Path]:
    """The generated CPU translation units under a ``.dacecache/<name>`` dir."""
    return sorted((cache / "src" / "cpu").glob("*.cpp"))


def resolve_targets(args) -> tuple[list[Path], Path | None]:
    """(translation units, build dir with compile_commands.json) from the CLI args."""
    if args.paths:
        tus = [Path(p) for p in args.paths]
        cache = tus[0].resolve().parents[2] if tus else None  # <cache>/src/cpu/x.cpp
    elif args.cache:
        cache = Path(args.cache)
        tus = find_kernel_tus(cache)
    elif args.name:
        root = default_cache_root()
        caches = [c for c in root.glob(f"{args.name}*") if (c / "src" / "cpu").is_dir()]
        if not caches:
            sys.exit(f"no cache '{args.name}*' under {root} -- build the kernel first")
        cache = max(caches, key=lambda c: c.stat().st_mtime)  # newest wins
        tus = find_kernel_tus(cache)
    else:
        sys.exit("give --name, --cache, or explicit .cpp paths (or --asan)")
    build = cache / "build" if cache and (cache / "build" / "compile_commands.json").is_file() else None
    return tus, build


def run_clang_tidy(tus: list[Path], build: Path | None) -> int:
    """clang-analyzer static-analysis only; needs compile_commands.json for includes."""
    exe = shutil.which("clang-tidy-21") or shutil.which("clang-tidy")
    if exe is None:
        print("clang-tidy not found; skipping", file=sys.stderr)
        return 0
    if build is None:
        print("no compile_commands.json (build with CMake first); skipping clang-tidy", file=sys.stderr)
        return 0
    # clang-analyzer-* only: the path-sensitive overrun / use-after-free / null-deref detector
    # (the compile-time analogue of the ASan run). bugprone-* is ~all false positives on
    # machine-generated code (__-internals, swappable params, widening) -- 200+ lines of noise.
    # Findings are advisory (clang-tidy exits 0 on warnings); the ASan run is the hard heap gate.
    cmd = [
        exe, "-p",
        str(build), "--quiet", "--checks=-*,clang-analyzer-*", "--header-filter=$^", *[str(t) for t in tus]
    ]
    return subprocess.call(cmd)


def run_cppcheck(tus: list[Path], build: Path | None) -> int:
    """Correctness-leaning cppcheck; prefer the compile DB so macros/includes resolve."""
    exe = shutil.which("cppcheck")
    if exe is None:
        print("cppcheck not found; skipping", file=sys.stderr)
        return 0
    # Suppress noise that is never our bug: vendored-header platform #errors (moodycamel),
    # findings inside external/ headers, and system-include gaps.
    cmd = [
        exe, "--enable=warning,performance,portability", "--inline-suppr", "--error-exitcode=1", "--quiet",
        "--suppress=preprocessorErrorDirective", "--suppress=*:*/external/*", "--suppress=missingIncludeSystem"
    ]
    if build is not None:
        cmd.append(f"--project={build / 'compile_commands.json'}")
    else:
        cmd.extend(["--language=c++", "--std=c++14", *[str(t) for t in tus]])
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="explicit .cpp translation units to lint")
    ap.add_argument("--name", help="kernel name; lints the newest .dacecache/<name>* build")
    ap.add_argument("--cache", help="a .dacecache/<name> directory")
    ap.add_argument("--asan", action="store_true", help="print the ASan run recipe and exit")
    args = ap.parse_args()

    if args.asan:
        print(asan_recipe())
        return 0

    tus, build = resolve_targets(args)
    if not tus:
        sys.exit("no generated .cpp found -- did the kernel build?")
    print(f"linting {len(tus)} TU(s): {', '.join(t.name for t in tus)}"
          f"{'' if build else '  (no compile DB -- best effort)'}")
    rc = run_clang_tidy(tus, build) | run_cppcheck(tus, build)
    print("clean" if rc == 0 else "issues found", file=sys.stderr)
    return 1 if rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
