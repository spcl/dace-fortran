# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Static analysis of the C++ DaCe generates for a built SDFG.

The generated code is already compiled with ``-Wall -Wextra`` (see the build's ``flags.make``), but nothing reads
the output, so undefined behaviour ships silently: an ICON halo body shipped a buffer sized from an uninitialised
local for months, and the only symptom was a glibc abort inside an unrelated ``free()``.

:data:`CRITICAL_WARNINGS` is the subset that means "this code has UB", not "this code is untidy" -- those are the
ones a build must never emit.  Deep analysis follows the compiler: gcc builds get ``-fanalyzer``, clang builds get
the LLVM static analyzer, so the analysis always matches the toolchain that produced the binary.

A missing tool raises: silently degrading to "nothing found" is how this class of bug survived in the first place.
"""

import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from dace.config import Config
from dace.sdfg import SDFG

# Warnings that indicate undefined behaviour rather than style.  A generated TU emitting any of these is a codegen
# bug: the reader of the generated code is a compiler, so there is no "intentional" uninitialised read to excuse.
CRITICAL_WARNINGS = (
    "uninitialized",
    "array-bounds",
    "stringop-overflow",
    "free-nonheap-object",
    "nonnull",
    "return-type",
    "sizeof-pointer-memaccess",
)

# Enabled and reported, but NOT a build failure.  ``-Wmaybe-uninitialized`` is gcc's speculative
# variant: it fires when the dataflow cannot PROVE a definition reaches a use, which a guarded
# define/use pair defeats whenever an opaque call sits between the two and the guard operands are
# reachable through a non-const pointer.  The input Fortran carries the same shapes, so gating on it
# scores gcc's heuristic rather than the generated code.  ``-Wuninitialized`` (the provable variant)
# stays critical.  Kept enabled so the diagnostic still prints as informational -- dropping it from
# CRITICAL_WARNINGS alone would also drop the ``-W`` flag and silence it entirely.
NONCRITICAL_WARNINGS = ("maybe-uninitialized", )

# clang names a few of these differently or not at all; passing an unknown -W to clang is only a warning, but
# keeping the list explicit documents what actually gets checked there.
CLANG_CRITICAL_WARNINGS = (
    "uninitialized",
    "array-bounds",
    "return-type",
    "sizeof-pointer-memaccess",
)

# The generator's own conventions are not defects: every DaCe symbol is double-underscored (__state, __i0,
# __dace_init_*), which is a reserved identifier by the letter of the standard and unavoidable here.
#
# security.ArrayBound is off for a sharper reason: heap extents here are symbolic (``new double[n*(m-1)+n]``
# indexed by ``i + j*n`` for i<n, j<m), and the analyzer cannot relate the two expressions, so it reports an
# overrun on provably-correct code.  It fires on the simplest valid kernel, which makes it useless as a gate --
# the bounds checking that does work on generated code is -Warray-bounds plus ASAN on a real run.
CLANG_TIDY_CHECKS = ("clang-analyzer-*,bugprone-*"
                     ",-bugprone-reserved-identifier"
                     ",-bugprone-easily-swappable-parameters"
                     ",-clang-analyzer-security.ArrayBound")
CPPCHECK_SUPPRESSIONS = ("preprocessorErrorDirective", "missingIncludeSystem", "checkersReport")

TOOLS = ("warnings", "analyzer", "clang-tidy", "cppcheck")


def generated_source(sdfg: SDFG) -> Path:
    """Path to the CPU C++ DaCe emitted for ``sdfg``.  Requires a completed build."""
    src = Path(sdfg.build_folder) / "src" / "cpu" / f"{sdfg.name}.cpp"
    if not src.is_file():
        raise FileNotFoundError(f"no generated C++ at {src}; compile the SDFG before analysing it")
    return src


def build_flags(sdfg: SDFG) -> List[str]:
    """The exact defines/includes/flags CMake used for the generated TU, read back from the build.

    Reusing the build's own flags keeps the analysis honest -- a hand-rebuilt include path analyses code the
    compiler never saw.  The Makefiles generator records them in ``flags.make``; the Ninja generator (used when
    ``ninja`` is on PATH) does not emit that file, so fall back to ``compile_commands.json``, which every
    generator writes and which carries the full command line for the TU.
    """
    build = Path(sdfg.build_folder) / "build"
    makeflags = build / "CMakeFiles" / f"{sdfg.name}.dir" / "flags.make"
    if makeflags.is_file():
        flags: List[str] = []
        for line in makeflags.read_text().splitlines():
            for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
                if line.startswith(key):
                    flags.extend(shlex.split(line.split("=", 1)[1]))
        return flags

    compdb = build / "compile_commands.json"
    if not compdb.is_file():
        raise FileNotFoundError(
            f"no {makeflags} and no {compdb}; the SDFG must be built (not just codegen'd) before analysis")
    src = generated_source(sdfg).resolve()
    entries = json.loads(compdb.read_text())
    # CMake records absolute paths, but ``build_folder`` is routinely relative (the default
    # ``.dacecache``, xdist's per-worker ``.dacecache_gwN``), so compare resolved paths -- a raw
    # string compare matches nothing and reads as "the TU was never built".
    match = next((e for e in entries if Path(e["file"]).resolve() == src), None)
    if match is None:
        # The database describes only the LAST cmake configure of this build folder, so a second
        # SDFG built into it (DaCe renames the collision to ``<name>_0``) evicts the first one's
        # entry.  Every generated TU of one project is compiled with the same -D/-I/-W set, so a
        # sibling in the same ``src/cpu`` carries the flags this TU saw; the per-target -o/-M*
        # tokens are the only difference and are dropped below anyway.
        match = next((e for e in entries if Path(e["file"]).resolve().parent == src.parent), None)
    if match is None:
        raise FileNotFoundError(f"{compdb} has no entry for {src}, and no sibling TU in {src.parent}")

    argv = shlex.split(match["command"]) if "command" in match else list(match["arguments"])
    # Drop the compiler, the -c/-o output pair, the source file, and the dependency-file flags;
    # keep every -D/-I/-W/-f flag the TU saw.  -MT/-MF/-MQ carry paths RELATIVE to the build
    # directory, which this analysis does not run in, so keeping them fails the compile on a file
    # it cannot open -- and a failed analysis compile reports no warnings, i.e. reads as clean.
    flags = []
    skip = False
    for tok in argv[1:]:  # argv[0] is the compiler
        if skip:
            skip = False
            continue
        if tok in ("-o", "-MT", "-MF", "-MQ"):
            skip = True  # also drop its argument
            continue
        if tok in ("-c", "-MD", "-MMD", "-MP") or tok == match["file"] or tok.endswith(".cpp"):
            continue
        flags.append(tok)
    return flags


def compiler_is_clang() -> bool:
    """Is DaCe configured to build with clang rather than gcc?"""
    return "clang" in Path(Config.get("compiler", "cpu", "executable") or "g++").name


def require(tool: str) -> str:
    """Absolute path to ``tool``, or a hard failure naming what to install."""
    found = shutil.which(tool)
    if not found:
        raise FileNotFoundError(f"{tool} not on PATH; install it (CI runners carry the full analysis toolchain)")
    return found


#: Non-critical warning tags that fire in the hundreds on generated code and carry no action -- excluded from
#: the informational print so a real non-critical warning is not buried.  (The generated code deliberately keeps
#: dead temps and never-returning stub decls; that is not a defect to chase.)
NONCRITICAL_NOISE = ("unused-but-set-variable", "unused-variable", "unused-parameter", "undefined-function-result")


def critical_tags(clang: Optional[bool] = None) -> set:
    """The ``[-Wname]`` tags that mark UB-class (critical) warnings for the active compiler."""
    if clang is None:
        clang = compiler_is_clang()
    return {f"[-W{w}]" for w in (CLANG_CRITICAL_WARNINGS if clang else CRITICAL_WARNINGS)}


def analyze(sdfg: SDFG, tool: str = "warnings", critical_only: bool = True) -> List[str]:
    """Run ``tool`` over the generated C++ and return the diagnostic lines it reported.

    ``warnings`` and ``analyzer`` follow the configured compiler (gcc ``-fanalyzer`` / clang ``--analyze``);
    the other two are their own binaries.  Only correctness checks are enabled, so a non-empty result is always
    actionable.

    :param critical_only: for ``warnings``/``analyzer``, keep only the UB-class (:data:`CRITICAL_WARNINGS`) tags;
        pass ``False`` to get every ``-W`` diagnostic the compiler reported (the caller then filters/prints them).
    """
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; expected one of {TOOLS}")
    src = generated_source(sdfg)
    flags = build_flags(sdfg)
    clang = compiler_is_clang()

    if tool in ("warnings", "analyzer"):
        enabled = CLANG_CRITICAL_WARNINGS if clang else CRITICAL_WARNINGS + NONCRITICAL_WARNINGS
        warn = [f"-W{w}" for w in enabled]
        # -O2 last so it overrides the build's -O0: the dataflow that powers -Wmaybe-uninitialized and
        # -Warray-bounds only runs with optimisation, so analysing at the build's own -O0 would report almost
        # nothing and read as a clean pass.
        cmd = [require("clang++" if clang else "g++"), "-c", "-o", "/dev/null", *flags, *warn, "-O2"]
        if tool == "analyzer":
            cmd.extend(["--analyze", "-Xclang", "-analyzer-output=text"] if clang else ["-fanalyzer"])
        cmd.append(str(src))
    elif tool == "clang-tidy":
        # Only the generated TU is ours to fix: an empty --header-filter drops diagnostics raised in the DaCe
        # runtime headers and the vendored third-party ones they pull in.
        cmd = [
            require("clang-tidy"), "--quiet", f"--checks={CLANG_TIDY_CHECKS}", "--header-filter=",
            "--system-headers=false",
            str(src), "--", *flags
        ]
    else:
        cmd = [require("cppcheck"), "--enable=warning", "--inline-suppr", "--quiet"]
        cmd.extend(f"--suppress={s}" for s in CPPCHECK_SUPPRESSIONS)
        cmd.extend(f for f in flags if f.startswith(("-I", "-D")))
        cmd.append(str(src))

    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = done.stdout + done.stderr
    if done.returncode != 0 and tool in ("warnings", "analyzer"):
        # A compile that FAILED reports no warnings, which is indistinguishable from a clean TU
        # once the lines are filtered below -- the exact silent degradation this module exists to
        # prevent (see the docstring).  A driver/front-end fatal (`cc1plus: fatal error:`) matches
        # neither ": warning:" nor ": error:", so it cannot be caught by the filter either.
        raise RuntimeError(f"codegen analysis compile failed (exit {done.returncode}) for {src}:\n"
                           f"{shlex.join(cmd)}\n{text[-3000:]}")
    diagnostics = [ln for ln in text.splitlines() if ": warning:" in ln or ": error:" in ln]
    # Belt-and-braces across all four tools: a diagnostic whose file is not the generated TU came from a header we
    # do not own, and nothing in this repo can act on it.
    diagnostics = [ln for ln in diagnostics if str(src) in ln]
    if critical_only and tool in ("warnings", "analyzer"):
        # The build's own flags carry -Wall -Wextra, so the compiler also reports style warnings
        # (-Wunused-but-set-variable fires in the hundreds on generated code).  Gate on the tag, not on the word
        # "warning", or the critical signal drowns in noise nobody will read.
        critical = critical_tags(clang)
        diagnostics = [ln for ln in diagnostics if any(tag in ln for tag in critical)]
    return diagnostics
