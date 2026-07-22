"""Static analysis of the C++ DaCe generates for a built SDFG.

The generated code is already compiled with ``-Wall -Wextra`` (see the build's ``flags.make``), but nothing reads
the output, so undefined behaviour ships silently: an ICON halo body shipped a buffer sized from an uninitialised
local for months, and the only symptom was a glibc abort inside an unrelated ``free()``.

:data:`CRITICAL_WARNINGS` is the subset that means "this code has UB", not "this code is untidy" -- those are the
ones a build must never emit.  Deep analysis follows the compiler: gcc builds get ``-fanalyzer``, clang builds get
the LLVM static analyzer, so the analysis always matches the toolchain that produced the binary.

A missing tool raises: silently degrading to "nothing found" is how this class of bug survived in the first place.
"""

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List

from dace.config import Config
from dace.sdfg import SDFG

# Warnings that indicate undefined behaviour rather than style.  A generated TU emitting any of these is a codegen
# bug: the reader of the generated code is a compiler, so there is no "intentional" uninitialised read to excuse.
CRITICAL_WARNINGS = (
    "uninitialized",
    "maybe-uninitialized",
    "array-bounds",
    "stringop-overflow",
    "free-nonheap-object",
    "nonnull",
    "return-type",
    "sizeof-pointer-memaccess",
)

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
    """The exact defines/includes/flags CMake used for the generated TU, read back from its ``flags.make``.

    Reusing the build's own flags keeps the analysis honest -- a hand-rebuilt include path analyses code the
    compiler never saw.
    """
    path = Path(sdfg.build_folder) / "build" / "CMakeFiles" / f"{sdfg.name}.dir" / "flags.make"
    if not path.is_file():
        raise FileNotFoundError(f"no {path}; the SDFG must be built (not just codegen'd) before analysis")
    flags: List[str] = []
    for line in path.read_text().splitlines():
        for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            if line.startswith(key):
                flags.extend(shlex.split(line.split("=", 1)[1]))
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


def analyze(sdfg: SDFG, tool: str = "warnings") -> List[str]:
    """Run ``tool`` over the generated C++ and return the diagnostic lines it reported.

    ``warnings`` and ``analyzer`` follow the configured compiler (gcc ``-fanalyzer`` / clang ``--analyze``);
    the other two are their own binaries.  Only correctness checks are enabled, so a non-empty result is always
    actionable.
    """
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; expected one of {TOOLS}")
    src = generated_source(sdfg)
    flags = build_flags(sdfg)
    clang = compiler_is_clang()

    if tool in ("warnings", "analyzer"):
        warn = [f"-W{w}" for w in (CLANG_CRITICAL_WARNINGS if clang else CRITICAL_WARNINGS)]
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
    diagnostics = [ln for ln in text.splitlines() if ": warning:" in ln or ": error:" in ln]
    # Belt-and-braces across all four tools: a diagnostic whose file is not the generated TU came from a header we
    # do not own, and nothing in this repo can act on it.
    diagnostics = [ln for ln in diagnostics if str(src) in ln]
    if tool in ("warnings", "analyzer"):
        # The build's own flags carry -Wall -Wextra, so the compiler also reports style warnings
        # (-Wunused-but-set-variable fires in the hundreds on generated code).  Gate on the tag, not on the word
        # "warning", or the critical signal drowns in noise nobody will read.
        critical = {f"[-W{w}]" for w in (CLANG_CRITICAL_WARNINGS if clang else CRITICAL_WARNINGS)}
        diagnostics = [ln for ln in diagnostics if any(tag in ln for tag in critical)]
    return diagnostics
