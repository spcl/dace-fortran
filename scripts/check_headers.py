#!/usr/bin/env python
# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enforce the copyright / SPDX header on the core Python package.

Every tracked ``.py`` file under ``dace_fortran/`` must begin with a copyright line
immediately followed by::

    # SPDX-License-Identifier: GPL-3.0-or-later

(An optional ``#!`` shebang and/or a PEP 263 ``coding`` line may precede it; the
copyright line is then required immediately after, with the SPDX line right below it.)

The copyright line is matched loosely (any single year or year range, any
``ETH Zurich and the <project> authors`` phrasing) so a file that predates the current
wording still counts -- ``--fix`` never rewrites or duplicates that line, it only
appends the missing SPDX line directly below it. A file with no copyright line at all
gets the full canonical two-line header inserted instead.

Scope -- the CORE package only: ``dace_fortran/``. ``tests/`` and top-level ``scripts/``
are a separate, deferred header pass (dace-fortran's ``tests/`` holds byte-compared
fixture sources, mirroring the dirs the format hook already treats specially).

Run standalone with no arguments and the tool discovers the scope via
``git ls-files``; pre-commit instead passes the staged files as positional
arguments (already narrowed by the hook's path filter). ``--fix`` inserts the
header in place; without it the tool only reports, exiting 1 when any in-scope
file is missing the header (the offenders and the fix command are printed).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SPDX_LINE = "# SPDX-License-Identifier: GPL-3.0-or-later"
# What --fix writes into a file with no copyright line at all (the canonical form).
HEADER = (
    "# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.",
    SPDX_LINE,
)
# A copyright line, loosely matched: any single year or year range, and any
# authors-org phrase, with the trailing "All rights reserved." optional -- so both the
# pre-existing DaCe-wide convention and the canonical HEADER above are recognized.
COPYRIGHT_RE = re.compile(
    r"^# Copyright \d{4}(-\d{4})? ETH Zurich and the [\w.-]+ authors\.(\s+All rights reserved\.)?$")

# Included root; dace-fortran has no ported-kernel / generated-distribution
# subtree analogous to an upstream project's vendored third-party code, so there is
# nothing to carve back out here.
SCOPE_PREFIXES = ("dace_fortran/", )

CODING_RE = re.compile(r"^[ \t\f]*#.*?coding[:=]")


def in_scope(rel: str) -> bool:
    posix = rel.replace("\\", "/")
    if not posix.endswith(".py"):
        return False
    return any(posix.startswith(p) for p in SCOPE_PREFIXES)


def tracked_python() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []


def prefix_len(lines: list[str]) -> int:
    """Number of leading lines (shebang, then optional coding) the header goes after."""
    idx = 0
    if idx < len(lines) and lines[idx].startswith("#!"):
        idx += 1
    if idx < len(lines) and CODING_RE.match(lines[idx]):
        idx += 1
    return idx


def copyright_line_index(lines: list[str]) -> int | None:
    """Index of the existing copyright line (after any shebang/coding), or None."""
    idx = prefix_len(lines)
    if idx < len(lines) and COPYRIGHT_RE.match(lines[idx]):
        return idx
    return None


def has_header(lines: list[str]) -> bool:
    cr_idx = copyright_line_index(lines)
    if cr_idx is None:
        return False
    return cr_idx + 1 < len(lines) and lines[cr_idx + 1] == SPDX_LINE


def insert_header(path: Path) -> bool:
    """Insert the missing piece (SPDX line, or the full header) in place. Returns True when it wrote."""
    text = path.read_text(encoding="utf-8")
    raw_lines = text.splitlines(keepends=True)
    stripped = [ln.rstrip("\r\n") for ln in raw_lines]
    if has_header(stripped):
        return False
    # Preserve the newline style already used at the insertion point.
    newline = "\r\n" if raw_lines and raw_lines[0].endswith("\r\n") else "\n"

    cr_idx = copyright_line_index(stripped)
    if cr_idx is not None:
        # Existing copyright line, no SPDX line yet -- append just the SPDX line below it.
        insert_at = cr_idx + 1
        block = [SPDX_LINE + newline]
    else:
        # No copyright line at all -- insert the full canonical two-line header.
        insert_at = prefix_len(stripped)
        block = [line + newline for line in HEADER]

    path.write_text("".join(raw_lines[:insert_at] + block + raw_lines[insert_at:]), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fix", action="store_true", help="insert the header in place instead of failing")
    ap.add_argument("files", nargs="*", help="explicit files to check (default: all tracked in-scope .py)")
    args = ap.parse_args(argv)

    candidates = args.files if args.files else tracked_python()
    targets = [rel for rel in sorted(set(candidates)) if in_scope(rel) and (REPO_ROOT / rel).is_file()]

    offenders = []
    for rel in targets:
        path = REPO_ROOT / rel
        if args.fix:
            if insert_header(path):
                offenders.append(rel)
        else:
            lines = [ln.rstrip("\r\n") for ln in path.read_text(encoding="utf-8").splitlines(keepends=True)]
            if not has_header(lines):
                offenders.append(rel)

    if args.fix:
        print(f"check-headers: inserted the header into {len(offenders)} of {len(targets)} in-scope file(s)")
        return 0
    if not offenders:
        print(f"check-headers: {len(targets)} in-scope file(s) OK")
        return 0
    print(f"check-headers: {len(offenders)} of {len(targets)} in-scope file(s) missing the copyright/SPDX header:\n")
    for rel in offenders:
        print(f"  {rel}")
    print("\nFix with:  python scripts/check_headers.py --fix")
    return 1


if __name__ == "__main__":
    sys.exit(main())
