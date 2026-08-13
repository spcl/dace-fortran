"""Give the serde header's namespace-scope functions ``inline`` linkage.

``include/serde_velocity_no_nproma.h`` is vendored and defines 53 free functions at namespace
scope, so any second translation unit that includes it is a duplicate-symbol link error. That
blocks ``src/velocity_split_dispatch.cpp``, which needs serde because ``VELOCITY_CALL_ARGS``
expands to ``serde::ARRAY_META_DICT_AT`` lookups for the three bare ``z_*`` arrays' extents.

``inline``, never ``static``. ``ARRAY_META_DICT()`` returns a pointer to a function-local ``static``
map that the runner fills during deserialisation. Internal linkage would give every translation
unit its own empty copy, so the dispatch TU's ``.at()`` would throw ``std::out_of_range`` at run
time -- a link error traded for a runtime one. ``inline`` merges the definition and shares the map;
``tests/`` covers exactly this.

Idempotent, so it is safe to re-run after the header is re-vendored.

Run:
    python tools/inline_serde_header.py            # patch in place
    python tools/inline_serde_header.py --check    # non-zero exit if any definition lacks inline
"""

import argparse
import re
import sys
from pathlib import Path

HEADER = Path("include/serde_velocity_no_nproma.h")

# Lines that open a type, namespace or directive rather than a function definition. ``template`` is
# excluded because templates already have vague linkage.
SKIP_PREFIXES = ("namespace", "struct", "class", "enum", "using", "template", "typedef", "extern", "#", "}", "//",
                 "inline")

# A namespace-scope definition: column 0, a return type, then a name and a parameter list.
DEFN_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9:<>,*&\s]*[*&\s][A-Za-z_][A-Za-z_0-9]*\s*\(")


def opens_body(lines: list[str], start: int) -> bool:
    """True if the signature at ``start`` opens a body rather than ending in a declaration."""
    for j in range(start, min(start + 8, len(lines))):
        text = lines[j].rstrip()
        if text.endswith(";"):
            return False
        if text.endswith("{") or (j == start and "{" in text):
            return True
    return False


def patch(lines: list[str]) -> tuple[list[str], list[int]]:
    out, touched = list(lines), []
    for i, line in enumerate(lines):
        if not line or line[0].isspace() or line.startswith(SKIP_PREFIXES):
            continue
        if not DEFN_RE.match(line) or not opens_body(lines, i):
            continue
        out[i] = "inline " + line
        touched.append(i + 1)
    return out, touched


def main() -> None:
    argp = argparse.ArgumentParser()
    argp.add_argument("--check", action="store_true", help="exit non-zero if the header is unpatched")
    args = argp.parse_args()

    target = Path(__file__).resolve().parent.parent / HEADER
    lines = target.read_text().split("\n")
    patched, touched = patch(lines)

    if args.check:
        if touched:
            print(f"{HEADER}: {len(touched)} definition(s) still need inline, first at line {touched[0]}",
                  file=sys.stderr)
            raise SystemExit(1)
        print(f"{HEADER}: every namespace-scope definition is inline")
        return

    if not touched:
        print(f"{HEADER}: already patched, nothing to do")
        return
    target.write_text("\n".join(patched))
    print(f"{HEADER}: inlined {len(touched)} definition(s)")


if __name__ == "__main__":
    main()
