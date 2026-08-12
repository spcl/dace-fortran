"""Verify ``include/call_velocity.h`` matches the signature of every variant.

Three invariants checked:

1. All 4 DaCe-generated ``__program_*`` signatures have identical parameter
   names and types (positional), modulo the variant-specific name prefix.
   This is what ``unify_variant_signatures`` is supposed to guarantee.
2. ``__dace_init_*`` of each variant has the same parameter list as its
   ``__program_*`` (minus the ``state_t *`` state slot).
3. The expression list emitted by ``tools/gen_call_site.py`` has one entry
   per param, in the same positional order.

Exits non-zero with a diff report on any failure.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from gen_call_site import parse_signature, parse_structs, map_param


VARIANTS = [
    "velocity_no_nproma_if_prop_lvn_only_0_istep_1",
    "velocity_no_nproma_if_prop_lvn_only_0_istep_2",
    "velocity_no_nproma_if_prop_lvn_only_1_istep_1",
    "velocity_no_nproma_if_prop_lvn_only_1_istep_2",
]


def _header(variant: str) -> Path:
    return Path(".dacecache") / variant / "include" / f"{variant}.h"


def _parse_init(header: Path, variant: str) -> List[Tuple[str, bool, str]]:
    """Same shape tuple as ``parse_signature``, but for ``__dace_init_*``.
    Also strips the returned state_t* (init has no state parameter)."""
    text = header.read_text()
    m = re.search(
        rf"\*\s*__dace_init_{variant}\((.*?)\);", text, re.S
    )
    if not m:
        raise SystemExit(f"couldn't find __dace_init for {variant}")
    raw = m.group(1)
    parts = [p.strip() for p in raw.split(",")]
    out: List[Tuple[str, bool, str]] = []
    for p in parts:
        toks = p.split()
        name = toks[-1]
        rest = " ".join(toks[:-1]).strip()
        ptr = "*" in rest
        base = rest.replace("*", " ").replace("__restrict__", " ").strip()
        base = re.sub(r"\s+", " ", base)
        out.append((base, ptr, name))
    return out


def _program_of(variant: str) -> List[Tuple[str, bool, str]]:
    return parse_signature(_header(variant), f"__program_{variant}")


def _diff_names(lhs: List[str], rhs: List[str], lhs_name: str, rhs_name: str) -> List[str]:
    out = []
    n = max(len(lhs), len(rhs))
    for i in range(n):
        a = lhs[i] if i < len(lhs) else "<missing>"
        b = rhs[i] if i < len(rhs) else "<missing>"
        if a != b:
            out.append(f"  [{i}] {lhs_name}: {a!r}   {rhs_name}: {b!r}")
    return out


def main():
    failed = False

    # 1. All 4 program signatures must match in names + types + order.
    sigs = {v: _program_of(v) for v in VARIANTS}
    ref = VARIANTS[0]
    ref_names = [t[2] for t in sigs[ref]]
    ref_types = [(t[0], t[1]) for t in sigs[ref]]
    for v in VARIANTS[1:]:
        names = [t[2] for t in sigs[v]]
        types = [(t[0], t[1]) for t in sigs[v]]
        if names != ref_names:
            failed = True
            diffs = _diff_names(ref_names, names, ref, v)
            print(f"FAIL: {v} param names differ from {ref}:")
            for d in diffs[:10]:
                print(d)
            if len(diffs) > 10:
                print(f"  ... {len(diffs) - 10} more")
        if types != ref_types:
            failed = True
            tdiffs = [
                f"  [{i}] {ref}: {ref_types[i]}   {v}: {types[i]}"
                for i in range(min(len(types), len(ref_types)))
                if types[i] != ref_types[i]
            ]
            print(f"FAIL: {v} param types differ from {ref}:")
            for d in tdiffs[:10]:
                print(d)
    if not failed:
        print(f"OK: all 4 __program_* signatures match ({len(ref_names)} params each)")

    # 2. __dace_init_* must match its sibling __program_* (modulo state slot).
    for v in VARIANTS:
        prog = sigs[v]
        init = _parse_init(_header(v), v)
        if [t[2] for t in init] != [t[2] for t in prog]:
            failed = True
            print(f"FAIL: {v} __dace_init_ parameter order != __program_")
            diffs = _diff_names([t[2] for t in prog], [t[2] for t in init], "program", "init")
            for d in diffs[:10]:
                print(d)
        elif [(t[0], t[1]) for t in init] != [(t[0], t[1]) for t in prog]:
            failed = True
            print(f"FAIL: {v} __dace_init_ parameter types != __program_")
    if not failed:
        print("OK: __dace_init_ parameter lists match __program_ for all 4 variants")

    # 3. The generator's output positions must match the reference signature,
    # and re-invoking map_param must reproduce the emitted expressions.
    structs = parse_structs(Path("include/velocity_tendencies_no_nproma.h"))
    expected_exprs = [
        map_param(structs, base, ptr, name) for base, ptr, name in sigs[ref]
    ]

    emitted_path = Path("include/call_velocity.h")
    if not emitted_path.exists():
        failed = True
        print(f"FAIL: {emitted_path} not present -- run tools/gen_call_site.py")
    else:
        text = emitted_path.read_text()
        # Grab everything after the macro body starts and before the #endif.
        body = text.split("lvn_only, ntnd) \\\n", 1)[1].split("#endif", 1)[0]
        # Each arg line ends with " \" (trailing continuation) or just the expr.
        args = []
        for line in body.splitlines():
            s = line.strip()
            if not s:
                continue
            s = s.rstrip("\\").rstrip()
            if s.endswith(","):
                s = s[:-1].rstrip()
            if s:
                args.append(s)
        if args != expected_exprs:
            failed = True
            print(
                f"FAIL: emitted macro args ({len(args)}) differ from map_param output ({len(expected_exprs)})"
            )
            diffs = _diff_names(expected_exprs, args, "map_param", "emitted")
            for d in diffs[:10]:
                print(d)
    if not failed:
        print(f"OK: include/call_velocity.h matches {len(expected_exprs)} expected expressions")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
