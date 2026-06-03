"""Standalone CLI for the Fortran source-text preprocess passes.

Wraps every pass that operates on a source string so build systems
(cmake / automake / make) can invoke them as a command step:

    python -m dace_fortran.preprocess_cli \\
        --rewrite-external \\
        --normalize-kind \\
        --search-dir src/utils \\
        --in src/kernel.f90 \\
        --out build/preprocessed/kernel.f90

Multiple ``--rewrite-*`` flags compose; the passes run in the same
order ``preprocess_fortran_source`` uses internally (merge -> strip-
OpenMP -> normalize-kind -> rewrite-integer-powers -> opt-in
extras).  Default behaviour matches ``preprocess_fortran_source``
defaults so the CLI is a thin wrapper, not a parallel
implementation.

Pattern 2 (string-enum) has a sidecar concern: the rewrite returns
``(rewritten_source, enum_maps)``.  When ``--rewrite-string-enum``
is on the CLI writes the enum maps to ``<out>.enum_maps.json``
alongside the rewritten source so the binding-generation step
downstream can consume them.

Exit codes:
    0 -- success (source written; maps written if any).
    2 -- argument / invocation error.
    3 -- a pass refused to rewrite (e.g. unresolved EXTERNAL the
         user expected resolved).  Stderr carries the diagnostic.
"""
import argparse
import json
import sys
from pathlib import Path

from dace_fortran.preprocess import (
    merge_used_modules,
    normalize_kind_parameters,
    preprocess_fortran,
    replace_external_with_modules,
    rewrite_integer_powers,
    rewrite_string_enum_to_integer,
    strip_openmp_directives,
)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="python -m dace_fortran.preprocess_cli",
        description="Apply DaCe-Fortran source-text preprocess passes.",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input .f90 / .F90 source.  Use '-' for stdin.")
    p.add_argument("--out",
                   dest="out_path",
                   help="Output path.  Default: stdout.  When set + "
                   "--rewrite-string-enum is on, an additional "
                   "<out>.enum_maps.json sidecar is written.")
    p.add_argument("--search-dir",
                   dest="search_dirs",
                   action="append",
                   default=[],
                   help="Directory (recursive) of sibling sources scanned by "
                   "module-resolving passes (merge / external).  Repeat.")

    # Pass switches.  All default off (caller picks what to apply).
    p.add_argument("--merge-modules", action="store_true", help="Inline every ``USE``-d module's source.")
    p.add_argument("--strip-openmp", action="store_true", help="Drop OpenMP / OpenACC sentinel directives.")
    p.add_argument("--rewrite-integer-powers", action="store_true", help="Expand ``x**2.0`` to ``x*x``.")
    p.add_argument("--normalize-kind",
                   action="store_true",
                   help="Substitute precision aliases (wp, sp, dp, qp) "
                   "with literal kind integers.")
    p.add_argument("--kind-passthrough",
                   action="store_true",
                   help="Force-skip the kind rewrite (e.g. when "
                   "upstream already resolved every alias).")
    p.add_argument("--kind-map",
                   action="append",
                   default=[],
                   help="Override one kind alias.  Format: NAME=N (e.g. "
                   "wp=4 for fp32).  NAME=NONE leaves the alias alone.")
    p.add_argument("--rewrite-external",
                   action="store_true",
                   help="Resolve ``EXTERNAL`` to ``USE`` imports against "
                   "modules under --search-dir.")
    p.add_argument("--rewrite-string-enum",
                   action="store_true",
                   help="Convert CHARACTER enum-style dummies to INTEGER "
                   "+ emit <out>.enum_maps.json sidecar for bindings.")
    p.add_argument("--rewrite-if-intvar", action="store_true", help="Rewrite ``IF (intvar)`` to ``IF (intvar /= 0)``.")
    p.add_argument("--all-defaults",
                   action="store_true",
                   help="Apply the same default mix as ``preprocess_"
                   "fortran_source``: merge + strip-OpenMP + "
                   "normalize-kind + rewrite-integer-powers.")
    return p


def _parse_kind_map(items) -> dict:
    """Parse ``--kind-map`` CLI items into the dict the pass accepts."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--kind-map expects NAME=N, got {item!r}")
        name, raw = item.split("=", 1)
        out[name.strip().lower()] = None if raw.strip().upper() == "NONE" else int(raw)
    return out


def main(argv=None) -> int:
    """Run the CLI; returns the process exit code.

    :param argv: optional list of command-line tokens (excluding the
        program name); defaults to ``sys.argv[1:]``.
    :returns: 0 on success, 2 on argument errors, 3 on pass refusal.
    """
    args = _build_parser().parse_args(argv)

    # Resolve input.
    if args.in_path == "-":
        source = sys.stdin.read()
    else:
        source = Path(args.in_path).read_text()

    # Apply passes in the canonical order so composition matches
    # what ``preprocess_fortran_source`` already exposes.
    if args.all_defaults:
        args.merge_modules = True
        args.strip_openmp = True
        args.normalize_kind = True
        args.rewrite_integer_powers = True

    if args.merge_modules:
        source = merge_used_modules(source, search_dirs=args.search_dirs)
    if args.strip_openmp:
        source = strip_openmp_directives(source)
    if args.normalize_kind and not args.kind_passthrough:
        source = normalize_kind_parameters(source, kind_map=_parse_kind_map(args.kind_map))
    if args.rewrite_integer_powers:
        source = rewrite_integer_powers(source)
    if args.rewrite_external:
        # Conservative: a pass with empty ``search_dirs`` is a no-op
        # by design.  Surface a clear hint if the user enabled
        # --rewrite-external without any --search-dir.
        if not args.search_dirs:
            print("warning: --rewrite-external is a no-op without --search-dir", file=sys.stderr)
        source = replace_external_with_modules(source, search_dirs=args.search_dirs)
    if args.rewrite_if_intvar:
        source = preprocess_fortran(source)

    enum_maps: dict = {}
    if args.rewrite_string_enum:
        source, enum_maps = rewrite_string_enum_to_integer(source)

    # Write rewritten source.
    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source)
        if args.rewrite_string_enum and enum_maps:
            sidecar = out.with_suffix(out.suffix + ".enum_maps.json")
            sidecar.write_text(json.dumps(enum_maps, indent=2, sort_keys=True))
            print(f"wrote {sidecar} ({sum(len(t) for t in enum_maps.values())} args)", file=sys.stderr)
    else:
        sys.stdout.write(source)
        if args.rewrite_string_enum and enum_maps:
            print(json.dumps(enum_maps, indent=2, sort_keys=True), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
