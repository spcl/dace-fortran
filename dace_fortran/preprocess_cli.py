"""CLI wrapper over the Fortran preprocess passes, for build-system steps.

Passes run in the same order as ``preprocess_fortran_source`` (merge ->
strip-openmp -> normalize-kind -> rewrite-integer-powers -> extras);
keep defaults in sync with it.

Exit codes: 0 success, 2 argument error, 3 pass refusal.
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
    p = argparse.ArgumentParser(
        prog="python -m dace_fortran.preprocess_cli",
        description="Apply DaCe-Fortran source-text preprocess passes.",
    )
    p.add_argument("--in",
                   dest="in_path",
                   required=True,
                   help="Input .f90 / .F90 source.  Use '-' for stdin.  "
                   "Repeatable when paired with --inplace -- the easiest "
                   "build-system-free way to apply the rewrites: each "
                   "file is rewritten in place, the user's existing "
                   "compiler builds the result with no further glue.",
                   action="append")
    p.add_argument("--out",
                   dest="out_path",
                   help="Output path.  Default: stdout.  When set + "
                   "--rewrite-string-enum is on, an additional "
                   "<out>.enum_maps.json sidecar is written.  Mutually "
                   "exclusive with --inplace.")
    p.add_argument("--inplace",
                   action="store_true",
                   help="Rewrite each --in path in place (atomic write "
                   "via a sibling tempfile + rename).  Skip the cmake / "
                   "automake glue entirely  --  run this once over your "
                   "source tree, then let your existing build system "
                   "compile the rewritten files.")
    p.add_argument("--backup-suffix",
                   default=None,
                   help="When --inplace is on, keep a backup of each "
                   "original next to it with this suffix (e.g. .orig).  "
                   "Default: no backup kept.")
    p.add_argument("--search-dir",
                   dest="search_dirs",
                   action="append",
                   default=[],
                   help="Directory (recursive) of sibling sources scanned by "
                   "module-resolving passes (merge / external).  Repeat.")

    # Pass switches (all default off).
    p.add_argument("--merge-modules", action="store_true", help="Inline every ``USE``-d module's source.")
    p.add_argument("--merge-engine",
                   choices=("regex", "fparser"),
                   default="regex",
                   help="Which --merge-modules engine to use: 'regex' "
                   "(default, the fparser-free text-splicer) or 'fparser' "
                   "(the AST inliner -- also desugars + prunes).")
    p.add_argument("--merge-entry",
                   default=None,
                   help="Entry procedure (plain name / module::proc / "
                   "mangled symbol) for the fparser engine's pruning; "
                   "ignored by the regex engine.")
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


def _apply_passes(source: str, args) -> tuple:
    """Apply the requested passes (in canonical order) to one source
    string.  Returns ``(rewritten_source, enum_maps)``."""
    if args.all_defaults:
        args.merge_modules = True
        args.strip_openmp = True
        args.normalize_kind = True
        args.rewrite_integer_powers = True
    if args.merge_modules:
        if args.merge_engine == "fparser":
            from dace_fortran.preprocess import _fparser_merge
            source = _fparser_merge(source, search_dirs=args.search_dirs, entry=args.merge_entry)
        else:
            source = merge_used_modules(source, search_dirs=args.search_dirs)
    if args.strip_openmp:
        source = strip_openmp_directives(source)
    if args.normalize_kind and not args.kind_passthrough:
        source = normalize_kind_parameters(source, kind_map=_parse_kind_map(args.kind_map))
    if args.rewrite_integer_powers:
        source = rewrite_integer_powers(source)
    if args.rewrite_external:
        source = replace_external_with_modules(source, search_dirs=args.search_dirs)
    if args.rewrite_if_intvar:
        source = preprocess_fortran(source)
    enum_maps: dict = {}
    if args.rewrite_string_enum:
        source, enum_maps = rewrite_string_enum_to_integer(source)
    return source, enum_maps


def _rewrite_inplace(in_path: Path, args) -> dict:
    """Read ``in_path``, apply passes, atomically replace the original
    via a sibling tempfile + rename.  Returns the enum_maps (empty
    when --rewrite-string-enum is off)."""
    import os
    import tempfile
    src_text = in_path.read_text()
    rewritten, emaps = _apply_passes(src_text, args)
    if rewritten == src_text:
        # No-op: skip the write to preserve mtime for incremental rebuilds.
        return emaps
    if args.backup_suffix:
        backup = in_path.with_name(in_path.name + args.backup_suffix)
        backup.write_text(src_text)
    fd, tmp = tempfile.mkstemp(dir=str(in_path.parent), prefix=f".{in_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(rewritten)
        os.replace(tmp, str(in_path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if emaps:
        sidecar = in_path.with_name(in_path.name + ".enum_maps.json")
        sidecar.write_text(json.dumps(emaps, indent=2, sort_keys=True))
    return emaps


def main(argv=None) -> int:
    """Runs the CLI; returns 0 on success, 2 on argument error, 3 on pass refusal."""
    args = _build_parser().parse_args(argv)

    # argparse can't express this 3-way exclusivity (--inplace/--out/stdout); check by hand.
    if args.inplace and args.out_path:
        raise SystemExit("--inplace and --out are mutually exclusive")
    if args.rewrite_external and not args.search_dirs:
        print("warning: --rewrite-external is a no-op without --search-dir", file=sys.stderr)

    # --inplace: build-system-free path, run once over the source tree.
    if args.inplace:
        # --inplace supports multiple --in (unlike --out, which uses only the last).
        for raw_in in args.in_path:
            if raw_in == "-":
                raise SystemExit("--inplace cannot be used with --in -")
            _rewrite_inplace(Path(raw_in), args)
        return 0

    # Non-inplace: multiple --in uses the last (argparse store-the-last); warn.
    if len(args.in_path) > 1:
        print("warning: multiple --in without --inplace -- using last", file=sys.stderr)
    in_path_str = args.in_path[-1]

    if in_path_str == "-":
        source = sys.stdin.read()
    else:
        source = Path(in_path_str).read_text()

    source, enum_maps = _apply_passes(source, args)

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
