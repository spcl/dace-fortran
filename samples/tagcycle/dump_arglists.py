"""Dump ``sdfg.arglist()`` for each .sdfgz given, in the arglists_094035d.index format.

Writes ``<stem>.arglist.txt`` beside every input (3 header lines, then ``idx\tname\tdtype\tshape``)
and prints one ``%5d  <relpath>`` index line per artifact on stdout.
"""
import os
import sys
from pathlib import Path

from dace import SDFG


def load(path: Path):
    """None when the artifact will not load (truncated .sdfgz from a killed writer)."""
    try:
        return SDFG.from_file(str(path))
    except Exception as exc:  # noqa: BLE001 -- a truncated pickle raises anything
        print(f"ARGLIST_DUMP_CORRUPT {path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        try:
            path.unlink()  # unusable either way; removing it makes the next warm rebuild instead
        except OSError as unlink_exc:
            print(f"  could not remove {path}: {unlink_exc}", file=sys.stderr, flush=True)
        return None


def dump(sdfg, path: Path, root: Path) -> str:
    args = sdfg.arglist()
    out = path.parent / (path.name[:-len(".sdfgz")] + ".arglist.txt")
    with out.open("w") as f:
        f.write(f"# {path.name}\n# sdfg.name={sdfg.name}\n# nargs={len(args)}\n")
        for i, (name, desc) in enumerate(args.items()):
            f.write(f"{i}\t{name}\t{desc.dtype.ctype}\t{tuple(desc.shape)}\n")
    # relpath, not Path.relative_to: an artifact reached through a symlink is not a subpath.
    return f"{len(args):5d}  {os.path.relpath(path, root)}"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: dump_arglists.py <root> <artifact.sdfgz> [...]", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    rc = 0
    for raw in sys.argv[2:]:
        path = Path(raw).resolve()
        # A corrupt artifact is skipped, not fatal: the warm job's per-artifact check reports it
        # as a MISSING dump (DIVERGED) without costing the other artifacts their dumps.
        sdfg = load(path)
        if sdfg is None:
            continue
        try:
            print(dump(sdfg, path, root), flush=True)
        except Exception as exc:  # one unreadable artifact must not cost the others their dump
            print(f"ARGLIST_DUMP_FAILED {path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
