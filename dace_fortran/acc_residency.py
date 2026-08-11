# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-argument OpenACC data residency for a Fortran routine.

The HLFIR parse path never sees ``!$ACC``: :func:`dace_fortran.preprocess.
strip_openmp_directives` deletes the sentinels before flang runs, and flang is
invoked without ``-fopenacc`` (``build.py``, ``emit_hlfir.py``,
``flang_codebase.py`` all pass ``-U_OPENACC``), so the directives are plain
comments and comments never enter a flang parse tree.  Residency is therefore
recovered by a text pre-pass over the ORIGINAL source, before preprocessing.

The pre-pass answers, for every dummy argument of a target routine, whether the
data is device- or host-resident at call time, and emits the sidecar
``<routine>.acc_residency.json`` consumed by the ICON binding generator::

    {"routine": ..., "source": ...,
     "args": {"<arg>": {"residency": "device"|"host",
                        "clause": "PRESENT", "evidence": "<file>:<line>"}},
     "unclassified": ["<arg>", ...]}

An argument with no ACC evidence is reported in ``unclassified``; it is never
defaulted into ``host``, because "no directive mentions it" and "the directives
say it is on the host" call for different actions on the binding side.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Clauses proving the data has a device copy at call time.  ``DELETE`` /
#: ``DETACH`` qualify: only device-resident data can be removed from the device.
DEVICE_CLAUSES = frozenset({
    "present",
    "copy",
    "copyin",
    "copyout",
    "create",
    "deviceptr",
    "device_resident",
    "attach",
    "detach",
    "delete",
    "device",
    "use_device",
    "link",
    "present_or_copy",
    "present_or_copyin",
    "present_or_copyout",
    "present_or_create",
    "pcopy",
    "pcopyin",
    "pcopyout",
    "pcreate",
    "pcopy_in",
    "pcopy_out",
})

#: Clauses that leave the data usable from the host.  ``NO_CREATE`` is the only
#: ACC clause admitting host memory as the fallback; ``UPDATE HOST/SELF`` marks
#: the host copy current.  Device evidence always outranks these (see
#: :func:`classify`), so an argument that is both ``PRESENT`` and later
#: ``UPDATE SELF`` stays ``device``.
HOST_CLAUSES = frozenset({"no_create", "host", "self"})

_ACC_SENTINEL_RE = re.compile(r"^\s*!\s*\$\s*acc\b", re.IGNORECASE)
_IDENT_RE = re.compile(r"\s*([A-Za-z_]\w*)")
_CLAUSE_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")
_ROUTINE_END_RE = re.compile(r"^\s*end\s*(?:subroutine|function)\b", re.IGNORECASE)
_ROUTINE_DECL_RE = re.compile(r"^\s*(?:[\w*(),:=\s]*?\b)?(?:subroutine|function)\s+[A-Za-z_]\w*", re.IGNORECASE)

#: Directive heads whose second word is part of the head itself.
_TWO_WORD_HEADS = frozenset({"end data", "enter data", "exit data", "end host_data"})

#: Heads opening a structured data region (depth +1).
_REGION_OPEN = frozenset({"data", "host_data"})
_REGION_CLOSE = frozenset({"end data", "end host_data"})


def _strip_comment(line: str) -> str:
    """Drop a trailing Fortran comment, honouring quoted strings."""
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return line[:i]
    return line


class _Logical:
    """A directive joined across ``&`` continuations, with offset->line map."""

    __slots__ = ("text", "_spans", "line")

    def __init__(self, pieces):
        parts, spans, off = [], [], 0
        for lineno, piece in pieces:
            parts.append(piece)
            spans.append((off, off + len(piece), lineno))
            off += len(piece) + 1
        self.text = " ".join(parts)
        self._spans = spans
        self.line = pieces[0][0] if pieces else 0

    def line_at(self, offset: int) -> int:
        for start, end, lineno in self._spans:
            if start <= offset <= end:
                return lineno
        return self.line


def _join_continuations(lines, index, sentinel):
    """Collect the physical pieces of one logical statement starting at ``index``."""
    pieces, i = [], index
    while i < len(lines):
        if sentinel is None:
            body = _strip_comment(lines[i])
        else:
            match = sentinel.match(lines[i])
            if match is None:
                break
            body = _strip_comment(lines[i][match.end():])
        body = body.rstrip()
        continued = body.endswith("&")
        if continued:
            body = body[:-1]
        body = body.strip()
        if body.startswith("&"):
            body = body[1:].strip()
        pieces.append((i + 1, body))
        if not continued:
            break
        i += 1
    return pieces, i


def acc_directives(source: str):
    """Yield every ``!$ACC`` directive of ``source`` as a :class:`_Logical`."""
    lines = source.splitlines()
    out, i = [], 0
    while i < len(lines):
        if _ACC_SENTINEL_RE.match(lines[i]):
            pieces, i = _join_continuations(lines, i, _ACC_SENTINEL_RE)
            out.append(_Logical(pieces))
        i += 1
    return out


def _head(text: str) -> str:
    words = re.findall(r"[A-Za-z_]\w*", text[:64])
    if not words:
        return ""
    two = " ".join(words[:2]).lower()
    if two in _TWO_WORD_HEADS:
        return two
    return words[0].lower()


def _clause_args(text: str, start: int, end: int):
    """Base names and absolute offsets of a clause's comma-separated items."""
    out, depth, token_start = [], 0, start
    for i in range(start, end + 1):
        ch = text[i] if i < end else ","
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            match = _IDENT_RE.match(text, token_start)
            if match is not None and match.end() <= i:
                out.append((match.group(1).lower(), match.start(1)))
            token_start = i + 1
    return out


def _clauses(logical: _Logical):
    """Yield ``(clause_name, [(base_name, offset), ...])`` for one directive."""
    text, i = logical.text, 0
    while True:
        match = _CLAUSE_RE.search(text, i)
        if match is None:
            return
        depth, j = 1, match.end()
        while j < len(text) and depth:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        yield match.group(1).lower(), _clause_args(text, match.end(), j - 1)
        i = j


def routine_span(source: str, routine: str):
    """1-based ``(first_line, last_line)`` of ``routine``'s body."""
    lines = source.splitlines()
    name_re = re.compile(r"\b(?:subroutine|function)\s+" + re.escape(routine) + r"\s*[(\s]", re.IGNORECASE)
    start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("#"):
            continue
        if _ROUTINE_END_RE.match(line):
            continue
        if name_re.search(_strip_comment(line)):
            start = i
            break
    if start is None:
        raise ValueError(f"routine {routine!r} not found in source")
    depth = 0
    for i in range(start + 1, len(lines)):
        stripped = lines[i].lstrip()
        if stripped.startswith("!") or stripped.startswith("#"):
            continue
        code = _strip_comment(lines[i])
        if _ROUTINE_END_RE.match(code):
            if depth == 0:
                return start + 1, i + 1
            depth -= 1
        elif _ROUTINE_DECL_RE.match(code):
            depth += 1
    return start + 1, len(lines)


def dummy_args(source: str, routine: str):
    """Dummy argument names of ``routine``, in declaration order, lowercased."""
    lines = source.splitlines()
    start = routine_span(source, routine)[0] - 1
    pieces, _ = _join_continuations(lines, start, None)
    stmt = " ".join(piece for _, piece in pieces)
    match = re.search(r"\b(?:subroutine|function)\s+" + re.escape(routine) + r"\s*\(", stmt, re.IGNORECASE)
    if match is None:
        return []
    depth, j = 1, match.end()
    while j < len(stmt) and depth:
        if stmt[j] == "(":
            depth += 1
        elif stmt[j] == ")":
            depth -= 1
        j += 1
    out = []
    for item in stmt[match.end():j - 1].split(","):
        ident = _IDENT_RE.match(item)
        if ident is not None:
            name = ident.group(1).lower()
            if name not in out:
                out.append(name)
    return out


def collect_evidence(source: str, routine: str):
    """Map ``arg -> [(depth, order, clause, line)]`` from the routine's directives.

    ``depth`` counts enclosing structured data regions; a region's own clauses
    are recorded at the depth they open, so a nested ``!$ACC DATA`` outranks the
    outer one while a compute construct directly inside a region ties with it
    and loses on ``order``.
    """
    first, last = routine_span(source, routine)
    args = set(dummy_args(source, routine))
    evidence, depth, order = {}, 0, 0
    for logical in acc_directives(source):
        if not first <= logical.line <= last:
            continue
        head = _head(logical.text)
        if head in _REGION_CLOSE:
            depth = max(0, depth - 1)
            continue
        if head in _REGION_OPEN:
            depth += 1
        for clause, items in _clauses(logical):
            if clause not in DEVICE_CLAUSES and clause not in HOST_CLAUSES:
                continue
            for name, offset in items:
                if name in args:
                    evidence.setdefault(name, []).append((depth, order, clause, logical.line_at(offset)))
                    order += 1
    return evidence


def classify(source: str, routine: str, source_name: str) -> dict:
    """Build the sidecar payload for ``routine`` in ``source``."""
    args = dummy_args(source, routine)
    evidence = collect_evidence(source, routine)
    classified, unclassified = {}, []
    for arg in args:
        found = evidence.get(arg, [])
        device = [e for e in found if e[2] in DEVICE_CLAUSES]
        host = [e for e in found if e[2] in HOST_CLAUSES]
        pool, residency = (device, "device") if device else (host, "host")
        if not pool:
            unclassified.append(arg)
            continue
        best = max(pool, key=lambda e: (e[0], -e[1]))
        classified[arg] = {
            "residency": residency,
            "clause": best[2].upper(),
            "evidence": f"{source_name}:{best[3]}",
        }
    return {"routine": routine, "source": source_name, "args": classified, "unclassified": unclassified}


def extract_acc_residency(source_path, routine: str) -> dict:
    """Classify ``routine`` in the Fortran file ``source_path``."""
    path = Path(source_path)
    return classify(path.read_text(), routine, path.name)


def write_acc_residency_sidecar(source_path, routine: str, out_dir) -> Path:
    """Write ``<routine>.acc_residency.json`` into ``out_dir``; return its path."""
    payload = extract_acc_residency(source_path, routine)
    out = Path(out_dir) / f"{routine}.acc_residency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dace_fortran.acc_residency",
                                     description="Extract per-argument OpenACC data residency for a Fortran routine.")
    parser.add_argument("source", type=Path, help="Fortran source, ACC directives intact.")
    parser.add_argument("--routine", required=True, help="Target routine name.")
    parser.add_argument("--out", type=Path, help="Write the sidecar JSON here (default: stdout).")
    parser.add_argument("--out-dir", type=Path, help="Write <routine>.acc_residency.json into this directory.")
    parser.add_argument("--table", action="store_true", help="Also print a human-readable table on stderr.")
    ns = parser.parse_args(argv)

    try:
        payload = extract_acc_residency(ns.source, ns.routine)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if ns.table:
        width = max([len(a) for a in list(payload["args"]) + payload["unclassified"]] or [1])
        for arg, info in payload["args"].items():
            print(f"{arg:<{width}}  {info['residency']:<6}  {info['clause']:<12}  {info['evidence']}", file=sys.stderr)
        for arg in payload["unclassified"]:
            print(f"{arg:<{width}}  {'-':<6}  {'unclassified':<12}  -", file=sys.stderr)

    text = json.dumps(payload, indent=2) + "\n"
    if ns.out_dir is not None:
        ns.out_dir.mkdir(parents=True, exist_ok=True)
        (ns.out_dir / f"{ns.routine}.acc_residency.json").write_text(text)
    if ns.out is not None:
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        ns.out.write_text(text)
    if ns.out is None and ns.out_dir is None:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
