# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-argument OpenACC data residency for a Fortran routine.

The HLFIR parse path never sees ``!$ACC``: :func:`dace_fortran.preprocess.
strip_openmp_directives` deletes the sentinels before flang runs, and flang is
invoked without ``-fopenacc`` (``build.py``, ``emit_hlfir.py``,
``flang_codebase.py`` all pass ``-U_OPENACC``), so the directives are plain
comments and comments never enter a flang parse tree.  Residency is therefore
recovered by a pre-pass over the ORIGINAL source, before preprocessing.

The routine structure (span, dummy arguments) comes from a real fparser AST --
a light line-preserving cpp arm selection (``_OPENACC`` defined, matching the
GPU build this pass models) makes raw ICON sources parseable -- and the
directive text itself (a comment even to fparser) is tokenised by a small
clause/entity parser that understands component refs (``p_diag%vt``), array
sections (``v(:,:,jb)``), continuation lines and multiple clauses per
directive.

The pre-pass answers, for every dummy argument of a target routine, whether the
data is device- or host-resident at call time, and emits the sidecar
``<routine>.acc_residency.json`` consumed by the ICON binding generator::

    {"routine": ..., "source": ...,
     "args": {"<arg>": {"residency": "device"|"host",
                        "clause": "PRESENT", "evidence": "<file>:<line>",
                        "ref": "<full entity ref, e.g. p_diag%vt(:,:,jb)>"}},
     "unclassified": ["<arg>", ...]}

``ref`` is additive over the original schema: entities are normalised to their
BASE variable name for the ``args`` keys (what the binding interface matches),
while the full reference string of the deciding clause item is retained.

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
from typing import Dict, Iterable, List, Optional, Tuple

from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003 as f03
from fparser.two.parser import ParserFactory
from fparser.two.utils import FortranSyntaxError, walk

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

#: cpp macros assumed defined when selecting preprocessor arms.  ``_OPENACC``
#: models the GPU build whose residency this pass reconstructs.
DEFAULT_CPP_DEFINES = frozenset({"_OPENACC"})

_ACC_SENTINEL_RE = re.compile(r"^\s*!\s*\$\s*acc\b", re.IGNORECASE)
_IDENT_RE = re.compile(r"\s*([A-Za-z_]\w*)")
_CLAUSE_RE = re.compile(r"([A-Za-z_]\w*)\s*\(")

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


# ---------------------------------------------------------------------------
# cpp arm selection (line-count preserving) + fparser parse
# ---------------------------------------------------------------------------


def _cpp_expr_true(expr: str, defines: frozenset) -> bool:
    """Best-effort truth of a ``#if`` expression: ``defined(X)`` resolves against
    ``defines``, unknown identifiers count as 0, unparseable means False."""
    e = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in defines else "0", expr)
    e = re.sub(r"defined\s+(\w+)", lambda m: "1" if m.group(1) in defines else "0", e)
    e = re.sub(r"\b[A-Za-z_]\w*\b", "0", e)
    e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    try:
        return bool(eval(e))  # arithmetic/logic over literals only after the rewrites
    except Exception:
        return False


def _mask_cpp(source: str, defines: frozenset) -> str:
    """Comment out cpp directive lines and every line of a not-taken arm,
    preserving the line count so fparser spans keep pointing at the original."""
    out: List[str] = []
    stack: List[List[bool]] = []  # [taken_now, taken_ever] per open #if
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            directive = stripped[1:].lstrip()
            word = re.match(r"\w*", directive).group(0)
            rest = directive[len(word):].strip()
            if word == "ifdef":
                taken = rest.split()[0] in defines if rest.split() else False
                stack.append([taken, taken])
            elif word == "ifndef":
                taken = rest.split()[0] not in defines if rest.split() else False
                stack.append([taken, taken])
            elif word == "if":
                taken = _cpp_expr_true(rest, defines)
                stack.append([taken, taken])
            elif word == "elif" and stack:
                taken = (not stack[-1][1]) and _cpp_expr_true(rest, defines)
                stack[-1] = [taken, stack[-1][1] or taken]
            elif word == "else" and stack:
                taken = not stack[-1][1]
                stack[-1] = [taken, True]
            elif word == "endif" and stack:
                stack.pop()
            out.append("!" + line)
            continue
        out.append(line if all(f[0] for f in stack) else "!" + line)
    return "\n".join(out)


_PARSER = None


def _parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = ParserFactory().create(std="f2008")
    return _PARSER


def _parse(source: str, defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> f03.Program:
    """fparser AST of ``source`` (comments retained), after cpp arm selection."""
    masked = _mask_cpp(source, frozenset(defines))
    try:
        return _parser()(FortranStringReader(masked, ignore_comments=False))
    except FortranSyntaxError as exc:
        raise ValueError(f"fparser could not parse the source: {exc}") from exc


_SCOPE_CLASSES = (f03.Subroutine_Subprogram, f03.Function_Subprogram)
_SCOPE_STMT_CLASSES = (f03.Subroutine_Stmt, f03.Function_Stmt)


def _routine_node(ast: f03.Program, routine: str):
    """The subprogram node defining ``routine`` (first match, like the old scan)."""
    want = routine.lower()
    for scope in walk(ast, _SCOPE_CLASSES):
        stmt = next(iter(walk(scope, _SCOPE_STMT_CLASSES)), None)
        if stmt is not None and str(stmt.children[1]).lower() == want:
            return scope
    raise ValueError(f"routine {routine!r} not found in source")


def _node_span(node) -> Tuple[int, int]:
    """1-based (first, last) source line of ``node``'s subtree."""
    spans = [n.item.span for n in walk(node) if getattr(n, "item", None) is not None]
    return min(s[0] for s in spans), max(s[1] for s in spans)


# ---------------------------------------------------------------------------
# Directive extraction (continuation-joined) + clause/entity tokenizer
# ---------------------------------------------------------------------------


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


def _sentinel_body(line: str) -> Tuple[str, bool]:
    """The directive text of one physical ``!$acc`` line, plus its continues-flag."""
    match = _ACC_SENTINEL_RE.match(line)
    body = _strip_comment(line[match.end():]).rstrip()
    continued = body.endswith("&")
    if continued:
        body = body[:-1]
    body = body.strip()
    if body.startswith("&"):
        body = body[1:].strip()
    return body, continued


def acc_directives(source: str) -> List[_Logical]:
    """Yield every ``!$ACC`` directive of ``source`` as a :class:`_Logical`."""
    lines = source.splitlines()
    out, i = [], 0
    while i < len(lines):
        if not _ACC_SENTINEL_RE.match(lines[i]):
            i += 1
            continue
        pieces = []
        body, continued = _sentinel_body(lines[i])
        pieces.append((i + 1, body))
        while continued and i + 1 < len(lines) and _ACC_SENTINEL_RE.match(lines[i + 1]):
            i += 1
            body, continued = _sentinel_body(lines[i])
            pieces.append((i + 1, body))
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


def _clause_entities(text: str, start: int, end: int):
    """``(base_name, full_ref, offset)`` per comma-separated clause item.

    An item is a variable reference: plain name, component path (``a%b%c``),
    or either with array-section subscripts (``v(:, :, jb)``); the base name
    is the leading identifier, the full ref is the item with whitespace
    removed.  Items that do not start with an identifier (e.g. a ``*``) are
    skipped.
    """
    out, depth, token_start = [], 0, start
    for i in range(start, end + 1):
        ch = text[i] if i < end else ","
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            item = text[token_start:i]
            match = _IDENT_RE.match(item)
            if match is not None:
                ref = re.sub(r"\s+", "", item)
                out.append((match.group(1).lower(), ref, token_start + match.start(1)))
            token_start = i + 1
    return out


def _clauses(logical: _Logical):
    """Yield ``(clause_name, [(base, ref, offset), ...])`` for one directive."""
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
        yield match.group(1).lower(), _clause_entities(text, match.end(), j - 1)
        i = j


# ---------------------------------------------------------------------------
# Public API (unchanged signatures + optional cpp ``defines``)
# ---------------------------------------------------------------------------


def routine_span(source: str, routine: str, defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> Tuple[int, int]:
    """1-based ``(first_line, last_line)`` of ``routine``'s definition."""
    return _node_span(_routine_node(_parse(source, defines), routine))


def dummy_args(source: str, routine: str, defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> List[str]:
    """Dummy argument names of ``routine``, in declaration order, lowercased."""
    node = _routine_node(_parse(source, defines), routine)
    stmt = next(iter(walk(node, _SCOPE_STMT_CLASSES)))
    args = next(iter(walk(stmt, f03.Dummy_Arg_List)), None)
    if args is None:
        return []
    out: List[str] = []
    for name in walk(args, f03.Name):
        low = name.string.lower()
        if low not in out:
            out.append(low)
    return out


def _collect_evidence_impl(source: str, routine: str, defines: Iterable[str]) -> Dict[str, list]:
    ast = _parse(source, defines)
    node = _routine_node(ast, routine)
    first, last = _node_span(node)
    stmt = next(iter(walk(node, _SCOPE_STMT_CLASSES)))
    args = set()
    arg_list = next(iter(walk(stmt, f03.Dummy_Arg_List)), None)
    if arg_list is not None:
        args = {n.string.lower() for n in walk(arg_list, f03.Name)}
    # Directive scan runs over the cpp-masked text so only live-arm directives
    # count, matching the parsed structure the spans came from.
    masked = _mask_cpp(source, frozenset(defines))
    evidence: Dict[str, list] = {}
    depth, order = 0, 0
    for logical in acc_directives(masked):
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
            for base, ref, offset in items:
                if base in args:
                    evidence.setdefault(base, []).append((depth, order, clause, logical.line_at(offset), ref))
                    order += 1
    return evidence


def collect_evidence(source: str,
                     routine: str,
                     defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> Dict[str, list]:
    """Map ``arg -> [(depth, order, clause, line, ref)]`` from the routine's
    directives.

    ``depth`` counts enclosing structured data regions; a region's own clauses
    are recorded at the depth they open, so a nested ``!$ACC DATA`` outranks the
    outer one while a compute construct directly inside a region ties with it
    and loses on ``order``.  ``ref`` is the full entity reference the clause
    named (``p_diag%vt(:,:,jb)``), whitespace-normalised; ``arg`` is its base.
    """
    return _collect_evidence_impl(source, routine, defines)


def classify(source: str, routine: str, source_name: str, defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> dict:
    """Build the sidecar payload for ``routine`` in ``source``."""
    args = dummy_args(source, routine, defines)
    evidence = collect_evidence(source, routine, defines)
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
            "ref": best[4],
        }
    return {"routine": routine, "source": source_name, "args": classified, "unclassified": unclassified}


def extract_acc_residency(source_path, routine: str, defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> dict:
    """Classify ``routine`` in the Fortran file ``source_path``."""
    path = Path(source_path)
    return classify(path.read_text(), routine, path.name, defines)


def write_acc_residency_sidecar(source_path, routine: str, out_dir, defines: Iterable[str] = DEFAULT_CPP_DEFINES) -> Path:
    """Write ``<routine>.acc_residency.json`` into ``out_dir``; return its path."""
    payload = extract_acc_residency(source_path, routine, defines)
    out = Path(out_dir) / f"{routine}.acc_residency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dace_fortran.acc_residency",
                                     description="Extract per-argument OpenACC data residency for a Fortran routine.")
    parser.add_argument("source", type=Path, help="Fortran source, ACC directives intact.")
    parser.add_argument("--routine", required=True, help="Target routine name.")
    parser.add_argument("--define",
                        action="append",
                        default=[],
                        help="Extra cpp macro assumed defined when selecting #if arms "
                        "(added to the default set: %s)." % ", ".join(sorted(DEFAULT_CPP_DEFINES)))
    parser.add_argument("--out", type=Path, help="Write the sidecar JSON here (default: stdout).")
    parser.add_argument("--out-dir", type=Path, help="Write <routine>.acc_residency.json into this directory.")
    parser.add_argument("--table", action="store_true", help="Also print a human-readable table on stderr.")
    ns = parser.parse_args(argv)

    defines = DEFAULT_CPP_DEFINES | set(ns.define)
    try:
        payload = extract_acc_residency(ns.source, ns.routine, defines)
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
