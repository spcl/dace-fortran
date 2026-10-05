# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The user-facing entry-point spelling (the Fortran name ``proc`` or ``module::proc``) and its resolution against
Fortran sources."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple


class QualifiedEntry(NamedTuple):
    module: str | None
    proc: str

    def __str__(self) -> str:
        return f"{self.module}::{self.proc}" if self.module else self.proc


def require_fortran_name(entry: str) -> str:
    """``entry`` itself, unless it is a Flang-mangled symbol (``_QM<mod>P<proc>``), which the API does not take.

    :raises ValueError: for a mangled symbol.
    """
    if entry.startswith("_Q"):
        raise ValueError(
            f"entry {entry!r} is a Flang-mangled symbol; name the procedure as in Fortran: 'proc' or 'module::proc'"
        )
    return entry


def split_qualified_entry(entry: str) -> QualifiedEntry:
    """Split ``module::proc`` into ``(module, proc)``, both lower-cased (flang lower-cases identifiers).

    A bare ``proc`` leaves the module unconstrained (``None``).

    :raises ValueError: for a Flang-mangled symbol (see :func:`require_fortran_name`).
    """
    module, _, proc = require_fortran_name(entry).lower().rpartition("::")
    return QualifiedEntry((module or None), proc)


_PROC_RE = re.compile(
    r"^\s*(?:(?:recursive|pure|impure|elemental|module)\s+)*(?:[\w*()]+\s+)*?(subroutine|function)\s+(\w+)",
    re.IGNORECASE,
)
_MOD_RE = re.compile(r"^\s*module\s+(\w+)\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*end\s*(module|interface|subroutine|function)?\b", re.IGNORECASE)
_IFACE_RE = re.compile(r"^\s*(?:abstract\s+)?interface\b", re.IGNORECASE)


def procedure_definitions(text: str) -> list[QualifiedEntry]:
    """The SUBROUTINE / FUNCTION *definitions* in Fortran source ``text``, lower-cased, with their enclosing module
    (``interface`` blocks, ``end`` lines and comments are skipped)."""
    module: str | None = None
    iface_depth = 0
    procs: list[QualifiedEntry] = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0]
        if _IFACE_RE.match(line):
            iface_depth += 1
            continue
        if m_end := _END_RE.match(line):
            kind = (m_end.group(1) or "").lower()
            if kind == "interface" and iface_depth:
                iface_depth -= 1
            elif kind == "module":
                module = None
            continue
        if m_mod := _MOD_RE.match(line):
            module = m_mod.group(1).lower()
            continue
        if not iface_depth and (m_proc := _PROC_RE.match(line)):
            procs.append(QualifiedEntry(module, m_proc.group(2).lower()))
    return procs


def resolve_entry_name(sources: Iterable[str], entry: str | None) -> QualifiedEntry:
    """The procedure ``entry`` names among the definitions in the Fortran source texts ``sources``, with its module.

    ``entry`` is a Fortran name (``solve_nh`` or ``mod::proc``), or ``None`` to take the single procedure the sources
    define (an SDFG targets one specific procedure -- no "first of many" guessing).

    :raises ValueError: if the named procedure is not found or is ambiguous, or (``None`` case) the sources define no
        or more than one procedure.
    """
    procs = [proc for text in sources for proc in procedure_definitions(text)]
    if entry is None:
        if not procs:
            raise ValueError("no SUBROUTINE/FUNCTION definition found to use as the SDFG entry; pass entry= explicitly")
        if len(procs) > 1:
            shown = ", ".join(p.proc for p in procs)
            raise ValueError(
                f"the source defines multiple procedures ({shown}); pass entry= (the Fortran name of the target one)"
            )
        return procs[0]
    want = split_qualified_entry(entry)
    matches = {p for p in procs if p.proc == want.proc and want.module in (None, p.module)}
    if not matches:
        raise ValueError(f"no procedure {entry!r} defined in the sources")
    if len(matches) > 1:
        shown = ", ".join(f"{p.module or '<free>'}::{p.proc}" for p in sorted(matches, key=str))
        raise ValueError(f"entry {entry!r} is ambiguous ({shown}); qualify it as module::proc")
    return matches.pop()
