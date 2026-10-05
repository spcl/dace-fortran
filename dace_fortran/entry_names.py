# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parsing of the user-facing entry-point spelling: the Fortran name ``proc`` or ``module::proc``."""

from __future__ import annotations

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
