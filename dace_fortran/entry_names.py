# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Parsing of the user-facing ``module::proc`` entry-point spelling."""

from __future__ import annotations

from typing import NamedTuple


class QualifiedEntry(NamedTuple):
    module: str | None
    proc: str


def split_qualified_entry(entry: str) -> QualifiedEntry:
    """Split ``module::proc`` into ``(module, proc)``, both lower-cased (flang lower-cases identifiers).

    A bare ``proc`` leaves the module unconstrained (``None``).
    """
    module, _, proc = entry.lower().rpartition("::")
    return QualifiedEntry((module or None), proc)
