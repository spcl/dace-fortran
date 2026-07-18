# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parse-time declaration of how a non-inlined Fortran procedure is handled: emitted as an external call, or dropped.

Stdlib-only (no dace imports) -- fparser inliner needs this without pulling in dace_fortran.external.
Invariant: do_not_emit is a subset of don't-inline; dont_inline_names() is the single source of truth.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Set

__all__ = ["ExternalFunction", "dont_inline_names", "validate"]


@dataclass(frozen=True)
class ExternalFunction:
    """A procedure that is NOT inlined and IS emitted as an external call.

    library=None leaves the symbol unresolved at link time (fine for extract-only/compile-check;
    the SDFG side supplies a real library for an executable run).
    """

    name: str
    c_function: Optional[str] = None
    library: Optional[str] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("ExternalFunction.name must be a non-empty procedure name")

    @property
    def symbol(self) -> str:
        """extern "C" symbol the call resolves to: c_function if set, else name."""
        return self.c_function or self.name


def dont_inline_names(external_functions: Iterable[ExternalFunction] = (), do_not_emit: Iterable[str] = ()) -> Set[str]:
    """Lower-cased union of names that must NOT be inlined: emitted external-function names plus do_not_emit.

    Single set the inliner consumes; lower-cased to match its case-insensitive Fortran matching.
    """
    names = {c.name.lower() for c in external_functions}
    names |= {n.lower() for n in do_not_emit}
    return names


def validate(external_functions: Iterable[ExternalFunction] = (), do_not_emit: Iterable[str] = ()) -> None:
    """Raise ValueError if a name is both emitted and ignored, or duplicated across emitted specs."""
    emit_names = [c.name.lower() for c in external_functions]
    dupes = sorted({n for n in emit_names if emit_names.count(n) > 1})
    if dupes:
        raise ValueError(f"external_functions has duplicate name(s): {dupes}")
    ignore = {n.lower() for n in do_not_emit}
    both = sorted(set(emit_names) & ignore)
    if both:
        raise ValueError(f"name(s) in both external_functions and do_not_emit "
                         f"(emit and ignore are mutually exclusive): {both}")
