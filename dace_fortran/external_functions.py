# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Parse-time declaration of how a non-inlined Fortran procedure is handled.

One small, dependency-free dataclass (:class:`ExternalFunction`) plus two
helpers express the whole "external function policy" the inliner and the bridge
share.  A caller declares, once per target, two collections:

* ``external_functions`` -- :class:`ExternalFunction` specs for procedures that
  are NOT inlined and ARE emitted as an external reference.  The bridge lowers
  each ``CALL`` to a :class:`dace_fortran.external.ExternalCall` library node,
  replicating the HLFIR call order and binding it to the supplied C-ABI symbol
  / library.
* ``do_not_emit`` -- plain names of procedures that are NOT inlined and NOT
  emitted (their call sites are dropped): timers, loggers, ``finish`` ...

The two behaviours share one structural invariant -- *ignore is a subset of
don't-inline* -- so :func:`dont_inline_names` is the single source of the
don't-inline set the inliner consumes.

This module is intentionally pure-stdlib (no ``dace`` / no bridge imports): the
fparser inliner imports it, and the inliner must stay free of the dace-heavy
:mod:`dace_fortran.external`.  The argument order/types of an emitted call are
NOT authored here -- HLFIR already carries the ``CALL f(a, b, c)``; an
:class:`ExternalFunction` only supplies what HLFIR cannot know (the
``extern "C"`` symbol and the library that exports it).

.. note::
   Naming: this :class:`ExternalFunction` is the user-facing, *parse-time
   declaration* of an external procedure.  It is deliberately distinct from
   :class:`dace_fortran.external.ExternalCall` (the lowered, code-gen-time SDFG
   *library node*) and from :class:`dace_fortran.external.ExternalSignature`
   (the bridge's full bind(c) ABI record) -- this spec *drives* their emission
   but carries only the minimum the user must supply.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Set

__all__ = ["ExternalFunction", "dont_inline_names", "validate"]


@dataclass(frozen=True)
class ExternalFunction:
    """A procedure that is NOT inlined and IS emitted as an external call.

    :ivar name: the Fortran call-site name, exactly as it appears at the
        ``CALL name(...)`` site.  Any name -- a plain procedure, or the generic
        an interface dispatches over (``sync_patch_array``); matching the
        generic to its concrete specifics (``sync_patch_array_3d_dp`` ...) is
        the inliner's job (see :func:`dont_inline_names`).
    :ivar c_function: the ``bind(c, name=...)`` / ``extern "C"`` symbol the
        emitted call invokes.  Defaults to :ivar:`name` (via :attr:`symbol`)
        when omitted -- the common case where the Fortran name is already the
        C symbol.
    :ivar library: path to the shared library (``.so``) that exports
        :attr:`symbol`.  ``None`` leaves the symbol unresolved at link time --
        fine for an extract-only / compile-check flow; the SDFG side supplies a
        real library when an executable run is built.
    """

    name: str
    c_function: Optional[str] = None
    library: Optional[str] = None

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("ExternalFunction.name must be a non-empty procedure name")

    @property
    def symbol(self) -> str:
        """The ``extern "C"`` symbol the call resolves to -- :ivar:`c_function`
        when given, else :ivar:`name`."""
        return self.c_function or self.name


def dont_inline_names(external_functions: Iterable[ExternalFunction] = (), do_not_emit: Iterable[str] = ()) -> Set[str]:
    """The (lower-cased) union of names that must NOT be inlined: every emitted
    external function's :attr:`ExternalFunction.name` plus every ``do_not_emit``
    name.

    This is the single set the inliner consumes; it never reads
    ``c_function`` / ``library`` (those are the bridge's concern).  Names are
    lower-cased to match the inliner's case-insensitive Fortran matching."""
    names = {c.name.lower() for c in external_functions}
    names |= {n.lower() for n in do_not_emit}
    return names


def validate(external_functions: Iterable[ExternalFunction] = (), do_not_emit: Iterable[str] = ()) -> None:
    """Reject an inconsistent policy: a name that is both emitted and ignored
    (ambiguous intent), or a duplicate ``name`` across the emitted specs.

    :raises ValueError: on either inconsistency.
    """
    emit_names = [c.name.lower() for c in external_functions]
    dupes = sorted({n for n in emit_names if emit_names.count(n) > 1})
    if dupes:
        raise ValueError(f"external_functions has duplicate name(s): {dupes}")
    ignore = {n.lower() for n in do_not_emit}
    both = sorted(set(emit_names) & ignore)
    if both:
        raise ValueError(f"name(s) in both external_functions and do_not_emit "
                         f"(emit and ignore are mutually exclusive): {both}")
