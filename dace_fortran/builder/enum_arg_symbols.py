"""Translate string arguments to integer enum values at SDFG call time.

Companion to ``rewrite_string_enum_to_integer`` (preprocess.py).  When
the preprocess pass converts a ``CHARACTER(LEN=*), INTENT(IN) :: var``
dummy that's used purely as an enum switch into ``INTEGER, INTENT(IN)
:: var``, it also returns a ``{procedure: {arg: {literal: int}}}``
mapping recording which string literal corresponds to which integer.
The Python side of the bridge still wants to accept the original
string at the call boundary: a caller writing ``sdfg(flag='c', ...)``
should get the same behaviour as ``sdfg(flag=0, ...)`` since under
the hood the kernel's body is integer-comparing.

This module installs a thin ``SDFG`` subclass mixin whose ``__call__``
walks every passed ``kwargs`` entry: if the kwarg's name is in the
enum-translation table and the value is a string, the string is
case-folded, looked up, and substituted by the integer.  Unknown
strings flow through unchanged so the SDFG either takes the int-
literal path the caller chose explicitly or surfaces a clean
"unsupported value" failure.

The mixin composes cleanly with ``_AutoDimSDFG`` (and any future
class-assignment wrapper): when ``install_enum_arg_translation`` is
called on an already-wrapped SDFG it inserts itself into the MRO
*above* whatever class the SDFG carries, so its ``super().__call__``
chains through the existing wrapper before reaching the underlying
``dace.SDFG``.

Typical lifecycle:

    rewritten, enum_maps = rewrite_string_enum_to_integer(source)
    sdfg = build_sdfg(rewritten, ...)
    sdfg = install_enum_arg_translation(sdfg, enum_maps)
    # Now both forms work:
    sdfg(flag='c', ...)     # -> flag=0
    sdfg(flag='C', ...)     # -> flag=0 (case-insensitive)
    sdfg(flag=0, ...)       # -> flag=0 (passthrough)
"""
import dace


class _EnumArgMixin(dace.SDFG):
    """Mixin that case-folds + integer-translates string kwargs whose
    names appear in ``self._enum_arg_maps``.

    Inherits from ``dace.SDFG`` so ``__class__`` assignment stays
    layout-compatible with the SDFG instance (Python's C-level
    ``__class__`` check requires the source and destination classes
    to share their underlying object layout, which is the case for
    any subclass of ``dace.SDFG``).

    Composes with the bridge's other class-assignment wrappers (most
    notably ``_AutoDimSDFG``) via cooperative ``super().__call__``:
    the runtime-synthesised wrapper class places the mixin ABOVE the
    current class in the MRO, so the mixin's ``super().__call__``
    chains into the existing wrapper before reaching ``dace.SDFG``.
    """

    def __call__(self, *args, **kwargs):
        emaps = getattr(self, "_enum_arg_maps", None)
        if emaps:
            for arg_name, lit_to_int in emaps.items():
                if arg_name not in kwargs:
                    continue
                v = kwargs[arg_name]
                if not isinstance(v, (str, bytes)):
                    continue
                key = (v.decode("ascii") if isinstance(v, bytes) else v).lower()
                if key in lit_to_int:
                    kwargs[arg_name] = lit_to_int[key]
        return super().__call__(*args, **kwargs)


def install_enum_arg_translation(sdfg: dace.SDFG, enum_maps: dict) -> dace.SDFG:
    """Rebind ``sdfg`` so direct calls auto-translate string arguments
    for the enum-mapped dummies recorded in ``enum_maps``.

    ``enum_maps`` is the dict ``rewrite_string_enum_to_integer``
    returns: ``{procedure_name: {arg_name: {literal_lower: int}}}``.
    Since the resulting SDFG is one entry, the per-procedure layer is
    flattened into a single ``{arg_name: {literal: int}}`` table on
    the SDFG.  Cross-procedure name collisions (the same arg name
    appearing under two procedures with different enum tables) keep
    the FIRST procedure's table -- a defensive choice; in practice
    the bridge surfaces one entry per SDFG so the collision is rare.

    No-op when ``enum_maps`` is empty: the SDFG flows through
    unmodified, no class assignment, no extra ``__call__`` overhead.

    :param sdfg: the freshly built SDFG (typically post-
        ``install_auto_dim_symbols``; the mixin composes via
        cooperative ``super().__call__``).
    :param enum_maps: the per-procedure enum tables; ``{}`` accepted.
    :returns: the same ``sdfg`` instance, re-classed with the mixin
        inserted into the MRO above its current class.
    """
    if not enum_maps:
        return sdfg

    # Flatten {proc: {arg: lit_map}} to {arg: lit_map}, FIRST-wins on
    # collisions.  ``setdefault`` keeps the earlier entry; iterate
    # procedures in dict-insertion order (Python 3.7+ guarantee).
    flat: dict = {}
    for proc_table in enum_maps.values():
        for arg, lit_map in proc_table.items():
            # Keep literals lowercased so the runtime lookup is
            # case-insensitive  --  matches the lower-grouping the
            # preprocess pass already applied.
            flat.setdefault(arg, {k.lower(): int(v) for k, v in lit_map.items()})

    # Synthesize a class with the mixin inserted ABOVE whatever class
    # the SDFG currently carries.  ``type(name, bases, namespace)``
    # builds a fresh class whose MRO is (mixin, current_class, ...,
    # object); a subsequent ``__class__`` assignment moves the
    # instance to the new class without re-running ``__init__``.
    base = type(sdfg)
    enum_class = type(f"_EnumArgSDFG_over_{base.__name__}", (_EnumArgMixin, base), {})
    sdfg.__class__ = enum_class
    sdfg._enum_arg_maps = flat
    return sdfg
