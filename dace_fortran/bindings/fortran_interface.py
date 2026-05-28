"""Outer Fortran interface  --  the caller-facing surface of the entry
subroutine, snapshotted from HLFIR BEFORE any normalising pass
(``hlfir-flatten-structs`` in particular) runs.

This is **auto-derived by default**.  ``SDFGBuilder.build`` calls the
bridge's ``HLFIRModule.get_fortran_interface(entry)`` -- which walks the
entry function's block arguments IN ORDER on the untransformed module --
and stashes the result on ``sdfg._fortran_interface_raw``.
``build_fortran_library`` turns it into an ``OriginalInterface`` via
``build_auto_interface`` whenever the caller does not pass one, so a
normal build needs no hand-written interface.

Per dummy the snapshot carries name / element dtype / rank / shape /
intent and, for a derived-type dummy, the type name + defining module
(recovered from the mangled ``_QM<mod>T<tname>``).  Member *layouts* are
deliberately NOT extracted: the emitter gets per-member accesses from the
``FlattenPlan``, so the interface only needs each dummy's outer surface
plus the ``use <mod>, only: ...`` list.  No fparser dependency  --  HLFIR's
types carry all of it.

A hand-written ``OriginalInterface`` is only needed for a dummy shape the
snapshot can't name (e.g. ``CHARACTER``); ``build_auto_interface`` raises
``unsupported dtype`` in that case so the caller knows to supply one.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# SDFG element dtype (as the bridge reports it) -> the ``iso_c_binding``
# Fortran type the wrapper declares for that dummy.  ``bool`` is the
# uniform image for any ``LOGICAL(KIND)`` (the logical-bridge converts the
# caller's kind width at the boundary); integer kinds map width-for-width.
# No unsigned entries: Fortran < 2023 has no UNSIGNED type and flang lowers
# everything to signless integers, so an unsigned dtype never reaches here.
_DTYPE_TO_FORTRAN_C = {
    "complex128": "complex(c_double)",
    "complex64": "complex(c_float)",
    "float64": "real(c_double)",
    "float32": "real(c_float)",
    "int8": "integer(c_int8_t)",
    "int16": "integer(c_int16_t)",
    "int32": "integer(c_int)",
    "int64": "integer(c_int64_t)",
    "bool": "logical(c_bool)",
}


@dataclass(frozen=True)
class Member:
    """One field of a Fortran derived type."""
    name: str  # 'u'
    fortran_type: str  # 'real(c_double)' | 'complex(c_double)' | 'integer(c_int)'
    rank: int
    # Symbolic / literal extents as they appear in the struct declaration.
    # For assumed-shape inside structs we fall back to '?' and let the
    # wrapper use ``size(st%u, dim=d)`` at call time.
    shape: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DerivedType:
    """Layout of one Fortran derived type referenced by the entry."""
    name: str  # 't_state'
    module: Optional[str]  # 'mo_state' if defined in a module
    members: Tuple[Member, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class OriginalArg:
    """One dummy argument of the entry subroutine, outer view."""
    name: str  # 'st'  --  Fortran-source name
    fortran_type: str  # 'real(c_double)' / 'type(t_state)' / 'logical' / ...
    rank: int
    shape: Tuple[str, ...] = field(default_factory=tuple)
    intent: str = ''  # 'in' | 'out' | 'inout' | ''
    # When fortran_type == 'type(<name>)', this points at the
    # DerivedType entry in ``OriginalInterface.struct_types``.
    struct_type: Optional[str] = None


@dataclass(frozen=True)
class OriginalInterface:
    """Caller-facing surface of the entry subroutine plus every
    derived type referenced by its dummies (transitively)."""
    entry: str  # 'compute_tendencies'
    args: Tuple[OriginalArg, ...]
    struct_types: Dict[str, DerivedType] = field(default_factory=dict)
    # Modules the wrapper needs to ``use <mod>, only: <syms>`` so the
    # derived types resolve when gfortran compiles the binding.
    used_modules: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Free SDFG symbols that the kernel reads from Fortran *module*
    # data (e.g. ICON's ``mo_parallel_config::nproma``) rather than
    # from a dummy argument.  The flatten plan can never supply these
    # via ``size(...)`` -- the binding has no dummy to query.  Maps
    # ``sym -> (module, member)``; the emitter renames the import to
    # ``<sym>__mod => <member>`` (the wrapper declares its own local
    # ``<sym>`` for the by-value SDFG call) and assigns
    # ``<sym> = int(<sym>__mod, c_int)`` in the symbol-population
    # block.  Default-empty: a no-op for flat kernels.
    module_symbol_sources: Dict[str, Tuple[str, str]] = field(default_factory=dict)


def build_auto_interface(raw: dict, entry: str) -> OriginalInterface:
    """Build an :class:`OriginalInterface` from the bridge's
    ``HLFIRModule.get_fortran_interface(entry)`` snapshot (taken before
    ``hlfir-flatten-structs``).  ``entry`` should be the final ``sdfg.name``
    so the wrapper's ``bind(c)`` symbols match the compiled SDFG exports.

    Derived-type dummies pick up their per-member layout from the
    ``struct_types`` sub-dict the bridge populates from each dummy's
    ``fir::RecordType``.  Members whose element dtype the bridge could
    not name (a nested record, ``allocatable`` / ``pointer`` /
    ``character``, complex) carry an empty ``dtype`` -- ``Member`` keeps
    the slot with a placeholder ``fortran_type`` (``'??'``) so the
    downstream ``bind_c_shim`` emitter can reject only the unsupported
    members and accept inline-flat ones from the same struct.

    :raises ValueError: a dummy uses a dtype the binding layer can't name
        (e.g. ``CHARACTER``) or a derived-type arg whose type name the bridge
        could not recover -- the caller then supplies an explicit interface.
    """
    args = []
    for a in raw["args"]:
        if a["is_struct"]:
            if not a["struct_name"]:
                raise ValueError(f"auto-iface: cannot name derived-type arg {a['name']!r}")
            fortran_type = f"type({a['struct_name']})"
            struct_type = a["struct_name"]
        else:
            fortran_type = _DTYPE_TO_FORTRAN_C.get(a["dtype"])
            if fortran_type is None:
                raise ValueError(f"auto-iface: unsupported dtype {a['dtype']!r} "
                                 f"for argument {a['name']!r}")
            struct_type = None
        args.append(OriginalArg(name=a["name"], fortran_type=fortran_type,
                                rank=int(a["rank"]), shape=tuple(a["shape"]),
                                intent=a["intent"], struct_type=struct_type))
    used_modules = {mod: tuple(syms) for mod, syms in raw["used_modules"].items()}
    struct_types = {}
    for sname, st in raw.get("struct_types", {}).items():
        members = []
        for m in st["members"]:
            members.append(Member(
                name=m["name"],
                fortran_type=_DTYPE_TO_FORTRAN_C.get(m["dtype"], "??"),
                rank=int(m["rank"]),
                shape=tuple(m["shape"]),
            ))
        struct_types[sname] = DerivedType(name=st["name"],
                                          module=st["module"] or None,
                                          members=tuple(members))
    return OriginalInterface(entry=entry, args=tuple(args),
                             struct_types=struct_types,
                             used_modules=used_modules)
