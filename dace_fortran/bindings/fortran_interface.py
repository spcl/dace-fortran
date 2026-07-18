"""Outer Fortran interface -- caller-facing surface of the entry
subroutine, snapshotted from HLFIR BEFORE any normalising pass
(``hlfir-flatten-structs``) runs.

Auto-derived by default: ``SDFGBuilder.build`` walks the entry's block
args via the bridge's ``HLFIRModule.get_fortran_interface(entry)`` and
stashes it on ``sdfg._fortran_interface_raw``; ``build_auto_interface``
turns it into an ``OriginalInterface`` when the caller passes none.

Member layouts are NOT extracted here -- the emitter gets per-member
accesses from the ``FlattenPlan``.  A hand-written ``OriginalInterface``
is only needed for a dummy shape the snapshot can't name (e.g.
``CHARACTER``); ``build_auto_interface`` then raises ``unsupported dtype``.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# SDFG dtype -> iso_c_binding Fortran type.  bool is the uniform image for
# any LOGICAL(KIND) (bridge converts kind width at the boundary).  No
# unsigned entries -- Fortran <2023 has none; flang lowers to signless ints.
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
    # Extents as declared; assumed-shape falls back to '?' (wrapper uses
    # size(st%u, dim=d) at call time).
    shape: Tuple[str, ...] = field(default_factory=tuple)
    # Nested-derived-type member: names the type so bind_c_shim can look
    # up its layout in OriginalInterface.struct_types and recurse.  Empty
    # for scalar/box-of-array/inline-flat members.
    struct_name: Optional[str] = None
    # 'allocatable' | 'pointer' | ''.  Unallocated/disassociated bounds are
    # undefined, so every marshal of this member must be presence-guarded --
    # gfortran's internal_pack at an unguarded site reads the garbage
    # descriptor and smashes the stack (ICON Held-Suarez: disassociated
    # t_nh_diag%ddt_ua_* pointers).
    alloc: str = ''


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
    # OPTIONAL dummy: wrapper forwards present(<name>) into the kernel's
    # <name>_present symbol, rather than defaulting it absent.
    optional: bool = False
    # fortran_type == 'type(<name>)' -> points at OriginalInterface.struct_types.
    struct_type: Optional[str] = None


@dataclass(frozen=True)
class OriginalInterface:
    """Caller-facing surface of the entry subroutine plus every
    derived type referenced by its dummies (transitively)."""
    entry: str  # 'compute_tendencies'
    args: Tuple[OriginalArg, ...]
    struct_types: Dict[str, DerivedType] = field(default_factory=dict)
    # Modules to `use <mod>, only: <syms>` so derived types resolve at compile time.
    used_modules: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Free SDFG symbols the kernel reads from Fortran module data (e.g.
    # ICON's mo_parallel_config::nproma) rather than a dummy arg.  Maps
    # sym -> (module, member); emitter imports as <sym>__mod => <member>
    # and assigns <sym> = int(<sym>__mod, c_int).  Empty = no-op for flat kernels.
    module_symbol_sources: Dict[str, Tuple[str, str]] = field(default_factory=dict)


def build_auto_interface(raw: dict, entry: str) -> OriginalInterface:
    """Build an :class:`OriginalInterface` from the bridge's
    ``HLFIRModule.get_fortran_interface(entry)`` snapshot (pre-flatten).
    ``entry`` should be the final ``sdfg.name`` so bind(c) symbols match
    the compiled SDFG exports.

    Members with an unnamed dtype (nested record, allocatable/pointer/
    character, complex) carry placeholder ``fortran_type='??'`` so
    ``bind_c_shim`` can reject only the unsupported members and accept
    inline-flat ones from the same struct.

    :raises ValueError: dtype the binding layer can't name, or an
        unrecoverable derived-type name.
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
        rank = int(a["rank"])
        shape = tuple(a["shape"])
        # Assumed-shape dummies report rank but no shape -- default to a
        # rank-length ':' tuple (matches the hand-authored interface shape
        # used by the velocity_full e2e).
        if rank > 0 and not shape:
            shape = (":", ) * rank
        args.append(
            OriginalArg(name=a["name"],
                        fortran_type=fortran_type,
                        rank=rank,
                        shape=shape,
                        intent=a["intent"],
                        optional=bool(a.get("optional", False)),
                        struct_type=struct_type))
    used_modules = {mod: tuple(syms) for mod, syms in raw["used_modules"].items()}
    struct_types = {}
    for sname, st in raw.get("struct_types", {}).items():
        members = []
        for m in st["members"]:
            nested_name = m.get("struct_name") or ""
            if nested_name:
                # Nested derived-type member: declared type(<nested>),
                # consistent with a top-level derived-type arg; bind_c_shim
                # follows struct_name to recurse.
                fortran_type = f"type({nested_name})"
            else:
                fortran_type = _DTYPE_TO_FORTRAN_C.get(m["dtype"], "??")
            members.append(
                Member(
                    name=m["name"],
                    fortran_type=fortran_type,
                    rank=int(m["rank"]),
                    shape=tuple(m["shape"]),
                    struct_name=nested_name or None,
                    # .get: pre-alloc-field bridge snapshots deserialise with no guard.
                    alloc=m.get("alloc", ""),
                ))
        struct_types[sname] = DerivedType(name=st["name"], module=st["module"] or None, members=tuple(members))
    return OriginalInterface(entry=entry, args=tuple(args), struct_types=struct_types, used_modules=used_modules)
