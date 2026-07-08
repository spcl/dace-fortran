"""Auto-generate a ``bind(c, name='<entry>_c')`` wrapper around the
emitted ``<entry>_dace`` Fortran module procedure.

Why a separate shim?  The ``<entry>_dace`` wrapper emitted by
:func:`dace_fortran.bindings.emit_bindings` is a *Fortran module
procedure* -- its symbol is mangled by gfortran (e.g.
``__velocity_tendencies_dace_bindings_MOD_velocity_tendencies_dace``)
and its dummies are Fortran shape descriptors, not flat C pointers.
A C / ``ctypes`` / Python caller cannot reach it without writing a
hand-authored Fortran shim that ``USE``\\s the binding module and
re-exports the call under a known ``bind(c)`` symbol with flat C-ABI
dummies (the ``run_sr`` pattern in ``tests/mpi_comm_e2e_test.py``, the
``run_velocity_flat_c`` pattern in ``tests/icon/full/...``).

This emitter writes that shim mechanically from the same
:class:`OriginalInterface` the bindings emitter already consumes, so
downstream callers get a standalone ``.so`` with one stable C entry
per kernel and no per-kernel hand-written Fortran glue.

Supported dummy shapes:

  * **flat scalar / array dummies** -- the MVP shape; the dummy maps
    to one C-ABI slot (by-value for ``intent(in)`` scalars, ``c_ptr``
    + ``c_f_pointer`` alias otherwise).
  * **derived-type dummies whose every member is inline-flat**
    (scalar or static-shape array of scalar) -- the Phase 2.4 struct
    extension.  The dummy expands to one C-ABI slot per member; the
    shim allocates a local instance of the derived type, copies each
    member in, calls ``<entry>_dace`` with the whole struct, and (for
    ``out``/``inout``) copies each member back out.

Non-supported shapes that today raise
:class:`UnsupportedShimInterfaceError`:

  * derived types with nested derived-type members
  * derived types with ``allocatable`` / ``pointer`` / dynamic-shape
    members (the same v2 boundary that ``MarshalExternalStructs``
    rejects with its strict ``isInlineFlatMember`` check).
"""
import re
from pathlib import Path
from typing import List

from dace_fortran.bindings.fortran_interface import (
    DerivedType,
    Member,
    OriginalArg,
    OriginalInterface,
)

# A Fortran identifier inside a shape expression (``nproma``, the
# ``n_zlev`` of ``n_zlev + 1``).  Used to recover the module-variable
# extents a flat array dummy's static shape references.
_SHAPE_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


class UnsupportedShimInterfaceError(NotImplementedError):
    """Raised when an :class:`OriginalInterface` carries a dummy the
    shim emitter cannot yet handle (nested struct members,
    ``allocatable`` / ``pointer`` members, dynamic shape, or any
    derived type the :class:`OriginalInterface` did not record a
    layout for in :attr:`OriginalInterface.struct_types`)."""


def _dim_spec(shape) -> str:
    """``(:,:)`` for rank-N, empty for scalars."""
    if not shape:
        return ""
    return "(" + ", ".join(":" for _ in shape) + ")"


def _shape_literal(shape) -> str:
    """``[d1, d2, ...]`` Fortran array constructor for the
    ``c_f_pointer`` extent argument."""
    return "[" + ", ".join(str(s) for s in shape) + "]"


def _free_shape_symbols(iface: OriginalInterface) -> List[str]:
    """Identifiers a flat array dummy's *static* shape references that
    are NOT themselves scalar dummy args  --  Fortran *module*
    variables the kernel declared its array extents with (ICON ocean
    ``tracer(nproma, n_zlev)`` -> ``nproma`` / ``n_zlev``; ``w(nproma,
    n_zlev + 1)`` -> the same two).

    A derived-type dummy rides its member-shape constants via the
    struct's module ``use`` line (``NX`` / ``NY``, see
    :func:`_struct_module_use`); a *flat* array dummy has no struct
    module to import from, so the bare extent names reach the
    ``c_f_pointer`` shape constructor undeclared (gfortran:
    ``Symbol 'n_zlev' has no IMPLICIT type``).  We forward each across
    the C ABI as an ``integer(c_int), value`` arg the caller supplies,
    so the shape resolves with no module dependency  --  robust against
    the per-library module-copy hazard (the caller passes the actual
    extent rather than the shim reading a possibly-stale module copy of
    ``nproma``).

    Assumed-shape (``:``) dummies are excluded: they already take
    runtime extents through the ``<name>_d<i>`` dynamic path in
    :func:`_emit_flat_arg`.  Pure-literal extents (``g(3, ...)``)
    contribute no identifier.  Returns the distinct symbols in
    first-appearance order.
    """
    scalar_dummies = {a.name for a in iface.args if a.rank == 0 and a.struct_type is None}
    seen: set = set()
    out: List[str] = []
    for a in iface.args:
        if a.struct_type is not None or a.rank == 0:
            continue
        if any(s in ("?", "*", ":") for s in a.shape):
            continue  # dynamic -- per-dim extent args cover it
        for entry in a.shape:
            for ident in _SHAPE_IDENT_RE.findall(str(entry)):
                if ident in scalar_dummies or ident in seen:
                    continue
                seen.add(ident)
                out.append(ident)
    return out


def _is_inline_flat_member(m: Member) -> bool:
    """``True`` iff the bind(c) shim can build a ``c_f_pointer`` alias
    for ``m``.  Accepts:

    * Scalar members (``rank == 0``).
    * Static-shape arrays of scalar -- every shape entry is a
      compile-time literal or named constant (no ``'?'``, ``'*'``,
      ``':'``).
    * **v2** Dynamic-shape arrays of scalar -- shape entries may be
      ``'?'`` (the bridge's "unknown extent at HLFIR time" marker for
      ``box<heap|ptr<seq<...>>>`` allocatable / pointer array members,
      surfaced after the box-aware member extractor in
      ``extract_vars.cpp``).  ``_emit_struct_arg`` takes the member's
      extents as separate ``integer(c_int), value`` args
      (``<name>_<member>_d<i>``) and feeds them into the
      ``c_f_pointer`` shape constructor at runtime.

    A nested-struct member (``struct_name`` set) is NOT inline-flat at
    this level; the validation walk recurses into the nested layout
    and the emission walk descends through it.  Returns ``False`` so
    callers know to recurse rather than emit a leaf.

    Refuses members whose Fortran type the bridge could not name
    (``fortran_type == '??'`` and no ``struct_name``): the
    ``c_f_pointer`` would have no element type to spell.
    """
    if m.struct_name:
        return False
    if m.fortran_type == "??":
        return False
    return m.rank >= 0


def _is_value_record(iface: OriginalInterface, struct_name: str) -> bool:
    """``True`` iff ``struct_name`` is a flat fixed-size *value* record --
    every member is a non-struct, fully-static-shape leaf (the canonical
    case is ``t_cartesian_coordinates`` with its single ``x(3)``).

    An ARRAY of such a record (``p_vn_dual(:,:,:)``, a struct member
    ``edge2vert_coeff_cc_t(:,:,:,:)``) is reconstructed *element-wise*
    (``arr(i)%x(j) = flat(i, j)``) rather than by whole-array
    ``arr%x`` descent, which Fortran forbids ("two or more part
    references with nonzero rank").  A *container* record (``t_patch_vert``
    -- multiple members, nested records, dynamic-shape allocatable members)
    returns ``False``: it is indexed at element ``(1)`` and descended into,
    not scattered.  Distinguishing the two needs no pointer/allocatable
    flag -- a value record's members are all static-shape scalars/arrays;
    a container's are not."""
    st = iface.struct_types.get(struct_name)
    if st is None or not st.members:
        return False
    for m in st.members:
        if m.struct_name:
            return False
        if any(s in ("?", "*", ":") for s in m.shape):
            return False
    return True


def _validate_struct_layout_recursive(iface: OriginalInterface, st: DerivedType, arg_name: str, path: str):
    """Walk ``st`` (and every nested derived-type member) and raise
    :class:`UnsupportedShimInterfaceError` on the first leaf the
    emitter can't handle.  ``path`` is the Fortran-side access path
    used in the error message (``a%cells%foo``)."""
    for m in st.members:
        if m.struct_name:
            nested = iface.struct_types.get(m.struct_name)
            if nested is None:
                raise UnsupportedShimInterfaceError(f"bind(c) shim: nested struct member {path}%{m.name} "
                                                    f"is type {m.fortran_type} but no layout was recorded "
                                                    f"for {m.struct_name!r} in OriginalInterface."
                                                    f"struct_types.  The bridge's recursive struct walker "
                                                    f"should have populated this; check that the bridge "
                                                    f"build is current.")
            _validate_struct_layout_recursive(iface, nested, arg_name, f"{path}%{m.name}")
            continue
        if not _is_inline_flat_member(m):
            raise UnsupportedShimInterfaceError(f"bind(c) shim: argument {arg_name!r} has member "
                                                f"{path}%{m.name} ({m.fortran_type}, rank={m.rank}, "
                                                f"shape={m.shape}) the shim cannot handle.  Only scalar "
                                                f"and array-of-scalar members (static or dynamic shape) "
                                                f"are supported; complex / character / function-pointer "
                                                f"members need a hand-authored shim.")


def _collect_nested_struct_modules(iface: OriginalInterface, st: DerivedType, out_lines: List[str], seen: set):
    """Walk ``st``'s nested-derived-type members and append a
    ``use <mod>, only: ...`` line for every distinct module the
    nested types reference, so the shim can spell ``type(<nested>)``
    on the locals it never declares (the outer ``type(<outer>),
    target :: <a>`` already covers nested instances structurally, but
    the modules must still resolve at compile time)."""
    for m in st.members:
        if not m.struct_name:
            continue
        u = _struct_module_use(iface, m.struct_name)
        if u and u not in seen:
            out_lines.append(u)
            seen.add(u)
        nested = iface.struct_types.get(m.struct_name)
        if nested is not None:
            _collect_nested_struct_modules(iface, nested, out_lines, seen)


def _struct_module_use(iface: OriginalInterface, struct_name: str) -> str:
    """``use <mod>, only: <struct_name>[, <shape_const_1>, ...]`` for
    the module that defines ``struct_name``.  The static-shape
    constants the struct's member shapes reference (e.g. ``NX`` /
    ``NY``) live in the same module and must ride the import so the
    shim can spell them in its ``c_f_pointer`` calls."""
    for mod, syms in iface.used_modules.items():
        if struct_name in syms:
            return f"  use {mod}, only: {', '.join(syms)}"
    return ""


def _emit_flat_arg(a: OriginalArg, header_args: List[str], decls_value: List[str], decls_ptr: List[str],
                   decls_local: List[str], c_f_calls: List[str], call_args: List[str]):
    """Append the per-dummy split for a non-struct argument: scalar
    inputs ride by value, scalar outputs and arrays ride as ``c_ptr``
    + ``c_f_pointer`` alias.  Mutates the parallel lists in place.

    Dynamic-shape arrays (any ``'?'`` / ``'*'`` / ``':'`` entry in
    :attr:`shape`) take their extents from per-dim ``integer(c_int),
    value`` args ``<name>_d<i>`` declared ahead of the pointer, in
    declaration order -- the C caller passes the dims before the
    pointer (same convention :func:`_emit_struct_members_recursive`
    uses for dynamic-shape struct members)."""
    if a.rank == 0 and a.intent in ('in', ''):
        header_args.append(a.name)
        decls_value.append(f"  {a.fortran_type}, value :: {a.name}")
        call_args.append(a.name)
        return
    is_dynamic = a.rank > 0 and any(s in ('?', '*', ':') for s in a.shape)
    if is_dynamic:
        ext_names = [f"{a.name}_d{i}" for i in range(a.rank)]
        for en in ext_names:
            header_args.append(en)
            decls_value.append(f"  integer(c_int), value :: {en}")
    ptr_name = f"{a.name}_p"
    header_args.append(ptr_name)
    decls_ptr.append(f"  type(c_ptr), value :: {ptr_name}")
    if a.rank == 0:
        # Length-1 array alias for scalar I/O (matches
        # ``feedback_scalar_io_convention``).
        decls_local.append(f"  {a.fortran_type}, pointer :: {a.name}(:)")
        c_f_calls.append(f"  call c_f_pointer({ptr_name}, {a.name}, [1])")
    else:
        decls_local.append(f"  {a.fortran_type}, pointer :: {a.name}{_dim_spec(a.shape)}")
        if is_dynamic:
            shape_tok = "[" + ", ".join(ext_names) + "]"
        else:
            shape_tok = _shape_literal(a.shape)
        c_f_calls.append(f"  call c_f_pointer({ptr_name}, {a.name}, "
                         f"{shape_tok})")
    call_args.append(a.name)


def _emit_value_record_array(iface: OriginalInterface, vt_name: str, outer_rank: int, inst_path: str, flat_prefix: str,
                             intent: str, header_args: List[str], decls_value: List[str], decls_ptr: List[str],
                             decls_local: List[str], c_f_calls: List[str], copy_in: List[str], copy_out: List[str]):
    """Reconstruct an ARRAY of a value record (``_is_value_record``,
    e.g. ``t_cartesian_coordinates``) element-wise.  ``outer_rank`` is the
    rank of the array carrying the record (the dummy's rank for a top-level
    ``p_vn_dual(:,:,:)``, the member's rank for a nested
    ``edge2vert_coeff_cc_t(:,:,:,:)``); ``inst_path`` is the Fortran array
    instance to assemble.

    Per value member ``v`` a flat C slot ``<flat>_<v>`` of rank
    ``outer_rank + v.rank`` carries the SoA companion -- outer dims
    first, member dims last, matching the wrapper's ``arr(i1..)%v(j1..)
    = flat(i1.., j1..)`` gather -- so the shim's scatter is its exact
    inverse.  ``intent`` (inherited from the outermost dummy) selects
    scatter (copy-in) and/or gather (copy-out).

    C-ABI ordering: the outer extents ride PER FIELD as
    ``integer(c_int), value`` args ``<flat>_<v>_d<i>`` immediately
    ahead of that field's pointer, so each field emits ``[<v>_d0..,
    <v>_p]``.  This mirrors the SDFG side, where the flatten pass
    lowers the value-record array to one INDEPENDENT SoA companion per
    field (``..._v1`` / ``..._v2``), each with its OWN symbolic
    extents; ``builder.emit_library``'s ``per_member_soa`` +
    ``dynamic_extents_abi`` path emits those extents per-leaf
    (``emit_library`` ~1651-1654 / ~1760-1764), so the two ABIs
    coincide leaf-for-leaf.  The fields share one runtime allocation
    (they are the same array-of-records) so the ``allocate`` uses the
    FIRST field's extents; every field's extents are equal at runtime.

    The ``allocate(inst(d0, ...))`` runs in ``copy_in`` ahead of the
    scatter: a top-level allocatable local or a nested pointer member both
    accept it."""
    st = iface.struct_types[vt_name]
    alloc_ext_names = None  # the first field's outer extents drive the shared allocate
    for v in st.members:
        flat_name = f"{flat_prefix}_{v.name}"
        ext_names = [f"{flat_name}_d{i}" for i in range(outer_rank)]
        for en in ext_names:
            header_args.append(en)
            decls_value.append(f"  integer(c_int), value :: {en}")
        if alloc_ext_names is None:
            alloc_ext_names = ext_names
            copy_in.append(f"  allocate({inst_path}({', '.join(ext_names)}))")
        ptr_name = f"{flat_name}_p"
        header_args.append(ptr_name)
        decls_ptr.append(f"  type(c_ptr), value :: {ptr_name}")
        total_rank = outer_rank + v.rank
        shape_toks = list(ext_names) + [str(s) for s in v.shape]
        decls_local.append(f"  {v.fortran_type}, pointer :: {flat_name}({', '.join(':' for _ in range(total_rank))})")
        c_f_calls.append(f"  call c_f_pointer({ptr_name}, {flat_name}, [{', '.join(shape_toks)}])")
        idx = [f"{flat_name}_i{k}" for k in range(total_rank)]
        decls_local.append(f"  integer :: {', '.join(idx)}")
        outer_idx, inner_idx = idx[:outer_rank], idx[outer_rank:]
        lhs = f"{inst_path}({', '.join(outer_idx)})%{v.name}"
        if inner_idx:
            lhs += f"({', '.join(inner_idx)})"
        rhs = f"{flat_name}({', '.join(idx)})"
        opens = ["  " + "  " * k + f"do {idx[k]} = 1, {shape_toks[k]}" for k in range(total_rank)]
        closes = ["  " + "  " * (total_rank - 1 - k) + "end do" for k in range(total_rank)]
        body = "  " + "  " * total_rank
        if intent in ("in", "inout", ""):
            copy_in.extend(opens)
            copy_in.append(f"{body}{lhs} = {rhs}")
            copy_in.extend(closes)
        if intent in ("out", "inout"):
            copy_out.extend(opens)
            copy_out.append(f"{body}{rhs} = {lhs}")
            copy_out.extend(closes)


#: A double-buffer lane's source expression, ``<prefix>%<aor>(<sym>)%<leaf>``
#: (e.g. ``p%prog(nnow)%rho``).  Emitted by ``splitDoubleBufferMembers`` /
#: ``recordStructArgEntry``: the literal ``(<sym>)`` only appears for a static
#: per-time-level lane, so this never matches a plain AoS member (whose
#: ``outer_expr`` is ``arr%w`` with the index riding ``$i`` placeholders).
_DBUF_OUTER_RE = re.compile(r'^(?P<prefix>.+)%(?P<aor>\w+)\((?P<sym>\w+)\)%(?P<leaf>.+)$')


def _build_dbuf_map(plan) -> dict:
    """Group the FlattenPlan's double-buffer lane recipes by their
    ``(struct-instance prefix, AoR member)`` so the shim can reconstruct the
    array-of-records with each per-time-level lane populated from its own
    C-ABI buffer.

    Returns ``{(prefix, aor): {sym: [leaf_info, ...]}}`` where ``leaf_info``
    is ``{leaf, flat, rank, dtype, intent}``.  Empty when ``plan`` is None or
    carries no double-buffer lanes."""
    dbuf: dict = {}
    if plan is None:
        return dbuf
    for e in plan.entries:
        m = _DBUF_OUTER_RE.match(e.outer_expr)
        if not m:
            continue
        r = e.recipe
        if not r.flat_names:
            continue
        info = {
            'leaf': m['leaf'],
            'flat': r.flat_names[0],
            'rank': r.rank,
            'dtype': r.scratch_dtype,
            'intent': e.writeback_intent,
        }
        dbuf.setdefault((m['prefix'], m['aor']), {}).setdefault(m['sym'], []).append(info)
    return dbuf


def _emit_double_buffer_member(inst_path: str, aor: str, syms: dict, header_args: List[str], decls_value: List[str],
                               decls_ptr: List[str], decls_local: List[str], c_f_calls: List[str], copy_in: List[str],
                               copy_out: List[str]):
    """Reconstruct an ICON double-buffer AoR member (``p%prog``) from the SDFG's
    per-time-level lane buffers.  The bridge split ``prog(nnow)`` / ``prog(nnew)``
    into static lanes (``p_prog_nnow_rho`` / ``p_prog_nnew_rho``); the binding
    aliases each lane back to ``p%prog(<sym>)%<leaf>``, so the shim must allocate
    the record array large enough for every time-level index and populate each
    one from its own C-ABI buffer.  ``syms`` maps each index symbol (already a
    by-value C-ABI scalar arg) to its leaf buffers."""
    sym_names = sorted(syms)
    max_expr = sym_names[0] if len(sym_names) == 1 else "max(" + ", ".join(sym_names) + ")"
    copy_in.append(f"  allocate({inst_path}%{aor}({max_expr}))")
    for sym in sym_names:
        for info in syms[sym]:
            flat, leaf, rank = info['flat'], info['leaf'], info['rank']
            ftype = _MOD_FORWARD_SCALAR_FTYPE.get(info['dtype'], 'real(c_double)')
            ext_names = [f"{flat}_d{i}" for i in range(rank)]
            for en in ext_names:
                header_args.append(en)
                decls_value.append(f"  integer(c_int), value :: {en}")
            ptr = f"{flat}_p"
            header_args.append(ptr)
            decls_ptr.append(f"  type(c_ptr), value :: {ptr}")
            dim_colons = "(" + ", ".join(":" for _ in range(rank)) + ")"
            decls_local.append(f"  {ftype}, pointer :: {flat}{dim_colons}")
            c_f_calls.append(f"  call c_f_pointer({ptr}, {flat}, [{', '.join(ext_names)}])")
            extents = "(" + ", ".join(ext_names) + ")"
            copy_in.append(f"  allocate({inst_path}%{aor}({sym})%{leaf}{extents})")
            if info['intent'] in ('', 'in', 'inout'):
                copy_in.append(f"  {inst_path}%{aor}({sym})%{leaf} = {flat}")
            if info['intent'] in ('out', 'inout'):
                copy_out.append(f"  {flat} = {inst_path}%{aor}({sym})%{leaf}")


def _emit_struct_members_recursive(iface: OriginalInterface,
                                   st: DerivedType,
                                   inst_path: str,
                                   flat_prefix: str,
                                   intent: str,
                                   header_args: List[str],
                                   decls_value: List[str],
                                   decls_ptr: List[str],
                                   decls_local: List[str],
                                   c_f_calls: List[str],
                                   copy_in: List[str],
                                   copy_out: List[str],
                                   shape_syms: set,
                                   dbuf_map: dict = None):
    """Walk ``st``'s members; for each leaf emit a C-ABI slot plus
    the matching ``c_f_pointer`` alias + copy-in / copy-out, for each
    nested-struct member descend into the nested layout with extended
    paths.  ``inst_path`` is the Fortran-side access path
    (``a%cells%foo``) used in struct-field assignments; ``flat_prefix``
    is the C-ABI naming root (``a_cells_foo``) used for header arg
    names and local pointer names; ``intent`` is inherited from the
    outermost dummy.

    ``shape_syms`` is the set of flat names a *flat array dummy*'s shape
    references (see :func:`_free_shape_symbols`).  A scalar member whose
    flat name lands in that set (``patch_3d_p_patch_2d_nblks_e``, which is
    both ``patch_3d%p_patch_2d(1)%nblks_e`` AND the trailing extent of
    ``vn(nproma, n_zlev, patch_3d_p_patch_2d_nblks_e)``) is forwarded
    ONCE, by value, under that name -- the array ``c_f_pointer`` shapes
    reference the same scalar directly, and the shape-symbol forwarding
    skips it as already-present.  Emitting it as a length-1 pointer alias
    instead would both double-declare the name and leave the array shapes
    referencing a rank-1 pointer where a scalar extent is required."""
    for m in st.members:
        if m.struct_name:
            if m.rank == 0:
                # Scalar nested record (``edges``, ``in_domain``): descend in
                # place, no index, no alloc.
                nested = iface.struct_types[m.struct_name]
                _emit_struct_members_recursive(iface, nested, f"{inst_path}%{m.name}", f"{flat_prefix}_{m.name}",
                                               intent, header_args, decls_value, decls_ptr, decls_local, c_f_calls,
                                               copy_in, copy_out, shape_syms)
            elif _is_value_record(iface, m.struct_name):
                # Array of a value record (``edge2vert_coeff_cc_t(:,:,:,:)``
                # of ``t_cartesian_coordinates``): scatter element-wise.
                _emit_value_record_array(iface, m.struct_name, m.rank, f"{inst_path}%{m.name}",
                                         f"{flat_prefix}_{m.name}", intent, header_args, decls_value, decls_ptr,
                                         decls_local, c_f_calls, copy_in, copy_out)
            elif dbuf_map and (inst_path, m.name) in dbuf_map:
                # ICON double-buffer AoR (``p%prog(nnow)`` / ``p%prog(nnew)``):
                # the bridge split it into per-time-level lanes.  Allocate the
                # record array to cover every time-level index and populate
                # each one from its own C-ABI lane buffer.
                _emit_double_buffer_member(inst_path, m.name, dbuf_map[(inst_path, m.name)], header_args, decls_value,
                                           decls_ptr, decls_local, c_f_calls, copy_in, copy_out)
            else:
                # Array of a container record (``p_patch_1d(:)`` /
                # ``p_patch_2d(:)`` of ``t_patch_vert`` / ``t_patch``).  The
                # ICON ocean kernels are single-patch -- the record array is
                # only ever indexed ``(1)`` -- so allocate the pointer array
                # to size 1 and descend into element ``(1)``; the wrapper's
                # ``1..size`` AoS gather then round-trips through size 1.
                copy_in.append(f"  allocate({inst_path}%{m.name}(1))")
                nested = iface.struct_types[m.struct_name]
                _emit_struct_members_recursive(iface, nested, f"{inst_path}%{m.name}(1)", f"{flat_prefix}_{m.name}",
                                               intent, header_args, decls_value, decls_ptr, decls_local, c_f_calls,
                                               copy_in, copy_out, shape_syms, dbuf_map)
            continue
        flat_name = f"{flat_prefix}_{m.name}"
        if m.rank == 0 and intent in ('in', '') and flat_name in shape_syms:
            # Scalar member that is also a flat array dummy's extent: one
            # by-value arg, shared by the struct copy and the array shapes.
            header_args.append(flat_name)
            decls_value.append(f"  {m.fortran_type}, value :: {flat_name}")
            copy_in.append(f"  {inst_path}%{m.name} = {flat_name}")
            continue
        ptr_name = f"{flat_name}_p"
        is_dynamic = any(s in ('?', '*', ':') for s in m.shape)
        if is_dynamic:
            # Per dim, ahead of the pointer: the lower bound then the extent,
            # both ``integer(c_int), value`` (``<flat>_lb<i>`` / ``<flat>_d<i>``).
            # Carrying the lower bound lets the shim rebuild the member at its
            # TRUE bounds, so the binding's ``offset_<member>_d<i> =
            # lbound(member)`` (and the SDFG's ``arr[(idx) - offset]`` indexing)
            # is correct for arrays ICON allocates with a non-default lower bound
            # -- the refinement-control index arrays
            # ``verts/cells/edges%{start,end}_{block,index}``, allocated
            # ``(min_rl : max_rl)`` and read at negative ``rl``.  Defaulting the
            # lower bound to 1 (the old behaviour) put ``end_block(-5)`` below
            # bound 1.
            lb_names = [f"{flat_name}_lb{i}" for i in range(m.rank)]
            ext_names = [f"{flat_name}_d{i}" for i in range(m.rank)]
            for lb, en in zip(lb_names, ext_names):
                header_args.append(lb)
                decls_value.append(f"  integer(c_int), value :: {lb}")
                header_args.append(en)
                decls_value.append(f"  integer(c_int), value :: {en}")
        header_args.append(ptr_name)
        decls_ptr.append(f"  type(c_ptr), value :: {ptr_name}")
        if m.rank == 0:
            decls_local.append(f"  {m.fortran_type}, pointer :: {flat_name}(:)")
            c_f_calls.append(f"  call c_f_pointer({ptr_name}, {flat_name}, [1])")
        else:
            decls_local.append(f"  {m.fortran_type}, pointer :: {flat_name}{_dim_spec(m.shape)}")
            if is_dynamic:
                shape_tok = "[" + ", ".join(ext_names) + "]"
            else:
                shape_tok = _shape_literal(m.shape)
            c_f_calls.append(f"  call c_f_pointer({ptr_name}, {flat_name}, "
                             f"{shape_tok})")
        if is_dynamic and m.rank > 0:
            # Dynamic-shape member is ALLOCATABLE / POINTER in the
            # outer struct's Fortran type definition.  ``=>`` would
            # only be valid for POINTER members and the bridge does
            # not yet distinguish POINTER vs ALLOCATABLE on
            # ``fir.BoxType`` extracts, so we take the universal
            # path: ``allocate`` the struct field at the runtime
            # extents (allocatable + pointer both accept this),
            # element-copy-in for ``intent(in / inout / '')``, call,
            # element-copy-back for ``intent(out / inout)``.  Costs
            # one extra copy per dynamic member per call; the
            # per_member_soa no-pack contract still holds on the
            # outer SDFG side (no AoS struct buffer is built --
            # only the per-leaf pointers cross the C ABI).
            #
            # Allocate at the member's TRUE bounds ``(lb : lb+d-1)`` (not the
            # default ``(d)`` == ``(1:d)``) so ``lbound(<member>)`` -- which the
            # binding reads into ``offset_<member>_d<i>`` -- is the real lower
            # bound.  The whole-array copy from the 1-based flat companion is by
            # position (Fortran intrinsic assignment is shape-conformable, not
            # bound-matched), so ``member(lb)`` takes ``flat(1)``.
            bounds_tok = "(" + ", ".join(f"{lb} : {lb} + {en} - 1" for lb, en in zip(lb_names, ext_names)) + ")"
            copy_in.append(f"  allocate({inst_path}%{m.name}{bounds_tok})")
            if intent in ('in', 'inout', ''):
                copy_in.append(f"  {inst_path}%{m.name} = {flat_name}")
            if intent in ('out', 'inout'):
                copy_out.append(f"  {flat_name} = {inst_path}%{m.name}")
            continue
        if intent in ('in', 'inout', ''):
            if m.rank == 0:
                copy_in.append(f"  {inst_path}%{m.name} = {flat_name}(1)")
            else:
                copy_in.append(f"  {inst_path}%{m.name} = {flat_name}")
        if intent in ('out', 'inout'):
            if m.rank == 0:
                copy_out.append(f"  {flat_name}(1) = {inst_path}%{m.name}")
            else:
                copy_out.append(f"  {flat_name} = {inst_path}%{m.name}")


def _emit_struct_arg(a: OriginalArg,
                     st: DerivedType,
                     iface: OriginalInterface,
                     header_args: List[str],
                     decls_value: List[str],
                     decls_ptr: List[str],
                     decls_local: List[str],
                     c_f_calls: List[str],
                     copy_in: List[str],
                     copy_out: List[str],
                     call_args: List[str],
                     shape_syms: set,
                     dbuf_map: dict = None):
    """Append the per-member split for a derived-type argument.

    The dummy becomes a local ``type(<struct>), target :: <name>``;
    every inline-flat leaf member (transitively, descending through
    nested derived-type members) rides as its own C-ABI slot
    ``<name>_<...path...>_<leaf>_p`` (``c_ptr, value``) aliased
    through ``c_f_pointer``.  Dynamic-shape leaf extents come as
    separate ``integer(c_int), value`` args ``<flat>_d<i>`` ahead of
    the pointer, in declaration order -- matching the marshal-
    expanded leaf ordering on the outer SDFG's emit_call side.

    Per :attr:`OriginalArg.intent` (inherited unchanged through
    nested struct members):

    * Static-shape leaves: copy-in / copy-out element-wise.
    * Dynamic-shape leaves: pointer-assign
      ``<a>%<...>%<leaf> => <flat>`` to alias the SDFG companion in
      place -- the bridge's struct flatten already arranged the
      storage layout so no element copy is needed.  Skipping the
      copy preserves the per_member_soa no-pack contract on the
      outer side.
    """
    if a.rank > 0:
        # Array-of-record dummy.  A value record (``p_vn_dual(:,:,:)`` of
        # ``t_cartesian_coordinates``) scatters element-wise into a local
        # allocatable; a top-level container-record array has no kernel here
        # and would need the size-1 record path -- reject it loudly rather
        # than silently emit a rank-mismatched scalar instance.
        if not _is_value_record(iface, a.struct_type):
            raise UnsupportedShimInterfaceError(f"bind(c) shim: dummy {a.name!r} is an array (rank {a.rank}) of the "
                                                f"container record {a.struct_type!r}; only arrays of flat value "
                                                f"records (every member a static-shape leaf) are reconstructed "
                                                f"element-wise today.")
        _emit_value_record_array(iface, a.struct_type, a.rank, a.name, a.name, a.intent, header_args, decls_value,
                                 decls_ptr, decls_local, c_f_calls, copy_in, copy_out)
        shape = ", ".join(":" for _ in range(a.rank))
        decls_local.append(f"  {a.fortran_type}, allocatable, target :: {a.name}({shape})")
        call_args.append(a.name)
        return
    _emit_struct_members_recursive(iface, st, a.name, a.name, a.intent, header_args, decls_value, decls_ptr,
                                   decls_local, c_f_calls, copy_in, copy_out, shape_syms, dbuf_map)
    decls_local.append(f"  {a.fortran_type}, target :: {a.name}")
    call_args.append(a.name)


# SDFG dtype -> ``iso_c_binding`` Fortran type for a pass-by-value
# scalar C ABI arg.  Keep in sync with the matching outer-side
# ``emit_library._sym2c`` casts so the two ABIs coincide.
_MOD_FORWARD_SCALAR_FTYPE = {
    "int32": "integer(c_int)",
    "int64": "integer(c_long_long)",
    "float32": "real(c_float)",
    "float64": "real(c_double)",
    "bool": "logical(c_bool)",
}


def _emit_module_symbol_forward(module_symbol_forward, header_args: List[str], decls_value: List[str],
                                decls_ptr: List[str], decls_local: List[str], c_f_calls: List[str], copy_in: List[str],
                                use_lines: List[str]):
    """Per ``(module, member, dtype, rank)`` in ``module_symbol_forward``,
    extend the bind(c) shim so the caller can write the INNER library's
    copy of ``<module>::<member>`` directly via the C ABI.

    For a scalar (rank == 0): append a pass-by-value ``<dtype>`` arg
    ``<member>_arg``, ``use <module>, only: <member>__sink => <member>``
    so the import resolves to *this* library's copy (which gfortran
    ships per-library, distinct from any other linked library's), and
    ``<member>__sink = <member>_arg`` in the body.

    For a rank-N fixed-shape array: append a ``type(c_ptr), value ::
    <member>_p`` arg, ``c_f_pointer`` it to a rank-N pointer aliased
    via the source-member's static shape (read off the imported
    ``<member>__sink``), and copy-assign whole-array.

    Mutates the parallel lists in place; the assignments land in
    ``copy_in`` so they run BEFORE the local-struct alloc + the
    ``<entry>_dace`` call, ensuring the inner kernel reads the value
    the outer side passed.
    """
    seen_use_aliases = set()
    for module, member, dtype, rank in module_symbol_forward:
        ftype = _MOD_FORWARD_SCALAR_FTYPE.get(dtype)
        if ftype is None:
            raise ValueError(f"bind_c_shim module_symbol_forward: unsupported dtype "
                             f"{dtype!r} for ``{module}::{member}``; extend "
                             f"``_MOD_FORWARD_SCALAR_FTYPE`` for new pass-by-value "
                             f"shapes.")
        # The same module may appear multiple times (e.g.
        # ``mo_run_config`` carries both ``lvert_nest`` and
        # ``timers_level``).  Collapse the ``use`` line via a single
        # ``only:`` list per module to keep the shim header tidy.
        alias = f"{member}__sink"
        if alias not in seen_use_aliases:
            use_lines.append(f"  use {module}, only: {alias} => {member}")
            seen_use_aliases.add(alias)
        if rank == 0:
            arg = f"{member}_arg"
            header_args.append(arg)
            decls_value.append(f"  {ftype}, value :: {arg}")
            copy_in.append(f"  {alias} = {arg}")
        else:
            # Rank-N fixed-shape array: receive a pointer, alias via
            # ``size`` queries on the source member, whole-array
            # assign.  ``size(<sink>, dim=d)`` works because the
            # ``use``d member is a static-shape array (its extents
            # are baked into the module's type info).
            ptr = f"{member}_p"
            local = f"{member}_buf"
            header_args.append(ptr)
            decls_ptr.append(f"  type(c_ptr), value :: {ptr}")
            shape_spec = ", ".join(":" for _ in range(rank))
            decls_local.append(f"  {ftype}, pointer :: {local}({shape_spec})")
            shape_args = ", ".join(f"size({alias}, dim={d + 1})" for d in range(rank))
            c_f_calls.append(f"  call c_f_pointer({ptr}, {local}, [{shape_args}])")
            copy_in.append(f"  {alias} = {local}")


def scalar_pointer_members(iface: OriginalInterface) -> frozenset:
    """Flat names of the rank-0 struct members this shim forwards as a
    ``type(c_ptr), value`` POINTER (aliased ``c_f_pointer(<x>_p, <x>, [1])``)
    rather than ``integer(c_int), value`` BY VALUE.

    A sibling-SDFG *caller* marshalling a struct argument to this shim
    (``emit_library`` ``per_member_soa`` + ``dynamic_extents_abi``) must pass a
    POINTER for each such member -- the address of a scratch cell holding the
    value -- EVEN when the caller itself holds the member as an SDFG *symbol* (a
    promoted grid extent / loop bound).  Passing the raw scalar value there,
    where the shim declares ``type(c_ptr), value`` and dereferences it via
    ``c_f_pointer``, would reinterpret the integer as an address.

    A rank-0 member crosses BY VALUE only in the shim's single by-value case
    (:func:`_emit_struct_members_recursive`): a read-only member that is ALSO a
    flat-array-dummy extent (``flat_name in shape_syms``).  Every other rank-0
    member is a pointer.  This mirrors that routine's routing exactly -- nested
    rank-0 records are descended in place; rank>0 struct members are arrays of
    records (value-record scatter / double-buffer AoR), whose leaves are arrays,
    never scalars."""
    shape_syms = set(_free_shape_symbols(iface))
    out: set = set()

    def walk(st: DerivedType, flat_prefix: str, intent: str):
        for m in st.members:
            flat_name = f"{flat_prefix}_{m.name}"
            if m.struct_name:
                if m.rank == 0:
                    walk(iface.struct_types[m.struct_name], flat_name, intent)
                continue
            if m.rank == 0 and not (intent in ("in", "") and flat_name in shape_syms):
                out.add(flat_name)

    for a in iface.args:
        if a.struct_type is None or a.rank != 0:
            continue
        walk(iface.struct_types[a.struct_type], a.name, a.intent)
    return frozenset(out)


def emit_bind_c_shim(iface: OriginalInterface,
                     out_path: str,
                     debug_prints: bool = False,
                     module_symbol_forward=(),
                     plan=None) -> Path:
    """Emit ``<entry>_c.f90`` -- a thin ``bind(c)`` wrapper around the
    binding module's ``<entry>_dace`` procedure.

    Per dummy:

    * **scalar input** (``rank == 0``, ``intent in / ''``): declared
      ``<fortran_type>, value`` -- the C-side passes the value
      directly, no pointer indirection.
    * **scalar output** (``rank == 0``, ``intent out / inout``):
      declared as a ``c_ptr, value`` and aliased through
      ``c_f_pointer`` to a length-1 array.  Matches
      ``feedback_scalar_io_convention`` -- inputs by value, outputs
      via pointer to a length-1 buffer.
    * **array** (``rank > 0``): declared as a ``c_ptr, value`` and
      aliased through ``c_f_pointer`` to the dummy's declared shape.
      The shape extents reference the scalar-input dummies preceding
      the array in C-ABI order, so the C caller passes dims first.
    * **derived-type dummy whose every member is inline-flat**: one
      C-ABI slot per member (named ``<dummy>_<member>_p``); a local
      ``type(<struct>), target :: <dummy>`` instance is assembled
      from the flat aliases (copy-in), passed whole to
      ``<entry>_dace``, and (for ``out``/``inout``) copied back out
      per member.

    After all aliases are set the shim calls
    ``<entry>_dace(...)`` with the *Fortran-side* names (the local
    aliases for flat dummies, the local instance for struct dummies)
    and finalises with ``<entry>_dace_finalize()`` so the DaCe handle
    is reference-counted out on the last call.

    :param iface: caller-facing Fortran interface.
    :param out_path: where to write ``<entry>_c.f90``.  Parent dirs
                     are created as needed; any existing file at the
                     path is overwritten.
    :returns: ``out_path`` as a :class:`~pathlib.Path` (just written).
    :raises UnsupportedShimInterfaceError: a struct dummy has a
            member shape the emitter cannot handle (nested struct,
            ``allocatable`` / ``pointer``, dynamic shape, or no
            recorded layout in
            :attr:`OriginalInterface.struct_types`).
    """
    # Validate every struct dummy (transitively through nested
    # derived-type members) has only emitter-handleable leaves.
    for a in iface.args:
        if a.struct_type is None:
            continue
        st = iface.struct_types.get(a.struct_type)
        if st is None:
            raise UnsupportedShimInterfaceError(f"bind(c) shim: dummy {a.name!r} is type {a.fortran_type} "
                                                f"but no layout was recorded for {a.struct_type!r} in "
                                                f"OriginalInterface.struct_types.  Supply a hand-authored "
                                                f"interface with the member list.")
        _validate_struct_layout_recursive(iface, st, a.name, a.name)

    entry = iface.entry
    c_name = f"{entry}_c"
    bind_mod = f"{entry}_dace_bindings"

    header_args: List[str] = []
    decls_value: List[str] = []
    decls_ptr: List[str] = []
    decls_local: List[str] = []
    c_f_calls: List[str] = []
    copy_in: List[str] = []
    copy_out: List[str] = []
    call_args: List[str] = []
    struct_use_lines: List[str] = []
    seen_struct_uses = set()
    # Flat names a flat array dummy's shape references -- a scalar struct
    # member that coincides with one is forwarded by value (and shared with
    # the array shapes) instead of as a length-1 pointer alias.
    shape_syms = set(_free_shape_symbols(iface))
    # Double-buffer AoR lanes (ICON ``p%prog(nnow)`` / ``p%prog(nnew)``) the
    # bridge split into static per-time-level companions, grouped by their
    # ``(struct-instance, AoR member)`` so the struct reconstruction can
    # allocate + populate every time-level element from its own lane buffer.
    dbuf_map = _build_dbuf_map(plan)
    for a in iface.args:
        if a.struct_type is None:
            _emit_flat_arg(a, header_args, decls_value, decls_ptr, decls_local, c_f_calls, call_args)
            continue
        st = iface.struct_types[a.struct_type]
        _emit_struct_arg(a, st, iface, header_args, decls_value, decls_ptr, decls_local, c_f_calls, copy_in, copy_out,
                         call_args, shape_syms, dbuf_map)
        # Pick up the ``use`` line for the struct's defining module
        # AND every nested-derived-type member's module so the shim
        # can spell ``type(<struct>)`` / ``type(<nested>)`` and any
        # shape constants the member declarations reference.
        u = _struct_module_use(iface, a.struct_type)
        if u and u not in seen_struct_uses:
            struct_use_lines.append(u)
            seen_struct_uses.add(u)
        _collect_nested_struct_modules(iface, st, struct_use_lines, seen_struct_uses)

    # Forward the module-variable extents a flat array dummy's static
    # shape references (``tracer(nproma, n_zlev)``) as ``integer(c_int),
    # value`` C args, declared ahead of the array pointers so the
    # ``c_f_pointer`` shape constructors resolve.  Prepended (extents
    # first) to a deterministic C-ABI arg order; skipped when the name
    # is already an arg (a scalar dummy or a ``<name>_d<i>`` dynamic
    # extent).  Struct member-shape constants are NOT included here --
    # they ride the struct's module ``use`` line.
    existing_args = set(header_args)
    shape_sym_args: List[str] = []
    shape_sym_decls: List[str] = []
    for sym in _free_shape_symbols(iface):
        if sym in existing_args:
            continue
        existing_args.add(sym)
        shape_sym_args.append(sym)
        shape_sym_decls.append(f"  integer(c_int), value :: {sym}")
    header_args = shape_sym_args + header_args
    decls_value = shape_sym_decls + decls_value

    # Forward Fortran module globals across the C ABI -- under
    # default ELF+gfortran linking, every shared library that USEs a
    # module gets its OWN BSS copy of the module's variables; an
    # outer caller writing ``mo_parallel_config::nproma`` in
    # libouter.so does NOT reach libinner.so's copy.  The shim
    # accepts the value as a C ABI arg and writes the INNER copy via
    # ``use <module>, only: <member>__sink => <member>`` so the
    # ``<entry>_dace`` call (in the inner library) reads the same
    # value the outer caller passed.  See the velocity dycore + ext
    # e2e ASan ODR diagnostic for the root cause.
    module_forward_use_lines: List[str] = []
    _emit_module_symbol_forward(module_symbol_forward, header_args, decls_value, decls_ptr, decls_local, c_f_calls,
                                copy_in, module_forward_use_lines)

    decl_block = "\n".join(decls_value + decls_ptr + decls_local)
    body_parts: List[str] = []
    if debug_prints:
        body_parts.append(f"  write(0, *) '[{c_name}] enter'")
        body_parts.append("  flush(0)")
    if c_f_calls:
        body_parts.append("\n".join(c_f_calls))
        if debug_prints:
            body_parts.append(f"  write(0, *) '[{c_name}] c_f_pointer done'")
            body_parts.append("  flush(0)")
    if copy_in:
        body_parts.append("\n".join(copy_in))
        if debug_prints:
            body_parts.append(f"  write(0, *) '[{c_name}] copy-in done'")
            body_parts.append("  flush(0)")
    if debug_prints:
        body_parts.append(f"  write(0, *) '[{c_name}] about to call {entry}_dace'")
        body_parts.append("  flush(0)")
    body_parts.append(f"  call {entry}_dace({', '.join(call_args)})")
    if debug_prints:
        body_parts.append(f"  write(0, *) '[{c_name}] {entry}_dace returned'")
        body_parts.append("  flush(0)")
    if copy_out:
        body_parts.append("\n".join(copy_out))
        if debug_prints:
            body_parts.append(f"  write(0, *) '[{c_name}] copy-out done'")
            body_parts.append("  flush(0)")
    body_parts.append(f"  call {entry}_dace_finalize()")
    if debug_prints:
        body_parts.append(f"  write(0, *) '[{c_name}] finalize done'")
        body_parts.append("  flush(0)")
    body_block = "\n".join(body_parts)

    use_lines = [
        # ``, intrinsic`` forces the real intrinsic module even when a
        # USE-imported source stubs a same-named ``iso_c_binding`` (matches the
        # generated bindings module; the shim needs the real ``c_*`` kinds).
        "  use, intrinsic :: iso_c_binding",
        *struct_use_lines,
        *module_forward_use_lines,
        f"  use {bind_mod}, only: {entry}_dace, {entry}_dace_finalize"
    ]
    lines = [
        "! AUTO-GENERATED by dace_fortran.bindings.bind_c_shim -- do not edit.",
        f"! bind(c) shim around module procedure {bind_mod}::{entry}_dace.",
        f"subroutine {c_name}({', '.join(header_args)}) "
        f"bind(c, name='{c_name}')",
        *use_lines,
        "  implicit none",
        decl_block,
        body_block,
        f"end subroutine {c_name}",
        "",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path
