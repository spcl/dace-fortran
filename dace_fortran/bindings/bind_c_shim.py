"""Auto-generate a ``bind(c, name='<entry>_c')`` wrapper around the
emitted ``<entry>_dace`` Fortran module procedure.

``<entry>_dace`` is a Fortran module procedure: gfortran mangles its
symbol and its dummies are Fortran shape descriptors, not flat C
pointers, so a C/ctypes/Python caller needs this shim instead.  Built
from the same :class:`OriginalInterface` the bindings emitter consumes.

Supported: flat scalar/array dummies; derived-type dummies whose every
member is inline-flat (scalar or static-shape array of scalar).
Unsupported (raises :class:`UnsupportedShimInterfaceError`): nested
derived-type members; allocatable/pointer/dynamic-shape members.
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

# Identifier in a shape expr (e.g. nproma) -- recovers module-variable extents.
_SHAPE_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


class UnsupportedShimInterfaceError(NotImplementedError):
    """Raised when a dummy has a shape the shim emitter can't handle:
    nested struct members, allocatable/pointer members, or a struct
    with no recorded layout."""


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
    """Identifiers a flat array dummy's static shape references that
    aren't scalar dummy args -- module vars the kernel used for extents
    (``tracer(nproma, n_zlev)`` -> ``nproma``, ``n_zlev``).

    Forwarded as by-value C args because a flat array dummy has no
    struct module to ``use`` them from -- otherwise gfortran fails with
    "Symbol 'n_zlev' has no IMPLICIT type".  Assumed-shape (``:``)
    dummies excluded (use the ``_d<i>`` dynamic path instead).  Returns
    distinct symbols in first-appearance order.
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
    """``True`` iff the shim can build a ``c_f_pointer`` alias for ``m``:
    scalar, static-shape array of scalar, or dynamic-shape array of
    scalar (``'?'`` entries, from allocatable/pointer members -- extents
    ride as separate ``<name>_<member>_d<i>`` C args).  ``False`` for a
    nested-struct member (recurse instead) or an unnamed Fortran type
    (``fortran_type == '??'``, no element type to spell).
    """
    if m.struct_name:
        return False
    if m.fortran_type == "??":
        return False
    return m.rank >= 0


def _is_value_record(iface: OriginalInterface, struct_name: str) -> bool:
    """``True`` iff ``struct_name`` is a flat fixed-size value record --
    every member a non-struct, static-shape leaf (e.g.
    ``t_cartesian_coordinates`` with ``x(3)``).

    An array of such a record is reconstructed element-wise
    (``arr(i)%x(j) = flat(i,j)``) because Fortran forbids whole-array
    ``arr%x`` descent ("two or more part references with nonzero
    rank").  A container record (multiple/nested/dynamic-shape members)
    returns ``False`` and is indexed at ``(1)`` and descended into.
    """
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
    """Walk ``st`` (and nested derived-type members), raise
    :class:`UnsupportedShimInterfaceError` on the first unhandleable leaf.
    ``path`` is the Fortran access path used in the error message."""
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
    """Append a ``use <mod>, only: ...`` line for every module a
    nested-derived-type member references, so ``type(<nested>)``
    resolves at compile time (the outer struct decl doesn't cover it)."""
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
    """``use <mod>, only: <struct_name>[, shape consts...]`` for the
    module defining ``struct_name`` -- member-shape constants (``NX``/
    ``NY``) must ride along so ``c_f_pointer`` calls can spell them."""
    for mod, syms in iface.used_modules.items():
        if struct_name in syms:
            return f"  use {mod}, only: {', '.join(syms)}"
    return ""


def _emit_flat_arg(a: OriginalArg, header_args: List[str], decls_value: List[str], decls_ptr: List[str],
                   decls_local: List[str], c_f_calls: List[str], call_args: List[str]):
    """Per-dummy split for a non-struct arg: scalar inputs by value,
    scalar outputs/arrays as ``c_ptr`` + ``c_f_pointer`` alias.
    Mutates the parallel lists in place.

    Dynamic-shape arrays take extents from ``<name>_d<i>`` by-value args
    declared BEFORE the pointer -- caller must pass dims before pointer
    (same convention as :func:`_emit_struct_members_recursive`)."""
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
        # Length-1 array alias for scalar I/O (feedback_scalar_io_convention).
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
    """Reconstruct an ARRAY of a value record (see :func:`_is_value_record`,
    e.g. ``t_cartesian_coordinates``) element-wise.  ``outer_rank`` is the
    record array's rank; ``inst_path`` is the Fortran instance to assemble.

    Per member ``v``, flat slot ``<flat>_<v>`` (rank ``outer_rank + v.rank``)
    carries the SoA companion, outer dims first -- exact inverse of the
    wrapper's ``arr(i1..)%v(j1..) = flat(i1.., j1..)`` gather.  ``intent``
    selects scatter (copy-in) / gather (copy-out).

    C-ABI order: per field, extents ``<flat>_<v>_d<i>`` immediately precede
    that field's pointer.  Must match ``emit_library``'s ``per_member_soa`` +
    ``dynamic_extents_abi`` per-leaf extent emission (~1651-1654/~1760-1764)
    so the two ABIs coincide leaf-for-leaf.  Fields share one runtime
    allocation, sized from the FIRST field's extents (all fields' extents
    are equal at runtime); that allocate runs in ``copy_in`` ahead of the
    scatter."""
    st = iface.struct_types[vt_name]
    alloc_ext_names = None  # first field's extents drive the shared allocate
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


#: Double-buffer lane source expr, ``<prefix>%<aor>(<sym>)%<leaf>`` (e.g.
#: ``p%prog(nnow)%rho``).  Never matches a plain AoS member (index there
#: rides ``$i`` placeholders, not a literal ``(<sym>)``).
_DBUF_OUTER_RE = re.compile(r'^(?P<prefix>.+)%(?P<aor>\w+)\((?P<sym>\w+)\)%(?P<leaf>.+)$')


def _build_dbuf_map(plan) -> dict:
    """Group the FlattenPlan's double-buffer lane recipes by
    ``(struct-instance prefix, AoR member)`` for per-time-level
    reconstruction.

    Returns ``{(prefix, aor): {sym: [leaf_info, ...]}}``; empty if
    ``plan`` is None or has no double-buffer lanes."""
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
    """Reconstruct an ICON double-buffer AoR member (``p%prog``) from the
    SDFG's per-time-level lane buffers (``prog(nnow)``/``prog(nnew)`` split
    into static ``p_prog_nnow_rho``/``p_prog_nnew_rho`` lanes).  Allocates
    the record array to cover every time-level index and populates each
    from its own C-ABI buffer.  ``syms`` maps index symbol -> leaf buffers."""
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
    """Walk ``st``'s members: emit a C-ABI slot + ``c_f_pointer`` alias +
    copy-in/copy-out per leaf; descend into nested-struct members with
    extended paths.  ``inst_path`` is the Fortran access path,
    ``flat_prefix`` the C-ABI naming root, ``intent`` inherited from the
    outermost dummy.

    ``shape_syms`` (see :func:`_free_shape_symbols`): a scalar member whose
    flat name is ALSO a flat array dummy's shape extent is forwarded ONCE,
    by value -- emitting it as a length-1 pointer instead would
    double-declare the name and give array shapes a pointer where a
    scalar extent is required."""
    for m in st.members:
        if m.struct_name:
            if m.rank == 0:
                # Scalar nested record: descend in place, no index/alloc.
                nested = iface.struct_types[m.struct_name]
                _emit_struct_members_recursive(iface, nested, f"{inst_path}%{m.name}", f"{flat_prefix}_{m.name}",
                                               intent, header_args, decls_value, decls_ptr, decls_local, c_f_calls,
                                               copy_in, copy_out, shape_syms)
            elif _is_value_record(iface, m.struct_name):
                # Array of a value record: scatter element-wise.
                _emit_value_record_array(iface, m.struct_name, m.rank, f"{inst_path}%{m.name}",
                                         f"{flat_prefix}_{m.name}", intent, header_args, decls_value, decls_ptr,
                                         decls_local, c_f_calls, copy_in, copy_out)
            elif dbuf_map and (inst_path, m.name) in dbuf_map:
                # ICON double-buffer AoR: bridge split into per-time-level lanes.
                _emit_double_buffer_member(inst_path, m.name, dbuf_map[(inst_path, m.name)], header_args, decls_value,
                                           decls_ptr, decls_local, c_f_calls, copy_in, copy_out)
            else:
                # Array of a container record: ICON ocean kernels are
                # single-patch, so allocate size 1 and descend into (1).
                copy_in.append(f"  allocate({inst_path}%{m.name}(1))")
                nested = iface.struct_types[m.struct_name]
                _emit_struct_members_recursive(iface, nested, f"{inst_path}%{m.name}(1)", f"{flat_prefix}_{m.name}",
                                               intent, header_args, decls_value, decls_ptr, decls_local, c_f_calls,
                                               copy_in, copy_out, shape_syms, dbuf_map)
            continue
        flat_name = f"{flat_prefix}_{m.name}"
        if m.rank == 0 and intent in ('in', '') and flat_name in shape_syms:
            # Also a flat-array-dummy extent: one shared by-value arg.
            header_args.append(flat_name)
            decls_value.append(f"  {m.fortran_type}, value :: {flat_name}")
            copy_in.append(f"  {inst_path}%{m.name} = {flat_name}")
            continue
        ptr_name = f"{flat_name}_p"
        is_dynamic = any(s in ('?', '*', ':') for s in m.shape)
        if is_dynamic:
            # Per dim: lower bound then extent, both by-value ints
            # (``<flat>_lb<i>``/``<flat>_d<i>``) ahead of the pointer.
            # Needed for arrays ICON allocates with non-default lower bound
            # (e.g. refinement-control index arrays at (min_rl:max_rl),
            # read at negative rl) -- defaulting lb to 1 broke end_block(-5).
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
            # Bridge can't distinguish POINTER vs ALLOCATABLE on fir.BoxType
            # extracts, so always ``allocate`` (works for both) + element
            # copy-in/copy-out rather than pointer-assign.  per_member_soa
            # no-pack contract still holds (only per-leaf pointers cross the
            # C ABI).
            #
            # Allocate at TRUE bounds ``(lb:lb+d-1)``, not default ``(1:d)``,
            # so lbound() matches offset_<member>_d<i>.  Whole-array copy from
            # the 1-based flat companion is by position, so member(lb) takes
            # flat(1).
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
    """Per-member split for a derived-type argument.

    The dummy becomes a local ``type(<struct>), target :: <name>``; each
    inline-flat leaf (transitively, through nested struct members) rides
    its own C-ABI slot ``<name>_..._<leaf>_p`` aliased via ``c_f_pointer``.
    Dynamic-shape extents come as ``<flat>_d<i>`` by-value args ahead of
    the pointer, matching the marshal-expanded leaf order on the outer
    SDFG's emit_call side.

    Per :attr:`OriginalArg.intent` (inherited through nested members):
    static-shape leaves copy-in/copy-out element-wise; dynamic-shape
    leaves pointer-assign ``<a>%..%<leaf> => <flat>`` in place (no copy,
    preserves the per_member_soa no-pack contract).
    """
    if a.rank > 0:
        # Array-of-record dummy: value record scatters element-wise;
        # container-record array has no path here -- reject loudly.
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


# SDFG dtype -> iso_c_binding Fortran type for by-value scalar C ABI args.
# Keep in sync with emit_library._sym2c so the two ABIs coincide.
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
    """Per ``(module, member, dtype, rank)``, extend the shim so the caller
    can write the INNER library's copy of ``<module>::<member>`` via the C
    ABI -- gfortran ships a per-library BSS copy of module vars, so an
    outer-library write doesn't reach this one.

    Scalar: by-value arg + ``use ... only: <member>__sink => <member>``
    write.  Rank-N array: pointer + ``c_f_pointer`` alias + whole-array
    copy.  Assignments land in ``copy_in``, running before the
    ``<entry>_dace`` call.
    """
    seen_use_aliases = set()
    for module, member, dtype, rank in module_symbol_forward:
        ftype = _MOD_FORWARD_SCALAR_FTYPE.get(dtype)
        if ftype is None:
            raise ValueError(f"bind_c_shim module_symbol_forward: unsupported dtype "
                             f"{dtype!r} for ``{module}::{member}``; extend "
                             f"``_MOD_FORWARD_SCALAR_FTYPE`` for new pass-by-value "
                             f"shapes.")
        # Same module may repeat -- collapse into one `only:` list per module.
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
            # Rank-N array: pointer + size() query alias (member is
            # static-shape) + whole-array assign.
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
    """Flat names of rank-0 struct members this shim takes as a
    ``c_ptr, value`` POINTER rather than BY VALUE.

    A caller marshalling a struct arg to this shim (``emit_library``
    ``per_member_soa``/``dynamic_extents_abi``) MUST pass a pointer for
    these, even if it holds the member as an SDFG symbol -- passing the
    raw value where the shim expects ``c_ptr`` reinterprets the int as
    an address.

    Mirrors :func:`_emit_struct_members_recursive`'s routing: only a
    read-only member that's ALSO a flat-array-dummy extent
    (``flat_name in shape_syms``) crosses by value; every other rank-0
    member is a pointer.
    """
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

    Scalar input: by value.  Scalar output: ``c_ptr`` aliased to a
    length-1 array (``feedback_scalar_io_convention``).  Array: ``c_ptr``
    aliased to the declared shape, with preceding scalar dims in C-ABI
    order.  Derived-type dummy with inline-flat members: one C-ABI slot
    per member, assembled into a local ``type(<struct>)`` instance,
    copied back out per member for ``out``/``inout``.

    Calls ``<entry>_dace(...)`` then ``<entry>_dace_finalize()`` to
    release the DaCe handle.  Raises
    :class:`UnsupportedShimInterfaceError` on a struct member shape it
    can't handle.
    """
    # Validate every struct dummy (transitively) has only handleable leaves.
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
    # Scalar struct member coinciding with a flat-array-dummy shape extent
    # forwards by value (shared), not as a length-1 pointer alias.
    shape_syms = set(_free_shape_symbols(iface))
    # ICON double-buffer AoR lanes, grouped by (struct-instance, AoR member)
    # so reconstruction can populate each time-level element from its lane.
    dbuf_map = _build_dbuf_map(plan)
    for a in iface.args:
        if a.struct_type is None:
            _emit_flat_arg(a, header_args, decls_value, decls_ptr, decls_local, c_f_calls, call_args)
            continue
        st = iface.struct_types[a.struct_type]
        _emit_struct_arg(a, st, iface, header_args, decls_value, decls_ptr, decls_local, c_f_calls, copy_in, copy_out,
                         call_args, shape_syms, dbuf_map)
        # Pull in the struct's module use-line plus every nested member's.
        u = _struct_module_use(iface, a.struct_type)
        if u and u not in seen_struct_uses:
            struct_use_lines.append(u)
            seen_struct_uses.add(u)
        _collect_nested_struct_modules(iface, st, struct_use_lines, seen_struct_uses)

    # Forward module-variable shape extents (``tracer(nproma, n_zlev)``) as
    # by-value C args, PREPENDED so they precede the array pointers that
    # need them.  Skipped if already an arg.  Struct shape constants ride
    # the struct's module use-line instead.
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

    # Forward module globals across the C ABI: under ELF+gfortran, each
    # shared library gets its OWN BSS copy of a used module's variables, so
    # an outer-library write doesn't reach this library's copy.  The shim
    # takes the value as a C arg and writes the INNER copy directly.
    # (Root cause: velocity dycore+ext e2e ASan ODR diagnostic.)
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
        # `, intrinsic` forces the real module even if a USE-imported
        # source stubs iso_c_binding.
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
