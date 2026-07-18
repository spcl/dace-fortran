"""Memlet subset construction, access-node caching, and indirect-index lifting.

``acc`` caches one access node per (state, name) so multi-tasklet reads/writes
in a state connect through one graph. ``build_memlet_index`` converts a bridge
``AccessInfo`` into a DaCe subset, offsetting Fortran 1-based indices to
0-based and resolving indirect-index expressions via ``collect_indirect``.
"""

import re
from types import SimpleNamespace

# Process-level (not per-SDFG) counter for unique '<arr>_at<gid>' names; avoids collisions across multi-file runs.
_INDIRECTION_GID_COUNTER = 0


def _next_indirection_gid() -> int:
    global _INDIRECTION_GID_COUNTER
    gid = _INDIRECTION_GID_COUNTER
    _INDIRECTION_GID_COUNTER += 1
    return gid


def iter_view_dim_map(view_dim_map):
    """Decode one ``section_alias`` ``view_dim_map`` entry: ``"_d<N>"`` =
    surviving dim (dummy-dim index N), else a dropped scalar's 1-based
    Fortran expr. Yields ``(src_dim, slot, dummy_dim | None)``; callers
    intentionally differ on 0- vs 1-based handling (``resolve_section_alias``
    keeps 1-based, the subscript/memset emitters convert inline) -- only
    this decode is shared.
    """
    for src_dim, slot in enumerate(view_dim_map):
        if slot.startswith('_d'):
            try:
                dummy_dim = int(slot[2:])
            except ValueError:
                dummy_dim = src_dim
            yield src_dim, slot, dummy_dim
        else:
            yield src_dim, slot, None


def resolve_object_member(builder, name: str):
    """Real flat descriptor for a member access on a whole-object pointer-rebind
    alias (``params_oce => v_params`` makes ``params_oce % a_veloc_v`` the same
    storage as ``v_params``'s ``a_veloc_v``), or ``None`` if not such an access.

    Resolves the base transitively through ``builder.object_aliases``, then
    either returns ``<src>_<member>`` or borrows the uniquely-flattened member
    of the same derived type from the bridge flatten plan. Returns ``None`` on
    ambiguity so the caller keeps the original name and a genuinely-missing
    descriptor still surfaces loudly.
    """
    if name in builder.arrays or name in builder.scalars or name in builder.symbols:
        return None
    aliases = vars(builder).get("object_aliases") or {}
    if not aliases:
        return None
    for base in sorted(aliases, key=len, reverse=True):
        if name == base or not name.startswith(base + "_"):
            continue
        member = name[len(base) + 1:]
        src = base
        seen = {src}
        while src in aliases and aliases[src] not in seen:
            src = aliases[src]
            seen.add(src)
        cand = f"{src}_{member}"
        if cand in builder.arrays or cand in builder.scalars or cand in builder.symbols:
            return cand
        hit = (vars(builder).get("object_alias_flat_members") or {}).get(member)
        if hit is not None and (hit in builder.arrays or hit in builder.scalars):
            return hit
        return None
    return None


def resolve_section_alias(builder, array_name: str, access):
    """If ``array_name`` is a trivial ``section_alias`` slice (full-range
    triplets + scalar drops only), return ``(source_name, spliced_access)``
    with the source's index list spliced via ``view_dim_map``; otherwise
    return ``(array_name, access)`` unchanged. Spliced indices stay Fortran
    1-based -- ``build_memlet_index`` offsets them uniformly.
    """
    # Whole-object rebind member: pure rename onto the real flattened storage;
    # indices unchanged (same rank/shape), only descriptor name/offset symbols differ.
    obj_real = resolve_object_member(builder, array_name)
    if obj_real is not None:
        return obj_real, access
    v = builder.arrays.get(array_name)
    if v is None or getattr(v, 'role', '') != 'section_alias':
        return array_name, access
    src = v.view_source
    if access is None:
        return src, access
    dummy_exprs = list(getattr(access, 'index_exprs', None) or [])
    dummy_vars = list(getattr(access, 'index_vars', None) or [])
    new_exprs, new_vars = [], []
    for _src_dim, slot, dummy_dim in iter_view_dim_map(v.view_dim_map):
        if dummy_dim is not None:
            new_exprs.append(dummy_exprs[dummy_dim] if dummy_dim < len(dummy_exprs) else '')
            new_vars.append(dummy_vars[dummy_dim] if dummy_dim < len(dummy_vars) else '')
        else:
            new_exprs.append(slot)
            new_vars.append('')
    spliced = SimpleNamespace(
        array_name=src,
        is_read=getattr(access, 'is_read', False),
        is_write=getattr(access, 'is_write', False),
        index_exprs=new_exprs,
        index_vars=new_vars,
    )
    return src, spliced


def resolve_full_dim_markers(view_subset, src_shape):
    """Replace ``":"`` full-dimension markers in a section subset with an
    explicit ``"0:<extent>"`` drawn from the parent array's SDFG shape.

    Dynamic ALLOCATABLE/POINTER bounds don't render in the bridge
    (``renderDesignateSubsetStrings`` emits bare ``":"``); using the real
    SDFG extent keeps the linking memlet scoped to the aliased slab instead
    of falling back to the whole array.
    """
    out = []
    for i, s in enumerate(view_subset):
        if s == ":":
            out.append(f"0:{src_shape[i]}" if i < len(src_shape) else s)
        else:
            out.append(s)
    return out


def cc_alias_view_spec(builder, name: str):
    """Synthesize a COMPLEX view spec for a complex-as-2-reals component alias
    (e.g. QE ``qvan2``'s ``REAL(8) :: qg(2, ngy)`` sequence-associated to a
    ``COMPLEX`` element): the bridge surfaces this as an inexpressible
    float-of-complex ``view_alias``, so recast it as a same-dtype COMPLEX view
    with the size-2 component axis dropped (handled per-access by the re/im
    mask). Returns a ``view_alias``-shaped namespace for the shared view-link
    code in ``acc``.

    Slab subset = alias base (``view_subset``, 0-based start) extended over
    the element extent (``shape_symbols`` minus the component dim), e.g.
    ``qgm(1:ngy, ijh)`` -> ``['0:dfftt_ngm', '(ijh)-1']``.
    """
    v = builder.complex_component_aliases[name]
    src_v = builder.arrays.get(v.view_source)
    base = [str(s) for s in (v.view_subset or [])]
    elem_ext = [str(s) for s in v.shape_symbols][1:]  # drop the size-2 component dim
    slab = []
    for j in range(len(base)):
        if j < len(elem_ext):
            slab.append(f"({base[j]}):({base[j]}) + ({elem_ext[j]})")
        else:
            slab.append(base[j])
    return SimpleNamespace(role='view_alias',
                           view_source=v.view_source,
                           view_subset=slab,
                           fortran_name=name,
                           shape=elem_ext,
                           dtype=(src_v.dtype if src_v is not None else v.dtype),
                           lower_bounds=list(v.lower_bounds)[1:])


def acc(builder, state, name: str):
    """Single access node for ``name`` in ``state``, reused across reads /
    writes.  Without this, every tasklet in the same state would fabricate
    its own disconnected access node, so a later read could not see the
    value produced by an earlier write in the same state.

    View-alias entries (Fortran storage-association reshape  --  see
    ``extract_vars::view_source`` / ``view_subset``) get an additional
    source->view linking memlet auto-installed the first time they're
    accessed in a state.  The link tells DaCe codegen which slab of
    the source array the view points at; subsequent reads / writes
    of the view in the same state pass through to the source.
    """
    # Trivial section-slice dummies (``role == 'section_alias'``) have
    # no SDFG descriptor  --  every access routes through the source array
    # with indices spliced via ``view_dim_map``.  Redirect the access-
    # node lookup to the source.
    v_alias = builder.arrays.get(name)
    if v_alias is not None and getattr(v_alias, 'role', '') == 'section_alias':
        return acc(builder, state, v_alias.view_source)
    # Whole-object rebind member: route the access node onto the real flattened
    # descriptor of the aliased object (``params_oce_a_veloc_v`` has no
    # descriptor of its own; ``p_phys_param_a_veloc_v`` is the real storage).
    obj_real = resolve_object_member(builder, name)
    if obj_real is not None:
        return acc(builder, state, obj_real)
    cache = getattr(state, '_hlfir_access', None)
    if cache is None:
        cache = {}
        state._hlfir_access = cache
    node = cache.get(name)
    if node is None:
        node = state.add_access(name)
        cache[name] = node
        v = builder.arrays.get(name)
        # Complex-as-2-reals component alias: re-cast the bridge's invalid
        # float-of-complex ``view_alias`` as a SAME-dtype COMPLEX view of the
        # spanned source slab, then fall through to the shared view-link code
        # (mirrors the ``bounds_remap_view`` synthesis below).
        if name in getattr(builder, 'complex_component_aliases', {}):
            v = cc_alias_view_spec(builder, name)
        # ``bounds_remap_view`` (multi-D POINTER remap of a 1D target,
        # e.g. ``p(1:M, 1:K) => arr1d``) needs the same source ->
        # view linking edge the rank-reinterpret ``view_alias`` path
        # uses.  Synthesise an equivalent VarInfo on the spot so the
        # shared code below handles both shapes uniformly.
        if v is not None and getattr(v, 'bounds_remap_view', False) \
                and v.bounds_remap_source \
                and v.bounds_remap_source in state.parent.arrays:
            # Prefer the surfaced source-SECTION subset (carries the
            # column offset, e.g. ``a[0:nrows, (c0)-1:c1]``) so the
            # original -> view linking memlet covers exactly the aliased
            # slab.  Fall back to the whole-array sentinel for the
            # rank-increasing embox case (``p(1:M, 1:K) => arr1d``) where
            # the view's own strides encode the reshape and there is no
            # source section to carry.
            src_subset = list(v.bounds_remap_source_subset) or [""]
            v = SimpleNamespace(role='view_alias',
                                view_source=v.bounds_remap_source,
                                view_subset=src_subset,
                                fortran_name=v.fortran_name)
        if v is not None and getattr(v, 'role', '') == 'view_alias' \
                and v.view_source and v.view_source in state.parent.arrays:
            from dace import Memlet
            src = v.view_source
            src_node = cache.get(src) or state.add_access(src)
            cache.setdefault(src, src_node)
            # Canonical DaCe view linking: source AccessNode ->
            # view ViewAccessNode via the ``views`` connector
            # (see d-face/tests/numpy/reshape_test.py).  The view's
            # own descriptor (shape + strides set in
            # ``descriptors.py``) handles the reinterpretation;
            # the linking memlet just carries the source name with
            # no explicit subset (defaults to full extent on both
            # sides, matching the view's storage span).
            if len(v.view_subset) == 1 and v.view_subset[0] == "":
                # Whole-array rank reinterpretation
                # (``ssor_tv(N)`` 1D -> ``buts_tv(5, M, K)`` 3D).
                # Build src_subset spanning the source's full flat
                # storage and other_subset spanning the view's
                # full multi-D shape.  Element counts match
                # (5445 == 5*33*33) so DaCe's dimensionality
                # check is satisfied even with the rank difference.
                src_dims = [str(d) for d in state.parent.arrays[src].shape]
                src_subset = ", ".join(f"0:{d}" for d in src_dims)
                view_dims = [str(d) for d in state.parent.arrays[name].shape]
                view_subset = ", ".join(f"0:{d}" for d in view_dims)
                state.add_edge(src_node, None, node, 'views',
                               Memlet(data=src, subset=src_subset, other_subset=view_subset))
            else:
                # Section-reshape view: source-side subset
                # describes which slab of ``src`` the view covers;
                # view-side subset spans the view's own shape.
                src_shape = [str(d) for d in state.parent.arrays[src].shape]
                src_subset = ", ".join(resolve_full_dim_markers(v.view_subset, src_shape))
                view_dims = [str(d) for d in state.parent.arrays[name].shape]
                view_subset = ", ".join(f"0:{d}" for d in view_dims)
                state.add_edge(src_node, None, node, 'views',
                               Memlet(data=src, subset=src_subset, other_subset=view_subset))
        elif v is not None and getattr(v, 'role', '') == 'view_alias':
            # A view with no resolvable source would be emitted as a bare
            # AccessNode and only fail much later, at SDFG validation, as an
            # opaque "Ambiguous or invalid edge to/from a View access node"
            # (exchange_data_r3d's ``send_ptr`` did exactly this when the
            # rebind trace stopped at the inlined ``recv`` dummy's declare).
            # Fail here, at the emission point, with the actual names.
            raise ValueError(f"view_alias '{name}' has no usable view source: "
                             f"view_source={v.view_source!r} is "
                             f"{'unset' if not v.view_source else 'not a registered array'} "
                             f"(state '{state.label}'). The bridge's rebind trace must resolve "
                             f"the view to a registered descriptor -- see extract_vars.cpp's "
                             f"pointer-view source walk.")
    return node


def get_access(accesses: list, array_name: str, is_read: bool):
    """Return the matching ``AccessInfo`` (exact read/write match preferred)."""
    for ac in accesses:
        if ac.array_name == array_name:
            if is_read and ac.is_read:
                return ac
            if not is_read and ac.is_write:
                return ac
    for ac in accesses:
        if ac.array_name == array_name:
            return ac
    return None


def _reserved_rewrite(name):
    """Map a single identifier to its ``program_<name>`` form when it
    collides with a sympy attribute (``im`` -> ``sympy.im`` is a
    ``FunctionClass``; arithmetic against a ``Symbol`` then fails).
    See ``builder.__init__._RESERVED_DACE_NAMES`` for the full set.
    Imported lazily to avoid a circular import at module load."""
    from dace_fortran.builder import _RESERVED_DACE_NAMES, _DACE_NAME_PREFIX
    if name in _RESERVED_DACE_NAMES:
        return _DACE_NAME_PREFIX + name
    return name


def rename_iters(expr, iter_map):
    """Whole-word substitution of Fortran iter names with their
    uniquified DaCe counterparts.  Word boundaries protect against
    partial matches inside identifiers (``i`` shouldn't rewrite
    inside ``input1``).  Pass-through for ``None`` / non-string.

    Does NOT apply the sympy-reserved-name rewrite -- this helper is
    used by ``emit_tasklet`` to rewrite the tasklet body code, where
    a Fortran identifier ``test`` must stay bare so the per-tasklet
    ``_rewrite_read_connectors`` can find it and map it to its input
    connector name; the SDFG-wide ``sdfg.replace("test",
    "program_test")`` later in ``_rename_reserved_collisions``
    handles the rename consistently for tasklet bodies + data names.
    Memlet subsets need the rewrite earlier (sympify runs at memlet
    construction time, before ``_rename_reserved_collisions``) --
    those sites call ``apply_reserved`` on the result."""
    if not iter_map or not isinstance(expr, str):
        return expr
    return re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: iter_map.get(m.group(1), m.group(1)), expr)


def apply_reserved(expr):
    """Whole-word rewrite of sympy-reserved Fortran identifiers
    (``im`` -> ``program_im`` etc.) inside a memlet-subset string.
    The SDFG-wide ``_rename_reserved_collisions`` only sees
    identifiers registered as arrays / symbols; loop counters and
    expression-internal names slip through, so the subset string
    is the right place to catch them before sympify."""
    if not isinstance(expr, str):
        return expr
    return re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: _reserved_rewrite(m.group(1)), expr)


def _remap_token(token, iter_map):
    """Rewrite a single subscript token through ``iter_map``.  Three
    forms collapse to one helper: integer literals pass through;
    arithmetic / parenthesised expressions go through whole-word
    substitution; bare identifiers do a direct dict lookup.  In
    every case the reserved-sympy-name rewrite (``im`` -> ``program_im``)
    is applied as a final guard so the memlet's sympified subset
    stays on plain Symbols.  This helper is ONLY used by memlet-subset
    paths -- the tasklet-body rewrite calls bare ``rename_iters``."""
    token = token.strip()
    if token.lstrip('-').isdigit():
        return token
    if any(op in token for op in "+-*/") or token.startswith("("):
        return apply_reserved(rename_iters(token, iter_map))
    mapped = iter_map.get(token, token) if iter_map else token
    return _reserved_rewrite(mapped)


def _format_offset_subset(arr, parts):
    """Wrap a per-dim expression list in the uniform offset-symbol
    form: ``arr[(p0) - offset_arr_d0, (p1) - offset_arr_d1, ...]``."""
    items = ", ".join(f"({p}) - {_offset_token(arr, d)}" for d, p in enumerate(parts))
    return f"{arr}[{items}]"


def sdfg_is_len1_array(sdfg, name: str) -> bool:
    """True when ``name`` is registered on the SDFG as a length-1 ``Array``
    rather than a ``Scalar``.  The bridge keeps some logical SCALARS as
    ``(1,)``-Arrays (``descriptors.py``): an ``intent(out)`` / ``inout``
    scalar -- the caller needs a writable buffer, e.g. a module global like
    ``kunit`` / ``npool`` the kernel updates -- and a complex ``intent(in)``
    scalar (pass-by-pointer ABI).  Such a var is a POINTER in the generated
    C, so a CODE-BLOCK reference (interstate-edge assignment / condition)
    must read it as ``name[0]``; a bare ``name`` leaks the pointer
    (``int* / int*``).  Tasklets already deref correctly via the memlet
    ``[0]``; only the bridge-built code-block expression strings need the
    explicit subscript.  Keyed off the real descriptor (``dtype is array``)
    rather than the intent classification, so any length-1 Array is caught."""
    import dace
    s = sdfg
    while s is not None:
        d = s.arrays.get(name)
        if d is not None:
            return isinstance(d, dace.data.Array) and tuple(d.shape) == (1, )
        # A nested-SDFG code block can reference a parent-scope length-1 Array
        # (a module global like ``kunit`` lives on the top SDFG); walk up.
        s = getattr(s, 'parent_sdfg', None)
    return False


def deref_len1_array_scalars(sdfg, expr: str) -> str:
    """Token-walk a code-block expression STRING and rewrite every bare
    reference to a length-1 Array scalar (see :func:`sdfg_is_len1_array`) as
    ``name[0]``.  Used to repair interstate-edge assignment values /
    conditions the bridge built before the descriptor was known to be a
    length-1 Array (so a logical scalar kept as a ``(1,)``-Array -- e.g. the
    inout module globals ``kunit`` / ``npool`` -- reads its element, not the
    bare pointer).  Already-subscripted occurrences (``name[...]``) and names
    not on the SDFG (tasklet connectors, symbols, locals) are left alone, so
    the rewrite is idempotent and connector-safe."""
    if not isinstance(expr, str) or sdfg is None or not expr:
        return expr
    out = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isalpha() or ch == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            tok = expr[i:j]
            already_subscripted = j < len(expr) and expr[j] == '['
            attr_access = i > 0 and expr[i - 1] == '.'
            if not already_subscripted and not attr_access and sdfg_is_len1_array(sdfg, tok):
                out.append(f"{tok}[0]")
            else:
                out.append(tok)
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def find_array_subscripts(expr, names):
    """Generator yielding ``(start, end, arr_name, parts)`` for each
    top-level ``<arr>[...]`` substring in ``expr`` whose ``<arr>`` is in
    ``names``.  Walks brackets balanced (handles nested subscripts
    like ``a[idx[i],j]``) and splits the inner range on top-level
    commas only.  Replaces the brittle ``^(\\w+)\\[([^\\]]*)\\]$``
    regex used by indirect_to_dace / indirect_host."""
    n = len(expr)
    i = 0
    while i < n:
        m = re.match(r'([A-Za-z_]\w*)\[', expr[i:])
        if not m:
            i += 1
            continue
        arr = m.group(1)
        if arr not in names:
            i += 1
            continue
        start = i
        inner_start = i + len(arr) + 1
        depth = 1
        j = inner_start
        while j < n and depth > 0:
            ch = expr[j]
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return  # unbalanced; bail
        inner = expr[inner_start:j]
        # Split top-level commas only.
        parts, d, sp = [], 0, 0
        for k, ch in enumerate(inner):
            if ch in '([{':
                d += 1
            elif ch in ')]}':
                d -= 1
            elif ch == ',' and d == 0:
                parts.append(inner[sp:k].strip())
                sp = k + 1
        parts.append(inner[sp:].strip())
        yield (start, j + 1, arr, parts)
        i = j + 1


def indirect_host(expr):
    """Given ``edge_idx[jc,1]`` return ``edge_idx``; empty for non-indirect.
    Robust to nested brackets via the bracket-balanced walker."""
    if not isinstance(expr, str) or '[' not in expr:
        return ""
    m = re.match(r'^([A-Za-z_]\w*)\[', expr)
    return m.group(1) if m and expr.endswith(']') else ""


def collect_indirect(builder, assigns: list) -> dict:
    """Walk every access in ``assigns`` and mint a fresh SDFG symbol for
    each distinct *inline* indirect index expression -- recursively, so
    nested forms like ``idx1[idx2[idx3[i]]]`` mint one symbol per
    indirection level.  Returns a map from the Fortran-style expression
    (``edge_idx[jc,1]``) to the symbol name.

    Iteration order in the returned dict matters: ``dict`` preserves
    insertion order, and we insert innermost-first so the caller can
    materialise interstate-edge assignments in the same order without
    forward references.

    Naming: ``<arr>_at<gid>`` -- the prefix carries the source array's
    Fortran name so the SDFG dump shows which load the symbol holds; the
    process-level monotonic ``gid`` disambiguates same-expression-different-
    call-site without us having to normalise the inner expression.
    """
    out: dict[str, str] = {}

    def _intern_recursive(expr: str):
        """Visit ``expr`` and intern any ``<arr>[...]`` substring with
        ``<arr>`` in ``builder.arrays``, innermost-first.  An expression
        like ``idx1[idx2[i]]`` produces two entries: first ``idx2[i]``
        (the inner load), then ``idx1[idx2[i]]`` (the outer load).
        """
        if not isinstance(expr, str) or '[' not in expr:
            return
        for start, end, arr, parts in find_array_subscripts(expr, builder.arrays):
            # Recurse into each part FIRST so inner indirections are
            # interned ahead of the enclosing one.
            for part in parts:
                _intern_recursive(part)
            sub = expr[start:end]
            if sub not in out:
                out[sub] = f"{arr}_at{_next_indirection_gid()}"

    for a in assigns:
        for ac in a.accesses:
            for expr in getattr(ac, 'index_exprs', None) or []:
                _intern_recursive(expr)
    return out


def materialize_indirect_view_sources(builder, state, indirect_syms: dict) -> None:
    """Install the ``source -> view`` linking memlet for every ``view_alias``
    array that an INTERSTATE-edge indirection assignment reads.

    :func:`collect_indirect` mints ``<arr>_at<gid>`` symbols whose assignments
    ride interstate edges (``emit_cfg``'s ``sym_<sym>`` states).  An interstate
    edge is not a state, so :func:`acc` -- which auto-installs a ``view_alias``'s
    source->view link the first time the view is touched IN a state -- never runs
    for those reads.  The view descriptor then reaches codegen with NO ``views``
    edge, so what it aliases is undefined; DaCe's allocation-lifetime pass
    (which synthesises a placeholder ``AccessNode`` for arrays referenced only by
    interstate-edge free symbols) calls ``get_view_edge`` on that placeholder and
    raises ``KeyError``, since the placeholder is in no state.

    ICON's real ``mo_velocity_advection`` hits this: the inlined
    ``cells2verts_scalar_ri_lib`` / ``rot_vertex_ri_lib`` rebind
    ``iidx => vert_cell_idx`` / ``iblk => vert_cell_blk`` (POINTER locals onto
    TARGET dummies) and then read ``iidx(jv, jb, 1..6)`` purely as inline indirect
    indices -- so the views were never touched from a state.

    Touching each ``view_alias`` source through :func:`acc` HERE -- in ``state``,
    which PRECEDES the ``sym_*`` states -- installs the link and gives the view a
    real AccessNode ahead of every interstate reference.  Order matters: DaCe
    records a state's ``data_nodes()`` before that state's edge-referenced
    placeholders and walks states topologically, so the real node is the FIRST
    recorded instance and ``get_view_edge`` resolves against it.
    """
    for expr in indirect_syms:
        for _start, _end, arr, _parts in find_array_subscripts(expr, builder.arrays):
            v = builder.arrays.get(arr)
            if v is not None and getattr(v, 'role', '') == 'view_alias':
                acc(builder, state, arr)


def _offset_token(arr: str, dim: int) -> str:
    """The per-axis offset-symbol name every memlet subtracts."""
    return f"offset_{arr}_d{dim}"


def array_read_to_dace_expr(builder, assign_node, iter_map: dict, sdfg=None) -> str:
    """Render a scalar-target assign's RHS as a DaCe interstate-edge
    expression, lifting EVERY array read to the uniform offset-symbol
    subscript form (``arr[(idx) - offset_arr_d<i>, ...]``).  Used to lift
    the assign onto an interstate edge so the value becomes a live SDFG
    symbol the consuming memlet can index by.

    The bridge emits ``expr`` with bare array names (``(dims * dims) + 1``)
    and a parallel, left-to-right-ordered ``accesses`` list (``dims(1)``,
    ``dims(2)``).  Walk ``expr`` and replace each bare array-name token
    with its subscript form, consuming the read accesses in order, so a
    COMPOUND RHS keeps all its terms.  The previous single-read form
    silently dropped everything past the first read (``k = dims(1)*
    dims(2)+1`` collapsed to ``dims(1)``), making a promoted index/size
    symbol wrong.  Falls back to ``expr`` when the RHS has no array read."""
    reads = [ac for ac in assign_node.accesses if ac.is_read and ac.array_name in builder.arrays]
    expr = assign_node.expr
    # Fast path: nothing to subscript (no array reads, and -- absent the SDFG
    # to consult for length-1 Array scalars -- no deref to add).
    if not reads and sdfg is None:
        return expr
    out = []
    ri = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isalpha() or ch == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            tok = expr[i:j]
            # A name already followed by ``[`` is subscripted -- leave it;
            # only a bare array-name occurrence maps to a read access (in
            # the order the bridge lists them).
            already_subscripted = j < len(expr) and expr[j] == '['
            if not already_subscripted and ri < len(reads) and tok == reads[ri].array_name:
                ac = reads[ri]
                ri += 1
                # Section-alias dummy (``igkp`` <- ``igk_exx(:, current_k)``):
                # resolve to the SOURCE array + spliced index list so the
                # interstate-edge read references real storage.  The alias has
                # no descriptor / ``offset_<alias>_d<i>`` symbols of its own
                # (``descriptors.py`` skips it), so emitting the bare alias
                # name here leaks ``igkp`` + ``offset_igkp_d0`` as unresolved
                # free symbols.  QE ``init_us_2``'s ``iv = igk_(ig)``
                # value-symbol read, reached after ``add_nlxx_pot`` inlines it.
                eff_name, eff_ac = resolve_section_alias(builder, ac.array_name, ac)
                parts = [_remap_token(raw, iter_map) for raw in (eff_ac.index_exprs or [])]
                out.append(_format_offset_subset(eff_name, parts))
            elif not already_subscripted and sdfg_is_len1_array(sdfg, tok):
                # A logical scalar kept as a length-1 Array (inout/out module
                # global like ``kunit`` / ``npool``, or a complex intent(in)):
                # it is a pointer in C, so the code-block expression must read
                # the single element ``tok[0]`` rather than leak the bare
                # pointer (``int* / int*`` compile error).
                out.append(f"{tok}[0]")
            else:
                out.append(tok)
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _rewrite_inner_indirects(part: str, indirect_syms: dict) -> str:
    """In-place substitute every ``<arr>[...]`` substring of ``part`` with
    its minted symbol from ``indirect_syms``.  Recurses through the part
    walker so a part like ``idx2[idx3[i]]`` collapses to whichever
    minted symbol covers it (innermost first, then the outer).
    Returns ``part`` unchanged if no nested indirect appears.
    """
    if not isinstance(part, str) or '[' not in part:
        return part
    # Walk inside-out: keep replacing the first match whose substring is
    # in ``indirect_syms`` until no more replacements are possible.  We
    # don't reuse ``find_array_subscripts`` since we need indexes into
    # ``part`` (not into a parent expression) for slicing.
    arr_names = set(indirect_syms.keys())
    # Sort longest-first so a ``a[b[i]]`` form picks the outer first only
    # after the inner ``b[i]`` has been substituted.  But since we scan
    # innermost-first via the bracket walker each pass, longest doesn't
    # actually matter -- still, keep deterministic ordering.
    changed = True
    out = part
    while changed:
        changed = False
        # Iterate through every <arr>[...] substring in out, replace the
        # first whose exact substring is interned.
        for st in range(len(out)):
            if out[st] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_':
                continue
            # Try to match an array name starting at st.
            j = st
            while j < len(out) and (out[j].isalnum() or out[j] == '_'):
                j += 1
            if j >= len(out) or out[j] != '[':
                continue
            # Walk balanced brackets to find end.
            depth = 1
            k = j + 1
            while k < len(out) and depth > 0:
                if out[k] == '[':
                    depth += 1
                elif out[k] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                break
            sub = out[st:k + 1]
            if sub in indirect_syms:
                out = out[:st] + indirect_syms[sub] + out[k + 1:]
                changed = True
                break  # restart the scan from the beginning
    return out


def indirect_to_dace(builder, expr: str, iter_map: dict, indirect_syms: dict | None = None) -> str:
    """Convert ``arr[i,j]`` (Fortran 1-based) into the uniform offset-
    symbol DaCe subscript form.  Robust to nested brackets via the
    bracket-balanced walker.

    When ``indirect_syms`` is supplied, every nested ``<inner>[...]``
    substring inside an index part is first replaced by its minted
    symbol -- so ``idx1[idx2[i]]`` becomes ``idx1[(idx2_at2) - offset_idx1_d0]``
    rather than dragging the raw inner ``idx2[i]`` text into the SDFG
    symbolic expression (where DaCe's sympy parser misreads ``[]`` and
    falls back to a function-call shape that the C++ codegen can't
    compile).
    """
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    # Single full-string match -- the typical inline-indirection shape.
    if len(matches) == 1:
        start, end, arr, parts = matches[0]
        if start == 0 and end == len(expr):
            if indirect_syms:
                parts = [_rewrite_inner_indirects(p, indirect_syms) for p in parts]
            # Section-alias dummy read on an interstate edge: resolve to the
            # source array + spliced dim_map (same gap as
            # ``array_read_to_dace_expr`` -- the alias has no offset symbols).
            v = builder.arrays.get(arr)
            if v is not None and getattr(v, 'role', '') == 'section_alias':
                _src, _sp = resolve_section_alias(builder, arr,
                                                  SimpleNamespace(index_exprs=parts, index_vars=[''] * len(parts)))
                arr, parts = _src, list(_sp.index_exprs)
            return _format_offset_subset(arr, [_remap_token(p, iter_map) for p in parts])
    return expr


def build_memlet_index(builder, array_name: str, access, iter_map: dict, indirect_syms: dict = None) -> str:
    """Build a memlet subset using the uniform offset-symbol form.

    For every dim of ``array_name``, the resulting subset token is
    ``(<fortran-1-based-expr>) - offset_<arr>_d<i>``  --  the offset symbol
    was declared in ``add_descriptors`` and gets folded by
    ``sdfg.specialize`` at the end of ``build()`` (default value ``1``
    collapses the form to ``expr - 1``).  Indirect-index symbols are
    used in place of the index_expr but otherwise follow the same
    form.

    Constants stay outside the subtraction: ``A(3)`` produces
    ``3 - offset_A_d0`` (sympy folds to ``2`` once specialise runs).

    Arrays not in ``builder.arrays`` (struct members the bridge
    synthesises ad hoc, etc.) fall back to a literal ``- 1`` so the
    memlet still validates  --  this matches the pre-symbolic behaviour
    for those cases and avoids a missing-symbol crash at specialise
    time.
    """
    indirect_syms = indirect_syms or {}
    arr = builder.arrays.get(array_name)
    if access is None:
        return ""
    exprs = list(access.index_exprs) if access.index_exprs else []
    ivars = list(access.index_vars)
    rank = max(len(ivars), len(exprs))

    has_offset_sym = arr is not None
    parts = []
    for dim in range(rank):
        v = ivars[dim] if dim < len(ivars) else ""
        expr = exprs[dim] if dim < len(exprs) else v
        offset_sym = f"offset_{array_name}_d{dim}" if has_offset_sym else "1"

        # Substitute every EMBEDDED indirect access in expr with its interned
        # symbol so a dim like ``(ikidx[je,2,jk,jb] - 1)`` -- arithmetic
        # wrapping an indirect read -- renders as ``(ikidx_at123 - 1)`` and
        # doesn't leak the nested-bracket commas into the memlet subset.  The
        # exact-match early-return below still catches the pure-indirect case
        # (``ikidx[...]`` with no surrounding arithmetic).
        if '[' in expr and indirect_syms:
            while True:
                replaced = False
                for sub_start, sub_end, _arr, _parts in find_array_subscripts(expr, builder.arrays):
                    sub = expr[sub_start:sub_end]
                    if sub in indirect_syms:
                        expr = expr[:sub_start] + indirect_syms[sub] + expr[sub_end:]
                        replaced = True
                        break  # positions invalidated; rescan
                if not replaced:
                    break

        # Indirect: substitute the minted symbol that holds the
        # Fortran 1-based runtime value, then offset uniformly.
        if '[' in expr and expr in indirect_syms:
            tok = indirect_syms[expr]
            parts.append(f"({tok}) - {offset_sym}")
            continue

        # Closed-form arithmetic: remap the iter names through the
        # current LoopRegion's uniquified iter_map, then offset.
        if any(op in expr for op in "+-*/") or expr.startswith("("):
            parts.append(f"({apply_reserved(rename_iters(expr, iter_map))}) - {offset_sym}")
            continue

        # Constant literal: keep as-is, offset symbolically (sympy
        # folds it to the right value after specialise).
        if expr.lstrip('-').isdigit():
            parts.append(f"{expr} - {offset_sym}")
            continue

        # Bare iter name: remap through iter_map, then offset.
        #
        # ``v`` is the bridge-side ``index_vars[dim]`` value  --
        # produced by ``resolveIndex(idx)`` which returns:
        #   * the loop-iter name for a tracked block-arg iter
        #     (``i``, ``je``, ...); ``iter_map`` folds the SSA rename.
        #   * ``traceToDecl(idx)`` as a fallback for everything
        #     else, including ``fir.load %dgt(%c)`` where ``%dgt``
        #     designates a flattened struct-member array  --  that
        #     trace returns the WHOLE array's name (``ind_indices``
        #     for ``fir.load %ind_indices_decl(%c1)``).
        #
        # The whole-array name is NOT a valid memlet index.  The
        # authoritative form lives on ``index_exprs[dim]`` (``expr``):
        # the bridge has already lifted the load to either a
        # ``__sym_<arr>_<n>`` symbol (constant-indexed read of a
        # read-only array) or to ``<arr>[idx]`` form (which the
        # caller has further folded via the indirect machinery).
        #
        # Defaulting the iter_map fallback to ``expr`` instead of
        # ``v`` keeps tracked iters working (iter_map.get(v, ...)
        # finds them) while letting non-iter v values fall through
        # to the richer ``expr`` rendering.  See the matching
        # ``internPosSymbol`` mutability gate in
        # ``bridge/ast/assigns.cpp``  --  the two together close the
        # ``arr(struct_member(const))`` indirect-index path
        # exercised by ``long_tasklet_test``.
        uid = iter_map.get(v, expr)
        # Shield sympy-reserved bare names (``im``, ``re``, ...) --
        # iter_map doesn't carry them, so the dict lookup returned
        # ``expr`` unchanged.  Without the rewrite, the memlet's
        # sympified subset would see ``sympy.im - Symbol(...)`` ->
        # ``FunctionClass - Symbol`` TypeError.
        uid = _reserved_rewrite(uid)
        parts.append(f"{uid} - {offset_sym}")

    return ", ".join(parts)
