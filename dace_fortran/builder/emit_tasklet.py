"""Tasklet emission.  ``emit_tasklet``: per-occurrence connector/memlet per
array-read occurrence (else repeated reads of the same array collapse onto one
connector and silently miscompute).  ``emit_scalar_assign``: flat fast path for
plain scalar assigns (``i = i + 1``, ``c = 0.5``).  ``assign_reads_array``
is the predicate ``emit_assign`` uses to pick between them.
"""

import re

from dace import Memlet

from dace_fortran.builder.access import (acc, build_memlet_index, get_access, indirect_host, rename_iters,
                                         resolve_object_member, resolve_section_alias)

# Excludes the imaginary-unit suffix of a complex literal (``1j`` in
# ``(0.0) + 1j*(0.0)``): negative lookbehind drops idents starting right after
# a digit/``.``, so a real scalar named ``j`` is still matched but ``1j``'s
# ``j`` isn't mistaken for it (else a spurious ``_in_j`` connector).
_IDENT_RE = re.compile(r'(?<![0-9.])[a-zA-Z_]\w*')


def _ident_tokens(expr: str) -> set:
    """Identifier tokens in ``expr``, excluding complex-literal imaginary
    units (see ``_IDENT_RE``)."""
    return set(_IDENT_RE.findall(expr))


def _is_len1_scalar_view(builder, nm: str) -> bool:
    """True when ``nm`` is a length-1 ``view_alias`` (a scalar POINTER rebind
    ``tmp => x`` lowered as a length-1-array View).  Reads/writes like a scalar,
    so emit paths wire it as ``_in_<nm>``/``<nm>[0]``, not an indexed occurrence."""
    a = builder.arrays.get(nm)
    return a is not None and getattr(a, 'role', '') == 'view_alias' \
        and list(a.shape_symbols) == ['1']


def _view_link_spec(builder, state, target: str):
    """Resolve ``(src, src_subset, view_subset)`` for a View ``target``'s source
    linking memlet, or ``None`` if not a View.  Normalises all three View flavours
    (complex-component alias, ``bounds_remap_view``, plain ``view_alias``) so the
    read-link and write-back paths agree on the same subsets."""
    if target in getattr(builder, 'complex_component_aliases', {}):
        from dace_fortran.builder.access import cc_alias_view_spec
        v = cc_alias_view_spec(builder, target)
    else:
        v = builder.arrays.get(target)
    if v is None:
        return None
    if getattr(v, 'bounds_remap_view', False) and v.bounds_remap_source \
            and v.bounds_remap_source in state.parent.arrays:
        from types import SimpleNamespace
        v = SimpleNamespace(role='view_alias',
                            view_source=v.bounds_remap_source,
                            view_subset=list(v.bounds_remap_source_subset) or [""],
                            fortran_name=v.fortran_name)
    if getattr(v, 'role', '') != 'view_alias':
        return None
    if not v.view_source or v.view_source not in state.parent.arrays:
        return None
    src = v.view_source
    if len(v.view_subset) == 1 and v.view_subset[0] == "":
        src_dims = [str(d) for d in state.parent.arrays[src].shape]
        src_subset = ", ".join(f"0:{d}" for d in src_dims)
    else:
        from dace_fortran.builder.access import resolve_full_dim_markers
        src_shape = [str(d) for d in state.parent.arrays[src].shape]
        src_subset = ", ".join(resolve_full_dim_markers(v.view_subset, src_shape))
    view_dims = [str(d) for d in state.parent.arrays[target].shape]
    view_subset = ", ".join(f"0:{d}" for d in view_dims)
    return src, src_subset, view_subset


def _ensure_view_writeback_link(builder, state, write_node, target: str):
    """Add the missing view -> source writeback link when a fresh write-side
    access node is created for a view alias (else ``get_view_edge`` sees no
    edge and validation fails).  Two rules keep it happy: use a FRESH source
    node, not the cached one (else ``src->view_read->tasklet->view_write->src``
    cycles); and drop ``target`` from the per-state cache after, so later reads
    mint a new read-side view instead of reusing this write node."""
    spec = _view_link_spec(builder, state, target)
    if spec is None:
        return
    src, src_subset, view_subset = spec
    src_node = state.add_access(src)
    cache = getattr(state, '_hlfir_access', None)
    if cache is not None:
        cache[src] = src_node
        cache.pop(target, None)
    state.add_edge(write_node, None, src_node, None, Memlet(data=src, subset=src_subset, other_subset=view_subset))


def _ensure_view_read_link(builder, state, read_node, target: str):
    """Install the source -> view linking memlet on a GIVEN read node of View
    ``target`` (read-direction counterpart of ``_ensure_view_writeback_link``).
    Needed when a library node's input is a View (e.g. in-place FFT over
    ``bounds_remap_view``): sharing ``acc()``'s cached read node with the
    matching write would self-cycle, so this uses a FRESH node on both ends."""
    spec = _view_link_spec(builder, state, target)
    if spec is None:
        return
    src, src_subset, view_subset = spec
    src_node = state.add_access(src)
    state.add_edge(src_node, None, read_node, 'views', Memlet(data=src, subset=src_subset, other_subset=view_subset))


def assign_reads_array(assign_node, arrays: dict) -> bool:
    """True iff any ``accesses`` entry on ``assign_node`` reads an array.
    Promotes a nominally-scalar assign (``s = d(i) + 1``) onto the
    per-occurrence-connector tasklet path so the read gets a real memlet."""
    for ac in assign_node.accesses:
        if ac.is_read and ac.array_name in arrays:
            return True
    return False


def _rewrite_read_connectors(code: str, sorted_tokens, scalar_reads, array_occ: dict) -> str:
    """Replace each read reference in a tasklet RHS with its per-occurrence
    input connector: scalar ``<name>`` -> ``_in_<name>``; Nth array occurrence
    ``<name>[...]`` -> ``_in_<name>_<N>`` with its balanced ``[...]`` consumed
    (leftover subscript trips DaCe's dimension validator).  ``sorted_tokens``
    must be longest-first so a short name can't shadow a longer one; brackets
    are walked by hand since ``re`` can't balance them."""
    for nm in sorted_tokens:
        if nm in scalar_reads:
            code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)
            continue
        new_chunks = []
        cursor = 0
        pat = re.compile(rf'\b{re.escape(nm)}\b')
        for m in pat.finditer(code):
            start = m.start()
            end = m.end()
            # If the very next char is '[', consume the balanced [...].
            if end < len(code) and code[end] == '[':
                depth = 1
                j = end + 1
                while j < len(code) and depth > 0:
                    ch = code[j]
                    if ch in '([{':
                        depth += 1
                    elif ch in ')]}':
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if depth == 0:
                    end = j + 1
            new_chunks.append(code[cursor:start])
            n = array_occ[nm]
            array_occ[nm] += 1
            new_chunks.append(f"_in_{nm}_{n}")
            cursor = end
        new_chunks.append(code[cursor:])
        code = ''.join(new_chunks)
    return code


def emit_tasklet(builder, state, assign_node, idx: int, iter_map: dict, indirect_syms: dict = None):
    """One Tasklet per array assignment.  Each RHS occurrence of an array
    (e.g. ``e_bln(jc,1)*z + e_bln(jc,2)*z``) gets its own input connector/memlet;
    collapsing them onto one connector would silently compute a wrong result."""
    indirect_syms = indirect_syms or {}
    accesses = assign_node.accesses

    tokens = _ident_tokens(assign_node.expr)
    r_arr = tokens & set(builder.arrays)
    r_scl = tokens & set(builder.scalars)
    # Whole-object rebind member tokens (alias name, no descriptor) miss the
    # array/scalar classification above and would leak as free symbols;
    # classify by the resolved real descriptor's kind.  Connector name stays
    # the alias (cosmetic) -- memlet + access node redirect to the real name.
    for tok in tokens - r_arr - r_scl:
        real = resolve_object_member(builder, tok)
        if real in builder.arrays:
            r_arr.add(tok)
        elif real in builder.scalars:
            r_scl.add(tok)
    # Name can collide between arrays/scalars when an inlined-callee's scalar
    # dummy shares a short Fortran name with an outer array (e.g. graupel's
    # 2D ``qr`` arg vs an inlined helper's scalar ``qr`` dummy).  Prefer the
    # array classification -- a scalar ``_in_qr``/``qr[0]`` memlet would fail
    # validation against the 2D ``qr.shape``.
    r_scl -= r_arr
    # Length-1 view_alias (scalar POINTER rebind) lives in r_arr but reads like
    # a scalar -- move to r_scl for a single ``_in_tmp`` connector, else it
    # renders as an unwired ``_in_tmp_0`` array occurrence.
    _len1_views = {nm for nm in r_arr if _is_len1_scalar_view(builder, nm)}
    r_arr -= _len1_views
    r_scl |= _len1_views
    target = assign_node.target

    # Index arrays (e.g. edge_idx) move onto the interstate edge as symbols, not connectors.
    indirect_arrays = {indirect_host(expr) for expr in indirect_syms}
    r_arr -= indirect_arrays

    # ONE connector + ONE memlet per textual occurrence (never dedup'd) -- dedup
    # used to misalign occurrence-to-access mapping when the accesses list and
    # expr disagreed on count (e.g. MIN/MAX cmp+select).  1:1 is the contract now.
    reads_by_name = {}
    for ac in accesses:
        if ac.is_read and ac.array_name in r_arr:
            reads_by_name.setdefault(ac.array_name, []).append(ac)

    # Connector substitution: scalars -> ``_in_<name>``, Nth array occurrence
    # -> ``_in_<name>_<N>`` (see ``_rewrite_read_connectors``).
    occ = {nm: 0 for nm in r_arr}
    sorted_tokens = sorted(r_arr | r_scl, key=len, reverse=True)

    in_c = {f"_in_{sc}" for sc in r_scl}
    for nm, acs in reads_by_name.items():
        for i in range(len(acs)):
            in_c.add(f"_in_{nm}_{i}")
    out_c = {f"_out_{target}"}

    # iter_map rename MUST run before the connector rewrite: ``d(i) = i*2.0``
    # in ``do i = 50, 54`` renders RHS ``i * 2.0``, but the LoopRegion's iter is
    # uniquified to ``i_0`` while an SDFG-level ``i`` symbol may also exist.
    # Without the rename the tasklet reads that ``i`` (typically zero), not ``i_0``.
    expr = rename_iters(assign_node.expr, iter_map)
    code = f"_out_{target} = {_rewrite_read_connectors(expr, sorted_tokens, r_scl, occ)}"
    # Bare ``?`` means the C++ AST builder hit a buildIndexExpr/leafExpr
    # fallback it couldn't trace; raise here instead of letting it reach
    # DaCe's ast.parse as an opaque ``SyntaxError`` at ``<unknown>:1``.
    if "?" in code:
        raise NotImplementedError(f"emit_tasklet: unresolved operand placeholder ``?`` in tasklet "
                                  f"body ``{code}`` (target={target!r}).  The C++ AST builder "
                                  "couldn't trace one of the operand chains -- check "
                                  "bridge/ast/assigns.cpp ``buildIndexExpr`` and "
                                  "expressions.cpp ``buildExpr`` for the fallback returning "
                                  "``?`` against this kernel's HLFIR.")
    t = state.add_tasklet(f"t_{idx}", in_c, out_c, code)

    for nm in sorted(reads_by_name):
        r = acc(builder, state, nm)
        # One edge per occurrence (1:1 with connectors); section-alias dummies
        # route through the source array via view_dim_map-spliced indices.
        for i, ac in enumerate(reads_by_name[nm]):
            eff_nm, eff_ac = resolve_section_alias(builder, nm, ac)
            ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
            state.add_edge(r, None, t, f"_in_{nm}_{i}", Memlet(f"{eff_nm}[{ix}]"))

    for sc in sorted(r_scl):
        r = acc(builder, state, sc)
        # The connector keeps the (cosmetic) alias token; the memlet + access
        # node bind the REAL descriptor when ``sc`` is an object-rebind member.
        eff_sc = resolve_object_member(builder, sc) or sc
        state.add_edge(r, None, t, f"_in_{sc}", Memlet(data=eff_sc, subset="0"))

    # Write-side access-node selection for the tasklet's output edge.
    # ``state._hlfir_access[name]`` caches the "live sink" -- the node later
    # reads pull from.  A NEW node (not the cached sink) is needed when: (1) the
    # write pairs with a read of the SAME name in the SAME tasklet (e.g.
    # ``i = i + 1``, ``d(1) = d(1)*2.0``) -- reusing it would put an in-edge and
    # an out-edge on one node, a cycle DaCe's validator rejects; (2) the cached
    # sink was already read by an earlier tasklet in this state -- sharing would
    # let the scheduler reorder the write before that read.  Otherwise reuse the
    # cached sink (multiple in-edges are legal).  Section-alias targets retarget
    # to the source array (the dummy has no SDFG descriptor); cache/self-update
    # bookkeeping keys off the source name.
    v_target = builder.arrays.get(target)
    eff_target = target
    if v_target is not None and getattr(v_target, 'role', '') == 'section_alias':
        eff_target = v_target.view_source
    else:
        # Whole-object rebind member write: retarget onto the real flattened
        # descriptor so the live update lands on real storage, not the alias.
        obj_real = resolve_object_member(builder, target)
        if obj_real is not None:
            eff_target = obj_real
    cache = getattr(state, '_hlfir_access', None)
    is_self_update = (target in r_scl) or (target in reads_by_name) \
                  or (eff_target in reads_by_name)
    cached_has_readers = False
    if cache is not None and eff_target in cache:
        cached_has_readers = state.out_degree(cache[eff_target]) > 0
    # View-edge rule: READ links source->view (``acc()``'s edge), WRITE links
    # view->source (writeback edge).  A WRITE to a View must go through
    # ``_ensure_view_writeback_link``, not ``acc()`` -- else the write never
    # propagates to the parent (parent looks uninitialised).  Covers the
    # pure-write case; self-update/cached-reader branch above handles RMW.
    v_eff = builder.arrays.get(eff_target)
    is_view_write = v_eff is not None and (getattr(v_eff, 'bounds_remap_view', False)
                                           or getattr(v_eff, 'role', '') == 'view_alias')
    if is_view_write or is_self_update or cached_has_readers:
        w = state.add_access(eff_target)
        if cache is not None:
            cache[eff_target] = w
        _ensure_view_writeback_link(builder, state, w, eff_target)
    else:
        w = acc(builder, state, eff_target)

    if target in builder.scalars or eff_target in builder.scalars:
        # Scalar target: no buildable index, subset is always element 0.
        state.add_edge(t, f"_out_{target}", w, None, Memlet(data=eff_target, subset="0"))
    else:
        ac = get_access(accesses, target, is_read=False)
        eff_nm, eff_ac = resolve_section_alias(builder, target, ac)
        ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
        state.add_edge(t, f"_out_{target}", w, None, Memlet(f"{eff_nm}[{ix}]"))


def emit_scalar_assign(builder, state, target: str, value: str):
    """Tasklet for ``target = value`` on a scalar target.  Identifier tokens
    naming an SDFG scalar each get their own input connector (so ``i = i + 1``
    self-updates work).  Whole-array fast path: when target and value are BOTH
    multi-dim arrays of the same shape, it's a pointer-rebind
    ``RewritePointerAssigns`` didn't collapse (e.g. ICON's
    ``icidx => p_patch%edges%cell_idx``) -- emit a whole-array copy memlet
    instead of a scalar tasklet with the wrong subset."""
    value = str(value)
    # Whole-OBJECT pointer rebind store (``params_oce => v_params``): a POINTER
    # association moves NO SDFG data -- member accesses resolve to the real
    # source via ``resolve_object_member`` on their own.  Emitting the store
    # would ``acc()`` a descriptor-less node that later crashes
    # ``prune_unused_arrays``.  Gate on REGISTERED rebind-store targets (not
    # merely-absent descriptor) so a genuinely-missing descriptor still surfaces.
    if target in (vars(builder).get("object_alias_defs") or set()):
        return
    # Bare ``?`` means the C++ AST builder couldn't trace an operand (designate
    # chain past ``kBuildIndexExprDepth``, missing indexStack entry, unresolved
    # memref).  Raise here instead of an opaque ``SyntaxError`` at ``<unknown>:1``.
    if "?" in value:
        raise NotImplementedError(f"emit_scalar_assign: unresolved operand placeholder ``?`` in "
                                  f"``{target} = {value}`` -- the C++ AST builder couldn't trace "
                                  "one of the operand chains.  Check bridge/ast/assigns.cpp "
                                  "``buildIndexExpr`` and control_flow.cpp ``leafExpr`` for the "
                                  "fallback returning ``?`` against this kernel's HLFIR.")
    src_name = value.strip()
    tgt_var = builder.arrays.get(target)
    tgt_is_array = (tgt_var is not None and getattr(tgt_var, "rank", 0) > 0
                    and len(tgt_var.shape_symbols) == tgt_var.rank)

    # Bounds-remap-view rebind (``p(1:M,1:K) => arr1d``): the View descriptor +
    # source->view linking edge (access.py) already establish the alias, so this
    # bare ``p = arr1d`` store is redundant.  Skip it -- else ``set_<target>``
    # writes a rank-1 subset against a multi-D View and the validator rejects it.
    if tgt_var is not None and getattr(tgt_var, "bounds_remap_view", False) \
            and getattr(tgt_var, "bounds_remap_source", "") == src_name:
        return

    # Plain section rebind (``p => a(:, j)``) lowered as view_alias: the
    # source->view link (access.py) already establishes it, so this bare store
    # is NOT a data copy -- skip it, else DaCe reports "ambiguous view".
    if tgt_var is not None and tgt_var.role == "view_alias" \
            and tgt_var.view_source == src_name:
        return

    if tgt_is_array:
        is_whole_array_copy = (src_name in builder.arrays and re.fullmatch(r'[A-Za-z_]\w*', src_name) is not None)
        if is_whole_array_copy:
            src_var = builder.arrays[src_name]
            if (src_var.rank == tgt_var.rank and len(src_var.shape_symbols) == src_var.rank):
                # Plain whole-array copy (pointer-rebind RewritePointerAssigns didn't
                # collapse): AccessNode(src) -> AccessNode(tgt), full-shape subsets, no tasklet.
                read = acc(builder, state, src_name)
                write = state.add_access(target)
                cache = getattr(state, '_hlfir_access', None)
                if cache is not None:
                    cache[target] = write
                _ensure_view_writeback_link(builder, state, write, target)
                subset = ",".join(f"0:{s}" for s in src_var.shape_symbols)
                state.add_edge(read, None, write, None, Memlet(f"{src_name}[{subset}]"))
                return

        # Whole-array fill: CONSTANT scalar -> multi-dim array (``ALLOCATE x; x
        # = 0`` zero-init prologues).  Fires for any RHS reading no data -- real
        # literal, complex literal (``1j``'s ``j`` excluded, see ``_IDENT_RE``),
        # or a complex128 cast.  ``add_mapped_tasklet`` wires the empty-input +
        # indexed-output edges itself.
        _val_toks = _ident_tokens(src_name)
        _reads_data = bool(_val_toks & (set(builder.arrays) | set(builder.scalars) | set(builder.symbols)))
        if not _reads_data:
            dims = tgt_var.shape_symbols
            ranges = {f"__i{k}": f"0:{s}" for k, s in enumerate(dims)}
            idx_expr = ",".join(f"__i{k}" for k in range(len(dims)))
            w = state.add_access(target)
            cache = getattr(state, '_hlfir_access', None)
            if cache is not None:
                cache[target] = w
            _ensure_view_writeback_link(builder, state, w, target)
            state.add_mapped_tasklet(
                name=f"set_{target}",
                map_ranges=ranges,
                inputs={},
                code=f"_out = {src_name}",
                outputs={"_out": Memlet(f"{target}[{idx_expr}]")},
                output_nodes={target: w},
                external_edges=True,
            )
            return

    tokens = _ident_tokens(value)

    # Length-1 view_alias (scalar POINTER rebind) reads like a scalar
    # (``tmp`` -> ``tmp[0]``) but lives in builder.arrays -- include it here,
    # else the tasklet emits invalid ``int*`` arithmetic (``tmp + 1`` on a view).
    # ``nm != target`` was wrong: ``i = i + 1`` needs a read edge on target itself.
    reads = [
        nm for nm in sorted(tokens, key=len, reverse=True) if nm in builder.scalars
        or _is_len1_scalar_view(builder, nm) or resolve_object_member(builder, nm) in builder.scalars
    ]

    code = value
    for nm in reads:
        code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)

    in_c = {f"_in_{nm}" for nm in reads}
    out_c = {'_out'}
    t = state.add_tasklet(f"set_{target}", in_c, out_c, f"_out = {code}")

    for nm in reads:
        r = acc(builder, state, nm)
        # A whole-object rebind scalar member (``params_oce_a_veloc_v_back``)
        # binds its real flattened descriptor; ``acc`` already redirected the
        # node, so the memlet must name the real descriptor too.
        eff_nm = resolve_object_member(builder, nm) or nm
        state.add_edge(r, None, t, f"_in_{nm}", Memlet(data=eff_nm, subset='0'))

    # Self-update (``i = i + 1``): read and write need DIFFERENT access nodes
    # so the state stays a DAG (no cycle on one node).  Same applies when an
    # EARLIER tasklet in this state already read ``target`` via the cached node.
    # Velocity-tendencies triggers this across two lines:
    # ``max_vcfl_dyn = MAX(p_diag%max_vcfl_dyn, ...)`` then
    # ``p_diag%max_vcfl_dyn = max_vcfl_dyn`` (2nd write's target was 1st read).
    cache = getattr(state, '_hlfir_access', None)
    cached_has_readers = (cache is not None and target in cache and state.out_degree(cache[target]) > 0)
    if (target in reads) or cached_has_readers:
        a = state.add_access(target)
        if cache is not None:
            cache[target] = a
        _ensure_view_writeback_link(builder, state, a, target)
    else:
        a = acc(builder, state, target)
    state.add_edge(t, '_out', a, None, Memlet(data=target, subset='0'))


def _cc_elem_subset(name, elem_exprs):
    """Element subset for a complex-component-alias access in the VIEW's own
    0-based coordinates: only subtracts the element dims' Fortran lower bound
    (``qg[(i) - offset_qg_d0, ...]``) -- the view's descriptor + linking memlet
    already encode the source slab, so no base folding here."""
    return ", ".join(f"({e}) - offset_{name}_d{k}" for k, e in enumerate(elem_exprs))


def emit_complex_component_assign(builder, state, node, idx: int, iter_map: dict, indirect_syms: dict = None):
    """``qg(c, i...) = <rhs>`` where ``qg`` is a complex-as-2-reals component
    alias (``REAL(2,N)`` dummy bound to a COMPLEX element -- QE's ``qvan2``
    ``qg(2,ngy)`` aliasing ``qgm(1,ijh)``), registered as a SAME-dtype COMPLEX
    View.  Staged RMW: read the complex value, set component ``c`` (1=real,
    else imag) to the rhs (``qg`` reads replaced by CURRENT component
    ``_cur``), write back via ``_ensure_view_writeback_link``.  Other rhs reads
    wire as ordinary per-occurrence connectors like ``emit_tasklet``."""
    indirect_syms = indirect_syms or {}
    name = node.target  # the COMPLEX view
    comp_dim = 0  # the leading (size-2) component dim of qg(2, ...)

    # The qg write access carries the index list ``[component, elem...]``.
    accesses = node.accesses or []
    wac = next((a for a in accesses if a.array_name == node.target and not a.is_read), None)
    if wac is None:
        raise NotImplementedError(f"complex_component_alias '{node.target}': "
                                  "assign has no write access to map onto the complex source")
    qg_exprs = list(wac.index_exprs)
    comp_expr = rename_iters(qg_exprs[comp_dim], iter_map)
    elem_exprs = [rename_iters(e, iter_map) for i, e in enumerate(qg_exprs) if i != comp_dim]
    elem_sub = _cc_elem_subset(name, elem_exprs)

    # rhs: replace bare ``qg`` reads with the CURRENT component ``_cur``.
    rhs = rename_iters(node.expr, iter_map)
    rhs = re.sub(rf'\b{re.escape(node.target)}\b', '_cur', rhs)

    # Collect the OTHER rhs reads (everything but ``qg``, now ``_cur``) and wire
    # like ``emit_tasklet``; the component selector scalar folds into the same set.
    tokens = _ident_tokens(rhs)
    r_arr = tokens & set(builder.arrays)
    r_scl = tokens & set(builder.scalars)
    r_scl -= r_arr
    _len1_views = {nm for nm in r_arr if _is_len1_scalar_view(builder, nm)}
    r_arr -= _len1_views
    r_scl |= _len1_views
    r_arr.discard(name)
    r_scl.discard(name)
    comp_is_scalar = comp_expr in builder.scalars
    if comp_is_scalar:
        r_scl.add(comp_expr)
    comp_ref = f"_in_{comp_expr}" if comp_is_scalar else comp_expr

    reads_by_name = {}
    for ac in accesses:
        if ac.is_read and ac.array_name in r_arr:
            reads_by_name.setdefault(ac.array_name, []).append(ac)
    occ = {nm: 0 for nm in r_arr}
    sorted_tokens = sorted(r_arr | r_scl, key=len, reverse=True)
    rhs_code = _rewrite_read_connectors(rhs, sorted_tokens, r_scl, occ)
    if "?" in rhs_code:
        raise NotImplementedError(f"emit_complex_component_assign: unresolved operand placeholder "
                                  f"``?`` in rhs ``{rhs_code}`` (target={name!r}).")

    in_conns = {'_in_z'} | {f"_in_{sc}" for sc in r_scl}
    for nm, acs in reads_by_name.items():
        for i in range(len(acs)):
            in_conns.add(f"_in_{nm}_{i}")
    # ``.real()``/``.imag()`` METHODS, not ``re()``/``im()`` helpers: a bare
    # ``im`` token collides with QE's kernel variable ``im`` (reserved-name
    # rewrite turns the call into a call on an int).  Attribute access isn't a
    # free Name, so symbol substitution can't touch it.
    code = (f"_cur = ((_in_z).real() if ({comp_ref} == 1) else (_in_z).imag())\n"
            f"_new = {rhs_code}\n"
            f"_out_z = (_new + 1j*(_in_z).imag()) if ({comp_ref} == 1) else ((_in_z).real() + 1j*_new)")
    t = state.add_tasklet(f"cc_{name}_{idx}", in_conns, {'_out_z'}, code)

    rz = acc(builder, state, name)  # COMPLEX view read (installs src -> view link)
    state.add_edge(rz, None, t, '_in_z', Memlet(f"{name}[{elem_sub}]"))
    for nm in sorted(reads_by_name):
        r = acc(builder, state, nm)
        for i, ac in enumerate(reads_by_name[nm]):
            eff_nm, eff_ac = resolve_section_alias(builder, nm, ac)
            ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
            state.add_edge(r, None, t, f"_in_{nm}_{i}", Memlet(f"{eff_nm}[{ix}]"))
    for sc in sorted(r_scl):
        r = acc(builder, state, sc)
        state.add_edge(r, None, t, f"_in_{sc}", Memlet(data=sc, subset="0"))
    # Fresh write node so ``view_read -> tasklet -> view_write`` is a clean RMW
    # chain; ``_ensure_view_writeback_link`` wires the view -> source direction.
    wz = state.add_access(name)
    state.add_edge(t, '_out_z', wz, None, Memlet(f"{name}[{elem_sub}]"))
    _ensure_view_writeback_link(builder, state, wz, name)
