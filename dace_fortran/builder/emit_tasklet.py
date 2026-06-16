"""Tasklet emission  --  both the full per-occurrence-connector path and the
flat ``emit_scalar_assign`` fast path.

``emit_tasklet`` is the heart of the frontend: takes an ``AssignNode``
from the bridge and produces a tasklet whose RHS expression has each
array-read occurrence wired to its own input connector with the right
memlet subset.  That per-occurrence wiring is what lets
``e_bln(jc,1)*z(...) + e_bln(jc,2)*z(...)`` produce three distinct
memlets instead of collapsing to one.

``emit_scalar_assign`` is the flush path for plain scalar assigns queued
on ``ctx.pending`` (``i = i + 1``, ``c = 0.5``)  --  simpler, no memlet
index math needed.

``assign_reads_array`` is the predicate used by ``emit_assign`` to
decide which path an incoming assign should take.
"""

import re

from dace import Memlet

from dace_fortran.builder.access import (acc, build_memlet_index, get_access, indirect_host, rename_iters,
                                         resolve_section_alias)

# Identifier scan that EXCLUDES the imaginary-unit suffix of a complex
# literal: in ``(0.0) + 1j * (0.0)`` (how the bridge renders a Fortran
# complex constant such as ``z = (1.0, 2.0)`` / ``c = 0.0_dp``), the
# ``j`` of ``1j`` is Python's imaginary unit, NOT a variable.  The
# negative lookbehind drops any word that starts right after a digit or
# ``.`` (``1j``, ``2.0j``), so a real scalar / loop iterator named ``j``
# is still picked up but ``1j`` is not mistaken for it -- otherwise a
# spurious ``_in_j`` read connector (or a missed constant-fill) appears
# wherever a complex literal meets a ``j`` in scope.
_IDENT_RE = re.compile(r'(?<![0-9.])[a-zA-Z_]\w*')


def _ident_tokens(expr: str) -> set:
    """Identifier tokens in ``expr``, excluding complex-literal imaginary
    units (see ``_IDENT_RE``)."""
    return set(_IDENT_RE.findall(expr))


def _is_len1_scalar_view(builder, nm: str) -> bool:
    """True when ``nm`` is a length-1 ``view_alias`` -- a Fortran scalar
    POINTER rebind (``tmp => x``) lowered as a length-1-array View.  It
    lives in ``builder.arrays`` but reads/writes like a scalar (``tmp``
    means ``tmp[0]``), so the emit paths wire it as a single ``_in_<nm>``
    connector (memlet ``<nm>[0]``) rather than an indexed array occurrence
    (an unwired ``_in_<nm>_0``) or a bare ``int*`` view reference."""
    a = builder.arrays.get(nm)
    return a is not None and getattr(a, 'role', '') == 'view_alias' \
        and list(a.shape_symbols) == ['1']


def _view_link_spec(builder, state, target: str):
    """Resolve the ``(src, src_subset, view_subset)`` for a View ``target``'s
    source linking memlet, or ``None`` if ``target`` is not a View.

    Handles all three View flavours uniformly  --  the complex-component alias
    (re-cast as a same-dtype COMPLEX view), the ``bounds_remap_view`` flatten
    pointer remap, and the plain ``view_alias`` reshape  --  by normalising each
    to a ``(view_source, view_subset)`` pair, then rendering the source-side slab
    (``:`` full-dim markers resolved against the parent shape; the ``[""]``
    whole-array sentinel spanning the full flat storage) and the view-side span.
    Shared by the read-link (``acc`` / ``_ensure_view_read_link``) and the
    write-back (``_ensure_view_writeback_link``) so both directions agree.
    """
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
    """When the Phase I self-update split creates a fresh access node
    for the write side of a view alias, the view's required source
    linking memlet (added by ``acc()`` on the read side) is missing
    on this new node  --  DaCe's ``get_view_edge`` returns None and
    validation fails.  Mirror the link in the writeback direction
    (view -> source) so the write node is also a recognised view edge.

    Two extra structural rules to keep ``get_view_edge`` happy:

    * Use a FRESH source access node for the writeback (not the
      cached one), otherwise ``src -> view_read -> tasklet ->
      view_write -> src`` forms a cycle on the source.
    * Drop ``target`` from the per-state cache after adding the
      writeback.  Subsequent reads in the same state then mint a new
      read-side view (with its own ``src -> view`` linking) instead
      of pulling through this write-side node  --  leaving the write
      node with the clean ``in=1 (tasklet) / out=1 (writeback)``
      shape ``get_view_edge`` recognises, and the read-side with the
      clean ``in=1 (linking) / out=N (tasklets)`` shape.
    """
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
    """Install the source -> view linking memlet on a GIVEN read node of a View
    ``target`` (the read-direction counterpart of
    ``_ensure_view_writeback_link``).

    ``acc()`` installs this link on its cached node, but a library node whose
    input is a View (e.g. an in-place FFT over the ``bounds_remap_view``
    ``prhoc_d``) must use a FRESH read node -- sharing ``acc``'s cached node
    with the matching write would self-cycle on the view.  This wires the
    ``views`` edge on that fresh node with a FRESH (un-cached) source node so the
    read-side source and the write-back source stay distinct (no cycle for the
    in-place case)."""
    spec = _view_link_spec(builder, state, target)
    if spec is None:
        return
    src, src_subset, view_subset = spec
    src_node = state.add_access(src)
    state.add_edge(src_node, None, read_node, 'views', Memlet(data=src, subset=src_subset, other_subset=view_subset))


def assign_reads_array(assign_node, arrays: dict) -> bool:
    """True iff any ``accesses`` entry on ``assign_node`` is a read against
    an array descriptor.  Used to promote a nominally-scalar assign
    (``s = d(i) + 1``) onto the per-occurrence-connector tasklet path so
    the array read gets a real memlet instead of a bare identifier in
    the code string.
    """
    for ac in assign_node.accesses:
        if ac.is_read and ac.array_name in arrays:
            return True
    return False


def _rewrite_read_connectors(code: str, sorted_tokens, scalar_reads, array_occ: dict) -> str:
    """Replace each read reference in a tasklet RHS with its
    per-occurrence input connector.

    A scalar ``<name>`` becomes ``_in_<name>``; the Nth array
    occurrence ``<name>[...]`` becomes ``_in_<name>_<N>`` and its
    balanced ``[...]`` subscript is consumed (the connector's memlet
    already targets that one element, so a leftover subscript would
    trip DaCe's dimension validator).  Balanced brackets are walked by
    hand since ``re`` can't.

    :param sorted_tokens: read names longest-first (so a short name
                          can't shadow a longer one sharing a prefix).
    :param scalar_reads: the subset of names that are scalar reads.
    :param array_occ: ``{array_name: next_occurrence_index}``, mutated
                      in place as occurrences are consumed.
    """
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
    """One Tasklet per array assignment.

    Expressions like ``e_bln(jc,1)*z_kin(...) + e_bln(jc,2)*z_kin(...)``
    access the same array at several positions.  Each *occurrence* in
    the RHS becomes its own tasklet input connector so every access
    carries the correct memlet; otherwise the generated code would
    collapse all three terms onto a single connector and silently
    compute a wrong result.
    """
    indirect_syms = indirect_syms or {}
    accesses = assign_node.accesses

    tokens = _ident_tokens(assign_node.expr)
    r_arr = tokens & set(builder.arrays)
    r_scl = tokens & set(builder.scalars)
    # A name can collide between ``builder.arrays`` and
    # ``builder.scalars`` when ``extract_vars``'s inlined-callee
    # local disambiguation didn't rename a SCALAR dummy whose short
    # Fortran name matches an outer ARRAY's name -- e.g. graupel's
    # ``qr(ivec, k_v)`` 2D arg plus an inlined helper that took a
    # scalar ``qr`` dummy.  The bridge's AccessInfo for the current
    # statement still carries the array shape (the token came from
    # a designate over the 2D qr declare), so prefer the array
    # classification and drop the scalar entry to avoid the
    # spurious ``_in_qr`` scalar connector + ``qr[0]`` 1D memlet
    # that fails validation against the 2D ``qr.shape``.
    r_scl -= r_arr
    # A length-1 ``view_alias`` (scalar POINTER rebind ``tmp => x``) lives in
    # ``builder.arrays`` so it lands in ``r_arr``, but it reads like a scalar
    # (``tmp`` -> ``tmp[0]``).  Move it to ``r_scl`` so it gets a single
    # ``_in_tmp`` connector (memlet ``tmp[0]``) -- otherwise it renders as an
    # unwired ``_in_tmp_0`` array occurrence (free symbol).
    _len1_views = {nm for nm in r_arr if _is_len1_scalar_view(builder, nm)}
    r_arr -= _len1_views
    r_scl |= _len1_views
    target = assign_node.target

    # Index arrays (e.g. edge_idx) show up in the RHS token scan but we
    # move their values onto the interstate edge as symbols.
    indirect_arrays = {indirect_host(expr) for expr in indirect_syms}
    r_arr -= indirect_arrays

    # One AccessInfo per textual occurrence, in the order buildExpr
    # produced.  We mint ONE connector + ONE memlet per occurrence  --
    # even when two occurrences read the same array element.  Sharing
    # connectors (dedup) used to misalign textual-occurrence-to-access
    # mapping when the bridge's accesses list and the textual expr
    # disagree on count (e.g., the MIN/MAX cmp+select pattern), so
    # the simpler 1:1 mapping is the contract now.
    reads_by_name = {}
    for ac in accesses:
        if ac.is_read and ac.array_name in r_arr:
            reads_by_name.setdefault(ac.array_name, []).append(ac)

    # Rewrite the RHS, replacing the Nth occurrence of each array name
    # (and consuming its balanced ``[...]`` subscript) with
    # ``_in_<name>_<N>``.  Scalars get a single bare-name connector
    # ``_in_<name>`` since they don't carry a subscript.
    # ``_in_<name>`` (scalars) / ``_in_<name>_<N>`` (Nth array
    # occurrence) connector substitution -- see _rewrite_read_connectors.
    occ = {nm: 0 for nm in r_arr}
    sorted_tokens = sorted(r_arr | r_scl, key=len, reverse=True)

    in_c = {f"_in_{sc}" for sc in r_scl}
    for nm, acs in reads_by_name.items():
        for i in range(len(acs)):
            in_c.add(f"_in_{nm}_{i}")
    out_c = {f"_out_{target}"}

    # Apply iter_map rename to bare symbol references in the RHS BEFORE
    # the array/scalar connector rewrite.  An assign like
    # ``d(i) = i*2.0`` inside a ``do i = 50, 54`` loop produces RHS
    # ``i * 2.0`` in the AST; the LoopRegion's iter is ``i_0`` (after
    # uniquification), but the SDFG-level symbol ``i`` may also exist
    # (a separate dummy with the same Fortran name, or just the
    # extract_vars symbol slot).  Without this rename, the tasklet
    # binds ``i`` to whatever the SDFG-level ``i`` symbol holds  --
    # typically zero  --  instead of the per-iteration value ``i_0``.
    expr = rename_iters(assign_node.expr, iter_map)
    code = f"_out_{target} = {_rewrite_read_connectors(expr, sorted_tokens, r_scl, occ)}"
    # Mirror the ``?`` guard in ``emit_scalar_assign``: a bare ``?`` in
    # the rendered RHS means the C++ AST builder hit a
    # ``buildIndexExpr`` / ``leafExpr`` fallback for an operand it
    # couldn't trace.  Without this raise the code reaches DaCe's
    # ``CodeBlock.ast.parse`` and surfaces as a ``SyntaxError`` at
    # ``<unknown>:1``, which is opaque about which AST node hit it.
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
        # One edge per textual occurrence (1:1 with connector names).
        # Section-alias dummies route through the source array with
        # indices spliced via ``view_dim_map``; the dummy itself has
        # no SDFG descriptor.
        for i, ac in enumerate(reads_by_name[nm]):
            eff_nm, eff_ac = resolve_section_alias(builder, nm, ac)
            ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
            state.add_edge(r, None, t, f"_in_{nm}_{i}", Memlet(f"{eff_nm}[{ix}]"))

    for sc in sorted(r_scl):
        r = acc(builder, state, sc)
        state.add_edge(r, None, t, f"_in_{sc}", Memlet(data=sc, subset="0"))

    # ----------------------------------------------------------------
    # Pick the write-side access node for the tasklet's output edge.
    # ----------------------------------------------------------------
    #
    # An SDFGState is a DAG of AccessNodes and Tasklets.  For each data
    # name we keep ONE "live sink" in ``state._hlfir_access[name]``  --
    # the access node that subsequent reads from that name should pull
    # from (because it holds the latest write).  Two rules govern
    # whether a new write reuses that sink or allocates a fresh one:
    #
    # 1. A write that is paired with a read of the SAME name in the
    #    SAME tasklet must target a NEW access node, not the one the
    #    read came from.  Otherwise the tasklet would have both an
    #    incoming and outgoing edge on the same node  --  a cycle  --  and
    #    DaCe's state validator would reject it.  Fortran patterns
    #    that trigger this: ``i = i + 1``, ``d(1) = d(1) * 2.0``,
    #    ``temp = min(d(1), temp)``.
    #
    # 2. A write whose cached sink has ALREADY been read by a later
    #    tasklet in this state must also get a new access node.
    #    Sharing it would let the DAG scheduler reorder the new write
    #    before the earlier read, changing observable data.
    #
    # Otherwise  --  pure write-only update over the latest sink  --  reuse
    # the cached access node.  Multiple in-edges to one node are legal;
    # sharing keeps the data-flow graph connected.
    # For section-alias targets, the write retargets to the source
    # array  --  the dummy has no SDFG descriptor.  Read-side reads also
    # route through the source name, so cache / self-update bookkeeping
    # must use the source name.
    v_target = builder.arrays.get(target)
    eff_target = target
    if v_target is not None and getattr(v_target, 'role', '') == 'section_alias':
        eff_target = v_target.view_source
    cache = getattr(state, '_hlfir_access', None)
    is_self_update = (target in r_scl) or (target in reads_by_name) \
                  or (eff_target in reads_by_name)
    cached_has_readers = False
    if cache is not None and eff_target in cache:
        cached_has_readers = state.out_degree(cache[eff_target]) > 0
    # Global view-edge rule: a READ of a View links ``source -> view`` (the
    # ``acc()`` read-side edge), a WRITE links ``view -> source`` (the
    # writeback edge).  ``acc()`` only emits the read direction, so a WRITE
    # whose target is a View must go through ``_ensure_view_writeback_link``
    # -- NOT ``acc()`` -- otherwise the write lands on a view node with an
    # incoming (read) view edge and never propagates to the parent (and the
    # parent looks uninitialised).  Covers the pure-write case the
    # self-update / cached-reader branch already handled for RMW.
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

    if target in builder.scalars:
        # Scalar target: no buildable index, subset is always element 0.
        state.add_edge(t, f"_out_{target}", w, None, Memlet(data=target, subset="0"))
    else:
        ac = get_access(accesses, target, is_read=False)
        eff_nm, eff_ac = resolve_section_alias(builder, target, ac)
        ix = build_memlet_index(builder, eff_nm, eff_ac, iter_map, indirect_syms)
        state.add_edge(t, f"_out_{target}", w, None, Memlet(f"{eff_nm}[{ix}]"))


def emit_scalar_assign(builder, state, target: str, value: str):
    """Tasklet for ``target = value`` on a scalar target.

    Inputs are derived from the identifier tokens that appear in
    ``value``  --  every one that names an SDFG scalar gets its own
    input connector so the tasklet can read ``i`` for ``i = i + 1``
    and similar self-updates.

    Whole-array fast path: an assign whose target and (single-token)
    value are BOTH multi-dim arrays of the same descriptor shape is a
    pointer-rebind that ``RewritePointerAssigns`` didn't collapse  --
    likely a chain shape (``ptr => derived_root%a%b``) the pass
    doesn't yet recognise.  Emit a whole-array copy memlet instead of
    a scalar tasklet with the wrong subset.  ICON's velocity_tendencies
    surfaces this with the ``icidx => p_patch%edges%cell_idx`` rebinds
    in its setup block.
    """
    value = str(value)
    # A bare ``?`` in the rendered value means the C++ AST builder hit
    # ``buildIndexExpr`` / ``leafExpr`` for an operand it could not
    # trace -- a designate chain past ``kBuildIndexExprDepth``, a
    # block-arg with no entry on ``indexStack``, or a load whose
    # memref didn't resolve to a declare.  Emitting it as Python code
    # gives a SyntaxError pointing at <unknown>:1 in DaCe's
    # ``ast.parse`` from a deeply-nested call site, which is opaque.
    # Raise here with the target / value so the gap is easy to find.
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

    # Bounds-remap-view rebind (``p(1:M, 1:K) => arr1d``): the View
    # descriptor itself + the source -> view linking edge in
    # ``access.py`` already establish the alias.  The Fortran-level
    # rebind ``p = arr1d`` lands here as an array-target assign with
    # a bare name RHS; the bridge's ``RewritePointerAssigns`` pass
    # leaves it intact when the LHS is tagged ``bounds_remap_view``
    # so we can emit a typed View instead of the per-element index
    # rewrite.  Skip the redundant tasklet entirely -- otherwise
    # ``set_<target>`` writes a rank-1 subset against a multi-D View
    # and the validator rejects the edge.
    if tgt_var is not None and getattr(tgt_var, "bounds_remap_view", False) \
            and getattr(tgt_var, "bounds_remap_source", "") == src_name:
        return

    # P3: a plain section rebind (``p => a(:, j)`` / ``p => store(3:7)``)
    # lowered as a ``view_alias``.  The source -> view linking edge in
    # access.py already establishes the alias; the bare ``p = a`` rebind
    # store is NOT a data copy -- skip it, else a spurious whole-array
    # ``a -> view`` edge is added on top of the section link and DaCe
    # reports an "ambiguous view" (two candidate sources).
    if tgt_var is not None and tgt_var.role == "view_alias" \
            and tgt_var.view_source == src_name:
        return

    if tgt_is_array:
        is_whole_array_copy = (src_name in builder.arrays and re.fullmatch(r'[A-Za-z_]\w*', src_name) is not None)
        if is_whole_array_copy:
            src_var = builder.arrays[src_name]
            if (src_var.rank == tgt_var.rank and len(src_var.shape_symbols) == src_var.rank):
                # Plain whole-array copy (pointer-rebind shape that
                # ``RewritePointerAssigns`` didn't collapse, e.g.
                # ``icidx => p_patch%edges%cell_idx``).
                # AccessNode(src) -> AccessNode(tgt) with full-shape
                # subsets on both sides; no tasklet needed.
                read = acc(builder, state, src_name)
                write = state.add_access(target)
                cache = getattr(state, '_hlfir_access', None)
                if cache is not None:
                    cache[target] = write
                _ensure_view_writeback_link(builder, state, write, target)
                subset = ",".join(f"0:{s}" for s in src_var.shape_symbols)
                state.add_edge(read, None, write, None, Memlet(f"{src_name}[{subset}]"))
                return

        # Whole-array fill: CONSTANT scalar -> multi-dim array.  The
        # bridge emits this for ``ALLOCATE x; x = 0`` and similar
        # zero-init prologues.  Fires for any RHS that reads no data --
        # a bare real literal (``0.0``), but also a complex literal
        # (``(0.0) + 1j * (0.0)`` for a COMPLEX ``x = 0.0_dp``) or a
        # ``dace.complex128(...)`` cast.  The imaginary-unit ``j`` in
        # ``1j`` is NOT a data identifier, so exclude any word preceded
        # by a digit / dot from the data-token check (a real variable
        # ``j`` is still caught).  Use ``add_mapped_tasklet`` so DaCe
        # wires the empty-input + indexed-output edges itself.
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

    # A length-1 ``view_alias`` (a Fortran scalar POINTER rebind ``tmp => x``
    # lowered as a length-1-array view) reads like a scalar: ``tmp`` in a
    # scalar expression means ``tmp[0]``.  It lives in ``builder.arrays``
    # (not ``builder.scalars``), so include it here -- otherwise the tasklet
    # references the bare ``tmp`` (an ``int*`` view) and codegen emits
    # ``tmp + 1`` -> invalid ``int*`` arithmetic.
    # ``nm != target`` was wrong  --  ``i = i + 1`` genuinely needs a read
    # edge on the target itself.
    reads = [
        nm for nm in sorted(tokens, key=len, reverse=True)
        if nm in builder.scalars or _is_len1_scalar_view(builder, nm)
    ]

    code = value
    for nm in reads:
        code = re.sub(rf'\b{re.escape(nm)}\b', f'_in_{nm}', code)

    in_c = {f"_in_{nm}" for nm in reads}
    out_c = {'_out'}
    t = state.add_tasklet(f"set_{target}", in_c, out_c, f"_out = {code}")

    for nm in reads:
        r = acc(builder, state, nm)
        state.add_edge(r, None, t, f"_in_{nm}", Memlet(data=nm, subset='0'))

    # Self-update (``i = i + 1``): the read and write need DIFFERENT
    # access nodes so the state remains a DAG  --  ``Access(i_read) ->
    # Tasklet -> Access(i_write)`` instead of a cycle on one node.
    # Same rule applies (Phase I) when an EARLIER tasklet in the same
    # state already read ``target`` through the cached access node  --
    # reusing it for our write would put both an in-edge and an
    # out-edge on the same node, creating the same cycle.  Velocity-
    # tendencies surfaces this with the two-line pattern
    # ``max_vcfl_dyn = MAX(p_diag%max_vcfl_dyn, ...)``
    # ``p_diag%max_vcfl_dyn = max_vcfl_dyn``: the second assign's
    # writeback target was the first assign's RHS read.
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
    0-based coordinates.  The COMPLEX view already encodes the source slab
    (base / extent / aliasing) via its descriptor + linking memlet, so an access
    ``qg(comp, i...)`` only subtracts the element dims' Fortran lower bound:
    ``qg[(i) - offset_qg_d0, ...]``.  No base folding here -- that was the
    descriptor-less form; the view subsumes it."""
    return ", ".join(f"({e}) - offset_{name}_d{k}" for k, e in enumerate(elem_exprs))


def emit_complex_component_assign(builder, state, node, idx: int, iter_map: dict, indirect_syms: dict = None):
    """``qg(c, i...) = <rhs>`` where ``qg`` is a complex-as-2-reals component
    alias (a ``REAL(2, N)`` dummy bound to a ``COMPLEX`` element -- QE's
    ``qvan2`` ``qg(2, ngy)`` aliasing ``qgm(1, ijh)``).

    ``qg`` is registered as a SAME-dtype COMPLEX View of the source slab (see
    ``cc_alias_view_spec`` / ``descriptors.py``), so the write is a staged
    read-modify-write on the VIEW element ``qg[i...]`` (the view's linking memlet
    maps it to the complex source): read the complex value, set its component
    ``c`` (``c == 1`` real, else imaginary) to the rhs (with any ``qg`` read in
    the rhs replaced by the CURRENT component ``_cur``), and write it back.  Any
    OTHER array / scalar reads in the rhs (QE's ``sig * ylmk0(ig,lp) * work``)
    are wired as ordinary tasklet input connectors -- the same per-occurrence
    connector machinery ``emit_tasklet`` uses.  Components are read / built with
    the ``.real()`` / ``.imag()`` methods (collision-proof) and the
    ``component + 1j*other`` reconstruction; the write-back direction is wired by
    ``_ensure_view_writeback_link``.
    """
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

    # Collect the OTHER reads in the rhs (everything but ``qg`` itself, now
    # ``_cur``) and wire them like ``emit_tasklet`` does: array occurrences get a
    # per-occurrence ``_in_<name>_<N>`` connector (its ``[...]`` subscript folded
    # into the memlet), scalars a single ``_in_<name>``.  The component selector
    # scalar is folded into this same scalar set so it is wired exactly once.
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
    # Read the components with the ``.real()`` / ``.imag()`` METHODS rather than
    # the ``re()`` / ``im()`` helper functions: a bare ``im`` token in tasklet
    # code collides with a kernel variable/symbol named ``im`` (QE declares one),
    # which the reserved-name rewrite renames to ``program_im`` -- turning the
    # function call into a call on an int.  An attribute access (``_in_z.imag()``)
    # is immune (it is not a free Name, so no symbol substitution touches it) and
    # compiles directly on ``std::complex`` / ``thrust::complex`` with no d-face
    # support needed.
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
