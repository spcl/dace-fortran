"""Control-flow-graph emission: assign, loop, while, conditional.

These produce the SDFG's CFG skeleton  --  LoopRegions, ConditionalBlocks,
and interstate-edge state-change assignments for symbol writes.  The
actual per-element compute lives in ``emit_tasklet``; this module only
stitches states and regions together.
"""

import re

import dace
from dace import InterstateEdge
from dace.sdfg.state import LoopRegion, ConditionalBlock, ControlFlowRegion

from dace_fortran.builder.access import (
    acc,
    array_read_to_dace_expr,
    collect_indirect,
    find_array_subscripts,
    indirect_to_dace,
    iter_view_dim_map,
)
from dace_fortran.builder.context import _Ctx
from dace_fortran.builder.descriptors import auto_declare_synth
from dace_fortran.builder.emit_tasklet import assign_reads_array, emit_tasklet

_DACE_CAST_RE = re.compile(r"dace\.(?:int32|int64|float32|float64)\(")


def _strip_dace_casts(expr):
    """Drop tasklet-style ``dace.<dtype>(...)`` casts from a symbolic
    interstate-edge expression.

    The bridge renders a Fortran ``fir.convert`` (e.g. a real->int
    truncation ``it = ap1``) as ``dace.int32(...)`` so a *tasklet* lowers it
    via ``static_cast``.  On an interstate edge the assignment is parsed by
    DaCe's symbolic engine, which has no ``dace`` symbol -- it would treat
    ``dace`` as a free symbol (``KeyError: 'dace'`` in ``arglist``).  The
    target symbol's own dtype performs the same truncation/widening on
    assignment, so dropping the wrapper is value-preserving here.
    """
    if not isinstance(expr, str) or "dace." not in expr:
        return expr
    while True:
        m = _DACE_CAST_RE.search(expr)
        if not m:
            return expr
        open_paren = m.end() - 1
        depth, i = 0, open_paren
        while i < len(expr):
            if expr[i] == "(":
                depth += 1
            elif expr[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if i >= len(expr):
            return expr  # unbalanced -- leave as-is
        expr = expr[:m.start()] + "(" + expr[open_paren + 1:i] + ")" + expr[i + 1:]


def _anchor_views_referenced_in_expr(builder, expr: str, region, pre, sdfg):
    """Ensure every ``view_alias`` array referenced (by name) in ``expr``
    has at least one real AccessNode in a state upstream of the
    interstate edge that will carry ``expr``.

    DaCe's framecode scans interstate-edge free_symbols and synthesises
    a bare ``AccessNode`` if the symbol is an array name with no real
    node yet  --  which then trips ``allocate_view -> get_view_edge ->
    state.in_edges`` because the synthetic node isn't in any state.
    The anchor state's call to ``acc()`` registers the view's
    ``src -> view`` linking memlet, so framecode finds a real instance
    first in topological order.

    Section-alias dummies are excluded  --  they have no SDFG descriptor
    and ``expr`` should have already been rewritten through
    ``_rewrite_section_aliases_in_expr`` before this runs.

    Returns the (possibly updated) ``pre`` state to chain off.
    """
    if not isinstance(expr, str):
        return pre
    view_aliases = {nm for nm, v in builder.arrays.items() if getattr(v, 'role', '') == 'view_alias'}
    if not view_aliases:
        return pre
    referenced = [nm for nm in view_aliases if re.search(rf'\b{re.escape(nm)}\b', expr)]
    if not referenced:
        return pre
    anchor = region.add_state(f"view_anchor_{builder.nid()}")
    region.add_edge(pre, anchor, InterstateEdge())
    for nm in referenced:
        acc(builder, anchor, nm)
    return anchor


def _rewrite_section_aliases_in_expr(builder, expr: str) -> str:
    """Rewrite ``dummy[i, j]`` to ``source[i, j, k_const]`` for every
    section_alias dummy referenced in ``expr``.

    Used by emit_cond / emit_loop when condition / bound expressions
    get lifted onto interstate-edge assignments  --  the dummy itself has
    no SDFG descriptor, so a bare ``dummy`` symbol in the edge's free
    symbols would trip ``sdfg.arglist`` (KeyError) and DaCe's
    allocation-lifetime tracker.

    The input subscripts are 0-based DaCe-form (built by
    ``buildExprWithSubscripts`` as ``(idx) - 1``); ``view_dim_map``'s
    scalar slots are 1-based Fortran-form, so we subtract 1 when
    splicing them in.
    """
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    section_dummies = {nm for nm, v in builder.arrays.items() if getattr(v, 'role', '') == 'section_alias'}
    if not section_dummies:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    if not matches:
        return expr
    out = expr
    for start, end, arr, parts in reversed(matches):
        if arr not in section_dummies:
            continue
        v = builder.arrays[arr]
        new_parts = []
        for _src_dim, slot, dummy_dim in iter_view_dim_map(v.view_dim_map):
            if dummy_dim is not None:
                new_parts.append(parts[dummy_dim] if dummy_dim < len(parts) else '0')
            else:
                new_parts.append(f"({slot}) - 1")
        out = out[:start] + f"{v.view_source}[{', '.join(new_parts)}]" + out[end:]
    return out


def emit_assign(builder, ctx: '_Ctx', n, region):
    """Scalar or symbol assignment.

    Routes by target kind:
      * ``symbols``    -> interstate-edge assignment that forces a new state.
      * ``array``      -> tasklet via ``emit_tasklet`` with per-occurrence
                         array-read connectors.
      * ``scalar`` whose RHS reads an array element (``s = d(2,1) + 1.0``)
                         -> same tasklet path; the subscripted read needs a
                         real memlet so the codegen sees a connector, not
                         a bare array-pointer identifier.
      * plain ``scalar`` (``i = i + 1``, ``c = 0.5``) -> queued on
                         ``ctx.pending`` for ``emit_scalar_assign`` at
                         flush time.
    """
    # Synthetic scalars (``__sc_N`` / ``__al_N``) from the faithful
    # scf.while walker don't come in as ``hlfir.declare`` ops, so
    # ``add_descriptors`` never saw them.  Register on first assign.
    auto_declare_synth(builder, n.target, ctx)
    if n.target in builder.symbols:
        # Symbol-target reads of an array (``ci0 = icidx(je, jb, 1)``  --
        # the scalar-staged indirection load) need the bridge's bare-name
        # ``n.expr`` (just ``"icidx"``) reconstructed into a full DaCe
        # subscript expression with Fortran->0-based offsets and
        # iter_map remap; otherwise the interstate edge would assign the
        # whole array to a scalar symbol.  Plain symbol writes
        # (``i = i + 1``) keep ``n.expr`` verbatim.
        # ``array_read_to_dace_expr`` lifts every builder.arrays read to the
        # uniform offset-subset form AND derefs length-1 Array scalars to
        # ``name[0]``.  Scalar I/O convention: ``intent(inout)`` /
        # ``intent(out)`` scalar dummies (and module globals like ``kunit`` /
        # ``npool`` the kernel updates) register in the SDFG as length-1
        # ``Array`` descriptors (writable buffer for the caller's binding);
        # ``intent(in)`` scalars register as ``Scalar``.  The C ABI binds an
        # Array as ``T*`` and a Scalar as ``T``, so a bare ``<name>`` on the
        # RHS of a symbol-target interstate-edge assignment leaks the pointer
        # (``int* / int*``) unless dereferenced.  The token-walk handles this
        # uniformly for both single-name (``indices_end = endidx``) and
        # COMPOUND (``nks = kunit * ifloor(nkbl / npool)``) right-hand sides;
        # plain ``i = i + 1`` passes through verbatim.
        rhs = array_read_to_dace_expr(builder, n, ctx.iter_map, ctx.sdfg)
        rhs = _strip_dace_casts(rhs)
        ctx.flush(builder, region)
        ctx.ensure(region)
        dst = region.add_state(f"post_{n.target}_{builder.nid()}")
        region.add_edge(ctx.cur, dst, InterstateEdge(assignments={n.target: rhs}))
        ctx.cur = dst
        return
    if n.target_is_array or assign_reads_array(n, builder.arrays):
        ctx.flush(builder, region)
        ctx.ensure(region)

        # WAR / WAW hazard guard.  When the current state already
        # contains tasklets that read or write any array this new
        # assign touches, force a new state so state-edge ordering
        # enforces Fortran's sequential semantics.  Without this,
        # the codegen scheduler (which sees only RAW dependencies
        # on shared AccessNodes) is free to reorder sibling tasklets
        # that share underlying storage  --  yielding the WAR-violation
        # diagnosed at commit a9bf02173 (cloudsc Section 4.5).
        # This is the realised-graph form of the same sequential-
        # semantics invariant ``_sibling_rw_hazard`` enforces on an
        # AST-list (``emit_loop``'s flat ``child_assigns`` path): a
        # correctness change to one must be mirrored in the other.
        # Here ``has_structured`` / IF-body emission processes children
        # one at a time and would otherwise share the current state, so
        # the check runs against ``ctx.cur``'s already-built nodes.
        from dace.sdfg.state import SDFGState
        from dace.sdfg.nodes import Tasklet, AccessNode

        if isinstance(ctx.cur, SDFGState):
            prior_writes = set()
            prior_reads = set()
            for nd in ctx.cur.nodes():
                if isinstance(nd, AccessNode):
                    has_w = any(isinstance(e.src, Tasklet) for e in ctx.cur.in_edges(nd))
                    has_r = any(isinstance(e.dst, Tasklet) for e in ctx.cur.out_edges(nd))
                    if has_w:
                        prior_writes.add(nd.data)
                    if has_r:
                        prior_reads.add(nd.data)
            new_reads = {ac.array_name for ac in n.accesses if ac.is_read and ac.array_name in builder.arrays}
            new_writes = ({n.target} if (n.target_is_array and n.target in builder.arrays) else set())
            new_writes |= {ac.array_name for ac in n.accesses if ac.is_write and ac.array_name in builder.arrays}
            # RAW: new reads vs prior writes (state-edge would chain through
            # the shared AccessNode anyway, but be explicit).
            # WAR: new writes vs prior reads.
            # WAW: new writes vs prior writes.
            hazard = ((new_reads & prior_writes) or (new_writes & prior_reads) or (new_writes & prior_writes))
            if hazard:
                ctx.new_state(builder, region, label=f"asn_{n.target}_{builder.nid()}")

        # Inline indirect accesses: ``vn(iqidx(je,jb,1), jk, iqblk(je,jb,1))``
        # inside an IF body skips ``emit_loop``'s batch path, so the
        # indirect symbols would otherwise never get minted and the
        # memlet subset would carry the bare array name (which DaCe
        # codegen renders as a pointer-vs-int multiply).  Mint them
        # here, chained one-symbol-per-state so inner indirects are
        # available to outer ones.
        indirect_syms = collect_indirect(builder, [n])
        if indirect_syms:
            for expr, sym in indirect_syms.items():
                rhs = _strip_dace_casts(indirect_to_dace(builder, expr, ctx.iter_map, indirect_syms))
                if sym not in ctx.sdfg.symbols:
                    ctx.sdfg.add_symbol(sym, dace.int64)
                nxt = region.add_state(f"sym_{sym}_{builder.nid()}")
                region.add_edge(ctx.cur, nxt, InterstateEdge(assignments={sym: rhs}))
                ctx.cur = nxt
        emit_tasklet(builder, ctx.cur, n, builder.nid(), ctx.iter_map, indirect_syms or None)
        return
    ctx.pending.append((n.target, n.expr))


def emit_symbol_init(builder, ctx: '_Ctx', n, region):
    """Stage a position-array -> SDFG-symbol read at SDFG entry.

    The bridge mints one of these for every ``arr(consts)`` it sees used
    as an array index, section bound, or shape extent (e.g.
    ``a(pos(1):pos(2))`` or ``allocate(buf(shp(1,2,1)))``).  ``n.target``
    is the symbol name (``__sym_pos_1`` / ``__sym_shp_1_2_1``), ``n.expr``
    the source array name, and ``n.pos_indices`` the per-dim 1-based
    Fortran indices.  We add the symbol and emit an interstate edge
    ``__sym_pos_1 = pos[0]`` (``__sym_shp_1_2_1 = shp[0, 1, 0]``) so every
    memlet / shape referencing the symbol resolves to a closed-form
    expression rather than a data lookup DaCe can't represent in a subset.
    """
    sym, arr = n.target, n.expr
    idxs = list(getattr(n, "pos_indices", None) or [])
    if not idxs:  # back-compat: scalar mirror on loop_lower
        idxs = [int(n.loop_lower)]
    if sym not in ctx.sdfg.symbols:
        ctx.sdfg.add_symbol(sym, dace.int64)
    ctx.flush(builder, region)
    ctx.ensure(region)
    dst = region.add_state(f"sym_init_{sym}_{builder.nid()}")
    # Reference the source without a subscript when it isn't a real
    # multi-element data array:
    #   * ``arr`` is already a SYMBOL -- a flattened struct-member scalar
    #     promoted to a shape symbol (QE ``dfftt%ngm`` -> ``dfftt_ngm``)
    #     holds the value directly; ``dfftt_ngm[0]`` would subscript a
    #     symbol and ``sdfg.validate`` raises ``KeyError`` looking it up
    #     as data.
    #   * ``arr`` is a Scalar on the SDFG (the bridge folds length-1
    #     transients to Scalar) -- a Scalar has no subscript.
    # Otherwise emit the (multi-dim) 0-based subscript; these source
    # arrays use the default lower bound 1.
    from dace.data import Scalar
    src_desc = ctx.sdfg.arrays.get(arr)
    if arr in ctx.sdfg.symbols or isinstance(src_desc, Scalar):
        read_expr = arr
    else:
        read_expr = f"{arr}[{', '.join(str(i - 1) for i in idxs)}]"
    region.add_edge(ctx.cur, dst, InterstateEdge(assignments={sym: read_expr}))
    ctx.cur = dst


def _fortran_subs_to_dace(expr, builder):
    """Rewrite every ``<arr>[<idx>, ...]`` substring in ``expr`` to
    DaCe 0-based form ``<arr>[(<idx>) - offset_<arr>_d<i>, ...]`` for
    each known array.  Used by ``emit_loop`` to convert Fortran-form
    bound expressions (e.g. ``row_ptr[(i_0+1)]``) into valid DaCe
    subscripts before they land in a LoopRegion's init / cond.
    Non-array names (or untracked synthesised arrays) are left
    unchanged.  Walks brackets balanced via ``find_array_subscripts``
    so nested subscripts are handled correctly."""
    if not isinstance(expr, str) or '[' not in expr:
        return expr
    matches = list(find_array_subscripts(expr, builder.arrays))
    if not matches:
        return expr
    out = []
    cursor = 0
    for start, end, arr, parts in matches:
        out.append(expr[cursor:start])
        new_inner = ", ".join(f"({p}) - offset_{arr}_d{d}" for d, p in enumerate(parts))
        out.append(f"{arr}[{new_inner}]")
        cursor = end
    out.append(expr[cursor:])
    return "".join(out)


def _is_trivial_bound(expr: str) -> bool:
    """A bound / condition expression is trivial when it's a bare
    identifier or a single integer literal -- hoisting it to a symbol
    would be pure ceremony.  Anything with operators, brackets, or
    whitespace is non-trivial and gets hoisted."""
    s = expr.strip()
    if not s:
        return True
    # Bare integer literal (incl. signed).
    if s.lstrip('-+').isdigit():
        return True
    # Bare identifier (single name, no operators, no brackets).
    if all(ch.isalnum() or ch == '_' for ch in s) and not s[0].isdigit():
        return True
    return False


def _deref_scalar_arrays_for_interstate(expr_str: str, ctx) -> str:
    """Rewrite bare references to scalar ``(1,)``-shape SDFG data
    descriptors as ``name[0]``.

    Interstate-edge assignments do not auto-dereference scalar-array
    data on the LHS / RHS the way tasklet codegen does (``int _in_x =
    x[0];``).  Any module-level scalar exposed on the SDFG as a
    ``(1,)``-Array (the bridge's default for module variables that
    are not promoted to symbols) must therefore be referenced as
    ``name[0]`` in interstate expressions or the C++ unparser emits
    ``int64_t loopend_N = (x - 1);`` -- pointer arithmetic against a
    ``int*`` parameter -- and the build fails.

    :param expr_str: a rendered expression string headed for an
        ``InterstateEdge.assignments`` value or condition.
    :param ctx: the build context (carries ``ctx.sdfg`` whose
        ``.arrays`` map is the source of truth for shapes).
    :returns: ``expr_str`` with bare names of scalar arrays substituted
        for ``name[0]``; identifiers that are not scalar arrays, are
        already subscripted, or aren't known to the SDFG are left
        untouched.
    """
    import re
    if not isinstance(expr_str, str) or not expr_str:
        return expr_str

    sdfg_arrays = ctx.sdfg.arrays

    def _is_scalar_array(name: str) -> bool:
        # Only ``(1,)``-shape ``dace.data.Array`` entries need
        # dereferencing -- ``dace.data.Scalar`` is passed by value in
        # the C++ signature, so subscripting one ('``n[0]``' where ``n``
        # is a plain ``int``) is a hard compile error.
        desc = sdfg_arrays.get(name)
        if not isinstance(desc, dace.data.Array):
            return False
        shape = tuple(desc.shape)
        if len(shape) != 1:
            return False
        try:
            return int(shape[0]) == 1
        except (TypeError, ValueError):
            return False

    out = []
    pos = 0
    for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr_str):
        name = m.group(0)
        out.append(expr_str[pos:m.start()])
        # Skip when already subscripted (next non-space char is ``[``).
        tail = expr_str[m.end():]
        already_subscripted = tail.lstrip().startswith('[')
        if _is_scalar_array(name) and not already_subscripted:
            out.append(f"{name}[0]")
        else:
            out.append(name)
        pos = m.end()
    out.append(expr_str[pos:])
    return "".join(out)


def _hoist_bound_to_symbol(ctx, region, builder, expr_str: str, prefix: str):
    """Stage a non-trivial loop-bound expression onto a fresh
    ``<prefix>_<nid>`` ``int64`` symbol via a pre-LoopRegion interstate
    edge, so the LoopRegion's init / cond carry only a symbol name.

    :param expr_str: the rendered bound expression.
    :param prefix: symbol-name prefix (``loopend`` / ``loopbegin``).
    :returns: the new symbol name, or ``None`` when ``expr_str`` is a
              trivial bound (the caller keeps its original value).
    """
    if _is_trivial_bound(expr_str):
        return None
    sym = f"{prefix}_{builder.nid()}"
    if sym not in ctx.sdfg.symbols:
        ctx.sdfg.add_symbol(sym, dace.int64)
    ctx.ensure(region)
    nxt = region.add_state(f"pre_{sym}")
    deref_expr = _deref_scalar_arrays_for_interstate(expr_str, ctx)
    region.add_edge(ctx.cur, nxt, InterstateEdge(assignments={sym: deref_expr}))
    ctx.cur = nxt
    return sym


def _sibling_rw_hazard(assigns) -> bool:
    """Sequential-semantics hazard test over a list of sibling assign
    ASTNodes (the AST-list form of the invariant ``emit_assign``'s
    in-state guard enforces on the realised graph).

    An inlined elemental body like ``f = g*g; g = g/(1+g)`` puts both
    tasklets in one state with no dataflow edge; since both access
    nodes back the same non-transient storage, DaCe's scheduler (which
    honours only RAW edges on shared AccessNodes) may reorder the write
    ahead of the read and clobber the value.  When this returns true
    the caller must emit one state per assign so state-edge ordering
    enforces Fortran order.  Both a forward pass (read after an earlier
    sibling's write) and a reverse pass (read before a later sibling's
    write) are checked.

    :param assigns: sibling assign ASTNodes in source order.
    :returns: ``True`` if any name-sharing R/W pair would race in one
        state.
    """
    write_names_so_far = set()
    for a in assigns:
        reads = {ac.array_name for ac in a.accesses if ac.is_read}
        if reads & write_names_so_far:
            return True
        for ac in a.accesses:
            if ac.is_write:
                write_names_so_far.add(ac.array_name)
            if ac.is_read and ac.array_name in {ac2.array_name for ac2 in a.accesses if ac2.is_write}:
                # self-update within one assign  --  fine (handled by
                # the write-sink logic in emit_tasklet), but any later
                # sibling that reads the same name must be in a new
                # state so it sees the updated value.
                write_names_so_far.add(ac.array_name)
    # Also catch later-writer / earlier-reader patterns that would
    # otherwise race within a single state.
    later_writes = set()
    for a in reversed(assigns):
        reads = {ac.array_name for ac in a.accesses if ac.is_read}
        if reads & later_writes:
            return True
        for ac in a.accesses:
            if ac.is_write:
                later_writes.add(ac.array_name)
    return False


def emit_loop(builder, ctx: '_Ctx', n, region, iter_map=None):
    """Fortran DO loop -> LoopRegion with exact Fortran bounds."""
    # Flush any pending scalar assigns from earlier siblings INTO the
    # parent region.  Without ``region`` here, ``ctx.flush`` would land
    # them in ``ctx.sdfg`` (the top-level SDFG)  --  disconnected from the
    # nested loop and orphaned: e.g. ``acc = 0.0d0`` ahead of an inner
    # ``do j = ...`` would surface as a duplicate top-level ``s_*``
    # state with no incoming edge, making the parent CFG's start block
    # ambiguous.
    ctx.flush(builder, region)
    # The bridge no longer uniquifies loop iter names -- the
    # ``UniqueLoopIterators`` post-pass (run from ``SDFGBuilder.build()``
    # via ``_run_post_gen_passes``) renames every ``LoopRegion.loop_var``
    # to a globally-unique ``_loop_it_<N>`` symbol and propagates the
    # rename through the body.  ``emit_loop`` therefore uses the
    # source-Fortran iter name verbatim, and ``iter_map`` is the
    # identity map kept here only so the few callers that still pipe
    # expressions through ``rename_iters`` see a no-op rather than a
    # missing dict.
    if iter_map is None:
        iter_map = dict(ctx.iter_map) if ctx.iter_map else {}

    uid = n.loop_iter

    # ``arr[idx]`` (Fortran 1-based) -> DaCe 0-based form so the
    # LoopRegion's init / cond hit the correct element.
    bound = _fortran_subs_to_dace(n.loop_bound, builder)
    lower_expr = (_fortran_subs_to_dace(n.loop_lower_expr, builder) if n.loop_lower_expr else '')
    lower = lower_expr if lower_expr else (n.loop_lower if n.loop_lower >= 0 else 1)

    # Hoist non-trivial bounds onto pre-LoopRegion symbols so the
    # LoopRegion's init / cond carry only symbol names -- the bridge
    # then doesn't need to embed expression-rewrite logic in bound
    # rendering (the hoisted assignment goes through the same
    # interstate-edge symbol-staging path indirect-array reads use).
    # Bare-symbol bounds are skipped; the staging would be pure noise.
    _b = _hoist_bound_to_symbol(ctx, region, builder, str(bound), "loopend")
    if _b is not None:
        bound = _b
    _l = _hoist_bound_to_symbol(ctx, region, builder, str(lower), "loopbegin")
    if _l is not None:
        lower = _l

    iter_map = {**iter_map, n.loop_iter: uid}

    # ``DO i = a, b, step`` semantics.  Flang's ``fir.do_loop``
    # carries (lower, upper, step) literally  --  for forward step the
    # iter walks lower->upper inclusive; for negative step the
    # MLIR-level lower is actually the START (e.g. NCLV-1 for ``DO
    # JN = NCLV-1, 1, -1``) and upper is the END (1).  The bridge
    # passes them through as ``loop_lower`` and ``loop_bound`` without
    # reordering, so emit_loop is responsible for picking the right
    # one as init.
    step = getattr(n, 'loop_step', 1)
    step_expr = getattr(n, 'loop_step_expr', '') or ''

    if step_expr:
        # Symbolic step  --  ``DO jbnd = jstart, jend, many_fft``
        # where ``many_fft`` is a runtime config integer, or
        # ``DO j = 1, n, stride_arr(idx)`` where the step reads an
        # array element.  Apply the same hoist-to-symbol path the
        # bounds use so the update_expr stays a bare symbol -- this
        # keeps the bridge's "loop iterator / array access / loop
        # bounds are symbols" design consistent, and the Fortran
        # 1-based -> DaCe 0-based conversion of ``arr(idx)`` to
        # ``arr[(idx) - offset_arr_d0]`` happens once on the
        # hoisted interstate-edge assignment instead of being
        # embedded in the loop body.  Treat as forward iteration;
        # runtime-negative symbols yield zero-or-one iterations
        # under ``uid <= bound``, matching Fortran's mismatched-
        # direction trip-count semantics.
        step_expr = _fortran_subs_to_dace(step_expr, builder)
        _s = _hoist_bound_to_symbol(ctx, region, builder, str(step_expr), "loopstep")
        if _s is not None:
            step_expr = _s
        loop = LoopRegion(
            label=f"loop_{uid}_{builder.nid()}",
            condition_expr=f"{uid} < {bound} + 1",
            loop_var=uid,
            initialize_expr=f"{uid} = {lower}",
            update_expr=f"{uid} = {uid} + {step_expr}",
        )
    elif step >= 0:
        loop = LoopRegion(
            label=f"loop_{uid}_{builder.nid()}",
            condition_expr=f"{uid} < {bound} + 1",
            loop_var=uid,
            initialize_expr=f"{uid} = {lower}",
            update_expr=(f"{uid} = {uid} + 1" if step == 1 else f"{uid} = {uid} + {step}"),
        )
    else:
        # Reverse: ``loop_lower`` is the START (the larger value),
        # ``loop_bound`` is the END (the smaller value).  Iter walks
        # from lower DOWN to bound, inclusive.
        loop = LoopRegion(
            label=f"loop_{uid}_{builder.nid()}",
            condition_expr=f"{uid} >= {bound}",
            loop_var=uid,
            initialize_expr=f"{uid} = {lower}",
            update_expr=(f"{uid} = {uid} - 1" if step == -1 else f"{uid} = {uid} + {step}"),
        )
    region.add_node(loop)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, loop, InterstateEdge())
    ctx.cur = loop

    # Cache .children once  --  nanobind copies on every access.
    children = n.children

    child_loops = [c for c in children if c.kind == "loop"]
    child_assigns = [c for c in children if c.kind == "assign"]
    # Anything beyond nested DO loops and plain assignments (IF/ELSE,
    # WHILE, reductions, library-node calls, ...) forces the generic
    # state-machine walk  --  the flat ``body`` tasklet path can't host
    # interstate edges.
    has_structured = any(c.kind not in ("loop", "assign") for c in children)

    if has_structured:
        inner_ctx = _Ctx(ctx.sdfg, builder)
        inner_ctx.iter_map = iter_map
        body_start = loop.add_state(f"body_{builder.nid()}", is_start_block=True)
        inner_ctx.cur = body_start
        builder._emit(inner_ctx, list(children), loop)
        inner_ctx.flush(builder, loop)
    elif child_loops:
        inner_ctx = _Ctx(ctx.sdfg, builder)
        inner_ctx.iter_map = iter_map
        for c in children:
            if c.kind == "loop":
                emit_loop(builder, inner_ctx, c, loop, iter_map)
            elif c.kind == "assign":
                emit_assign(builder, inner_ctx, c, loop)
        inner_ctx.flush(builder, loop)
    elif child_assigns:
        # Inline indirect accesses (``z_kin(edge_idx(jc,k), jk)``) mint a
        # fresh ``<arr>_at<gid>`` SDFG symbol per occurrence; the value is
        # assigned on an interstate edge so a new state is forced before
        # the compute tasklet runs.
        indirect_syms = collect_indirect(builder, child_assigns)

        # Scalar-staged indirection (``ci0 = icidx(je, jb, 1); w(ci0,...)``):
        # the bridge classifies ``ci0`` as a symbol (it feeds an
        # ``hlfir.designate`` index downstream), so the assign cannot land
        # as a tasklet  --  DaCe has no array named ``ci0``.  Lift each such
        # assign onto the same pre->body interstate edge that hosts the
        # inline indirect symbols; the consuming tasklet then reads the
        # symbol value uniformly.  The compute tasklets run on the
        # remaining (array-target) child assigns.
        symbol_assigns = [a for a in child_assigns if a.target in builder.symbols]
        compute_assigns = [a for a in child_assigns if a.target not in builder.symbols]

        # Serialise sibling assigns that share an array as RW (see
        # ``_sibling_rw_hazard``): one state per assign when hazardous.
        serialise = _sibling_rw_hazard(compute_assigns)

        # For each indirect symbol, emit an ``InterstateEdge`` carrying
        # its assignment.  Nested indirection (``idx1[idx2[idx3[i]]]``)
        # mints one symbol per level, innermost-first; placing each one
        # on its OWN interstate edge in that order lets the outer levels
        # read the inner symbol's value through DaCe's normal symbol
        # propagation -- a single edge with all assignments triggers the
        # race-condition validator because edge-side assignments are
        # treated as parallel.  Empty intermediate states are fine; DaCe
        # collapses the chain at codegen time.
        per_sym_assigns: list[tuple[str, str]] = []
        for expr, sym in indirect_syms.items():
            rhs = indirect_to_dace(builder, expr, iter_map, indirect_syms)
            per_sym_assigns.append((sym, rhs))
            if sym not in ctx.sdfg.symbols:
                ctx.sdfg.add_symbol(sym, dace.int64)
        symbol_assign_pairs: list[tuple[str, str]] = []
        for a in symbol_assigns:
            symbol_assign_pairs.append((a.target, array_read_to_dace_expr(builder, a, iter_map, ctx.sdfg)))

        if per_sym_assigns or symbol_assign_pairs:
            pre = loop.add_state(f"pre_{builder.nid()}")
            cur = pre
            # Indirect symbols first (innermost -> outermost).  Each
            # gets a fresh state so its assignment can reference the
            # symbol set on the previous edge.
            for sym, rhs in per_sym_assigns:
                nxt = loop.add_state(f"sym_{sym}_{builder.nid()}")
                loop.add_edge(cur, nxt, InterstateEdge(assignments={sym: rhs}))
                cur = nxt
            # Stage-staged scalar->symbol writes.  Most are independent
            # (``ci0 = icidx(je, jb, 1)``) and share one edge, but some
            # CHAIN (``qvan2_i1 = qvan2_i0 + 1`` right after ``qvan2_i0 =
            # int32(qm) + 1``).  DaCe forbids an interstate assignment
            # whose RHS reads a name written on the SAME edge (no defined
            # order), so a pair whose RHS references a target written
            # EARLIER in the current batch must open a NEW edge (the prior
            # batch then sits on the preceding edge).  Dependency detected
            # via the RHS's SYMBOLIC free-symbols (robust to substrings /
            # the ``int32(...)`` typecast function, which is not a symbol).
            edge = InterstateEdge()
            batch_lhs: set = set()
            for tgt, rhs in symbol_assign_pairs:
                try:
                    rhs_syms = {str(s) for s in dace.symbolic.pystr_to_symbolic(rhs).free_symbols}
                except Exception:
                    rhs_syms = set(re.findall(r'[A-Za-z_]\w*', str(rhs)))
                if edge.assignments and (rhs_syms & batch_lhs):
                    nxt = loop.add_state(f"sym_chain_{builder.nid()}")
                    loop.add_edge(cur, nxt, edge)
                    cur, edge, batch_lhs = nxt, InterstateEdge(), set()
                edge.assignments[tgt] = rhs
                batch_lhs.add(tgt)
            body = loop.add_state('body')
            loop.add_edge(cur, body, edge)
        else:
            body = loop.add_state('body')

        if not serialise:
            for idx, a in enumerate(compute_assigns):
                emit_tasklet(builder, body, a, idx, iter_map, indirect_syms)
        else:
            prev = body
            for idx, a in enumerate(compute_assigns):
                if idx == 0:
                    emit_tasklet(builder, prev, a, idx, iter_map, indirect_syms)
                    continue
                nxt = loop.add_state(f"body_{builder.nid()}")
                loop.add_edge(prev, nxt, InterstateEdge())
                emit_tasklet(builder, nxt, a, idx, iter_map, indirect_syms)
                prev = nxt


def _stage_cond_scalar(builder, ctx, region, pre, sym, cond, cond_accesses):
    """Compute an array-dependent control-flow condition into a SCALAR
    transient via a tasklet; return ``(new_pre_state, scalar_name)``.

    Per the codeblock rule the LoopRegion / ConditionalBlock then reads the
    scalar by its BARE name (no ``[0]``).  Internal condition values are
    never length-1 Arrays -- those are reserved for values returned outside
    the SDFG.  Registering ``sym`` in ``builder.scalars`` routes
    ``emit_tasklet``'s scalar-output path (``Memlet(sym, subset='0')``); the
    array reads still wire through per-occurrence ``_in_<arr>_<n>``
    connectors + memlets.
    """
    from types import SimpleNamespace
    if sym not in ctx.sdfg.arrays:
        ctx.sdfg.add_scalar(sym, dace.int64, transient=True, find_new_name=False)
    builder.scalars.setdefault(
        sym,
        SimpleNamespace(fortran_name=sym, intent='', dtype='int64', rank=0,
                        is_dynamic=False, role='scalar', shape_symbols=[], lower_bounds=[]))
    pre = _anchor_views_referenced_in_expr(builder, cond, region, pre, ctx.sdfg)
    nxt = region.add_state(f"pre_{sym}")
    region.add_edge(pre, nxt, InterstateEdge())
    ctx.cur = nxt
    synth = SimpleNamespace(kind='assign', target=sym, expr=cond,
                            target_is_array=False, accesses=cond_accesses)
    emit_tasklet(builder, nxt, synth, builder.nid(), ctx.iter_map)
    return nxt, sym


def emit_while(builder, ctx: '_Ctx', n, region):
    """Fortran ``DO WHILE``  --  lifted by ``lift-cf-to-scf`` into scf.while
    and extracted as ``kind="while"``.  Emit a DaCe LoopRegion whose
    condition is ``True`` (the bridge's faithful walker folds any
    break-on-false into a ``break`` child node inside the body).
    """
    pre = ctx.flush_and_ensure(builder, region)
    # ``?`` is the bridge's placeholder for an unextractable condition.
    # Default to ``True`` so ast.parse succeeds and leaves the faithful
    # structure visible in the SDFG for inspection.
    cond = n.condition if n.condition and n.condition != "?" else "True"

    # Mirror the ``emit_cond`` cond-rewrite pipeline so a real
    # ``DO WHILE (a(i) > thr)`` doesn't relapse the bugs already
    # fixed for ``IF``.  Today the bridge folds break-on-false into
    # ``break`` nodes and sends ``True`` here -- but the day a real
    # condition lands, every gap below was an emit_cond fix:
    cond = _rewrite_section_aliases_in_expr(builder, cond)

    # When the condition references arrays, lift it through a
    # tasklet-into-scalar-transient (same per-occurrence connector
    # machinery emit_tasklet uses for assigns) so the array reads
    # get proper memlets instead of bare-pointer interstate-edge
    # free symbols.  The scalar-``[0]`` rewrite for inout scalars
    # is skipped on the lift path (would survive past the rewriter
    # and surface as ``_in_<nm>[0]`` against a scalar connector).
    cond_accesses = []
    if n.accesses:
        seen_acc = set()
        for ac in n.accesses:
            key = (ac.array_name, tuple(ac.index_exprs), ac.is_read)
            if key in seen_acc:
                continue
            seen_acc.add(key)
            cond_accesses.append(ac)
    cond_array_reads = [ac for ac in cond_accesses
                        if ac.is_read and ac.array_name in builder.arrays]
    will_lift = bool(cond_array_reads)
    if will_lift:
        from collections import Counter
        text_occ = Counter()
        for tok in re.findall(r'\b([A-Za-z_]\w*)\b', cond):
            if tok in builder.arrays:
                text_occ[tok] += 1
        access_count = Counter()
        for ac in cond_array_reads:
            access_count[ac.array_name] += 1
        mismatched = any(text_occ[k] != access_count[k]
                         for k in set(text_occ) | set(access_count))
        if mismatched:
            will_lift = False
    if not will_lift:
        # Legacy path: rewrite intent(out)/inout scalars to ``nm[0]``
        # so the LoopRegion's condition_expr reads the size-1
        # backing array's element 0.
        for nm, v in builder.scalars.items():
            if v.intent in ('out', 'inout'):
                cond = re.sub(rf'\b{re.escape(nm)}\b', f"{nm}[0]", cond)

    if will_lift and not _is_trivial_bound(cond):
        # Lift an array-dependent loop condition into a SCALAR transient;
        # the LoopRegion reads it by its bare name (no ``[0]``).
        sym = f"while_cond_{builder.nid()}"
        pre, cond = _stage_cond_scalar(builder, ctx, region, pre, sym, cond, cond_accesses)

    loop = LoopRegion(label=f"while_{builder.nid()}", condition_expr=cond)
    region.add_node(loop)
    if pre is not None:
        region.add_edge(pre, loop, InterstateEdge())
    ctx.cur = loop

    body_start = loop.add_state(f"while_body_{builder.nid()}", is_start_block=True)
    inner_ctx = _Ctx(ctx.sdfg, builder)
    inner_ctx.cur = body_start
    builder._emit(inner_ctx, list(n.children), loop)
    inner_ctx.flush(builder, loop)


def emit_cond(builder, ctx: '_Ctx', n, region):
    """``if (cond) then ... else ... end if`` -> ``ConditionalBlock`` with
    a ``ControlFlowRegion`` per branch.  Subsequent statements land in a
    fresh successor state wired from the block.
    """
    pre = ctx.flush_and_ensure(builder, region)

    cond = n.condition if n.condition and n.condition != "?" else "True"
    # Section-alias dummies (trivial section slices) have no SDFG
    # descriptor  --  rewrite ``dummy[i, j]`` references in the condition
    # to ``source[i, j, k_const]`` via the view_dim_map.  Without this,
    # the interstate-edge assignment carries the dummy name as a free
    # symbol and ``sdfg.arglist`` raises a KeyError when scanning.
    cond = _rewrite_section_aliases_in_expr(builder, cond)
    # Scalar OUTPUTS land as size-1 Arrays on the SDFG signature, so
    # referring to a bare name in a branch condition would pick up the
    # array pointer.  Subscript each one to read element 0.  Scalar
    # INPUTS (``intent(in)`` / ``VALUE``) are true Scalars and need no
    # subscript -- they're addressable as the bare name in C++.
    #
    # Skip when the condition will be lifted into a tasklet
    # (downstream array-read path): emit_tasklet's
    # ``_rewrite_read_connectors`` handles scalar connectors as
    # ``_in_<nm>`` -- a textual ``[0]`` already in the body would
    # leak through as ``_in_<nm>[0]`` and fail validation as a
    # subscript-on-scalar (the IndexError that surfaced
    # test_sqrt_in_if / test_exp_in_if).
    will_lift = bool(n.accesses) and any(
        ac.is_read and ac.array_name in builder.arrays for ac in n.accesses)
    if not will_lift:
        for nm, v in builder.scalars.items():
            if v.intent in ('out', 'inout'):
                cond = re.sub(rf'\b{re.escape(nm)}\b', f"{nm}[0]", cond)

    # Hoist non-trivial conditions to a pre-state symbol so the
    # ConditionalBlock branch carries only a symbol name -- one path
    # for every IF lowering, no per-branch expression-rewrite logic.
    # Trivial cases (a bare name or ``True`` / ``False``) skip the
    # staging.
    if not _is_trivial_bound(cond):
        # Conditions that reference ARRAYS (e.g. graupel's
        # ``MAX(q_x_1[1,iv,k], q_x_1[5,iv,k], q_x_1[6,iv,k]) > qmin``)
        # cannot be lifted onto a plain interstate-edge assignment --
        # DaCe treats bare array names there as Symbols (no connector
        # + no memlet) and the C++ codegen emits the data pointer
        # where a scalar was expected (``double* > 1e-15`` type-error
        # in graupel's ``if_cond_38``).
        #
        # Detection: if the bridge populated ``n.accesses`` for the
        # conditional AND any of the read targets is in
        # ``builder.arrays``, route through the per-occurrence-
        # connector tasklet path (same machinery ``emit_tasklet``
        # uses for assigns) into a fresh ``if_cond_<nid>`` scalar
        # transient.  The conditional then reads the transient
        # element-0 as a regular scalar.
        # Dedupe accesses by ``(array_name, index_exprs)`` -- the
        # bridge walker recurses through ``arith.maximumf`` /
        # ``cmpf`` / ``select`` chains and visits the same
        # ``hlfir.designate`` from multiple paths, so the raw
        # accesses list has duplicates (e.g. 20 entries for a,b,c
        # each accessed once textually).  Dedupe to 1 entry per
        # unique (name, indices) so the per-occurrence connector
        # count matches the text-occurrence count.
        cond_accesses = []
        if n.accesses:
            seen_acc = set()
            for ac in n.accesses:
                key = (ac.array_name, tuple(ac.index_exprs), ac.is_read)
                if key in seen_acc:
                    continue
                seen_acc.add(key)
                cond_accesses.append(ac)
        cond_array_reads = [ac for ac in cond_accesses
                             if ac.is_read and ac.array_name in builder.arrays]
        # Only lift via the tasklet path when accesses can be matched
        # 1:1 to text occurrences -- otherwise we mint connectors that
        # the rewritten code references but the access list can't
        # bind to memlets, producing the ``_in_<arr>_<n>`` unresolved-
        # free-symbol error.  Mismatch surfaces when the bridge
        # collapsed a slice ``arr[i, 0:4]`` into ONE access while
        # ``buildBoolExpr`` expanded the slice into FOUR textual
        # ``arr[i, 0], arr[i, 1], ...`` references (graupel's
        # ``MIN(kmin[iv,0:4]) >`` shape).
        if cond_array_reads:
            from collections import Counter
            text_occ = Counter()
            for tok in re.findall(r'\b([A-Za-z_]\w*)\b', cond):
                if tok in builder.arrays:
                    text_occ[tok] += 1
            access_count = Counter()
            for ac in cond_array_reads:
                access_count[ac.array_name] += 1
            mismatched = any(text_occ[k] != access_count[k]
                              for k in set(text_occ) | set(access_count))
            if mismatched:
                cond_array_reads = []  # fall through to legacy path
        if cond_array_reads:
            # Lift the array-reading branch condition into a SCALAR
            # transient (DaCe takes any numeric-truthy value on the
            # branch); the ConditionalBlock reads it by its bare name.
            # ``emit_tasklet``'s per-occurrence array-read connector
            # machinery wires each ``arr[i, k]`` to ``_in_arr_N`` + memlet.
            sym = f"if_cond_{builder.nid()}"
            pre, cond = _stage_cond_scalar(builder, ctx, region, pre, sym, cond, cond_accesses)
        else:
            sym = f"if_cond_{builder.nid()}"
            if sym not in ctx.sdfg.symbols:
                ctx.sdfg.add_symbol(sym, dace.int64)
            # If the condition references any view_alias array, anchor it
            # in a state upstream of the interstate-edge assignment so
            # DaCe's framecode finds a real AccessNode first.
            pre = _anchor_views_referenced_in_expr(builder, cond, region, pre, ctx.sdfg)
            nxt = region.add_state(f"pre_{sym}")
            region.add_edge(pre, nxt, InterstateEdge(assignments={sym: cond}))
            pre = nxt
            ctx.cur = nxt
            cond = sym

    uid = builder.nid()
    cond_block = ConditionalBlock(f"if_{uid}")
    region.add_node(cond_block, ensure_unique_name=True)
    if pre is not None:
        region.add_edge(pre, cond_block, InterstateEdge())

    def _populate_branch(label: str, children: list) -> ControlFlowRegion:
        branch = ControlFlowRegion(label, sdfg=ctx.sdfg)
        inner = _Ctx(ctx.sdfg, builder)
        inner.iter_map = ctx.iter_map
        builder._emit(inner, children, branch)
        inner.flush(builder, branch)
        # An empty branch (e.g. the EXIT arm of a Flang-lowered DO+EXIT)
        # still needs a start block, otherwise the validator complains.
        if len(branch.nodes()) == 0:
            branch.add_state(f"{label}_noop", is_start_block=True)
        return branch

    then_region = _populate_branch(f"if_{uid}_then", list(n.children))
    cond_block.add_branch(cond, then_region)

    else_children = list(n.else_children)
    if else_children:
        else_region = _populate_branch(f"if_{uid}_else", else_children)
        cond_block.add_branch(None, else_region)

    # The ConditionalBlock is itself the "current" control-flow node;
    # subsequent statements get a fresh state edge-connected to it.
    ctx.cur = cond_block
