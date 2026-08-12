"""Fuse a full-range copy map with an adjacent endpoint-write map.

Pattern (extracted from velocity_no_nproma stage 3, e.g. ``z_w_con_c``):

    state A:
        Map [lev = K_LO:K_HI, *other_axes]   # e.g. lev = 0..89, row = startidx-1..endidx
            <body>                            # writes ``dst[*other_axes, lev] = expr(...)``
            -> dst[*other_axes, lev]

    state B (immediate, unconditional successor of A):
        Map [*other_axes]                     # same row range, NO ``lev`` axis
            Tasklet (constant, e.g. = 0.0)    # writes ``dst[*other_axes, K_HI] = const``
            -> dst[*other_axes, K_HI]

The two writes together cover ``dst[*other_axes, K_LO:K_HI+1]``. The
fused result is a single map iterating ``lev = K_LO:K_HI+1`` whose body
dispatches via a ``ConditionalBlock``:

    Map [lev = K_LO:K_HI+1, *other_axes]
        NestedSDFG containing:
            ConditionalBlock
                branch ("lev < K_HI"): the original A-body tasklet
                branch (else):         the original B-body tasklet
        -> dst[*other_axes, lev]

State B is removed; A's interstate-edge successors are rewired to point
at B's old successors.

The transformation only fires when the two states are sequentially
adjacent and connected by an unconditional interstate edge (cond `1` /
True) -- otherwise we'd risk reordering side effects from intervening
work. State A and B may live in any ``ControlFlowRegion`` (top-level
SDFG, ``loop_body`` of a ``LoopRegion``, etc.) as long as their parent
graph supports state insertion / removal.

Constraints checked:

- A's map is exactly one axis larger than B's map, and that extra axis
  iterates a contiguous integer range ending at the constant ``K_HI``
  that B writes.
- Both maps' OTHER axes match (same params, same range expressions).
- B's tasklet has no inputs, one scalar output, and its code is a pure
  literal assignment to that output (e.g. ``out = 0.0``).
- Both MapExits write into the same destination AccessNode, with
  matching subset shapes (full row range on the non-fused axes).
- The interstate edge A->B is unconditional; A has exactly one
  successor and B has exactly one predecessor.

The transformation is NOT velocity-specific. It applies to any SDFG
with the (full-range copy, endpoint-constant-write) shape pair.
"""

import copy as _copy
from typing import Dict, List, Optional, Tuple

import dace
from dace import dtypes, memlet as mm
from dace.properties import CodeBlock
from dace.sdfg import nodes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


def _scope_subgraph_nodes(state: SDFGState, map_entry: nodes.MapEntry):
    """All nodes inside the scope of ``map_entry`` (excluding the entry/exit)."""
    return [n for n in state.scope_subgraph(map_entry).nodes()
            if n is not map_entry and n is not state.exit_node(map_entry)]


def _single_top_level_map(state: SDFGState) -> Optional[nodes.MapEntry]:
    """Return the unique top-level (no enclosing scope) MapEntry in
    ``state``, or None if there is zero or more than one."""
    candidates = [n for n in state.nodes()
                  if isinstance(n, nodes.MapEntry) and state.entry_node(n) is None]
    return candidates[0] if len(candidates) == 1 else None


def _writes_to(state: SDFGState, map_entry: nodes.MapEntry) -> List[Tuple[nodes.AccessNode, dace.Memlet]]:
    """The (AccessNode, memlet) pairs written by ``map_entry``'s
    MapExit. Single-output maps return a single pair."""
    mx = state.exit_node(map_entry)
    out: List[Tuple[nodes.AccessNode, dace.Memlet]] = []
    for oe in state.out_edges(mx):
        if isinstance(oe.dst, nodes.AccessNode):
            out.append((oe.dst, oe.data))
    return out


def _is_constant_zero_init_tasklet(t: nodes.Tasklet) -> Optional[str]:
    """If ``t`` is a single-output literal-assign tasklet (typically
    ``out = 0.0``), return the constant string. Otherwise None."""
    if not isinstance(t, nodes.Tasklet):
        return None
    if len(t.in_connectors) != 0 or len(t.out_connectors) != 1:
        return None
    out_conn = next(iter(t.out_connectors))
    code = t.code.as_string.strip().rstrip(';').strip()
    # Expect ``<out> = <literal>`` (Python or C++).
    prefix = f"{out_conn} = "
    if not code.startswith(prefix):
        return None
    rhs = code[len(prefix):].strip()
    return rhs


def _interstate_successors(parent: ControlFlowRegion, state: SDFGState):
    return list(parent.out_edges(state))


def _interstate_predecessors(parent: ControlFlowRegion, state: SDFGState):
    return list(parent.in_edges(state))


def _direct_unconditional(parent: ControlFlowRegion, src: SDFGState, dst: SDFGState) -> bool:
    edges = [e for e in parent.out_edges(src) if e.dst is dst]
    if len(edges) != 1:
        return False
    cond = edges[0].data.condition
    cond_str = cond.as_string.strip().lower()
    return cond_str in ('1', 'true') and not edges[0].data.assignments


def _identify_endpoint_axis(full_params, full_ranges, endp_params, endp_ranges,
                            endp_subset, full_subset) -> Optional[Tuple[int, str, int]]:
    """Determine which axis of ``dst`` is the one being fused.

    Returns (axis_index, fused_param, k_hi) or None on no match.

    ``axis_index`` is the dimension (in ``dst``'s shape) along which
    A iterates [K_LO:K_HI] and B writes the single index K_HI.
    ``fused_param`` is the iteration variable name from A. ``k_hi`` is
    the integer endpoint that B writes (one past the end of A's range).
    """
    # B's params are a subset of A's; the missing one is the fused axis.
    missing = [p for p in full_params if p not in endp_params]
    if len(missing) != 1:
        return None
    fused_param = missing[0]
    fused_idx_in_full = full_params.index(fused_param)
    fused_range = full_ranges[fused_idx_in_full]
    # Range is (begin, end, step); end is INCLUSIVE for DaCe.
    begin, end, step = fused_range
    if step != 1 and str(step) != '1':
        return None

    # Find the dst-axis whose subset in A is the fused range and in B is
    # a single index equal to (end + 1) (i.e. K_HI in our docstring,
    # since DaCe end is inclusive).
    dst_ndim = len(full_subset)
    for axis in range(dst_ndim):
        a_lo, a_hi, a_step = full_subset[axis]
        b_lo, b_hi, b_step = endp_subset[axis]
        if str(a_lo) == str(b_lo) and str(a_hi) == str(b_hi):
            # Same range on this axis -- not the fused axis.
            continue
        # Otherwise this axis must be the fused one: A spans a range,
        # B is a single point at the endpoint.
        if str(b_lo) != str(b_hi):
            return None
        # B's single index == A's hi + 1 (DaCe end is inclusive, K_HI
        # in docstring is the exclusive upper bound = b_lo).
        try:
            k_hi_b = int(str(b_lo))
        except (TypeError, ValueError):
            return None
        # A's high end + 1 should equal B's index
        try:
            a_hi_int = int(str(a_hi))
        except (TypeError, ValueError):
            return None
        if a_hi_int + 1 != k_hi_b:
            return None
        return (axis, fused_param, k_hi_b)
    return None


def _params_and_ranges_match(a_params, a_ranges, b_params, b_ranges) -> bool:
    """B's (params, ranges) is a subset of A's (params, ranges)
    preserving order, except for one missing param."""
    b_iter = iter(zip(b_params, b_ranges))
    try:
        bp, br = next(b_iter)
    except StopIteration:
        return False
    for ap, ar in zip(a_params, a_ranges):
        if ap == bp:
            if str(ar) != str(br):
                return False
            try:
                bp, br = next(b_iter)
            except StopIteration:
                return True
    return False  # B had params not in A


def _try_fuse_pair(parent: ControlFlowRegion, state_a: SDFGState, state_b: SDFGState,
                   verbose: bool) -> bool:
    """Attempt to fuse (A, B). Returns True on success (mutation done)."""
    me_a = _single_top_level_map(state_a)
    me_b = _single_top_level_map(state_b)
    if me_a is None or me_b is None:
        return False

    writes_a = _writes_to(state_a, me_a)
    writes_b = _writes_to(state_b, me_b)
    if len(writes_a) != 1 or len(writes_b) != 1:
        return False
    dst_a, memlet_a = writes_a[0]
    dst_b, memlet_b = writes_b[0]
    if dst_a.data != dst_b.data:
        return False

    # B must be a constant-assign tasklet under its map.
    b_inner = _scope_subgraph_nodes(state_b, me_b)
    b_tasklets = [n for n in b_inner if isinstance(n, nodes.Tasklet)]
    if len(b_tasklets) != 1:
        return False
    b_const_rhs = _is_constant_zero_init_tasklet(b_tasklets[0])
    if b_const_rhs is None:
        return False

    a_params, a_ranges = list(me_a.map.params), list(me_a.map.range)
    b_params, b_ranges = list(me_b.map.params), list(me_b.map.range)
    if not _params_and_ranges_match(a_params, a_ranges, b_params, b_ranges):
        return False

    info = _identify_endpoint_axis(a_params, a_ranges, b_params, b_ranges,
                                   memlet_b.subset, memlet_a.subset)
    if info is None:
        return False
    fused_axis, fused_param, k_hi = info

    if verbose:
        print(f"[fuse_full_and_endpoint] candidate: dst={dst_a.data}  "
              f"axis={fused_axis}  fused_param={fused_param}  k_hi={k_hi}")

    # Build the fused state; it replaces state_a + state_b.
    sdfg = parent.sdfg if hasattr(parent, 'sdfg') else state_a.parent
    fused_state = parent.add_state(f"_fused_{state_a.label}_{state_b.label}")

    # A's map params/ranges with the fused axis extended to K_HI+1.
    new_ranges = list(a_ranges)
    begin, _, step = new_ranges[a_params.index(fused_param)]
    new_ranges[a_params.index(fused_param)] = (begin, k_hi, step)

    # Move A's body INTO an inner SDFG with a ConditionalBlock dispatch
    # so the read of A's source is gated on (fused_param < k_hi).
    inner = SDFG(f"_fused_body_{dst_a.data}")
    inner.add_symbol(fused_param, dace.int64)
    cb = ConditionalBlock(f"cb_{dst_a.data}_endpoint")
    inner.add_node(cb, is_start_block=True)

    # The "main" branch: original A's body, copied verbatim.
    main_region = ControlFlowRegion(label=f"_main_{dst_a.data}", sdfg=inner)
    main_state = main_region.add_state("copy", is_start_block=True)
    _copy_inner_scope(state_a, me_a, inner, main_state, dst_a.data, "out")

    # The "endpoint" branch: B's constant tasklet. ``out`` is already
    # registered as a scalar in the inner SDFG by ``_copy_inner_scope``.
    endp_region = ControlFlowRegion(label=f"_endpoint_{dst_a.data}", sdfg=inner)
    endp_state = endp_region.add_state("zero", is_start_block=True)
    et = endp_state.add_tasklet("endp", set(), {"o"}, f"o = {b_const_rhs}")
    endp_state.add_edge(et, "o", endp_state.add_write("out"), None, mm.Memlet("out[0]"))

    cb.add_branch(CodeBlock(f"{fused_param} < {k_hi}"), main_region)
    cb.add_branch(CodeBlock("True"), endp_region)

    # Build the outer fused map.
    map_ranges = {p: f"{r[0]}:{r[1] + 1}:{r[2]}" for p, r in zip(a_params, new_ranges)}
    inputs_to_inner = _replicate_external_reads(state_a, me_a, fused_state, inner)
    out_arr = sdfg.arrays[dst_a.data]
    # Inner SDFG sees all of the fused map's params as symbols (used by
    # subset expressions inside its branches), plus any free symbol the
    # original A-body referenced.
    for p in a_params:
        if p not in inner.symbols:
            inner.add_symbol(p, dace.int64)
    nsdfg_node = fused_state.add_nested_sdfg(
        inner,
        set(inputs_to_inner.keys()),
        {"out"},
        symbol_mapping={s: s for s in inner.free_symbols},
    )
    me, mx = fused_state.add_map("fused", map_ranges)
    # External AccessNodes: one per input array, plus the destination.
    ext_reads = {arr_name: fused_state.add_read(arr_name) for arr_name in inputs_to_inner}
    ext_write = fused_state.add_write(dst_a.data)

    for arr_name, (full_subset, inner_conn) in inputs_to_inner.items():
        in_conn = f"IN_{arr_name}"
        out_conn = f"OUT_{arr_name}"
        me.add_in_connector(in_conn)
        me.add_out_connector(out_conn)
        fused_state.add_edge(ext_reads[arr_name], None, me, in_conn,
                             mm.Memlet(data=arr_name, subset=full_subset))
        fused_state.add_edge(me, out_conn, nsdfg_node, inner_conn,
                             mm.Memlet(data=arr_name, subset=full_subset))

    # Write side: NSDFG out -> MapExit -> dst.
    #
    # The PER-ITERATION memlet (NSDFG -> MapExit edge) must reference a
    # SINGLE element of ``dst``, addressed by the map's iteration vars
    # -- e.g. ``dst[row, lev]``. Reuse the original A-tasklet's output
    # memlet for this; it already has the right shape, with all
    # subscripts being single-index map params.
    inner_write_subset = _original_inner_write_subset(state_a, me_a, dst_a.data)

    mx.add_in_connector("IN_out")
    mx.add_out_connector("OUT_out")
    fused_state.add_edge(nsdfg_node, "out", mx, "IN_out",
                         mm.Memlet(data=dst_a.data, subset=inner_write_subset))

    # The OUTER memlet (MapExit -> AccessNode) is the full union footprint
    # along the fused axis (extended from ``A``'s subset).
    out_subset_outer = list(memlet_a.subset)
    out_subset_outer[fused_axis] = (
        new_ranges[a_params.index(fused_param)][0],
        new_ranges[a_params.index(fused_param)][1],
        new_ranges[a_params.index(fused_param)][2],
    )
    fused_state.add_edge(mx, "OUT_out", ext_write, None,
                         mm.Memlet(data=dst_a.data, subset=dace.subsets.Range(out_subset_outer)))

    # Re-wire interstate edges: A's predecessors -> fused_state -> B's successors.
    for ie in list(parent.in_edges(state_a)):
        parent.add_edge(ie.src, fused_state, ie.data)
        parent.remove_edge(ie)
    for oe in list(parent.out_edges(state_b)):
        parent.add_edge(fused_state, oe.dst, oe.data)
        parent.remove_edge(oe)

    # Drop the unconditional A->B edge and the original states.
    for e in list(parent.edges_between(state_a, state_b)):
        parent.remove_edge(e)
    parent.remove_node(state_a)
    parent.remove_node(state_b)

    return True


def _scalar_dtype_of(state: SDFGState, map_entry: nodes.MapEntry, dst_name: str):
    sdfg = state.parent
    return sdfg.arrays[dst_name].dtype


def _copy_inner_scope(src_state: SDFGState, src_me: nodes.MapEntry,
                      inner_sdfg: SDFG, dst_state: SDFGState,
                      dst_arr_name: str, out_scalar: str):
    """Copy the tasklet + connections inside ``src_me``'s scope into
    ``dst_state``. The tasklet's output ``dst_arr_name``-write becomes
    a write to scalar ``out_scalar`` in ``inner_sdfg``."""
    inner_sdfg.add_scalar(out_scalar, src_state.parent.arrays[dst_arr_name].dtype, transient=False)
    src_mx = src_state.exit_node(src_me)
    inner_nodes = [n for n in src_state.scope_subgraph(src_me).nodes()
                   if n is not src_me and n is not src_mx]
    for n in inner_nodes:
        if isinstance(n, nodes.Tasklet):
            new_t = dst_state.add_tasklet(n.label, set(n.in_connectors), set(n.out_connectors),
                                          n.code.as_string, language=n.language)
            for ie in src_state.in_edges(n):
                if ie.src is src_me:
                    arr_name = ie.data.data
                    if arr_name not in inner_sdfg.arrays:
                        # Mirror the outer descriptor as a scalar input
                        ext_arr = src_state.parent.arrays[arr_name]
                        inner_sdfg.add_array(arr_name, ext_arr.shape, ext_arr.dtype,
                                             transient=False, strides=ext_arr.strides)
                    rd = dst_state.add_read(arr_name)
                    dst_state.add_edge(rd, None, new_t, ie.dst_conn,
                                       mm.Memlet(data=arr_name, subset=str(ie.data.subset)))
            for oe in src_state.out_edges(n):
                if oe.dst is src_mx:
                    wr = dst_state.add_write(out_scalar)
                    dst_state.add_edge(new_t, oe.src_conn, wr, None,
                                       mm.Memlet(f"{out_scalar}[0]"))


def _replicate_external_reads(src_state: SDFGState, src_me: nodes.MapEntry,
                              fused_state: SDFGState, inner_sdfg: SDFG):
    """Discover the external arrays the original A-map reads and return
    {arr_name: (subset_str_for_outer_memlet, inner_connector_name)}."""
    out: Dict[str, Tuple[str, str]] = {}
    for ie in src_state.in_edges(src_me):
        if not isinstance(ie.src, nodes.AccessNode):
            continue
        out[ie.src.data] = (str(ie.data.subset), ie.src.data)
    return out


def _build_indexed_subset(orig_subset, axis: int, param_name: str):
    """Return a Range that's identical to ``orig_subset`` except that
    ``axis`` is replaced with the single index ``param_name``."""
    new = list(orig_subset)
    new[axis] = (param_name, param_name, 1)
    return dace.subsets.Range(new)


def _original_inner_write_subset(state: SDFGState, map_entry: nodes.MapEntry,
                                 dst_name: str):
    """Locate the original tasklet-side write memlet to ``dst_name``
    inside ``map_entry``'s scope and return its subset (a per-iteration
    address like ``dst[row, lev]``).

    Used by the fuser to wire the new NSDFG -> outer-MapExit edge with
    the same single-element memlet the original A-body produced --
    avoids reconstructing the (axis -> map-param) mapping by hand.
    """
    mx = state.exit_node(map_entry)
    for ie in state.in_edges(mx):
        if ie.data is not None and ie.data.data == dst_name:
            return dace.subsets.Range(list(ie.data.subset))
    raise ValueError(
        f"no inner write memlet for {dst_name!r} found inside "
        f"map {map_entry.label!r} in state {state.label!r}"
    )


def fuse_full_and_endpoint(sdfg: SDFG, target_array: Optional[str] = None,
                           verbose: bool = False) -> int:
    """Find every (full-range copy, endpoint-constant-write) adjacent
    pair in ``sdfg`` and fuse them in place. When ``target_array`` is
    set, only fuse pairs whose destination is that array.

    Returns the number of fusions applied. The pass walks both the
    top-level SDFG and any ``ControlFlowRegion`` (e.g. ``loop_body``)
    nested inside it.
    """
    fused = 0
    parents: List[ControlFlowRegion] = [sdfg]
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, ControlFlowRegion) and n is not sdfg:
            parents.append(n)

    for parent in parents:
        # Walk pairs in topological order; rewriting invalidates the
        # traversal so re-collect after each successful fuse.
        progress = True
        while progress:
            progress = False
            for state in list(parent.nodes()):
                if not isinstance(state, SDFGState):
                    continue
                succs = [e for e in parent.out_edges(state) if isinstance(e.dst, SDFGState)]
                if len(succs) != 1:
                    continue
                nxt = succs[0].dst
                if not _direct_unconditional(parent, state, nxt):
                    continue
                if target_array is not None:
                    me_a = _single_top_level_map(state)
                    if me_a is None:
                        continue
                    writes = _writes_to(state, me_a)
                    if len(writes) != 1 or writes[0][0].data != target_array:
                        continue
                if _try_fuse_pair(parent, state, nxt, verbose):
                    fused += 1
                    progress = True
                    break
    return fused
