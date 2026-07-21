"""Make intra-state Fortran statement order explicit.

A run of consecutive Fortran statements is lowered into ONE ``SDFGState``.  Two statements that
touch the same container land in disjoint dataflow components whenever the second one does not
consume the first one's value (``x = x - c`` / ``d = x - o`` / ``x = o``): nothing in the graph
orders them, so the emitted order is a scheduler tie-break and any later transformation may flip
it.  This pass adds empty (ordering-only) memlets that pin the source order into the graph.

The rule mirrors DaCe's ``StateFusionExtended``: an ordering edge never targets the reader itself,
it targets the reader's CONSUMER -- the read is only done once the node that consumes it has run,
so a later write must be sequenced after that consumer, not after the access node.

This runs on the realised graph, so it needs no per-emitter cooperation and no name matching.  Both
matter: ``emit_cfg``'s emit-time guards split states only on the paths that reach them (a scalar
assign flushed from ``_Ctx.flush`` never does), and they compare Fortran names while ``access.acc``
has already resolved every alias onto the source container's access node.

Source order is recovered from ``state.node_id``: access nodes are created while their statement is
emitted, so increasing node id is increasing Fortran statement order.  The pass therefore has to run
on the freshly emitted graph, before anything renumbers nodes.
"""
from collections import defaultdict

from dace import Memlet
from dace.sdfg import nodes


def descendants(state, source):
    """Every node reachable from ``source`` along dataflow edges."""
    seen = set()
    stack = [source]
    while stack:
        node = stack.pop()
        for edge in state.out_edges(node):
            if edge.dst not in seen:
                seen.add(edge.dst)
                stack.append(edge.dst)
    return seen


def completion_anchor(state, node):
    """Node whose completion implies ``node`` has run (a scope stands for its whole body)."""
    return state.exit_node(node) if isinstance(node, nodes.EntryNode) else node


def start_anchor(state, node):
    """Node whose start implies ``node`` has not run yet (a scope stands for its whole body)."""
    return state.entry_node(node) if isinstance(node, nodes.ExitNode) else node


def store_anchors(state, node, seen=None):
    """Nodes that perform the store into ``node``.

    Walks back through intervening AccessNodes so an aliased write (``p(i) = c(i)`` with
    ``p => a``, lowered as ``tasklet -> p -> a``) anchors on the tasklet, not on the view: an
    edge into the view would leave the tasklet itself free to float above the earlier access.
    """
    seen = {id(node)} if seen is None else seen
    anchors = []
    for edge in state.in_edges(node):
        source = edge.src
        if not isinstance(source, nodes.AccessNode):
            anchors.append(start_anchor(state, source))
        elif id(source) not in seen:
            seen.add(id(source))
            anchors.extend(store_anchors(state, source, seen))
    return anchors or [node]


def load_anchors(state, node, seen=None):
    """Nodes whose completion implies every read of ``node`` has happened.

    The consumer rule: a later write is sequenced after the reader's CONSUMER, never after the
    reader.  Intervening AccessNodes (a source -> view link) are walked through for the same
    reason ``store_anchors`` walks back through them.
    """
    seen = {id(node)} if seen is None else seen
    anchors = []
    for edge in state.out_edges(node):
        sink = edge.dst
        if not isinstance(sink, nodes.AccessNode):
            anchors.append(completion_anchor(state, sink))
        elif id(sink) not in seen:
            seen.add(id(sink))
            anchors.extend(load_anchors(state, sink, seen))
    return anchors or [node]


def order_state(state):
    """Add ordering edges for every unordered same-container pair in ``state``.

    :return: list of ``(container, src, dst)`` triples that had to be skipped to keep the state
        acyclic.
    """
    scopes = state.scope_dict()
    by_container = defaultdict(list)
    for node in state.data_nodes():
        if scopes[node] is None:
            by_container[node.root_data].append(node)

    skipped = []
    for container, access_nodes in by_container.items():
        if len(access_nodes) < 2:
            continue
        ordered = sorted(access_nodes, key=state.node_id)
        reach = {id(n): descendants(state, n) for n in ordered}

        for index, earlier in enumerate(ordered):
            for later in ordered[index + 1:]:
                if later in reach[id(earlier)] or earlier in reach[id(later)]:
                    continue
                earlier_writes = state.in_degree(earlier) > 0
                later_writes = state.in_degree(later) > 0
                if not earlier_writes and not later_writes:
                    continue
                if later_writes:
                    sources = load_anchors(state, earlier)
                    targets = store_anchors(state, later)
                else:
                    sources = [earlier]
                    targets = [later]
                for src in sources:
                    for dst in targets:
                        if src is dst:
                            continue
                        src_reach = reach.get(id(src))
                        if src_reach is None:
                            src_reach = descendants(state, src)
                            reach[id(src)] = src_reach
                        if dst in src_reach:
                            continue
                        dst_reach = reach.get(id(dst))
                        if dst_reach is None:
                            dst_reach = descendants(state, dst)
                            reach[id(dst)] = dst_reach
                        if src in dst_reach:
                            skipped.append((container, src, dst))
                            continue
                        state.add_nedge(src, dst, Memlet())
                        gained = dst_reach | {dst}
                        for key, known in reach.items():
                            if key == id(src) or src in known:
                                known |= gained
    return skipped


def enforce_statement_order(sdfg):
    """Pin Fortran statement order into every state of ``sdfg`` and its nested SDFGs."""
    skipped = []
    for nested in sdfg.all_sdfgs_recursive():
        for state in nested.states():
            skipped.extend(order_state(state))
    return skipped
