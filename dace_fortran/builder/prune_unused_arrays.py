"""Prune unused-but-still-registered array descriptors from a built SDFG.

Struct-flatten + marshal-expansion mints SoA companions for every derived-type
member, including ones only used inside an external callee whose expansion
refused (box/pointer/allocatable/dynamic-shape) -- these stay in
``sdfg.arrays`` with zero references, bloating the kernel signature.  Uses
``get_used_data`` to find genuinely dead names; ``binding_names`` keeps ones
the bindings layer still needs.
"""
from __future__ import annotations

from typing import Set

import dace
from dace.sdfg import nodes as _nodes
from dace.sdfg.sdfg import SDFG


def _collect_live_names(sdfg: SDFG) -> Set[str]:
    """Every array name referenced anywhere on this SDFG.  Two surfaces:
    DaCe's ``get_used_data`` per state (AccessNodes/memlets/read-write sets),
    and an identifier-grep over tasklet code / interstate assignments+conditions
    / frame init-exit code (needed because a length-1 scalar promoted to a free
    symbol is only referenced textually).  The grep is over-broad in the safe
    direction -- may keep a dead array, never over-prunes."""
    import re
    _IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\b")
    from dace.sdfg.utils import get_used_data
    live: Set[str] = set()
    for state in sdfg.all_states():
        live |= get_used_data(state)
    array_names = set(sdfg.arrays.keys())

    def _grep(text: str):
        if text:
            live.update(set(_IDENT_RE.findall(text)) & array_names)

    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, _nodes.Tasklet):
                for cb in (n.code, n.code_global, n.code_init, n.code_exit):
                    _grep(getattr(cb, 'as_string', None) or str(cb or ""))
    for isedge in sdfg.all_interstate_edges():
        # assignments map target -> CodeBlock-or-string RHS; condition is its own CodeBlock.
        for v in isedge.data.assignments.values():
            _grep(str(v) if v is not None else "")
        cond = isedge.data.condition
        _grep(getattr(cond, 'as_string', None) or str(cond or ""))
    # Control-flow regions (LoopRegion/ConditionalBlock) carry their own
    # conditions naming SDFG arrays directly -- interstate edges don't have them.
    from dace.sdfg.state import ConditionalBlock, LoopRegion
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            for attr in ('loop_condition', 'init_statement', 'update_statement'):
                cb = getattr(region, attr, None)
                if cb is not None:
                    _grep(getattr(cb, 'as_string', None) or str(cb))
        elif isinstance(region, ConditionalBlock):
            for cond, _branch in region.branches:
                if cond is not None:
                    _grep(getattr(cond, 'as_string', None) or str(cond))
    for cb in (sdfg.init_code.get('frame'), sdfg.exit_code.get('frame')):
        _grep(getattr(cb, 'as_string', None) or str(cb or ""))
    return live


def _prune_one(sdfg: SDFG, binding_names: Set[str]) -> Set[str]:
    """Drop every non-persistent, non-globally-scoped descriptor with no live
    reference and no bindings-layer reference.  Returns the set of pruned names."""
    live = _collect_live_names(sdfg)
    live |= set(sdfg.symbols.keys())  # symbol-promoted scalars
    live |= {str(s) for s in sdfg.free_symbols}  # free symbol references
    live |= binding_names  # FlattenPlan + arglist keepers
    dropped: Set[str] = set()
    for name in list(sdfg.arrays.keys()):
        if name in live:
            continue
        desc = sdfg.arrays[name]
        # Non-transient descriptors are the caller-binding contract -- dropping
        # one breaks the argument list against any caller.  A module-level
        # scalar's reads can vanish after SCCP+symbol-dce constant-folds its
        # BSS-zero init, but the caller still pre-sets it, so skip non-transients
        # entirely; the live-name walker is transients-only.
        if not desc.transient:
            continue
        # Persistent/Global lifetime descriptors are kept regardless of dataflow
        # visibility -- the runtime allocator manages them; dropping breaks codegen.
        lifetime = getattr(desc, 'lifetime', None)
        if lifetime in (dace.AllocationLifetime.Persistent, dace.AllocationLifetime.Global):
            continue
        del sdfg.arrays[name]
        dropped.add(name)
    return dropped


def prune_unused_arrays(sdfg: SDFG, binding_names: Set[str] = frozenset()) -> Set[str]:
    """Recursively prune dead-descriptor arrays from ``sdfg`` and every
    NestedSDFG body reachable from it (mutated in place).  ``binding_names``
    keeps names the bindings layer needs regardless of dataflow visibility
    (FlattenPlan's ``flat_names``, writeback companions); empty set when no
    binding-emission concern (e.g. direct-call SDFGs)."""
    dropped = _prune_one(sdfg, binding_names)
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, _nodes.NestedSDFG):
                # Nested SDFG descriptors live in their own table; parent's binding-name set doesn't apply.
                dropped |= prune_unused_arrays(n.sdfg)
    return dropped
