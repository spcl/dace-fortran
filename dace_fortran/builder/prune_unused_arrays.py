"""Prune unused-but-still-registered array descriptors from a built SDFG.

The bridge's struct-flatten + marshal-expansion pipeline mints SoA
companion descriptors for every member of every derived-type dummy
the bridge can see -- including members whose only live use was
inside an external callee whose marshal expansion silently refused
(box / pointer / allocatable / dynamic-shape members; see Phase
2.3.E v2 boundary).  When the refused callee then registers as
``keep_external`` with no expansion possible, the SoA companions
remain in ``sdfg.arrays`` with zero references from any AccessNode,
memlet, tasklet, or NestedSDFG mapping -- pure dead descriptors
that nevertheless show up in :func:`dace.SDFG.arglist`,
:attr:`dace.SDFG.free_symbols`, and the binding emission's
per-argument loop, bloating the kernel signature with parameters
the kernel never reads or writes.

The pass leans on DaCe's canonical ``get_used_data`` utility
(``dace.sdfg.utils.get_used_data`` -> ``state.read_and_write_sets``),
which covers AccessNodes, memlets, interstate-edge assignments /
conditions, and nested-SDFG closures -- the surface area that
``sdfg.arglist`` consumes.  Anything that survives the union of
used data on every reachable state is genuinely dead at the data
layer and safe to drop.

Bindings emission is not at risk: it walks the live
``sdfg.arglist`` (descriptors that survive are exactly the
arguments the kernel needs), and the bridge's binding wrappers
``c_loc`` / ``c_f_pointer`` only the per-member companions the
:class:`FlattenPlan` recipes name AS surviving flat arrays.  When
an external call DOES need a member array, the memlet edges from
the ExternalCall node keep the descriptor live and the prune skips
it.
"""
from __future__ import annotations

from typing import Set

import dace
from dace.sdfg import nodes as _nodes
from dace.sdfg.sdfg import SDFG


def _collect_live_names(sdfg: SDFG) -> Set[str]:
    """Every array name referenced anywhere on this SDFG.

    Walks two surfaces:

    1. DaCe's :func:`dace.sdfg.utils.get_used_data` per state --
       covers AccessNodes + memlets + the read / write sets the
       canonical "what data does this SDFG actually use" view
       defines.
    2. Identifier-grep over every tasklet code-block, interstate
       edge assignment / condition, and SDFG frame init / exit code
       block.  A length-1 scalar arg the bridge promotes to a free
       symbol still lives in ``sdfg.arrays`` (it remains a kernel
       parameter) but its only references at the *use* layer are
       textual: a tasklet that names it directly, or an
       interstate-edge condition.  An identifier match against
       ``sdfg.arrays.keys()`` is over-broad in the safe direction
       (matches a same-named local in the code text -> we keep an
       array that's actually dead; cheap cost), and it never
       over-prunes (the failure mode that would surface as
       ``unresolved free symbol``).
    """
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
        # Each assignment maps target -> CodeBlock-or-string RHS;
        # the branch ``condition`` is its own CodeBlock.
        for v in isedge.data.assignments.values():
            _grep(str(v) if v is not None else "")
        cond = isedge.data.condition
        _grep(getattr(cond, 'as_string', None) or str(cond or ""))
    # Control-flow regions carry their own conditions (LoopRegion's
    # condition_expr / init_statement / update_statement, ConditionalBlock's
    # per-branch condition, ...) that name SDFG arrays directly --
    # interstate edges don't carry them.  Walk every reachable region.
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
    """Drop every non-persistent, non-globally-scoped descriptor
    from ``sdfg.arrays`` that has no live reference in the given
    SDFG and is not referenced by the bindings layer.  Returns the
    set of pruned names."""
    live = _collect_live_names(sdfg)
    live |= set(sdfg.symbols.keys())  # symbol-promoted scalars
    live |= {str(s) for s in sdfg.free_symbols}  # free symbol references
    live |= binding_names  # FlattenPlan + arglist keepers
    dropped: Set[str] = set()
    for name in list(sdfg.arrays.keys()):
        if name in live:
            continue
        desc = sdfg.arrays[name]
        # Non-transient descriptors are the SDFG's caller-binding
        # contract -- dropping one breaks the kernel's argument list
        # against any caller (Python kwargs, bindings emitter, frozen
        # signature).  The bridge surfaces module-level scalar inputs
        # / outputs as non-transient ``(1,)``-Array entries; their
        # reads can vanish after SCCP + symbol-dce constant-fold the
        # BSS-zero init across every load, but the caller still
        # pre-sets them and the SDFG must accept that value.  Skip
        # the prune for any non-transient entry -- the live-name
        # walker is for transients only.
        if not desc.transient:
            continue
        # Persistent-lifetime / global descriptors are kept regardless
        # of dataflow visibility -- the runtime allocator manages them
        # and dropping the descriptor would break codegen.
        lifetime = getattr(desc, 'lifetime', None)
        if lifetime in (dace.AllocationLifetime.Persistent, dace.AllocationLifetime.Global):
            continue
        del sdfg.arrays[name]
        dropped.add(name)
    return dropped


def prune_unused_arrays(sdfg: SDFG, binding_names: Set[str] = frozenset()) -> Set[str]:
    """Recursively prune dead-descriptor arrays from ``sdfg`` and
    every NestedSDFG body reachable from it.

    :param sdfg: SDFG to prune (mutated in place).
    :param binding_names: extra names the bindings layer needs to
        keep visible regardless of the dataflow walker's verdict --
        the :class:`FlattenPlan` recipes' ``flat_names``, the
        wrapper's writeback companions, and any other names the
        emitter spells in its generated Fortran wrapper.  Pass an
        empty set when the prune runs in a context with no
        binding-emission concern (e.g. direct-call SDFGs).
    :returns: Set of dropped names.
    """
    dropped = _prune_one(sdfg, binding_names)
    for state in sdfg.all_states():
        for n in state.nodes():
            if isinstance(n, _nodes.NestedSDFG):
                # Nested SDFG descriptors live in their own table;
                # the parent's binding-name set does not apply.
                dropped |= prune_unused_arrays(n.sdfg)
    return dropped
