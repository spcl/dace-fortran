"""Drop leftover init-edge assignments whose RHS is a C struct access,
and promote every ``__f2dace_*`` identifier to a global ``int32`` symbol.

After ``DealiasSymbols``, ``ResolveExtentOffsets`` and
``ResolveExtentSizes`` have had their way with the init edge, whatever
remains is a struct-pointer walk the pipeline couldn't dealias (nested
access, unhandled shapes, ...). Drop those assignments: the LHS names
stay in ``sdfg.symbols`` and become **external** kernel parameters that
the caller (e.g. ``main_per.cu``) supplies directly -- no generated-code
side needs the struct-pointer expression.

Detection: we look for ``.`` (or the original ``->``) anywhere in the RHS.
An identifier-only RHS stays (e.g. ``nblks_c = p_patch.nblks_c`` would
match but ``nblks_c = 1`` would not; we only drop the first kind).

After the init scaffold is cleaned, the pass walks the whole SDFG and
makes every ``__f2dace_*`` identifier a global ``int32`` symbol:

- A scalar-like descriptor in ``sdfg.arrays`` is dropped and re-added to
  ``sdfg.symbols`` as ``int32``.
- A name referenced in a memlet / tasklet / interstate-edge / loop /
  conditional / array-shape but absent from both ``sdfg.symbols`` and
  ``sdfg.arrays`` is added to ``sdfg.symbols`` as ``int32``.
- Any interstate-edge assignment whose LHS is a ``__f2dace_*`` name is
  dropped after this (the value now comes from the caller).
- Real multi-dim arrays whose name contains ``__f2dace_`` are left alone.
"""

import re
from typing import Iterable, Optional, Set

import dace
from dace import SDFG, data, properties
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


_STRUCT_ACCESS_RE = re.compile(r"[A-Za-z_]\w*\s*(?:\.|->)\s*[A-Za-z_]\w*")
_F2DACE_IDENT_RE = re.compile(r"\b(__f2dace_[A-Za-z0-9_]+)\b")


@properties.make_properties
@xf.explicit_cf_compatible
class ExternalizeStructAccesses(ppl.Pass):
    """Delete init-edge assignments with struct-access RHS so their LHS
    symbols become external kernel parameters, then collapse the
    ``PrepareBaseline`` marker scaffold when it's no longer needed. Also
    promotes every ``__f2dace_*`` identifier to a global ``int32`` symbol
    so the same externalisation applies to names introduced via shape
    expressions / memlet subsets rather than init assignments."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG | ppl.Modifies.States | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _):
        entry = next((s for s in sdfg.states() if s.label == "__prep_init_entry"), None)
        post = next((s for s in sdfg.states() if s.label == "__prep_init_post"), None)

        dropped = 0
        if entry is not None and post is not None:
            for oe in sdfg.out_edges(entry):
                kept = {}
                for k, v in oe.data.assignments.items():
                    if _STRUCT_ACCESS_RE.search(v):
                        dropped += 1
                        continue
                    kept[k] = v
                oe.data.assignments = kept

            _collapse_marker_scaffold(sdfg, entry, post)

        promoted, added, dropped_f2dace = _promote_f2dace_to_int32_symbols(sdfg)

        if not (dropped or promoted or added or dropped_f2dace):
            return None
        return {
            "dropped_struct_assignments": dropped,
            "promoted_from_arrays": sorted(promoted),
            "added_from_references": sorted(added),
            "dropped_f2dace_assignments": dropped_f2dace,
        }


def _collapse_marker_scaffold(sdfg: SDFG, entry, post):
    """Replace the ``entry -> post -> original_start`` chain with a single
    edge carrying whatever assignments survive on ``entry -> post``.

    If no assignments remain, the chain collapses entirely and
    ``original_start`` becomes the SDFG start block.
    """
    entry_out = sdfg.out_edges(entry)
    post_out = sdfg.out_edges(post)
    if len(entry_out) != 1 or len(post_out) != 1:
        return

    assign_edge = entry_out[0]
    passthru = post_out[0]
    original_start = passthru.dst
    remaining = dict(assign_edge.data.assignments or {})

    import dace

    sdfg.remove_edge(assign_edge)
    sdfg.remove_edge(passthru)
    # remove_node handles the inner tasklets + any stragglers.
    sdfg.remove_node(entry)
    sdfg.remove_node(post)

    if remaining:
        init = sdfg.add_state("__baseline_init", is_start_block=True)
        sdfg.add_edge(init, original_start, dace.InterstateEdge(assignments=remaining))
    else:
        sdfg.start_block = sdfg.node_id(original_start)


# ---------------------------------------------------------------------------
# f2dace identifier promotion
# ---------------------------------------------------------------------------


def _is_scalar_like(desc) -> bool:
    return isinstance(desc, data.Scalar) or (
        isinstance(desc, data.Array) and all(s == 1 for s in desc.shape)
    )


def _collect_f2dace_identifiers(sdfg: SDFG) -> Set[str]:
    """Every ``__f2dace_*`` name referenced anywhere in ``sdfg``, plus
    every ``__f2dace_*`` key currently in ``sdfg.arrays`` / ``sdfg.symbols``."""
    found: Set[str] = set()

    def _grab(text: Optional[str]):
        if text:
            found.update(_F2DACE_IDENT_RE.findall(text))

    for name in list(sdfg.arrays.keys()) + list(sdfg.symbols.keys()):
        if "__f2dace_" in name:
            found.update(_F2DACE_IDENT_RE.findall(name))
            if name.startswith("__f2dace_"):
                found.add(name)

    for desc in sdfg.arrays.values():
        for dim in getattr(desc, "shape", ()):
            for s in getattr(dim, "free_symbols", ()):
                sn = str(s)
                if sn.startswith("__f2dace_"):
                    found.add(sn)

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.Tasklet):
                _grab(node.code.as_string or "")
        for edge in state.edges():
            if edge.data is None:
                continue
            _grab(edge.data.data)
            for sub_attr in ("subset", "other_subset"):
                sub = getattr(edge.data, sub_attr, None)
                if sub is None:
                    continue
                for s in getattr(sub, "free_symbols", ()):
                    sn = str(s)
                    if sn.startswith("__f2dace_"):
                        found.add(sn)

    for cfg in sdfg.all_control_flow_regions():
        for iedge in cfg.edges():
            for k, v in (iedge.data.assignments or {}).items():
                _grab(k)
                _grab(v)
            if iedge.data.condition is not None:
                _grab(iedge.data.condition.as_string)

    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ConditionalBlock):
            for cond, _region in block.branches:
                if cond is not None:
                    _grab(cond.as_string)
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            for code in (region.loop_condition, region.init_statement, region.update_statement):
                if code is not None:
                    _grab(code.as_string)

    return found


def _register_as_int32_symbols(sdfg: SDFG, names: Iterable[str]):
    """Ensure every name in ``names`` is a ``sdfg.symbols`` entry of type
    ``int32``. Returns ``(promoted_from_arrays, added_from_references)``."""
    promoted: Set[str] = set()
    added: Set[str] = set()
    for name in names:
        if name in sdfg.symbols:
            continue
        if name in sdfg.arrays:
            desc = sdfg.arrays[name]
            if not _is_scalar_like(desc):
                continue
            sdfg.remove_data(name, validate=False)
            sdfg.add_symbol(name, dace.int32)
            promoted.add(name)
        else:
            sdfg.add_symbol(name, dace.int32)
            added.add(name)
    return promoted, added


def _drop_f2dace_assignments(sdfg: SDFG) -> int:
    """Remove any interstate-edge assignment whose LHS is an ``__f2dace_*``
    identifier. Returns the number of dropped assignments."""
    dropped = 0
    for cfg in sdfg.all_control_flow_regions():
        for edge in cfg.edges():
            assigns = edge.data.assignments or {}
            to_drop = [k for k in assigns if _F2DACE_IDENT_RE.fullmatch(k) or "__f2dace_" in k]
            if not to_drop:
                continue
            edge.data.assignments = {k: v for k, v in assigns.items() if k not in to_drop}
            dropped += len(to_drop)
    return dropped


def _promote_f2dace_to_int32_symbols(sdfg: SDFG):
    """Run the promotion across every SDFG. Returns
    ``(promoted_from_arrays, added_from_references, dropped_assignments)``
    totals."""
    promoted_total: Set[str] = set()
    added_total: Set[str] = set()
    dropped_total = 0
    for nested in list(sdfg.all_sdfgs_recursive()):
        names = _collect_f2dace_identifiers(nested)
        if names:
            promoted, added = _register_as_int32_symbols(nested, names)
            promoted_total.update(promoted)
            added_total.update(added)
        dropped_total += _drop_f2dace_assignments(nested)
    return promoted_total, added_total, dropped_total
