"""Two f2dace-symbol cleanup passes that run back-to-back.

``StripSSuffix``
    Drops every ``_s_<idx>`` sequence-number segment from f2dace-prefixed
    names so defs and uses (which can pick up different ``_s_<idx>`` values
    after specialisation + simplify) converge on one canonical name.
    Handles both top-level f2dace names (``__f2dace_SOA_foo_d_0_s_42``)
    and f2dace chunks nested inside ``StructToContainerGroups`` mangles
    (``__CG_p_int_m___f2dace_SOA_cells_aw_verts_d_0_s_31``).

``RegisterReferencedF2daceSymbols``
    Scans every memlet subset / tasklet code / interstate edge / branch
    condition / loop expression / array shape for ``__f2dace_*``
    identifiers, and re-registers any that are referenced but absent from
    ``sdfg.symbols`` / ``sdfg.arrays``. Those names become free symbols
    and enter ``arglist()`` (and therefore the generated kernel
    signature). Fixes the "symbol referenced but not declared" C++
    compile errors that occur when an earlier pass prunes an ``SA`` /
    ``SOA`` / ``OA`` / ``A`` symbol from the symbol table while a
    subscript expression in the body still names it.

Both passes can be invoked anywhere in the pipeline; they run recursively
across nested SDFGs.
"""

import re
from typing import Dict, Iterable, List, Optional, Set

import dace
from dace import SDFG, properties
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


# Matches ``__f2dace_<KIND>_<field>_d_<dim>_s_<idx>`` anywhere in the
# identifier (not anchored to the start, since ``StructToContainerGroups``
# emits mangled forms like ``__CG_p_int_m___f2dace_SOA_..._d_0_s_31``
# where the f2dace chunk is nested inside a container-group prefix).
# Captures everything up to (but not including) ``_s_<idx>`` so ``re.sub``
# drops the ``_s_<idx>`` segment while preserving the rest of the name.
_STRIP_RE = re.compile(r"(__f2dace_[A-Za-z]+_.+?_d_\d+)_s_\d+")


def _strip(name: str) -> str:
    """Drop every ``_s_<digits>`` segment from an f2dace name (including
    f2dace chunks embedded in the middle of a mangled identifier)."""
    return _STRIP_RE.sub(r"\1", name)


@properties.make_properties
@xf.explicit_cf_compatible
class StripSSuffix(ppl.Pass):
    """Remove the ``_s_<int>`` sequence-number segment from every f2dace name."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _):
        total: Dict[str, str] = {}
        for nested in list(sdfg.all_sdfgs_recursive()):
            rewrites = _collect_rewrites(nested)
            if not rewrites:
                continue
            _apply_rewrites(nested, rewrites)
            total.update(rewrites)
        return total or None


def _collect_rewrites(sdfg: SDFG) -> Dict[str, str]:
    """Build ``{old: stripped}`` for every array / symbol name whose
    stripped form differs from its current form."""
    rewrites: Dict[str, str] = {}
    for name in list(sdfg.arrays.keys()) + list(sdfg.symbols.keys()):
        stripped = _strip(name)
        if stripped != name:
            rewrites.setdefault(name, stripped)
    return rewrites


def _apply_rewrites(sdfg: SDFG, rewrites: Dict[str, str]):
    """Rename every key in ``rewrites`` across ``sdfg.arrays``,
    ``sdfg.symbols``, and every graph reference. When two old names
    collide on the same stripped name, the first one iterated wins and
    subsequent descriptors / stypes are discarded."""
    # Snapshot the descriptors / stypes we need to reinstall under
    # stripped names so we can delete the old keys without losing data.
    array_descs = {
        old: sdfg.arrays[old] for old in rewrites if old in sdfg.arrays
    }
    symbol_stypes = {
        old: sdfg.symbols[old] for old in rewrites if old in sdfg.symbols
    }

    for old in array_descs:
        sdfg.remove_data(old, validate=False)
    for old in symbol_stypes:
        sdfg.remove_symbol(old)

    # First old per stripped name wins.
    winners: Dict[str, str] = {}
    for old, new in rewrites.items():
        winners.setdefault(new, old)

    for new, old in winners.items():
        if old in array_descs and new not in sdfg.arrays:
            sdfg.add_datadesc(new, array_descs[old])
        if old in symbol_stypes and new not in sdfg.symbols:
            sdfg.add_symbol(new, symbol_stypes[old])

    # ``replace_keys=False`` leaves arrays/symbols dicts alone (we've
    # already reinstalled them under the canonical names); the call still
    # rewrites every graph-side reference (memlets, access-nodes,
    # interstate-edge assignments/conditions, conditional-block
    # conditions, loop-region expressions, tasklet code).
    sdfg.replace_dict(rewrites, replace_keys=False)

    # Post-check: every renamed old name must be completely gone from the
    # SDFG. ``replace_dict`` can silently miss occurrences (e.g. inside a
    # nested structure it doesn't walk, or a property it doesn't touch),
    # and that's the exact failure mode that produces "symbol referenced
    # but not declared" compile errors downstream. Fail loudly here so the
    # offending site is obvious.
    for old in rewrites:
        _assert_name_gone(sdfg, old)


def _token_in(name: str, text: Optional[str]) -> bool:
    return bool(text) and re.search(r"\b" + re.escape(name) + r"\b", text) is not None


def _assert_name_gone(sdfg: SDFG, name: str):
    """Fail loudly if ``name`` still appears anywhere in ``sdfg``."""
    if name in sdfg.arrays:
        raise AssertionError(f"{name!r} still in sdfg.arrays after StripSSuffix")
    if name in sdfg.symbols:
        raise AssertionError(f"{name!r} still in sdfg.symbols after StripSSuffix")
    if name in getattr(sdfg, "constants_prop", {}):
        raise AssertionError(f"{name!r} still in sdfg.constants_prop after StripSSuffix")

    # Array shapes / strides that still reference the old name.
    for aname, desc in sdfg.arrays.items():
        for dim in getattr(desc, "shape", ()):
            if name in {str(s) for s in getattr(dim, "free_symbols", ())}:
                raise AssertionError(
                    f"{name!r} still appears in shape of array {aname!r} = {desc.shape}"
                )

    # State-local references: access nodes, memlets, tasklet code.
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == name:
                raise AssertionError(
                    f"{name!r} still present as AccessNode in state {state.label!r}"
                )
            if isinstance(node, nodes.Tasklet):
                if _token_in(name, node.code.as_string or ""):
                    raise AssertionError(
                        f"{name!r} still present in tasklet code at {state.label}/{node.label}"
                    )
        for edge in state.edges():
            if edge.data is None:
                continue
            if edge.data.data == name:
                raise AssertionError(
                    f"{name!r} still present in memlet.data in state {state.label!r}"
                )
            for sub_attr in ("subset", "other_subset"):
                sub = getattr(edge.data, sub_attr, None)
                if sub is not None and name in {str(s) for s in getattr(sub, "free_symbols", ())}:
                    raise AssertionError(
                        f"{name!r} still present in memlet.{sub_attr} in state {state.label!r}"
                    )

    # Interstate edges (assignments + conditions).
    for cfg in sdfg.all_control_flow_regions():
        for iedge in cfg.edges():
            for k, v in (iedge.data.assignments or {}).items():
                if _token_in(name, k) or _token_in(name, v):
                    raise AssertionError(
                        f"{name!r} still present in interstate-edge assignment {k!r}={v!r}"
                    )
            cond = iedge.data.condition.as_string if iedge.data.condition else None
            if _token_in(name, cond):
                raise AssertionError(
                    f"{name!r} still present in interstate-edge condition {cond!r}"
                )

    # Conditional-block branch conditions.
    for block in sdfg.all_control_flow_blocks():
        if isinstance(block, ConditionalBlock):
            for cond, _region in block.branches:
                if cond is not None and _token_in(name, cond.as_string):
                    raise AssertionError(
                        f"{name!r} still present in ConditionalBlock {block.label!r} cond {cond.as_string!r}"
                    )

    # Loop-region init / condition / update.
    for region in sdfg.all_control_flow_regions():
        if isinstance(region, LoopRegion):
            for code in (region.loop_condition, region.init_statement, region.update_statement):
                if code is not None and _token_in(name, code.as_string):
                    raise AssertionError(
                        f"{name!r} still present in LoopRegion {region.label!r} expression {code.as_string!r}"
                    )


# ---------------------------------------------------------------------------
# PromoteF2daceSymbolsToGlobals
# ---------------------------------------------------------------------------

# Matches any f2dace-family identifier referenced in a code/expr string.
_F2DACE_IDENT_RE = re.compile(r"\b(__f2dace_[A-Za-z0-9_]+)\b")


@properties.make_properties
@xf.explicit_cf_compatible
class PromoteF2daceSymbolsToGlobals(ppl.Pass):
    """Make every f2dace ``SA`` / ``SOA`` / ``OA`` / ``A`` scalar a global
    ``int32`` symbol so it ends up in the kernel signature.

    Three steps per SDFG, recursively:

    1. Collect every ``__f2dace_*`` identifier referenced anywhere in the
       graph (memlet subsets, memlet.data, tasklet code, interstate-edge
       assignments/conditions, ``ConditionalBlock`` / ``LoopRegion``
       expressions, array shape ``free_symbols``) and every ``__f2dace_*``
       key in ``sdfg.arrays`` / ``sdfg.symbols``.
    2. For each such name: if it's a scalar-like descriptor in
       ``sdfg.arrays`` (``Scalar`` or length-1 ``Array``), delete the
       descriptor and add it to ``sdfg.symbols`` as ``int32``. If it
       isn't in ``sdfg.symbols`` / ``sdfg.arrays`` at all, just add it as
       ``int32``. Real multi-dim arrays are left untouched.
    3. Drop any interstate-edge assignment whose LHS is a promoted
       ``__f2dace_*`` name -- those values now come from the caller.
       Cleans both the ``__prep_init_entry`` edge (if present) and every
       other interstate edge, in case prior passes installed assignments
       outside the init scaffold.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _):
        promoted_total: Set[str] = set()
        added_total: Set[str] = set()
        dropped_assigns = 0
        for nested in list(sdfg.all_sdfgs_recursive()):
            names = _collect_f2dace_identifiers(nested)
            if not names:
                continue
            promoted, added = _register_as_int32_symbols(nested, names)
            dropped = _drop_f2dace_assignments(nested, names)
            promoted_total.update(promoted)
            added_total.update(added)
            dropped_assigns += dropped
        if not (promoted_total or added_total or dropped_assigns):
            return None
        return {
            "promoted_from_arrays": sorted(promoted_total),
            "added_from_references": sorted(added_total),
            "dropped_init_assignments": dropped_assigns,
        }


def _collect_f2dace_identifiers(sdfg: SDFG) -> Set[str]:
    """Every ``__f2dace_*`` name referenced anywhere in ``sdfg``, plus
    every ``__f2dace_*`` key currently in ``sdfg.arrays`` / ``sdfg.symbols``."""
    found: Set[str] = set()

    def _grab(text: Optional[str]):
        if text:
            found.update(_F2DACE_IDENT_RE.findall(text))

    # Array / symbol dict keys.
    for name in list(sdfg.arrays.keys()) + list(sdfg.symbols.keys()):
        if name.startswith("__f2dace_") or "__f2dace_" in name:
            # Only register the actual f2dace chunks; the surrounding mangle
            # (__CG_x_m___f2dace_...) is also a legal identifier so grab
            # both via the regex.
            found.update(_F2DACE_IDENT_RE.findall(name))
            if name.startswith("__f2dace_"):
                found.add(name)

    # Array shapes / strides.
    for desc in sdfg.arrays.values():
        for dim in getattr(desc, "shape", ()):
            for s in getattr(dim, "free_symbols", ()):
                sn = str(s)
                if sn.startswith("__f2dace_"):
                    found.add(sn)

    # State-local references.
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

    # Interstate edges.
    for cfg in sdfg.all_control_flow_regions():
        for iedge in cfg.edges():
            for k, v in (iedge.data.assignments or {}).items():
                _grab(k)
                _grab(v)
            if iedge.data.condition is not None:
                _grab(iedge.data.condition.as_string)

    # Conditional-block and loop-region expressions.
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


def _is_scalar_like(desc) -> bool:
    from dace import data as _data
    return isinstance(desc, _data.Scalar) or (
        isinstance(desc, _data.Array) and all(s == 1 for s in desc.shape)
    )


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
                # Real multi-dim array -- leave it.
                continue
            sdfg.remove_data(name, validate=False)
            sdfg.add_symbol(name, dace.int32)
            promoted.add(name)
        else:
            sdfg.add_symbol(name, dace.int32)
            added.add(name)
    return promoted, added


def _drop_f2dace_assignments(sdfg: SDFG, names: Iterable[str]) -> int:
    """Remove any interstate-edge assignment whose LHS is in ``names``.
    Returns the number of dropped assignments."""
    targets = {n for n in names if n.startswith("__f2dace_") or "__f2dace_" in n}
    dropped = 0
    for cfg in sdfg.all_control_flow_regions():
        for edge in cfg.edges():
            assigns = edge.data.assignments or {}
            to_drop = [k for k in assigns if k in targets]
            if not to_drop:
                continue
            new = {k: v for k, v in assigns.items() if k not in to_drop}
            edge.data.assignments = new
            dropped += len(to_drop)
    return dropped
