"""Post-``StructToContainerGroups`` cleanup as a proper DaCe ``Pass``.

The f2dace-generated SDFG, after AoS->SoA flattening, carries three things
that need fixing up before the baseline is usable on mainline DaCe:

- ``sdfg.global_code['frame']`` contains ``int <name>;`` declarations for
  scalars the generated code reads as globals. Each becomes an SDFG
  symbol (dtype derived from the C declaration, ``int32`` by default).
- ``sdfg.init_code['frame']`` contains ``<name> = <C expression>;``
  assignments wiring the symbols to struct-pointer walks. Every LHS is
  registered as a global ``int32`` symbol; the RHS struct walks are *not*
  installed on an interstate edge. The raw ``{lhs: rhs}`` map is
  returned instead so ``DealiasSymbols`` can rename ``tmp_struct_symbol_N``
  to the corresponding container-group mangled name, and the remaining
  ``__f2dace_*`` LHS stay as free kernel parameters (their values are
  supplied by the caller).
- Flatten tasklets survive from ``StructToContainerGroups``: they read an
  AoS struct and write the SoA leaf arrays. After we drop the AoS side the
  flatten tasklet is redundant -- its SoA outputs become source AccessNodes
  (no in-edge) and their data descriptors flip to non-transient so they
  behave as kernel inputs.

``Structure`` descriptors are intentionally *not* removed here; they get
dropped once nothing references them (see ``_drop_unused_structure_descriptors``).
"""

import re
from typing import Dict, Optional

import dace
from dace import SDFG, SDFGState, dtypes, nodes, properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


_DECL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ \t*]*?)\s+([A-Za-z_]\w*)\s*;\s*$")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*;\s*$")


_C_TO_DACE: Dict[str, dtypes.typeclass] = {
    "int": dace.int32,
    "int32_t": dace.int32,
    "long": dace.int64,
    "long long": dace.int64,
    "int64_t": dace.int64,
    "short": dace.int16,
    "unsigned": dace.uint32,
    "unsigned int": dace.uint32,
    "size_t": dace.uint64,
    "float": dace.float32,
    "double": dace.float64,
    "bool": dace.bool_,
}


def parse_global_code(code: str) -> Dict[str, str]:
    """Return ``{name: ctype}`` for each ``<ctype> <name>;`` line in ``code``."""
    result: Dict[str, str] = {}
    for line in code.splitlines():
        m = _DECL_RE.match(line)
        if m:
            result[m.group(2)] = m.group(1).strip()
    return result


def parse_init_code(code: str) -> Dict[str, str]:
    """Return ``{lhs: rhs}`` for each ``<lhs> = <rhs>;`` line in ``code``.

    C pointer-struct access ``->`` is rewritten to attribute access ``.``
    so the resulting strings are Python-parseable (DaCe's ``replace_dict``
    walks every interstate-edge value through ``ast.parse``). Downstream
    codegen treats ``.`` on a struct-pointer descriptor as ``->``.
    """
    result: Dict[str, str] = {}
    for line in code.splitlines():
        m = _ASSIGN_RE.match(line)
        if m:
            result[m.group(1)] = m.group(2).strip().replace("->", ".")
    return result


def _c_to_dace_dtype(ctype: str) -> dtypes.typeclass:
    key = re.sub(r"\s+", " ", ctype.strip())
    key = re.sub(r"^const\s+", "", key).rstrip("*").strip()
    return _C_TO_DACE.get(key, dace.int32)


@properties.make_properties
@xf.explicit_cf_compatible
class PrepareBaseline(ppl.Pass):
    """Bake f2dace global/init code and flatten conversions into SDFG state.

    Apply after ``StructToContainerGroups`` and before any main-DaCe pass
    that expects standard SDFG form. Returns the raw init-code
    ``{lhs: rhs}`` mapping (or ``None`` when nothing changed) for use by a
    later pointer-dealiasing pass.
    """

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return (
            ppl.Modifies.CFG
            | ppl.Modifies.States
            | ppl.Modifies.Symbols
            | ppl.Modifies.Descriptors
            | ppl.Modifies.Nodes
            | ppl.Modifies.Memlets
        )

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[str, str]]:
        globals_decl: Dict[str, str] = {}
        init_map: Dict[str, str] = {}
        for cb in sdfg.global_code.values():
            globals_decl.update(parse_global_code(cb.as_string))
        for cb in sdfg.init_code.values():
            init_map.update(parse_init_code(cb.as_string))

        changed = False

        # Register every ``global_code`` declaration as a symbol (uses the
        # declared C type when resolvable, int32 otherwise).
        for name, ctype in globals_decl.items():
            if name in sdfg.symbols or name in sdfg.arrays:
                continue
            sdfg.add_symbol(name, _c_to_dace_dtype(ctype))
            changed = True

        # Every ``init_code`` LHS is also a scalar kernel parameter: ensure
        # it has a concrete symbol declaration. We do NOT install the
        # struct-walk RHS as an interstate-edge assignment -- the value
        # comes from the caller directly. ``DealiasSymbols`` downstream
        # consumes ``init_map`` (returned from here) to decide which of
        # these symbols should be renamed to their container-group mangled
        # RHS equivalent.
        for lhs in init_map:
            if lhs in sdfg.symbols or lhs in sdfg.arrays:
                continue
            ctype = globals_decl.get(lhs)
            stype = _c_to_dace_dtype(ctype) if ctype else dace.int32
            sdfg.add_symbol(lhs, stype)
            changed = True

        empty = properties.CodeBlock("")
        for loc in list(sdfg.global_code.keys()):
            if sdfg.global_code[loc].as_string:
                sdfg.global_code[loc] = empty
                changed = True
        for loc in list(sdfg.init_code.keys()):
            if sdfg.init_code[loc].as_string:
                sdfg.init_code[loc] = empty
                changed = True

        if _lift_flatten_outputs(sdfg):
            changed = True

        return init_map if changed else None


def _lift_flatten_outputs(sdfg: SDFG) -> bool:
    """Remove every ``Flattener`` / ``Deflattener`` node and promote the
    SoA leaf AccessNodes they bridge to non-transient.

    ``StructToContainerGroups`` inserts ``Flattener`` nodes (class from
    ``dace.transformation.passes.struct_to_container_group``, not plain
    tasklets) that read an AoS struct and write the SoA leaf arrays, and
    ``Deflattener`` nodes that do the reverse at the output boundary.
    After AoS->SoA is baked in, both are no-op translators.

    For flatten: the SoA arrays on the *output* side become kernel
    **inputs** -- flip them to non-transient and leave their downstream
    out-edges in place (they become source nodes, in-degree 0).
    For deflatten: the SoA arrays on the *input* side become kernel
    **outputs** -- flip to non-transient, leave in-edges in place (they
    become sink nodes, out-degree 0).

    Struct-side AccessNodes that fed flatten (or were fed by deflatten)
    lose their only edge and are removed as orphans. The ``Structure``
    descriptors themselves are dropped from ``sdfg.arrays`` once nothing
    references them any more.
    """
    changed = False
    for node, parent in list(sdfg.all_nodes_recursive()):
        if not isinstance(parent, SDFGState):
            continue
        cls_name = type(node).__name__
        if cls_name not in ("Flattener", "Deflattener"):
            continue
        is_flatten = cls_name == "Flattener"

        soa_side = (
            list(parent.out_edges(node)) if is_flatten else list(parent.in_edges(node))
        )
        struct_side = (
            list(parent.in_edges(node)) if is_flatten else list(parent.out_edges(node))
        )

        for e in soa_side:
            ac = e.dst if is_flatten else e.src
            if isinstance(ac, nodes.AccessNode):
                desc = sdfg.arrays.get(ac.data)
                if desc is not None and getattr(desc, "transient", False):
                    desc.transient = False

        for e in struct_side + soa_side:
            parent.remove_edge(e)
        parent.remove_node(node)

        for e in struct_side:
            ac = e.src if is_flatten else e.dst
            if isinstance(ac, nodes.AccessNode) and parent.degree(ac) == 0:
                parent.remove_node(ac)

        # Drop SoA-side AccessNodes that have no other consumer/producer
        # -- removing flatten/deflatten orphaned them; they'd fail the
        # ``Isolated node`` validator if left in place.
        for e in soa_side:
            ac = e.dst if is_flatten else e.src
            if isinstance(ac, nodes.AccessNode) and parent.degree(ac) == 0:
                parent.remove_node(ac)

        changed = True

    _drop_unused_structure_descriptors(sdfg)
    return changed


def _drop_unused_structure_descriptors(sdfg: SDFG):
    """Delete ``Structure`` descriptors from ``sdfg.arrays`` that have no
    remaining ``AccessNode`` referencing them."""
    from dace import data as _data  # local import to avoid top-level cycle

    live = set()
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.AccessNode):
            live.add(node.data)

    for name, desc in list(sdfg.arrays.items()):
        if isinstance(desc, _data.Structure) and name not in live:
            try:
                sdfg.remove_data(name, validate=False)
            except Exception:
                del sdfg.arrays[name]
