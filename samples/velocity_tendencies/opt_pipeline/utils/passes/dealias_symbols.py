"""Dealias ``tmp_struct_symbol_N`` scalars to their SoA container-group symbols.

``PrepareBaseline`` parses ``sdfg.init_code`` (e.g.

    tmp_struct_symbol_0 = global_data->nproma
    tmp_struct_symbol_2 = global_data->nproma
    tmp_struct_symbol_3 = p_patch->nblks_c
    __f2dace_SOA_neighbor_idx_d_0_s_146_cells_p_patch_2 = p_patch->cells->__f2dace_SOA_neighbor_idx_d_0_s_146
    ...

), registers every LHS as a global ``int32`` symbol, and hands this pass
the raw ``{lhs: rhs}`` map. This pass then:

1. Selects only the aliasing entries -- the ``tmp_struct_symbol_N`` LHS
   whose RHS matches the one-hop ``<container>.<member>`` shape. These
   are "accumulator aliases" that point at an SoA leaf the container-
   group pass has already named (``global_data.nproma`` ->
   ``__CG_global_data_m_nproma``).
2. Promotes each mangled target into ``sdfg.symbols`` (demoting any
   matching scalar data descriptor to a symbol in the process).
3. Rewrites every occurrence of the aliased LHS names (memlets,
   interstate-edge expressions, tasklet code, conditions, shapes) to
   the mangled symbol via ``sdfg.replace_dict`` and drops the LHS from
   ``sdfg.symbols``.

Nested accesses like ``p_patch->cells->__f2dace_SOA_...`` -- and every
``__f2dace_SA_*`` / ``__f2dace_SOA_*`` / ``__f2dace_A_*`` /
``__f2dace_OA_*`` LHS -- are *not* touched; those stay as free kernel
parameters (the caller passes them directly). Later passes
``ResolveExtentOffsets`` (offsets, SOA/OA family) and
``ResolveExtentSizes`` (sizes, SA/A family) fold the subset of those
whose observed lbound / size is a known literal or a well-known scalar
like ``nproma``.

Returns the applied ``{lhs: mangled}`` mapping, or ``None`` when nothing
aliased.
"""

import re
from typing import Dict, List, Optional, Set

import dace
from dace import SDFG, data, properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


# ``PrepareBaseline`` stores init-code RHS values with ``->`` rewritten to
# ``.`` so they're Python-parseable; the one-hop shape we can dealias looks
# like ``global_data.nproma`` -- a single ``.`` between two identifiers.
_SIMPLE_ACCESS_RE = re.compile(r"^([A-Za-z_]\w*)\.([A-Za-z_]\w*)$")


def _mangle(container: str, member: str) -> str:
    # Matches the ``StructToContainerGroups`` output format: double
    # underscore before ``m_``. One-underscore form (``__CG_<c>_m_<m>``)
    # would alias to a *different* symbol, so the caller would have to
    # pass the same value twice.
    return f"__CG_{container}__m_{member}"


@properties.make_properties
@xf.explicit_cf_compatible
class DealiasSymbols(ppl.Pass):
    """Rewrite ``tmp_struct_symbol_N`` aliases to their container-group
    mangled names, given the ``init_map`` captured by ``PrepareBaseline``."""

    CATEGORY: str = "Simplification"

    init_map = properties.DictProperty(
        key_type=str,
        value_type=str,
        default={},
        desc="{lhs: rhs} map from PrepareBaseline -- only one-hop struct "
        "walks are dealiased; the rest are left as free symbols.",
    )

    def __init__(self, init_map: Optional[Dict[str, str]] = None):
        super().__init__()
        self.init_map = dict(init_map or {})

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[str, str]]:
        if not self.init_map:
            return None

        rewrites = _collect_rewrites(self.init_map)
        if not rewrites:
            return None

        _promote_mangled_targets(sdfg, set(rewrites.values()))

        # ``replace_keys=False`` leaves every LHS in ``sdfg.symbols``. If the
        # graph-wide rewrite is complete the LHS becomes an unused symbol
        # (DaCe's ``arglist`` drops unused ones automatically). If *any*
        # reference leaks through -- anything from an obscure memlet subset
        # to a stringified interstate-edge RHS -- the LHS is still a valid
        # declared symbol and shows up in arglist, so codegen never emits
        # an "undeclared" reference.
        sdfg.replace_dict(rewrites, replace_keys=False)

        # Belt-and-suspenders: ``sdfg.replace_dict`` routes through
        # ``ast.parse`` which can miss references in complex nested
        # interstate-edge expressions. Manually walk every interstate
        # edge / conditional / loop expression and do a token-bounded
        # string replacement.
        from utils.prune_names import replace_on_interstate_edges
        replace_on_interstate_edges(sdfg, rewrites)

        return rewrites


def _collect_rewrites(assignments: Dict[str, str]) -> Dict[str, str]:
    """Group ``{lhs: rhs}`` by RHS; for every RHS that matches the one-hop
    ``<container>.<member>`` shape, map each aliased LHS to the mangled name."""
    by_rhs: Dict[str, List[str]] = {}
    for lhs, rhs in assignments.items():
        by_rhs.setdefault(rhs, []).append(lhs)

    rewrites: Dict[str, str] = {}
    for rhs, lhs_list in by_rhs.items():
        m = _SIMPLE_ACCESS_RE.match(rhs)
        if not m:
            continue
        mangled = _mangle(m.group(1), m.group(2))
        for lhs in lhs_list:
            rewrites[lhs] = mangled
    return rewrites


def _promote_mangled_targets(sdfg: SDFG, targets: Set[str]):
    """Ensure each name in ``targets`` exists as a DaCe symbol. Scalar /
    length-1 array data descriptors of the same name are deleted first."""
    for name in targets:
        if name in sdfg.symbols:
            continue
        desc = sdfg.arrays.get(name)
        if desc is None:
            sdfg.add_symbol(name, dace.int32)
            continue
        if isinstance(desc, data.Scalar) or (
            isinstance(desc, data.Array) and all(s == 1 for s in desc.shape)
        ):
            dtype = desc.dtype
            sdfg.remove_data(name, validate=False)
            sdfg.add_symbol(name, dtype)
