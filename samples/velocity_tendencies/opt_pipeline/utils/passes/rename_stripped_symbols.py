"""Collapse every f2dace-family name into a short, filter-safe form.

Two motivations:

1. DaCe's codegen hardcodes a filter that drops any free symbol starting
   with ``__f2dace_SA``, ``__f2dace_SOA``, or ``tmp_struct_symbol`` from
   the generated kernel signature
   (see ``dace/codegen/targets/framecode.py:55-60``, author-marked
   ``TODO: Hack, remove!``). We can't touch that filter, so our symbols
   must not match any of those prefixes at compile time.

2. After ``StructToContainerGroups`` + ``PrepareBaseline``, an SDFG
   accumulates three flavours of f2dace-style scalar names:

     - bare prefix:    ``__f2dace_<KIND>_<field>_d_<dim>_s_<idx>[_<tail>]``
                         where KIND is one of SA / SOA / A / OA / STRUCTARRAY
     - CG-nested:      ``__CG_<c>__m___f2dace_<KIND>_<field>_d_<dim>_s_<idx>``
     - ``tmp_struct_symbol_<N>``

   Readability suffers (especially in kernel signatures) and the per-
   occurrence ``_s_<idx>`` sequence number is noise once the SoA layout
   is baked in.

This pass applies three cheap, order-independent string rewrites that
strip the ``__f2dace_`` token everywhere it appears while keeping the
kind tag (SA/SOA/A/OA/etc.) as the new prefix:

  - ``__f2dace_<KIND>_`` -> ``<KIND>_``     (KIND = any f2dace subword)
  - ``_s_<digits>``       removed
  - ``tmp_struct_symbol_`` -> ``TSS_``

Resulting shapes:

  ``__f2dace_SA_cell_blk_d_1_s_168_edges_p_patch_4``
        -> ``SA_cell_blk_d_1_edges_p_patch_4``
  ``__CG_p_diag__m___f2dace_SA_ddt_vn_apc_pc_d_0_s_300``
        -> ``__CG_p_diag__m_SA_ddt_vn_apc_pc_d_0``
  ``__f2dace_A_z_kin_hor_e_d_0_s_363``
        -> ``A_z_kin_hor_e_d_0``
  ``__f2dace_STRUCTARRAY_foo_d_0_s_5``
        -> ``STRUCTARRAY_foo_d_0``
  ``tmp_struct_symbol_0``
        -> ``TSS_0``

The kind tags come from disjoint f2dace namespaces (SA = struct-array
size, SOA = struct-array offset, A = array size, OA = array offset,
STRUCTARRAY = struct-array itself). Keeping them preserves size-vs-
offset / struct-vs-plain semantics and avoids collisions if the same
``<field>`` name exists in two namespaces.

Must run after every other rewriting pass (so the rewrite applies to
the final surviving set) and before compile. Idempotent: already-
renamed names contain no f2dace tokens and stay unchanged. Raises on
any collision (two distinct old names would rename to the same new
name) so we never silently alias two SDFG symbols onto one C
identifier.
"""

import re
from typing import Dict, Optional

from dace import SDFG, properties
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation as xf


# Match ``__f2dace_<KIND>_`` with KIND being any f2dace subword. The
# capture group is used as the replacement prefix -- so ``__f2dace_FOO_``
# becomes ``FOO_``, irrespective of which flavour f2dace emitted.
_F2DACE_SEG = re.compile(r"__f2dace_([A-Za-z][A-Za-z0-9]*)_")
_S_SEG = re.compile(r"_s_\d+")
_TSS_PREFIX = "tmp_struct_symbol_"


def _rename(name: str) -> Optional[str]:
    """Return the canonicalised form if ``name`` contains any f2dace or
    tmp_struct_symbol token; ``None`` otherwise."""
    new = _F2DACE_SEG.sub(lambda m: f"{m.group(1)}_", name)
    new = _S_SEG.sub("", new)
    if new.startswith(_TSS_PREFIX):
        new = "TSS_" + new[len(_TSS_PREFIX):]
    if new == name:
        return None
    return new


@properties.make_properties
@xf.explicit_cf_compatible
class RenameStrippedSymbols(ppl.Pass):
    """Canonicalise every ``__f2dace_<KIND>_*`` / ``tmp_struct_symbol_*``
    name across the SDFG so the generated kernel signature is short,
    consistent, and clear of DaCe's framecode signature-filter hack."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Dict[str, str]]:
        total: Dict[str, str] = {}
        for nested in list(sdfg.all_sdfgs_recursive()):
            rewrites: Dict[str, str] = {}
            for name in list(nested.symbols) + list(nested.arrays):
                new = _rename(name)
                if new is not None and new != name:
                    rewrites.setdefault(name, new)

            reverse: Dict[str, str] = {}
            for old, new in rewrites.items():
                if new in reverse:
                    raise ValueError(
                        f"RenameStrippedSymbols collision: {old!r} and "
                        f"{reverse[new]!r} both rename to {new!r}"
                    )
                reverse[new] = old

            if not rewrites:
                continue

            nested.replace_dict(rewrites)

            # Belt-and-suspenders: DaCe's ``replace_dict`` goes through
            # ``ast.parse`` / ``ASTFindReplace`` and can miss references
            # in complex interstate-edge expressions. Manually walk
            # interstate edges, conditional / loop expressions.
            from utils.prune_names import replace_on_interstate_edges
            replace_on_interstate_edges(nested, rewrites)

            total.update(rewrites)

        return total or None
