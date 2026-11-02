# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Allocate transients whose size is fixed for the whole run once, in ``__dace_init``."""

from collections.abc import Collection
from typing import Any

from dace import SDFG, dtypes, symbolic
from dace import data as dt
from dace.sdfg.state import LoopRegion
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl

KEEP_LIFETIMES = (
    dtypes.AllocationLifetime.Persistent,
    dtypes.AllocationLifetime.Global,
    dtypes.AllocationLifetime.External,
)


def run_constant_symbols(sdfg: SDFG, fixed_symbols: Collection[str] = ()) -> set[str]:
    """Constants and ``fixed_symbols`` that nothing in the tree reassigns.

    A free symbol is a per-call argument, but ``__dace_init`` runs once per compiled object: only the caller can
    promise it never changes.
    """
    reassigned: set[str] = set()
    for nested in sdfg.all_sdfgs_recursive():
        for edge in nested.all_interstate_edges():
            reassigned.update(edge.data.assignments)
        for cfg in nested.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion) and cfg.loop_variable:
                reassigned.add(cfg.loop_variable)
    return (set(fixed_symbols) | sdfg.constants_prop.keys()) - reassigned


def root_names(nsdfg: SDFG, names: set[str]) -> set[str] | None:
    """``names`` rewritten through every ``symbol_mapping`` out to the root, or None if one is unbound."""
    while nsdfg.parent_nsdfg_node is not None:
        mapping = nsdfg.parent_nsdfg_node.symbol_mapping
        outer: set[str] = set()
        for name in names - nsdfg.constants_prop.keys():
            if name not in mapping:
                return None
            outer.update(str(s) for s in symbolic.pystr_to_symbolic(str(mapping[name])).free_symbols)
        names, nsdfg = outer, nsdfg.parent_sdfg
    return names


class MakeTransientsPersistent(ppl.Pass):
    """Make every transient array allocated outside all scopes, whose size reads only run-constant symbols,
    Persistent.

    :param fixed_symbols: free symbols the caller guarantees for every call of the compiled SDFG (ICON's ``nproma``).
    """

    def __init__(self, fixed_symbols: Collection[str] = ()):
        super().__init__()
        self.fixed_symbols = fixed_symbols

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[int, set[str]] | None:
        """:returns: cfg_id -> promoted names, or None if nothing was promoted."""
        allowed = run_constant_symbols(sdfg, self.fixed_symbols)
        promoted: dict[int, set[str]] = {}
        for nsdfg in sdfg.all_sdfgs_recursive():
            ok: set[str] = set()
            refused: set[str] = set()
            for state in nsdfg.states():
                for node in state.data_nodes():
                    desc = node.desc(nsdfg)
                    if node.data in refused or (
                        node.root_data != node.data
                        and nsdfg.arrays[node.root_data].lifetime != dtypes.AllocationLifetime.Persistent
                    ):
                        continue
                    size = {str(s) for s in symbolic.pystr_to_symbolic(desc.total_size).free_symbols}
                    at_root = root_names(nsdfg, size)
                    if (
                        node.data in nsdfg.constants_prop
                        or not desc.transient
                        or type(desc) is not dt.Array
                        or desc.storage == dtypes.StorageType.Register
                        or desc.lifetime in KEEP_LIFETIMES
                        or at_root is None
                        or at_root - allowed
                        or xfh.get_parent_map(state, node) is not None
                    ):
                        refused.add(node.data)
                    else:
                        ok.add(node.data)
            for name in ok - refused:
                nsdfg.arrays[name].lifetime = dtypes.AllocationLifetime.Persistent
            if ok - refused:
                promoted[nsdfg.cfg_id] = ok - refused
        return promoted or None
