"""Layout permutation driver for stage6.

Wraps :class:`dace.transformation.layout.PermuteDimensions` with a
``PermuteConfig`` dataclass so callers can serialize / iterate over a list
of named permutation choices instead of constructing the pass directly.

Map-dim shuffling (``MapDimShuffle``) is applied opportunistically when
available; pre-yakup/dev DaCe forks may lack it, in which case the wrapper
falls back to the bare permute.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import dace


def _inverse(p: List[int]) -> List[int]:
    out = [0] * len(p)
    for i, v in enumerate(p):
        out[v] = i
    return out


@dataclass
class PermuteConfig:
    name: str
    permute_map: Dict[str, List[int]]
    inverse_permute_map: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.inverse_permute_map:
            self.inverse_permute_map = {k: _inverse(v) for k, v in self.permute_map.items()}


def permute_layout(sdfg: dace.SDFG, config: PermuteConfig, shuffle_map: bool = True) -> int:
    """Apply ``config`` to ``sdfg`` in place. Returns the number of permuted arrays."""
    if not config.permute_map:
        return 0
    # Deferred import: the listing tools (``list_layout_configs.py``)
    # only need ``PermuteConfig`` and shouldn't fail to load when DaCe
    # is on a branch that does not yet ship ``permute_dimensions``.
    from dace.transformation.layout.permute_dimensions import PermuteDimensions
    PermuteDimensions(
        permute_map=config.permute_map,
        add_permute_maps=True,
        column_major=False,
    ).apply_pass(sdfg=sdfg, pipeline_results={})

    if shuffle_map:
        try:
            from dace.transformation.dataflow import MapDimShuffle
        except ImportError:
            MapDimShuffle = None
        if MapDimShuffle is not None:
            for state in sdfg.all_states():
                for node in list(state.nodes()):
                    if not isinstance(node, dace.nodes.MapEntry):
                        continue
                    if len(node.map.params) < 2:
                        continue
                    try:
                        MapDimShuffle.apply_to(sdfg=sdfg, map_entry=node)
                    except Exception:
                        # Best effort: not every map admits shuffling.
                        continue

    sdfg.validate()
    return len(config.permute_map)
