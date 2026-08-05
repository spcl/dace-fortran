# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-SDFG optimization recipes for Fortran frontend output.

``optimize`` is the end-to-end parallelization pipeline:

    specialize -> short-loop-unroll -> simplify -> state-fusion-extended
      -> [scalar-fission] -> loop2map -> state-fusion-extended -> mapfusion

``scalar_fission`` runs BEFORE loop2map (it splits scalar-carried loop bodies so the loop can
map): CloudSC needs it for parallelism, the ICON ocean kernels do not. ``specialize`` bakes the
known compile-time constants (CloudSC: nclv, ncldqi, ...) so the shape and branch folding
downstream have literals to work on.
"""
from typing import Dict, Optional, Union

from dace import SDFG
from dace.sdfg.utils import specialize_scalars, specialize_symbols
from dace.transformation.dataflow.map_fusion import MapFusion
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.scalar_fission import ScalarFission

Const = Union[float, int, str]


def optimize(sdfg: SDFG,
             *,
             symbols: Optional[Dict[str, Const]] = None,
             scalars: Optional[Dict[str, Const]] = None,
             unroll_limit: int = 8,
             scalar_fission: bool = False,
             validate: bool = True) -> SDFG:
    """Run the parallelization pipeline in place and return ``sdfg``.

    :param sdfg: the SDFG to optimize (mutated in place).
    :param symbols: free symbols to bake to constants (batched, one recursive walk).
    :param scalars: scalar data containers to bake to constants (batched).
    :param unroll_limit: fully unroll constant-trip loops at or below this many iterations.
    :param scalar_fission: run ScalarFission before loop2map (CloudSC parallelism prep).
    :param validate: validate the SDFG after each structural stage.
    """
    if symbols:
        specialize_symbols(sdfg, symbols)
    if scalars:
        specialize_scalars(sdfg, scalars)

    ShortLoopUnroll(unroll_limit).apply_pass(sdfg, {})

    sdfg.simplify(validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)

    if scalar_fission:
        # ScalarFission depends on the ScalarWriteShadowScopes analysis; a bare apply_pass gets an
        # empty pipeline_results and KeyErrors. A Pipeline resolves depends_on() first.
        Pipeline([ScalarFission()]).apply_pass(sdfg, {})

    sdfg.apply_transformations_repeated(LoopToMap, validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)
    sdfg.apply_transformations_repeated(MapFusion, validate=validate)

    if validate:
        sdfg.validate()
    return sdfg


def num_maps(sdfg: SDFG) -> int:
    """Map entries anywhere in ``sdfg``, nested SDFGs included -- the parallelization yardstick."""
    from dace.sdfg import nodes
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
