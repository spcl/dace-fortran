# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Standalone copy of the parallelization pipeline the velocity GPU lane runs.

Faithful duplicate of ``dace_fortran.pipelines.optimize`` (dace-fortran @ ci-fix) as called by
``samples/velocity_tendencies/run_velocity_perf.py``: ``optimize(sdfg)`` with no ``symbols`` and
no ``scalars`` (velocity specializes nothing), so the sequence is

    short-loop-unroll -> unique-loop-iterators -> scalar-fission -> simplify
      -> state-fusion-extended -> loop2map -> state-fusion-extended -> mapfusion
      -> make-transients-persistent

Every pass is a stock DaCe pass, so nothing here imports dace_fortran. GPU deltas versus the
source: none in the sequence itself -- the device schedule and the transfers are applied
afterwards by ``gpu_offload.apply_gpu_offload``, which is the only step the CPU lane lacks.
"""
from __future__ import annotations

from dace import SDFG
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.persistent_transients import MakeTransientsPersistent
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

from dace.sdfg import nodes as _nd

UNROLL_LIMIT = 8


def optimize_velocity(sdfg: SDFG, validate: bool = True) -> SDFG:
    """Run the velocity parallelization pipeline in place and return ``sdfg``."""
    ShortLoopUnroll(UNROLL_LIMIT).apply_pass(sdfg, {})
    UniqueLoopIterators().apply_pass(sdfg, {})
    Pipeline([ScalarFission()]).apply_pass(sdfg, {})

    sdfg.simplify(validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)

    sdfg.apply_transformations_repeated(LoopToMap, validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)
    sdfg.apply_transformations_repeated(MapFusionVertical, validate=validate)
    MakeTransientsPersistent().apply_pass(sdfg, {})

    if validate:
        sdfg.validate()
    return sdfg


def num_maps(sdfg: SDFG) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, _nd.MapEntry))
