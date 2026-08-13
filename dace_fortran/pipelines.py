# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-SDFG optimization recipes for Fortran frontend output.

``optimize`` is the end-to-end parallelization pipeline:

    specialize -> short-loop-unroll -> unique-loop-iterators -> scalar-fission -> simplify
      -> state-fusion-extended -> loop2map -> state-fusion-extended -> mapfusion

``scalar_fission`` runs unconditionally, BEFORE simplify (it splits scalar-carried loop bodies so
the loop can map downstream): LoopToMap needs it in general, not just CloudSC. ``specialize``
bakes the known compile-time constants (CloudSC: nclv, ncldqi, ...) so the shape and branch
folding downstream have literals to work on.
"""
import copy
from typing import Any, Dict, Optional, Set, Union

from dace.transformation.dataflow.map_collapse import MapCollapse
import numpy as np

from dace import SDFG
from dace.sdfg.utils import specialize_scalars, specialize_symbols
from dace.transformation.dataflow.map_fusion_vertical import MapFusionVertical
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

Const = Union[float, int, str]


def accepted_call_args(sdfg: SDFG) -> Set[str]:
    """Names ``sdfg`` can be called with: its arglist plus its free symbols."""
    return set(sdfg.arglist()) | {str(s) for s in sdfg.free_symbols}


def verify_numerics(reference: SDFG, optimized: SDFG, inputs: Dict[str, Any]) -> None:
    """Run both SDFGs on the same inputs and require BIT-IDENTICAL results.

    The pipeline reorders statements and forms maps but never reassociates arithmetic, so anything
    short of bit-identical is a bug, not rounding -- same bar the e2e lanes hold (``max_diff == 0``).

    Every argument is compared, not just a declared output set: these kernels write through their
    arguments, so an in-place output is indistinguishable from an input by signature alone.

    :param reference: the pre-optimization SDFG.
    :param optimized: the post-optimization SDFG.
    :param inputs: concrete call arguments; each SDFG gets its own copy, since they mutate in place.
                   Names either SDFG does not accept are dropped from its call: specializing bakes
                   constants out of the signature, so the two signatures need not agree.
    :raises AssertionError: if any argument differs after the two runs.
    """
    # The two SDFGs share a name, and the build folder is keyed on it -- compiling both would put
    # them in one directory and the second would clobber the first.
    reference = copy.deepcopy(reference)
    reference.name = f"{reference.name}_preopt"

    def fresh() -> Dict[str, Any]:
        return {
            k: (v.copy(order='F' if v.flags.f_contiguous else 'C') if isinstance(v, np.ndarray) else v)
            for k, v in inputs.items()
        }

    ref_accepts, opt_accepts = accepted_call_args(reference), accepted_call_args(optimized)
    ref_args, opt_args = fresh(), fresh()
    reference(**{k: v for k, v in ref_args.items() if k in ref_accepts})
    optimized(**{k: v for k, v in opt_args.items() if k in opt_accepts})

    mismatched = []
    # Only names BOTH sides were handed are evidence: one the optimized SDFG never received is
    # untouched by construction and would read as a spurious mismatch.
    for name in sorted(ref_accepts & opt_accepts):
        ref_val = ref_args.get(name)
        if not isinstance(ref_val, np.ndarray):
            continue
        opt_val = opt_args[name]
        if np.array_equal(ref_val, opt_val, equal_nan=np.issubdtype(ref_val.dtype, np.inexact)):
            continue
        if np.issubdtype(ref_val.dtype, np.number):
            delta = np.nanmax(np.abs(ref_val.astype(np.float64) - opt_val.astype(np.float64)))
            mismatched.append(f"{name}: max|d|={delta:.6e}")
        else:
            mismatched.append(f"{name}: differs")
    if mismatched:
        raise AssertionError("optimize() changed the numerics (must be bit-exact):\n  " + "\n  ".join(mismatched))


def optimize(sdfg: SDFG,
             *,
             symbols: Optional[Dict[str, Const]] = None,
             scalars: Optional[Dict[str, Const]] = None,
             unroll_limit: int = 8,
             validate: bool = True,
             verify_inputs: Optional[Dict[str, Any]] = None) -> SDFG:
    """Run the parallelization pipeline in place and return ``sdfg``.

    :param sdfg: the SDFG to optimize (mutated in place).
    :param symbols: free symbols to bake to constants (batched, one recursive walk).
    :param scalars: scalar data containers to bake to constants (batched).
    :param unroll_limit: fully unroll constant-trip loops at or below this many iterations.
    :param validate: validate the SDFG after each structural stage. STRUCTURAL only -- it says
                     nothing about whether the transformations preserved values.
    :param verify_inputs: call arguments to check numerics with once the pipeline is done. The
                          pre-optimization SDFG is snapshotted and both are run on these inputs,
                          requiring bit-identical results (:func:`verify_numerics`). Costs a
                          deepcopy plus a second compile and run, so it is opt-in. Arguments must
                          be supplied by the caller rather than generated here: these kernels index
                          through connectivity arrays, so random integers read out of bounds.
    """
    reference = copy.deepcopy(sdfg) if verify_inputs is not None else None

    if symbols:
        specialize_symbols(sdfg, symbols)
    if scalars:
        specialize_scalars(sdfg, scalars)

    ShortLoopUnroll(unroll_limit).apply_pass(sdfg, {})
    # default assign_loop_iterator_post_value=True keeps Fortran counted-DO exit-value semantics;
    # shared iterator names otherwise make LoopToMap refuse merged siblings.
    UniqueLoopIterators().apply_pass(sdfg, {})
    # ScalarFission depends on the ScalarWriteShadowScopes analysis; a bare apply_pass gets an
    # empty pipeline_results and KeyErrors. A Pipeline resolves depends_on() first.
    Pipeline([ScalarFission()]).apply_pass(sdfg, {})

    sdfg.simplify(validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)

    sdfg.apply_transformations_repeated(LoopToMap, validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)
    sdfg.apply_transformations_repeated(MapFusionVertical, validate=validate)
    sdfg.apply_transformations_repeated(MapCollapse, validate=validate)

    from dace.transformation.passes.persistent_transients import MakeTransientsPersistent
    MakeTransientsPersistent().apply_pass(sdfg, {})

    if validate:
        sdfg.validate()
    if verify_inputs is not None:
        verify_numerics(reference, sdfg, verify_inputs)
    return sdfg


def num_maps(sdfg: SDFG) -> int:
    """Map entries anywhere in ``sdfg``, nested SDFGs included -- the parallelization yardstick."""
    from dace.sdfg import nodes
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))
