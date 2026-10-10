# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-SDFG optimization recipes for Fortran frontend output.

``optimize`` is the end-to-end parallelization pipeline:

    specialize -> len1-to-scalar -> short-loop-unroll -> unique-loop-iterators -> scalar-fission
      -> simplify -> state-fusion-extended -> demote-non-structural-symbols -> scalar-fission
      -> order-free-reductions -> loop2map -> state-fusion-extended -> mapfusion -> map-collapse
      -> mapfusion -> make-transients-persistent -> bind-omp-thread-count

``scalar_fission`` runs unconditionally, BEFORE simplify (it splits scalar-carried loop bodies so
the loop can map downstream): LoopToMap needs it in general, not just CloudSC. ``specialize``
runs FIRST and bakes the known compile-time constants (CloudSC: nclv, ncldqi, ...), so the
short-loop unroll sees literal trip counts and the shape and branch folding downstream have
literals to work on. ``len1_to_scalar`` runs with ``preserve_abi``, so the rest of the pipeline
sees a plain scalar instead of a one-element buffer while the SDFG signature keeps its array form. The builder already runs it on frontend output; repeating it here
covers an SDFG that reached ``optimize`` some other way, and the pass is idempotent.
``mapfusion`` is ``FullMapFusion`` (vertical +
horizontal fusion run together to a fixed point), applied a second time after ``map-collapse``
since a freshly-collapsed nest can expose fusions the first pass missed. ``bind-omp-thread-count``
defines the team-size symbol of thread-strided persistent maps at SDFG entry, when the SDFG uses it.
"""

import ast
import copy
from typing import Any, Union, cast

import numpy as np
from dace import SDFG, Memlet, dtypes, subsets
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.sdfg.utils import specialize_scalars, specialize_symbols
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.wcr_conversion import AugAssignToWCR
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.demote_non_structural_symbols import DemoteNonStructuralSymbols
from dace.transformation.passes.full_map_fusion import FullMapFusion
from dace.transformation.passes.length_one_array_scalar_conversion import (
    STAGING_STATE_PREFIXES,
    ConvertLengthOneArraysToScalars,
)
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll
from dace.transformation.passes.scalar_fission import ScalarFission
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

from dace_fortran.omp_threads import BindOmpThreadCount

Const = Union[float, int, str]


def accepted_call_args(sdfg: SDFG) -> set[str]:
    """Names ``sdfg`` can be called with: its arglist plus its free symbols."""
    return set(sdfg.arglist()) | {str(s) for s in sdfg.free_symbols}


def abi_proxy_transients(sdfg: SDFG) -> set[str]:
    """Transients that exist only as the copy-in/copy-out shadow of a NON-transient.

    ``ConvertLengthOneArraysToScalars(preserve_abi=True)`` keeps the signature array and stages it
    through a fresh transient scalar, so that scalar's value leaves the SDFG through the copy-out --
    it is an ABI proxy, externally visible, and must be treated like the non-transient it stands for.
    Derived from the staging edges themselves rather than the ``scal_`` name the pass happens to mint,
    and keyed on that pass's OWN state-label constant so a rename there cannot silently unhook this.
    """
    proxies: set[str] = set()
    for state in sdfg.all_states():
        if not state.label.startswith(STAGING_STATE_PREFIXES):
            continue
        for edge in state.edges():
            if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                continue
            src, dst = sdfg.arrays[edge.src.data], sdfg.arrays[edge.dst.data]
            if src.transient != dst.transient:
                proxies.add(edge.src.data if src.transient else edge.dst.data)
    return proxies


def fission_scalars(sdfg: SDFG) -> dict[str, set[str]]:
    """Run ``ScalarFission`` on ``sdfg``, leaving ABI-proxy transients whole.

    Fissioning a proxy splits its value over several shadows while the copy-out still reads exactly
    one of them, so a kernel write landing in another shadow is silently lost to the caller. The
    proxies are hidden behind the pass's OWN non-transient gate for the duration of the run --
    genuine internal transients (``difcoef`` and friends) still fission.

    ``ScalarFission`` depends on the ``ScalarWriteShadowScopes`` analysis; a bare ``apply_pass`` gets
    an empty ``pipeline_results`` and KeyErrors, so a ``Pipeline`` resolves ``depends_on()`` first.
    """
    proxies = abi_proxy_transients(sdfg)
    for name in proxies:
        sdfg.arrays[name].transient = False
    try:
        return (Pipeline([ScalarFission()]).apply_pass(sdfg, {}) or {}).get("ScalarFission") or {}
    finally:
        for name in proxies:
            sdfg.arrays[name].transient = True


#: Updates whose parallel combination gives the same bits in any order, so forming them keeps ``optimize`` bit-exact;
#: a floating-point sum would round differently.
ORDER_FREE_UPDATES = frozenset({"max", "min", "or", "and"})
#: Updates exact in any order on integers only (a counter such as ICON's ``clip_count = clip_count + 1``).
ORDER_FREE_INTEGER_UPDATES = frozenset({"+", "|", "&", "^"})


def _is_integer(sdfg: SDFG, name: str) -> bool:
    """Whether container ``name`` holds integers (Fortran ``LOGICAL`` included, stored as an integer)."""
    return np.issubdtype(sdfg.arrays[name].dtype.type, np.integer)


#: The reduction a boolean constant store is equivalent to: ``x = True`` is ``x = x or True``.
_IDEMPOTENT_STORE_WCR = {True: "lambda a, b: a or b", False: "lambda a, b: a and b"}


def _enclosing_loop_vars(state: SDFGState) -> set[str]:
    """The iteration variables of the loops around ``state`` within its SDFG."""
    names: set[str] = set()
    region = state.parent_graph
    while region is not None and not isinstance(region, SDFG):
        if isinstance(region, LoopRegion) and region.loop_variable:
            names.add(region.loop_variable)
        region = region.parent_graph
    return names


def order_free_reductions(sdfg: SDFG) -> int:
    """Turn every update ``LoopToMap`` would see as a loop-carried write into the reduction it is.

    * ``x = max(x, e)`` / ``min`` / ``x = x or e`` / ``and`` becomes a WCR write (``AugAssignToWCR``);
    * a boolean constant store to an element no enclosing loop's iterator selects (Fortran's
      ``levmask(jb, jk) = .TRUE.`` inside the ``jc`` loop) becomes the OR / AND reduction it is equivalent to.

    Only :data:`ORDER_FREE_UPDATES`, and :data:`ORDER_FREE_INTEGER_UPDATES` on integers, are formed, so the parallel
    result is bit-identical. Returns the updates rewritten.
    """
    rewritten = 0
    xform = AugAssignToWCR()
    for sd in sdfg.all_sdfgs_recursive():
        changed = True
        while changed:
            changed = False
            for state in sd.states():
                for tasklet in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
                    if tasklet.language is not dtypes.Language.Python or state.entry_node(tasklet) is not None:
                        continue
                    outs = state.out_edges(tasklet)
                    if len(outs) != 1 or not isinstance(outs[0].dst, nodes.AccessNode) or outs[0].data.wcr:
                        continue
                    out = outs[0]
                    if state.in_degree(tasklet) == 0:
                        rewritten += _idempotent_store(sd, state, tasklet, out)
                        continue
                    update = AugAssignToWCR.python_update(tasklet, state.in_edges(tasklet), out)
                    if update is None or not (
                        update[0] in ORDER_FREE_UPDATES
                        or (update[0] in ORDER_FREE_INTEGER_UPDATES and _is_integer(sd, out.dst.data))
                    ):
                        continue
                    source = update[1].src
                    if not isinstance(source, nodes.AccessNode) or source.data != out.dst.data:
                        continue
                    xform.setup_match(
                        sd,
                        sd.cfg_id,
                        sd.node_id(state) if state.parent_graph is sd else -1,
                        # Bound by node OBJECT, not by node id: ``PatternNode`` resolves a non-int value as-is.
                        cast(
                            dict[Any, int],
                            {
                                AugAssignToWCR.input: source,
                                AugAssignToWCR.tasklet: tasklet,
                                AugAssignToWCR.output: out.dst,
                            },
                        ),
                        0,
                        override=True,
                    )
                    if not xform.can_be_applied(state, 0, sd):
                        continue
                    xform.apply(state, sd)
                    rewritten += 1
                    changed = True  # the apply may fission ``state``: sweep again
                    break
                if changed:
                    break
    return rewritten


def _idempotent_store(sdfg: SDFG, state: SDFGState, tasklet: nodes.Tasklet, out: MultiConnectorEdge[Memlet]) -> int:
    """Give a boolean constant store, ``out = True`` or ``out = False``, the reduction it is equivalent to when an
    enclosing loop writes the same element from every iteration."""
    code = tasklet.code.code
    statement = code[0] if isinstance(code, list) and len(code) == 1 else None
    if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Constant):
        return 0
    value = statement.value.value
    name = out.dst.data
    if not isinstance(value, bool) or not (sdfg.arrays[name].dtype == dtypes.bool_ or _is_integer(sdfg, name)):
        return 0
    written = out.data.get_dst_subset(out, state)
    if not isinstance(written, subsets.Subset) or not _enclosing_loop_vars(state) - {
        str(s) for s in written.free_symbols
    }:
        return 0
    out.data.wcr = _IDEMPOTENT_STORE_WCR[value]
    return 1


def verify_numerics(reference: SDFG, optimized: SDFG, inputs: dict[str, Any]) -> None:
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
    cast(Any, reference).name = f"{reference.name}_preopt"  # dace declares SDFG.name as a Property

    def fresh() -> dict[str, Any]:
        return {
            k: (v.copy(order="F" if v.flags.f_contiguous else "C") if isinstance(v, np.ndarray) else v)
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


def optimize(
    sdfg: SDFG,
    *,
    symbols: dict[str, Const] | None = None,
    scalars: dict[str, Const] | None = None,
    unroll_limit: int = 8,
    validate: bool = True,
    verify_inputs: dict[str, Any] | None = None,
) -> SDFG:
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

    ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})

    ShortLoopUnroll(unroll_limit).apply_pass(sdfg, {})
    # default assign_loop_iterator_post_value=True keeps Fortran counted-DO exit-value semantics;
    # shared iterator names otherwise make LoopToMap refuse merged siblings.
    UniqueLoopIterators().apply_pass(sdfg, {})
    fission_scalars(sdfg)

    sdfg.simplify(validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)

    # After the last simplify, whose scalar-to-symbol promotion would undo it: a counter or guard that bounds no loop
    # and indexes no array becomes a scalar again, so the update below can turn into a reduction.
    DemoteNonStructuralSymbols().apply_pass(sdfg, {})
    # A demoted symbol is one name for every loop that assigned it; split it per write scope again, as the symbol
    # was private to each iteration, or LoopToMap sees one container shared by all those loops.
    fission_scalars(sdfg)
    order_free_reductions(sdfg)
    sdfg.apply_transformations_repeated(LoopToMap, validate=validate)
    sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate)
    # FullMapFusion: MapFusionVertical + MapFusionHorizontal run together to a fixed point, not just
    # vertical -- horizontal fuses maps that only share an input, no producer/consumer edge.
    FullMapFusion(validate=validate).apply_pass(sdfg, {})
    sdfg.apply_transformations_repeated(MapCollapse, validate=validate)
    # A freshly-collapsed nest can expose a fusion the pass above missed.
    FullMapFusion(validate=validate).apply_pass(sdfg, {})

    from dace.transformation.passes.persistent_transients import MakeTransientsPersistent

    MakeTransientsPersistent().apply_pass(sdfg, {})
    BindOmpThreadCount().apply_pass(sdfg, {})

    if validate:
        sdfg.validate()
    if verify_inputs is not None and reference is not None:
        verify_numerics(reference, sdfg, verify_inputs)
    return sdfg


def num_maps(sdfg: SDFG) -> int:
    """Map entries anywhere in ``sdfg``, nested SDFGs included -- the parallelization yardstick."""
    from dace.sdfg import nodes

    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def num_loops(sdfg: SDFG) -> int:
    """Loops anywhere in ``sdfg``, nested SDFGs included -- what the pipeline left sequential."""
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))
