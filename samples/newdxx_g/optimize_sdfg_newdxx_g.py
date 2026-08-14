#!/usr/bin/env python3
"""Run the dace_fortran parallelization pipeline on the built newdxx_g SDFG
(clone of samples/vexx_bp_k/optimize_sdfg_vexx_bp_k.py).

This replicates ``dace_fortran.pipelines.optimize`` STAGE BY STAGE (same passes, same
order) instead of calling it directly, for three reasons:

1. BUGFIX (applied here as a monkeypatch so no dace/dace-fortran source changes):
   ``ConvertLengthOneArraysToScalars(preserve_abi=True)`` decides copy-in/copy-out from
   ``_descriptor_is_read``/``_descriptor_is_written``, which only look at AccessNodes.
   Gates read EXCLUSIVELY on interstate edges / branch conditions would be staged into
   transient scalars (scal_*) with NO copy-in: uninitialized gate reads at runtime, and
   a LoopToMap crash later (its loop-local-container scan has the same blind spot).
   The patch widens ``_descriptor_is_read`` to also count interstate-edge and
   LoopRegion/ConditionalBlock condition reads, so the pass wires the copy-ins itself.

2. GATE COPY-IN REPAIR (bug 3 of dace-fortran-fixes-needed-6c99810.md): the shipped
   ``outputs/newdxx_g.sdfg`` was BUILT without the monkeypatch above (the builder
   itself runs the same pass), so ``scal_okvan`` / ``scal_gamma_only`` (and for
   newdxx_g also ``scal_dfftt_ngm``) are already staged writer-less INSIDE the file.
   ``repair_missing_gate_copyins`` wires the missing ``scal_<x> <- <x>[0]`` copy-in
   edges in a fresh start state -- the SDFG-level equivalent of the ``.cpp`` copy-in
   hunks in dace-fortran-manual-fixes-6c99810.patch, applied BEFORE the pipeline
   because both LoopToMap and the staging check below reject writer-less gates.

3. Per-stage impact reporting (loops -> maps, unrolled loops, fusions, collapses).

Output: ``outputs/newdxx_g_opt.sdfg`` (binding signature refrozen, same array ABI).

    cd /workspace/dace-fortran/samples/newdxx_g && \
        PYTHONHASHSEED=0 python3 optimize_sdfg_newdxx_g.py
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
NAME = "newdxx_g"

if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")

import dace  # noqa: E402
from dace.sdfg import nodes as dace_nodes  # noqa: E402
from dace.sdfg.state import ConditionalBlock, LoopRegion  # noqa: E402
from dace.transformation.dataflow.map_collapse import MapCollapse  # noqa: E402
from dace.transformation.interstate.loop_to_map import LoopToMap  # noqa: E402
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended  # noqa: E402
from dace.transformation.pass_pipeline import Pipeline  # noqa: E402
from dace.transformation.passes.full_map_fusion import FullMapFusion  # noqa: E402
import dace.transformation.passes.length_one_array_scalar_conversion as l1mod  # noqa: E402
from dace.transformation.passes.parallelization_prep import ShortLoopUnroll  # noqa: E402
from dace.transformation.passes.persistent_transients import MakeTransientsPersistent  # noqa: E402
from dace.transformation.passes.scalar_fission import ScalarFission  # noqa: E402
from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators  # noqa: E402

from dace_fortran.bindings.frozen_signature import refreeze  # noqa: E402
from dace_fortran.pipelines import num_maps  # noqa: E402

# --------------------------------------------------------------------------- bugfix patch
_orig_is_read = l1mod._descriptor_is_read


def _is_read_incl_interstate(sdfg, name):
    """_descriptor_is_read + interstate-edge and region-condition reads (see module docstring)."""
    if _orig_is_read(sdfg, name):
        return True
    for e in sdfg.all_interstate_edges():
        if name in e.data.free_symbols:
            return True
    for blk in sdfg.all_control_flow_blocks():
        if isinstance(blk, (LoopRegion, ConditionalBlock)):
            for code in blk.get_meta_codeblocks():
                if name in code.get_free_symbols():
                    return True
    return False


l1mod._descriptor_is_read = _is_read_incl_interstate

# --------------------------------------------------------------------------- metrics


def n_loops(sdfg):
    return sum(1 for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion))


def n_states(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.SDFGState))


def stats(sdfg):
    return {"loops": n_loops(sdfg), "maps": num_maps(sdfg),
            "nodes": sum(1 for _ in sdfg.all_nodes_recursive())}


def writerless_edge_read_transients(sdfg):
    """Transient containers read on interstate edges/conditions but never written -- must be empty."""
    written = set()
    for node, parent in sdfg.all_nodes_recursive():
        if isinstance(node, dace_nodes.AccessNode) and parent.in_degree(node) > 0:
            written.add(node.data)
    bad = set()
    for s in sdfg.all_sdfgs_recursive():
        refs = set()
        for e in s.all_interstate_edges():
            refs |= set(e.data.free_symbols)
        for blk in s.all_control_flow_blocks():
            if isinstance(blk, (LoopRegion, ConditionalBlock)):
                for code in blk.get_meta_codeblocks():
                    refs |= set(code.get_free_symbols())
        for name in refs:
            if name in s.arrays and s.arrays[name].transient and name not in written:
                # symbol-mapped nested reads resolve through the NSDFG symbol mapping instead
                bad.add((s.label, name))
    return bad


def repair_missing_gate_copyins(sdfg):
    """Wire the missing ``scal_<x> <- <x>[0]`` copy-ins (see module docstring, item 2).

    For every writer-less transient ``scal_*`` container that IS read (on interstate
    edges / region conditions, or by AccessNodes), add the copy-in edge from its
    length-1 source array in a new start state -- byte-for-byte what the patched
    ``ConvertLengthOneArraysToScalars`` staging would have emitted at build time.
    A writer-less ``scal_*`` without a same-SDFG source array is FATAL.
    """
    repaired = []
    for s in sdfg.all_sdfgs_recursive():
        written, node_reads = set(), set()
        for st in s.all_states():
            for n in st.nodes():
                if isinstance(n, dace_nodes.AccessNode):
                    if st.in_degree(n) > 0:
                        written.add(n.data)
                    if st.out_degree(n) > 0:
                        node_reads.add(n.data)
        refs = set(node_reads)
        for e in s.all_interstate_edges():
            refs |= set(e.data.free_symbols)
        for blk in s.all_control_flow_blocks():
            if isinstance(blk, (LoopRegion, ConditionalBlock)):
                for code in blk.get_meta_codeblocks():
                    refs |= set(code.get_free_symbols())
        todo = sorted(name for name in refs
                      if name.startswith("scal_") and name in s.arrays
                      and s.arrays[name].transient and name not in written)
        if not todo:
            continue
        copyin = s.add_state_before(s.start_state, "scal_copyin_fix3", is_start_block=True)
        for name in todo:
            src = name[len("scal_"):]
            if src not in s.arrays:
                sys.exit(f"FATAL: cannot repair writer-less {name}: no source array "
                         f"'{src}' in SDFG '{s.label}'")
            copyin.add_nedge(copyin.add_access(src), copyin.add_access(name),
                             dace.Memlet(data=src, subset="0"))
            repaired.append((s.label, name, src))
    return repaired


class StageTimer:
    def __init__(self, sdfg):
        self.sdfg = sdfg
        self.prev = stats(sdfg)
        self.t0 = time.time()

    def report(self, label, extra=""):
        cur, t1 = stats(self.sdfg), time.time()
        d = {k: cur[k] - self.prev[k] for k in cur}
        delta = ", ".join(f"{k} {self.prev[k]}->{cur[k]} ({d[k]:+d})" for k in cur if d[k]) or "no structural change"
        print(f"[{label:24s}] {delta}{'  |  ' + extra if extra else ''}  [{t1 - self.t0:.0f}s]", flush=True)
        self.prev, self.t0 = cur, t1


# --------------------------------------------------------------------------- pipeline


def run_pipeline(sdfg, unroll_limit=8, validate=True, max_stages=99):
    """The optimize() stages, with per-stage impact reporting (see module docstring).

    ``max_stages`` truncates the pipeline after the N-th stage (1=len1, 2=unroll,
    3=unique-iter, 4=fission, 5=simplify, 6=fuse1, 7=loop2map, 8=fuse2, 9=mapfus1,
    10=collapse, 11=mapfus2, 12=persistent) -- for bisecting a miscompiling stage.
    """
    st = StageTimer(sdfg)

    rep = repair_missing_gate_copyins(sdfg)
    st.report("gate-copyin-repair",
              f"{len(rep)} copy-ins wired: {sorted(r[1] for r in rep)}" if rep else "none needed")

    def stop_after(k):
        if max_stages <= k:
            print(f"run_pipeline: stopping after stage {k} (max_stages={max_stages})", flush=True)
            if validate:
                sdfg.validate()
            return True
        return False

    # Same stage order as dace_fortran.pipelines.optimize (no specialize: no constants baked).
    l1mod.ConvertLengthOneArraysToScalars(preserve_abi=True).apply_pass(sdfg, {})
    n_copyin = 0
    for state in sdfg.all_states():
        if state.label.startswith("stage_copyin"):
            n_copyin += sum(1 for n in state.nodes()
                            if isinstance(n, dace_nodes.AccessNode) and state.in_degree(n) > 0)
    st.report("len1-to-scalar", f"{n_copyin} copy-ins staged")
    bad = writerless_edge_read_transients(sdfg)
    if bad:
        sys.exit(f"FATAL: writer-less edge-read transients after staging: {sorted(bad)}")
    print("    gate-staging check OK: every edge-read transient has a writer", flush=True)
    if stop_after(1):
        return sdfg

    ShortLoopUnroll(unroll_limit).apply_pass(sdfg, {})
    st.report("short-loop-unroll")
    if stop_after(2):
        return sdfg

    UniqueLoopIterators().apply_pass(sdfg, {})
    st.report("unique-loop-iterators")
    if stop_after(3):
        return sdfg

    Pipeline([ScalarFission()]).apply_pass(sdfg, {})
    st.report("scalar-fission")
    if stop_after(4):
        return sdfg

    sdfg.simplify(validate=validate)
    st.report("simplify")
    if stop_after(5):
        return sdfg

    n = sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate) or 0
    st.report("state-fusion-1", f"{n} fusions")
    if stop_after(6):
        return sdfg

    n = sdfg.apply_transformations_repeated(LoopToMap, validate=validate) or 0
    st.report("loop-to-map", f"{n} loops converted to maps")
    if stop_after(7):
        return sdfg

    n = sdfg.apply_transformations_repeated(StateFusionExtended, validate=validate) or 0
    st.report("state-fusion-2", f"{n} fusions")
    if stop_after(8):
        return sdfg

    FullMapFusion(validate=validate).apply_pass(sdfg, {})
    st.report("map-fusion-1")
    if stop_after(9):
        return sdfg

    n = sdfg.apply_transformations_repeated(MapCollapse, validate=validate) or 0
    st.report("map-collapse", f"{n} collapses")
    if stop_after(10):
        return sdfg

    FullMapFusion(validate=validate).apply_pass(sdfg, {})
    st.report("map-fusion-2")
    if stop_after(11):
        return sdfg

    MakeTransientsPersistent().apply_pass(sdfg, {})
    st.report("persistent-transients")

    if validate:
        sdfg.validate()
        print("final validate OK", flush=True)
    bad = writerless_edge_read_transients(sdfg)
    if bad:
        sys.exit(f"FATAL: writer-less edge-read transients at end of pipeline: {sorted(bad)}")
    return sdfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=OUT / f"{NAME}.sdfg", help="built SDFG to optimize")
    ap.add_argument("--output", type=Path, default=OUT / f"{NAME}_opt.sdfg", help="where to save the result")
    ap.add_argument("--unroll-limit", type=int, default=8, help="ShortLoopUnroll trip-count limit")
    ap.add_argument("--no-validate", action="store_true", help="skip structural validation between stages")
    ap.add_argument("--no-refreeze", action="store_true", help="skip re-snapshotting the frozen binding signature")
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"input SDFG not found: {args.input} (run run_regex_default_{NAME}.py first)")

    t0 = time.time()
    sdfg = dace.SDFG.from_file(str(args.input))
    s0 = stats(sdfg)
    print(f"loaded {args.input.name}: {s0['nodes']} nodes, {s0['loops']} loops, {s0['maps']} maps "
          f"in {time.time() - t0:.0f}s", flush=True)

    run_pipeline(sdfg, unroll_limit=args.unroll_limit, validate=not args.no_validate)

    if not args.no_refreeze:
        # NOTE: on a file-loaded SDFG this snapshots the CURRENT signature (the builder's
        # in-memory freeze does not survive save/load); the real drift check happens in
        # build_opt_lib.py, which rebuilds via SDFGBuilder and refreezes against it.
        refreeze(sdfg)
        print("refreeze(): binding signature re-snapshotted", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sdfg.save(str(args.output))
    s1 = stats(sdfg)
    print(f"saved {args.output}", flush=True)
    print(f"TOTAL: loops {s0['loops']}->{s1['loops']}, maps {s0['maps']}->{s1['maps']}, "
          f"nodes {s0['nodes']}->{s1['nodes']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
