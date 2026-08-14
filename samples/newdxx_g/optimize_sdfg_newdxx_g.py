#!/usr/bin/env python3
"""Run the dace_fortran parallelization pipeline on the built newdxx_g SDFG
(clone of samples/vexx_bp_k/optimize_sdfg_vexx_bp_k.py).

The pipeline itself is ``dace_fortran.pipelines.optimize``, CALLED -- this sample runs
the same stage list as CloudSC, not a copy of it.  Two sample-specific bookends wrap
the call:

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

3. A writer-less edge-read transient audit after the pipeline, before/after
   loop/map/node counts around each bookend, and ``assign_blas_implementations``:
   a BLAS library node at the top of a host state lowers to OpenBLAS instead of
   the ``pure`` fallback (one inside a map scope keeps ``pure`` -- see there).

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
import dace.transformation.passes.length_one_array_scalar_conversion as l1mod  # noqa: E402

from dace_fortran.bindings.frozen_signature import refreeze  # noqa: E402
from dace_fortran.pipelines import num_maps, optimize  # noqa: E402

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


def top_level_library_nodes(sdfg):
    """Library nodes at the top of a HOST-level state.

    ``scope_dict()`` restarts per nested SDFG, so a node at the top of a nested
    SDFG's state looks unscoped even when the nested SDFG itself sits in a map --
    the flag has to be threaded down instead.
    """
    def walk(g, in_scope):
        for st in g.all_states():
            sdict = st.scope_dict()
            for n in st.nodes():
                if isinstance(n, dace_nodes.NestedSDFG):
                    yield from walk(n.sdfg, in_scope or sdict[n] is not None)
                elif isinstance(n, dace_nodes.LibraryNode) and not (in_scope or sdict[n] is not None):
                    yield n

    yield from walk(sdfg, False)


def assign_blas_implementations(sdfg, impl="OpenBLAS"):
    """Top-level BLAS library nodes lower to a real host BLAS, not to ``pure``.

    Top-level ONLY: inside a map scope the call would be issued once per iteration
    from within an OpenMP region, and on the GPU lane that same node ends up inside
    a kernel where no host BLAS can be linked at all.  Those keep ``pure``, which
    ``offload_<kernel>.py`` decides on its own side.
    """
    picked = []
    for node in top_level_library_nodes(sdfg):
        if impl in type(node).implementations:
            node.implementation = impl
            picked.append(f"{type(node).__name__} '{node.label}'")
    print(f"[blas-implementation     ] {impl}: {picked or 'no top-level BLAS library node'}", flush=True)


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


def run_pipeline(sdfg, unroll_limit=8, validate=True):
    """``dace_fortran.pipelines.optimize`` -- the shared pipeline, CALLED not copied.

    Only the two sample-specific bookends stay here: the gate copy-in repair,
    which has to run BEFORE the pipeline (both LoopToMap and len1 staging reject
    writer-less gates, module docstring item 2), and the writer-less transient
    audit afterwards.
    """
    st = StageTimer(sdfg)

    rep = repair_missing_gate_copyins(sdfg)
    st.report("gate-copyin-repair",
              f"{len(rep)} copy-ins wired: {sorted(r[1] for r in rep)}" if rep else "none needed")

    optimize(sdfg, unroll_limit=unroll_limit, validate=validate)
    st.report("pipelines.optimize")

    bad = writerless_edge_read_transients(sdfg)
    if bad:
        sys.exit(f"FATAL: writer-less edge-read transients at end of pipeline: {sorted(bad)}")
    assign_blas_implementations(sdfg)
    if validate:
        print("final validate OK", flush=True)
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
