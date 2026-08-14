#!/usr/bin/env python3
"""addusxx_g full-GPU offload: CPU optimize() pipeline + velocity-style GPU offload.

Method cloned from ``samples/velocity_tendencies/offload_velocity.py`` (schedule
assignment first, then a manual data-movement transformation that mirrors every
kernel-touched non-transient to a ``gpu_<name>`` sibling with copy-in/copy-out
states, then transient promotion / prefix renames / connector reconciliation /
storage propagation).  Parallelization structure follows the hand-written full
GPU offloads of this kernel in ``baseline/ref`` (usxx_kernels.cu: per-g-vector
threads, sequential na/ih/jh accumulation).

Differences vs the velocity script, scoped to THIS kernel's structural contract
(flag='c' arm, blocksize-256 G-loops, see run_optimize_original.py):

1. The input is the raw builder SDFG (``outputs/addusxx_g_original.sdfg``), so
   the CPU ``optimize()`` pipeline runs first (imported from
   optimize_sdfg_addusxx_g: gate copy-in repair + len1 monkeypatch included) to
   turn the Fortran loops into maps at all.
2. ``_rebase_eigts_lbounds``: bug 4 of dace-fortran-fixes-needed-6c99810.md
   (lost ``(-nr:nr)`` lbound of eigts1/2/3) is fixed at the SDFG level here --
   the CPU flow regex-patches the generated .cpp instead, which does not carry
   over to CUDA codegen.
3. ``_convert_scatter_loop_to_map``: the sequential ``rhoc(nl(ig)) += aux2``
   scatter loop is a bare host tasklet driven by an interstate indirection
   read; a host tasklet cannot touch device arrays, so it is manually rebuilt
   as a map whose tasklet does the ``dfftt_nl`` indirection in-kernel.  No WCR:
   within one map instance the written indices are distinct because QE's
   ``dfftt%nl`` is injective (a G-vector -> FFT-grid permutation), and separate
   (na, iblock) instances are serialized by the host loop nest.
4. ``_collapse_kernel_launches``: stock DaCe transformations only
   (MoveLoopIntoMap / StateFusionExtended / FullMapFusion to a fixed point,
   plus a range normalization of the constant-256 zero-fill maps so they fuse
   with their Min(256, ngm-offset)-ranged consumers).  This folds the jh/ih
   accumulation loops INSIDE the kernels -- the structure of
   ``kernel_addusxx_baseline`` in baseline/ref -- instead of launching one
   256-thread kernel per (jh, ih, na, iblock).
5. ``_unblock_cache_blocking_loop``: QE's ``DO iblock = 1, numblock`` is CPU
   cache blocking, not parallelism -- its iterations tile ``1..ngm`` into
   disjoint 256-element chunks.  It is deleted and the block maps re-ranged to
   the full sphere, so a kernel gets ngm threads instead of 256 and the launch
   count loses its ``numblock`` factor.
6. Reductions stay Reduce LIBRARY NODES and run on the GPU (the velocity
   script tasklet-ized them instead).  The one Reduce in this graph (the
   3-element ``(xk-xkq).tau`` phase dot product) sits inside the eigqts
   kernel, so it gets ScheduleType.Sequential + identity=0: the 'auto'
   expansion then lowers it to the deterministic sequential accumulator in
   device code at compile time.  A top-level (host-launched) Reduce would get
   ScheduleType.GPU_Device and expand through ExpandReduceGPUAuto.
7. No explicit stream-sync tasklets: velocity runs under a __DACE_NO_SYNC=1
   contract; this sample keeps DaCe's default synchronization.

Output: ``outputs/addusxx_g_gpu.sdfg`` (Reduce nodes unexpanded, renamed to
``addusxx_g_gpu`` so compilation gets its own .dacecache directory).

    cd /workspace/dace-fortran/samples/addusxx_g && \
        PYTHONHASHSEED=0 python3 offload_addusxx.py
"""
import argparse
import copy
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
NAME = "addusxx_g"

if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")
sys.path.insert(0, str(HERE))

import sympy  # noqa: E402

import dace  # noqa: E402
from dace import data, dtypes  # noqa: E402
from dace import memlet as mm  # noqa: E402
from dace import subsets as sbs  # noqa: E402
from dace import symbolic  # noqa: E402
from dace.config import Config  # noqa: E402
from dace.sdfg import nodes  # noqa: E402
from dace.sdfg.state import LoopRegion, ConditionalBlock, SDFGState  # noqa: E402
from dace.transformation.interstate.move_loop_into_map import MoveLoopIntoMap  # noqa: E402
from dace.transformation.interstate.state_fusion_with_happens_before import StateFusionExtended  # noqa: E402
from dace.transformation.passes.analysis import loop_analysis  # noqa: E402
from dace.transformation.passes.full_map_fusion import FullMapFusion  # noqa: E402

# importing this applies the len1-staging interstate-read monkeypatch; run_pipeline
# also wires the missing scal_* gate copy-ins of the shipped original SDFG (bug 3)
from optimize_sdfg_addusxx_g import run_pipeline  # noqa: E402

_CPU_STORAGES = (
    dtypes.StorageType.Default,
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.Register,
)

_BOUNDARY_NODE_TYPES = (nodes.MapEntry, nodes.MapExit, nodes.NestedSDFG)

_EIGTS = ("eigts1", "eigts2", "eigts3")

_BLOCKSIZE = 256  # QE's `INTEGER, PARAMETER :: blocksize` in addusxx_g/newdxx_g


def OffloadAddusxxToGPU(sdfg: dace.SDFG, exclude_from_offload=(), collapse_rounds=4):
    """Scoped to addusxx_g's post-optimize() structural contract, not a general Pass.

    ``exclude_from_offload`` names arrays that must stay CPU-side regardless.
    ``collapse_rounds=0`` skips the launch-collapse fixed point (debug fallback:
    one kernel per innermost map state, correct but launch-bound).
    """
    # -1 emits no streams: every launch and copy takes the default stream (velocity
    # does the same; here it just serializes the per-(na, iblock) kernel chain).
    Config.set("compiler", "cuda", "max_concurrent_streams", value="-1")

    _rebase_eigts_lbounds(sdfg)
    _convert_scatter_loop_to_map(sdfg)
    if collapse_rounds:
        _collapse_kernel_launches(sdfg, rounds=collapse_rounds)
    _unblock_cache_blocking_loop(sdfg)
    _fix_nsdfg_symbol_scoping(sdfg)
    _assign_schedules(sdfg)
    # Needs schedules-independent structure only; interstate reads at host level
    # must keep their CPU arrays (a mirrored one would hand host code a device ptr).
    excluded = frozenset(exclude_from_offload) | _host_interstate_read_names(sdfg)
    _mirror_kernelside_nontransients_to_gpu(sdfg, excluded)
    _promote_transient_arrays_to_gpu(sdfg)
    # Renaming inner arrays leaves connector bindings stale: never validate between these.
    _ensure_gpu_prefix_for_gpu_storage_arrays(sdfg)
    _reconcile_nsdfg_connector_names(sdfg)
    _propagate_gpu_storage_into_nested_sdfgs(sdfg)
    _configure_reduce_nodes(sdfg)
    _dedup_block_labels(sdfg)
    sdfg.validate()


# --------------------------------------------------------------------- manual fix: eigts


def _rebase_eigts_lbounds(sdfg: dace.SDFG):
    """SDFG-level form of build_opt_lib.py's cpp fix 4.

    QE declares eigts1(-nr1:nr1, nat); the frontend lowers module allocatables
    1-based, so kernel memlets read ``eigts1[mill_at - 1, na - 1]`` -- out of
    bounds for mill <= 0 and the wrong structure factor everywhere else.  The
    true first-dim index is ``mill + nr = (mill - 1) + (d0+1)//2``.  Only
    single-index dim-0 subsets are shifted; the full-range [0:d0] boundary
    memlets into maps/nested SDFGs stay as they are.
    """
    shifted = {k: 0 for k in _EIGTS}
    for g, _ in _walk_sdfgs(sdfg):
        for e in g.all_interstate_edges():
            for m in e.data.get_read_memlets(g.arrays):
                if m.data in _EIGTS:
                    raise RuntimeError(f"eigts read on interstate edge of {g.label}: "
                                       "rebase only handles dataflow memlets")
        for state in g.all_states():
            for e in state.edges():
                md = e.data
                if md is None or md.is_empty() or md.data not in _EIGTS:
                    continue
                rng = md.subset
                b, en, st_ = rng[0]
                if b != en:  # structural sympy equality: shift only single-index subsets
                    continue
                off = symbolic.pystr_to_symbolic(f"({md.data}_d0 + 1) // 2")
                new_dims = [(b + off, en + off, st_)] + [tuple(r) for r in rng[1:]]
                md.subset = sbs.Range(new_dims)
                shifted[md.data] += 1
    if any(n == 0 for n in shifted.values()):
        raise RuntimeError(f"eigts rebase incomplete: {shifted} (expected >=1 site each)")
    print(f"[eigts-rebase          ] dim-0 +((d0+1)//2) at {shifted}", flush=True)


# ---------------------------------------------------------- manual fix: scatter loop->map


def _convert_scatter_loop_to_map(sdfg: dace.SDFG):
    """Rebuild the sequential indirect-scatter loop as a (future GPU) map.

    Pattern (the ``loop__doit`` region the optimize() pipeline leaves behind):
    LoopRegion whose only non-empty state holds ONE tasklet doing
    ``arr[idx(sym)] = arr[idx(sym)] + val`` where ``sym`` is assigned on an
    internal interstate edge as an indirect read ``sym = ind[expr(it)]``.
    Replacement: a single state with a map over the loop range; the tasklet
    reads ``ind[expr(it)]`` through a scalar connector and does the dynamic
    read-modify-write through full-range volume-1 memlets.  Race-free without
    WCR: see module docstring item 3.
    """
    target = None
    for region in sdfg.all_control_flow_regions(recursive=True):
        if not isinstance(region, LoopRegion):
            continue
        blocks = list(region.nodes())
        if not all(isinstance(b, SDFGState) for b in blocks):
            continue
        nonempty = [b for b in blocks if b.number_of_nodes() > 0]
        if len(nonempty) != 1:
            continue
        body = nonempty[0]
        tasklets = [n for n in body.nodes() if isinstance(n, nodes.Tasklet)]
        if len(tasklets) != 1 or any(not isinstance(n, (nodes.Tasklet, nodes.AccessNode))
                                     for n in body.nodes()):
            continue
        body_subset_syms = set()
        for be in body.edges():
            if be.data is not None and not be.data.is_empty() and be.data.subset is not None:
                body_subset_syms |= {str(s) for s in be.data.subset.free_symbols}
        indirection = {}  # sym name -> Memlet of the indirect read feeding a body subset
        for e in region.all_interstate_edges():
            if not e.data.assignments:
                continue
            reads = e.data.get_read_memlets(region.sdfg.arrays)
            for sym in e.data.assignments:
                if str(sym) in body_subset_syms and reads:
                    indirection[str(sym)] = reads[0]
        if indirection:
            if target is not None:
                raise RuntimeError("more than one scatter-loop candidate found")
            target = (region, body, tasklets[0], indirection)
    if target is None:
        raise RuntimeError("scatter loop (loop__doit) not found -- structure drifted")

    region, body, tasklet, indirection = target
    if len(indirection) != 1:
        raise RuntimeError(f"expected exactly one indirection symbol, got {indirection}")
    ind_sym, ind_memlet = next(iter(indirection.items()))

    itervar = region.loop_variable
    start = loop_analysis.get_init_assignment(region)
    end = loop_analysis.get_loop_end(region)
    if start is None or end is None:
        raise RuntimeError("scatter loop bounds not analyzable")
    # Strip a defensive Max(0, x): the enclosing iblock loop never emits an empty
    # block, and the plain Min(...) form is what the sibling compute maps carry --
    # identical ranges are what lets FullMapFusion fuse the scatter kernel in.
    if isinstance(end, sympy.Max) and any(a == 0 for a in end.args):
        end = next(a for a in end.args if a != 0)

    in_edges = [e for e in body.in_edges(tasklet) if e.data is not None]
    out_edges = [e for e in body.out_edges(tasklet) if e.data is not None]
    if len(out_edges) != 1:
        raise RuntimeError("scatter tasklet must have exactly one output")
    scatter_arr = out_edges[0].data.data
    scatter_subset = out_edges[0].data.subset
    val_edges = [e for e in in_edges if e.data.data != scatter_arr]
    if len(val_edges) != 1:
        raise RuntimeError("scatter tasklet must have exactly one value input")
    val_arr = val_edges[0].data.data
    val_subset = copy.deepcopy(val_edges[0].data.subset)

    # Index expression with the interstate symbol replaced by the connector name.
    idx_expr = scatter_subset[0][0].subs(symbolic.pystr_to_symbolic(ind_sym),
                                         symbolic.pystr_to_symbolic("_nl"))
    idx = symbolic.symstr(idx_expr, cpp_mode=False)

    parent = region.parent_graph
    was_start = parent.start_block is region
    st = parent.add_state("scatter_gpu", is_start_block=was_start)

    desc = region.sdfg.arrays[scatter_arr]
    me, _mx = st.add_map("scatter_map", {itervar: sbs.Range([(start, end, 1)])})
    t = st.add_tasklet("scatter", {"_val", "_nl", "_rin"}, {"_rout"},
                       f"_rout[{idx}] = _rin[{idx}] + _val")

    st.add_memlet_path(st.add_read(val_arr), me, t, dst_conn="_val",
                       memlet=mm.Memlet(data=val_arr, subset=val_subset))
    st.add_memlet_path(st.add_read(ind_memlet.data), me, t, dst_conn="_nl",
                       memlet=copy.deepcopy(ind_memlet))

    def _full_dynamic():
        m = mm.Memlet(data=scatter_arr, subset=sbs.Range([(0, desc.shape[0] - 1, 1)]))
        m.dynamic = True
        m.volume = 1
        return m

    st.add_memlet_path(st.add_read(scatter_arr), me, t, dst_conn="_rin", memlet=_full_dynamic())
    st.add_memlet_path(t, _mx, st.add_write(scatter_arr), src_conn="_rout", memlet=_full_dynamic())

    for ie in parent.in_edges(region):
        parent.add_edge(ie.src, st, ie.data)
    for oe in parent.out_edges(region):
        parent.add_edge(st, oe.dst, oe.data)
    parent.remove_node(region)
    if ind_sym in sdfg.symbols:  # dangling indirection symbol would stay a free symbol
        sdfg.remove_symbol(ind_sym)
    print(f"[scatter-loop-to-map   ] {region.label}: {scatter_arr}[{idx}] += {val_arr}, "
          f"range {itervar}={start}:{end} (indirection via {ind_memlet.data})", flush=True)


# ------------------------------------------------------------------- launch collapsing


def _n_maps(sdfg) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _normalize_block_map_ranges(sdfg: dace.SDFG) -> int:
    """Shrink the constant [1:256] zero-fill maps to the Min(256, ngm-offset) range
    of their sibling compute maps.  Elements past the block remainder are never
    read downstream (they were zeroed purely defensively), and identical ranges
    are a precondition for FullMapFusion / MoveLoopIntoMap to collapse the ih/jh
    launch storm.  Only maps whose state writes exclusively transient arrays
    qualify (the aux1/aux2 zero fills)."""
    const256 = sbs.Range([(1, 256, 1)])
    canonical = None
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry) and len(n.map.range) == 1:
            en = n.map.range[0][1]
            s = str(en)
            if "Min" in s and "scal_dfftt_ngm" in s:
                if canonical is None:
                    canonical = en
                elif (canonical - en).simplify() != 0:
                    raise RuntimeError(f"conflicting block ranges: {canonical} vs {en}")
    if canonical is None:
        return 0
    changed = 0
    for n, parent in sdfg.all_nodes_recursive():
        if not isinstance(n, nodes.MapEntry) or n.map.range != const256:
            continue
        state = parent
        writes_ok = all(
            state.sdfg.arrays[x.data].transient
            for x in state.nodes()
            if isinstance(x, nodes.AccessNode) and state.in_degree(x) > 0)
        if not writes_ok:
            continue
        n.map.range = sbs.Range([(1, canonical, 1)])
        changed += 1
    if changed:
        print(f"[normalize-zero-ranges ] {changed} maps [1:256] -> [1:{canonical}]", flush=True)
    return changed


def _collapse_kernel_launches(sdfg: dace.SDFG, rounds=4):
    """Fixed point that folds the sequential jh/ih accumulation loops INTO the
    block kernels (see module docstring item 4).  MoveLoopIntoMap / StateFusion /
    FullMapFusion are stock DaCe; ``_merge_accumulate_into_block_kernel`` is the
    one manual step (FullMapFusion refuses the read-modify-write intermediate),
    and unlocks MoveLoopIntoMap on the ih loop in the following round.
    Validation deferred to the end of the offload."""
    for rnd in range(1, rounds + 1):
        n_mv = sdfg.apply_transformations_repeated(MoveLoopIntoMap, validate=False) or 0
        n_rng = _normalize_block_map_ranges(sdfg)
        n_sf = sdfg.apply_transformations_repeated(StateFusionExtended, validate=False) or 0
        try:
            n_mg = _merge_accumulate_into_block_kernel(sdfg)
        except RuntimeError as exc:
            print(f"[merge-acc-kernel      ] SKIPPED (non-fatal): {exc}", flush=True)
            n_mg = 0
        maps_before = _n_maps(sdfg)
        FullMapFusion(validate=False).apply_pass(sdfg, {})
        n_mf = maps_before - _n_maps(sdfg)
        print(f"[collapse round {rnd}      ] loop-into-map {n_mv}, range-norm {n_rng}, "
              f"state-fusions {n_sf}, acc-merges {n_mg}, map-fusions {n_mf}", flush=True)
        if n_mv + n_rng + n_sf + n_mg + n_mf == 0:
            break


def _merge_accumulate_into_block_kernel(sdfg: dace.SDFG) -> int:
    """Fold the per-ih ``aux2 += aux1 * conj(becphi)`` map and the ``aux1 = 0``
    zero-fill map into the jh-accumulation kernel.

    After MoveLoopIntoMap(jh) + state fusion, a loop-ih body state carries two
    top-level maps over the same block range: a PRODUCER whose nested SDFG
    contains the jh LoopRegion (writing the aux1 intermediate) and a CONSUMER
    (single-state nested SDFG, one tasklet) reading it.  FullMapFusion refuses
    the fusion because aux1/aux2 are read-modify-write at the map boundary, but
    per element it is plainly ``aux1 = 0; for jh: aux1 += ...; aux2 += ...``:
    the consumer tasklet is appended as a final state INSIDE the producer's
    nested SDFG (map param renamed), the zero fill becomes an init state there,
    and the sibling zero-fill state is emptied.  This leaves the ih loop body as
    ONE single-map state, which MoveLoopIntoMap then folds into the kernel too."""
    applied = 0
    for L in [r for r in sdfg.all_control_flow_regions(recursive=True) if isinstance(r, LoopRegion)]:
        for S in [b for b in L.nodes() if isinstance(b, SDFGState)]:
            sdict = S.scope_dict()
            tops = [n for n in S.nodes() if isinstance(n, nodes.MapEntry) and sdict[n] is None]
            if len(tops) != 2:
                continue

            def _sole_nsdfg(me):
                sub = S.scope_subgraph(me, include_entry=False, include_exit=False)
                inner = [n for n in sub.nodes()]
                if len(inner) == 1 and isinstance(inner[0], nodes.NestedSDFG):
                    return inner[0]
                return None

            classified = {}
            for me in tops:
                ns = _sole_nsdfg(me)
                if ns is None:
                    classified = None
                    break
                has_loop = any(isinstance(r, LoopRegion)
                               for r in ns.sdfg.all_control_flow_regions(recursive=True))
                classified[me] = (ns, has_loop)
            if not classified:
                continue
            prods = [me for me, (_, hl) in classified.items() if hl]
            conss = [me for me, (_, hl) in classified.items() if not hl]
            if len(prods) != 1 or len(conss) != 1:
                continue
            prod_me, cons_me = prods[0], conss[0]
            prod_ns, cons_ns = classified[prod_me][0], classified[cons_me][0]
            if prod_me.map.range != cons_me.map.range:
                continue
            prod_mx, cons_mx = S.exit_node(prod_me), S.exit_node(cons_me)

            # consumer must be a single state with a single tasklet
            cons_states = [st for st in cons_ns.sdfg.all_states() if st.number_of_nodes() > 0]
            if len(cons_states) != 1 or list(cons_ns.sdfg.all_interstate_edges()):
                continue
            cst = cons_states[0]
            ctasks = [n for n in cst.nodes() if isinstance(n, nodes.Tasklet)]
            if len(ctasks) != 1:
                continue
            ct = ctasks[0]

            # the intermediate: written by producer exit, read by consumer entry
            inter = None
            for e in S.out_edges(prod_mx):
                if isinstance(e.dst, nodes.AccessNode) and any(
                        e2.dst is cons_me for e2 in S.out_edges(e.dst)):
                    inter = e.dst
            if inter is None:
                continue
            inter_name = inter.data

            p_prod = symbolic.pystr_to_symbolic(prod_me.map.params[0])
            p_cons = symbolic.pystr_to_symbolic(cons_me.map.params[0])

            def _renamed(subset):
                return sbs.Range([tuple(x.subs(p_cons, p_prod) if symbolic.issymbolic(x) else x
                                        for x in r) for r in subset])

            # ---- init state: aux1 element zero fill inside the producer NSDFG
            J = prod_ns.sdfg
            elem_subset = None
            for e in cst.in_edges(ct):
                if e.data.data == inter_name:
                    elem_subset = _renamed(e.data.subset)
            if elem_subset is None:
                raise RuntimeError(f"consumer tasklet does not read intermediate {inter_name}")
            init = J.add_state_before(J.start_block, "acc_merge_init", is_start_block=True)
            zt = init.add_tasklet("zero_fill", {}, {"_out_z"}, "_out_z = (0.0 + (1j * 0.0))")
            init.add_edge(zt, "_out_z", init.add_write(inter_name), None,
                          mm.Memlet(data=inter_name, subset=copy.deepcopy(elem_subset)))

            # ---- final state: the consumer tasklet, appended inside the producer
            sinks = [b for b in J.nodes() if J.out_degree(b) == 0]
            assert len(sinks) == 1, f"producer NSDFG has {len(sinks)} sinks"
            acc = J.add_state_after(sinks[0], "acc_merge_tail")
            newt = acc.add_tasklet(ct.label, set(ct.in_connectors), set(ct.out_connectors),
                                   ct.code.as_string)
            moved_arrays = set()
            for e in cst.in_edges(ct):
                nm = e.data.data
                if nm not in J.arrays:
                    J.add_datadesc(nm, copy.deepcopy(cons_ns.sdfg.arrays[nm]))
                    moved_arrays.add(nm)
                acc.add_edge(acc.add_read(nm), None, newt, e.dst_conn,
                             mm.Memlet(data=nm, subset=_renamed(e.data.subset)))
            out_e = [e for e in cst.out_edges(ct)][0]
            nm = out_e.data.data
            if nm not in J.arrays:
                J.add_datadesc(nm, copy.deepcopy(cons_ns.sdfg.arrays[nm]))
                moved_arrays.add(nm)
            acc.add_edge(newt, out_e.src_conn, acc.add_write(nm), None,
                         mm.Memlet(data=nm, subset=_renamed(out_e.data.subset)))

            # ---- outer wiring: route the consumer's non-intermediate arrays
            # through the producer map instead
            for arr in sorted(moved_arrays):
                desc = J.arrays[arr]
                full = mm.Memlet.from_array(arr, desc)
                srcs = [e.src for e in S.in_edges(cons_me)
                        if isinstance(e.src, nodes.AccessNode) and e.src.data == arr]
                dsts = [e.dst for e in S.out_edges(cons_mx)
                        if isinstance(e.dst, nodes.AccessNode) and e.dst.data == arr]
                if srcs:
                    prod_ns.add_in_connector(arr, force=True)
                    cin, cout = f"IN_{arr}", f"OUT_{arr}"
                    prod_me.add_in_connector(cin)
                    prod_me.add_out_connector(cout)
                    S.add_edge(srcs[0], None, prod_me, cin, copy.deepcopy(full))
                    S.add_edge(prod_me, cout, prod_ns, arr, copy.deepcopy(full))
                if dsts:
                    prod_ns.add_out_connector(arr, force=True)
                    cin, cout = f"IN_{arr}", f"OUT_{arr}"
                    prod_mx.add_in_connector(cin)
                    prod_mx.add_out_connector(cout)
                    S.add_edge(prod_ns, arr, prod_mx, cin, copy.deepcopy(full))
                    S.add_edge(prod_mx, cout, dsts[0], None, copy.deepcopy(full))

            # symbols the RENAMED content actually uses (ijkb0, the ih iterator)
            # must reach J -- NOT the consumer's dead map parameter
            needed = set()
            for st2 in (init, acc):
                for e in st2.edges():
                    if e.data is not None and e.data.subset is not None:
                        needed |= {str(s) for s in e.data.subset.free_symbols}
            have = {str(k) for k in prod_ns.symbol_mapping}
            for s in sorted(needed - have):
                sv = next((v for k, v in cons_ns.symbol_mapping.items() if str(k) == s), None)
                if sv is None:
                    continue  # map param of the producer itself, or locally defined
                prod_ns.symbol_mapping[s] = copy.deepcopy(sv)
                if s not in J.symbols:
                    J.add_symbol(s, dtypes.int64)

            # ---- drop the consumer map and dangling access nodes
            for n in [cons_me, cons_mx, cons_ns]:
                S.remove_node(n)
            for n in list(S.nodes()):
                if isinstance(n, nodes.AccessNode) and S.degree(n) == 0:
                    S.remove_node(n)

            # ---- remove the sibling zero-fill state of the intermediate; the
            # zero fill now runs inside the kernel.  Removal (not just emptying)
            # leaves the merged state as the loop body's ONLY block, which is
            # what MoveLoopIntoMap needs to fold the ih loop in next round.
            for Z in [b for b in L.nodes() if isinstance(b, SDFGState) and b is not S]:
                zmaps = [n for n in Z.nodes() if isinstance(n, nodes.MapEntry)]
                if len(zmaps) != 1:
                    continue
                zwrites = {e.dst.data for e in Z.out_edges(Z.exit_node(zmaps[0]))
                           if isinstance(e.dst, nodes.AccessNode)}
                if zwrites != {inter_name}:
                    continue
                zout = L.out_edges(Z)
                removable = (L.in_degree(Z) == 0 and len(zout) == 1
                             and zout[0].data.is_unconditional())
                # Loop-INVARIANT assignments on the zero state's outgoing edge
                # (the jh bound ``loopend = nh[nt-1]``) hoist to the loop's
                # entry edges; itervar-dependent ones block the removal.
                assigns = dict(zout[0].data.assignments)
                itv = str(L.loop_variable)
                if removable and assigns:
                    parent = L.parent_graph
                    entries = parent.in_edges(L)
                    written_in_L = {n.data for st2 in L.all_states() for n in st2.nodes()
                                    if isinstance(n, nodes.AccessNode) and st2.in_degree(n) > 0}
                    read_arrays = {m.data for m in zout[0].data.get_read_memlets(L.sdfg.arrays)}
                    hoistable = (bool(entries) and not (read_arrays & written_in_L) and all(
                        itv not in symbolic.free_symbols_and_functions(v)
                        and all(k not in e.data.assignments for e in entries)
                        for k, v in assigns.items()))
                    if hoistable:
                        for e in entries:
                            e.data.assignments.update(copy.deepcopy(assigns))
                    else:
                        removable = False
                for n in list(Z.nodes()):
                    Z.remove_node(n)
                if removable:
                    L.remove_node(Z)
                    if L.start_block is not zout[0].dst:
                        raise RuntimeError("start block did not fall to the merged state")
            applied += 1
    return applied


# ------------------------------------------------------------------------ cache unblocking


def _find_cache_block_loop(sdfg: dace.SDFG):
    """``(LoopRegion, total)`` of QE's 256-element cache-blocking loop.

    Recognised through the maps it drives rather than by name: every block map
    ends at ``Min(256, total - 256*iblock + 256)``, so the loop variable and the
    unblocked trip count both fall out of that one range expression.
    """
    found = set()
    for n, _ in sdfg.all_nodes_recursive():
        if not isinstance(n, nodes.MapEntry) or len(n.map.range) != 1:
            continue
        end = n.map.range[0][1]
        if not isinstance(end, sympy.Min) or _BLOCKSIZE not in end.args:
            continue
        rest = next(a for a in end.args if a != _BLOCKSIZE)
        for s in sorted(rest.free_symbols, key=str):
            total = sympy.simplify(rest + _BLOCKSIZE * s - _BLOCKSIZE)
            if s not in total.free_symbols:
                found.add((str(s), total))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one cache-blocking loop, found {sorted(map(str, found))}")
    var, total = next(iter(found))
    loops = [
        r for r in sdfg.all_control_flow_regions(recursive=True)
        if isinstance(r, LoopRegion) and str(r.loop_variable) == var
    ]
    if len(loops) != 1:
        raise RuntimeError(f"no unique LoopRegion for block variable {var}")
    return loops[0], total


def _widen_block_scratch(sdfg: dace.SDFG, total) -> list:
    """The 256-element aux scratch gets one element per G-vector, at every level.

    Persistent lifetime has to go with it: the new extent is a data-dependent
    symbol (``dfftt%ngm``, staged from an argument), which the init-time
    allocation of a Persistent transient cannot see.  Scope is what the sibling
    ngm-sized transients in this kernel family already use.
    """
    names = {
        name
        for g, _ in _walk_sdfgs(sdfg)
        for name, desc in g.arrays.items() if isinstance(desc, data.Array) and tuple(desc.shape) == (_BLOCKSIZE, )
    }
    old_full = sbs.Range([(0, _BLOCKSIZE - 1, 1)])
    widened = []
    for g, in_kernel in _walk_sdfgs(sdfg):
        for name in sorted(names & set(g.arrays)):
            desc = g.arrays[name]
            # A kernel-side TRANSIENT is per-thread scratch, which _promote_transient_arrays_to_gpu
            # sends to Register -- a data-dependent extent there is a device VLA.  The kernel-side
            # NON-transients here are just connector shadows of the host-level array; those are fine.
            if in_kernel and desc.transient:
                raise RuntimeError(f"{g.label}:{name} is per-thread in-kernel scratch -- unblocking it "
                                   "needs it hoisted out of the kernel to GPU_Global first")
            desc.shape = (total, )
            desc.strides = (1, )
            desc.total_size = total
            if desc.lifetime == dtypes.AllocationLifetime.Persistent:
                desc.lifetime = dtypes.AllocationLifetime.Scope
            widened.append(f"{g.label}:{name}")
        for state in g.all_states():
            for e in state.edges():
                if e.data is not None and e.data.data in names and e.data.subset == old_full:
                    e.data.subset = sbs.Range([(0, total - 1, 1)])
    return widened


def _substitute_symbol_with_one(sdfg: dace.SDFG, var: str):
    """Bind ``var`` to 1 in every subset, interstate read and symbol mapping."""
    pat = re.compile(r"(?<![\w.])" + re.escape(var) + r"(?![\w])")
    for g, _ in _walk_sdfgs(sdfg):
        for state in g.all_states():
            for e in state.edges():
                for attr in ("subset", "other_subset"):
                    sub = getattr(e.data, attr, None) if e.data is not None else None
                    if sub is not None and var in sub.free_symbols:
                        sub.replace({var: 1})
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    node.symbol_mapping.pop(var, None)
        for e in g.all_interstate_edges():
            e.data.assignments = {
                k: (pat.sub("1", v) if isinstance(v, str) else v)
                for k, v in e.data.assignments.items()
            }
            if e.data.condition is not None:
                from dace.properties import CodeBlock
                e.data.condition = CodeBlock(pat.sub("1", e.data.condition.as_string), e.data.condition.language)
        if var in g.symbols:
            g.remove_symbol(var)


def _unblock_cache_blocking_loop(sdfg: dace.SDFG):
    """Delete QE's cache-blocking loop; every block map runs the whole G sphere.

    ``DO iblock = 1, numblock`` slices the G loop into disjoint 256-element
    chunks purely for CPU cache reuse: iteration ``iblock`` touches
    ``g = 256*(iblock-1) + 1 .. 256*(iblock-1) + realblocksize`` and nothing
    else, and the chunks tile ``1..ngm`` exactly.  So pinning ``iblock = 1``
    (which turns every ``offset + ig`` back into a plain global ``ig``) and
    running the maps over ``1:ngm`` reproduces the union of all iterations.
    The only loop-carried state is the aux1/aux2 scratch, widened to one element
    per G-vector so each thread keeps its own.  The scatter stays race-free for
    the same reason it already was: ``dfftt%nl`` is injective over the WHOLE
    sphere, not just within a block.

    Launches per (nt, na) drop by a factor ``numblock`` (73 on BaO_nat002) and
    each kernel gets ngm threads instead of 256.
    """
    loop, total = _find_cache_block_loop(sdfg)
    var = str(loop.loop_variable)
    blocks = loop.nodes()
    if len(blocks) != 1:
        raise RuntimeError(f"{loop.label} has {len(blocks)} child blocks, expected exactly 1")

    widened = _widen_block_scratch(sdfg, total)
    retimed = 0
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry) and var in n.map.range.free_symbols:
            n.map.range = sbs.Range([(1, total, 1)])
            retimed += 1
    _substitute_symbol_with_one(sdfg, var)

    parent, child = loop.parent_graph, blocks[0]
    loop.remove_node(child)
    parent.add_node(child, is_start_block=parent.start_block is loop)
    for ie in parent.in_edges(loop):
        parent.add_edge(ie.src, child, ie.data)
    for oe in parent.out_edges(loop):
        parent.add_edge(child, oe.dst, oe.data)
    parent.remove_node(loop)
    sdfg.reset_cfg_list()

    print(
        f"[unblock-cache-loop    ] {loop.label} ({var}) removed: {retimed} maps now 1:{total}, "
        f"scratch widened {widened}",
        flush=True)


def _fix_nsdfg_symbol_scoping(sdfg: dace.SDFG):
    """MoveLoopIntoMap forwards a nested SDFG's free symbols only when they are
    registered in ``sdfg.symbols`` -- interstate-ASSIGNED symbols (the loopend_*
    bounds read from ``nh``) are not, so the moved-in loop's bound ends up
    neither declared inside nor in the symbol mapping and codegen dies with a
    KeyError in ``arglist``.  Declare + identity-map every missing free symbol,
    resolving the dtype from the parent chain instead of re-minting blindly."""
    fixed = []
    for g, _ in _walk_sdfgs(sdfg):
        node = g.parent_nsdfg_node
        if node is None:
            continue
        for s in sorted(map(str, g.free_symbols)):
            missing_map = s not in node.symbol_mapping
            missing_decl = s not in g.symbols
            if not (missing_map or missing_decl):
                continue
            if missing_map:
                node.symbol_mapping[s] = symbolic.pystr_to_symbolic(s)
            if missing_decl:
                dtype, p = None, g.parent_sdfg
                while dtype is None and p is not None:
                    dtype = p.symbols.get(s)
                    p = p.parent_sdfg
                g.add_symbol(s, dtype if dtype is not None else dtypes.int64)
            fixed.append(f"{g.label}:{s}")
    if fixed:
        print(f"[nsdfg-symbol-scoping  ] declared/mapped: {fixed}", flush=True)


# ------------------------------------------------------------------------- schedules


def _walk_sdfgs(g: dace.SDFG, in_kernel: bool = False) -> Iterator[tuple]:
    """Yield (sdfg, in_kernel) pairs; in_kernel is True once the nesting chain has
    crossed a map scope (gotcha: scope_dict restarts per NestedSDFG, so the flag
    must be threaded down instead of trusting scope lookups)."""
    yield g, in_kernel
    for state in g.all_states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                yield from _walk_sdfgs(node.sdfg, in_kernel or sdict[node] is not None)


def _assign_schedules(sdfg: dace.SDFG):
    """Top-of-state maps/libnodes on host-level SDFGs -> GPU_Device; anything in a
    scope or inside a kernel-side nested SDFG -> Sequential."""
    for g, in_kernel in _walk_sdfgs(sdfg):
        for state in g.all_states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry):
                    node.map.schedule = (dtypes.ScheduleType.Sequential
                                         if in_kernel or sdict[node] is not None
                                         else dtypes.ScheduleType.GPU_Device)
                elif isinstance(node, nodes.LibraryNode):
                    node.schedule = (dtypes.ScheduleType.Sequential
                                     if in_kernel or sdict[node] is not None
                                     else dtypes.ScheduleType.GPU_Device)


def _configure_reduce_nodes(sdfg: dace.SDFG):
    """Reductions stay Reduce library nodes and run on the GPU (module docstring
    item 5).  Sum reduces get identity=0 so the 'auto' expansion picks the
    deterministic sequential accumulator inside kernels."""
    from dace.libraries.standard.nodes.reduce import Reduce
    found = 0
    for g, in_kernel in _walk_sdfgs(sdfg):
        for state in g.all_states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if not isinstance(node, Reduce):
                    continue
                found += 1
                device_side = in_kernel or sdict[node] is not None
                node.schedule = (dtypes.ScheduleType.Sequential
                                 if device_side else dtypes.ScheduleType.GPU_Device)
                if node.identity is None and "+" in str(node.wcr):
                    node.identity = 0
                print(f"[reduce-libnode        ] '{node.label}' in state '{state.label}': "
                      f"schedule={node.schedule.name}, identity={node.identity} "
                      f"({'in-kernel' if device_side else 'host-launched kernel'})", flush=True)
    if found == 0:
        print("[reduce-libnode        ] none found", flush=True)


# ----------------------------------------------------------- manual data movement (mirror)


def _host_interstate_read_names(sdfg: dace.SDFG) -> set:
    """Containers read by host-level interstate edges / region conditions (gates,
    loop bounds, type tables).  Mirroring one hands host code a device pointer."""
    names = set()
    for g, in_kernel in _walk_sdfgs(sdfg):
        if in_kernel:
            continue
        for e in g.all_interstate_edges():
            names |= {m.data for m in e.data.get_read_memlets(g.arrays)}
        for blk in g.all_control_flow_blocks():
            if isinstance(blk, (LoopRegion, ConditionalBlock)):
                for code in blk.get_meta_codeblocks():
                    names |= {s for s in code.get_free_symbols() if s in g.arrays}
    return names


def _arrays_needing_gpu_mirror(sdfg: dace.SDFG) -> set:
    """Non-transient CPU-storage Arrays with a kernel-side AccessNode."""
    candidates = {
        name
        for name, arr in sdfg.arrays.items()
        if isinstance(arr, data.Array) and not arr.transient and arr.storage in _CPU_STORAGES
    }
    needs = set()
    for state in sdfg.all_states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data in candidates:
                if _is_kernel_side(node, state, sdict):
                    needs.add(node.data)
    return needs


def _is_kernel_side(node: nodes.AccessNode, state, sdict) -> bool:
    """In a scope, or wired to a scope boundary (map entry/exit, nested SDFG)."""
    if sdict[node] is not None:
        return True
    if any(isinstance(e.src, _BOUNDARY_NODE_TYPES) for e in state.in_edges(node)):
        return True
    return any(isinstance(e.dst, _BOUNDARY_NODE_TYPES) for e in state.out_edges(node))


def _edge_is_kernel_side(edge, sdict, retargeted_nodes: set) -> bool:
    if id(edge.src) in retargeted_nodes or id(edge.dst) in retargeted_nodes:
        return True
    if sdict.get(edge.src) is not None or sdict.get(edge.dst) is not None:
        return True
    return isinstance(edge.src, _BOUNDARY_NODE_TYPES) or isinstance(edge.dst, _BOUNDARY_NODE_TYPES)


def _mirror_kernelside_nontransients_to_gpu(sdfg: dace.SDFG, excluded: frozenset = frozenset()):
    """``gpu_<name>`` siblings for kernel-touched non-transients, plus copy states.

    This is the manual data-movement transformation: one copy-in state before the
    start block uploads every mirrored array; one copy-out state after the unique
    end block downloads ONLY the arrays kernels actually WRITE (rhoc) -- a D2H
    of read-only mirrors would move qgm's 100+MB back for nothing.  Kernel-side
    AccessNodes/memlets are retargeted to the ``gpu_`` mirror while host-side
    ones keep the CPU array."""
    mirror_names = sorted(_arrays_needing_gpu_mirror(sdfg) - excluded)
    if not mirror_names:
        print("[gpu-mirror            ] nothing to mirror", flush=True)
        return

    # Kernel-side WRITTEN subset (before the copy states exist, so the scan
    # only sees compute states).
    kernel_written = set()
    for state in sdfg.all_states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if (isinstance(node, nodes.AccessNode) and node.data in mirror_names
                    and state.in_degree(node) > 0 and _is_kernel_side(node, state, sdict)):
                kernel_written.add(node.data)

    # Before retargeting, so the copy states reference the ORIGINAL names.
    pre = sdfg.add_state_before(sdfg.start_state, label="_cpu_to_gpu_copy_in", is_start_block=True)
    ends = [s for s in sdfg.nodes() if sdfg.out_degree(s) == 0]
    assert len(ends) == 1, f"OffloadAddusxxToGPU expects exactly one end state, got {len(ends)}"
    post = sdfg.add_state_after(ends[0], label="_gpu_to_cpu_copy_out")

    for name in mirror_names:
        arr = sdfg.arrays[name]
        gname = "gpu_" + name
        assert gname not in sdfg.arrays
        gpu_arr = copy.deepcopy(arr)
        gpu_arr.transient = True
        gpu_arr.storage = dtypes.StorageType.GPU_Global
        # Persistent: sizes are init-time symbols, and per-call cudaMalloc of the
        # 100+MB qgm mirror would dominate the benchmark.
        gpu_arr.lifetime = dtypes.AllocationLifetime.Persistent
        sdfg.add_datadesc(gname, gpu_arr)

        pre.add_edge(pre.add_read(name), None, pre.add_write(gname), None,
                     mm.Memlet.from_array(name, arr))
        if name in kernel_written:
            post.add_edge(post.add_read(gname), None, post.add_write(name), None,
                          mm.Memlet.from_array(gname, gpu_arr))

    # Host-side AccessNodes keep the original name and the CPU array.
    retargeted_nodes = set()
    for state in sdfg.all_states():
        if state is pre or state is post:
            continue
        sdict = state.scope_dict()
        for node in list(state.nodes()):
            if not isinstance(node, nodes.AccessNode) or node.data not in mirror_names:
                continue
            if _is_kernel_side(node, state, sdict):
                node.data = "gpu_" + node.data
                retargeted_nodes.add(id(node))

    for state in sdfg.all_states():
        if state is pre or state is post:
            continue
        sdict = state.scope_dict()
        for edge in state.edges():
            if edge.data is None or edge.data.data not in mirror_names:
                continue
            if _edge_is_kernel_side(edge, sdict, retargeted_nodes):
                edge.data.data = "gpu_" + edge.data.data

    print(f"[gpu-mirror            ] {len(mirror_names)} arrays mirrored (upload): {mirror_names}; "
          f"copy-out (kernel-written only): {sorted(kernel_written)}", flush=True)
    # Inner descriptors shadow the outer binding; a CPU-storage inner is an IllegalCopy.
    _propagate_gpu_storage_into_nested_sdfgs(sdfg)


# ------------------------------------------------------------------ storage promotion


def _promote_transient_arrays_to_gpu(sdfg: dace.SDFG):
    """Host-level transient Arrays -> GPU_Global; kernel-side transient Arrays ->
    Register (a GPU_Global allocation cannot be made from device code); every
    Scalar -> Register (a CPU_Heap scalar fed into a kernel is an IllegalCopy)."""
    for g, in_kernel in _walk_sdfgs(sdfg):
        for arr in g.arrays.values():
            if arr.storage not in _CPU_STORAGES:
                continue
            if isinstance(arr, data.Array) and arr.transient:
                if in_kernel:
                    arr.storage = dtypes.StorageType.Register
                    arr.lifetime = dtypes.AllocationLifetime.Scope
                else:
                    arr.storage = dtypes.StorageType.GPU_Global
            elif isinstance(arr, data.Scalar):
                arr.storage = dtypes.StorageType.Register
                if in_kernel:
                    arr.lifetime = dtypes.AllocationLifetime.Scope


# ------------------------------------------- gpu_ prefix / connector renames (velocity)


def _ensure_gpu_prefix_for_gpu_storage_arrays(sdfg: dace.SDFG):
    """Any Array on GPU_Global must carry the ``gpu_`` name prefix."""
    for g, _ in _walk_sdfgs(sdfg):
        renames = {}
        for name, desc in sorted(g.arrays.items()):
            if not isinstance(desc, data.Array):
                continue
            if desc.storage != dtypes.StorageType.GPU_Global or name.startswith("gpu_"):
                continue
            new_name = "gpu_" + name
            if new_name in g.arrays:
                continue
            renames[name] = new_name
        for old, new in renames.items():
            _rename_array_in_sdfg(g, old, new)


def _rename_array_in_sdfg(g: dace.SDFG, old: str, new: str):
    """Avoids ``sdfg.replace``: that routes interstate values through sympy, which
    collapses literals like ``-1.79e+308`` to ``-inf`` (velocity's rationale)."""
    if old == new or old not in g.arrays:
        return
    g.arrays[new] = g.arrays.pop(old)
    for state in g.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.AccessNode) and node.data == old:
                node.data = new
        for edge in state.edges():
            if edge.data is not None and edge.data.data == old:
                edge.data.data = new
    pat = re.compile(r"(?<![\w.])" + re.escape(old) + r"(?![\w])")
    for e in g.all_interstate_edges():
        e.data.assignments = {
            (pat.sub(new, k) if isinstance(k, str) else k):
            (pat.sub(new, v) if isinstance(v, str) else v)
            for k, v in e.data.assignments.items()
        }
        if e.data.condition is not None:
            cs = e.data.condition.as_string
            new_cs = pat.sub(new, cs)
            if new_cs != cs:
                from dace.properties import CodeBlock
                e.data.condition = CodeBlock(new_cs, e.data.condition.language)


def _reconcile_nsdfg_connector_names(sdfg: dace.SDFG):
    """Connector name must equal its outer memlet ``data``; the mirror/prefix
    phases break that.  Fixed-point, because renaming one level exposes
    mismatches one level down."""
    for _ in range(8):
        changed = False
        for state in sdfg.all_states():
            for node in list(state.nodes()):
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                for e in list(state.in_edges(node)):
                    if e.data is None or e.data.data is None:
                        continue
                    if e.dst_conn and e.dst_conn != e.data.data:
                        _rename_nsdfg_connector(node, e.dst_conn, e.data.data, direction="in")
                        e.dst_conn = e.data.data
                        changed = True
                for e in list(state.out_edges(node)):
                    if e.data is None or e.data.data is None:
                        continue
                    if e.src_conn and e.src_conn != e.data.data:
                        _rename_nsdfg_connector(node, e.src_conn, e.data.data, direction="out")
                        e.src_conn = e.data.data
                        changed = True
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    _reconcile_nsdfg_connector_names(node.sdfg)
        if not changed:
            return


def _rename_nsdfg_connector(nsdfg_node: nodes.NestedSDFG, old_name: str, new_name: str,
                            direction: str):
    if old_name == new_name:
        return
    conns = nsdfg_node.in_connectors if direction == "in" else nsdfg_node.out_connectors
    if old_name not in conns:
        return
    dtype = conns[old_name]
    if direction == "in":
        nsdfg_node.remove_in_connector(old_name)
        nsdfg_node.add_in_connector(new_name, dtype=dtype, force=True)
    else:
        nsdfg_node.remove_out_connector(old_name)
        nsdfg_node.add_out_connector(new_name, dtype=dtype, force=True)
    if new_name in nsdfg_node.sdfg.arrays:
        # Already renamed by the other direction; connector lines up, don't collide.
        return
    _rename_array_in_sdfg(nsdfg_node.sdfg, old_name, new_name)


def _propagate_gpu_storage_into_nested_sdfgs(sdfg: dace.SDFG):
    """Inner descriptors inherit their GPU_Global outer binding, recursively."""
    for state in sdfg.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            parent_sdfg = state.sdfg if hasattr(state, "sdfg") else state.parent
            host_only_inner = _inner_names_read_on_interstate_edges(node.sdfg)
            for e in list(state.in_edges(node)) + list(state.out_edges(node)):
                if e.data is None or e.data.data is None:
                    continue
                outer_desc = parent_sdfg.arrays.get(e.data.data)
                if outer_desc is None or outer_desc.storage != dtypes.StorageType.GPU_Global:
                    continue
                conn = e.dst_conn if e.dst is node else e.src_conn
                if conn is None or conn in host_only_inner:
                    continue
                inner = node.sdfg.arrays.get(conn)
                if isinstance(inner, data.Array):
                    inner.storage = dtypes.StorageType.GPU_Global
            _propagate_gpu_storage_into_nested_sdfgs(node.sdfg)


def _inner_names_read_on_interstate_edges(inner_sdfg: dace.SDFG) -> set:
    """Names referenced on interstate edges of HOST-level descendants of
    ``inner_sdfg`` only: device-level interstate reads (the mill/ijtoh
    indirections inside kernels) execute in the kernel and NEED the GPU copy."""
    names = set()
    arr_names = set(inner_sdfg.arrays.keys())
    for g, in_kernel in _walk_sdfgs(inner_sdfg):
        if in_kernel:
            continue
        for e in g.all_interstate_edges():
            texts = list(e.data.assignments.values())
            if e.data.condition is not None:
                texts.append(e.data.condition.as_string)
            for t in texts:
                if not isinstance(t, str):
                    continue
                for n in arr_names:
                    if n in t:
                        names.add(n)
    return names


def _dedup_block_labels(sdfg: dace.SDFG):
    """The frontend reuses block labels across regions; state fusion / loop
    restructuring can land two same-named blocks in ONE region, which validation
    rejects.  Deterministic rename in region node order."""
    n = 0
    for g, _ in _walk_sdfgs(sdfg):
        for region in g.all_control_flow_regions(recursive=True):
            if isinstance(region, ConditionalBlock):
                continue
            seen = set()
            for blk in region.nodes():
                if blk.label in seen:
                    i = 1
                    while f"{blk.label}_dedup{i}" in seen:
                        i += 1
                    blk.label = f"{blk.label}_dedup{i}"
                    n += 1
                seen.add(blk.label)
    if n:
        print(f"[dedup-block-labels    ] {n} blocks renamed", flush=True)


# ------------------------------------------------------------------------------- main


def _summary(sdfg: dace.SDFG):
    from collections import Counter
    sched = Counter()
    launch_sites = 0
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry):
            sched[n.map.schedule.name] += 1
            if n.map.schedule == dtypes.ScheduleType.GPU_Device:
                launch_sites += 1
        elif isinstance(n, nodes.LibraryNode):
            sched[f"LIB:{n.schedule.name}"] += 1
    print(f"schedules: {dict(sched)}; static GPU kernel launch sites: {launch_sites}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=OUT / f"{NAME}_original.sdfg")
    ap.add_argument("--output", type=Path, default=OUT / f"{NAME}_gpu.sdfg")
    ap.add_argument("--collapse-rounds", type=int, default=4,
                    help="launch-collapse fixed-point rounds (0 = skip)")
    ap.add_argument("--unroll-limit", type=int, default=8)
    args = ap.parse_args()

    if not args.input.is_file():
        sys.exit(f"input SDFG not found: {args.input}")

    t0 = time.time()
    sdfg = dace.SDFG.from_file(str(args.input))
    print(f"loaded {args.input.name}: {sum(1 for _ in sdfg.all_nodes_recursive())} nodes "
          f"in {time.time() - t0:.0f}s", flush=True)

    # CPU parallelization pipeline (loops -> maps); includes gate copy-in repair.
    run_pipeline(sdfg, unroll_limit=args.unroll_limit, validate=True)

    t0 = time.time()
    OffloadAddusxxToGPU(sdfg, collapse_rounds=args.collapse_rounds)
    print(f"GPU offload done in {time.time() - t0:.0f}s", flush=True)

    sdfg.name = f"{NAME}_gpu"  # own .dacecache directory, distinct from the CPU builds
    try:
        from dace_fortran.bindings.frozen_signature import refreeze
        refreeze(sdfg)
        print("refreeze(): binding signature re-snapshotted", flush=True)
    except Exception as exc:  # noqa: BLE001 -- Fortran bindings not needed for the GPU bench
        print(f"refreeze skipped (non-fatal for the Python-call bench): {exc}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sdfg.save(str(args.output))
    print(f"saved {args.output}", flush=True)
    _summary(sdfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
