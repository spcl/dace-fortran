#!/usr/bin/env python3
"""newdxx_g full-GPU offload: CPU optimize() pipeline + velocity-style GPU offload.

Clone of ``samples/addusxx_g/offload_addusxx.py`` (method:
``samples/velocity_tendencies/offload_velocity.py`` -- schedule assignment
first, then a manual data-movement transformation that mirrors every
kernel-touched non-transient to a ``gpu_<name>`` sibling with copy-in/copy-out
states, then transient promotion / prefix renames / connector reconciliation /
storage propagation), adapted to newdxx_g's post-optimize() structure.

newdxx_g differs from addusxx_g in ways that SIMPLIFY the offload:

* No indirect-scatter loop: the ``deexx(ikb0+i) += ...`` updates live INSIDE
  the per-(iblock, na) map over the projector index (LoopToMap already proved
  independence), and the ``vc(nl(g))`` indirection is a GATHER map over the
  full G sphere.  The addusxx scatter-loop-to-map transformation is therefore
  absent here; a structural check refuses the offload if such a loop appears.
* The block reductions became LIBRARY NODES already: the per-(ih, jh) block
  dot product is a BLAS ``Dot`` node, and the 3-element ``(xk-xkq).tau`` phase
  reduction is a ``Reduce`` node -- both sit inside kernels and stay library
  nodes running on the GPU (``_configure_library_nodes``).
* No launch storm: per (iblock, na) there are exactly two kernels (block
  structure-factor gather + projector map).  The collapse fixed point still
  runs -- its guards make it a no-op where nothing applies.

newdxx-specific additions:

* ``_wrap_bare_device_tasklets``: the frontend leaves a bare host tasklet
  (``set_fact``: fact = omega) writing a transient that promotion sends to
  GPU_Global; a host tasklet cannot write device memory, so it is wrapped in a
  one-iteration map that becomes a trivial kernel.
* Dual-residency mirroring: ``nh`` is read by HOST interstate edges (the
  projector-count loop bounds) AND flows into kernels as dataflow.  Unlike the
  velocity/addusxx exclusion (which would strand the kernel side on a CPU
  pointer), only names with NO kernel-side AccessNode are excluded; ``nh``
  gets a ``gpu_nh`` mirror for the kernel side while host interstate reads
  keep the CPU array (interstate reads reference the name directly, not an
  AccessNode, so per-node retargeting leaves them alone -- and connector
  reconciliation rewrites the DEVICE-level interstate reads inside the nested
  SDFG to the mirror).
* ``_rebase_eigts_lbounds`` is unchanged from addusxx (bug 4 of
  dace-fortran-fixes-needed-6c99810.md, fixed on the memlet subsets so it
  survives CUDA codegen).

The two DaCe 2.0.0a5 workarounds are carried over unchanged:
``_fix_nsdfg_symbol_scoping`` (MoveLoopIntoMap drops interstate-ASSIGNED
symbols from nested SDFGs) and ``_dedup_block_labels`` (state fusion can land
two same-named blocks in one region).

Output: ``outputs/newdxx_g_gpu.sdfg`` (library nodes unexpanded, renamed to
``newdxx_g_gpu`` so compilation gets its own .dacecache directory).

    cd /workspace/dace-fortran/samples/newdxx_g && \
        PYTHONHASHSEED=0 python3 offload_newdxx.py
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
NAME = "newdxx_g"

if (Path.cwd() / "dace").is_dir():
    sys.exit("run from a directory without a 'dace' subdir")
sys.path.insert(0, str(HERE))

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
from dace.transformation.passes.full_map_fusion import FullMapFusion  # noqa: E402

# importing this applies the len1-staging interstate-read monkeypatch; run_pipeline
# also wires the missing scal_* gate copy-ins of the shipped original SDFG (bug 3)
from optimize_sdfg_newdxx_g import run_pipeline  # noqa: E402

_CPU_STORAGES = (
    dtypes.StorageType.Default,
    dtypes.StorageType.CPU_Heap,
    dtypes.StorageType.Register,
)

_BOUNDARY_NODE_TYPES = (nodes.MapEntry, nodes.MapExit, nodes.NestedSDFG)

_EIGTS = ("eigts1", "eigts2", "eigts3")


def OffloadNewdxxToGPU(sdfg: dace.SDFG, exclude_from_offload=(), collapse_rounds=4):
    """Scoped to newdxx_g's post-optimize() structural contract, not a general Pass."""
    Config.set("compiler", "cuda", "max_concurrent_streams", value="-1")

    _rebase_eigts_lbounds(sdfg)
    _dedup_tasklet_inputs(sdfg)
    _assert_no_host_indirection_loops(sdfg)
    _wrap_bare_device_tasklets(sdfg)
    if collapse_rounds:
        _collapse_kernel_launches(sdfg, rounds=collapse_rounds)
    _fix_nsdfg_symbol_scoping(sdfg)
    _assign_schedules(sdfg)
    _mirror_kernelside_nontransients_to_gpu(sdfg, frozenset(exclude_from_offload))
    _promote_transient_arrays_to_gpu(sdfg)
    # Renaming inner arrays leaves connector bindings stale: never validate between these.
    _ensure_gpu_prefix_for_gpu_storage_arrays(sdfg)
    _reconcile_nsdfg_connector_names(sdfg)
    _propagate_gpu_storage_into_nested_sdfgs(sdfg)
    _configure_library_nodes(sdfg)
    _dedup_block_labels(sdfg)
    sdfg.validate()


# --------------------------------------------------------------------- manual fix: eigts


def _rebase_eigts_lbounds(sdfg: dace.SDFG):
    """SDFG-level form of build_opt_lib.py's cpp fix 4 (see addusxx clone)."""
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


def _dedup_tasklet_inputs(sdfg: dace.SDFG):
    """Merge duplicate tasklet input edges (same data, same subset) into one
    connector.  The frontend emits one connector per syntactic use
    (``_in_auxvc_0`` / ``_in_auxvc_1`` for the same ``auxvc[g]``), and DaCe
    2.0.0a5's nested-SDFG codegen then drops the ``const`` qualifier on that
    array's function parameter -- the CUDA build fails with a const/non-const
    argument mismatch against the (correctly const) kernel parameter."""
    merged = 0
    for g, _ in _walk_sdfgs(sdfg):
        for state in g.all_states():
            for t in [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]:
                groups = {}
                for e in state.in_edges(t):
                    if e.data is None or e.data.is_empty() or e.dst_conn is None:
                        continue
                    groups.setdefault((e.data.data, str(e.data.subset)), []).append(e)
                for (_dname, _sub), edges in sorted(groups.items()):
                    if len(edges) < 2:
                        continue
                    keep = edges[0]
                    code = t.code.as_string
                    for e in edges[1:]:
                        code = re.sub(rf"\b{re.escape(e.dst_conn)}\b", keep.dst_conn, code)
                        t.remove_in_connector(e.dst_conn)
                        state.remove_edge(e)
                        # drop the now-dangling duplicate source node, if any
                        if isinstance(e.src, nodes.AccessNode) and state.degree(e.src) == 0:
                            state.remove_node(e.src)
                        merged += 1
                    from dace.properties import CodeBlock
                    t.code = CodeBlock(code, t.code.language)
    if merged:
        print(f"[dedup-tasklet-inputs  ] {merged} duplicate input edges merged", flush=True)


# ------------------------------------------------- structural guard + bare-tasklet wrap


def _assert_no_host_indirection_loops(sdfg: dace.SDFG):
    """newdxx_g has no addusxx-style host scatter loop (a LoopRegion whose only
    content is one tasklet driven by an interstate indirection read).  If one
    appears, the offload needs the addusxx scatter-loop-to-map transformation."""
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
        if len(tasklets) == 1 and all(isinstance(n, (nodes.Tasklet, nodes.AccessNode))
                                      for n in body.nodes()):
            has_ind = any(e.data.get_read_memlets(region.sdfg.arrays)
                          for e in region.all_interstate_edges())
            if has_ind:
                raise RuntimeError(f"host indirection loop '{region.label}' found -- port "
                                   "the addusxx scatter-loop-to-map transformation")
    print("[host-indirection-check] none found (deexx updates are in-kernel)", flush=True)


def _wrap_bare_device_tasklets(sdfg: dace.SDFG):
    """Wrap top-level host tasklets that WRITE transient Arrays into 1-iteration
    maps.  Promotion sends those arrays to GPU_Global (they are read inside
    kernels), and a bare host tasklet cannot write device memory; as a
    one-iteration GPU map it becomes a trivial kernel (set_fact: fact=omega)."""
    wrapped = []
    for state in sdfg.all_states():
        sdict = state.scope_dict()
        for t in [n for n in state.nodes()
                  if isinstance(n, nodes.Tasklet) and sdict[n] is None]:
            outs = [e for e in state.out_edges(t)
                    if e.data is not None and e.data.data is not None]
            if not any(isinstance(sdfg.arrays.get(e.data.data), data.Array)
                       and sdfg.arrays[e.data.data].transient for e in outs):
                continue
            me, mx = state.add_map(f"wrap_{t.label}", {"__wrap_it": sbs.Range([(0, 0, 1)])})
            in_edges = list(state.in_edges(t))
            for i, e in enumerate(in_edges):
                state.remove_edge(e)
                cin, cout = f"IN_{i}", f"OUT_{i}"
                me.add_in_connector(cin)
                me.add_out_connector(cout)
                state.add_edge(e.src, e.src_conn, me, cin, copy.deepcopy(e.data))
                state.add_edge(me, cout, t, e.dst_conn, copy.deepcopy(e.data))
            if not in_edges:
                state.add_nedge(me, t, mm.Memlet())
            for i, e in enumerate(list(state.out_edges(t))):
                if e.dst is mx:
                    continue
                state.remove_edge(e)
                cin, cout = f"IN_w{i}", f"OUT_w{i}"
                mx.add_in_connector(cin)
                mx.add_out_connector(cout)
                state.add_edge(t, e.src_conn, mx, cin, copy.deepcopy(e.data))
                state.add_edge(mx, cout, e.dst, e.dst_conn, copy.deepcopy(e.data))
            wrapped.append(t.label)
    print(f"[wrap-bare-tasklets    ] {wrapped if wrapped else 'none needed'}", flush=True)


# ------------------------------------------------------------------- launch collapsing


def _n_maps(sdfg) -> int:
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry))


def _normalize_block_map_ranges(sdfg: dace.SDFG) -> int:
    """Shrink constant [1:256] zero-fill maps to the Min(256, ngm-offset) block
    range of the sibling compute maps (precondition for fusion; the tail
    elements are never read).  Only maps whose state writes exclusively
    transient arrays qualify."""
    const256 = sbs.Range([(1, 256, 1)])
    canonical = None
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, nodes.MapEntry) and len(n.map.range) == 1:
            en = n.map.range[0][1]
            s = str(en)
            if "Min" in s and "ngms" in s:
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
    """Same fixed point as the addusxx clone.  newdxx_g has no launch storm
    (the projector loop is already a map), so most rounds are no-ops; the
    guards in every stage make that safe and visible."""
    for rnd in range(1, rounds + 1):
        n_mv = sdfg.apply_transformations_repeated(MoveLoopIntoMap, validate=False) or 0
        n_rng = _normalize_block_map_ranges(sdfg)
        n_sf = sdfg.apply_transformations_repeated(StateFusionExtended, validate=False) or 0
        maps_before = _n_maps(sdfg)
        FullMapFusion(validate=False).apply_pass(sdfg, {})
        n_mf = maps_before - _n_maps(sdfg)
        print(f"[collapse round {rnd}      ] loop-into-map {n_mv}, range-norm {n_rng}, "
              f"state-fusions {n_sf}, map-fusions {n_mf}", flush=True)
        if n_mv + n_rng + n_sf + n_mf == 0:
            break


def _fix_nsdfg_symbol_scoping(sdfg: dace.SDFG):
    """DaCe 2.0.0a5 workaround: MoveLoopIntoMap forwards a nested SDFG's free
    symbols only when registered in ``sdfg.symbols``; interstate-ASSIGNED loop
    bounds are not, leaving them undeclared/unmapped -> codegen KeyError in
    ``arglist``.  Declare + identity-map, resolving dtype from the parent chain."""
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
    """Yield (sdfg, in_kernel); the flag is threaded down because scope_dict
    restarts per NestedSDFG."""
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


def _configure_library_nodes(sdfg: dace.SDFG):
    """Reductions stay LIBRARY NODES and run on the GPU.  In-kernel Reduce
    nodes get Sequential + identity=0 (the 'auto' expansion then emits the
    deterministic sequential accumulator in device code); other in-kernel
    library nodes (the per-(ih,jh) block-dot ``Dot``) are forced to their
    'pure' expansion -- the default BLAS pick must not link a host library into
    a kernel.  Host-level ones would get GPU_Device."""
    from dace.libraries.standard.nodes.reduce import Reduce
    found = 0
    for g, in_kernel in _walk_sdfgs(sdfg):
        for state in g.all_states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if not isinstance(node, nodes.LibraryNode):
                    continue
                found += 1
                device_side = in_kernel or sdict[node] is not None
                node.schedule = (dtypes.ScheduleType.Sequential
                                 if device_side else dtypes.ScheduleType.GPU_Device)
                if isinstance(node, Reduce):
                    if node.identity is None and "+" in str(node.wcr):
                        node.identity = 0
                elif device_side and "pure" in type(node).implementations:
                    node.implementation = "pure"
                print(f"[library-node          ] {type(node).__name__} '{node.label}' in "
                      f"'{state.label}': schedule={node.schedule.name}, "
                      f"impl={node.implementation}"
                      f"{', identity=' + str(node.identity) if isinstance(node, Reduce) else ''}"
                      f" ({'in-kernel' if device_side else 'host-launched kernel'})", flush=True)
    if found == 0:
        print("[library-node          ] none found", flush=True)


# ----------------------------------------------------------- manual data movement (mirror)


def _host_interstate_read_names(sdfg: dace.SDFG) -> set:
    """Containers read by host-level interstate edges / region conditions."""
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

    Copy-in uploads every mirror; copy-out downloads ONLY kernel-written arrays
    (deexx).  Unlike the addusxx/velocity flow, names read by HOST interstate
    edges are NOT excluded when they also have kernel-side AccessNodes (nh):
    interstate reads reference the NAME, not an AccessNode, so per-node
    retargeting leaves the host reads on the CPU array while the kernel side
    gets the mirror -- and connector reconciliation rewrites the DEVICE-level
    interstate reads inside nested SDFGs to the mirror name."""
    host_reads = _host_interstate_read_names(sdfg)
    kernel_needed = _arrays_needing_gpu_mirror(sdfg)
    dual = sorted(host_reads & kernel_needed)
    if dual:
        print(f"[gpu-mirror            ] dual residency (host interstate reads stay on the "
              f"CPU array): {dual}", flush=True)
    mirror_names = sorted(kernel_needed - excluded)
    if not mirror_names:
        print("[gpu-mirror            ] nothing to mirror", flush=True)
        return

    kernel_written = set()
    for state in sdfg.all_states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if (isinstance(node, nodes.AccessNode) and node.data in mirror_names
                    and state.in_degree(node) > 0 and _is_kernel_side(node, state, sdict)):
                kernel_written.add(node.data)

    pre = sdfg.add_state_before(sdfg.start_state, label="_cpu_to_gpu_copy_in", is_start_block=True)
    ends = [s for s in sdfg.nodes() if sdfg.out_degree(s) == 0]
    assert len(ends) == 1, f"OffloadNewdxxToGPU expects exactly one end state, got {len(ends)}"
    post = sdfg.add_state_after(ends[0], label="_gpu_to_cpu_copy_out")

    for name in mirror_names:
        arr = sdfg.arrays[name]
        gname = "gpu_" + name
        assert gname not in sdfg.arrays
        gpu_arr = copy.deepcopy(arr)
        gpu_arr.transient = True
        gpu_arr.storage = dtypes.StorageType.GPU_Global
        gpu_arr.lifetime = dtypes.AllocationLifetime.Persistent
        sdfg.add_datadesc(gname, gpu_arr)

        pre.add_edge(pre.add_read(name), None, pre.add_write(gname), None,
                     mm.Memlet.from_array(name, arr))
        if name in kernel_written:
            post.add_edge(post.add_read(gname), None, post.add_write(name), None,
                          mm.Memlet.from_array(gname, gpu_arr))

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

    print(f"[gpu-mirror            ] {len(mirror_names)} arrays mirrored (upload): "
          f"{mirror_names}; copy-out (kernel-written only): {sorted(kernel_written)}",
          flush=True)
    _propagate_gpu_storage_into_nested_sdfgs(sdfg)


# ------------------------------------------------------------------ storage promotion


def _promote_transient_arrays_to_gpu(sdfg: dace.SDFG):
    """Host-level transient Arrays -> GPU_Global; kernel-side transient Arrays ->
    Register; every Scalar -> Register."""
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
    """Avoids ``sdfg.replace`` (sympy collapses interstate literals)."""
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
    """Connector name must equal its outer memlet data; fixed-point."""
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
    """Names referenced on interstate edges of HOST-level descendants only:
    device-level interstate reads (mill / dfftt_nl / nh indirections inside
    kernels) execute in the kernel and NEED the GPU copy."""
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
    """DaCe 2.0.0a5 workaround: state fusion can land two same-named blocks in
    one region, which validation rejects.  Deterministic rename."""
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
            sched[f"LIB:{type(n).__name__}:{n.schedule.name}"] += 1
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

    run_pipeline(sdfg, unroll_limit=args.unroll_limit, validate=True)

    t0 = time.time()
    OffloadNewdxxToGPU(sdfg, collapse_rounds=args.collapse_rounds)
    print(f"GPU offload done in {time.time() - t0:.0f}s", flush=True)

    sdfg.name = f"{NAME}_gpu"  # own .dacecache directory, distinct from the CPU builds
    try:
        from dace_fortran.bindings.frozen_signature import refreeze
        refreeze(sdfg)
        print("refreeze(): binding signature re-snapshotted", flush=True)
    except Exception as exc:  # noqa: BLE001 -- Fortran bindings not needed for the Python bench
        print(f"refreeze skipped (non-fatal for the Python-call bench): {exc}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sdfg.save(str(args.output))
    print(f"saved {args.output}", flush=True)
    _summary(sdfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
