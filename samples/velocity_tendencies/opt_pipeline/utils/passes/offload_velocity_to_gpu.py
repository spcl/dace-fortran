"""Velocity-tendencies stage-4 GPU offload.

Takes a stage-3 SDFG (all map schedules ``Default``, transients at the
root SDFG with ``AllocationLifetime.SDFG``, mixed storage) and drives
DaCe's own ``GPUTransformSDFG`` over the *host-level* SDFG tree:

1. **Block maps** -- the per-patch ``DO jb = i_startblk .. i_endblk``
   loops -- are identified structurally (their range depends on a
   symbol that some interstate edge assigns from a ``*_start_block`` /
   ``*_end_block`` array) and handed to ``GPUTransformSDFG`` as
   ``host_maps``. They orchestrate kernel launches and stay
   ``Sequential`` on the host.
2. ``GPUTransformSDFG`` does the rest at that level: clone
   non-transients to ``gpu_<name>`` with copy-in / copy-out states,
   schedule top-level maps ``GPU_Device``, sequentialize inner maps,
   promote transients, wrap host tasklets that touch device data in
   1-iteration GPU maps, and add host copies for containers read by
   host-code interstate edges.
3. Recurse into the NestedSDFGs *inside* block maps -- that is where
   the velocity kernels live and ``GPUTransformSDFG`` does not descend
   on its own. Inner connector descriptors inherit the storage of
   their outer binding first, so the recursive application sees them
   as already-resident and neither clones nor copies them.

**Host-only data stays on the host.** Any array that no kernel scope
ever touches (patch index descriptors, ``*_is_associated`` flags, the
``max_vcfl_dyn`` round-trip output, fields consumed only by host-side
code) is analysed up front, passed as ``host_data``, and its clone --
``GPUTransformSDFG`` clones every read array regardless -- is reverted.
Without that, host code ends up dereferencing device pointers and
validation fails with "accessed on host".
"""
import re

from dace import SDFG, data, dtypes
from dace.sdfg import nodes
from dace.sdfg.state import AbstractControlFlowRegion
from dace.transformation.interstate import GPUTransformSDFG

from utils.passes.replace_reductions_with_tasklets import replace_reductions_with_tasklets

_COPY_IN_LABEL = '_cpu_to_gpu_copy_in'
_COPY_OUT_LABEL = '_gpu_to_cpu_copy_out'


def OffloadVelocityToGPU(sdfg: SDFG, exclude_from_offload=()):
    """Drive the offload. Not a general-purpose DaCe ``Pass`` -- scoped
    to velocity stage 3's structural contract.

    ``exclude_from_offload``: array names the caller insists on keeping
    CPU-side on top of the ones the host-only analysis finds."""
    # Pin the default stream. ``max_concurrent_streams = -1`` makes
    # DaCe's CUDA codegen create no streams at all and pass ``nullptr``
    # to every kernel launch and async copy. Multi-stream scheduling
    # mixes host/device ordering in ways that complicate the
    # ``__DACE_NO_SYNC=1`` contract; on the default stream the only
    # syncs needed are at the copy boundaries.
    import dace as _dace
    _dace.Config.set("compiler", "cuda", "max_concurrent_streams", value="-1")

    block_syms = block_loop_symbols(sdfg)
    host_only = host_only_arrays(sdfg, block_syms)
    host_only[id(sdfg)] |= set(exclude_from_offload)
    offload_host_level(sdfg, block_syms, host_only, root=True)
    bracket_copy_states_with_syncs(sdfg)
    # Swap standard-library Reduce nodes for tasklets that dispatch to
    # the velocity-owned reduction helpers (CPU / GPU launch / device
    # inline, chosen from detected storage + scope). Runs last so it
    # sees the final storage layout.
    replace_reductions_with_tasklets(sdfg)
    sdfg.validate()


# -- structural block-map identity ---------------------------------------


def block_loop_symbols(sdfg: SDFG) -> frozenset:
    """Symbols an interstate edge assigns from a ``*_start_block`` /
    ``*_end_block`` descriptor array. A map whose range depends on one
    of them iterates patch blocks."""
    syms = set()
    for g in all_sdfgs(sdfg):
        for e in g.all_interstate_edges():
            reads = e.data.get_read_memlets(g.arrays)
            if not reads:
                continue
            for sym, value in e.data.assignments.items():
                text = str(value)
                for m in reads:
                    if (m.data.endswith('_start_block') or m.data.endswith('_end_block')) and m.data in text:
                        syms.add(sym)
    return frozenset(syms)


def is_block_map(map_node: 'nodes.Map', block_syms: frozenset) -> bool:
    return bool({str(s) for s in map_node.range.free_symbols} & block_syms)


def block_map_entries(g: SDFG, block_syms: frozenset) -> list:
    """Top-level block maps of ``g``, as ``(state, entry)`` pairs."""
    out = []
    for state in g.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry) and sdict[node] is None and is_block_map(node.map, block_syms):
                out.append((state, node))
    return out


# -- host-only data analysis ---------------------------------------------


def nsdfg_bindings(g: SDFG, out=None) -> dict:
    """``(id(inner_sdfg), connector) -> (id(outer_sdfg), array)`` for
    every NestedSDFG in the tree."""
    out = {} if out is None else out
    for state in g.all_states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            for e in state.in_edges(node):
                if e.data is not None and e.data.data and e.dst_conn:
                    out[(id(node.sdfg), e.dst_conn)] = (id(g), e.data.data)
            for e in state.out_edges(node):
                if e.data is not None and e.data.data and e.src_conn:
                    out[(id(node.sdfg), e.src_conn)] = (id(g), e.data.data)
            nsdfg_bindings(node.sdfg, out)
    return out


def host_only_arrays(sdfg: SDFG, block_syms: frozenset) -> dict:
    """Per host-level SDFG, the arrays that no kernel scope touches.

    Walks the host-level tree the same way the offload does -- root,
    then the bodies of block maps -- and marks everything reachable
    from a would-be kernel, resolving inner names to their outer
    binding. What is left over is host-only and must not go to device
    storage: patch index descriptors, ``*_is_associated`` flags, the
    ``max_vcfl_dyn`` round-trip output, host-launched reduction
    results. Returns ``{id(sdfg): names}``."""
    binds = nsdfg_bindings(sdfg)

    def outer_key(g, name):
        key, seen = (id(g), name), set()
        while key in binds and key not in seen:
            seen.add(key)
            key = binds[key]
        return key

    touched = set()
    host_level = []

    def mark_kernel(g, state, entry):
        for node in state.scope_subgraph(entry).nodes():
            if isinstance(node, nodes.AccessNode):
                touched.add(outer_key(g, node.data))
            elif isinstance(node, nodes.NestedSDFG):
                for inner in all_sdfgs(node.sdfg):
                    for name in inner.arrays:
                        touched.add(outer_key(inner, name))
        for e in state.in_edges(entry) + state.out_edges(state.exit_node(entry)):
            if e.data is not None and e.data.data:
                touched.add(outer_key(g, e.data.data))

    def descend(g):
        host_level.append(g)
        for state in g.all_states():
            sdict = state.scope_dict()
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and sdict[node] is None:
                    if is_block_map(node.map, block_syms):
                        for inner in state.scope_subgraph(node).nodes():
                            if isinstance(inner, nodes.NestedSDFG):
                                descend(inner.sdfg)
                    else:
                        mark_kernel(g, state, node)
                elif isinstance(node, nodes.NestedSDFG) and sdict[node] is None:
                    descend(node.sdfg)

    descend(sdfg)
    spread_through_wrapped_tasklets(host_level, touched, outer_key)
    return {id(g): {name for name in g.arrays if outer_key(g, name) not in touched} for g in host_level}


def spread_through_wrapped_tasklets(host_level, touched, outer_key):
    """``GPUTransformSDFG`` moves free host tasklets into 1-iteration
    GPU maps, dragging everything they touch into device code. It wraps
    a tasklet as soon as one of its edges carries an Array -- only
    tasklets that talk exclusively to ``Scalar`` descriptors stay on
    the host -- and again for anything adjacent to device data. Data
    such a tasklet touches is therefore not host-only either; leaving
    it on the host is an ``Illegal copy`` at codegen (the validator
    does not catch this direction). Close the set over that rule."""
    changed = True
    while changed:
        changed = False
        for g in host_level:
            for state in g.all_states():
                sdict = state.scope_dict()
                for node in state.nodes():
                    if not isinstance(node, nodes.Tasklet) or sdict[node] is not None:
                        continue
                    keys, wrapped = set(), False
                    for e in state.in_edges(node) + state.out_edges(node):
                        if e.data is None or not e.data.data:
                            continue
                        key = outer_key(g, e.data.data)
                        keys.add(key)
                        if isinstance(g.arrays.get(e.data.data), data.Array) or key in touched:
                            wrapped = True
                    if wrapped and not keys <= touched:
                        touched |= keys
                        changed = True


# -- the offload itself ---------------------------------------------------


def offload_host_level(g: SDFG, block_syms: frozenset, host_only: dict, root: bool):
    """Apply ``GPUTransformSDFG`` to one host-level SDFG, then recurse
    into the bodies of its block maps."""
    host_entries = block_map_entries(g, block_syms)
    keep_host = {n for n in host_only.get(id(g), ()) if n in g.arrays}
    promote_interstate_read_transients(g, keep_host)

    options = {
        'host_maps': [entry.guid for _, entry in host_entries],
        'host_data': sorted(keep_host),
        'simplify': False,
        'sequential_innermaps': True,
    }
    if not root:
        # Inner non-transients are connector bindings: the caller
        # already placed them, so no copy states inside the body.
        bindings = ','.join(sorted(n for n, d in g.arrays.items() if not d.transient))
        options['exclude_copyin'] = bindings
        options['exclude_copyout'] = bindings

    before = set(g.arrays)
    host_storage = {n: g.arrays[n].storage for n in sorted(keep_host)}
    # No intermediate validation: the tree is only consistent again
    # once the recursion below has processed the block-map bodies.
    g.apply_transformations(GPUTransformSDFG, options=options, validate=False, validate_all=False)
    revert_gpu_clones(g, sorted(n for n in keep_host if n in before and not g.arrays[n].transient), before)
    # ``host_data`` is not honoured everywhere: the transformation also
    # pushes the outputs of GPU-scheduled library nodes and nested
    # SDFGs to device storage without consulting it (a Reduce writing a
    # host-read scalar hits this). Put host-only descriptors back.
    for name, storage in host_storage.items():
        if name in g.arrays and g.arrays[name].storage != storage:
            g.arrays[name].storage = storage

    # Whatever GPUTransformSDFG left on ``Default`` at this level is
    # host code: block maps, and maps it kept back because they touch
    # host_data. Default is not enough -- DaCe's type inference
    # promotes a Default-schedule top-level map to GPU_Device for a
    # CUDA target, which would put host loops on the device.
    for state in g.states():
        sdict = state.scope_dict()
        for node in state.nodes():
            if (isinstance(node, nodes.MapEntry) and sdict[node] is None
                    and node.map.schedule == dtypes.ScheduleType.Default):
                node.map.schedule = dtypes.ScheduleType.Sequential

    for state, entry in host_entries:
        for node in state.scope_subgraph(entry).nodes():
            if isinstance(node, nodes.NestedSDFG):
                inherit_binding_storage(g, state, node)
                offload_host_level(node.sdfg, block_syms, host_only, root=False)


def promote_interstate_read_transients(g: SDFG, keep_host: set):
    """``GPUTransformSDFG`` records containers that are *already* on
    GPU storage when it starts, so it can copy them back for host-code
    interstate edges. Transients it promotes itself (later, in its
    storage step) never make that list and end up read on the host.
    Promote them here so the transformation sees them."""
    read_names = set()
    for e in g.all_interstate_edges():
        read_names |= {m.data for m in e.data.get_read_memlets(g.arrays)}
    for block in g.all_control_flow_blocks():
        if isinstance(block, AbstractControlFlowRegion):
            read_names |= {m.data for m in block.get_meta_read_memlets()}
    for name in sorted(read_names):
        desc = g.arrays[name]
        if isinstance(desc, data.Array) and desc.transient and name not in keep_host:
            desc.storage = dtypes.StorageType.GPU_Global


def inherit_binding_storage(g: SDFG, state, nsdfg_node):
    """Give each connector-bound inner descriptor the storage of its
    outer binding, so the recursive ``GPUTransformSDFG`` sees resident
    data and neither clones nor copies it."""
    for e in list(state.in_edges(nsdfg_node)) + list(state.out_edges(nsdfg_node)):
        if e.data is None or not e.data.data:
            continue
        conn = e.dst_conn if e.dst is nsdfg_node else e.src_conn
        if conn is None or conn not in nsdfg_node.sdfg.arrays:
            continue
        outer, inner = g.arrays[e.data.data], nsdfg_node.sdfg.arrays[conn]
        if isinstance(inner, data.Array) and isinstance(outer, data.Array):
            inner.storage = outer.storage


def revert_gpu_clones(g: SDFG, names, before: set):
    """Undo the ``gpu_<name>`` clone ``GPUTransformSDFG`` makes for
    every array it reads -- including the ones marked ``host_data``,
    which it clones anyway. Drops the clone's copy edges, points every
    remaining reference back at the host array, and removes the
    descriptor."""
    for name in names:
        clone = find_clone(g, name, before)
        if clone is None:
            continue
        for state in g.all_states():
            for e in list(state.edges()):
                if (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)
                        and {e.src.data, e.dst.data} == {name, clone}):
                    state.remove_edge(e)
                    for endpoint in (e.src, e.dst):
                        if state.degree(endpoint) == 0:
                            state.remove_node(endpoint)
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data == clone:
                    node.data = name
            for e in state.edges():
                if e.data is not None and e.data.data == clone:
                    e.data.data = name
        rename_in_interstate_edges(g, clone, name)
        del g.arrays[clone]


def find_clone(g: SDFG, name: str, before: set):
    """The descriptor ``GPUTransformSDFG`` added for ``name``:
    ``gpu_<name>``, or ``gpu_<name>_<k>`` when that collided."""
    exact = 'gpu_' + name
    if exact in g.arrays and exact not in before:
        return exact
    for cand in sorted(g.arrays):
        if cand in before or not cand.startswith(exact + '_'):
            continue
        if cand[len(exact) + 1:].isdigit():
            return cand
    return None


def rename_in_interstate_edges(g: SDFG, old: str, new: str):
    """Word-boundary string substitution on interstate assignments and
    conditions. Avoids ``sdfg.replace``, which routes values through
    sympy and collapses literals like ``-1.79e+308`` to ``-inf``."""
    pat = re.compile(r'(?<![\w.])' + re.escape(old) + r'(?![\w])')
    for e in g.all_interstate_edges():
        e.data.assignments = {
            (pat.sub(new, k) if isinstance(k, str) else k): (pat.sub(new, v) if isinstance(v, str) else v)
            for k, v in e.data.assignments.items()
        }
        if e.data.condition is not None:
            text = e.data.condition.as_string
            retext = pat.sub(new, text)
            if retext != text:
                from dace.properties import CodeBlock
                e.data.condition = CodeBlock(retext, e.data.condition.language)
    for block in g.all_control_flow_blocks():
        if isinstance(block, AbstractControlFlowRegion):
            block.replace_meta_accesses({old: new})


# -- copy-state syncs ------------------------------------------------------


def bracket_copy_states_with_syncs(sdfg: SDFG):
    """Relabel ``GPUTransformSDFG``'s copy states to the names stage 4
    keys its timer off, and sync stream 0 after each of them."""
    for state in sdfg.states():
        if state.label == sdfg.label + '_copyin':
            state.label = _COPY_IN_LABEL
        elif state.label == sdfg.label + '_copyout':
            state.label = _COPY_OUT_LABEL
    for label, sync_label in ((_COPY_IN_LABEL, '_sync_after_copy_in'), (_COPY_OUT_LABEL, '_sync_after_copy_out')):
        state = next((s for s in sdfg.states() if s.label == label), None)
        if state is None:
            continue
        add_stream_sync_tasklet(sdfg.add_state_after(state, label=sync_label))


def add_stream_sync_tasklet(state) -> None:
    """Parameterless Tasklet synchronizing the default stream, which is
    the only stream in play (``max_concurrent_streams = -1``: the
    context holds no streams and every launch takes ``nullptr``). The
    ``gpu*`` aliases in ``dace/runtime/include/dace/cuda/
    cudacommon.cuh`` expand to ``cuda*`` for NVIDIA and ``hip*`` for
    AMD. ``side_effects=True`` keeps DaCe from eliminating a tasklet
    with no inputs or outputs."""
    state.add_tasklet(
        name='_stream0_sync',
        inputs={},
        outputs={},
        code='DACE_GPU_CHECK(gpuStreamSynchronize(nullptr));',
        language=dtypes.Language.CPP,
        side_effects=True,
    )


# -- helpers --------------------------------------------------------------


def all_sdfgs(sdfg: SDFG):
    yield sdfg
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.NestedSDFG):
            yield node.sdfg
