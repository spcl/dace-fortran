"""Replace ``dace.libraries.standard.Reduce`` nodes with Tasklets that
call the velocity pipeline's own reduction helpers.

Runs once, at the end of ``OffloadVelocityToGPU`` (stage 4), when map
schedules and array storages are final. For every Reduce node the
pass picks exactly one backend and emits a Tasklet that calls that
backend's helper -- no runtime preprocessor dispatch:

    * **device** (inline ``__device__``) -- the Reduce sits inside a
      GPU kernel scope. Decided by walking the scope dict and any
      enclosing NestedSDFGs until a ``GPU_Device`` / ``GPU_ThreadBlock``
      MapEntry shows up.
    * **gpu** (host-launched CUB via ``stream``) -- the Reduce is at
      host level but its input memlet's array lives on GPU storage
      (``GPU_Global`` / ``GPU_Shared``).
    * **cpu** -- input is on CPU storage and the Reduce is host-level.

The pass also injects the lifecycle pair ``reduce_gpu_init`` /
``reduce_gpu_finalize`` into ``sdfg.init_code['cuda']`` /
``sdfg.exit_code['cuda']`` whenever at least one ``gpu`` tasklet was
emitted, so the SDFG's ``__dace_init_cuda_*`` and ``__dace_exit_cuda_*``
warm / release the scratch buffers at load / unload.

Headers are injected via ``global_code``:
    * ``reductions_cpu.h``    -> host TUs (always needed if a cpu or
                                 gpu tasklet exists; carries the
                                 prototypes that stay host-visible).
    * ``reductions_kernel.cuh`` -> when any ``gpu`` tasklet exists.
    * ``reductions_device.cuh`` -> when any ``device`` tasklet exists.
"""
import re

import dace
from dace import SDFG, dtypes, memlet as mm
from dace.sdfg import nodes
from dace.libraries.standard import Reduce


_MAX_WCR_PAT = re.compile(r"max\s*\(\s*a\s*,\s*b\s*\)|max\s*\(\s*b\s*,\s*a\s*\)")
_OR_WCR_PAT = re.compile(r"a\s*\|\s*b|b\s*\|\s*a")
_SUM_WCR_PAT = re.compile(r"a\s*\+\s*b|b\s*\+\s*a")

_GPU_STORAGES = (
    dtypes.StorageType.GPU_Global,
    dtypes.StorageType.GPU_Shared,
)

_GPU_SCHEDULES = (
    dtypes.ScheduleType.GPU_Device,
    dtypes.ScheduleType.GPU_ThreadBlock,
    dtypes.ScheduleType.GPU_ThreadBlock_Dynamic,
)


def replace_reductions_with_tasklets(sdfg: SDFG) -> int:
    """Rewrite every Reduce in ``sdfg`` (and its nested SDFGs) as a
    Tasklet calling the backend helper. Returns the count rewritten.

    For top-level GPU reductions whose result is a host-readable
    scalar, the emitted tasklet uses the ``*_async_host_gpu`` helper
    (stream-enqueued, no host stall). A single deferred
    ``gpuStreamSync`` tasklet is added in the earliest downstream
    state that reads the scalar, so any GPU work enqueued between
    the reduction and the consumer overlaps with the reduction's
    D2H copy on stream 0.
    """
    backends_used = set()
    # Collected async-output scalars -> (containing_sdfg, data_name).
    async_scalars = []
    count = 0
    for g in _all_sdfgs(sdfg):
        for state in list(g.all_states()):
            for node in list(state.nodes()):
                if isinstance(node, Reduce):
                    backend, async_out = _rewrite_one(state, node)
                    backends_used.add(backend)
                    if async_out is not None:
                        async_scalars.append((g, async_out))
                    count += 1
    if count == 0:
        return 0

    # Global include plumbing. ``reductions_cpu.h`` is always safe
    # (host-only prototypes, plain C linkage). ``reductions_kernel.cuh``
    # pulls in ``cuda_runtime.h`` so it belongs in any TU that compiles
    # through nvcc/hipcc -- which in the stage-4 pipeline is everything
    # (DaCe's .cpp goes through nvcc with ``-x cu``). ``reductions_device.cuh``
    # uses ``__device__`` and is only valid inside the .cu.
    _inject_global_cpp(sdfg, 'frame', '#include "reductions_cpu.h"\n')
    if 'gpu' in backends_used:
        # The kernel.cuh prototypes are called from both the host
        # ``.cpp`` (kernel launches) and the ``.cu`` (init / exit code
        # that the pass also emits below). Both TUs need the include;
        # 'frame' covers the .cpp, 'cuda' covers the .cu.
        for loc in ('frame', 'cuda'):
            _inject_global_cpp(sdfg, loc, '#include "reductions_kernel.cuh"\n')
    if 'device' in backends_used:
        _inject_global_cpp(sdfg, 'cuda',
                           '#include "reductions_device.cuh"\n')

    # Lifecycle hooks -- only relevant if a host-launched GPU tasklet
    # exists. Emitted into the CUDA-target init/exit so the SDFG's
    # ``__dace_init_cuda_*`` warms the scratch caches (currently a
    # no-op, see ``reductions_kernel.cu``) and ``__dace_exit_cuda_*``
    # frees them.
    if 'gpu' in backends_used:
        _inject_init_cpp(sdfg, 'cuda', 'reduce_gpu_init();\n')
        _inject_exit_cpp(sdfg, 'cuda', 'reduce_gpu_finalize();\n')

    # Deferred sync for every async scalar output. Done last so every
    # consumer state is reachable from the reduction state by the
    # time we walk.
    for g, scalar_name in async_scalars:
        _insert_sync_before_first_consumer(g, scalar_name)

    return count


def _rewrite_one(state, red: Reduce):
    """Rewrite a single Reduce into a tasklet. Returns
    ``(backend, async_scalar_name_or_None)``. ``async_scalar_name`` is
    set when the emitted call is the non-blocking ``*_async_host_gpu``
    variant -- the caller then schedules a deferred sync tasklet
    before the first consumer state of that scalar."""
    g: SDFG = state.sdfg if hasattr(state, 'sdfg') else state.parent
    in_e, = state.in_edges(red)
    out_e, = state.out_edges(red)

    in_desc = g.arrays[in_e.data.data]
    out_desc = g.arrays[out_e.data.data]

    family = _family_from_wcr(red.wcr)
    shape_kind = _out_shape_kind(out_e.data, out_desc)
    backend = _pick_backend(state, red, in_desc)
    size_expr = _size_expression(in_e.data)

    # Pick async variant when: (1) backend is ``gpu`` (host-launched
    # from the SDFG, so the sync point is hoistable), and (2) the
    # output is a scalar that downstream consumers read on the host
    # (to_scalar case). The ``to_address`` case already writes to a
    # GPU-resident slot and is naturally async.
    use_async = (backend == 'gpu' and shape_kind == 'to_scalar'
                 and family in ('max', 'sum', 'any'))

    if backend == 'gpu':
        # Default stream: DaCe is pinned to max_concurrent_streams=-1,
        # so no context streams exist and every launch takes nullptr.
        stream_arg = ', nullptr'
    else:
        stream_arg = ''

    if use_async:
        fn = f'reduce_{family}_async_host_gpu'
        # Async variant takes ``&out`` (writes via async memcpy) and
        # returns void. ``side_effects=True`` keeps DaCe from
        # reordering the stream op against later kernels.
        code = f'{fn}(&in_arr[0], &out, {size_expr}{stream_arg});'
        async_scalar = out_e.data.data
    else:
        fn = _function_name(family, shape_kind, backend)
        if shape_kind == 'to_scalar':
            code = f'out = {fn}(&in_arr[0], {size_expr}{stream_arg});'
        else:
            # ``to_address`` outputs: DaCe wires the connector as a
            # pointer bound at ``<base> + offset``, so pass ``out``
            # directly (not ``&out``).
            code = f'{fn}(&in_arr[0], out, {size_expr}{stream_arg});'
        async_scalar = None

    tasklet = state.add_tasklet(
        name=f'{family}_{shape_kind}_{backend}',
        inputs={'in_arr': dtypes.pointer(in_desc.dtype)},
        outputs={'out': (dtypes.pointer(out_desc.dtype)
                         if shape_kind == 'to_address'
                         else out_desc.dtype)},
        code=code,
        language=dtypes.Language.CPP,
        side_effects=True if use_async else None,
    )

    state.add_edge(in_e.src, in_e.src_conn, tasklet, 'in_arr', in_e.data)
    state.add_edge(tasklet, 'out', out_e.dst, out_e.dst_conn, out_e.data)
    state.remove_edge(in_e)
    state.remove_edge(out_e)
    state.remove_node(red)
    return backend, async_scalar


def _function_name(family: str, shape_kind: str, backend: str) -> str:
    """Resolve ``{family, shape_kind, backend}`` to the actual helper
    function name exposed by the ``reductions_*.{h,cuh}`` surface.

    * ``any`` only has a scalar-return form.
    * ``max`` / ``sum`` expose both ``<family>_<backend>`` (scalar
      return) and ``<family>_store_<backend>`` (store-to-pointer).
    """
    if family == 'any':
        return f'reduce_any_{backend}'
    if shape_kind == 'to_scalar':
        return f'reduce_{family}_{backend}'
    return f'reduce_{family}_store_{backend}'


def _pick_backend(state, red: Reduce, in_desc) -> str:
    if _inside_gpu_kernel(state, red):
        return 'device'
    if getattr(in_desc, 'storage', None) in _GPU_STORAGES:
        return 'gpu'
    return 'cpu'


def _inside_gpu_kernel(state, node) -> bool:
    """True iff any ancestor MapEntry -- within this state's SDFG or
    any outer SDFG reached by climbing parent NSDFGs -- has a GPU
    schedule. Captures the case where a Reduce at the top of a nested
    SDFG is actually emitted inside a kernel because the parent SDFG
    wrapped the NestedSDFG in a ``GPU_Device`` map."""
    current_sdfg = state.sdfg if hasattr(state, 'sdfg') else state.parent
    current_state = state
    current_node = node
    while current_sdfg is not None:
        sdict = current_state.scope_dict()
        anc = sdict.get(current_node)
        while anc is not None:
            if isinstance(anc, nodes.MapEntry) and anc.map.schedule in _GPU_SCHEDULES:
                return True
            anc = sdict.get(anc)
        parent_nsdfg = getattr(current_sdfg, 'parent_nsdfg_node', None)
        parent_state = getattr(current_sdfg, 'parent', None)
        if parent_nsdfg is None or parent_state is None:
            return False
        current_node = parent_nsdfg
        current_state = parent_state
        current_sdfg = (parent_state.sdfg if hasattr(parent_state, 'sdfg')
                        else getattr(parent_state, 'parent', None))
    return False


def _family_from_wcr(wcr: str) -> str:
    """``max(a, b)`` → ``max``, ``a | b`` → ``any``, ``a + b`` → ``sum``.
    Accepts both argument orderings."""
    if wcr is None:
        raise ValueError('Reduce node has no wcr')
    if _MAX_WCR_PAT.search(wcr):
        return 'max'
    if _OR_WCR_PAT.search(wcr):
        return 'any'
    if _SUM_WCR_PAT.search(wcr):
        return 'sum'
    raise NotImplementedError(
        f'Unsupported reduction wcr: {wcr!r}. '
        f'Supported: max / bitwise-or (any) / sum.')


def _out_shape_kind(out_memlet: mm.Memlet, out_desc) -> str:
    """``to_scalar`` iff the target is a 1-element Scalar or shape-[1]
    Array; ``to_address`` if writing a single element into a larger
    array."""
    if isinstance(out_desc, dace.data.Scalar):
        return 'to_scalar'
    if getattr(out_desc, 'shape', None) is not None \
            and tuple(out_desc.shape) == (1,):
        return 'to_scalar'
    try:
        if int(out_memlet.volume) == 1:
            return 'to_address'
    except Exception:
        pass
    raise NotImplementedError(
        f'Cannot classify reduction output shape. memlet={out_memlet}, '
        f'descriptor shape={getattr(out_desc, "shape", "?")}')


def _size_expression(in_memlet: mm.Memlet) -> str:
    """Total element count of the input subset as a C int expression.
    ``cppunparse`` keeps DaCe symbols as plain identifiers so the
    emitted code can reference them directly."""
    from dace.codegen import cppunparse
    vol = in_memlet.volume
    try:
        return f'(int)({cppunparse.pyexpr2cpp(str(vol))})'
    except Exception:
        return f'(int)({vol})'


def _inject_global_cpp(sdfg: SDFG, location: str, cpp_code: str) -> None:
    """Append a CPP-language CodeBlock to ``sdfg.global_code[location]``.
    Set rather than append because the stock SDFG's ``global_code['frame']``
    is a Python-language empty block whose ``code`` field is a list;
    ``list += str`` would iterate the string into chars and break save."""
    _inject_cpp(sdfg.global_code, location, cpp_code)


def _inject_init_cpp(sdfg: SDFG, location: str, cpp_code: str) -> None:
    _inject_cpp(sdfg.init_code, location, cpp_code)


def _inject_exit_cpp(sdfg: SDFG, location: str, cpp_code: str) -> None:
    _inject_cpp(sdfg.exit_code, location, cpp_code)


def _inject_cpp(container, location: str, cpp_code: str) -> None:
    from dace.properties import CodeBlock
    existing = container.get(location)
    prior = existing.code if existing is not None \
        and isinstance(existing.code, str) else ''
    container[location] = CodeBlock(prior + cpp_code, dtypes.Language.CPP)


def _all_sdfgs(root: SDFG):
    yield root
    for n, _ in root.all_nodes_recursive():
        if isinstance(n, nodes.NestedSDFG):
            yield n.sdfg


def _insert_sync_before_first_consumer(g: SDFG, scalar_name: str) -> None:
    """Insert a dedicated sync state on every interstate edge that
    reads ``scalar_name`` (directly or via its AccessNode's register
    name). The sync tasklet issues ``gpuStreamSynchronize`` on
    stream 0 before the consumer state evaluates the assignment.

    The async-write happens on the stream; without this sync the
    consumer would read stale / uninitialised memory. Placed on the
    interstate edge rather than inside the producer state so any GPU
    work queued between the reduction tasklet and the interstate-
    edge exit can run concurrently with the D2H copy (on the same
    stream, so strictly serial execution, but the host-side launch
    pipeline keeps going -- that's the overlap win).
    """
    import re as _re
    patt = _re.compile(r'(?<![\w])' + _re.escape(scalar_name) + r'(?![\w])')

    # Also look for any array that transitively aliases the scalar
    # via ``<sym> = <scalar_name>`` assignments. In our pipeline this
    # surfaces as ``tmp_call_18 = _red_tmp_tmp_call_18`` -- so we
    # gate on both names.
    aliases = {scalar_name}
    for e in g.all_interstate_edges():
        for k, v in e.data.assignments.items():
            v_str = v.as_string if hasattr(v, 'as_string') else str(v)
            if patt.search(v_str):
                aliases.add(k)
    alias_patt = _re.compile(
        r'(?<![\w])(' + '|'.join(_re.escape(a) for a in aliases) + r')(?![\w])')

    def _references_any_alias(e) -> bool:
        for v in e.data.assignments.values():
            s = v.as_string if hasattr(v, 'as_string') else str(v)
            if alias_patt.search(s):
                return True
        if e.data.condition is not None and alias_patt.search(
                e.data.condition.as_string):
            return True
        return False

    # Collect source states of edges that reference the async scalar
    # (or any alias). Insert a sync state between that source and
    # each dst, preserving edge data.
    candidates = [e for e in g.all_interstate_edges()
                  if _references_any_alias(e)]
    seen_sources = set()
    for e in candidates:
        src = e.src
        # One sync state per producer is sufficient: if multiple
        # downstream edges read the alias, they all flow through the
        # same post-producer sync state.
        if id(src) in seen_sources:
            continue
        seen_sources.add(id(src))
        _insert_post_state_sync(g, src)


def _insert_post_state_sync(g: SDFG, producer_state) -> None:
    """Add a new state right after ``producer_state`` containing a
    ``gpuStreamSynchronize`` side-effect tasklet. Re-wires every
    out-edge of ``producer_state`` to pass through the new state
    first."""
    sync_state = g.add_state_after(producer_state,
                                   label='_reduce_async_sync')
    sync_state.add_tasklet(
        name='_reduce_async_sync',
        inputs={}, outputs={},
        code='DACE_GPU_CHECK(gpuStreamSynchronize(nullptr));',
        language=dtypes.Language.CPP,
        side_effects=True,
    )
