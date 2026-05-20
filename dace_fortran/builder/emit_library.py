"""Library-node + terminator emissions.

These are the shortest per-kind emitters: each one stamps a single DaCe
library node (CopyLibraryNode / MemsetLibraryNode / MatMul / Transpose /
Dot / Reduce) or control-flow terminator (BreakBlock / ReturnBlock) and
wires its memlets.

All share the same shape: flush pending scalars, ensure a state, add the
node, attach edges.  Kept together because they're structurally cousins
and none is big enough to earn its own file.
"""

import importlib
import math

from dace import InterstateEdge, Memlet

from dace_fortran.builder.access import acc, iter_view_dim_map

# Per-library-node connector conventions.  Kept here rather than on
# ``LibNodeIntrinsic`` because the names are a property of the DaCe
# library node, not of the Fortran intrinsic.  Each entry maps a
# bridge-side ``LibNodeIntrinsic`` callee tag to ``(input_conns, output_conn)``;
# library nodes with their own dedicated emitters (CopyLibraryNode,
# MemsetLibraryNode, MergeLibraryNode, CountLibraryNode) bypass this
# generic dispatch table and live in the per-emitter functions below.
_LIBCALL_CONNECTORS = {
    "MatMul": (("_a", "_b"), "_c"),
    "Dot": (("_x", "_y"), "_result"),
    "Transpose": (("_inp", ), "_out"),
    # Must mirror the node classes' *_CONNECTOR_NAME constants
    # (MergeLibraryNode.TRUE/FALSE/MASK/OUTPUT_CONNECTOR_NAME,
    # CountLibraryNode.INPUT/OUTPUT_CONNECTOR_NAME) -- restyled to the
    # ``copy_node`` / ``memset_node`` prefixed-constant convention.
    "MergeLibraryNode": (("_mrg_t", "_mrg_f", "_mrg_mask"), "_mrg_out"),
    "CountLibraryNode": (("_cnt_in", ), "_cnt_out"),
}


def _parse_reduce_identity(s: str):
    """Resolve a reduce-accumulator-identity string to its Python value.

    Replaces a prior ``eval`` (flagged as a code smell; its globals were
    hand-patched with ``inf=math.inf``).  Rather than a closed whitelist
    -- brittle if a new reduction emits a different literal -- this
    parses the literal forms the two producers can emit and any plain
    numeric: the bridge ``kRedTable`` (``bridge/ast/dispatch.cpp``:
    ``0``/``1``/``inf``/``-inf``/``False``/``True``; bare ``inf`` so the
    section-reduce path's cppunparse maps it to ``INFINITY``) and the
    Python ``REDUCTIONS`` registry (``math.inf`` / ``-math.inf``), plus
    any future int/float identity (``0.0``, ``1.0``, ...).  Unknown
    non-numeric tokens (e.g. a symbolic ``huge(x)``) raise loudly rather
    than silently mis-reducing.

    :param s: the identity string carried on the reduce ASTNode.
    :returns: ``bool`` / ``int`` / ``float`` accumulator identity.
    :raises NotImplementedError: on an unrecognised non-numeric token.
    """
    named = {
        'True': True,
        'False': False,
        'inf': math.inf,
        '-inf': -math.inf,
        'math.inf': math.inf,
        '-math.inf': -math.inf,
    }
    if s in named:
        return named[s]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        raise NotImplementedError(f"unsupported reduction identity {s!r}")


def emit_copy(builder, ctx, n, region):
    """Whole-array ``b = a`` -> ``CopyLibraryNode``, with memlets covering
    the full source / destination arrays.  Connector names come from the
    node class (``_cpy_in`` / ``_cpy_out`` in the current libnode) so this
    stays correct if the libnode renames them."""
    from dace.libraries.standard.nodes import CopyLibraryNode
    state = ctx.flush_and_ensure(builder, region)

    src_name = n.reduce_src  # buildCopyNode stored the source here
    tgt_name = n.target
    src_desc = ctx.sdfg.arrays[src_name]
    tgt_desc = ctx.sdfg.arrays[tgt_name]

    cp = CopyLibraryNode(name=f"copy_{tgt_name}_{builder.nid()}")
    state.add_node(cp)

    src_access = acc(builder, state, src_name)
    tgt_access = acc(builder, state, tgt_name)
    state.add_edge(src_access, None, cp, CopyLibraryNode.INPUT_CONNECTOR_NAME,
                   Memlet.from_array(src_name, src_desc))
    state.add_edge(cp, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, tgt_access, None,
                   Memlet.from_array(tgt_name, tgt_desc))


def emit_memset(builder, ctx, n, region):
    """Scalar-zero -> array fill -> ``MemsetLibraryNode`` with a single
    output memlet covering the destination.  The memset transitions
    to a fresh successor state so any later element write to the same
    array lands in a new state (and on a new access node) instead of
    racing with the array-wide write inside one state's DAG."""
    from dace.libraries.standard.nodes import MemsetLibraryNode
    state = ctx.flush_and_ensure(builder, region)

    tgt_name = n.target
    # Section-alias dummies route memset through the source array,
    # writing to the slab carved out by view_dim_map (surviving dims =
    # full range, scalar dims = point index).
    v_tgt = builder.arrays.get(tgt_name)
    if v_tgt is not None and getattr(v_tgt, 'role', '') == 'section_alias':
        src_name = v_tgt.view_source
        src_desc = ctx.sdfg.arrays[src_name]
        slab_parts = []
        for src_dim, slot, dummy_dim in iter_view_dim_map(v_tgt.view_dim_map):
            if dummy_dim is not None:
                slab_parts.append(f"0:{src_desc.shape[src_dim]}")
            else:
                slab_parts.append(f"({slot}) - 1")
        slab_subset = ", ".join(slab_parts)
        ms = MemsetLibraryNode(name=f"memset_{tgt_name}_{builder.nid()}")
        state.add_node(ms)
        tgt_access = acc(builder, state, tgt_name)  # redirects to src via resolve
        state.add_edge(ms, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, tgt_access, None,
                       Memlet(data=src_name, subset=slab_subset))
        ctx.new_state(builder, region)
        return

    tgt_desc = ctx.sdfg.arrays[tgt_name]

    ms = MemsetLibraryNode(name=f"memset_{tgt_name}_{builder.nid()}")
    state.add_node(ms)

    tgt_access = acc(builder, state, tgt_name)
    state.add_edge(ms, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, tgt_access, None,
                   Memlet.from_array(tgt_name, tgt_desc))

    # Force a state break so a subsequent element write doesn't share
    # the memset's access node.  Two incoming memlets on one access
    # node race in DaCe's dataflow DAG.
    ctx.new_state(builder, region)


def emit_libcall(builder, ctx, n, region):
    """``target = matmul(a, b)`` / ``transpose(a)`` / ``dot_product(x, y)``
    lowered to the matching DaCe library node.  ``MatMul`` specializes
    internally (GEMM / GEMV / Dot) based on operand ranks.
    """
    from dace_fortran.intrinsics import libnode_spec

    state = ctx.flush_and_ensure(builder, region)

    spec = libnode_spec(n.callee)
    if spec is None:
        raise RuntimeError(f"unregistered libnode intrinsic {n.callee!r}")
    mod = importlib.import_module(f"dace.libraries.{spec.module}.nodes")
    cls = getattr(mod, spec.node_cls)
    in_conns, out_conn = _LIBCALL_CONNECTORS[spec.node_cls]

    # ``Transpose`` needs an explicit ``dtype`` so its expansion can
    # produce the right element type; ``CountLibraryNode`` consumes its
    # Fortran-1-based ``dim`` from ``reduce_axes`` (set by the bridge's
    # ``buildLibCallNode`` when the source ``hlfir.count`` carries a dim
    # operand); every other library node picks types up from the
    # attached memlets.
    tgt_desc = ctx.sdfg.arrays[n.target]
    if spec.node_cls == "Transpose":
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dtype=tgt_desc.dtype)
    elif spec.node_cls == "CountLibraryNode":
        # Bridge stores the (0-based) reduce axis the same way it does
        # for whole-array vs per-dim Reduce nodes.  CountLibraryNode's
        # constructor wants Fortran 1-based, so convert back.
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else -1
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dim=dim)
    else:
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}")
    state.add_node(node)

    # ``call_arg_subsets`` is parallel to ``call_args``; an empty entry =
    # whole-array source, a non-empty entry = a DaCe-0-based subset like
    # ``"0:3"`` for ``dot_product(arg1(1:3), arg2(1:3))``.  Older bridge
    # builds may not populate the field; default to empty for each arg.
    arg_subsets = list(getattr(n, 'call_arg_subsets', None) or [])
    arg_subsets += [''] * (len(n.call_args) - len(arg_subsets))
    for conn, src, sub in zip(in_conns, n.call_args, arg_subsets):
        src_desc = ctx.sdfg.arrays[src]
        if sub:
            in_memlet = Memlet(f"{src}[{sub}]")
        else:
            in_memlet = Memlet.from_array(src, src_desc)
        state.add_edge(acc(builder, state, src), None, node, conn, in_memlet)

    # Element-designate destination (``res1(1) = dot_product(...)``):
    # the bridge populates ``n.accesses[0]`` with the per-dim write
    # index so the output memlet covers a single element instead of
    # the whole array (which would fail validation for scalar-output
    # libcalls like dot_product, count, ...).
    write_acc = next((ac for ac in n.accesses if ac.is_write), None)
    if write_acc is not None:
        from dace_fortran.builder.access import build_memlet_index
        ix = build_memlet_index(builder, n.target, write_acc, ctx.iter_map)
        out_memlet = Memlet(f"{n.target}[{ix}]")
    else:
        out_memlet = Memlet.from_array(n.target, tgt_desc)
    state.add_edge(node, out_conn, acc(builder, state, n.target), None, out_memlet)


def emit_mpi(builder, ctx, n, region):
    """Lower a recognised Fortran MPI point-to-point call
    (``kind == 'mpicall'``) to a ``dace.libraries.mpi`` library node.

    ``n.callee`` / ``n.call_args``:

    * ``mpi_send`` / ``mpi_recv``  -- ``[buffer, partner, tag]``
    * ``mpi_isend`` / ``mpi_irecv`` -- ``[buffer, partner, tag, request]``
    * ``mpi_wait``                 -- ``[request]``

    ``partner`` is the dest rank for (i)send, the source rank for
    (i)recv.  count is implicit in the buffer memlet, the MPI datatype
    is derived from the buffer descriptor.  The communicator is
    ``MPI_COMM_WORLD`` unless the C++ bridge appended a runtime/user
    communicator, in which case the Fortran ``integer`` handle is
    retyped to an ``opaque(MPI_Comm)`` SDFG input wired to the libnode's
    ``_comm`` connector (the c-binding wrapper does ``MPI_Comm_f2c``).
    The non-blocking request is threaded Isend/Irecv
    -> Wait through a synthesised transient ``_mpireq_<req>`` of
    ``opaque("MPI_Request")`` keyed by the Fortran request variable, so
    the dataflow edge enforces the completion ordering.  ``MPI_Wait``'s
    status fields are ignored (``MPI_STATUS_IGNORE``) -- wired to
    write-only scratch.  Mirrors ``dace/frontend/python/replacements/mpi.py``.

    Each MPI call is emitted into its **own fresh state** (chained by
    an interstate edge), so program order between side-effecting MPI
    nodes is enforced by state sequencing -- a state is a dataflow
    graph, so two MPI nodes with no connecting memlet placed in one
    state would be order-unspecified (reorder / deadlock risk).
    ``has_side_effects`` (True for every ``MPINode``) only prevents
    DCE, not reordering; the per-statement state is what orders them,
    matching DaCe's Python-frontend MPI lowering.

    :raises NotImplementedError: for an unsupported MPI op.
    """
    import dace

    # Fresh successor state per MPI call -> interstate-edge ordering.
    ctx.new_state(builder, region)
    state = ctx.cur

    # Belt-and-suspenders: a shared len-1 ``__mpi_order`` transient that
    # every MPI op's state reads *and* writes (via a tiny sequencing
    # tasklet -- the MPI library nodes have fixed connector sets, so the
    # token can't ride a node connector).  The RAW chain on
    # ``__mpi_order`` across the per-call states is an explicit data
    # dependency between the side-effecting MPI nodes, surviving even if
    # a later transform were to fuse states.  The value is irrelevant
    # (just incremented); only the read+write matters.
    _tok = "__mpi_order"
    if _tok not in ctx.sdfg.arrays:
        ctx.sdfg.add_array(_tok, [1], dace.int32, transient=True)
    _seq = state.add_tasklet(f"_mpi_seq_{builder.nid()}", {"_o_in"}, {"_o_out"}, "_o_out = _o_in + 1")
    state.add_edge(state.add_read(_tok), None, _seq, "_o_in", Memlet(f"{_tok}[0]"))
    state.add_edge(_seq, "_o_out", state.add_write(_tok), None, Memlet(f"{_tok}[0]"))

    def _req_array(req: str) -> str:
        """Ensure the per-request ``opaque(MPI_Request)`` transient
        exists; return its name (shared by the Isend/Irecv producer and
        the matching Wait consumer)."""
        name = f"_mpireq_{req}"
        if name not in ctx.sdfg.arrays:
            ctx.sdfg.add_array(name, [1], dace.dtypes.opaque("MPI_Request"), transient=True)
        return name

    if n.callee == 'mpi_wait':
        from dace.libraries.mpi.nodes.wait import Wait
        (req, ) = n.call_args
        rname = _req_array(req)
        node = Wait(f'_mpi_wait_{builder.nid()}')
        node.in_connectors = {c: (dace.pointer(dace.dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in node.in_connectors.items()}
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, rname), node, dst_conn='_request',
                              memlet=Memlet.simple(rname, "0:1", num_accesses=1))
        # status ignored (MPI_STATUS_IGNORE) -> write-only scratch.
        for conn in ('_stat_tag', '_stat_source'):
            sname = f'_mpistat{conn}_{builder.nid()}'
            ctx.sdfg.add_array(sname, [1], dace.int32, transient=True)
            state.add_memlet_path(node, acc(builder, state, sname), src_conn=conn,
                                  memlet=Memlet.simple(sname, "0:1", num_accesses=1))
        return

    buffer, partner, tag = n.call_args[0], n.call_args[1], n.call_args[2]
    bdesc = ctx.sdfg.arrays[buffer]
    bptr = dace.pointer(bdesc.dtype)
    partner_memlet = Memlet.from_array(partner, ctx.sdfg.arrays[partner])
    tag_memlet = Memlet.from_array(tag, ctx.sdfg.arrays[tag])
    buf_memlet = Memlet.from_array(buffer, bdesc)

    # Optional trailing user communicator (the C++ bridge appends it only
    # when non-default; default ``MPI_COMM_WORLD`` adds nothing).  Base
    # ``call_args`` length is 3 for send/recv, 4 for isend/irecv (the
    # request); one extra entry is the comm.
    _comm_base = 4 if n.callee in ('mpi_isend', 'mpi_irecv') else 3
    comm = n.call_args[_comm_base] if len(n.call_args) > _comm_base else None
    if comm is not None:
        # Retype the Fortran ``integer`` comm dummy to an
        # ``opaque(MPI_Comm)`` SDFG input; the c-binding wrapper does
        # ``MPI_Comm_f2c`` on the integer handle.  The opaque-dtype
        # exemption in the length-1<->scalar passes keeps it a by-value
        # ``Scalar`` (an ``MPI_Comm`` handle, not a pointer).
        ctx.sdfg.arrays[comm].dtype = dace.dtypes.opaque("MPI_Comm")

    def _wire_comm(node):
        """Add a ``_comm`` input connector + memlet when a user
        communicator is present (no-op for default WORLD)."""
        if comm is None:
            return
        node.add_in_connector('_comm', dace.dtypes.opaque("MPI_Comm"))
        state.add_memlet_path(acc(builder, state, comm), node, dst_conn='_comm',
                              memlet=Memlet(data=comm, subset='0'))

    if n.callee == 'mpi_send':
        from dace.libraries.mpi.nodes.send import Send
        node = Send(f'_mpi_send_{builder.nid()}')
        node.in_connectors = {c: (bptr if c == '_buffer' else t) for c, t in node.in_connectors.items()}
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, buffer), node, dst_conn='_buffer', memlet=buf_memlet)
        state.add_memlet_path(acc(builder, state, partner), node, dst_conn='_dest', memlet=partner_memlet)
        state.add_memlet_path(acc(builder, state, tag), node, dst_conn='_tag', memlet=tag_memlet)
        _wire_comm(node)
    elif n.callee == 'mpi_recv':
        from dace.libraries.mpi.nodes.recv import Recv
        node = Recv(f'_mpi_recv_{builder.nid()}')
        node.out_connectors = {c: (bptr if c == '_buffer' else t) for c, t in node.out_connectors.items()}
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, partner), node, dst_conn='_src', memlet=partner_memlet)
        state.add_memlet_path(acc(builder, state, tag), node, dst_conn='_tag', memlet=tag_memlet)
        state.add_memlet_path(node, acc(builder, state, buffer), src_conn='_buffer', memlet=buf_memlet)
        _wire_comm(node)
    elif n.callee == 'mpi_isend':
        from dace.libraries.mpi.nodes.isend import Isend
        rname = _req_array(n.call_args[3])
        node = Isend(f'_mpi_isend_{builder.nid()}')
        node.in_connectors = {c: (bptr if c == '_buffer' else t) for c, t in node.in_connectors.items()}
        node.out_connectors = {c: (dace.pointer(dace.dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in node.out_connectors.items()}
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, buffer), node, dst_conn='_buffer', memlet=buf_memlet)
        state.add_memlet_path(acc(builder, state, partner), node, dst_conn='_dest', memlet=partner_memlet)
        state.add_memlet_path(acc(builder, state, tag), node, dst_conn='_tag', memlet=tag_memlet)
        state.add_edge(node, '_request', acc(builder, state, rname), None,
                       Memlet.simple(rname, "0:1", num_accesses=1))
        _wire_comm(node)
    elif n.callee == 'mpi_irecv':
        from dace.libraries.mpi.nodes.irecv import Irecv
        rname = _req_array(n.call_args[3])
        node = Irecv(f'_mpi_irecv_{builder.nid()}')
        node.out_connectors = {c: (bptr if c == '_buffer' else dace.pointer(dace.dtypes.opaque("MPI_Request")) if c == '_request' else t) for c, t in node.out_connectors.items()}
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, partner), node, dst_conn='_src', memlet=partner_memlet)
        state.add_memlet_path(acc(builder, state, tag), node, dst_conn='_tag', memlet=tag_memlet)
        state.add_memlet_path(node, acc(builder, state, buffer), src_conn='_buffer', memlet=buf_memlet)
        state.add_edge(node, '_request', acc(builder, state, rname), None,
                       Memlet.simple(rname, "0:1", num_accesses=1))
        _wire_comm(node)
    else:
        raise NotImplementedError(f"MPI op {n.callee!r} not supported")


def emit_call(builder, ctx, n, region):
    """Lower a *registered* external ``bind(c)`` call to an
    :class:`dace_fortran.external.ExternalCall` library node.

    The callee is matched against
    ``dace_fortran.external.lookup_external``.  Unregistered callees
    are a **no-op** (preserves prior behaviour -- ``kind="call"`` had
    no emitter), so unrelated kernels are unaffected.

    For a registered callee the signature drives the library node's
    connectors and call body: array args are pointer connectors
    (read / written per ``intent``), scalar args by-value connectors,
    shape-only free symbols are referenced inline in the call body.
    The library node carries the ``extern "C"`` declaration; its
    expansion at code-gen time produces the side-effecting CPP
    tasklet.  Linking the separately-compiled ``.so`` happens via the
    scoped ``compiler.linker.args`` set by ``register_external``.

    :raises ValueError: registered arg count disagrees with the call.
    """
    import dace
    from dace_fortran.external import ExternalCall, lookup_external

    callee = n.callee.lstrip('@')
    if callee.startswith('_QP'):
        callee = callee[3:]
    sig = lookup_external(callee)
    if sig is None:
        return  # not registered -> unchanged (kind="call" had no emitter)

    names = list(n.call_args)
    if len(names) != len(sig.args):
        raise ValueError(f"external {callee!r}: signature declares {len(sig.args)} "
                         f"argument(s) but the call site passed {len(names)}")

    state = ctx.flush_and_ensure(builder, region)

    # Argument-order invariant: every per-arg list below is built in one
    # forward pass over ``sig.args`` (with the matching call-site name),
    # so the i-th element of each list corresponds to the i-th
    # signature parameter.  The C call body is ``c_name(terms[0], ...,
    # terms[n-1])`` -- the order the C compiler sees, the order
    # ``c_decl`` declares, and the order ``sig.args`` was registered in.
    # Nothing downstream re-orders ``terms``: connector names embed the
    # position via ``_a{i}`` and the LibraryNode is constructed with
    # explicit ordered ``inputs`` / ``outputs`` lists (not sets).
    #
    # Per arg, decide how it reaches the C call.  A data container ->
    # library-node connector(s): array = pointer (distinct in ``_aI`` /
    # out ``_aI_o`` names -- the expanded tasklet may not reuse one
    # name across in & out; both memlet the same array so codegen
    # aliases them), scalar = by-value ``_aI``.  A shape-only free
    # symbol (``n`` from ``a(n)``) has no container -> referenced
    # directly by name in the call body.
    in_conns: list = []
    out_conns: list = []
    ptr_of: dict = {}
    edges: list = []  # (name, conn, direction)  direction: 'r' | 'w'
    terms: list = []
    for i, (a, name) in enumerate(zip(sig.args, names)):
        if name not in ctx.sdfg.arrays:
            terms.append(name)  # free symbol -- in scope in the code
            continue
        dt = ctx.sdfg.arrays[name].dtype
        if a.kind == 'array':
            reads = a.intent in ('in', 'inout')
            writes = a.intent in ('out', 'inout')
            cin, cout = f"_a{i}", f"_a{i}_o"
            if reads:
                in_conns.append(cin); ptr_of[cin] = dt; edges.append((name, cin, 'r'))
            if writes:
                out_conns.append(cout); ptr_of[cout] = dt; edges.append((name, cout, 'w'))
            # The C call uses the writable pointer when it writes,
            # else the read pointer (both alias the same array).
            terms.append(cout if writes else cin)
        else:
            cin = f"_a{i}"
            in_conns.append(cin); edges.append((name, cin, 'r'))
            terms.append(cin)

    node = ExternalCall(name=f"_ext_{callee}_{builder.nid()}",
                        c_name=sig.c_name,
                        c_decl=sig.c_declaration(),
                        body=f"{sig.c_name}({', '.join(terms)});",
                        inputs=in_conns,
                        outputs=out_conns)
    state.add_node(node)

    for name, conn, direction in edges:
        mem = Memlet.from_array(name, ctx.sdfg.arrays[name])
        if direction == 'r':
            state.add_memlet_path(state.add_read(name), node, dst_conn=conn, memlet=mem)
        else:
            state.add_memlet_path(node, state.add_write(name), src_conn=conn, memlet=mem)

    # Array connectors carry a pointer; data scalars stay by-value.
    node.in_connectors = {c: (dace.pointer(ptr_of[c]) if c in ptr_of else d)
                          for c, d in node.in_connectors.items()}
    node.out_connectors = {c: dace.pointer(ptr_of[c]) for c, d in node.out_connectors.items()}


def emit_reduce(builder, ctx, n, region):
    """``target = sum(src)`` (and product / minval / maxval) lowered as a
    DaCe ``standard.Reduce`` library node via
    ``state.add_reduce(wcr, axes, identity)``.

    ``axes=None`` reduces all dimensions (whole-array scalar result); a
    non-empty ``reduce_axes`` list reduces along those dims only.

    When ``n.target_is_array`` is true and ``n.accesses[0]`` carries a
    write AccessInfo (LHS was ``res(i) = MINVAL(...)``), the output
    memlet covers only that element  --  otherwise multiple reductions
    in the same routine all write through the whole destination and
    the last one wins.
    """
    from dace_fortran.builder.access import build_memlet_index

    state = ctx.flush_and_ensure(builder, region)

    src_name = n.reduce_src
    src_desc = ctx.sdfg.arrays.get(src_name)
    if src_desc is None:
        raise RuntimeError(f"reduction source {src_name!r} not registered as SDFG data")
    axes = list(n.reduce_axes) if n.reduce_axes else None

    # DaCe's Reduce expects a value (or None) for ``identity``.  The
    # bridge emits the float-extreme identities as bare ``inf`` /
    # ``-inf`` (so the section-reduce init tasklet renders to a valid
    # ``INFINITY`` C++ literal); patch the eval namespace so this
    # whole-array path resolves them too.
    #
    # Fortran spec: ``MINVAL`` / ``MAXVAL`` on an empty array returns
    # ``HUGE(x)`` / ``-HUGE(x)`` (the dtype's representable extreme),
    # not ``+/-inf``.  Substitute the identity per destination dtype so
    # the empty-array case matches gfortran exactly and the integer
    # path doesn't truncate ``inf`` to a garbage int.
    import numpy as np
    tgt_desc = ctx.sdfg.arrays[n.target]
    identity_val = None
    if n.reduce_identity:
        identity_val = _parse_reduce_identity(n.reduce_identity)
        if identity_val in (math.inf, -math.inf):
            np_dt = tgt_desc.dtype.as_numpy_dtype()
            if np.issubdtype(np_dt, np.integer):
                info = np.iinfo(np_dt)
                identity_val = info.max if identity_val == math.inf else info.min
            elif np.issubdtype(np_dt, np.floating):
                info = np.finfo(np_dt)
                identity_val = float(info.max if identity_val == math.inf else info.min)

    red = state.add_reduce(n.reduce_wcr, axes, identity_val)

    src_access = acc(builder, state, src_name)
    tgt_access = acc(builder, state, n.target)
    state.add_edge(src_access, None, red, None, Memlet.from_array(src_name, src_desc))

    write_acc = next((ac for ac in n.accesses if ac.is_write), None) if n.accesses else None
    if n.target_is_array and write_acc is not None and write_acc.index_exprs:
        subset = build_memlet_index(builder, n.target, write_acc, iter_map={})
        out_memlet = Memlet(f"{n.target}[{subset}]")
    else:
        out_memlet = Memlet.from_array(n.target, tgt_desc)
    state.add_edge(red, None, tgt_access, None, out_memlet)


def _emit_terminator_block(builder, ctx, region, block_cls, prefix: str):
    """Add a leaf control-flow terminator (``BreakBlock`` /
    ``ReturnBlock``) to ``region``, wired from ``ctx.cur`` -- or marked
    the region's start block when the terminator is its first statement.

    :param block_cls: the DaCe terminator block class to instantiate.
    :param prefix: node-name prefix (``break`` / ``return``).
    """
    ctx.flush(builder, region)
    is_start = ctx.cur is None
    blk = block_cls(f"{prefix}_{builder.nid()}")
    region.add_node(blk, is_start_block=is_start)
    if ctx.cur is not None:
        region.add_edge(ctx.cur, blk, InterstateEdge())
    ctx.cur = blk


def emit_break(builder, ctx, n, region):
    """Fortran ``EXIT`` -> ``BreakBlock`` added to the current region.
    The block is a leaf and implicitly transfers control to the nearest
    enclosing loop's exit edge at codegen time.  When the break is the
    region's first block (a branch body whose only statement is
    ``exit``), it becomes the region's start block.
    """
    from dace.sdfg.state import BreakBlock
    _emit_terminator_block(builder, ctx, region, BreakBlock, "break")


def emit_return(builder, ctx, n, region):
    """Fortran ``RETURN`` -> ``ReturnBlock``.  Added to the current region
    so RETURNs nested inside a loop or conditional get placed correctly;
    codegen still emits a plain ``return`` that bails out of the whole
    subroutine.
    """
    from dace.sdfg.state import ReturnBlock
    _emit_terminator_block(builder, ctx, region, ReturnBlock, "return")
