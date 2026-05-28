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

# When a Fortran kernel passes a runtime user communicator to MPI calls,
# :func:`emit_mpi` installs a :class:`FortranProcessGrid` on the SDFG --
# its ``init_code`` builds an ``MPI_Cart_create``-cartesian sub-comm out
# of the user's MPI_Comm at ``__dace_init`` time.  The pgrid + the
# symbols that drive it live under fixed names so the bindings layer
# and downstream tests can find them without grepping ``sdfg.symbols``.
_USER_COMM_SYMBOL = "dace_user_comm"        # opaque(MPI_Comm) symbol -- f2c result from the wrapper
_USER_COMM_SIZE_SYMBOL = "dace_user_comm_size"  # int -- MPI_Comm_size(dace_user_comm), 1-D pgrid extent
_USER_PGRID_NAME = "dace_user_pgrid"        # FortranProcessGrid descriptor name

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
    # A Fortran whole-array assignment conforms, so source and destination
    # hold the same number of elements per dim.  An allocatable's transient,
    # however, carries its own ALLOCATE extent symbol (e.g. ``x_alloc1_d0``)
    # distinct from the source's (``n``), and CopyLibraryNode's same-rank
    # expansion can't prove the two symbolic shapes equal.  Drive the
    # destination memlet off the SOURCE descriptor's shape when the two
    # differ symbolically (same rank) so both subsets align; conformance
    # guarantees the destination subset stays in bounds.
    same_rank_diff_shape = (len(src_desc.shape) == len(tgt_desc.shape) and list(src_desc.shape) != list(tgt_desc.shape))
    tgt_memlet = (Memlet.from_array(tgt_name, src_desc) if same_rank_diff_shape else Memlet.from_array(
        tgt_name, tgt_desc))
    state.add_edge(src_access, None, cp, CopyLibraryNode.INPUT_CONNECTOR_NAME, Memlet.from_array(src_name, src_desc))
    state.add_edge(cp, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, tgt_access, None, tgt_memlet)


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
    state.add_edge(ms, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, tgt_access, None, Memlet.from_array(tgt_name, tgt_desc))

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


def _install_user_pgrid(ctx, comm_arg: str):
    """Install the :class:`FortranProcessGrid` (+ its driving symbols)
    on the SDFG if not already present, and remove the orphan Fortran
    ``comm`` integer scalar from ``sdfg.arrays``.

    After this call the SDFG has:
      * symbol ``__user_comm`` of dtype ``opaque(MPI_Comm)`` -- the C
        ``MPI_Comm`` passed by the bindings wrapper at ``__dace_init``.
      * symbol ``__user_comm_size`` of dtype ``int64`` -- the
        ``MPI_Comm_size`` of the user comm (used as the 1-D pgrid
        extent).
      * descriptor ``__user_pgrid`` (:class:`FortranProcessGrid`),
        whose ``init_code`` runs ``MPI_Cart_create(__user_comm, 1,
        [__user_comm_size], ...)``.

    The bindings layer is responsible for populating both symbols from
    ``MPI_Comm_f2c`` + ``MPI_Comm_size`` in the wrapper, and for
    *omitting* the ``comm`` integer from the program call.

    :param ctx: the build context (``ctx.sdfg`` is the target SDFG).
    :param comm_arg: name of the Fortran ``integer`` ``comm`` dummy in
        ``sdfg.arrays`` -- removed by this call.
    """
    import dace
    from dace_fortran.data import FortranProcessGrid

    sdfg = ctx.sdfg
    if _USER_PGRID_NAME in sdfg.arrays:
        # Already installed by an earlier MPI call in the same kernel.
        # Still need to remove the orphan ``comm`` scalar -- it might be
        # a distinct dummy in a later call (rare but easy to support).
        if comm_arg in sdfg.arrays:
            sdfg.arrays.pop(comm_arg, None)
        return

    sdfg.add_symbol(_USER_COMM_SYMBOL, dace.dtypes.opaque("MPI_Comm"))
    sdfg.add_symbol(_USER_COMM_SIZE_SYMBOL, dace.dtypes.int64)
    sdfg.add_datadesc(
        _USER_PGRID_NAME,
        FortranProcessGrid(
            name=_USER_PGRID_NAME,
            shape=[dace.symbol(_USER_COMM_SIZE_SYMBOL)],
            parent_comm_symbol=_USER_COMM_SYMBOL,
        ))
    sdfg.append_init_code(sdfg.arrays[_USER_PGRID_NAME].init_code())
    sdfg.append_exit_code(sdfg.arrays[_USER_PGRID_NAME].exit_code())
    # ``ProcessGrid``'s ``init_code`` references state fields
    # (``__state->__user_pgrid``, ``..._group``, ``..._rank``, etc.)
    # but DaCe codegen only emits those fields when an MPI
    # :class:`dace.libraries.mpi.Dummy` node declares them.  Place a
    # Dummy on the SDFG's start state with the field list mirroring
    # the stock ``add_pgrid`` pattern from
    # ``dace/frontend/python/replacements/mpi.py``.
    from dace.libraries.mpi import Dummy
    start_state = sdfg.start_state
    dummy = Dummy(_USER_PGRID_NAME, [
        f'MPI_Comm {_USER_PGRID_NAME};',
        f'MPI_Group {_USER_PGRID_NAME}_group;',
        f'int {_USER_PGRID_NAME}_coords[1];',
        f'int {_USER_PGRID_NAME}_dims[1];',
        f'int {_USER_PGRID_NAME}_rank;',
        f'int {_USER_PGRID_NAME}_size;',
        f'bool {_USER_PGRID_NAME}_valid;',
    ])
    start_state.add_node(dummy)
    wnode = start_state.add_write(_USER_PGRID_NAME)
    start_state.add_edge(dummy, None, wnode, None, Memlet())
    # Drop the original ``comm`` integer dummy from the SDFG signature
    # -- the MPI nodes now wire to ``__user_pgrid`` instead, and the
    # bindings wrapper routes the f2c'd MPI_Comm into ``__user_comm``
    # at init time, not into the kernel call.
    sdfg.arrays.pop(comm_arg, None)
    # Remember the Fortran integer dummy that originally held this
    # communicator -- the bindings wrapper needs the outer-dummy name
    # to call ``MPI_Comm_f2c`` on it.  ``__user_comm_size`` is
    # populated by an ``MPI_Comm_size`` call in the same wrapper
    # block; track it alongside.
    sdfg._fortran_user_comm_source = comm_arg


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
        node.in_connectors = {
            c: (dace.pointer(dace.dtypes.opaque("MPI_Request")) if c == '_request' else t)
            for c, t in node.in_connectors.items()
        }
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, rname),
                              node,
                              dst_conn='_request',
                              memlet=Memlet.simple(rname, "0:1", num_accesses=1))
        # status ignored (MPI_STATUS_IGNORE) -> write-only scratch.
        for conn in ('_stat_tag', '_stat_source'):
            sname = f'_mpistat{conn}_{builder.nid()}'
            ctx.sdfg.add_array(sname, [1], dace.int32, transient=True)
            state.add_memlet_path(node,
                                  acc(builder, state, sname),
                                  src_conn=conn,
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
        # Replace the opaque-MPI_Comm scalar wiring with a
        # ``FortranProcessGrid`` whose ``MPI_Cart_create`` parent is the
        # user-supplied communicator.  At ``__dace_init`` time the
        # bindings wrapper converts the Fortran integer comm to a C
        # ``MPI_Comm`` and passes it as the ``__user_comm`` symbol; the
        # pgrid's ``init_code`` then runs
        # ``MPI_Cart_create(__user_comm, ...)`` and the resulting
        # cartesian sub-comm lives in ``__state->__user_pgrid``.  Every
        # MPI library node wires ``_comm`` (matching the stock
        # Send/Recv connector contract) to an access node on
        # ``__user_pgrid`` -- the codegen substitutes the
        # ``__state->__user_pgrid`` reference, so the C tasklet line
        # ``MPI_Send(..., _comm)`` invokes on the cartesian comm.
        _install_user_pgrid(ctx, comm)

    def _wire_comm(node):
        """Add a ``_comm`` input connector + memlet wired to the
        ``FortranProcessGrid`` access node when a user communicator is
        present (no-op for default ``MPI_COMM_WORLD``)."""
        if comm is None:
            return
        node.add_in_connector('_comm', dace.dtypes.opaque("MPI_Comm"))
        state.add_memlet_path(
            acc(builder, state, _USER_PGRID_NAME), node,
            dst_conn='_comm',
            memlet=Memlet(data=_USER_PGRID_NAME, subset='0'))

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
        node.out_connectors = {
            c: (dace.pointer(dace.dtypes.opaque("MPI_Request")) if c == '_request' else t)
            for c, t in node.out_connectors.items()
        }
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, buffer), node, dst_conn='_buffer', memlet=buf_memlet)
        state.add_memlet_path(acc(builder, state, partner), node, dst_conn='_dest', memlet=partner_memlet)
        state.add_memlet_path(acc(builder, state, tag), node, dst_conn='_tag', memlet=tag_memlet)
        state.add_edge(node, '_request', acc(builder, state, rname), None, Memlet.simple(rname, "0:1", num_accesses=1))
        _wire_comm(node)
    elif n.callee == 'mpi_irecv':
        from dace.libraries.mpi.nodes.irecv import Irecv
        rname = _req_array(n.call_args[3])
        node = Irecv(f'_mpi_irecv_{builder.nid()}')
        node.out_connectors = {
            c: (bptr if c == '_buffer' else dace.pointer(dace.dtypes.opaque("MPI_Request")) if c == '_request' else t)
            for c, t in node.out_connectors.items()
        }
        state.add_node(node)
        state.add_memlet_path(acc(builder, state, partner), node, dst_conn='_src', memlet=partner_memlet)
        state.add_memlet_path(acc(builder, state, tag), node, dst_conn='_tag', memlet=tag_memlet)
        state.add_memlet_path(node, acc(builder, state, buffer), src_conn='_buffer', memlet=buf_memlet)
        state.add_edge(node, '_request', acc(builder, state, rname), None, Memlet.simple(rname, "0:1", num_accesses=1))
        _wire_comm(node)
    else:
        raise NotImplementedError(f"MPI op {n.callee!r} not supported")


def emit_io(builder, ctx, n, region):
    """Lower a recognised Fortran I/O statement (``kind == 'iocall'``) to a
    ``dace.libraries.fortran_io`` node.

    The C++ recognizer folds an ``open`` / ``read`` / ``write`` / ``close``
    region into one node: ``n.callee`` is ``'read'`` / ``'write'`` /
    ``'namelist_read'``, ``n.target`` is the literal filename (baked into the
    node -- DaCe cannot pass a string at runtime), ``n.expr`` is the namelist
    group name (namelist only), and ``n.call_args`` are the transferred
    array / scalar names (a namelist member's name is its variable's name).  A
    READ writes each item (output connectors ``_out_i``); a WRITE reads each
    item (input connectors ``_in_i``).  Whole-array memlets -- the Fortran
    statement transfers each item in full.
    """
    nodes_mod = importlib.import_module("dace.libraries.fortran_io.nodes")
    # Each I/O statement gets its own SDFG state so the side-effecting
    # statements run in program order.  Nodes within one state have no mutual
    # dependency, so DaCe is free to reorder them -- which would corrupt
    # ordered I/O (e.g. a write then a read of the same file).  Sequential
    # states are a strict order, giving each I/O call its own state.
    ctx.flush_and_ensure(builder, region)
    state = ctx.new_state(builder, region)
    items = list(n.call_args)

    if n.callee == "namelist_read":
        node = nodes_mod.NamelistRead(f"namelist_{builder.nid()}", filename=n.target, group=n.expr, members=items)
        state.add_node(node)
        for i, name in enumerate(items):
            state.add_edge(node, f"_out_{i}", acc(builder, state, name), None,
                           Memlet.from_array(name, ctx.sdfg.arrays[name]))
        return

    is_read = n.callee == "read"
    cls = nodes_mod.Read if is_read else nodes_mod.Write
    node = cls(f"{n.callee}_{builder.nid()}", filename=n.target, num_items=len(items))
    state.add_node(node)
    for i, name in enumerate(items):
        memlet = Memlet.from_array(name, ctx.sdfg.arrays[name])
        if is_read:
            state.add_edge(node, f"_out_{i}", acc(builder, state, name), None, memlet)
        else:
            state.add_edge(acc(builder, state, name), None, node, f"_in_{i}", memlet)


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

    # Normalise the bridge's callee name to the registry key the user
    # registered.  The C++ side may hand us an MLIR symbol (leading
    # ``@``) and flang's free-procedure mangling (``_QP<name>``); strip
    # both so ``register_external("foo", ...)`` matches a ``CALL foo``.
    # Module-procedure callees (``_QMmodP<name>``) are not stripped --
    # register those under their bare ``<name>`` if needed.
    callee = n.callee.lstrip('@')
    if callee.startswith('_QP'):
        callee = callee[3:]
    sig = lookup_external(callee)
    if sig is None:
        return  # not registered -> unchanged (kind="call" had no emitter)
    if sig.stub:
        return  # stubbed external -> body already stripped; drop the call

    from collections import defaultdict
    from math import prod

    names = list(n.call_args)
    groups = list(n.aos_marshal_groups)  # flat [start, count, ...]
    group_pairs = [(groups[i], groups[i + 1]) for i in range(0, len(groups), 2)]

    # Expand ``sig.args`` to a per-call-arg plan.  An ``aos`` signature arg was
    # split by ``hlfir-marshal-external-structs`` into the struct's member
    # call-args (the i-th ``aos`` arg <-> the i-th ``[start, count]`` group);
    # every other arg maps to one call-arg.  ``plan[i]`` = ``(kind, dtype,
    # intent, gid)`` parallel to ``names`` (``gid`` is the group index for an
    # aos member, else ``None``).
    plan: list = []
    gi = 0
    for a in sig.args:
        if a.kind == 'aos':
            if gi >= len(group_pairs):
                # ``hlfir-marshal-external-structs`` only tags structs whose
                # every member is inline-flat (scalar or static-shape array
                # of scalar) -- the strict v1 ``isInlineFlatMember`` check
                # (see ``MarshalExternalStructs.cpp``).  Box-typed
                # (``allocatable`` / ``pointer``) members, nested derived
                # types, and dynamic-shape members are the v2 boundary and
                # the marshal pass silently skips them.  When that happens
                # the call site keeps the whole-struct operand; the
                # ``[start,count,...]`` tag the bridge would copy is absent;
                # and this lookup fails.  Two ways forward for such a
                # callee: (a) drop ``keep_external`` and use
                # :func:`dace_fortran.external.inline_external` (the SDFG
                # inlines callee's body, bypassing AoS marshalling
                # entirely); (b) wait for the v2 marshal expansion that
                # supports non-inline-flat members.
                raise ValueError(
                    f"external {callee!r}: 'aos' arg #{gi} has no "
                    f"marshalling group.  Most likely "
                    f"hlfir-marshal-external-structs skipped this callee "
                    f"because its struct has non-inline-flat members "
                    f"(allocatable / pointer arrays, nested derived types, "
                    f"dynamic shape).  Workarounds: (1) use "
                    f"dace_fortran.external.inline_external to fold the "
                    f"callee's SDFG into the caller (no marshalling "
                    f"needed); (2) restructure the callee to take only "
                    f"inline-flat members; (3) wait for the v2 permissive "
                    f"marshal expansion (Phase 2.3.E).")
            _, count = group_pairs[gi]
            for _ in range(count):
                plan.append(('aos', a.dtype, a.intent, gi))
            gi += 1
        else:
            plan.append((a.kind, a.dtype, a.intent, None))
    if len(plan) != len(names):
        raise ValueError(f"external {callee!r}: expanded signature expects "
                         f"{len(plan)} argument(s) but the call site passed "
                         f"{len(names)}")

    state = ctx.flush_and_ensure(builder, region)

    # Build the library-node connectors per call-arg.  array / aos-member =
    # pointer (distinct ``_aI`` / ``_aI_o`` names; both memlet the same array
    # so codegen aliases them); scalar = by-value ``_aI``; MPI communicator =
    # by-value ``opaque(MPI_Comm)``; a shape-only free symbol has no container
    # and is referenced by name.  ``logical_terms`` is one entry per *signature*
    # arg ((``'lit'``, term) or (``'aos'``, gid)); an aos group becomes one
    # ``(void*)&buffer`` C argument.
    in_conns: list = []
    out_conns: list = []
    ptr_of: dict = {}
    comm_conns: set = set()
    edges: list = []  # (name, conn, direction)  direction: 'r' | 'w'
    logical_terms: list = []
    # gid -> [(ctype, n_elems, in_conn|None, out_conn|None)]; n_elems == 1 for a
    # scalar member, else the member array's total element count.
    group_members: dict = defaultdict(list)
    prev_gid = None
    for i, (kind, dtype, intent, gid) in enumerate(plan):
        name = names[i]
        if gid is not None:
            desc = ctx.sdfg.arrays.get(name)
            if desc is None:
                raise ValueError(f"external {callee!r}: aos member {name!r} is not "
                                 f"an SDFG array")
            dt = desc.dtype
            ctype = dt.ctype  # the member's concrete C scalar type (e.g. "double")
            # ``int(prod(shape))`` fails for symbolic shapes -- ICON's
            # dynamic-extent box members render their shape as symbol
            # products (``s_w_d0 * s_w_d1 * s_w_d2``).  The AoS pack/unpack
            # body for such a member is not statically computable; the
            # node is destined for ``inline_external`` rewrite before
            # codegen so the body is never emitted, but ``emit_call``
            # still needs a syntactically valid placeholder.  ``nel == 0``
            # signals "skip pack/unpack" -- handled by the body lines
            # below.
            shape = getattr(desc, "shape", None)
            if shape:
                try:
                    nel = int(prod(shape))
                except TypeError:
                    nel = 0
            else:
                nel = 1
            reads = intent in ('in', 'inout')
            writes = intent in ('out', 'inout')
            cin, cout = f"_a{i}", f"_a{i}_o"
            if reads:
                in_conns.append(cin); ptr_of[cin] = dt; edges.append((name, cin, 'r'))
            if writes:
                out_conns.append(cout); ptr_of[cout] = dt; edges.append((name, cout, 'w'))
            group_members[gid].append((ctype, nel, cin if reads else None,
                                       cout if writes else None))
            if gid != prev_gid:
                logical_terms.append(('aos', gid))
            prev_gid = gid
            continue
        prev_gid = None
        if name not in ctx.sdfg.arrays:
            logical_terms.append(('lit', name))  # free symbol -- in scope
            continue
        if kind == 'comm':
            ctx.sdfg.arrays[name].dtype = dace.dtypes.opaque("MPI_Comm")
            cin = f"_a{i}"
            in_conns.append(cin); comm_conns.add(cin); edges.append((name, cin, 'r'))
            logical_terms.append(('lit', cin))
            continue
        dt = ctx.sdfg.arrays[name].dtype
        if kind == 'array':
            reads = intent in ('in', 'inout')
            writes = intent in ('out', 'inout')
            cin, cout = f"_a{i}", f"_a{i}_o"
            if reads:
                in_conns.append(cin); ptr_of[cin] = dt; edges.append((name, cin, 'r'))
            if writes:
                out_conns.append(cout); ptr_of[cout] = dt; edges.append((name, cout, 'w'))
            logical_terms.append(('lit', cout if writes else cin))
        else:  # 'scalar'
            cin = f"_a{i}"
            in_conns.append(cin); edges.append((name, cin, 'r'))
            logical_terms.append(('lit', cin))

    # Assemble the C body.  For an aos group, declare a local AoS buffer, pack
    # its pointer members in (in/inout), pass ``&buffer``, then unpack out
    # (out/inout) -- so the SoA<->AoS cast lives entirely in this tasklet.
    body_lines: list = []
    call_args_c: list = []
    bufname: dict = {}
    for kind, val in logical_terms:
        if kind == 'lit':
            call_args_c.append(val)
            continue
        gid = val
        mems = group_members[gid]
        buf = f"_aosbuf{gid}"; bufname[gid] = buf
        # One struct field per member: ``T mK;`` (scalar) or ``T mK[N];``
        # (array).  The field layout mirrors the Fortran derived type, so the
        # external's AoS pointer addresses the same contiguous bytes.
        # ``nel == 0`` is the dynamic-shape sentinel: AoS pack/unpack for
        # that member is statically unrenderable and the node is destined
        # for ``inline_external`` rewrite before codegen.  Render the field
        # as a pointer placeholder so the surrounding struct stays well-
        # formed and skip the pack/unpack lines below.
        fields = " ".join(
            (f"{ct} m{k};" if nel == 1 else
             f"{ct}* m{k};" if nel == 0 else
             f"{ct} m{k}[{nel}];")
            for k, (ct, nel, _, _) in enumerate(mems))
        body_lines.append(f"struct {{ {fields} }} {buf};")
        for k, (ct, nel, cin, cout) in enumerate(mems):
            if cin is None or nel == 0:
                continue
            if nel == 1:
                body_lines.append(f"{buf}.m{k} = (*{cin});")
            else:
                body_lines.append(f"for (int _i = 0; _i < {nel}; ++_i) "
                                  f"{buf}.m{k}[_i] = {cin}[_i];")
        call_args_c.append(f"(void*)(&{buf})")
    body_lines.append(f"{sig.c_name}({', '.join(call_args_c)});")
    for gid, mems in group_members.items():
        for k, (ct, nel, cin, cout) in enumerate(mems):
            if cout is None or nel == 0:
                continue
            if nel == 1:
                body_lines.append(f"(*{cout}) = {bufname[gid]}.m{k};")
            else:
                body_lines.append(f"for (int _i = 0; _i < {nel}; ++_i) "
                                  f"{cout}[_i] = {bufname[gid]}.m{k}[_i];")

    node = ExternalCall(name=f"_ext_{callee}_{builder.nid()}",
                        c_name=sig.c_name,
                        c_decl=sig.c_declaration(),
                        body="\n".join(body_lines),
                        inputs=in_conns,
                        outputs=out_conns)
    state.add_node(node)

    for name, conn, direction in edges:
        if conn in comm_conns:
            # Comm: by-value opaque scalar (subset '0', single element).
            mem = Memlet(data=name, subset='0')
        else:
            mem = Memlet.from_array(name, ctx.sdfg.arrays[name])
        if direction == 'r':
            state.add_memlet_path(state.add_read(name), node, dst_conn=conn, memlet=mem)
        else:
            state.add_memlet_path(node, state.add_write(name), src_conn=conn, memlet=mem)

    # Array connectors carry a pointer; data scalars stay by-value;
    # ``comm`` connectors carry ``opaque(MPI_Comm)`` by value (matches
    # the C ``MPI_Comm`` parameter type the shim declares).
    def _retype_in(c, d):
        if c in ptr_of:
            return dace.pointer(ptr_of[c])
        if c in comm_conns:
            return dace.dtypes.opaque("MPI_Comm")
        return d

    node.in_connectors = {c: _retype_in(c, d) for c, d in node.in_connectors.items()}
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
