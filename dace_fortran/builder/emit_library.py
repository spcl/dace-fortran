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

import dace.symbolic
from dace import InterstateEdge, Memlet

from dace_fortran.builder.access import acc, iter_view_dim_map

# When a Fortran kernel passes a runtime user communicator to MPI calls,
# :func:`emit_mpi` installs a :class:`FortranProcessGrid` on the SDFG --
# its ``init_code`` builds an ``MPI_Cart_create``-cartesian sub-comm out
# of the user's MPI_Comm at ``__dace_init`` time.  The pgrid + the
# symbols that drive it live under fixed names so the bindings layer
# and downstream tests can find them without grepping ``sdfg.symbols``.
_USER_COMM_SYMBOL = "dace_user_comm"  # opaque(MPI_Comm) symbol -- f2c result from the wrapper
_USER_COMM_SIZE_SYMBOL = "dace_user_comm_size"  # int -- MPI_Comm_size(dace_user_comm), 1-D pgrid extent
_USER_PGRID_NAME = "dace_user_pgrid"  # FortranProcessGrid descriptor name

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
    # Fortran MINLOC / MAXLOC -> ``ArgMin`` / ``ArgMax``.  Optional
    # ``_mask`` connector is wired only when the source
    # ``hlfir.minloc`` / ``hlfir.maxloc`` carries a mask operand;
    # ``emit_libcall`` adds it after the mandatory ``_x``.
    "ArgMin": (("_x", ), "_idx"),
    "ArgMax": (("_x", ), "_idx"),
    # Fortran CSHIFT / EOSHIFT -- single-array input + shift-via-symbol output.
    "CShift": (("_x", ), "_out"),
    "EOShift": (("_x", ), "_out"),
    # Fortran NORM2 -- single-array input, scalar output.
    "Norm2": (("_x", ), "_out"),
    # Fortran SPREAD -- single-array source, broadcasted destination.
    "Broadcast": (("_src", ), "_dst"),
}

# SDFG dtype -> C scalar type for ``extern "C"`` declarations on
# Fortran module BSS symbols forwarded via
# ``ExternalSignature.module_symbol_forward``.  Mirrors the
# bind_c_shim's ``_MOD_FORWARD_SCALAR_FTYPE`` so the two ABIs match
# byte-for-byte.
_MOD_FORWARD_CTYPE = {
    "int32": "int",
    "int64": "long long",
    "float32": "float",
    "float64": "double",
    "bool": "bool",
}


def _sym2c(s) -> str:
    """Render a symbolic shape entry as a C expression suitable for an
    ``(int)(...)`` cast inside an external-call body.

    The expression is evaluated in the surrounding kernel scope where
    every SDFG symbol is in scope; ``sym2cpp`` honours the connector /
    free-symbol naming the C++ tasklet expects."""
    from dace.codegen.common import sym2cpp
    return sym2cpp(s)


def _shape_is_symbolic(shape) -> bool:
    """True iff any shape entry isn't an integer literal -- the
    ``dynamic_extents_abi`` callee then needs a runtime extent per dim
    rather than a baked compile-time literal in its ``c_f_pointer``."""
    if not shape:
        return False
    for s in shape:
        try:
            int(s)
        except (TypeError, ValueError):
            return True
    return False


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
    import dace.dtypes as dtypes

    state = ctx.flush_and_ensure(builder, region)

    # ``hlfir.matmul_transpose`` -- ``C = MATMUL(TRANSPOSE(A), B)``.
    # Compose into a Transpose libcall (A -> A_T transient) plus a
    # MatMul libcall (A_T, B -> C) instead of adding a new lib node;
    # the existing Transpose + MatMul expansions cover every backend
    # (pure / OpenBLAS / MKL / cuBLAS) at the cost of one transient
    # copy.  Future fused-cuBLAS / fused-MKL path can swap in a
    # MatMul with ``transA=True`` once that flag is wired through
    # SpecializeMatMul.
    if n.callee == "matmul_transpose":
        from types import SimpleNamespace
        if len(n.call_args) != 2:
            raise RuntimeError(f"matmul_transpose: expected 2 operands, got {len(n.call_args)}")
        a_name, b_name = n.call_args
        a_desc = ctx.sdfg.arrays[a_name]
        if len(a_desc.shape) != 2:
            raise NotImplementedError("matmul_transpose: rank != 2 LHS not yet supported")
        at_shape = [a_desc.shape[1], a_desc.shape[0]]
        at_name = f"__matmul_t_{builder.nid()}"
        ctx.sdfg.add_transient(at_name, at_shape, a_desc.dtype)
        # Synthesise a Transpose libcall ASTNode and reuse the
        # generic path.  ``ASTNode`` is a read-only C++ binding so
        # use a duck-typed ``SimpleNamespace`` carrying the field set
        # the generic emit code reads (kind / callee / target /
        # target_is_array / call_args / call_arg_subsets / accesses /
        # reduce_axes / options).
        t_node = SimpleNamespace(
            kind="libcall",
            callee="transpose",
            target=at_name,
            target_is_array=True,
            call_args=[a_name],
            call_arg_subsets=[''],
            accesses=[],
            reduce_axes=[],
            options={},
        )
        emit_libcall(builder, ctx, t_node, region)
        # Then MatMul of (A_T, B) into the original target.
        m_node = SimpleNamespace(
            kind="libcall",
            callee="matmul",
            target=n.target,
            target_is_array=n.target_is_array,
            call_args=[at_name, b_name],
            call_arg_subsets=['', ''],
            accesses=list(n.accesses),
            reduce_axes=[],
            options={},
        )
        emit_libcall(builder, ctx, m_node, region)
        return

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
    has_mask = False
    if spec.node_cls == "Transpose":
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dtype=tgt_desc.dtype)
    elif spec.node_cls == "CountLibraryNode":
        # Bridge stores the (0-based) reduce axis the same way it does
        # for whole-array vs per-dim Reduce nodes.  CountLibraryNode's
        # constructor wants Fortran 1-based, so convert back.
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else -1
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dim=dim)
    elif spec.node_cls in ("ArgMin", "ArgMax"):
        # ``MINLOC`` / ``MAXLOC`` -- pull dim / back from the bridge
        # ASTNode.  The bridge stores ``dim`` 0-based in
        # ``reduce_axes`` (mirroring the Reduce path); ``back`` lives
        # in ``options`` (set by ``buildLibCallNode``'s minloc/maxloc
        # operand walker).  A mask= argument is signalled by an extra
        # ``call_args`` entry past the first ``_x`` source.
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else None
        back = bool((getattr(n, 'options', None) or {}).get('back', False))
        has_mask = len(n.call_args) > 1
        node = cls(
            f"{spec.name}_{n.target}_{builder.nid()}",
            one_based=True,
            back=back,
            dim=dim,
            mask=has_mask,
        )
    elif spec.node_cls in ("CShift", "EOShift"):
        # Fortran CSHIFT / EOSHIFT -- bridge stores the shift (and
        # optional boundary, for EOSHIFT) expressions in
        # ``options['shift']`` / ``options['boundary']`` as Python-
        # compatible strings; the optional axis lives in
        # ``reduce_axes`` 0-based.  Symbol-promotion for the shift
        # expression's free symbols runs after node creation so a
        # Fortran scalar INTENT(IN) arg the libcall references gets
        # an SDFG-level symbol.
        opts = getattr(n, 'options', None) or {}
        shift_expr = opts.get('shift', None)
        boundary_expr = opts.get('boundary', None)
        shift = dace.symbolic.pystr_to_symbolic(shift_expr) if shift_expr else None  # noqa: F405
        boundary = dace.symbolic.pystr_to_symbolic(boundary_expr) if boundary_expr else None  # noqa: F405
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else 1
        if spec.node_cls == "CShift":
            node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dim=dim, shift=shift)
        else:
            node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dim=dim, shift=shift, boundary=boundary)
        # Promote every free symbol the shift expression depends on
        # to an SDFG-level symbol -- the scalar-INTENT(IN) Fortran
        # arg might otherwise land as a Scalar array (e.g. when it
        # has no other memlet / tasklet use) and the lib node's
        # symbolic-property reference would not match up at arglist
        # time.
        import dace.dtypes as dtypes
        if shift is not None:
            for sym in shift.free_symbols:
                name = str(sym)
                if name in ctx.sdfg.symbols:
                    continue
                if name in ctx.sdfg.arrays:
                    sym_name = f"__{name}_sym"
                    if sym_name not in ctx.sdfg.symbols:
                        ctx.sdfg.add_symbol(sym_name, dtypes.int64)
                    nxt = region.add_state(f"pre_{spec.name}_{sym_name}")
                    region.add_edge(ctx.cur, nxt, InterstateEdge(assignments={sym_name: f"{name}[0]"}))
                    ctx.cur = nxt
                    state = ctx.flush_and_ensure(builder, region)
                    node.shift = node.shift.subs(sym, dace.symbolic.symbol(sym_name))  # noqa: F405
                else:
                    ctx.sdfg.add_symbol(name, dtypes.int64)
    elif spec.node_cls == "Norm2":
        # Fortran NORM2 -- optional 1-based dim from reduce_axes (empty
        # for whole-array scalar).
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else None
        node = cls(f"{spec.name}_{n.target}_{builder.nid()}", dim=dim)
    elif spec.node_cls == "Broadcast":
        # Fortran SPREAD -- bridge stores the inserted axis (Fortran
        # 1-based) in ``reduce_axes[0]`` (0-based).
        dim = (n.reduce_axes[0] + 1) if n.reduce_axes else 1
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
    # ArgMin / ArgMax with mask=True expose an extra ``_mask`` input
    # connector that ``_LIBCALL_CONNECTORS`` does NOT list (because
    # mask is optional).  Append it on the fly so the iteration below
    # binds the bridge's second positional arg to ``_mask``.
    effective_in_conns = list(in_conns)
    if has_mask and spec.node_cls in ("ArgMin", "ArgMax"):
        effective_in_conns.append("_mask")
    for conn, src, sub in zip(effective_in_conns, n.call_args, arg_subsets):
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

    if n.callee == 'mpi_alltoall':
        # ``call_args``: [sendbuf, recvbuf, optional comm].  The Alltoall
        # library node has fixed ``_inbuffer`` / ``_outbuffer`` connectors
        # and derives the count from the buffer memlets.
        from dace.libraries.mpi.nodes.alltoall import Alltoall
        sendbuf = n.call_args[0]
        recvbuf = n.call_args[1]
        node = Alltoall(f'_mpi_alltoall_{builder.nid()}')
        state.add_node(node)
        send_desc = ctx.sdfg.arrays[sendbuf]
        recv_desc = ctx.sdfg.arrays[recvbuf]
        state.add_edge(state.add_read(sendbuf), None, node, '_inbuffer', Memlet.from_array(sendbuf, send_desc))
        state.add_edge(node, '_outbuffer', state.add_write(recvbuf), None, Memlet.from_array(recvbuf, recv_desc))
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
        state.add_memlet_path(acc(builder, state, _USER_PGRID_NAME),
                              node,
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


def emit_fft_interpolate(builder, ctx, n, region):
    """Lower a recognised QE ``fft_interpolate_*`` call to an
    :class:`dace.libraries.fft.nodes.FFTInterpolate` lib node.

    ``n.callee`` is ``"real"`` or ``"complex"`` (the variant suffix).
    ``n.call_args`` is ``[v_in, v_out]`` -- the value arrays on the
    source and target grids.  The descriptor arguments (``dfft_in`` /
    ``dfft_out``) are ignored at recognition time and the lib node
    derives the rank / extents from the memlets at expansion.
    """
    import importlib

    fft_nodes = importlib.import_module("dace.libraries.fft.nodes")
    ctx.flush_and_ensure(builder, region)
    state = ctx.new_state(builder, region)

    vin, vout = n.call_args[0], n.call_args[1]
    node = fft_nodes.FFTInterpolate(f"fft_interpolate_{builder.nid()}", dtype_kind=n.callee)
    state.add_node(node)
    in_desc = ctx.sdfg.arrays[vin]
    out_desc = ctx.sdfg.arrays[vout]
    state.add_edge(state.add_read(vin), None, node, "_inp", Memlet.from_array(vin, in_desc))
    state.add_edge(node, "_out", state.add_write(vout), None, Memlet.from_array(vout, out_desc))


def emit_unsupported_libcall(builder, ctx, n, region):
    """Raise a clear ``NotImplementedError`` for a Fortran call site that
    matches a recognised library's call convention (MPI / FFTW3 / BLAS /
    LAPACK) but isn't in the bridge's supported subset yet.

    The C++ side's near-miss detector emits this ASTNode in place of the
    generic ``call`` fallback so the failure surfaces with the library
    family name + the canonical routine name instead of degrading to a
    silently-invalid ``_out = ?`` tasklet body.
    """
    family_help = {
        "mpi": "extend ``mpiCalleeTag`` in dispatch.cpp and add a handler in ``emit_mpi``",
        "fftw3": "extend ``fftw3CalleeTag`` and ``buildFftw3CallNode`` in dispatch.cpp",
        "blas": "extend ``blasCalleeTag`` and ``buildBlasCallNode`` in dispatch.cpp + add a handler to ``emit_blas``",
        "lapack": "extend ``lapackCalleeTag`` and ``buildLapackCallNode`` + extend ``emit_lapack``",
    }
    hint = family_help.get(n.expr, "extend the bridge's library recognition")
    raise NotImplementedError(f"Fortran call to {n.callee!r} matches the {n.expr.upper()} library convention "
                              f"but is not in the bridge's supported subset.  To add support: {hint}.")


def emit_blas(builder, ctx, n, region):
    """Lower a recognised Fortran BLAS call (``kind == 'blascall'``) to a
    :mod:`dace.libraries.blas` library node.

    Currently supported routines (real32 / real64; complex twins out of
    scope for the first wave):

    * ``daxpy`` / ``saxpy``     -- ``y := alpha*x + y``
    * ``dscal`` / ``sscal``     -- ``x := alpha*x``
    * ``dgemv`` / ``sgemv``     -- ``y := alpha*op(A)*x + beta*y``
    * ``dgemm`` / ``sgemm``     -- ``C := alpha*op(A)*op(B) + beta*C``

    ``ddot`` is special-cased on the C++ side and threads through the
    matching ``hlfir.assign`` site (not via this emitter).

    ``n.expr`` carries the character flag literals (TRANSA/TRANSB / etc.)
    when the routine has them; ``n.call_args`` carries the resolved
    operand decl-names in the order documented in
    :func:`buildBlasCallNode` (dispatch.cpp).
    """
    import importlib

    import dace
    import dace.symbolic as _ds

    blas_nodes = importlib.import_module("dace.libraries.blas.nodes")
    ctx.flush_and_ensure(builder, region)
    state = ctx.new_state(builder, region)
    routine = n.callee.lower()

    # Collect scalar promotions (e.g. alpha, beta arriving as ``REAL :: alpha``
    # length-1 arrays).  Stage them on the inbound interstate edge of the
    # BLAS state so the symbol is bound BEFORE the lib node executes.
    promotions: dict[str, str] = {}  # sym -> "array_name[0]"

    def _scalar(name):
        """Resolve a scalar literal / dummy to a value usable as a
        :class:`SymbolicProperty` on the BLAS lib node.
        """
        try:
            return float(name)
        except (TypeError, ValueError):
            pass
        if name in ctx.sdfg.symbols:
            return _ds.symbol(name)
        if name in ctx.sdfg.arrays:
            desc = ctx.sdfg.arrays[name]
            sym = f"__blas_{name}_{builder.nid()}"
            ctx.sdfg.add_symbol(sym, desc.dtype)
            promotions[sym] = f"{name}[0]"
            return _ds.symbol(sym)
        ctx.sdfg.add_symbol(name, dace.float64)
        return _ds.symbol(name)

    def _apply_promotions():
        """Stage any pending scalar promotions on the BLAS state's inbound edge."""
        if not promotions:
            return
        for in_edge in ctx.sdfg.in_edges(state):
            for sym, expr in promotions.items():
                in_edge.data.assignments[sym] = expr
            break

    if routine in ("daxpy", "saxpy"):
        alpha, x, y = n.call_args
        node = blas_nodes.Axpy(f"axpy_{builder.nid()}", a=_scalar(alpha))
        _apply_promotions()
        state.add_node(node)
        x_desc = ctx.sdfg.arrays[x]
        y_desc = ctx.sdfg.arrays[y]
        state.add_edge(state.add_read(x), None, node, "_x", Memlet.from_array(x, x_desc))
        state.add_edge(state.add_read(y), None, node, "_y", Memlet.from_array(y, y_desc))
        state.add_edge(node, "_res", state.add_write(y), None, Memlet.from_array(y, y_desc))
        return

    if routine in ("dscal", "sscal"):
        alpha, x = n.call_args
        node = blas_nodes.Scal(f"scal_{builder.nid()}", a=_scalar(alpha))
        _apply_promotions()
        state.add_node(node)
        x_desc = ctx.sdfg.arrays[x]
        state.add_edge(state.add_read(x), None, node, "_x", Memlet.from_array(x, x_desc))
        state.add_edge(node, "_res", state.add_write(x), None, Memlet.from_array(x, x_desc))
        return

    if routine in ("dgemv", "sgemv"):
        trans = n.expr.strip().strip("'\"").upper()[:1] or "N"
        alpha, A, x, beta, y = n.call_args
        node = blas_nodes.Gemv(f"gemv_{builder.nid()}", transA=(trans == "T"), alpha=_scalar(alpha), beta=_scalar(beta))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (x, "_x"), (y, "_y")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        y_desc = ctx.sdfg.arrays[y]
        state.add_edge(node, "_y", state.add_write(y), None, Memlet.from_array(y, y_desc))
        return

    if routine in ("dgemm", "sgemm"):
        tA, tB = (s.strip("'\"").upper()[:1] or "N" for s in n.expr.split(","))
        alpha, A, B, beta, C = n.call_args
        node = blas_nodes.Gemm(f"gemm_{builder.nid()}",
                               transA=(tA == "T"),
                               transB=(tB == "T"),
                               alpha=_scalar(alpha),
                               beta=_scalar(beta))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_a"), (B, "_b"), (C, "_c")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        c_desc = ctx.sdfg.arrays[C]
        state.add_edge(node, "_c", state.add_write(C), None, Memlet.from_array(C, c_desc))
        return

    # ----- new-extension BLAS L1/L2/L3 lib nodes -------------------------------

    def _wire_inplace_single(node_cls, x, **kwargs):
        """Lib nodes with ``_x`` in -> ``_res`` out (Scal-style) on a single array."""
        node = node_cls(f"{routine}_{builder.nid()}", **kwargs)
        _apply_promotions()
        state.add_node(node)
        x_desc = ctx.sdfg.arrays[x]
        state.add_edge(state.add_read(x), None, node, "_x", Memlet.from_array(x, x_desc))
        state.add_edge(node, "_res", state.add_write(x), None, Memlet.from_array(x, x_desc))

    if routine in ("dcopy", "scopy"):
        x, y = n.call_args
        node = blas_nodes.Copy(f"copy_{builder.nid()}")
        _apply_promotions()
        state.add_node(node)
        x_desc = ctx.sdfg.arrays[x]
        y_desc = ctx.sdfg.arrays[y]
        state.add_edge(state.add_read(x), None, node, "_x", Memlet.from_array(x, x_desc))
        state.add_edge(node, "_y", state.add_write(y), None, Memlet.from_array(y, y_desc))
        return

    if routine in ("dswap", "sswap"):
        x, y = n.call_args
        node = blas_nodes.Swap(f"swap_{builder.nid()}")
        _apply_promotions()
        state.add_node(node)
        x_desc = ctx.sdfg.arrays[x]
        y_desc = ctx.sdfg.arrays[y]
        state.add_edge(state.add_read(x), None, node, "_xin", Memlet.from_array(x, x_desc))
        state.add_edge(state.add_read(y), None, node, "_yin", Memlet.from_array(y, y_desc))
        state.add_edge(node, "_xout", state.add_write(x), None, Memlet.from_array(x, x_desc))
        state.add_edge(node, "_yout", state.add_write(y), None, Memlet.from_array(y, y_desc))
        return

    if routine in ("dger", "sger"):
        alpha, x, y, A = n.call_args
        node = blas_nodes.Ger(f"ger_{builder.nid()}", alpha=_scalar(alpha))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (x, "_x"), (y, "_y")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        a_desc = ctx.sdfg.arrays[A]
        state.add_edge(node, "_res", state.add_write(A), None, Memlet.from_array(A, a_desc))
        return

    if routine in ("dtrsv", "strsv", "dtrmv", "strmv"):
        is_trsv = routine.endswith("trsv")
        flags = n.expr.split(",")
        uplo_l = flags[0].strip("'\"").upper()[:1] or "L"
        trans_l = flags[1].strip("'\"").upper()[:1] or "N"
        diag_l = flags[2].strip("'\"").upper()[:1] or "N"
        A, x = n.call_args
        cls = blas_nodes.Trsv if is_trsv else blas_nodes.Trmv
        node = cls(f"{routine}_{builder.nid()}",
                   uplo=(uplo_l == "U"),
                   transA=(trans_l == "T"),
                   unit_diag=(diag_l == "U"))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (x, "_xin")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        x_desc = ctx.sdfg.arrays[x]
        state.add_edge(node, "_xout", state.add_write(x), None, Memlet.from_array(x, x_desc))
        return

    if routine in ("dsymv", "ssymv"):
        uplo_l = n.expr.strip("'\"").upper()[:1] or "L"
        alpha, A, x, beta, y = n.call_args
        node = blas_nodes.Symv(f"symv_{builder.nid()}", uplo=(uplo_l == "U"), alpha=_scalar(alpha), beta=_scalar(beta))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (x, "_x"), (y, "_yin")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        y_desc = ctx.sdfg.arrays[y]
        state.add_edge(node, "_yout", state.add_write(y), None, Memlet.from_array(y, y_desc))
        return

    if routine in ("dtrsm", "strsm", "dtrmm", "strmm"):
        is_trsm = routine.endswith("trsm")
        flags = n.expr.split(",")
        side_l = flags[0].strip("'\"").upper()[:1] or "L"
        uplo_l = flags[1].strip("'\"").upper()[:1] or "L"
        trans_l = flags[2].strip("'\"").upper()[:1] or "N"
        diag_l = flags[3].strip("'\"").upper()[:1] or "N"
        alpha, A, B = n.call_args
        cls = blas_nodes.Trsm if is_trsm else blas_nodes.Trmm
        node = cls(f"{routine}_{builder.nid()}",
                   side=(side_l == "R"),
                   uplo=(uplo_l == "U"),
                   transA=(trans_l == "T"),
                   unit_diag=(diag_l == "U"),
                   alpha=_scalar(alpha))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (B, "_Bin")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        b_desc = ctx.sdfg.arrays[B]
        state.add_edge(node, "_Bout", state.add_write(B), None, Memlet.from_array(B, b_desc))
        return

    if routine in ("dsymm", "ssymm"):
        flags = n.expr.split(",")
        side_l = flags[0].strip("'\"").upper()[:1] or "L"
        uplo_l = flags[1].strip("'\"").upper()[:1] or "L"
        alpha, A, B, beta, C = n.call_args
        node = blas_nodes.Symm(f"symm_{builder.nid()}",
                               side=(side_l == "R"),
                               uplo=(uplo_l == "U"),
                               alpha=_scalar(alpha),
                               beta=_scalar(beta))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (B, "_B"), (C, "_Cin")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        c_desc = ctx.sdfg.arrays[C]
        state.add_edge(node, "_Cout", state.add_write(C), None, Memlet.from_array(C, c_desc))
        return

    if routine in ("dsyrk", "ssyrk"):
        flags = n.expr.split(",")
        uplo_l = flags[0].strip("'\"").upper()[:1] or "L"
        trans_l = flags[1].strip("'\"").upper()[:1] or "N"
        alpha, A, beta, C = n.call_args
        node = blas_nodes.Syrk(f"syrk_{builder.nid()}",
                               uplo=(uplo_l == "U"),
                               transA=(trans_l == "T"),
                               alpha=_scalar(alpha),
                               beta=_scalar(beta))
        _apply_promotions()
        state.add_node(node)
        for arr, conn in ((A, "_A"), (C, "_Cin")):
            desc = ctx.sdfg.arrays[arr]
            state.add_edge(state.add_read(arr), None, node, conn, Memlet.from_array(arr, desc))
        c_desc = ctx.sdfg.arrays[C]
        state.add_edge(node, "_Cout", state.add_write(C), None, Memlet.from_array(C, c_desc))
        return


def emit_lapack(builder, ctx, n, region):
    """Lower a recognised Fortran LAPACK call (``kind == 'lapackcall'``)
    to a :mod:`dace.libraries.lapack` library node.

    Supported routines:

    * ``dgetrf`` / ``sgetrf``  -- LU factorisation
    * ``dpotrf`` / ``spotrf``  -- Cholesky factorisation
    """
    import importlib

    lapack_nodes = importlib.import_module("dace.libraries.lapack.nodes")
    ctx.flush_and_ensure(builder, region)
    state = ctx.new_state(builder, region)
    routine = n.callee.lower()

    if routine in ("dgetrf", "sgetrf"):
        A, ipiv, info = n.call_args
        node = lapack_nodes.Getrf(f"getrf_{builder.nid()}")
        state.add_node(node)
        a_desc = ctx.sdfg.arrays[A]
        state.add_edge(state.add_read(A), None, node, "_xin", Memlet.from_array(A, a_desc))
        state.add_edge(node, "_xout", state.add_write(A), None, Memlet.from_array(A, a_desc))
        if ipiv in ctx.sdfg.arrays:
            ipiv_desc = ctx.sdfg.arrays[ipiv]
            state.add_edge(node, "_ipiv", state.add_write(ipiv), None, Memlet.from_array(ipiv, ipiv_desc))
        if info in ctx.sdfg.arrays:
            info_desc = ctx.sdfg.arrays[info]
            state.add_edge(node, "_res", state.add_write(info), None, Memlet.from_array(info, info_desc))
        return

    if routine in ("dpotrf", "spotrf"):
        uplo_literal = n.expr.strip().strip("'\"").upper()[:1] or "L"
        A, info = n.call_args
        node = lapack_nodes.Potrf(f"potrf_{builder.nid()}", lower=(uplo_literal == "L"))
        state.add_node(node)
        a_desc = ctx.sdfg.arrays[A]
        state.add_edge(state.add_read(A), None, node, "_xin", Memlet.from_array(A, a_desc))
        state.add_edge(node, "_xout", state.add_write(A), None, Memlet.from_array(A, a_desc))
        if info in ctx.sdfg.arrays:
            info_desc = ctx.sdfg.arrays[info]
            state.add_edge(node, "_res", state.add_write(info), None, Memlet.from_array(info, info_desc))
        return

    if routine in ("dpotrs", "spotrs"):
        uplo_literal = n.expr.strip().strip("'\"").upper()[:1] or "L"
        A, B, info = n.call_args
        node = lapack_nodes.Potrs(f"potrs_{builder.nid()}", lower=(uplo_literal == "L"))
        state.add_node(node)
        a_desc = ctx.sdfg.arrays[A]
        b_desc = ctx.sdfg.arrays[B]
        state.add_edge(state.add_read(A), None, node, "_a", Memlet.from_array(A, a_desc))
        state.add_edge(state.add_read(B), None, node, "_bin", Memlet.from_array(B, b_desc))
        state.add_edge(node, "_bout", state.add_write(B), None, Memlet.from_array(B, b_desc))
        if info in ctx.sdfg.arrays:
            info_desc = ctx.sdfg.arrays[info]
            state.add_edge(node, "_res", state.add_write(info), None, Memlet.from_array(info, info_desc))
        return

    if routine in ("dgeqrf", "sgeqrf"):
        A, tau, info = n.call_args
        node = lapack_nodes.Geqrf(f"geqrf_{builder.nid()}")
        state.add_node(node)
        a_desc = ctx.sdfg.arrays[A]
        state.add_edge(state.add_read(A), None, node, "_ain", Memlet.from_array(A, a_desc))
        state.add_edge(node, "_aout", state.add_write(A), None, Memlet.from_array(A, a_desc))
        if tau in ctx.sdfg.arrays:
            tau_desc = ctx.sdfg.arrays[tau]
            state.add_edge(node, "_tau", state.add_write(tau), None, Memlet.from_array(tau, tau_desc))
        if info in ctx.sdfg.arrays:
            info_desc = ctx.sdfg.arrays[info]
            state.add_edge(node, "_res", state.add_write(info), None, Memlet.from_array(info, info_desc))
        return

    if routine in ("dorgqr", "sorgqr"):
        A, tau, info = n.call_args
        node = lapack_nodes.Orgqr(f"orgqr_{builder.nid()}")
        state.add_node(node)
        a_desc = ctx.sdfg.arrays[A]
        tau_desc = ctx.sdfg.arrays[tau]
        state.add_edge(state.add_read(A), None, node, "_ain", Memlet.from_array(A, a_desc))
        state.add_edge(state.add_read(tau), None, node, "_tau", Memlet.from_array(tau, tau_desc))
        state.add_edge(node, "_aout", state.add_write(A), None, Memlet.from_array(A, a_desc))
        if info in ctx.sdfg.arrays:
            info_desc = ctx.sdfg.arrays[info]
            state.add_edge(node, "_res", state.add_write(info), None, Memlet.from_array(info, info_desc))
        return


def emit_fft(builder, ctx, n, region):
    """Lower a recognised FFTW3 ``fftw_execute_dft`` call site
    (``kind == 'fftcall'``) to a :class:`dace.libraries.fft.nodes.FFT`
    (forward) or :class:`dace.libraries.fft.nodes.IFFT` (backward)
    library node.

    The C++ side absorbs the matching ``fftw_plan_dft_*`` and
    ``fftw_destroy_plan`` calls and only emits an ``fftcall`` ASTNode
    for the executing call.  The plan's ``rank`` / ``dims`` /
    ``direction`` are looked up at recognition time and carried on the
    ``ASTNode``:

    * ``n.expr`` -- ``"forward"`` (CUFFT_FORWARD / FFTW_FORWARD) or
      ``"backward"``.
    * ``n.call_args`` -- ``[in_array, out_array, dim0, dim1[, dim2]]``.

    A fresh successor state is opened so the side-effecting FFT runs
    in program order with respect to any other library calls
    (mirroring the MPI / I/O patterns).
    """
    import importlib

    fft_nodes = importlib.import_module("dace.libraries.fft.nodes")
    ctx.flush_and_ensure(builder, region)
    state = ctx.new_state(builder, region)

    in_arr, out_arr = n.call_args[0], n.call_args[1]
    is_inverse = (n.expr == "backward")
    cls = fft_nodes.IFFT if is_inverse else fft_nodes.FFT
    node = cls(f"{'i' if is_inverse else ''}fft_{builder.nid()}")
    state.add_node(node)

    in_desc = ctx.sdfg.arrays[in_arr]
    out_desc = ctx.sdfg.arrays[out_arr]
    # Use ``add_read`` / ``add_write`` (fresh nodes) rather than the cached
    # ``acc`` helper: when the Fortran source is in-place (the same array
    # for ``in`` and ``out``) the cache returns one shared access node and
    # the resulting in+out edge on a single node forms a self-cycle that
    # SDFG validation rejects.  A fresh read + write pair binds to the same
    # underlying array but lets the dataflow stay acyclic.
    state.add_edge(state.add_read(in_arr), None, node, "_inp", Memlet.from_array(in_arr, in_desc))
    state.add_edge(node, "_out", state.add_write(out_arr), None, Memlet.from_array(out_arr, out_desc))


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
    # ``@``) and flang's free-procedure mangling (``_QP<name>``) or
    # module-procedure mangling (``_QM<mod>P<name>``); strip both so
    # ``register_external("foo", ...)`` matches both a free-procedure
    # ``CALL foo`` and a ``CALL <module>::foo``.  We first try the bare
    # name; if no registration matches, fall through and try the
    # untouched mangled form so a caller can still register module-
    # qualified if they want to disambiguate two same-named callees.
    callee_raw = n.callee.lstrip('@')
    callee = callee_raw
    if callee.startswith('_QP'):
        callee = callee[3:]
    elif callee.startswith('_QM'):
        # ``_QM<mod>P<name>`` -- the bare proc name is everything after
        # the *last* ``P`` (handles module names that contain a ``P``).
        p_idx = callee.rfind('P')
        if p_idx > 2:
            callee = callee[p_idx + 1:]
    sig = lookup_external(callee)
    if sig is None and callee != callee_raw:
        sig = lookup_external(callee_raw)
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
    # aos member, else ``None``).  ``group_c_abi[gid]`` records each aos
    # group's :meth:`Arg.resolved_c_abi` so the body-generation step
    # picks between the AoS-struct pack/unpack path
    # (``'aos_struct_ptr'``) and the per-member SoA pass-through
    # (``'per_member_soa'``); decoupling Arg's Fortran-side ``kind``
    # from its C-ABI shape this way is what lets a sibling-SDFG callee
    # receive the same SoA flats the marshal expansion already produces,
    # with no intermediate AoS round-trip.
    plan: list = []
    group_c_abi: dict = {}
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
                raise ValueError(f"external {callee!r}: 'aos' arg #{gi} has no "
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
            group_c_abi[gi] = a.resolved_c_abi()
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
    # gid -> [(ctype, n_elems, in_conn|None, out_conn|None, shape)]; n_elems
    # == 1 for a scalar member, else the member array's total element count.
    # ``shape`` is the SDFG-array shape tuple (sympy expressions), used
    # when ``sig.dynamic_extents_abi`` and ``n_elems == 0`` to prepend
    # one ``int`` extent per dim before the leaf pointer.
    group_members: dict = defaultdict(list)
    # Per-``logical_terms``-position record for ``kind='array'`` args
    # whose connected SDFG array carries a symbolic shape; consumed by
    # the body / decl-types builders when ``sig.dynamic_extents_abi`` is
    # true so each such pointer arg gets one ``int`` extent per dim
    # prepended (the C ABI :func:`emit_bind_c_shim` exports).  Maps the
    # logical-term position to ``(shape_tuple,)``.
    array_shape_at_term: dict = {}
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
                in_conns.append(cin)
                ptr_of[cin] = dt
                edges.append((name, cin, 'r'))
            if writes:
                out_conns.append(cout)
                ptr_of[cout] = dt
                edges.append((name, cout, 'w'))
            group_members[gid].append(
                (ctype, nel, cin if reads else None, cout if writes else None, tuple(shape) if shape else ()))
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
            in_conns.append(cin)
            comm_conns.add(cin)
            edges.append((name, cin, 'r'))
            logical_terms.append(('lit', cin))
            continue
        dt = ctx.sdfg.arrays[name].dtype
        if kind == 'array':
            reads = intent in ('in', 'inout')
            writes = intent in ('out', 'inout')
            cin, cout = f"_a{i}", f"_a{i}_o"
            if reads:
                in_conns.append(cin)
                ptr_of[cin] = dt
                edges.append((name, cin, 'r'))
            if writes:
                out_conns.append(cout)
                ptr_of[cout] = dt
                edges.append((name, cout, 'w'))
            arr_shape = tuple(getattr(ctx.sdfg.arrays[name], "shape", ()) or ())
            if sig.dynamic_extents_abi and _shape_is_symbolic(arr_shape):
                array_shape_at_term[len(logical_terms)] = arr_shape
            logical_terms.append(('lit', cout if writes else cin))
        else:  # 'scalar'
            cin = f"_a{i}"
            in_conns.append(cin)
            edges.append((name, cin, 'r'))
            logical_terms.append(('lit', cin))

    # Assemble the C body.  Per aos group, ``group_c_abi[gid]`` picks
    # the route:
    #   * ``aos_struct_ptr`` -- declare a local AoS buffer, pack its
    #     pointer members in (in/inout), pass ``&buffer``, then unpack
    #     out (out/inout) -- the SoA<->AoS cast lives entirely in this
    #     tasklet.  Today's default; the opaque-C-library shape.
    #   * ``per_member_soa`` -- forward each leaf member's connector
    #     directly to the external in marshal-expansion order; no AoS
    #     buffer, no pack/unpack copy.  Used for sibling-SDFG callees
    #     that already speak SoA (their ``bind_c_shim`` receives the
    #     per-member slots the marshal pass produced).
    body_lines: list = []
    call_args_c: list = []
    bufname: dict = {}
    for term_index, (kind, val) in enumerate(logical_terms):
        if kind == 'lit':
            shape = array_shape_at_term.get(term_index)
            if shape:
                for s in shape:
                    call_args_c.append(f"(int)({_sym2c(s)})")
            call_args_c.append(val)
            continue
        gid = val
        mems = group_members[gid]
        abi = group_c_abi.get(gid, 'aos_struct_ptr')
        if abi == 'per_member_soa':
            # Per-leaf pass-through: every leaf forwards its writable
            # connector when present (so codegen sees the write
            # dependency), else its readable one.  No struct buffer,
            # no copy in or out.  When the callee's ABI is
            # ``dynamic_extents_abi`` (a bind_c_shim'd sibling SDFG),
            # each dynamic-shape leaf (``nel == 0``) gets one ``int``
            # extent per dim prepended to feed the shim's
            # ``c_f_pointer`` shape constructor.
            for ct, nel, cin, cout, shape in mems:
                tok = cout if cout is not None else cin
                if sig.dynamic_extents_abi and nel == 0 and shape:
                    for s in shape:
                        call_args_c.append(f"(int)({_sym2c(s)})")
                call_args_c.append(tok)
            continue
        # ``aos_struct_ptr`` path (today's default).
        buf = f"_aosbuf{gid}"
        bufname[gid] = buf
        # One struct field per member: ``T mK;`` (scalar) or ``T mK[N];``
        # (array).  The field layout mirrors the Fortran derived type, so the
        # external's AoS pointer addresses the same contiguous bytes.
        # ``nel == 0`` is the dynamic-shape sentinel: AoS pack/unpack for
        # that member is statically unrenderable and the node is destined
        # for ``inline_external`` rewrite before codegen.  Render the field
        # as a pointer placeholder so the surrounding struct stays well-
        # formed and skip the pack/unpack lines below.
        fields = " ".join((f"{ct} m{k};" if nel == 1 else f"{ct}* m{k};" if nel == 0 else f"{ct} m{k}[{nel}];")
                          for k, (ct, nel, _, _, _) in enumerate(mems))
        body_lines.append(f"struct {{ {fields} }} {buf};")
        for k, (ct, nel, cin, cout, _shape) in enumerate(mems):
            if cin is None or nel == 0:
                continue
            if nel == 1:
                body_lines.append(f"{buf}.m{k} = (*{cin});")
            else:
                body_lines.append(f"for (int _i = 0; _i < {nel}; ++_i) "
                                  f"{buf}.m{k}[_i] = {cin}[_i];")
        call_args_c.append(f"(void*)(&{buf})")
    # Forward Fortran module globals across the C ABI: read each
    # ``__<module>_MOD_<member>`` symbol directly from the OUTER
    # library's BSS (where the outer's wrapper has already written
    # it from the caller's args via the existing ``use ...`` import
    # path) and append the value to the call.  The matching
    # ``bind_c_shim`` slot writes the INNER library's copy so the
    # callee sees the same value.  See ``ExternalSignature.
    # module_symbol_forward`` for the rationale (per-library
    # Fortran-module-globals issue).  Pulled into a separate prefix
    # list so the ``extern`` declarations for each ``__<mod>_MOD_<mem>``
    # symbol get rendered once just before the call.
    module_extern_decls: list = []
    for module, member, dtype, rank in sig.module_symbol_forward:
        sym = f"__{module}_MOD_{member}"
        ct = _MOD_FORWARD_CTYPE.get(dtype)
        if ct is None:
            raise ValueError(f"external {callee!r}: unsupported module_symbol_forward "
                             f"dtype {dtype!r} for ``{module}::{member}``")
        if rank == 0:
            # gfortran emits the scalar BSS as a ``<ct>``.  Pass the
            # value by value.  ``extern`` (no language linkage --
            # the body declarations are inside a function scope where
            # ``extern "C"`` is illegal; the symbol's ABI is fixed by
            # gfortran's mangling regardless).
            module_extern_decls.append(f'extern {ct} {sym};')
            call_args_c.append(sym)
        else:
            # Rank-N module array: gfortran emits a flat BSS region;
            # the symbol decays to a pointer, which the C ABI takes
            # directly.
            module_extern_decls.append(f'extern {ct} {sym}[];')
            call_args_c.append(sym)
    body_lines = module_extern_decls + body_lines
    body_lines.append(f"{sig.c_name}({', '.join(call_args_c)});")
    # AoS-struct-ptr copy-out (per_member_soa needs no unpack: writes
    # land in the connector directly via the call).
    for gid, mems in group_members.items():
        if group_c_abi.get(gid, 'aos_struct_ptr') != 'aos_struct_ptr':
            continue
        for k, (ct, nel, cin, cout, _shape) in enumerate(mems):
            if cout is None or nel == 0:
                continue
            if nel == 1:
                body_lines.append(f"(*{cout}) = {bufname[gid]}.m{k};")
            else:
                body_lines.append(f"for (int _i = 0; _i < {nel}; ++_i) "
                                  f"{cout}[_i] = {bufname[gid]}.m{k}[_i];")

    # Build the ``extern "C"`` declaration at the call site so an
    # ``Arg(kind='aos', c_abi='per_member_soa')`` arg expands to its
    # actual leaf signature (one ``<ctype>*`` per leaf member, in
    # marshal-expansion order).  An ``aos_struct_ptr`` group keeps the
    # single ``void*`` shape; any non-aos arg lifts its
    # ``Arg.c_decl_type()`` verbatim.  When the callee's ABI is
    # ``dynamic_extents_abi``, each dynamic-shape leaf (per_member_soa
    # member with ``nel == 0`` or ``kind='array'`` with symbolic
    # shape) is prefixed with one ``int`` per dim -- matching the
    # extents the body emission prepends.
    decl_types: list = []
    sig_arg_iter = iter(sig.args)
    cur_sig_arg = next(sig_arg_iter, None)
    last_gid_seen = None
    last_member_idx = -1
    plan_term_index = -1  # mirrors ``logical_terms`` indexing for non-aos
    for kind, dtype, intent, gid in plan:
        if gid is None:
            plan_term_index += 1
            arr_shape = array_shape_at_term.get(plan_term_index)
            if arr_shape:
                decl_types.extend(["int"] * len(arr_shape))
            decl_types.append(cur_sig_arg.c_decl_type())
            cur_sig_arg = next(sig_arg_iter, None)
            last_gid_seen = None
            last_member_idx = -1
            continue
        if gid != last_gid_seen:
            plan_term_index += 1
            last_gid_seen = gid
            last_member_idx = -1
            cur_sig_arg = next(sig_arg_iter, None)  # consume the aos sig arg
        last_member_idx += 1
        if group_c_abi.get(gid) == 'per_member_soa':
            ct, nel, _cin, _cout, shape = group_members[gid][last_member_idx]
            if sig.dynamic_extents_abi and nel == 0 and shape:
                decl_types.extend(["int"] * len(shape))
            decl_types.append(f"{ct}*")
        elif last_member_idx == 0:  # aos_struct_ptr: emit once per group
            decl_types.append("void *")
    # ``module_symbol_forward`` values append AFTER every other arg
    # in the same order ``bind_c_shim`` declares them, so the inner
    # shim's signature lines up with what we pass here.  Scalars
    # ride by value (``int`` / ``double`` / ...); rank-N module
    # arrays decay to the matching pointer (``<ct>*``) on the C ABI.
    for module, member, dtype, rank in sig.module_symbol_forward:
        ct = _MOD_FORWARD_CTYPE.get(dtype)
        if ct is None:
            raise ValueError(f"external {callee!r}: unsupported module_symbol_forward "
                             f"dtype {dtype!r} for ``{module}::{member}``")
        decl_types.append(f"{ct}*" if rank > 0 else ct)
    c_decl = f'extern "C" void {sig.c_name}({", ".join(decl_types) or "void"});'

    node = ExternalCall(name=f"_ext_{callee}_{builder.nid()}",
                        c_name=sig.c_name,
                        c_decl=c_decl,
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
