"""SDFG descriptor registration + type mapping + synthetic-scalar lazy decl.

``add_descriptors`` is called once from ``SDFGBuilder.build()`` to register
symbols, arrays, and scalars on the fresh SDFG.  ``auto_declare_synth``
runs on-demand from the emit path when the bridge introduces synthetic
scalars (``__sc_N`` / ``__al_N``) that weren't in the original variable
classification.
"""

from types import SimpleNamespace

import dace
from dace import SDFG

DTYPE = {
    'float64': dace.float64,
    'float32': dace.float32,
    'int8': dace.int8,
    'int16': dace.int16,
    'int32': dace.int32,
    'int64': dace.int64,
    # MLIR ``index`` (pointer-width integer) backs array extents and the
    # AoS-allocatable ``cap_<base>_<member>`` symbol; it is an integer,
    # not the ``float64`` default.
    'index': dace.int64,
    'bool': dace.bool_,
    'uint8': dace.uint8,
    'complex64': dace.complex64,
    'complex128': dace.complex128,
}


def dt(s: str) -> dace.typeclass:
    """Map a Fortran type string to its DaCe ``typeclass`` (default ``float64``)."""
    return DTYPE.get(s, dace.float64)


def sdfg_name(builder) -> str:
    """Derive the SDFG name -- and therefore the generated ``.so``
    library name -- from the procedure being built.

    Prefers the explicit ``entry`` symbol passed to :class:`SDFGBuilder`
    (demangled to the procedure name -- ``_QMmoduleP<proc>`` or
    ``_QP<proc>`` -> ``<proc>``).  Falls back to the first ``_QF<proc>``
    mangled name on a registered variable, then to a generic ``sdfg``.

    Using the entry-procedure name means
    ``build_sdfg_from_hlfir(..., entry="_QMmo_velocity_advectionPvelocity_tendencies")``
    produces ``libvelocity_tendencies.so`` instead of a generic name,
    so a registered external callee can be linked against it by
    function-keyed library name.
    """
    entry = getattr(builder, "entry", None)
    if entry:
        proc = entry.rsplit("P", 1)[-1] if "P" in entry else entry
        if proc:
            return proc
    for v in builder.arrays.values():
        mn = v.mangled_name
        if '_QF' in mn and 'E' in mn:
            return mn.split('_QF')[1].split('E')[0]
    return "sdfg"


def _fortran_strides(dims):
    """Column-major strides: ``stride[i]`` is the product of
    ``dims[0..i-1]``.  Fortran's declaration ``real :: a(nproma, nlev,
    nblks_e)`` has nproma as the fastest-varying index (stride 1),
    matching what Flang's HLFIR expects  --  so the SDFG descriptor must
    advertise the same layout or DaCe's C-order default will mis-index
    when called with numpy F-order inputs.

    :param dims: ordered extents (ints or symbolic).
    :returns: column-major stride list, same length as ``dims``.
    """
    strides = []
    acc = 1
    for d in dims:
        strides.append(acc)
        acc = acc * d
    return strides


def add_descriptors(builder, sdfg: SDFG):
    """Add symbols, arrays, and scalars to ``sdfg`` from ``builder``'s
    classified variable dicts.

    Scalar rule: locals (``intent=''``) -> ``dace.data.Scalar`` transient.
    Scalar OUTPUTS (``intent in/out/inout`` other than pure ``in``) land as
    length-1 ``dace.data.Array`` because the caller needs a writable buffer.
    Scalar INPUTS (``intent(in)`` or ``REAL(8), VALUE :: x``) register as
    non-transient ``dace.data.Scalar`` so callers pass plain ``int`` /
    ``float`` and the C++ codegen reads ``x`` directly.
    """
    # Named Fortran symbols (nproma, nlev, ...).
    for v in builder.symbols.values():
        sdfg.add_symbol(v.fortran_name, dt(v.dtype))

    # Per-dim ``?`` entries (e.g. ``vn_ie(nproma, nlev+1, nblks_e)``  --
    # ``resolveShapeSyms`` returns ``"?"`` for the arith-derived middle
    # extent) become a synthetic ``<arr>_d<dim>`` name so DaCe sees a
    # legal symbol; the caller-side binding emitter still passes the real
    # extent at call time.  Same shape as the whole-list-empty fallback
    # in the bridge, just applied per-dim.  ``v.shape_symbols`` is a
    # fresh-copy nanobind property on every read, so we materialise the
    # rewritten list in a Python-side dict keyed by name and route the
    # downstream loops through it.
    shape_syms = {}
    for v in builder.arrays.values():
        syms = list(v.shape_symbols)
        for dim, s in enumerate(syms):
            if s == "?":
                syms[dim] = f"{v.fortran_name}_d{dim}"
        shape_syms[v.fortran_name] = syms

    # Closed-form extent expressions (``traceExtentExpr`` output for a
    # dynamic gather-temp first dim, e.g. ``"max((endcol - startcol) +
    # 1, 0)"``).  Detect by presence of arithmetic operators or
    # parentheses -- a bare symbol/literal won't contain any of those.
    def _is_expr(s: str) -> bool:
        return any(c in s for c in '+-*/()')

    # Synthetic symbols for dims that stayed unresolved after passes.
    # Literal-integer dimensions (e.g. the "3" in ``edge_idx(nc, 3)``) stay
    # as Python ints and do not need a symbol registration.  Expression
    # strings (gather-temp dynamic extents) reference leaf symbols that
    # are registered separately (via the Pass 2c triplet-bound
    # promotion) -- skip the registration here.
    known = {v.fortran_name for v in builder.variables}
    for v in builder.arrays.values():
        for s in shape_syms[v.fortran_name]:
            if s.lstrip('-').isdigit() or _is_expr(s):
                continue
            if s not in known and s not in sdfg.symbols:
                sdfg.add_symbol(s, dace.int64)

    def _dim(s: str):
        if s.lstrip('-').isdigit():
            return int(s)
        if _is_expr(s):
            return dace.symbolic.pystr_to_symbolic(s)
        return dace.symbol(s)

    # Flang emits internal temporaries with dotted names (e.g.
    # ``.tmp.arrayctor`` for the array constructor backing an
    # ``out = [n, m]``-style RHS, and ``.c.<type>`` /
    # ``.dt.<type>`` for type-info records on derived-type uses).
    # DaCe's ``NestedDict`` treats dots as nesting separators and
    # rejects them as keys.  Skip these declares at descriptor time
    #  --  the bridge's accesses still reference the original
    # dummy/declare names, so dropping the internals is safe.
    def _is_flang_internal(nm: str) -> bool:
        return nm.startswith(".")

    def _is_char_literal(dtype) -> bool:
        """A Fortran ``CHARACTER`` constant from flang's literal pool (an I/O
        ``filename`` / ``status`` / format string).  DaCe has no string data
        type and these are never compute data -- they feed I/O statements,
        which the ``fortran_io`` recognizer handles separately -- so they must
        not be registered as SDFG descriptors.  Doing so mints invalid array
        names (the hex-encoded literal contents, e.g. ``747874`` for ``txt``,
        which also start with a digit)."""
        return str(dtype).startswith("!fir.char")

    # Per-axis offset symbols for every array.  ``offset_<arr>_d<i>`` is
    # the value subtracted from the Fortran 1-based index in every
    # memlet (see ``access.py::build_memlet_index``).  Default value for
    # a Fortran array is ``1`` (the standard lb); ``dimension(20:24)``
    # picks up ``20`` from the declare's shape_shift; ``dimension(lo:hi)``
    # with caller-supplied ``lo`` falls through to ``None`` and the
    # symbol stays free on the SDFG signature.  Populated here so
    # ``builder.offset_values`` is fully filled before any AST emission
    # references the symbols in memlet subsets.
    def _offset_value(s: str):
        s = s.strip()
        if s == "?" or not s:
            return None
        if s.lstrip('-').isdigit():
            return int(s)
        # Symbolic lb (e.g. caller-supplied ``arrsize``).  If the symbol
        # is already declared on the SDFG (a known dummy / Fortran sym),
        # pass the name through; sdfg.specialize will alias one symbol
        # to the other.  Otherwise leave unknown so the offset stays
        # free.
        return s if s in sdfg.symbols else None

    for v in builder.arrays.values():
        if _is_flang_internal(v.fortran_name) or _is_char_literal(v.dtype):
            continue
        if v.role == 'section_alias':
            # Trivial section slice  --  no SDFG descriptor, no offset
            # symbols.  Accesses through the inlined-body dummy rewrite
            # to source-array memlets via ``view_dim_map`` in
            # ``access.py`` / ``emit_tasklet.py``.
            continue
        dims = [_dim(s) for s in shape_syms[v.fortran_name]]
        if v.bounds_remap_view:
            # Fortran 2003 bounds-remapping pointer assignment lowered
            # to a 1-D contiguous view of the parent array.  See
            # ``passes/MarkBoundsRemapViews.cpp`` for the detection
            # contract.  The View shares storage with
            # ``v.bounds_remap_source``; element ``ptr(i)`` lowers to
            # ``parent_flat[offset_<ptr>_d0 + i - 1]`` at codegen.
            #
            # Shape: ``[total_extent]`` -- the flat 1-D size flang
            # encodes on the rebox's shape-shift extent operand.  When
            # ``v.bounds_remap_total_extent`` is a plain symbol /
            # arithmetic expression the bridge extracted from the
            # rebox, use it directly; otherwise fall back to a
            # synthesised ``<ptr>_total_extent_d0`` symbol the caller
            # binds.
            #
            # Strides: ``[1]`` -- per the spec, the contiguous case.
            # QE's two sites flatten a column-major rank-2 slice over
            # the last dim, which IS stride 1.  Other contiguous-
            # column shapes also land here; non-contiguous slices
            # wouldn't have triggered the mark pass in the first
            # place (those produce a different rebox shape).
            #
            # Offset: ``0`` -- the View descriptor itself is offset-0.
            # The per-rebind column offset into the parent is bound to
            # a fresh ``offset_<ptr>_d0`` symbol (minted below
            # alongside every array's offset symbols), assigned per
            # loop iteration via interstate edge (a follow-up commit
            # wires the assignment).
            extent_str = v.bounds_remap_total_extent
            if not extent_str:
                # Synth fallback: caller binds a free total-extent
                # symbol when the rebox's extent operand wasn't
                # statically expressible.
                extent_str = f"{v.fortran_name}_total_extent_d0"
                if extent_str not in sdfg.symbols:
                    sdfg.add_symbol(extent_str, dace.int64)
            sdfg.add_view(
                v.fortran_name,
                shape=[dace.symbolic.pystr_to_symbolic(extent_str)],
                dtype=dt(v.dtype),
                strides=[1],
            )
        elif v.role == 'view_alias':
            # Pointer alias of ``v.view_source``  --  no separate storage.
            # ``sdfg.add_view`` registers a static reference that DaCe
            # codegen lowers to a typed pointer into the source's
            # buffer; per-state linking memlets (added by the ``acc``
            # factory) tell DaCe which slice of the source the view
            # covers.
            #
            # View strides are derived from the source array's Fortran
            # column-major strides times each surviving section dim's
            # triplet stride.  Example: source ``a(100, 10)`` has
            # strides ``(1, 100)``; section ``a(:, 1:10:2)`` keeps both
            # dims (full range on dim 0, stride-2 on dim 1) so the view
            # has shape ``(100, 5)`` strides ``(1, 200)``.  Source dims
            # collapsed to a scalar are dropped.  Section ``a(i, :)``
            # has shape ``(10,)`` stride ``(100,)``.
            src_v = builder.arrays.get(v.view_source)
            src_dims = (shape_syms.get(v.view_source) if src_v is not None else None)
            src_strides = (_fortran_strides([_dim(s) for s in src_dims]) if src_dims and len(src_dims) > 1 else None)
            view_strides = []
            if src_strides is not None and len(v.view_subset) == len(src_strides):
                for src_d, sub in enumerate(v.view_subset):
                    if ':' not in sub:
                        continue  # scalar dim  --  drops out of the view
                    parts = sub.split(':')
                    sec_stride = int(parts[2]) if len(parts) >= 3 else 1
                    view_strides.append(src_strides[src_d] * sec_stride)
            # Only honour the derived strides if their length matches
            # the view's rank.  ``view_reshape`` cases use a
            # ``fir.convert`` to flatten the section (rank reduction
            # beyond just scalar dims), so the per-surviving-section-
            # dim stride list has more entries than the view has dims.
            # In those cases the section is contiguous in storage and
            # ``[1, ...]`` is correct.
            if len(view_strides) != len(dims):
                view_strides = [1] * len(dims) if len(dims) > 0 else None
            sdfg.add_view(
                v.fortran_name,
                shape=dims,
                dtype=dt(v.dtype),
                strides=view_strides,
            )
        else:
            # Length-1 transient arrays become Scalars for better
            # compatibility with DaCe transformations (which often
            # recognise ``Scalar`` natively but skip ``Array(shape=(1,))``).
            # Caller-provided length-1 args stay ``Array(1,)`` -- the
            # caller owns the buffer and the pass-by-pointer ABI needs
            # a real array descriptor.  Rank-0 source vars don't hit
            # this branch (they live in ``builder.scalars``); this only
            # triggers for explicit ``REAL :: x(1)`` declarations whose
            # source already names a length-1 array.
            #
            # ``transient`` follows the bake / kwarg classification
            # mirrored from ``hlfir-preserve-mutable-globals`` -- a
            # baked constant (PARAMETER or function-scope local) is a
            # transient backed by ``add_constant`` data; every other
            # intent-empty global is a caller kwarg (non-transient
            # (1,)-Array surfacing on the SDFG signature).
            from dace_fortran.builder import _global_is_baked_constant
            transient = (v.intent == '' and _global_is_baked_constant(v))
            is_length_one = len(dims) == 1 and dims[0] == 1
            if transient and is_length_one:
                sdfg.add_scalar(v.fortran_name, dtype=dt(v.dtype), transient=True)
            else:
                sdfg.add_array(
                    v.fortran_name,
                    shape=dims,
                    dtype=dt(v.dtype),
                    transient=transient,
                    strides=_fortran_strides(dims) if len(dims) > 1 else None,
                )
        # Declare an offset symbol per dim, sized from the SDFG array's
        # rank (not ``v.lower_bounds`` which may be shorter for some
        # synth shapes).  Unknown lower bounds default to ``1``.
        rank = len(dims)
        for d in range(rank):
            sym_name = f"offset_{v.fortran_name}_d{d}"
            if sym_name not in sdfg.symbols:
                sdfg.add_symbol(sym_name, dace.int64)
            if v.bounds_remap_view:
                # The offset of a bounds-remap view is *dynamic*: it
                # encodes the column offset into the parent array and
                # changes per surrounding-loop iteration (e.g. each
                # ``jbnd`` step in QE's ``addusxx_g`` rebinds
                # ``prhoc_d`` to a different column slice).
                # Setting ``offset_values`` to ``None`` keeps the
                # symbol free on the SDFG signature so it survives
                # ``_specialize_symbol``'s constant-fold sweep; the
                # interstate-edge assignment that binds it lives in
                # the View's linking-memlet wiring (follow-up).
                builder.offset_values[sym_name] = None
                continue
            lb = v.lower_bounds[d] if d < len(v.lower_bounds) else "1"
            builder.offset_values[sym_name] = _offset_value(lb)

    for v in builder.scalars.values():
        if _is_flang_internal(v.fortran_name) or _is_char_literal(v.dtype):
            continue
        # Cross-role collision guard.  When ``hlfir-inline-all`` splices
        # multiple callees into the entry, the bridge's collector can
        # surface several declares with the same bare ``fortran_name``
        # in different roles -- e.g. graupel's ``qs`` shows up once as
        # the entry's INTENT(INOUT) ARRAY dummy (added via ``arrays``
        # above) AND again as the scalar dummy of each inlined PURE
        # FUNCTION (``snow_lambda``, ``cloud_to_snow``, ...).  Without
        # this guard, ``sdfg.add_scalar`` raises FileExistsError on
        # the second add.  The array binding already represents the
        # storage the caller hands in; the inlined-callee scalars are
        # value-passed locals whose downstream uses route through the
        # array's access node, so skipping the scalar add is the right
        # choice.  Same applies to symbol collisions (a callee-local
        # whose name collides with a shape symbol).
        if v.fortran_name in sdfg.arrays or v.fortran_name in sdfg.symbols:
            continue
        if v.intent == '':
            # Local transient scalar.
            sdfg.add_scalar(v.fortran_name, dtype=dt(v.dtype), transient=True)
        elif v.intent in ('out', 'inout'):
            # Scalar OUTPUT must remain a length-1 array on the SDFG
            # signature -- the runtime needs a writable buffer the
            # caller hands in (Python ``float`` would be pass-by-value
            # so updates wouldn't surface on the caller side).
            sdfg.add_array(v.fortran_name, shape=(1, ), dtype=dt(v.dtype), transient=False)
        else:
            # Scalar INPUT (``intent(in)`` or ``REAL(8), VALUE :: x``).
            # Register as a true Scalar -- DaCe accepts plain Python
            # ``int`` / ``float`` for these and the C++ codegen reads
            # ``x`` directly instead of ``x[0]``.  Matches Fortran's
            # pass-by-value semantics (the kernel gets its own copy
            # of the constant).
            sdfg.add_scalar(v.fortran_name, dtype=dt(v.dtype), transient=False)


def declare_synth_array(builder, name: str, shape, dtype: str, ctx):
    """Register a bridge-synthesised transient array on the SDFG and in
    ``builder.arrays``.  Used by the ``kind="declare_transient"`` AST
    handler: when the bridge emits a per-element loop that fills a
    scratch mask before a reduction or select library node, this is the
    one-stop helper that creates the array descriptor.

    ``shape`` is a list of strings; literal-integer entries are parsed
    as Python ints, anything else is treated as a symbol name and looked
    up via ``dace.symbol``.  No-op if ``name`` already exists.
    """
    if name in ctx.sdfg.arrays:
        return
    dims = []
    for s in shape:
        if isinstance(s, int):
            dims.append(s)
            continue
        s_str = str(s).strip()
        if s_str.lstrip('-').isdigit():
            dims.append(int(s_str))
        else:
            if s_str not in ctx.sdfg.symbols:
                ctx.sdfg.add_symbol(s_str, dace.int64)
            dims.append(dace.symbol(s_str))
    # Fortran-style transient: rank > 1 -> column-major strides so the
    # matmul / transpose / dot_product library nodes (which inherit
    # layout from the source operands' strides) write the result in the
    # same layout the bridge-declared dummy arrays use.  Single-rank
    # transients (or scalars) take DaCe's default contiguous stride.
    strides = _fortran_strides(dims) if len(dims) > 1 else None
    # Length-1 synthesised transient -> Scalar (same rule as the
    # outer add_descriptors path).  Better DaCe-pass compatibility.
    if len(dims) == 1 and dims[0] == 1:
        ctx.sdfg.add_scalar(name, dtype=dt(dtype), transient=True)
    else:
        ctx.sdfg.add_array(name, shape=dims, dtype=dt(dtype), transient=True, strides=strides)
    # Mirror the entry into ``builder.arrays`` so subsequent emit_assign
    # / emit_libcall calls find it via the existing arrays-dict lookups.
    builder.arrays[name] = SimpleNamespace(
        fortran_name=name,
        intent='',
        dtype=dtype,
        rank=len(shape),
        is_dynamic=False,
        role='array',
        shape_symbols=[str(s) for s in shape],
        lower_bounds=['1'] * len(shape),
    )
    # Per-axis offset symbols + values (always 1 for bridge-synthesised
    # transients  --  they're allocated fresh with Fortran's default lb).
    for d in range(len(shape)):
        sym_name = f"offset_{name}_d{d}"
        if sym_name not in ctx.sdfg.symbols:
            ctx.sdfg.add_symbol(sym_name, dace.int64)
        builder.offset_values[sym_name] = 1


def emit_declare_transient(builder, ctx, n, region):
    """Handler for ASTNode kind=\"declare_transient\".

    Reads ``n.target`` (name), ``n.expr`` (dtype as string), and shape
    from ``n.accesses[0].index_exprs`` (one string per dim).  Calls
    ``declare_synth_array`` to register the SDFG descriptor.

    Resolves ``<src>_d<N>`` shape placeholders against the source array's
    actual SDFG descriptor: the bridge emits these for synthesised
    transients derived from a known source (e.g. a transpose-of-A
    transient picked up by ``MATMUL(A, TRANSPOSE(B))``) but doesn't
    have access to the source's already-registered symbolic shape.
    Without this rewrite the transient would carry a fresh symbol like
    ``b_d1`` that is symbolically distinct from ``b.shape[1]`` and the
    downstream lib node's same-dim validation rejects the mismatch.
    """
    import re

    shape = list(n.accesses[0].index_exprs) if n.accesses else []
    resolved = []
    for entry in shape:
        s = str(entry)
        m = re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*)_d(\d+)', s)
        if m:
            src, dim = m.group(1), int(m.group(2))
            if src in ctx.sdfg.arrays:
                desc = ctx.sdfg.arrays[src]
                if 0 <= dim < len(desc.shape):
                    resolved.append(desc.shape[dim])
                    continue
        resolved.append(entry)
    declare_synth_array(builder, n.target, resolved, n.expr or "int32", ctx)


def auto_declare_synth(builder, name: str, ctx):
    """Lazy-declare a synthetic scalar minted by the bridge's faithful
    scf.while walker.  ``__sc_N`` names materialise ``scf.if -> T``
    results; ``__al_N`` names come from bare ``fir.alloca`` ops that
    lift-cf-to-scf uses as scratch counters.  Both need an SDFG
    descriptor + an entry in ``builder.scalars`` so ``emit_assign``'s
    existing dispatch (scalar pending, or symbol state-change) can
    fire normally.  Treated as transient ints  --  they only live for
    the loop's lifetime and are read only by downstream generated
    conditions.
    """
    if name in builder.scalars or name in builder.symbols:
        return
    if not (name.startswith("__sc_") or name.startswith("__al_")):
        return
    # Fake a VarInfo-like record so _add_descriptors-consistent paths work.
    # A ``SimpleNamespace`` is enough  --  scalar dispatch only reads
    # ``.intent`` and ``.dtype``.
    v = SimpleNamespace(fortran_name=name,
                        intent='',
                        dtype='int32',
                        rank=0,
                        is_dynamic=False,
                        role='scalar',
                        shape_symbols=[],
                        lower_bounds=[])
    builder.scalars[name] = v
    if name not in ctx.sdfg.arrays:
        ctx.sdfg.add_scalar(name, dtype=dace.int32, transient=True)
