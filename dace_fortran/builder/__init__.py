"""``SDFGBuilder``  --  walks the HLFIR AST to directly construct a DaCe SDFG.

Pipeline:
    flang-20 -fc1 -emit-hlfir code.f90 -o code.hlfir
    sdfg = generate_sdfg("code.hlfir")   # -> dace.SDFG, validated

Architecture:
    The builder parses the HLFIR via the C++ bridge (``hlfir_bridge.so``),
    runs the default pass pipeline, then walks the recursive ASTNode tree
    and emits DaCe constructs:

        kind="assign"      -> Tasklet / interstate-edge assignment
        kind="loop"        -> LoopRegion (nested for-loop)
        kind="while"       -> LoopRegion (while form, post lift-cf-to-scf)
        kind="conditional" -> ConditionalBlock + ControlFlowRegion per branch
        kind="copy"        -> CopyLibraryNode
        kind="memset"      -> MemsetLibraryNode
        kind="libcall"     -> BLAS / standard library node (MatMul, Dot, ...)
        kind="reduce"      -> standard.Reduce
        kind="break"       -> BreakBlock
        kind="return"      -> ReturnBlock

Per-emitter implementations live in sibling modules under this package.
``SDFGBuilder`` itself keeps only orchestration  --  ``__init__``, ``build``,
``nid``, and the ``_emit`` dispatch.

State-change rules:
    - Write to a symbol -> interstate edge with assignment (emit_assign).
    - Every other write -> tasklet in the current state.
    - LoopRegion / ConditionalBlock open a fresh region; their children
      run in a nested ``_Ctx``.

NOTE on nanobind bindings:
    Every read of a std::vector-typed attribute (e.g. ast_node.children,
    var.shape_symbols, assign.accesses) returns a FRESH Python list copy.
    Hot paths cache such attributes into locals.
"""

from dace import InterstateEdge, SDFG

from dace_fortran.build_bridge import hb

from dace_fortran.builder.auto_dim_symbols import install_auto_dim_symbols
from dace_fortran.builder.context import _Ctx
from dace_fortran.builder.descriptors import (
    DTYPE,
    add_descriptors,
    auto_declare_synth,
    dt,
    emit_declare_transient,
    sdfg_name,
)
from dace_fortran.builder.emit_library import (
    emit_break,
    emit_call,
    emit_copy,
    emit_libcall,
    emit_memset,
    emit_mpi,
    emit_reduce,
    emit_return,
)
from dace_fortran.builder.emit_cfg import (
    emit_assign,
    emit_cond,
    emit_loop,
    emit_symbol_init,
    emit_while,
)
from dace_fortran.builder.emit_tasklet import emit_scalar_assign, emit_tasklet

# Default bridge pass pipeline.  Order matters  --  see ``README.md``.
DEFAULT_PIPELINE = (
    # Lower fir.select_case -> arith.cmp + cf.cond_br BEFORE inline-all.
    # The upstream ``mlir::inlineCall`` mishandles fir.select_case's
    # block-operand remap and segfaults when a callee containing one is
    # inlined.  Pre-lowering side-steps the inliner crash and produces
    # a plain CFG that lift-cf-to-scf turns back into nested scf.if for
    # the bridge to consume.
    "lower-fir-select-case,"
    # Structurize callees BEFORE inlining: an early ``RETURN`` (``if (c)
    # return``) or other in-callee CFG makes the callee a multi-block
    # function, and ``mlir::inlineCall`` then splices those blocks into the
    # call site -- corrupting any structured ``fir.if`` / ``fir.do_loop``
    # region the call sits in (``'fir.if' op expects region #0 to have 0 or 1
    # blocks``).  Lifting cf -> scf first folds the early-return guard into a
    # single-block ``scf.if`` so every callee is single-block and inlines
    # cleanly; the trailing lift-cf-to-scf handles whatever inlining exposes.
    "lift-cf-to-scf,"
    "hlfir-inline-all,"
    # Erase element-scoped alias declares left by inlining scalar-arg
    # procedures (elemental subroutines, most commonly)  --  runs before
    # flatten-structs so the rewrite's designate chains are already
    # single-declare rooted.
    "hlfir-fold-element-aliases,"
    # Replace ``hlfir.associate`` of an ``hlfir.elemental`` (Flang's
    # copy-in temp for noncontiguous slice arguments) with an explicit
    # ``fir.alloca`` + gather DO loop.  After inline-all so the
    # surrounding callee dummy declare aliasing the temp resolves
    # through the materialised hlfir.declare.
    "hlfir-expand-vector-subscript-gather,"
    # Scatter sibling: rewrites ``hlfir.region_assign`` whose lhs region
    # carries an ``hlfir.elemental_addr`` (Fortran ``d(cols) = source``)
    # into an explicit DO loop of per-element scalar assigns.
    "hlfir-expand-vector-subscript-scatter,"
    # Drop private callee bodies once inlined  --  otherwise their
    # declares leak into extract_vars as stray scalars.
    "symbol-dce,"
    # Statically devirtualise resolvable ``fir.dispatch`` /
    # ``fir.select_type`` ops.  The bridge supports CLASS-as-
    # monomorphic-box only  --  surviving polymorphic ops are caught
    # by ``hlfir-reject-polymorphism`` immediately after.
    "fir-polymorphic-op,"
    "hlfir-reject-polymorphism,"
    # Collapse Fortran sequence-association adapters (caller passing a
    # scalar element of an array where the formal expects an
    # explicit-shape array) into an explicit section designate of the
    # parent.  Runs AFTER inline-all (so the inlined callee body's
    # declare-of-converted-ref is visible) and BEFORE flatten-structs
    # (so the section view feeds into the usual designate-rewrite).
    "hlfir-rewrite-sequence-association,"
    # Lift ``type(t), allocatable :: f(:)`` struct members (alloc-array
    # of records  --  ICON's ``p_patch%pprog(jg)`` shape) into top-level
    # companions with a leading runtime-extent dim.  Runs BEFORE
    # flatten-structs so the outer struct sees clean top-level arrays
    # after the lift.  Bails silently when no such member is present;
    # FlattenStructs's opaque-skip for alloc-array-of-records members
    # provides the safety net for un-handled patterns.
    "hlfir-lift-alloc-array-of-records,"
    "hlfir-flatten-structs,"
    # Collapse Fortran ``ptr => target`` rebinds under the strict-no-
    # aliasing assumption: every read or write of the pointer becomes
    # an access to the rebind target's storage.  Runs AFTER
    # flatten-structs so a target like ``s%a`` has already been
    # rewritten to a flat ``s_a`` declare; we trace the rebind through
    # the resulting box+embox chain to that flat declare and rewrite
    # all pointer reads accordingly.  Emits a warning per rewrite to
    # surface the no-alias assumption (Fortran allows aliased pointer
    # access; relying on alias semantics is unsafe under this pass).
    "hlfir-rewrite-pointer-assigns,"
    "hlfir-propagate-shapes,"
    # Lift array-reducing intrinsics (sum/maxval/minval/product/any/all)
    # that appear as INLINE expression operands into a preceding
    # scalar-temp assign.  ``buildExpr`` can't render reductions in a
    # tasklet expression  --  ``out = max(x, MAXVAL(slice))`` would otherwise
    # surface as ``_out = max(_in_x, ?)`` and crash Python ast.parse.
    # After this pass, the lifted ``temp = MAXVAL(slice)`` is a top-
    # level assign the existing reduce-emit dispatch handles, and the
    # outer expression sees a clean scalar load.
    "hlfir-lift-reduction-operands,"
    "hlfir-default-intent,"
    # Lift cf.br / cf.cond_br loops into scf.while so extract_ast can walk them.
    "lift-cf-to-scf,"
    # Constant propagation + fold + CSE after every HLFIR rewrite has
    # exposed as many constants as it will.
    "sccp,canonicalize,cse")

# Multi-file pipeline: flatten cross-file calls into the entry, drop
# the now-dead sibling definitions, fail fast on anything left
# unresolved, then run the usual HLFIR rewrite chain.  The
# ``hlfir-inline-all`` pass needs the per-dialect
# DialectInlinerInterface to be attached to the MLIRContext, which
# the bridge's constructor now does via ``mlir::func::
# registerInlinerExtension`` + ``fir::addFIRInlinerExtension``.
MULTI_FILE_PIPELINE = ("hlfir-inline-all,"
                       "hlfir-fold-element-aliases,"
                       "symbol-dce,"
                       "hlfir-verify-no-unresolved-calls,"
                       "hlfir-flatten-structs,"
                       "hlfir-propagate-shapes,"
                       "hlfir-default-intent,"
                       "lift-cf-to-scf,"
                       "sccp,canonicalize,cse")

# Sympy module-level attributes that turn user-source identifiers into
# parser hazards.  ``test`` / ``doctest`` are ``LazyFunction`` wrappers
# that fail sympify with ``cannot sympify object of type LazyFunction``
# whenever a string referencing them is parsed (interstate-edge
# expressions, memlet subsets, etc.).  The bridge renames any matching
# Fortran identifier to ``program_<name>`` at the SDFG layer; the binding
# emitter restores the original name on the Python wrapper.
_RESERVED_DACE_NAMES = frozenset({"test", "doctest"})

_DACE_NAME_PREFIX = "program_"


def _rename_reserved_collisions(sdfg) -> dict:
    """Walk ``sdfg.arrays`` / ``sdfg.symbols`` for entries whose name
    collides with a reserved sympy attribute and apply a deterministic
    ``program_<name>`` rename via ``sdfg.replace`` (which sweeps every
    memlet, code string, interstate-edge expression, and access node
    in lockstep).  Returns ``{user_fortran_name: dace_name}`` for the
    binding emitter; empty dict when nothing collided.
    """
    renames = {}
    for name in list(sdfg.arrays.keys()) + list(sdfg.symbols.keys()):
        if name in _RESERVED_DACE_NAMES:
            renames[name] = _DACE_NAME_PREFIX + name
    for old, new in renames.items():
        sdfg.replace(old, new)
    return renames


class SDFGBuilder:
    """Walks the HLFIR ASTNode tree and emits a DaCe SDFG.

    Public surface:
        builder = SDFGBuilder("code.hlfir")
        sdfg    = builder.build()

    After construction:
        self.variables   --  full VarInfo list from the bridge.
        self.arrays      --  {name: VarInfo} for rank>0 variables.
        self.symbols     --  {name: VarInfo} for scalars used in shapes / bounds /
                          control-flow conditions (pass 2b-2d of extract_vars).
        self.scalars     --  {name: VarInfo} for remaining scalars.
    """

    DTYPE = DTYPE

    def __init__(self, hlfir_path: str, pipeline: str = DEFAULT_PIPELINE, entry: str | None = None):
        """Parse HLFIR, run the pass pipeline, and classify variables.

        If ``entry`` is set, every other ``func.func`` in the module is
        made private before the pipeline runs so ``symbol-dce`` drops
        them after ``hlfir-inline-all`` has flattened their bodies in.
        Needed when the source contains a module-scope callee that would
        otherwise leak dummy-arg declares into ``extract_vars``.
        """
        self.module = hb.HLFIRModule()
        if not self.module.parse_file(hlfir_path):
            raise RuntimeError(f"Cannot parse {hlfir_path}")

        if entry is not None:
            self.module.set_entry_symbol(entry)

        # Run bridge passes BEFORE extracting variables so assumed-shape
        # dummies pick up real names and the rest of the rewrites have
        # settled.
        if pipeline:
            self.module.run_passes(pipeline)

        self._classify()

    @classmethod
    def from_files(cls, hlfir_paths, *, entry: str, pipeline: str = MULTI_FILE_PIPELINE) -> "SDFGBuilder":
        """Parse and merge several HLFIR files, keep ``entry`` as the only
        public function, verify every remaining call resolves, then run
        the rewrite chain.

        Use this when the entry subroutine and its dependencies live in
        separate ``.hlfir`` files  --  e.g. the ICON multi-module flow where
        each module compiles to its own HLFIR.  ``parse_files`` dedups
        by symbol name so shared external declarations don't conflict.

        Arguments:
            hlfir_paths: list of paths to HLFIR files.  The first file
                         becomes the base; the rest are merged in.
            entry:       mangled Flang symbol (``_QPkernel`` /
                         ``_QMmodPsub``) of the subroutine the SDFG
                         should represent.
            pipeline:    pass pipeline to run before extraction.
        """
        obj = cls.__new__(cls)
        obj.module = hb.HLFIRModule()
        if not obj.module.parse_files(list(hlfir_paths)):
            raise RuntimeError(f"Cannot parse one of {hlfir_paths}")
        obj.module.set_entry_symbol(entry)
        if pipeline:
            obj.module.run_passes(pipeline)
        remaining = obj.module.list_functions()
        if entry not in remaining:
            raise RuntimeError(f"entry '{entry}' dropped by pipeline; remaining: {remaining}")
        obj._classify()
        return obj

    def _classify(self):
        """Shared post-parse extraction: variables + AST + role split."""
        self.variables = self.module.get_variables()
        self.ast = self.module.get_ast()
        # ``view_alias`` participates in the array dictionary so the
        # emitter routes accesses to it normally; ``add_descriptors``
        # registers it via ``sdfg.add_view`` (pointer alias of its
        # source array, no separate storage) and the ``acc`` factory
        # adds a per-state linking memlet so DaCe codegen knows
        # ``dd``'s reads/writes propagate to ``d``.
        self.arrays = {v.fortran_name: v for v in self.variables if v.role in ("array", "view_alias", "section_alias")}
        self.symbols = {v.fortran_name: v for v in self.variables if v.role == "symbol"}
        self.scalars = {v.fortran_name: v for v in self.variables if v.role == "scalar"}
        # Per-axis offset symbols: ``offset_<arr>_d<i>`` is the SDFG
        # symbol every memlet of array ``<arr>`` subtracts on dim ``i``.
        # Populated by ``add_descriptors`` from each VarInfo's
        # ``lower_bounds``.  Values are int (constant-folded by
        # ``sdfg.specialize``), str (substituted with another symbol
        # name), or ``None`` (unknown  --  symbol stays free, caller
        # passes it).
        self.offset_values: dict[str, int | str | None] = {}

    def build(self) -> SDFG:
        """Construct the SDFG, run the unconditional offset-symbol
        specialisation pass, and attach a frozen-signature snapshot.

        The snapshot is later verified by ``codegen.generate_code``
        before any C++ header gets emitted  --  any downstream
        transformation that drifts the argument list will then raise
        rather than silently invalidate a generated Fortran binding.
        """
        self._id_counter = 0
        sdfg = SDFG(sdfg_name(self))
        add_descriptors(self, sdfg)
        # Constant-pool (Flang's ``_QQro.<...>`` globals): for every
        # ``parameter``-attributed declare whose backing global carries
        # a dense init, register the data via ``sdfg.add_constant``.
        # That bakes the values into codegen so the kernel's reads land
        # on the right data instead of an uninitialised transient.  The
        # array descriptor stays in ``sdfg.arrays`` (created by
        # ``add_descriptors``); the constant table just attaches the
        # initial-value tuple to it.
        self._register_constants(sdfg)
        # View aliases need no copy staging here: ``add_descriptors``
        # registers each as an ``sdfg.add_view`` (a typed pointer into
        # the source buffer) and the ``acc`` factory adds the per-state
        # linking memlets lazily on first access, so reads/writes hit
        # the source storage directly.
        ctx = _Ctx(sdfg, self)
        # Seed writable-init transients (module globals the kernel WRITES that
        # carry an init value -- "not really constant") at SDFG entry, before
        # the body runs.  Read-only module data / PARAMETERs stay in the
        # constant pool (see _register_constants).
        self._seed_written_inits(ctx, sdfg)
        self._emit(ctx, self.ast, sdfg)
        ctx.flush(self)
        # User-source identifiers that collide with sympy module-level
        # names (``test`` / ``doctest`` are ``LazyFunction`` attributes
        # that crash sympify; bare letters like ``I`` resolve to
        # ``ImaginaryUnit``).  Rewrite to ``program_<name>`` so DaCe's
        # symbolic parsers stop reaching for the sympy attribute.  The
        # binding emitter consults ``self.dace_name_map`` to expose the
        # original Fortran name on the Python wrapper, so user-side
        # calls (``sdfg(test=arr)``) keep working.
        self.dace_name_map = _rename_reserved_collisions(sdfg)
        # Always-on post-emit substitution.  ``offset_values`` carries
        # two flavours of mapping: int constants (``offset_d_d0 = 50``
        # for ``dimension(50:54)``) and symbol aliases (``offset_d_d0 =
        # "arrsize"`` for ``dimension(arrsize:arrsize+4)``).  They take
        # different paths because ``sdfg.specialize`` only handles
        # constants  --  feeding it a string would land on ``add_constant``
        # and downstream casting tries ``int64("arrsize")`` and
        # ValueError-s.
        # TODO(future): replace the ``specialize`` call with
        # ``sdfg.replace_dict`` so the offset symbols get erased from
        # ``sdfg.symbols`` entirely (they currently linger as bound
        # constants and bloat the symbol table).  An attempt at this
        # broke ``test_fortran_frontend_type_array`` /
        # ``test_fortran_frontend_type_array2`` in ``type_test.py``:
        # for non-default lower bounds (``dimension(7:12)``) the
        # ``replace_dict`` substitution didn't apply uniformly to
        # every memlet subset, leaving raw-Fortran indices that went
        # out-of-bounds against the 0-based flat companion.  Needs a
        # careful audit of which property paths ``replace_dict``
        # walks (vs. what ``specialize`` does in-place via the
        # constants table) before re-trying.
        const_offsets, alias_offsets = {}, {}
        for k, v in self.offset_values.items():
            if v is None:
                continue
            (alias_offsets if isinstance(v, str) else const_offsets)[k] = v
        if const_offsets:
            sdfg.specialize(const_offsets)
        # Symbol-to-symbol aliasing (``offset_d_d0 = arrsize``): rename
        # every reference and drop the now-redundant offset symbol from
        # the SDFG so its signature only carries ``arrsize`` as a free
        # symbol.
        for src, dst in alias_offsets.items():
            sdfg.replace(src, dst)
            if src in sdfg.symbols:
                sdfg.symbols.pop(src)
        # Post-gen cleanups (Stage 4b in dace_fortran/README.md).
        # Run BEFORE the FrozenSignature snapshot so the snapshot
        # captures the post-cleanup signature (matters for the
        # downstream codegen drift check).
        self._run_post_gen_passes(sdfg)
        self._attach_frozen_signature(sdfg)
        # Every Fortran extent stays a required SDFG input; resolve the
        # synthetic ``<arr>_d<i>`` symbols a direct caller omits from
        # the passed arrays (correct extent) or a don't-care default.
        sdfg = install_auto_dim_symbols(sdfg)
        # Validate the SDFG exactly as returned -- after every mutation
        # (post-gen passes, frozen-signature snapshot, auto-dim retype) --
        # so a caller never receives an unvalidated graph.
        sdfg.validate()
        return sdfg

    def _register_constants(self, sdfg: SDFG):
        """Attach Flang's constant-pool data to the SDFG.

        Every ``VarInfo`` with non-empty ``const_data`` represents a
        ``_QQro.<shape>x<dtype>.<counter>`` global  --  the read-only
        backing for an array or scalar literal in the source.  The
        bridge has already added a transient descriptor for it via
        ``add_descriptors``; this hook attaches the dense values so
        DaCe's codegen materialises them into the binary.

        The data widens to ``double`` on the bridge side for
        transport; we narrow to the descriptor's actual dtype here
        and reshape back to the rank-N companion shape.  Scalar
        constants (rank 0, single value) are uncommon  --  Fortran
        ``parameter`` scalars typically inline as ``arith.constant``
         --  but the path supports them with a trivial 1-element array.
        """
        import numpy as np
        from dace.data import Scalar
        for v in self.variables:
            if not v.const_data:
                continue
            # A module global the kernel WRITES is not a read-only constant:
            # it's a writable transient seeded with its init value at SDFG
            # entry (see ``_seed_written_inits``).  A ``constexpr`` here would
            # make the kernel's store to it fail to compile.
            if getattr(v, 'is_written', False):
                continue
            if v.fortran_name not in sdfg.arrays:
                continue
            desc = sdfg.arrays[v.fortran_name]
            np_dtype = desc.dtype.as_numpy_dtype()
            arr = np.asarray(v.const_data, dtype=np.float64).astype(np_dtype)
            # Scalar (rank 0) globals  --  e.g. ``real :: bob = 1`` at
            # module scope  --  pass through as a Python scalar so DaCe's
            # ``framecode.generate_constants`` writes a
            # ``constexpr <T> name = <val>`` (not the array form which
            # tries to ``sym2cpp`` a numpy array and chokes with
            # ``unhashable type: 'numpy.ndarray'``).
            if isinstance(desc, Scalar) or v.rank == 0:
                if arr.size != 1:
                    continue
                sdfg.add_constant(v.fortran_name, arr.item(), desc)
                continue
            # Array globals: reshape from row-major doubles transport
            # to the descriptor's declared shape.
            shape = tuple(int(d) for d in desc.shape)
            if arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape, order='C')
            sdfg.add_constant(v.fortran_name, arr, desc)

    def _seed_written_inits(self, ctx, sdfg):
        """Seed module globals the kernel WRITES that carry an init value
        (``is_written`` + ``const_data``) with that value at SDFG entry.

        These are "not really constant" -- a lazy-init flag like
        ``tables_are_initialized = .false.`` that an init routine later sets
        ``.true.``.  The constant pool would make them read-only ``constexpr``;
        instead they are writable transients (or symbols) seeded once up front:
        a scalar via a tasklet in the entry state, a symbol via an
        interstate-edge assignment.  Rank-0 only -- a written-before-read
        scratch array (a lookup table) needs no seed, and a written array WITH
        an init is not yet handled.
        """
        scalar_inits, symbol_inits = [], []
        for v in self.variables:
            if not (getattr(v, 'is_written', False) and v.const_data and v.rank == 0):
                continue
            val = v.const_data[0]
            is_int = v.dtype.startswith('int') or v.dtype == 'bool'
            expr = str(int(round(val))) if is_int else repr(float(val))
            if v.fortran_name in self.symbols:
                symbol_inits.append((v.fortran_name, expr))
            elif v.fortran_name in self.scalars:
                scalar_inits.append((v.fortran_name, expr))
        if not scalar_inits and not symbol_inits:
            return
        for tgt, expr in scalar_inits:
            ctx.pending.append((tgt, expr))
        ctx.flush_and_ensure(self, sdfg)  # emit scalar seeds into the entry state
        nxt = sdfg.add_state(f"s_{self.nid()}")
        edge = InterstateEdge(assignments=dict(symbol_inits)) if symbol_inits else InterstateEdge()
        sdfg.add_edge(ctx.cur, nxt, edge)
        ctx.cur = nxt

    def _run_post_gen_passes(self, sdfg: SDFG):
        """Run the post-generation cleanup passes that take a freshly-
        emitted bridge SDFG to its canonical shape.  See Stage 4b in
        ``dace_fortran/README.md`` for the pipeline.

        Currently:
            * ``UniqueLoopIterators`` -- rewrites every ``LoopRegion``'s
              loop variable to a globally-unique ``_loop_it_<N>`` symbol
              and propagates the rename through the body.  Enabled with
              ``assign_loop_iterator_post_value=True``: bridge-emitted
              SDFGs land in Fortran callers that read the iterator after
              the loop end (gfortran/ifort/flang convention: one stride
              past the last attained value), so the pass also stages a
              postfix-assignment state for that read.
            * ``replace_length_one_arrays_with_scalars`` -- folds
              every length-1 ``Array`` on the SDFG signature down to a
              true ``Scalar``.  The bridge already emits scalar inputs
              directly as ``Scalar``; this pass cleans up leftover
              length-1 OUTPUTS and any local 1-element transients so
              callers can bind plain ``int`` / ``float`` instead of
              wrapping in a numpy 1-array.
        """
        from dace.transformation.passes import ConvertLengthOneArraysToScalars
        from dace_fortran.builder.scalar_shape_symbol_cleanup import RemoveScalarFortranShapeSymbols
        from dace_fortran.integer_power_exponents import IntegerizePowerExponents
        from dace.transformation.passes.unique_loop_iterators import UniqueLoopIterators

        # Empty-region cleanup: any ControlFlowRegion (LoopRegion,
        # ConditionalBlock branch, the top-level SDFG, etc.) that
        # ended up with zero internal blocks gets a single empty
        # state added.  Validation requires every CFG region to
        # have a defined start block; an empty region triggers
        # "Ambiguous starting block".  Such empties arise legitimately
        # from Fortran source  --  a ``do i = 1, N; <only-stripped-by-
        # flatten>; end do`` whose body became a no-op after
        # AoS+allocatable flattening, an empty IF branch, etc.
        # The empty state is semantically equivalent to the source
        # construct (a loop iterating over a no-op body is still
        # a no-op overall).
        for region in list(sdfg.all_control_flow_regions()):
            if len(list(region.nodes())) == 0:
                region.add_state("empty_body", is_start_block=True)

        uniq_loop_iter_pass = UniqueLoopIterators()
        uniq_loop_iter_pass.assign_loop_iterator_post_value = True
        uniq_loop_iter_pass.apply_pass(sdfg, {})
        # ``transient_only=True``: only fold LOCAL 1-element transients
        # (e.g. accumulators left as length-1 arrays by the bridge).  The
        # signature convention is preserved: ``intent(out)`` / ``inout``
        # scalars stay as length-1 ``Array`` so callers can pass a numpy
        # 1-element buffer to receive the value.  ``intent(in)`` /
        # ``VALUE`` scalars are already emitted as ``Scalar`` directly by
        # ``descriptors.py`` and don't need this pass.
        ConvertLengthOneArraysToScalars(recursive=True, transient_only=True).apply_pass(sdfg, {})

        # A ``Scalar`` has no shape / offset, so the bridge's synthesised
        # ``<s>_d<i>`` / ``offset_<s>_d<i>`` symbols for it are dead; under
        # the current DaCe core a stray free symbol becomes a required
        # call argument, so drop them.
        RemoveScalarFortranShapeSymbols(recursive=True).apply_pass(sdfg, {})

        # Retype integer-valued float ``**`` exponents to ``int`` so
        # codegen uses repeated-multiply ``ipow`` (bit-matching the
        # Fortran reference) rather than libm ``pow``.
        IntegerizePowerExponents().apply_pass(sdfg, {})

    def _attach_frozen_signature(self, sdfg: SDFG):
        """Snapshot ``sdfg.arglist()`` + free symbols into a
        ``FrozenSignature`` and pin it on the SDFG.

        ``kind`` is read off the live SDFG descriptor, not the builder's
        role split: a scalar OUTPUT (``intent(out)`` / ``intent(inout)``)
        registers in ``self.scalars`` but lives on the SDFG as a length-1
        ``Array`` -- the bindings emitter must see ``kind='array'`` so it
        emits ``type(c_ptr), value`` (pointer) instead of a pass-by-value
        scalar binding.  A scalar INPUT (``intent(in)`` / ``VALUE``) lives
        as a true ``Scalar`` and gets ``kind='scalar'``.
        """
        # Local import keeps the binding machinery optional -- plain
        # ``import dace_fortran`` doesn't drag it in.
        from dace import dtypes
        from dace.data import Array, Scalar
        from dace_fortran.bindings.frozen_signature import FrozenArg, FrozenSignature

        # Auto-detected Fortran module-global provenance, keyed by the
        # bridge's short Fortran name.  Populated from every VarInfo
        # the bridge tagged with a ``_QM<mod>E<entity>`` origin (see
        # ``extract_vars.cpp``); consumed both for module-global args
        # below and for free symbols (a scalar module global lifted
        # into a shape / bound).  The binding generator merges this
        # with any hand-authored override map.
        origin_by_name = {
            v.fortran_name: (v.module_origin_mod, v.module_origin_name)
            for v in self.variables if getattr(v, 'module_origin_mod', '') and getattr(v, 'module_origin_name', '')
        }
        module_symbol_origins: dict = {}

        args_list = []
        # Reverse the rename map so we can recover the user-source
        # Fortran name from the SDFG-internal name.  Empty dict when no
        # reserved-name collision fired, so the lookup becomes a no-op.
        dace_to_user = {v: k for k, v in getattr(self, 'dace_name_map', {}).items()}
        for sdfg_name_, desc in sdfg.arglist().items():
            user_key = dace_to_user.get(sdfg_name_, sdfg_name_)
            v = (self.arrays.get(user_key) or self.symbols.get(user_key) or self.scalars.get(user_key))
            _dt = getattr(desc, 'dtype', None)
            if isinstance(_dt, dtypes.opaque) and _dt.ctype == 'MPI_Comm':
                # A Fortran ``integer`` communicator dummy whose SDFG
                # descriptor ``emit_mpi`` retyped to ``opaque(MPI_Comm)``;
                # the binding wrapper does ``MPI_Comm_f2c`` on the
                # integer handle.
                kind = 'mpi_comm'
            elif user_key in self.symbols:
                kind = 'symbol'
            elif isinstance(desc, Scalar):
                kind = 'scalar'
            elif isinstance(desc, Array):
                kind = 'array'
            else:
                kind = 'scalar'
            dtype_obj = getattr(desc, 'dtype', None)
            if isinstance(dtype_obj, dtypes.opaque):
                # ``opaque.to_string()`` is unimplemented in this dace
                # (no ``typename``); the ctype is the stable identity.
                dtype_str = dtype_obj.ctype
            else:
                dtype_str = (getattr(dtype_obj, 'to_string', lambda: str(dtype_obj))()
                             if dtype_obj is not None else '?')
            shape = tuple(str(s) for s in getattr(desc, 'shape', ()))
            origin = origin_by_name.get(user_key)
            if origin is not None:
                module_symbol_origins[sdfg_name_] = origin
            args_list.append(
                FrozenArg(
                    fortran_name=v.fortran_name if v is not None else user_key,
                    sdfg_name=sdfg_name_,
                    kind=kind,
                    dtype=dtype_str,
                    rank=len(shape) if kind == 'array' else 0,
                    shape=shape,
                    intent=(v.intent if v is not None else ''),
                ))
        # Free symbols carrying module-global provenance: a scalar
        # module global the bridge lifted into a shape / bound symbol
        # (no SDFG arg, so the loop above never saw it).  The SDFG
        # symbol name is the bridge's short Fortran name.
        free_syms = tuple(sorted(str(s) for s in sdfg.free_symbols))
        for s in free_syms:
            if s not in module_symbol_origins and s in origin_by_name:
                module_symbol_origins[s] = origin_by_name[s]
        # Module globals that survived as baked constants / seeded
        # transients rather than kwargs: a read-only module datum WITH an
        # initialiser (e.g. ICON's ``i_am_accel_node = .FALSE.``) takes the
        # ``const_data`` path, so it is neither an arglist arg nor a free
        # symbol and the loops above miss it.  Record its provenance too --
        # the binding ``USE``-imports the host value; the baked initialiser
        # is the default when no host override is supplied.
        name_map = getattr(self, 'dace_name_map', {})
        for name, origin in origin_by_name.items():
            module_symbol_origins.setdefault(name_map.get(name, name), origin)
        fs = FrozenSignature(
            entry=sdfg.name,
            mangled=next((v.mangled_name for v in self.arrays.values() if getattr(v, 'mangled_name', '')), sdfg.name),
            args=tuple(args_list),
            free_symbols=free_syms,
            module_symbol_origins=module_symbol_origins,
        )
        sdfg._frozen_signature = fs

    def nid(self) -> int:
        """Globally unique integer.  Shared across ``_Ctx`` instances so
        loop variable names (``jk_0``, ``jc_1``, ``jk_2``, ...) never
        collide.
        """
        i = self._id_counter
        self._id_counter += 1
        return i

    _EMIT_DISPATCH = {
        "assign": emit_assign,
        "loop": emit_loop,
        "while": emit_while,
        "conditional": emit_cond,
        "reduce": emit_reduce,
        "copy": emit_copy,
        "memset": emit_memset,
        "libcall": emit_libcall,
        "mpicall": emit_mpi,
        "call": emit_call,
        "break": emit_break,
        "return": emit_return,
        "declare_transient": emit_declare_transient,
        "symbol_init": emit_symbol_init,
    }

    def _emit(self, ctx: '_Ctx', nodes: list, region):
        """Recursive dispatcher  --  maps each ASTNode.kind to its emitter."""
        for n in nodes:
            fn = self._EMIT_DISPATCH.get(n.kind)
            if fn is not None:
                fn(self, ctx, n, region)
            # ``kind="call"`` dispatches to ``emit_call``: a no-op for
            # an unregistered callee, a CPP tasklet for one registered
            # via ``dace_fortran.external``.

    # Scalar-assign is called from _Ctx.flush; keep it as a method on the
    # builder for that caller's convenience.
    def emit_scalar_assign(self, state, target: str, value: str):
        """Emit ``target = value`` as a scalar assignment in ``state``
        (method form so ``_Ctx.flush`` can call it on the builder)."""
        emit_scalar_assign(self, state, target, value)


def generate_sdfg(path: str = None, *, pipeline: str = None, entry: str = None, hlfir_files=None) -> SDFG:
    """Build an SDFG from one or several HLFIR files.

    Single-file form (back-compat):
        ``generate_sdfg("code.hlfir")``  --  parses + DEFAULT_PIPELINE.

    Multi-file form (ICON-style linked entry):
        ``generate_sdfg(entry="_QPkernel", hlfir_files=[...])``  --  parses
        every file, merges them, drops non-entry siblings, errors on
        unresolved calls, then runs the HLFIR rewrite chain.
    """
    if hlfir_files is not None:
        if entry is None:
            raise ValueError("entry= is required when hlfir_files= is supplied")
        return SDFGBuilder.from_files(
            hlfir_files,
            entry=entry,
            pipeline=(pipeline if pipeline is not None else MULTI_FILE_PIPELINE),
        ).build()
    if path is None:
        raise TypeError("generate_sdfg: pass a path or hlfir_files=[...]")
    return SDFGBuilder(
        path,
        pipeline=(pipeline if pipeline is not None else DEFAULT_PIPELINE),
    ).build()
