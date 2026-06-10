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
    emit_blas,
    emit_break,
    emit_call,
    emit_copy,
    emit_fft,
    emit_fft_interpolate,
    emit_io,
    emit_lapack,
    emit_libcall,
    emit_memset,
    emit_mpi,
    emit_reduce,
    emit_return,
    emit_unsupported_libcall,
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
    # Erase dispatch-table bindings (``fir.dt_entry``) the entry never
    # dynamically invokes, so the symbol-dce below can drop the unreachable
    # (often polymorphic, e.g. ``fir.select_type`` over a ``class(*)`` key)
    # procedure clusters a merged USE-closure drags in only for its types.
    # Runs before symbol-dce / the structurizing passes; a no-op when there is
    # no dispatch table.
    "hlfir-prune-unreachable,"
    # Drop unreachable functions FIRST.  ``set_entry_symbol`` has marked the
    # entry public and every other function private; symbol-dce then removes
    # the private functions the entry never (transitively) calls.  This matters
    # for a merged single-TU build of a large USE-closure (the cross-module
    # inline-everything path): infrastructure modules pulled in only for their
    # types carry features the bridge does not lower (e.g. ``fir.select_type``
    # polymorphic dispatch), and lowering them would fail the pipeline even
    # though the entry's call tree never reaches them.  Removing the dead code
    # up front means the structurizing / lowering passes below only ever see
    # live code.  A no-op for an ordinary single-procedure input.
    "symbol-dce,"
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
    # Delete every ``CALL errore(...)`` / ``CALL finish(...)`` site
    # BEFORE the inliner runs.  These helpers are universally
    # ``IF (ierr <= 0) RETURN`` + ``WRITE`` + ``STOP 1`` -- their error
    # branch is unreachable under any valid input, and ``STOP`` is a
    # noreturn terminator the ``scf`` dialect doesn't model, so
    # ``lift-cf-to-scf`` leaves them multi-block.  Inlining a multi-
    # block callee into the caller's surrounding ``scf`` region
    # crashes flang's ``mlir::inlineCall``.  Stripping the call sites
    # leaves the orphan callee for ``symbol-dce`` to remove.  See
    # ``passes/StripErrorHelpers.cpp`` for the default helper-name
    # list and the ``HLFIR_ERROR_HELPERS`` extension knob.
    "hlfir-strip-error-helpers,"
    # Delete every ``fir.call @_FortranAio*`` whose cookie chain
    # does NOT touch a ``SetFile`` call -- stdout / stderr WRITEs,
    # PRINTs, and the QE stop_clock diagnostic.  File-bound chains
    # (``OPEN (..., FILE='...') ... READ (u, *) y``) are preserved
    # for the AST-extraction-time recognizer at
    # ``bridge/ast/dispatch.cpp::recognizeIoCall``, which maps them
    # to ``dace.libraries.fortran_io`` library nodes.  Same slot as
    # strip-error-helpers (pre-inline) so the cookie-threading chains
    # never reach ``hlfir-inline-all``.  See
    # ``passes/StripRuntimeIo.cpp``.
    "hlfir-strip-runtime-io,"
    # Delete every ``fir.call @_FortranACharacter*`` -- string compare
    # (CharacterCompareScalar1), Trim, Adjust, etc.  Lowered from
    # string-keyed dispatch helpers (QE's ``start_clock(name)`` walks
    # a clock-name table via ``_FortranACharacterCompareScalar1``); the
    # bridge's numerical-equivalence contract does not model character
    # data, and the AST builder's ``leafExpr`` falls through to ``?``
    # on these calls, so they must be elided before AST extraction.
    # See ``passes/StripCharacterRuntime.cpp``.
    "hlfir-strip-character-runtime,"
    "hlfir-inline-all,"
    # Unwrap ``hlfir.eval_in_mem`` blocks into ``fir.alloca`` + body +
    # reads.  flang's HLFIR wraps any array-valued expression that
    # has to be evaluated into pre-allocated memory in eval_in_mem;
    # after ``hlfir-inline-all`` inlines a callee whose return is an
    # array-by-value (graupel's ``update = precip1(...)``, NPB-LU's
    # ``snow_*`` helpers), the inlined body sits inside an
    # eval_in_mem whose result is an ``!hlfir.expr<NxT>`` value the
    # bridge's expression resolver cannot read.  This pass rewrites
    # each eval_in_mem to a plain alloca + body + plain reads so the
    # bridge's existing extract_vars + AST emitter handle it the
    # same way they handle a stack-local Fortran array.  No-op when
    # there are no eval_in_mem ops left.
    "hlfir-unwrap-eval-in-mem,"
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
    # Lift an AoS-of-records-with-pointer-only-members (Graupel's
    # ``TYPE(t_qx_ptr) :: q(N)`` with ``REAL, POINTER :: p(:), x(:,:)``)
    # to flat per-member concat transients with copy-in / copy-out.
    # Without this, ``flatten-structs`` rejects the shape (its docstring
    # says ``Pointer members and AoS-with-allocatable members are still
    # out of scope``) and the bridge emits a 2-D subscript against a
    # scalar-shape descriptor that Python's interstate-edge parser then
    # chokes on.  Runs BEFORE flatten-structs so flatten sees only
    # ordinary flat arrays.
    "hlfir-lift-aos-pointer-records,"
    # Pre-stage the alloc-array-of-records dummy splits BEFORE
    # marshal-external-structs.  ``splitDoubleBufferMembers`` rewrites
    # ``s%prog(nnow)`` element designates to a fresh scalar-struct
    # dummy ``s_prog_nnow``; ``splitMultiDimAoRScalarMembers`` does the
    # same for ``s%X(i,j,k)%v1`` with scalar inner members.  Marshal
    # then sees the call's struct args under the NEW dummy names so
    # the per-member designates it generates land on ``s_prog_nnow%w``
    # (not ``s%prog(nnow)%w``) -- a chain that the scalar-struct
    # flatten below resolves cleanly to the flat leaf.  The full
    # ``hlfir-flatten-structs`` re-runs the splits (idempotent no-op)
    # and adds ``planAndReplaceStructArgs`` on top.
    "hlfir-split-aor-dummies,"
    # Expand the struct argument of a registered external (``keep_external``)
    # call into its individual members, so flatten-structs turns each into the
    # SoA flat the SDFG dataflow uses; the binding emitter re-packs the SoA
    # flats into a local AoS buffer inside the generated C tasklet.  Runs BEFORE
    # flatten-structs so the member designates feed the usual designate-rewrite;
    # a no-op when no external takes a struct.
    "hlfir-marshal-external-structs,"
    "hlfir-flatten-structs,"
    # Tag Fortran 2003 bounds-remapping pointer assignments
    # (``ptr(1:N*K) => target(:, slice)``) with
    # ``hlfir_bridge.bounds_remap_view`` on the LHS pointer declare.
    # Runs BEFORE ``hlfir-rewrite-pointer-assigns`` so the
    # rewriter skips the marked declares (its index-rewriting model
    # can't express a rank reshape).  The actual SDFG-side View
    # emission lives in ``descriptors.py``: it reads the tag, traces
    # the rebox chain to the parent array, and emits
    # ``sdfg.add_view(shape=[total_extent], strides=[1])`` with a
    # fresh ``offset_<ptr>_d0`` symbol bound per surrounding loop
    # iteration via interstate assignment.  See
    # ``passes/MarkBoundsRemapViews.cpp`` and
    # ``tests/bounds_remap_view/`` for the detection contract.
    "hlfir-mark-bounds-remap-views,"
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
    # Classify ``fir.global`` ops as INPUT vs MUTABLE based on whether
    # the IR writes them.  INPUT globals (no in-IR writes -- the caller
    # mutates them from OUTSIDE via the bindings layer) have their
    # init body cleared so ``sccp`` cannot fold their loads to the
    # BSS initializer.  Without this, a Fortran module-level scalar
    # the caller pre-sets (LU's ``dt``, QE's per-call config scalars,
    # ...) gets baked to its in-source initial value and the SDFG
    # silently ignores the runtime kwarg.  Must run AFTER
    # ``hlfir-inline-all`` (so we see inlined writes) and BEFORE
    # ``sccp,canonicalize,cse``.  See
    # ``passes/PreserveMutableGlobals.cpp``.
    "hlfir-preserve-mutable-globals,"
    # Fold ``fir.box_rank`` / ``fir.is_assumed_size`` to constants
    # when the rank-erased dummy traces back to a concrete-rank
    # caller declare.  Without this every ``SELECT RANK`` dispatch
    # in an assumed-rank ``DIMENSION(..)`` callee reaches AST
    # extraction with all branches live (the bridge would see
    # multiple ``hlfir.declare`` ops with the same uniq_name but
    # different ranks).  Must run AFTER ``hlfir-inline-all`` (so
    # the callee body is inlined and the convert chain is visible)
    # and BEFORE ``canonicalize`` (so the cmpi / scf.if / select
    # chain reduces to just the matching branch).
    "hlfir-fold-assumed-rank-queries,"
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
MULTI_FILE_PIPELINE = (
    # Structurize callees BEFORE inlining (mirrors ``DEFAULT_PIPELINE``):
    # an early ``RETURN`` makes a callee multi-block, and ``mlir::
    # inlineCall`` skips multi-block callees per
    # ``passes/InlineAll.cpp`` line 162.  For LU's contained subroutines
    # (``domain`` / ``setcoeff`` / ``ssor`` / etc.), every one was
    # multi-block until ``lift-cf-to-scf`` ran, so the inliner left
    # them all as separate functions and the AST extractor saw only
    # 9 opaque call nodes -- the entire LU body invisible to the
    # SDFG builder.  Lifting first folds each callee into a single
    # scf-wrapped block; the inliner then absorbs every level of the
    # call tree in its existing fixed-point loop.  This is the
    # ``WP-2`` fix; the LU numerical correctness test passes
    # element-wise against the gfortran reference after this lands.
    "lift-cf-to-scf,"
    "hlfir-inline-all,"
    "hlfir-fold-element-aliases,"
    "symbol-dce,"
    "hlfir-verify-no-unresolved-calls,"
    "hlfir-flatten-structs,"
    "hlfir-propagate-shapes,"
    "hlfir-default-intent,"
    "hlfir-preserve-mutable-globals,"
    "hlfir-fold-assumed-rank-queries,"
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


def _global_is_baked_constant(v) -> bool:
    """Mirror of the ``hlfir-preserve-mutable-globals`` rule on the
    Python side.  A module-level Fortran global is "baked" (becomes a
    compile-time constant in the SDFG) iff the caller has no symbol to
    bind it -- which is exactly two cases:

    * PARAMETER constants -- flang marks them with the ``EC`` separator
      before the variable name (``_QM<mod>EC<var>`` / ``_QM<mod>F<func>EC<var>``).
    * Function-scope locals declared with a source-level initialiser
      (``real :: bob = 1`` inside a subroutine) -- flang marks the
      enclosing scope with an uppercase ``F`` segment after the
      leading ``_Q`` (``_QF<func>E<var>`` / ``_QM<mod>F<func>E<var>``).

    Every other module-level initialised global is a caller-overridable
    default (LU ``dt``, a config lookup table the caller may override)
    and surfaces as a kwarg on the SDFG signature.

    Flang lowercases every Fortran identifier; an uppercase ``F`` or
    ``EC`` after ``_Q`` is therefore always a scope / attribute marker
    and never coincides with a module / function / variable name.
    """
    mangled = getattr(v, 'mangled_name', '') or ''
    if not mangled.startswith('_Q'):
        return False
    # Flang's synthetic literal-pool globals back every array / string
    # literal in the source.  Two prefixes:
    #
    #   _QQro.<shape>x<dtype>.<counter>     -- array literal read-only data
    #                                          (see ``_register_constants``
    #                                          docstring and
    #                                          ``bridge/extract_vars.h``
    #                                          line 45 / 118)
    #   _QQclX<hex>                         -- character literal decoded
    #                                          inline in ``bridge/ast/
    #                                          dispatch.cpp`` line 835
    #
    # Neither carries a Fortran-source symbol the caller could bind --
    # always bake.
    if mangled.startswith('_QQro') or mangled.startswith('_QQcl'):
        return True
    tail = mangled[2:]
    return 'EC' in tail or 'F' in tail


def _specialize_symbol(sdfg: SDFG, symbol_name: str, value):
    """Bake a free symbol to a constant value, recursively through nested SDFGs.

    Substitutes ``symbol_name`` with ``value`` in every subset, memlet,
    tasklet, interstate edge, and array descriptor, walks every nested
    SDFG via :meth:`all_sdfgs_recursive`, and strips the symbol from
    each :class:`NestedSDFG` node's ``symbol_mapping``.  The symbol is
    removed from the top-level ``sdfg.symbols`` so the SDFG signature
    sheds the now-redundant entry entirely -- and transformations that
    pattern-match on integer constants in shapes / strides / subsets
    see the literal value instead of a bound symbol.

    A direct port of :func:`dace.sdfg.utils.specialize_symbol` from
    yakup/dev (not yet on d2/FaCe).  Switch to the dace import once it
    lands upstream.

    :param sdfg: The SDFG to specialize.
    :param symbol_name: The symbol name to replace.
    :param value: The constant value to substitute in.
    """
    from dace.sdfg.nodes import NestedSDFG
    val = str(value)
    for sd in list(sdfg.all_sdfgs_recursive()):
        if (symbol_name in sd.symbols or any(str(s) == symbol_name for s in sd.free_symbols)):
            sd.replace_dict({symbol_name: val})
        if symbol_name in sd.symbols:
            sd.remove_symbol(symbol_name)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, NestedSDFG):
            node.symbol_mapping.pop(symbol_name, None)


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
        # Cache the entry name so ``sdfg_name`` can name the SDFG (and
        # therefore the generated ``.so``) after the actual procedure.
        self.entry = entry

        # Snapshot the caller-facing dummy list BEFORE any pass runs --
        # ``hlfir-flatten-structs`` destroys the AoS view of struct dummies,
        # so the only place to read the original interface is here.  Used to
        # auto-derive an ``OriginalInterface`` for the binding emitter.
        self._fortran_interface_raw = self.module.get_fortran_interface(entry or "")

        # Keep every registered external callee (``dace_fortran.external``) as a
        # declaration through ``hlfir-inline-all``: a ``keep_external`` procedure
        # whose Fortran body is present in a merged translation unit would
        # otherwise be inlined into the entry, dragging its implementation (and
        # everything only it reaches) into the lowered code.  No-op when the
        # registry is empty, so ordinary single-procedure builds are unaffected.
        from dace_fortran.external import registered_names
        ext_names = registered_names()
        if ext_names:
            self.module.externalize_symbols(ext_names)
            # Record the same names so hlfir-marshal-external-structs knows
            # which calls take their struct args as array-of-structs and must
            # be expanded to per-member arguments (deep-copy marshalling).
            self.module.set_external_symbols(ext_names)

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
        # Pre-flatten interface snapshot (see ``__init__``).
        obj._fortran_interface_raw = obj.module.get_fortran_interface(entry or "")
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
        # Array-element values promoted to symbols while resolving extents
        # (e.g. ``z_raylfac(nrdmax(jg))`` -> ``__sym_nrdmax_jg``).  Must be read
        # right after ``get_variables`` (which populates them).  ``build`` seeds
        # each from its element read and asserts the element stays constant.
        self.value_symbols = self.module.get_value_symbols()
        self.ast = self.module.get_ast()
        # ``view_alias`` participates in the array dictionary so the
        # emitter routes accesses to it normally; ``add_descriptors``
        # registers it via ``sdfg.add_view`` (pointer alias of its
        # source array, no separate storage) and the ``acc`` factory
        # adds a per-state linking memlet so DaCe codegen knows
        # ``dd``'s reads/writes propagate to ``d``.
        # Role-keyed lookups -- ``builder.arrays`` is the canonical
        # ARRAY (rank>0) dict, ``builder.scalars`` is rank-0, etc.
        #
        # Collision resolution: when multiple VarInfos share the same
        # short ``fortran_name`` -- typically a kernel-entry block-arg
        # ARRAY whose Fortran name happens to match a SCALAR dummy of
        # an inlined helper subroutine (graupel's
        # ``qr(ivec, k_v)`` 2D arg vs an inlined-callee scalar
        # ``qr``) -- the extract_vars Pass-0 disambiguation skips
        # entry-scope block args by design (renaming one would mint
        # a phantom flat-scalar SDFG kwarg).  The colliding callee
        # SCALAR then leaked into ``builder.scalars`` AND the
        # ``builder.arrays`` array, causing emit_tasklet's
        # ``r_arr | r_scl`` token classifier to fire BOTH paths --
        # adding a spurious ``qr[0]`` 1D memlet that failed validation
        # against the 2D ``qr`` shape.
        #
        # Fix: when a name appears in both ARRAY and SCALAR roles,
        # ARRAY wins (the array IS the user-visible top-level entity;
        # the colliding scalar is an inlined-callee remnant whose
        # accesses route to the array via the same designate
        # rewriting that produces the 2D AccessInfo for the
        # surrounding statement).  Detected at builder-init time so
        # downstream code (emit_tasklet, access.py, descriptors.py)
        # never sees the inconsistency.
        array_names = {v.fortran_name for v in self.variables
                        if v.role in ("array", "view_alias", "section_alias")}
        symbol_names = {v.fortran_name for v in self.variables if v.role == "symbol"}
        self.arrays = {v.fortran_name: v for v in self.variables
                        if v.role in ("array", "view_alias", "section_alias")}
        self.symbols = {v.fortran_name: v for v in self.variables
                        if v.role == "symbol" and v.fortran_name not in array_names}
        self.scalars = {v.fortran_name: v for v in self.variables
                        if v.role == "scalar"
                        and v.fortran_name not in array_names
                        and v.fortran_name not in symbol_names}
        # Post-condition: the three role-keyed dicts are disjoint.
        # A name in two of them is a Pass-0 disambiguation gap and
        # surfaces downstream as the spurious-edge / wrong-rank shape
        # described above.  Loud-fail here so the gap is caught at
        # extract time, not at SDFG validation 200 states later
        # with an opaque ``InvalidSDFGEdgeError``.
        _array_keys = set(self.arrays)
        _symbol_keys = set(self.symbols)
        _scalar_keys = set(self.scalars)
        _ax = _array_keys & _scalar_keys
        _ay = _array_keys & _symbol_keys
        _xy = _scalar_keys & _symbol_keys
        if _ax or _ay or _xy:
            collisions = sorted(_ax | _ay | _xy)
            raise RuntimeError(
                f"variable name collision across role-keyed dicts: {collisions[:5]}.  "
                "A VarInfo's Fortran short name appears in more than one of "
                "builder.arrays / builder.scalars / builder.symbols after the "
                "array-wins de-collision -- means Pass-0 disambiguation in "
                "extract_vars.cpp didn't catch a same-name shadow (typically "
                "an inlined-callee scalar dummy whose name matches an outer "
                "array).  Fix in the disambiguation pass; downstream code "
                "assumes the three dicts are disjoint.")
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
        # Seed array-element value-symbols (``__sym_<arr>_<idx>``) before the
        # body, so shapes/memlets that reference them resolve to a value.
        self._seed_value_symbols(ctx, sdfg)
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
        # "arrsize"`` for ``dimension(arrsize:arrsize+4)``).  Both are
        # erased from ``sdfg.symbols``: constants bake into every
        # subset / memlet / shape as a literal ``sympy.Integer`` (which
        # downstream transformations require -- they pattern-match on
        # integer constants, not on a free symbol bound to a constant),
        # aliases get renamed to the source symbol.
        const_offsets, alias_offsets = {}, {}
        for k, v in self.offset_values.items():
            if v is None:
                continue
            (alias_offsets if isinstance(v, str) else const_offsets)[k] = v
        # Snapshot the inferred per-axis offsets onto the SDFG before the
        # specialise pass zeroes their symbols out, so tests / diagnostics
        # can still inspect the inferred values without grepping memlet
        # subsets.  ``sdfg.constants`` no longer carries these entries
        # because ``_specialize_symbol`` substitutes them as literal
        # integers in every subset.
        sdfg._fortran_offset_values = dict(const_offsets)
        # Constant offsets: ``_specialize_symbol`` walks every nested
        # SDFG and strips the symbol from each NestedSDFG node's
        # ``symbol_mapping``, so the symbol leaves the signature
        # entirely.  This was the gap behind the older ``replace_dict``
        # attempt that broke ``type_array`` /
        # ``type_array2`` tests for non-default lower bounds.
        for k, v in const_offsets.items():
            _specialize_symbol(sdfg, k, v)
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
        # Prune SoA companions left orphaned by a marshal-expansion
        # refusal (Phase 2.3.E v2 boundary): with no AccessNode and no
        # memlet referring to them they bloat the signature with kernel
        # parameters the kernel never reads.  Bindings emission stays
        # safe -- the :class:`FlattenPlan` recipes' ``flat_names`` are
        # forwarded as ``binding_names`` keepers, so a member the
        # bridge generated for the bindings ``c_loc`` aliasing path
        # survives even if the SDFG dataflow itself never reads it.
        from dace_fortran.builder.prune_unused_arrays import prune_unused_arrays
        _plan_raw = self.module.get_flatten_plan() or {}
        _binding_keep = {
            f
            for e in (_plan_raw.get('entries') or [])
            for f in (e.get('recipe', {}).get('flat_names') or [])
        }
        prune_unused_arrays(sdfg, binding_names=_binding_keep)
        self._attach_frozen_signature(sdfg)
        # Stash the pre-flatten caller interface + the post-flatten plan so
        # the binding emitter can auto-derive both an ``OriginalInterface``
        # and a ``FlattenPlan`` (built on demand against the final
        # ``sdfg.name``, so a post-build rename is honoured).
        sdfg._fortran_interface_raw = self._fortran_interface_raw
        sdfg._flatten_plan_raw = self.module.get_flatten_plan()
        # Every Fortran extent stays a required SDFG input; resolve the
        # synthetic ``<arr>_d<i>`` symbols a direct caller omits from
        # the passed arrays (correct extent) or a don't-care default.
        sdfg = install_auto_dim_symbols(sdfg)
        # Soundness check for array-element value-symbols: the backing array of
        # every ``__sym_<arr>_<idx>`` must be constant in the symbol's scope
        # (no write would change the value it froze).  Run on the final graph.
        self._check_value_symbols_constant(sdfg)
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
            # Mirror the MLIR-side ``hlfir-preserve-mutable-globals`` rule
            # on the Python side: only globals that the caller can NOT
            # bind get baked into the SDFG constant pool.  Two such
            # shapes -- PARAMETERs (true compile-time constants;
            # flang's mangled marker is the ``EC`` separator before
            # the var name) and routine-local ``SAVE``-init globals
            # (function-scope, flang marks scope with an uppercase
            # ``F`` segment).  Every other module-level initialised
            # global is a caller-overridable default (LU ``dt``, a
            # module lookup table the caller may override) and must
            # surface as a kwarg, not as a baked constant.
            if not _global_is_baked_constant(v):
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
                # Keep a float constant as its typed numpy scalar (e.g.
                # ``np.float32`` for a ``real(4)`` parameter) so the SDFG
                # constant carries its true precision rather than a
                # dtype-erased Python ``float`` widened to double; integers /
                # booleans pass through as Python scalars (round-trip- and
                # ``isinstance(..., int)``-friendly).
                val = arr.reshape(())[()]
                sdfg.add_constant(v.fortran_name, val if np.issubdtype(arr.dtype, np.floating) else val.item(), desc)
                continue
            # Array globals: reshape from row-major doubles transport
            # to the descriptor's declared shape.
            shape = tuple(int(d) for d in desc.shape)
            if arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape, order='C')
            sdfg.add_constant(v.fortran_name, arr, desc)

    def _seed_written_inits(self, ctx, sdfg):
        """Seed globals the kernel WRITES that carry an init value
        (``is_written`` + ``const_data``) with that value at SDFG entry.

        These are "not really constant": a read-only ``constexpr`` would make
        the kernel's store fail to compile, so they become writable transients
        seeded once up front.  Three shapes:

          * scalar  -- a tasklet in the entry state (a lazy-init flag like
            ``tables_are_initialized = .false.``);
          * symbol  -- an interstate-edge assignment;
          * array (an "array of constants" the kernel also mutates, e.g. a
            DATA-statement array later assigned to) -- a writable transient
            whose initial values are unfolded into per-element init tasklets
            in the entry state.  (A read-only array of constants instead bakes
            into a DaCe ``constexpr`` array via ``_register_constants``.)
        """
        import numpy as np
        from dace import Memlet
        from dace.data import Scalar
        scalar_inits, symbol_inits, array_inits = [], [], []
        for v in self.variables:
            if not (getattr(v, 'is_written', False) and v.const_data):
                continue
            if v.rank == 0:
                val = v.const_data[0]
                is_int = v.dtype.startswith('int') or v.dtype == 'bool'
                expr = str(int(round(val))) if is_int else repr(float(val))
                if v.fortran_name in self.symbols:
                    symbol_inits.append((v.fortran_name, expr))
                elif v.fortran_name in self.scalars:
                    scalar_inits.append((v.fortran_name, expr))
            elif v.fortran_name in sdfg.arrays and not isinstance(sdfg.arrays[v.fortran_name], Scalar):
                array_inits.append(v)
        if not scalar_inits and not symbol_inits and not array_inits:
            return
        for tgt, expr in scalar_inits:
            ctx.pending.append((tgt, expr))
        ctx.flush_and_ensure(self, sdfg)  # emit scalar seeds into the entry state
        for v in array_inits:
            desc = sdfg.arrays[v.fortran_name]
            shape = tuple(int(d) for d in desc.shape)
            arr = np.asarray(v.const_data, dtype=np.float64)
            if arr.size != int(np.prod(shape)):
                continue
            # Make it a writable transient (a kwarg would force the caller to
            # supply it) and unfold the dense init into one tasklet per
            # element.  Reshape column-major (``order='F'``) so each logical
            # ``a[i, j]`` write picks the Fortran-storage value the kernel
            # later reads through the array's column-major strides -- the
            # read-only ``_register_constants`` path instead stores the flat
            # ``const_data`` verbatim and relies on the strides, so it uses
            # ``order='C'``; a per-element write must map logical -> value
            # itself.  (Identical for rank 1.)
            desc.transient = True
            arr = arr.reshape(shape, order='F')
            is_int = v.dtype.startswith('int') or v.dtype == 'bool'
            acc = ctx.cur.add_write(v.fortran_name)
            for idx in np.ndindex(*shape):
                val = arr[idx]
                expr = str(int(round(float(val)))) if is_int else repr(float(val))
                tname = "init_%s_%s" % (v.fortran_name, "_".join(str(i) for i in idx))
                t = ctx.cur.add_tasklet(tname, set(), {"_o"}, "_o = %s" % expr)
                ctx.cur.add_edge(t, "_o", acc, None,
                                 Memlet("%s[%s]" % (v.fortran_name, ", ".join(str(i) for i in idx))))
        nxt = sdfg.add_state(f"s_{self.nid()}")
        edge = InterstateEdge(assignments=dict(symbol_inits)) if symbol_inits else InterstateEdge()
        sdfg.add_edge(ctx.cur, nxt, edge)
        ctx.cur = nxt

    def _seed_value_symbols(self, ctx, sdfg):
        """Seed each array-element value-symbol (``__sym_<arr>_<idx>``, minted
        when a runtime-indexed element like ``nrdmax(jg)`` sizes an array) from
        its element read via an interstate edge at SDFG entry, so shapes and
        memlets referencing the symbol resolve to a value rather than a data
        lookup DaCe cannot place in a subset.  Records provenance for the
        constancy check (:meth:`_check_value_symbols_constant`)."""
        import dace
        self._value_symbol_provenance: dict[str, tuple[str, str]] = {}
        seeds = {}
        for vs in getattr(self, "value_symbols", None) or []:
            sym, arr, idx = vs.symbol, vs.array, vs.index_expr
            if arr not in sdfg.arrays:
                continue  # array not on the SDFG surface (trimmed)
            if sym not in sdfg.symbols:
                sdfg.add_symbol(sym, dace.int64)
            # 1-based Fortran element read; assumes lower bound 1, matching the
            # constant-index ``emit_symbol_init`` seeding.
            seeds[sym] = f"{arr}[({idx}) - 1]"
            self._value_symbol_provenance[sym] = (arr, idx)
        if not seeds:
            return
        ctx.flush(self, sdfg)
        ctx.ensure(sdfg)
        dst = sdfg.add_state(f"value_symbol_seed_{self.nid()}")
        sdfg.add_edge(ctx.cur, dst, InterstateEdge(assignments=seeds))
        ctx.cur = dst

    def _check_value_symbols_constant(self, sdfg: SDFG):
        """Additional correctness check (re-runnable after transformations):
        every array whose element was frozen into a value-symbol
        (``__sym_<arr>_<idx>``) must stay constant in that symbol's scope.  A
        write to the backing array anywhere in the assembled SDFG means the
        symbol could hold a stale value, so refuse it.  Conservative: flags any
        write to the array, not just the exact element.

        :raises ValueError: the backing array of a value-symbol is written.
        """
        prov = getattr(self, "_value_symbol_provenance", None)
        if not prov:
            return
        written = set()
        for state in sdfg.all_states():
            for node in state.data_nodes():
                if state.in_degree(node) > 0:
                    written.add(node.data)
        for sym, (arr, idx) in prov.items():
            if arr in written:
                raise ValueError(f"array-element value '{arr}({idx})' is used as a data-access "
                                 f"dimension (promoted to symbol '{sym}'), but '{arr}' is "
                                 f"written in the SDFG -- the symbol would capture a stale "
                                 f"value.  This promotion requires '{arr}' to stay constant "
                                 f"within the scope where '{sym}' is live.")

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
        # USE-SITE-DERIVED symbol set.  Robust against the core-dace change
        # that lifts an unused transient's shape symbols into
        # ``sdfg.free_symbols`` even when no tasklet, memlet, NSDFG
        # mapping, or interstate edge references them: such symbols
        # appear in ``free_symbols`` but never in ``needed`` here, so
        # downstream consumers (the diagnostic, the frozen-signature
        # snapshot, the ``arglist`` consumer) can filter them out.
        # Pre-flight: an AccessNode whose ``data`` field isn't registered
        # in ``sdfg.arrays`` -- e.g. the bare struct base name surfacing
        # from an unflattened ``s%X(...)%Y`` chain that landed as an
        # access node -- KeyErrors inside ``state.used_symbols`` when
        # ``n.desc(sdfg)`` looks the name up.  Surface it directly so
        # ``_collect_needed_symbols`` doesn't trip on the same lookup.
        self._diagnose_unresolved_access_nodes(sdfg)
        needed_syms = self._collect_needed_symbols(sdfg)
        self._diagnose_unresolved_free_symbols(sdfg, needed_syms)
        # Pre-add placeholder entries for any leaked-but-unused free
        # symbol so ``sdfg.arglist()`` doesn't ``KeyError`` on
        # ``self.symbols[k]``.  Restored after the loop; the leaked
        # entries never enter ``args_list`` because the filter below
        # drops them.
        from dace import dtypes as _dtypes
        leaked_syms = [
            k for k in sdfg.free_symbols
            if k not in sdfg.symbols and k not in sdfg.arrays and not k.startswith('__dace') and k not in needed_syms
        ]
        for k in leaked_syms:
            sdfg.symbols[k] = _dtypes.int64
        try:
            arglist_items = list(sdfg.arglist().items())
        finally:
            for k in leaked_syms:
                sdfg.symbols.pop(k, None)
        for sdfg_name_, desc in arglist_items:
            if sdfg_name_ in leaked_syms:
                continue  # leaked-but-unused; not a real argument
            user_key = dace_to_user.get(sdfg_name_, sdfg_name_)
            v = (self.arrays.get(user_key) or self.symbols.get(user_key) or self.scalars.get(user_key))
            _dt = getattr(desc, 'dtype', None)
            if sdfg_name_ in ('dace_user_comm', 'dace_user_comm_size'):
                # SDFG free symbols seeded by ``emit_mpi._install_user_pgrid``
                # -- the bindings wrapper sources their values by calling
                # ``MPI_Comm_f2c`` + ``MPI_Comm_size`` on the original
                # Fortran integer communicator dummy (recorded on
                # ``sdfg._fortran_user_comm_source``) and threads them
                # through ``dace_init_<entry>`` so the pgrid's
                # ``MPI_Cart_create`` runs with the user's comm as
                # parent.  Skip from ``args_list`` -- they belong in the
                # init-only path the bindings handle via
                # ``free_symbols``.
                continue
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
                    is_written=bool(getattr(v, 'is_written', False)),
                ))
        # Free symbols carrying module-global provenance: a scalar
        # module global the bridge lifted into a shape / bound symbol
        # (no SDFG arg, so the loop above never saw it).  The SDFG
        # symbol name is the bridge's short Fortran name.  Source the
        # set from ``sdfg.free_symbols`` filtered against the
        # use-site-derived ``needed_syms`` so leaked unused-transient
        # shape symbols don't bloat the signature.
        free_syms = tuple(sorted(str(s) for s in sdfg.free_symbols if s not in leaked_syms))
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
            user_comm_source=getattr(sdfg, '_fortran_user_comm_source', None),
        )
        sdfg._frozen_signature = fs

    def _diagnose_unresolved_access_nodes(self, sdfg: SDFG):
        """Raise if any ``AccessNode`` references a ``data`` field that
        isn't registered in ``sdfg.arrays``.

        Same dominant cause as :meth:`_diagnose_unresolved_free_symbols`:
        the bridge collapsed an unflattened struct-member access chain
        (e.g. ``s%X(...)%Y``) to the bare struct base name and landed
        the result as an access node.  Walked up front because the
        downstream dace walker ``n.desc(sdfg)`` raises an opaque
        ``KeyError`` on the first unresolved name with no use-site
        information attached.

        :param sdfg: The SDFG to walk.
        :raises RuntimeError: If at least one access node references an
            unregistered name.
        """
        from dace.sdfg.nodes import AccessNode
        bad: list = []
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, AccessNode) and n.data not in sdfg.arrays:
                    bad.append((state.label, n.data))
                    if len(bad) >= 5:
                        break
            if len(bad) >= 5:
                break
        if not bad:
            return
        bullet = '\n  '.join(f'access[{lbl}] data={d!r}' for lbl, d in bad)
        raise RuntimeError(f'unresolved access-node data field(s) in SDFG (not in '
                           f'``sdfg.arrays``):\n  {bullet}\n\n'
                           'Most likely cause: the bridge collapsed an unflattened '
                           'struct-member access chain (e.g. ``s%X(...)%Y``) to the '
                           'bare struct base name and landed the result as an access '
                           'node.  Common pattern: an alloc-array-of-records member '
                           '(``box<heap<array<? x record>>>``) whose inner '
                           'record-element members are read as scalars -- the flatten '
                           'pass skips the member and the access chain dead-ends at '
                           'the struct root.')

    def _collect_needed_symbols(self, sdfg: SDFG) -> set:
        """Return the set of symbol names ACTUALLY referenced at a use
        site in the SDFG.

        Equivalent to ``sdfg.free_symbols`` except for the
        array-descriptor sweep at the end of
        :meth:`dace.SDFG._used_symbols_internal`, which adds the shape /
        stride / offset symbols of EVERY array -- including unused
        transients the SDFG never accesses.  Here we restrict that
        sweep to arrays that appear as an ``AccessNode``'s data or as a
        memlet's ``data`` field, so a transient whose shape symbol is
        never referenced anywhere doesn't bloat the signature.

        Implemented on top of each node's / region's / edge's own
        ``free_symbols`` (the dace public API), so no string parsing or
        whitelist is required.

        :param sdfg: SDFG to walk.
        :returns: A set of symbol names (str) referenced at some use
            site, with array names removed.
        """
        from dace.sdfg.nodes import AccessNode
        from dace.sdfg.state import ControlFlowRegion
        # Defined-at-SDFG: array names + module constants are not free
        # symbols.  Seed ``defined_syms`` with them so the walker
        # excludes them from the free set.
        defined = set(sdfg.arrays.keys()) | set(sdfg.constants_prop.keys())
        # Call the parent class's walker (``ControlFlowRegion``)
        # directly to get the correct block-scope handling (LoopRegion
        # loop_variables, MapEntry params, interstate-edge LHS
        # assignments) WITHOUT triggering the SDFG override's
        # array-descriptor sweep at the bottom of
        # ``SDFG._used_symbols_internal`` -- which adds every array's
        # shape / stride symbols regardless of whether the array is
        # referenced anywhere.
        free_syms, defined_syms, _ = ControlFlowRegion._used_symbols_internal(sdfg,
                                                                              all_symbols=True,
                                                                              defined_syms=set(defined),
                                                                              free_syms=set(),
                                                                              used_before_assignment=set(),
                                                                              with_contents=True)
        # SDFG-declared symbols are part of the free set when
        # ``all_symbols=True`` (mirrors the SDFG override at
        # ``sdfg.py:3099``).
        free_syms |= set(sdfg.symbols.keys())
        free_syms -= defined_syms
        # USED-array shape / stride / offset symbols.  Restrict to
        # arrays actually referenced by an AccessNode or a memlet
        # ``data`` field; an unused transient contributes nothing.
        used_arrays: set = set()
        for state in sdfg.all_states():
            for n in state.nodes():
                if isinstance(n, AccessNode):
                    used_arrays.add(n.data)
            for edge in state.edges():
                m = edge.data
                if m is not None and getattr(m, 'data', None):
                    used_arrays.add(m.data)
        for name in used_arrays:
            arr = sdfg.arrays.get(name)
            if arr is None:
                continue
            free_syms |= {str(s) for s in arr.used_symbols(all_symbols=True)}
        return (free_syms - defined_syms) - set(sdfg.arrays.keys())

    def _diagnose_unresolved_free_symbols(self, sdfg: SDFG, needed: set):
        """Surface a precise error before ``sdfg.arglist()`` raises an
        opaque ``KeyError``.

        ``sdfg.arglist`` looks every free symbol up in ``sdfg.symbols`` and
        raises a bare ``KeyError`` on the first missing name -- with no
        indication of where the symbol is referenced or why it's unresolved.
        The dominant cause is the bridge collapsing an unflattened
        struct-member access chain (e.g. ``s%X(...)%Y``) to the bare struct
        base name in ``expressions.cpp::buildExpr`` -- the chain dead-ends
        at the struct dummy because flatten-structs left the member
        untouched (typically an alloc-array-of-records inner member).

        The check fires only on symbols ACTUALLY USED at some site (so an
        unused-transient shape symbol leaking through
        ``sdfg.free_symbols`` is filtered out).  For each unresolved
        symbol, point at a representative tasklet / memlet that
        references it.

        :param sdfg: The freshly built SDFG, immediately before
            ``arglist()`` is consulted.
        :param needed: The use-site-derived symbol set produced by
            :meth:`_collect_needed_symbols`.
        :raises RuntimeError: If any actually-used symbol is neither
            registered on the SDFG nor a dace-internal name.
        """
        import re
        from dace.sdfg.nodes import Tasklet, NestedSDFG
        unresolved = sorted(k for k in needed
                            if k not in sdfg.symbols and k not in sdfg.arrays and not k.startswith('__dace'))
        if not unresolved:
            return
        # Find up to three representative use sites for the first symbol so
        # the user sees the exact tasklet / memlet that's referencing the
        # bare name.  Word-boundary match to avoid substring noise (e.g. a
        # symbol ``p_patch`` would otherwise match ``p_patch_nlev``).
        target = unresolved[0]
        pat = re.compile(rf'(?<![A-Za-z0-9_]){re.escape(target)}(?![A-Za-z0-9_])')
        examples: list = []
        for state in sdfg.all_states():
            for edge in state.edges():
                m = edge.data
                if m is None or not getattr(m, 'data', None):
                    continue
                subset_str = f'{m.subset}'
                if pat.search(subset_str):
                    examples.append(f'memlet[{state.label}] data={m.data} '
                                    f'subset={subset_str[:160]}')
                    if len(examples) >= 3:
                        break
            if len(examples) >= 3:
                break
            for n in state.nodes():
                if isinstance(n, Tasklet) and pat.search(n.code.as_string):
                    examples.append(f'tasklet[{state.label}::{n.label}] '
                                    f'code={n.code.as_string[:160]}')
                    if len(examples) >= 3:
                        break
                if isinstance(n, NestedSDFG):
                    for k, v in n.symbol_mapping.items():
                        if pat.search(str(v)):
                            examples.append(f'nsdfg[{state.label}::{n.label}] '
                                            f'symbol_mapping[{k}]={v}')
                            if len(examples) >= 3:
                                break
            if len(examples) >= 3:
                break
        hint = ('Most likely cause: the bridge collapsed an unflattened struct-'
                'member access chain (e.g. ``s%X(...)%Y``) to the bare struct '
                'base name.  Common pattern: an alloc-array-of-records member '
                '(``box<heap<array<? x record>>>``) whose inner record-element '
                'members are read as scalars -- the flatten pass skips the '
                'member and the access chain dead-ends at the struct root.')
        bullet = '\n  '.join(examples) if examples else '(no use site found)'
        raise RuntimeError(f'unresolved free symbol(s) in SDFG: {unresolved[:5]}'
                           f'{"" if len(unresolved) <= 5 else f" (+{len(unresolved) - 5} more)"}'
                           f'\nfirst representative use site(s) for {target!r}:\n  {bullet}'
                           f'\n\n{hint}')

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
        "iocall": emit_io,
        "fftcall": emit_fft,
        "blascall": emit_blas,
        "lapackcall": emit_lapack,
        "fft_interpolate": emit_fft_interpolate,
        "unsupported_libcall": emit_unsupported_libcall,
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
