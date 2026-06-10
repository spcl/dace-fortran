// ============================================================================
// LiftReductionOperands.cpp  --  pre-lower reduction intrinsics that appear as
// inline expression operands.
// ============================================================================
//
// Motivation:
//     The bridge's ``buildExpr`` (expressions.cpp) handles scalar-shape
//     ops (arith.*, math.*, hlfir.designate, fir.load, ...) but has no
//     case for array-reducing intrinsics  --  ``hlfir.sum`` / ``maxval`` /
//     ``minval`` / ``product`` / ``any`` / ``all``.  Those map to scalar
//     values but can't be rendered as a tasklet expression: a tasklet
//     reads scalars, not array slices.
//
//     The dispatcher in ``dispatch.cpp`` already handles the top-level
//     case: ``target = MAXVAL(arr)`` routes to ``buildReduceNode`` /
//     ``buildSectionReduceAssign`` / ``buildElementalAnyAllReduce``.  The
//     gap is reductions used as INLINE operands:
//
//         max_vcfl_dyn = MAX(p_diag%max_vcfl_dyn, MAXVAL(vcflmax(s:e)))
//
//     ``buildExpr`` returns ``"?"`` for the inner ``MAXVAL``, the resulting
//     tasklet code ``_out = max(_in_..., ?)`` fails Python ``ast.parse``,
//     and the SDFG can't build.
//
// What the pass does:
//     For each ``hlfir.assign`` in the module, walk its RHS subtree for
//     any reduction op that is NOT the immediate RHS.  For each such
//     "nested" reduction:
//
//         1. Insert ``%tmp = fir.alloca T`` + ``%tmp_decl = hlfir.declare``
//            in the function entry, where T is the reduction's scalar
//            result type.
//         2. Insert ``hlfir.assign <reduction_op_result> to %tmp_decl#0``
//            immediately BEFORE the consuming assign.
//         3. Replace uses of the reduction op's result in the RHS subtree
//            with ``fir.load %tmp_decl#0``.
//
//     After this pass:
//         - The lifted ``temp = MAXVAL(slice)`` is a top-level reduction
//           assign  --  ``buildSectionReduceAssign`` handles it.
//         - The outer ``max_vcfl_dyn = MAX(p_diag_..., load(temp))``
//           sees only a scalar load  --  the existing buildExpr arith.maxnumf
//           handler renders it correctly.
//
// Pipeline position:
//     After ``hlfir-flatten-structs`` (so designate chains on flattened
//     companions are already rewritten) and BEFORE the AST extractor's
//     dispatch (``buildAssignNode`` / ``buildExpr``).  Insertion point in
//     the bridge's DEFAULT_PIPELINE: directly before
//     ``hlfir-default-intent`` is fine.
//
// Out of scope:
//     * ``hlfir.count``  --  already routed through a libcall in the dispatch
//       table; that codepath supports inline use via the libcall's emit
//       path.  If it ever surfaces as a problem, fold in here.
//     * Reductions whose source is an ``hlfir.elemental`` (a compound
//       boolean expression).  The dispatcher's Mode-C path already
//       materialises a transient mask before calling the reduce; that
//       same path covers the lifted top-level case here.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// True iff ``op`` is one of the array-reducing intrinsics that
/// ``buildExpr`` cannot render inline.  ``hlfir.count`` is excluded  --
/// the dispatcher routes it through ``CountLibraryNode`` which handles
/// inline use via the libcall emit path.
static bool isReductionOp(mlir::Operation *op) {
  if (!op) return false;
  return mlir::isa<hlfir::SumOp, hlfir::ProductOp, hlfir::MinvalOp,
                   hlfir::MaxvalOp, hlfir::AnyOp, hlfir::AllOp>(op);
}

/// True iff ``op`` is one of the dense-linalg intrinsics that the
/// dispatcher routes through a libcall (``hlfir.matmul`` /
/// ``hlfir.matmul_transpose`` / ``hlfir.transpose`` /
/// ``hlfir.dot_product``).  ``buildExpr`` returns ``?`` when these
/// appear nested inside a larger expression (e.g.
/// ``MATMUL(TRANSPOSE(a), q) / tpi`` -- the division consumes the
/// matmul result element-wise, the matmul itself never reaches the
/// libcall dispatcher).  Same lift-to-temp strategy as the reduction
/// path, but the temp is an array (or a scalar for ``dot_product``).
///
/// QE's ``vcut_get`` (and ``vexx_bp_k_gpu`` callees) trip this:
/// ``i_real = (MATMUL(TRANSPOSE(vcut % a), q)) / tpi`` lowers to an
/// ``hlfir.matmul_transpose`` whose result is consumed by an
/// ``hlfir.elemental`` (the per-element ``/`` over the rank-1
/// matmul result).  The bridge's elemental walker tries to render
/// the matmul value inline, fails, and emits ``?`` into the
/// tasklet body.  Lifting it to a top-level
/// ``temp = MATMUL_TRANSPOSE(...)`` assign lets the libcall
/// machinery materialise the GEMM (with the transpose flag, no
/// separate ``Transpose`` libcall on the input matrix) into an
/// explicit transient that the elemental can then load
/// element-by-element.
static bool isLiftableLinalgOp(mlir::Operation *op) {
  if (!op) return false;
  return mlir::isa<hlfir::MatmulOp, hlfir::MatmulTransposeOp,
                   hlfir::TransposeOp, hlfir::DotProductOp>(op);
}

/// True iff ``op`` is an op the pass should lift -- reduction or
/// liftable linalg.
static bool isLiftableOp(mlir::Operation *op) {
  return isReductionOp(op) || isLiftableLinalgOp(op);
}

/// Find every liftable op transitively used by ``rootOp`` (the RHS of
/// an assign) -- except the rootOp itself.  "Liftable" covers both
/// array reductions (sum/product/min/max/any/all) and dense-linalg
/// libcalls (matmul/matmul_transpose/transpose/dot_product); see
/// ``isLiftableOp``.  Returns them in postorder so callers process
/// inner ops before outer (lifting an inner ``matmul`` to a temp
/// before its consumer is rewritten avoids dangling references in
/// the outer rewrite).
static void collectNestedLiftable(
    mlir::Operation *rootOp, llvm::SmallVectorImpl<mlir::Operation *> &out) {
  if (!rootOp) return;
  llvm::SmallVector<mlir::Operation *, 8> stack;
  llvm::SmallPtrSet<mlir::Operation *, 16> seen;
  for (auto v : rootOp->getOperands())
    if (auto *def = v.getDefiningOp())
      if (seen.insert(def).second) stack.push_back(def);
  while (!stack.empty()) {
    auto *op = stack.pop_back_val();
    if (isLiftableOp(op)) out.push_back(op);
    for (auto v : op->getOperands())
      if (auto *def = v.getDefiningOp())
        if (seen.insert(def).second) stack.push_back(def);
  }
}

struct LiftReductionOperandsPass
    : public mlir::PassWrapper<LiftReductionOperandsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftReductionOperandsPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-lift-reduction-operands";
  }
  llvm::StringRef getDescription() const final {
    return "Lift array-reducing intrinsics that appear as inline "
           "expression operands into a preceding scalar-temp assign, "
           "so the AST extractor sees only top-level reductions and "
           "scalar loads in the consuming expression.";
  }

  void runOnOperation() override {
    // Counter for unique temp names per function.
    llvm::DenseMap<mlir::func::FuncOp, unsigned> liftCounter;

    // Two-pass: collect first, mutate after  --  modifying the IR
    // mid-walk would invalidate iterators.
    struct Job {
      hlfir::AssignOp consumer;
      mlir::Operation *redOp;
    };
    llvm::SmallVector<Job, 16> jobs;

    getOperation().walk([&](hlfir::AssignOp assign) {
      auto rhs = assign.getRhs();
      auto *rhsOp = rhs.getDefiningOp();
      if (!rhsOp) return;
      // If the RHS itself is a liftable op, the dispatcher already
      // handles it  --  leave alone.  Only lift NESTED ones.
      llvm::SmallVector<mlir::Operation *, 4> nested;
      collectNestedLiftable(rhsOp, nested);
      for (auto *r : nested) jobs.push_back({assign, r});
    });

    for (auto &job : jobs) lift(job.consumer, job.redOp, liftCounter);
  }

  /// Materialise a temp local for the liftable op's result, emit
  /// ``hlfir.assign <op> to <local>`` before the consuming assign,
  /// and rewrite the consuming RHS to read from the local.  Scalar
  /// results get a ``fir.alloca`` + ``fir.load`` pair; array
  /// results (``hlfir.matmul`` / ``transpose`` / dim-reduction)
  /// get a ``fir.alloca !fir.array<NxT>`` + ``hlfir.declare`` and
  /// uses are rewritten to the declare's box result.
  void lift(hlfir::AssignOp consumer, mlir::Operation *redOp,
            llvm::DenseMap<mlir::func::FuncOp, unsigned> &liftCounter) {
    auto func = consumer->getParentOfType<mlir::func::FuncOp>();
    if (!func) return;
    if (redOp->getNumResults() != 1) return;
    auto resTy = redOp->getResult(0).getType();

    // Classify the result shape: scalar, FIR-typed array, or
    // HLFIR-expr array.  Scalar results land in the original
    // alloca + fir.load pattern.  Array results need an HLFIR
    // associate (the FIR allocate-and-declare pattern matches a
    // Fortran-source local) so the consuming hlfir.elemental can
    // load element-by-element.
    bool isScalar = true;
    int64_t arrayRank = 0;
    mlir::Type arrayEltTy;
    if (mlir::isa<fir::SequenceType>(resTy)) {
      isScalar = false;
      auto seq = mlir::cast<fir::SequenceType>(resTy);
      arrayRank = seq.getDimension();
      arrayEltTy = seq.getEleTy();
    } else if (auto exprTy = mlir::dyn_cast<hlfir::ExprType>(resTy)) {
      if (!exprTy.isScalar()) {
        isScalar = false;
        arrayRank = exprTy.getShape().size();
        arrayEltTy = exprTy.getElementType();
      }
    }

    // Array-result lift: previously attempted via ``fir.alloca +
    // hlfir.declare + hlfir.assign + hlfir.as_expr`` but the
    // downstream ``buildLibCallNode``'s ``traceToDecl`` does not
    // walk through ``hlfir.as_expr``, leaving the libcall source
    // name empty.  Until that gap is closed, only EMIT THE LOUD
    // ERROR for matmul/transpose ops whose result feeds a
    // non-liftable consumer (an arith op, an hlfir.elemental's
    // hlfir.apply, etc.) -- i.e. the actual inline-in-expression
    // case the bridge can't render.
    //
    // Skip the error when every user of the op's result is ANOTHER
    // liftable op (e.g. ``hlfir.matmul(hlfir.transpose(A), q)``:
    // the transpose's only user is the matmul, which is itself a
    // liftable op and will be processed by the libcall dispatcher
    // as a whole-assign ``MATMUL(TRANSPOSE(...))``).  Same for the
    // consuming ``hlfir.assign`` (if the lifted op IS the
    // assign's direct RHS, the dispatcher already handles it --
    // collectNestedLiftable would not have collected it but the
    // safety check stays explicit).
    if (!isScalar) {
      if (isLiftableLinalgOp(redOp)) {
        bool allUsersHandled = true;
        for (auto *user : redOp->getResult(0).getUsers()) {
          if (isLiftableOp(user)) continue;
          if (mlir::isa<hlfir::AssignOp>(user)) continue;
          // ``hlfir.destroy`` is a cleanup marker; not a real
          // consumer.
          if (user->getName().getStringRef() == "hlfir.destroy") continue;
          allUsersHandled = false;
          break;
        }
        if (!allUsersHandled) {
          redOp->emitError()
              << "hlfir-lift-reduction-operands: inline "
              << redOp->getName().getStringRef()
              << " inside a larger expression not yet supported.  "
              << "Workaround: assign the linalg result to a Fortran "
              << "local first ``tmp = MATMUL(TRANSPOSE(A), q); res = "
              << "tmp / scalar``, then the libcall dispatcher routes "
              << "the whole-assign matmul through its GEMM lib node "
              << "(with the transpose flag).  TODO: extend "
              << "traceToDecl to peel ``hlfir.as_expr`` (or wire "
              << "hlfir-optimized-bufferization upstream of this "
              << "pass) to lift inline matmul automatically.";
          signalPassFailure();
        }
      }
      return;
    }

    unsigned gid = liftCounter[func]++;
    auto loc = redOp->getLoc();
    auto *ctx = func.getContext();

    // Create the temp local at the function entry block  --  putting
    // it inline at the consuming assign's location works too, but
    // hoisting to entry keeps the pattern uniform with how flang
    // emits other Fortran-source ``REAL :: tmp`` locals.
    mlir::OpBuilder b(&func.front(), func.front().begin());
    auto allocaTy = fir::ReferenceType::get(resTy);
    auto alloca = b.create<fir::AllocaOp>(loc, resTy);
    std::string uniqName = "_QQred_lift_" + std::to_string(gid);
    mlir::NamedAttrList attrs;
    attrs.append("uniq_name", mlir::StringAttr::get(ctx, uniqName));
    // operandSegmentSizes for hlfir.declare: memref + (no shape) +
    // (no typeparams) + (no dummy_scope).
    attrs.append("operandSegmentSizes", b.getDenseI32ArrayAttr({1, 0, 0, 0}));
    auto decl =
        b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{allocaTy, allocaTy},
                                   mlir::ValueRange{alloca.getResult()}, attrs);

    // Emit the lifted assign and load IMMEDIATELY AFTER the
    // reduction op  --  placing them at the consumer's location would
    // put the load AFTER existing uses of the reduction (e.g.
    // ``arith.cmpf %scalar, %maxval`` followed by
    // ``arith.select`` followed by the assign), and rewriting those
    // earlier uses to reference the load would violate dominance.
    // After-the-reduction placement keeps the load before every
    // existing use; the reduction op itself stays at its original
    // position and the new ``hlfir.assign`` plus ``fir.load`` form
    // a tight pair right behind it.  The dispatcher then sees the
    // lifted assign as a top-level ``temp = REDUCTION(...)`` and
    // routes through the existing reduce-emit machinery; consuming
    // sites read the scalar load uniformly.
    b.setInsertionPointAfter(redOp);
    auto liftedAssign =
        b.create<hlfir::AssignOp>(loc, redOp->getResult(0), decl.getResult(0));
    auto load = b.create<fir::LoadOp>(loc, decl.getResult(0));

    // Replace every existing use of ``redOp`` with the load,
    // EXCEPT the just-emitted ``hlfir.assign`` (which intentionally
    // takes the reduction's original result as its source).
    llvm::SmallPtrSet<mlir::Operation *, 4> exceptions{liftedAssign};
    redOp->getResult(0).replaceAllUsesExcept(load.getResult(), exceptions);
  }

  /// Materialise an inline ``hlfir.matmul`` / ``transpose`` / dim-
  /// reducing intrinsic whose result is an array.  Emits a
  /// ``fir.alloca !fir.array<...x T>`` + ``hlfir.declare`` (matches
  /// the FIR shape Flang emits for a Fortran-source local), then
  /// ``hlfir.assign <op-result> to <declare-result>``, and rewrites
  /// every use of the op's result to read from the declare's box
  /// (#0) result.  The declare's #0 has the SAME box / ref type the
  /// consuming ``hlfir.elemental`` (or other array consumer) expects
  /// when the operand came from a normal Fortran-source array
  /// declaration -- so no per-consumer rewrite is needed beyond
  /// ``replaceAllUsesExcept``.
  void liftArrayResult(
      hlfir::AssignOp consumer, mlir::Operation *redOp,
      llvm::DenseMap<mlir::func::FuncOp, unsigned> &liftCounter,
      int64_t rank, mlir::Type eltTy) {
    auto func = consumer->getParentOfType<mlir::func::FuncOp>();
    if (!func || rank <= 0 || !eltTy) return;

    unsigned gid = liftCounter[func]++;
    auto loc = redOp->getLoc();
    auto *ctx = func.getContext();

    // Determine the shape from the op's first operand for matmul /
    // transpose; for dim-reductions the shape would need to be
    // computed from the source array minus the reduced dim, which
    // hlfir.expr<NxT> already encodes -- so we read the static
    // dims from the result type when we have them.  When a dim is
    // dynamic (``?``), we cannot pre-allocate a static array; bail
    // and leave the inline reference for the downstream emitter to
    // either handle or surface the ``?`` placeholder.
    llvm::SmallVector<int64_t, 4> shape;
    auto resTy = redOp->getResult(0).getType();
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(resTy)) {
      for (auto d : seq.getShape()) shape.push_back(d);
    } else if (auto exprTy = mlir::dyn_cast<hlfir::ExprType>(resTy)) {
      for (auto d : exprTy.getShape()) shape.push_back(d);
    } else {
      return;
    }
    for (auto d : shape)
      if (d == mlir::ShapedType::kDynamic ||
          d == fir::SequenceType::getUnknownExtent())
        return;  // dynamic shape -- can't pre-allocate statically.

    // ``fir.alloca !fir.array<N x T>`` + ``hlfir.declare`` at the
    // function entry block -- mirrors Flang's lowering for a
    // Fortran-source local of the same shape, so downstream passes
    // see a normal source-local declare to walk.
    mlir::OpBuilder b(&func.front(), func.front().begin());
    auto arrTy = fir::SequenceType::get(shape, eltTy);
    auto refTy = fir::ReferenceType::get(arrTy);
    auto alloca = b.create<fir::AllocaOp>(loc, arrTy);
    std::string uniqName = "_QQlift_linalg_" + std::to_string(gid);
    mlir::NamedAttrList attrs;
    attrs.append("uniq_name", mlir::StringAttr::get(ctx, uniqName));
    // Build a shape value for hlfir.declare -- one fir.shape with
    // the constant extents from ``shape``.
    llvm::SmallVector<mlir::Value, 4> extents;
    for (auto d : shape) {
      auto c =
          b.create<mlir::arith::ConstantOp>(loc, b.getIndexType(),
                                            b.getIndexAttr(d));
      extents.push_back(c);
    }
    auto shapeOp = b.create<fir::ShapeOp>(loc, extents);
    // operandSegmentSizes for hlfir.declare: memref(1) + shape(1) +
    // typeparams(0) + dummy_scope(0).
    attrs.append("operandSegmentSizes", b.getDenseI32ArrayAttr({1, 1, 0, 0}));
    auto decl = b.create<hlfir::DeclareOp>(loc,
                                            mlir::TypeRange{refTy, refTy},
                                            mlir::ValueRange{alloca.getResult(),
                                                              shapeOp.getResult()},
                                            attrs);

    // Emit ``hlfir.assign <op-result> to <decl#0>`` immediately
    // after the linalg op, then convert the materialised variable
    // back to an ``!hlfir.expr<...>`` via ``hlfir.as_expr`` so the
    // existing consumers (``hlfir.shape_of`` / ``hlfir.apply`` /
    // ``hlfir.elemental``) keep their expected operand type.  The
    // declare's #0 result (a ``!fir.ref<!fir.array<...x T>>``) can
    // NOT replace the op's result directly because those consumers
    // require the HLFIR expression type, not a FIR ref -- the
    // ``hlfir.as_expr`` round-trip recovers the type so use
    // replacement is a simple SSA swap.
    b.setInsertionPointAfter(redOp);
    auto liftedAssign = b.create<hlfir::AssignOp>(loc, redOp->getResult(0),
                                                    decl.getResult(0));
    auto asExpr = b.create<hlfir::AsExprOp>(loc, resTy, decl.getResult(0),
                                              /*mustFree=*/mlir::Value{});

    // Replace every existing use of the op's result with the
    // as_expr result, EXCEPT the just-emitted hlfir.assign.
    llvm::SmallPtrSet<mlir::Operation *, 4> exceptions{liftedAssign};
    redOp->getResult(0).replaceAllUsesExcept(asExpr.getResult(), exceptions);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftReductionOperandsPass() {
  return std::make_unique<LiftReductionOperandsPass>();
}

}  // namespace hlfir_bridge
