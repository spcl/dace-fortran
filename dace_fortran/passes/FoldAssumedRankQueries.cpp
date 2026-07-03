// ============================================================================
// FoldAssumedRankQueries.cpp  --  fold ``fir.box_rank`` /
//                                 ``fir.is_assumed_size`` when the box's
//                                 source rank/shape is statically known.
// ============================================================================
//
// Fortran 2018's assumed-rank dummies (``DIMENSION(..)``) reach the IR
// as ``!fir.box<!fir.array<*:T>>``.  Inside the callee a ``SELECT
// RANK`` construct dispatches on ``fir.box_rank`` to a per-rank
// branch; ``fir.is_assumed_size`` separates the assumed-size case.
//
// When the bridge inlines a caller -> assumed-rank callee chain, the
// caller already knows the actual's rank (it has a concrete-rank
// ``hlfir.declare``).  Flang lowers the call as:
//
//   %actual = hlfir.declare ... : (!fir.box<!fir.array<?x?xf64>>, ...)
//   %erased = fir.convert %actual#0 :
//             (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<*:f64>>
//   fir.call @inner(%erased)
//
// After ``hlfir-inline-all`` the callee's body is spliced into the
// caller with %erased flowing as the assumed-rank dummy's memref.  The
// callee body still has:
//
//   %r = fir.box_rank %dummy : (!fir.box<!fir.array<*:f64>>) -> i8
//   %as = fir.is_assumed_size %dummy : ... -> i1
//   scf.if .../switch %r ...
//
// Neither ``canonicalize`` nor ``sccp`` folds ``fir.box_rank`` of a
// converted-from-concrete box -- the operations have no built-in
// fold rule for the convert-from-concrete-rank pattern.  Every
// dispatch branch survives into AST extraction; the bridge sees
// multiple ``hlfir.declare`` ops with the same ``uniq_name`` but
// different concrete ranks and rejects the SDFG.
//
// This pass replaces:
//   * ``fir.box_rank %X`` -> ``arith.constant <N>`` when ``%X`` traces
//     back through ``fir.convert`` ops to a ``fir.box<!fir.array<...>>``
//     of statically-known rank N.
//   * ``fir.is_assumed_size %X`` -> ``arith.constant false`` when the
//     traced source is a concrete-rank box (assumed-size is only true
//     for ``DIMENSION(*)`` dummies, whose box has a sentinel last
//     extent flag; an explicit-shape actual never carries it).
//
// Canonicalize then folds the surrounding ``arith.cmpi`` / ``scf.if`` /
// ``fir.select_case`` chain to just the matching branch.  The
// remaining ``hlfir.declare`` of the rank-N converted box reaches AST
// extraction as a regular concrete-rank dummy, no special-case needed.
//
// Trace rules (matched against the IR shape Flang's
// ``hlfir-inline-all`` produces):
//
//   * ``fir.convert``  --  rank-laundering between box types (the
//     exact op that erases concrete rank to ``*``).  Peel through to
//     the source.
//   * ``hlfir.declare`` --  the dummy_scope declare on the assumed-
//     rank dummy.  Follow ``getMemref()`` to the convert that lowered
//     into it from the caller.
//   * ``fir.rebox``    --  shape-change rebox; same role as
//     ``fir.convert`` for tracing.
//
// Stops at the first defining op that's a ``hlfir.declare`` with a
// CONCRETE-rank result type (the caller's own array declare).  If the
// chain bottoms out at a still-assumed-rank box or at a function
// argument typed as assumed-rank (the genuine through-pass-around
// case), the pass leaves the query op alone -- the canonicalizer
// can't fold it further without runtime info.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// Maximum convert-chain depth to walk while tracing.
static constexpr unsigned kMaxTraceDepth = 16;

/// Strip ``fir.box`` / ``fir.ref`` / ``fir.heap`` / ``fir.ptr`` layers
/// off a type until we hit a ``fir.array`` (``fir.SequenceType``).
static mlir::Type peelToArray(mlir::Type t) {
  for (int i = 0; i < 8; ++i) {
    if (auto bx = mlir::dyn_cast<fir::BoxType>(t)) {
      t = bx.getEleTy();
      continue;
    }
    if (auto rf = mlir::dyn_cast<fir::ReferenceType>(t)) {
      t = rf.getEleTy();
      continue;
    }
    if (auto hp = mlir::dyn_cast<fir::HeapType>(t)) {
      t = hp.getEleTy();
      continue;
    }
    if (auto pt = mlir::dyn_cast<fir::PointerType>(t)) {
      t = pt.getEleTy();
      continue;
    }
    break;
  }
  return t;
}

/// Trace ``v`` back through ``fir.convert`` / ``fir.rebox`` /
/// ``hlfir.declare`` chains.  Returns the rank of the first
/// statically-known-rank ``fir.array`` we encounter, or ``-1`` if the
/// chain bottoms out at an assumed-rank box (rank ``*``) or an
/// untraceable defining op.
///
/// A rank-``*`` array type is distinguished by its single dimension
/// being marked unknown (``fir.SequenceType::getUnknownExtent()``)
/// AND having shape size exactly one with no concrete entries; we
/// match more directly via ``hasAssumedRank()`` since flang
/// represents the rank-``*`` case with that bit.
static int traceStaticRank(mlir::Value v) {
  for (unsigned i = 0; i < kMaxTraceDepth && v; ++i) {
    auto t = peelToArray(v.getType());
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
      // Assumed-rank carries a single ``unknown`` extent and the
      // shape's ``hasUnknownShape`` query reports rank-erased.  Use
      // the dedicated helper rather than checking shape sizes.
      if (!seq.hasUnknownShape()) return seq.getDimension();
    }
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      v = rb.getBox();
      continue;
    }
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      v = decl.getMemref();
      continue;
    }
    break;
  }
  return -1;
}

struct FoldAssumedRankQueriesPass
    : public mlir::PassWrapper<FoldAssumedRankQueriesPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldAssumedRankQueriesPass)

  llvm::StringRef getArgument() const final { return "hlfir-fold-assumed-rank-queries"; }
  llvm::StringRef getDescription() const final {
    return "Fold fir.box_rank / fir.is_assumed_size on concrete-rank "
           "actuals so SELECT RANK dispatches reduce to a single branch.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::OpBuilder b(&getContext());

    // ``fir.box_rank``: replace with the actual's static rank when
    // the convert chain traces back to a concrete-rank ``fir.array``.
    llvm::SmallVector<fir::BoxRankOp, 8> rankQueries;
    module.walk([&](fir::BoxRankOp op) { rankQueries.push_back(op); });
    for (auto op : rankQueries) {
      int rank = traceStaticRank(op.getBox());
      if (rank < 0) continue;
      b.setInsertionPoint(op);
      auto resTy = op.getResult().getType();
      // ``fir.box_rank`` returns ``i8`` per the op definition, but
      // build the constant with the result type so the rewrite
      // tolerates any width Flang chose at lowering time.
      auto cst =
          b.create<mlir::arith::ConstantOp>(op.getLoc(), resTy, b.getIntegerAttr(resTy, static_cast<int64_t>(rank)));
      op.getResult().replaceAllUsesWith(cst.getResult());
      op.erase();
    }

    // ``fir.box_dims %X, %d`` returns ``(lower_bound, extent,
    // byte_stride)`` at dim ``%d`` of the box ``%X``.  When ``%X``
    // traces back to a concrete-shape ``fir.array<D0xD1x...xT>``,
    // the extent at dim ``d`` is the static literal ``Dd`` and the
    // lower bound defaults to 1.  Stride is byte-stride (element
    // size * column-major product of preceding dims); leave it
    // alone since the bridge doesn't lower it.  Folding extent +
    // lower-bound lets the surrounding ``do j = 1, size(a, k)``
    // loop bound become a literal so the dummy doesn't carry a
    // free shape symbol into the SDFG signature.
    auto dimAsIndex = [](mlir::Value v) -> std::optional<int64_t> {
      auto* d = v.getDefiningOp();
      if (!d) return std::nullopt;
      auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d);
      if (!cst) return std::nullopt;
      // ``arith.constant`` wraps either ``IntegerAttr`` (i8/i32/...)
      // or ``IndexAttr`` (the ``index`` type used in shape ops).
      // ``fir.box_dims``'s dim operand is ``index``-typed in the
      // post-canonicalize shape; accept both attribute kinds.
      auto attr = cst.getValue();
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr)) return ia.getInt();
      return std::nullopt;
    };
    auto extentAt = [](mlir::Value v, int64_t dim) -> std::optional<int64_t> {
      // Walk back through the convert / rebox / declare / embox chain
      // looking for the first ``fir.array`` whose ``dim``-th extent is
      // a concrete literal.  A ``<?x?>`` or rank-erased ``<*>`` ancestor
      // doesn't terminate the trace -- the rank-mismatched chain may
      // re-promote a concrete-shape source back to ``<?x?>`` via the
      // SELECT RANK redeclare, and the concrete extents live further
      // back at the original ``fir.embox`` site.
      for (unsigned i = 0; i < kMaxTraceDepth && v; ++i) {
        auto t = peelToArray(v.getType());
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
          if (!seq.hasUnknownShape()) {
            auto shape = seq.getShape();
            if (dim >= 0 && dim < (int64_t)shape.size()) {
              int64_t e = shape[dim];
              if (e != fir::SequenceType::getUnknownExtent()) return e;
            }
          }
          // Either rank-erased ``<*>`` or ``<?...>`` at this dim --
          // keep tracing back through the def chain in case an
          // earlier ancestor carries the concrete extent.
        }
        auto* def = v.getDefiningOp();
        if (!def) break;
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
          v = cv.getValue();
          continue;
        }
        if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
          v = rb.getBox();
          continue;
        }
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
          v = decl.getMemref();
          continue;
        }
        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
          v = eb.getMemref();
          continue;
        }
        break;
      }
      return std::nullopt;
    };
    llvm::SmallVector<fir::BoxDimsOp, 8> dimQueries;
    module.walk([&](fir::BoxDimsOp op) { dimQueries.push_back(op); });
    for (auto op : dimQueries) {
      auto dim = dimAsIndex(op.getDim());
      if (!dim) continue;
      auto ext = extentAt(op.getVal(), *dim);
      if (!ext) continue;
      b.setInsertionPoint(op);
      auto idxTy = b.getIndexType();
      // Lower bound defaults to 1 for Fortran arrays (the
      // sequence-association reshape never carries a non-default
      // lower bound; the rank-aware ``hlfir.declare`` re-stamps
      // bounds when needed and that re-stamp is what reaches
      // user code, not this raw query).
      auto lb = b.create<mlir::arith::ConstantOp>(op.getLoc(), idxTy, b.getIndexAttr(1));
      auto exC = b.create<mlir::arith::ConstantOp>(op.getLoc(), idxTy, b.getIndexAttr(*ext));
      op.getLowerBound().replaceAllUsesWith(lb.getResult());
      op.getExtent().replaceAllUsesWith(exC.getResult());
      // Leave byte-stride to canonicalize -- replace only the two
      // results we can statically resolve.  If ``getByteStride()``
      // has no users, the op DCEs on its own.
      if (op.getByteStride().use_empty()) op.erase();
    }

    // ``fir.is_assumed_size``: replace with ``false`` when the
    // traced source has any static rank.  ``DIMENSION(*)`` actuals
    // also reach the IR with a typed array, but their box carries
    // an extra sentinel bit; in practice the assumed-size case
    // appears only when the caller itself has an assumed-size
    // declare in its arglist (which the bridge already rejects).
    // Folding to ``false`` is sound for every actual the bridge
    // accepts.
    llvm::SmallVector<fir::IsAssumedSizeOp, 8> sizeQueries;
    module.walk([&](fir::IsAssumedSizeOp op) { sizeQueries.push_back(op); });
    for (auto op : sizeQueries) {
      int rank = traceStaticRank(op.getVal());
      if (rank < 0) continue;
      b.setInsertionPoint(op);
      auto resTy = op.getResult().getType();
      auto cst = b.create<mlir::arith::ConstantOp>(op.getLoc(), resTy, b.getIntegerAttr(resTy, 0));
      op.getResult().replaceAllUsesWith(cst.getResult());
      op.erase();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createFoldAssumedRankQueriesPass() {
  return std::make_unique<FoldAssumedRankQueriesPass>();
}

}  // namespace hlfir_bridge
