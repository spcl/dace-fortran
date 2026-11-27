// ============================================================================
// LowerArithMinMax.cpp  --  Rewrite arith.max/min{si,ui} to arith.cmpi + arith.select (LLVM 23 only).
// ============================================================================
// Flang 23 emits some of its extent clamps ``select(n > 0, n, 0)`` as ``arith.maxsi(n, 0)`` (section extents)
// and MLIR 23's canonicalizer folds the rest; LLVM 21 and 22 leave the cmpi + select pair.  The bridge's
// extent, shape and bound tracing (isExtentClamp, traceConstInt, the declared-shape readers) match the
// select shape, so this pass turns the ops back into it at the start of the pipeline and again after the
// last canonicalize.  A no-op before LLVM 23, where
// an ``arith.maxsi`` is Fortran's own MAX/MIN and the bridge reads it as such.
// ============================================================================

#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct LowerArithMinMaxPass : public mlir::PassWrapper<LowerArithMinMaxPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerArithMinMaxPass)

  llvm::StringRef getArgument() const final { return "lower-arith-minmax"; }
  llvm::StringRef getDescription() const final {
    return "Rewrite arith.max/min{si,ui} into arith.cmpi + arith.select, the shape LLVM 21/22 keep (LLVM 23 only).";
  }

  void runOnOperation() override {
#if LLVM_VERSION_MAJOR >= 23
    using mlir::arith::CmpIPredicate;
    llvm::SmallVector<mlir::Operation*, 16> ops;
    getOperation().walk([&](mlir::Operation* op) {
      if (mlir::isa<mlir::arith::MaxSIOp, mlir::arith::MinSIOp, mlir::arith::MaxUIOp, mlir::arith::MinUIOp>(op))
        ops.push_back(op);
    });
    for (auto* op : ops) {
      mlir::OpBuilder b(op);
      CmpIPredicate pred = mlir::isa<mlir::arith::MaxSIOp>(op)   ? CmpIPredicate::sgt
                           : mlir::isa<mlir::arith::MinSIOp>(op) ? CmpIPredicate::slt
                           : mlir::isa<mlir::arith::MaxUIOp>(op) ? CmpIPredicate::ugt
                                                                 : CmpIPredicate::ult;
      auto lhs = op->getOperand(0);
      auto rhs = op->getOperand(1);
      auto cmp = b.create<mlir::arith::CmpIOp>(op->getLoc(), pred, lhs, rhs);
      op->getResult(0).replaceAllUsesWith(b.create<mlir::arith::SelectOp>(op->getLoc(), cmp, lhs, rhs));
      op->erase();
    }
#endif
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createLowerArithMinMaxPass() { return std::make_unique<LowerArithMinMaxPass>(); }

}  // namespace hlfir_bridge
