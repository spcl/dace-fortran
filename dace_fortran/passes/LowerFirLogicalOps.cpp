// ============================================================================
// LowerFirLogicalOps.cpp  --  Lower fir.logical_and/or and fir.eqv/neqv to i1 arithmetic.
// ============================================================================
// LLVM 23 lowers ``.and.`` / ``.or.`` / ``.eqv.`` / ``.neqv.`` on LOGICAL(K) to
// ``fir.logical_and`` / ``fir.logical_or`` / ``fir.eqv`` / ``fir.neqv`` on ``!fir.logical<K>``.
// LLVM 21 and 22 emit ``fir.convert`` to ``i1``, ``arith.andi`` / ``ori`` / ``cmpi eq`` / ``xori``,
// and ``fir.convert`` back, and the bridge's expression builders only know that shape.  Rewriting
// the four ops into it keeps every reader unchanged.  The ops are matched by name, so this file
// builds against any LLVM major.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct LowerFirLogicalOpsPass : public mlir::PassWrapper<LowerFirLogicalOpsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFirLogicalOpsPass)

  llvm::StringRef getArgument() const final { return "lower-fir-logical-ops"; }
  llvm::StringRef getDescription() const final {
    return "Lower LLVM 23's fir.logical_and/or and fir.eqv/neqv to fir.convert + arith on i1, the shape LLVM 21/22 "
           "emit.";
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::Operation*, 16> ops;
    getOperation().walk([&](mlir::Operation* op) {
      auto nm = op->getName().getStringRef();
      if ((nm == "fir.logical_and" || nm == "fir.logical_or" || nm == "fir.eqv" || nm == "fir.neqv") &&
          op->getNumOperands() == 2 && op->getNumResults() == 1)
        ops.push_back(op);
    });
    for (auto* op : ops) {
      mlir::OpBuilder b(op);
      auto loc = op->getLoc();
      auto i1 = b.getI1Type();
      auto lhs = b.create<fir::ConvertOp>(loc, i1, op->getOperand(0));
      auto rhs = b.create<fir::ConvertOp>(loc, i1, op->getOperand(1));
      auto nm = op->getName().getStringRef();
      mlir::Value res;
      if (nm == "fir.logical_and")
        res = b.create<mlir::arith::AndIOp>(loc, lhs, rhs);
      else if (nm == "fir.logical_or")
        res = b.create<mlir::arith::OrIOp>(loc, lhs, rhs);
      else if (nm == "fir.eqv")
        res = b.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
      else
        res = b.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
      op->getResult(0).replaceAllUsesWith(b.create<fir::ConvertOp>(loc, op->getResult(0).getType(), res));
      op->erase();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createLowerFirLogicalOpsPass() { return std::make_unique<LowerFirLogicalOpsPass>(); }

}  // namespace hlfir_bridge
