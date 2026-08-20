// ============================================================================
// FoldConstantScalars.cpp  --  Replace loads from a scalar alloca with a
// single constant store by that constant, then erase the dead memory.
// ============================================================================
// Problem:
//     Flang lowers local scalar assignments like ``IWARMRAIN = 2`` to a
//     ``fir.alloca`` + ``fir.store`` + subsequent ``fir.load`` chain.  The
//     upstream ``sccp`` pass works on SSA values, not memory, so the loads
//     survive and the bridge emits a runtime scalar transient that hides the
//     constant from DaCe.  The resulting SDFG keeps whole ``IF (x == 1) ...
//     ELSEIF (x == 2) ...`` chains alive even though ``x`` is statically known.
//
// Fix:
//     For every scalar ``fir.alloca`` that has exactly one constant store and
//     only load uses, replace every load with the stored constant and erase the
//     store/alloca/declare chain.  This exposes the constant to the existing
//     ``sccp,canonicalize,cse`` run at the end of the pipeline, which then
//     folds the branch conditions and removes dead code.
//
// Safety:
//     Only folds when the single store dominates every load and there are no
//     other memory users (calls, addresses, unknown ops) of the alloca.  This
//     keeps the pass a local, obviously-correct memory-to-SSA promotion.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// Trace a value back through ``hlfir.declare`` / ``fir.convert`` to the
/// underlying ``fir.alloca``.  Returns the alloca value when the allocated
/// type is a scalar integer, float, or logical; otherwise returns nullptr.
mlir::Value getScalarAlloca(mlir::Value v) {
  for (int i = 0; i < 32 && v; ++i) {
    if (auto alloca = mlir::dyn_cast_or_null<fir::AllocaOp>(v.getDefiningOp())) {
      mlir::Type t = alloca.getAllocatedType();
      if (mlir::isa<mlir::IntegerType, mlir::FloatType, fir::LogicalType>(t)) {
        return v;
      }
      return nullptr;
    }
    if (auto decl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(v.getDefiningOp())) {
      v = decl.getMemref();
      continue;
    }
    if (auto conv = mlir::dyn_cast_or_null<fir::ConvertOp>(v.getDefiningOp())) {
      v = conv.getValue();
      continue;
    }
    break;
  }
  return nullptr;
}

/// Return the ultimate ``arith.constant`` op feeding ``v`` through a chain of
/// ``fir.convert`` ops, or nullptr if the value is not a constant.
mlir::arith::ConstantOp getConstantOp(mlir::Value v) {
  for (int i = 0; i < 16 && v; ++i) {
    if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>()) {
      return cst;
    }
    if (auto conv = mlir::dyn_cast_or_null<fir::ConvertOp>(v.getDefiningOp())) {
      v = conv.getValue();
      continue;
    }
    break;
  }
  return nullptr;
}

/// Recursively check that all users of ``v`` (through transparent
/// declare/convert wrappers) are either loads, ``fir.store``s, or scalar
/// ``hlfir.assign``s of the same scalar alloca, or other transparent wrappers.
/// On success, append the loads and the single write into the output vectors
/// and return true.
bool collectMemoryUsers(mlir::Value v, llvm::SmallVectorImpl<fir::LoadOp>& loads,
                        llvm::SmallVectorImpl<fir::StoreOp>& stores, llvm::SmallVectorImpl<hlfir::AssignOp>& assigns) {
  for (auto* u : v.getUsers()) {
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
      for (mlir::Value r : decl.getResults()) {
        if (!collectMemoryUsers(r, loads, stores, assigns)) return false;
      }
      continue;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(u)) {
      if (!collectMemoryUsers(conv.getResult(), loads, stores, assigns)) return false;
      continue;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) {
      loads.push_back(ld);
      continue;
    }
    if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) {
      stores.push_back(st);
      continue;
    }
    if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u)) {
      // Only treat scalar RHS assignments as stores.  Array/box assignments to
      // the same symbol cannot appear for a scalar alloca, but we still guard
      // the RHS type to keep the transform obviously local and correct.
      mlir::Type rhsTy = as.getRhs().getType();
      if (mlir::isa<mlir::IntegerType, mlir::FloatType, fir::LogicalType>(rhsTy)) {
        assigns.push_back(as);
        continue;
      }
      return false;
    }
    // Any other user (fir.call by reference, address_of, ...) is unsafe.
    return false;
  }
  return true;
}

struct FoldConstantScalarsPass
    : public mlir::PassWrapper<FoldConstantScalarsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldConstantScalarsPass)

  llvm::StringRef getArgument() const final { return "hlfir-fold-constant-scalars"; }
  llvm::StringRef getDescription() const final {
    return "Promote scalar stack variables with a single constant store to "
           "SSA constants so downstream SCCP can fold their uses.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    bool changed = false;

    module.walk([&](mlir::func::FuncOp func) {
      if (func.isExternal()) return;

      llvm::SmallVector<fir::AllocaOp> allocas;
      func.walk([&](fir::AllocaOp a) {
        mlir::Type t = a.getAllocatedType();
        if (mlir::isa<mlir::IntegerType, mlir::FloatType, fir::LogicalType>(t)) {
          allocas.push_back(a);
        }
      });

      mlir::DominanceInfo dom(func);

      for (auto alloca : allocas) {
        mlir::Value base = alloca.getResult();
        llvm::SmallVector<fir::LoadOp, 4> loads;
        llvm::SmallVector<fir::StoreOp, 1> stores;
        llvm::SmallVector<hlfir::AssignOp, 1> assigns;
        if (!collectMemoryUsers(base, loads, stores, assigns)) continue;
        if (stores.size() + assigns.size() != 1) continue;
        if (loads.empty()) continue;

        mlir::Value stored;
        mlir::Operation* writeOp = nullptr;
        if (!stores.empty()) {
          stored = stores.front().getValue();
          writeOp = stores.front();
        } else {
          stored = assigns.front().getRhs();
          writeOp = assigns.front();
        }
        if (!getConstantOp(stored)) continue;

        // The write must dominate every load.
        bool dominated = true;
        for (auto ld : loads) {
          if (!dom.dominates(writeOp, ld)) {
            dominated = false;
            break;
          }
        }
        if (!dominated) continue;

        // Replace loads with the constant value and erase the memory chain.
        for (auto ld : loads) {
          ld.replaceAllUsesWith(stored);
          ld->erase();
          changed = true;
        }
        writeOp->erase();

        // Drop the now-dead alloca/declare wrappers if nothing else uses them.
        auto eraseIfUnused = [](mlir::Operation* op) {
          if (!op) return;
          if (op->use_empty()) {
            op->erase();
          }
        };
        for (auto* u : llvm::make_early_inc_range(base.getUsers())) {
          eraseIfUnused(u);
        }
        eraseIfUnused(alloca);
      }
    });

    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createFoldConstantScalarsPass() { return std::make_unique<FoldConstantScalarsPass>(); }

}  // namespace hlfir_bridge
