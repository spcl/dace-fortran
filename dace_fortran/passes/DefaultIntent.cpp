// ============================================================================
// DefaultIntent.cpp  --  Fill in ``intent_inout`` for dummies that lack an
// intent.
// ============================================================================
// Problem:
//     Fortran lets a subroutine declare a dummy argument without any INTENT
//     attribute; the standard treats it as effectively ``intent(inout)``
//     for callers.  Flang emits such a dummy as
//         hlfir.declare %arg0 dummy_scope %0 {uniq_name = ...} : ...
//     with no ``fortran_attrs`` attribute at all.  Downstream consumers
//     (our extract_vars, code that chooses between transient/non-transient
//     SDFG descriptors, ...) can't tell the difference between "a local"
//     and "a dummy without explicit intent" from the attribute alone.
//
// What this pass does:
//     For every func.func in the module, walk its entry-block arguments,
//     find the hlfir.declare that uses each arg, and if that declare's
//     fortran_attrs has no intent bit set, OR the attribute is absent
//     entirely, set it to include ``intent_inout``.
//
// Why it's safe:
//     The standard requires a dummy to be treated like intent_inout when
//     no intent is given, so canonicalising the IR to say so doesn't
//     change semantics  --  it just makes the subsequent passes / extract
//     code trivially correct.
//
// Written integer dummies:
//     An integer scalar read as an index or in a condition becomes an SDFG
//     symbol, and a symbol passes by value: a write would never reach the
//     caller. The body runs on a local copy instead, stored back to the dummy
//     before every return.
// ============================================================================

#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// Runs ``func`` on a local copy of integer scalar dummy ``decl`` when the body writes the dummy and reads the result.
void localizeWrittenDummy(mlir::func::FuncOp func, hlfir::DeclareOp decl) {
  using Flags = fir::FortranVariableFlagsEnum;
  auto flags = decl.getFortranAttrs().value_or(Flags::None);
  auto eleTy = fir::dyn_cast_ptrEleTy(decl.getBase().getType());
  if (!eleTy || !eleTy.isSignlessInteger() || !bitEnumContainsAny(flags, Flags::intent_out | Flags::intent_inout) ||
      bitEnumContainsAny(flags, Flags::optional | Flags::value))
    return;

  // Entry-block position of each write: the local takes over at the first one, or at the end of the entry block when
  // the writes all sit in later blocks. Reads before it see the unchanged dummy, which keeps specification expressions
  // such as the extent of ``a(n)`` on the dummy.
  auto& entry = func.front();
  mlir::Operation* takeover = nullptr;
  bool reads = false;
  bool writes = false;
  llvm::SmallVector<mlir::OpOperand*, 8> uses;
  for (mlir::Value base : decl->getResults())
    for (auto& use : base.getUses()) {
      uses.push_back(&use);
      auto* user = use.getOwner();
      bool const store = mlir::isa<fir::StoreOp>(user) && use.getOperandNumber() == 1;
      bool const assign = mlir::isa<hlfir::AssignOp>(user) && use.getOperandNumber() == 1;
      reads |= store || !assign;  // a DO variable stores through fir.store and is a symbol even when unread
      if (!store && !assign && !mlir::isa<fir::CallOp>(user)) continue;
      writes = true;
      auto* top = entry.findAncestorOpInBlock(*user);
      if (top && (!takeover || top->isBeforeInBlock(takeover))) takeover = top;
    }
  if (!reads || !writes) return;
  if (!takeover) takeover = entry.getTerminator();

  mlir::OpBuilder b(takeover);
  auto loc = decl.getLoc();
  auto local = b.create<fir::AllocaOp>(loc, eleTy);
  auto name = b.getStringAttr(decl.getUniqName().str() + "_local");
  auto localDecl = b.create<hlfir::DeclareOp>(loc, local, name.getValue());
  b.create<hlfir::AssignOp>(loc, b.create<fir::LoadOp>(loc, decl.getBase()), localDecl.getBase());

  for (auto* use : uses) {
    auto* top = entry.findAncestorOpInBlock(*use->getOwner());
    if (!top || top == takeover || takeover->isBeforeInBlock(top))
      use->set(use->get() == decl.getResult(1) ? localDecl.getResult(1) : localDecl.getResult(0));
  }
  func.walk([&](mlir::func::ReturnOp ret) {
    mlir::OpBuilder at(ret);
    at.create<hlfir::AssignOp>(loc, at.create<fir::LoadOp>(loc, localDecl.getBase()), decl.getBase());
  });
}

struct DefaultIntentPass : public mlir::PassWrapper<DefaultIntentPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DefaultIntentPass)

  llvm::StringRef getArgument() const final { return "hlfir-default-intent"; }
  llvm::StringRef getDescription() const final {
    return "Stamp intent_inout onto dummy-argument hlfir.declare ops "
           "that lack an explicit intent, mirroring Fortran's default.";
  }

  void runOnOperation() override {
    auto* ctx = &getContext();
    using Flags = fir::FortranVariableFlagsEnum;
    auto hasIntent = [](Flags f) {
      return bitEnumContainsAny(f, Flags::intent_in) || bitEnumContainsAny(f, Flags::intent_out) ||
             bitEnumContainsAny(f, Flags::intent_inout);
    };

    getOperation().walk([&](mlir::func::FuncOp func) {
      if (func.isExternal()) return;
      auto& block = func.front();
      for (auto arg : block.getArguments()) {
        hlfir::DeclareOp decl;
        for (auto* u : arg.getUsers())
          if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
            decl = d;
            break;
          }
        if (!decl) continue;

        auto current = decl.getFortranAttrs();
        Flags flags = current ? *current : Flags::None;
        if (!hasIntent(flags)) {
          flags = flags | Flags::intent_inout;
          decl->setAttr("fortran_attrs", fir::FortranVariableFlagsAttr::get(ctx, flags));
        }
        localizeWrittenDummy(func, decl);
      }
    });
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createDefaultIntentPass() { return std::make_unique<DefaultIntentPass>(); }

}  // namespace hlfir_bridge
