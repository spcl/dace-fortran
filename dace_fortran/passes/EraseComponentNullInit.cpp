// ============================================================================
// EraseComponentNullInit.cpp  --  Drop LLVM 23's component-wise null initialisation of derived-type locals.
// ============================================================================
// For a scalar derived-type local (or INTENT(OUT) dummy) whose type has allocatable / pointer components and no
// component default, LLVM 23 nullifies each such component in place:
//
//     %c = fir.coordinate_of %s#0, w : (...) -> !fir.ref<!fir.box<!fir.heap<...>>>
//     fir.store %null_box to %c
//
// LLVM 21 and 22 instead copy the type's ``DerivedInit`` global (``fir.copy``), which FlattenStructs'
// ``seedLeafDefaults`` erases without seeding box leaves: the SDFG never models a descriptor's null state.  The
// ``fir.coordinate_of`` users keep FlattenStructs from flattening the struct at all, so erase the stores and their
// now-dead address and value chains, which leaves the shape 21/22 reach after that erase.  A no-op before LLVM 23.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

// True for flang's unallocated / disassociated descriptor (fir.embox of fir.zero_bits) and null procedure pointer.
bool isNullBox(mlir::Value v) {
  auto* d = v.getDefiningOp();
  if (auto eb = mlir::dyn_cast_or_null<fir::EmboxOp>(d)) return eb.getMemref().getDefiningOp<fir::ZeroOp>() != nullptr;
  if (auto ep = mlir::dyn_cast_or_null<fir::EmboxProcOp>(d))
    return ep.getFunc().getDefiningOp<fir::ZeroOp>() != nullptr;
  return false;
}

// True when v is a chain of field-name fir.coordinate_of ops rooted at an hlfir.declare.
bool isComponentOfDeclare(mlir::Value v) {
  bool any = false;
  while (auto co = v.getDefiningOp<fir::CoordinateOp>()) {
    auto idx = co.getFieldIndices();
    if (!co.getCoor().empty() || !idx || idx->size() != 1 || !mlir::isa<fir::ReferenceType>(co.getRef().getType()) ||
        !mlir::isa<fir::RecordType>(fir::unwrapRefType(co.getRef().getType())))
      return false;
    any = true;
    v = co.getRef();
  }
  return any && v.getDefiningOp<hlfir::DeclareOp>();
}

// Erase op and, transitively, the defining ops of its operands that it left trivially dead; a declare stays, as it
// names a dummy.
void eraseDeadChain(mlir::Operation* op) {
  if (!op || mlir::isa<hlfir::DeclareOp>(op) || !mlir::isOpTriviallyDead(op)) return;
  llvm::SmallVector<mlir::Operation*, 4> defs;
  for (auto v : op->getOperands()) defs.push_back(v.getDefiningOp());
  op->erase();
  for (auto* d : defs) eraseDeadChain(d);
}

struct EraseComponentNullInitPass
    : public mlir::PassWrapper<EraseComponentNullInitPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseComponentNullInitPass)

  llvm::StringRef getArgument() const final { return "erase-component-null-init"; }
  llvm::StringRef getDescription() const final {
    return "Erase LLVM 23's fir.coordinate_of + null-box fir.store component initialisation of derived-type locals, "
           "the shape LLVM 21/22 reach once FlattenStructs drops the DerivedInit copy.";
  }

  void runOnOperation() override {
    llvm::SmallVector<fir::StoreOp, 8> stores;
    getOperation().walk([&](fir::StoreOp st) {
      if (isNullBox(st.getValue()) && isComponentOfDeclare(st.getMemref())) stores.push_back(st);
    });
    for (auto st : stores) {
      auto* addr = st.getMemref().getDefiningOp();
      auto* val = st.getValue().getDefiningOp();
      st.erase();
      eraseDeadChain(addr);
      eraseDeadChain(val);
    }
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createEraseComponentNullInitPass() {
  return std::make_unique<EraseComponentNullInitPass>();
}

}  // namespace hlfir_bridge
