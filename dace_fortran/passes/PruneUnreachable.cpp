// ============================================================================
// PruneUnreachable.cpp  --  drop the dispatch-table bindings that keep
// entry-unreachable procedures alive, so symbol-dce can remove them.
// ============================================================================
//
// Merging a large USE-closure into one translation unit (the cross-module
// "inline everything" path) pulls in infrastructure modules the entry uses only
// for their *types* -- key/value stores, hash tables, axis registries, and the
// like, whose type-bound procedures often do ``fir.select_type`` over a
// ``class(*)`` payload.  When the entry never dynamically dispatches to such a
// binding, its procedures are dead, but the merged module's ``fir.type_info``
// dispatch tables (``fir.dt_entry`` bindings) still reference them, so
// ``symbol-dce`` treats them as used and keeps them -- and the structurizing /
// polymorphism passes then choke on an unsupported ``select_type`` sitting in
// code the entry never reaches.
//
// This pass records the method names invoked by ``fir.dispatch`` ops reachable
// from the public entry point(s) over static ``fir.call`` / ``func.call``
// edges, then erases every ``fir.dt_entry`` whose binding method no reachable
// dispatch invokes: that drops the only static reference holding the bound
// procedures, so the following ``symbol-dce`` can remove whichever of them are
// otherwise unreferenced.  Matching by method name is intentionally
// conservative -- a name a reachable dispatch invokes keeps that binding on
// *every* type, so a binding a reachable dispatch genuinely needs is never
// erased (devirtualisation downstream still finds it); only bindings no
// reachable dispatch could ever select are dropped.  The procedures themselves
// are left for ``symbol-dce`` to remove -- it accounts for every reference
// kind, avoiding dangling-symbol bugs.
//
// The dispatch tables are not the only hold, though: flang also emits, per
// derived type, a ``linkonce_odr`` binding-table vtable
// (``fir.global @...E.v.<type>``) whose elements are ``fir.address_of`` the
// type-bound procedures, alongside the descriptor / component / name RTTI
// globals (``E.dt.`` / ``E.c.`` / ``E.n.`` / ...).  Those globals are not
// ``private``, so ``symbol-dce`` treats them as roots and keeps the bound
// procedures -- and everything they reach -- alive even once the dt_entries are
// gone.  ``linkonce_odr`` is precisely flang's "discard if unused" linkage for
// this compiler-generated RTTI, so this pass privatises it: the following
// ``symbol-dce`` then discards the descriptors no reachable code references
// (cascading through the proc-pointer web), while keeping the descriptors the
// entry's live data genuinely uses.
//
// This does NOT remove a procedure the entry's call graph genuinely reaches --
// e.g. a hash table reached transitively through a halo-exchange wrapper is
// live and survives, by design.  The pass only unsticks the dead dispatch
// tables.
//
// Runs before ``symbol-dce`` and the structurizing passes.  A no-op for an
// ordinary single-procedure input (no dispatch tables, nothing unreachable).
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct PruneUnreachablePass : public mlir::PassWrapper<PruneUnreachablePass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PruneUnreachablePass)

  llvm::StringRef getArgument() const final { return "hlfir-prune-unreachable"; }
  llvm::StringRef getDescription() const final {
    return "Erase dispatch-table (fir.dt_entry) bindings the entry never "
           "dynamically invokes, so symbol-dce can drop the unreachable "
           "(often polymorphic) procedure clusters a merged USE-closure "
           "drags in.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Index functions by symbol name; gather the public functions as roots
    // (``set_entry_symbol`` leaves only the entry public, but seeding from
    // every public function is correct regardless of whether it ran).
    llvm::StringMap<mlir::func::FuncOp> bySym;
    llvm::SmallVector<mlir::func::FuncOp, 8> roots;
    module.walk([&](mlir::func::FuncOp f) {
      bySym[f.getSymName()] = f;
      if (!f.isDeclaration() && mlir::SymbolTable::getSymbolVisibility(f) == mlir::SymbolTable::Visibility::Public)
        roots.push_back(f);
    });
    if (roots.empty()) return;  // no entry point to anchor reachability

    auto funcForRef = [&](mlir::SymbolRefAttr r) -> mlir::func::FuncOp {
      auto it = bySym.find(r.getLeafReference().getValue());
      return it == bySym.end() ? mlir::func::FuncOp() : it->second;
    };

    // BFS the reachable functions and record the dispatch methods reachable
    // code can invoke.  We do NOT follow a dispatch into its bindings: that
    // would pull in whole OOP clusters (the receiver type's bound procedures
    // and everything they transitively reference) that the entry's direct call
    // graph never executes.  Recording the method only keeps that dispatch's
    // own table entries below, so ``fir-polymorphic-op`` can still devirtualise
    // a monomorphic dispatch (and ``hlfir-reject-polymorphism`` reject a truly
    // runtime one with a clear error).
    llvm::SmallPtrSet<mlir::Operation*, 32> reachable;
    llvm::StringSet<> reachedMethods;
    llvm::SmallVector<mlir::func::FuncOp, 64> work(roots.begin(), roots.end());
    auto enqueue = [&](mlir::func::FuncOp f) {
      if (f && !reachable.contains(f.getOperation())) work.push_back(f);
    };
    while (!work.empty()) {
      mlir::func::FuncOp f = work.pop_back_val();
      if (!reachable.insert(f.getOperation()).second) continue;
      f.walk([&](mlir::Operation* op) {
        // Walk direct-call edges only.  This set exists solely to collect the
        // methods reachable ``fir.dispatch`` ops invoke, so it must stay tight:
        // following ``fir.address_of`` would drag in whole OOP infrastructure
        // clusters whose own ``fir.dispatch`` ops re-mark their methods
        // reached, keeping the dispatch tables alive and defeating the prune.
        // The trade-off is that a procedure reached only as a proc-pointer
        // value
        // (``fir.address_of`` + indirect call) is not scanned for the methods
        // it dispatches; should it dispatch one whose binding is then pruned,
        // ``fir-polymorphic-op`` / ``hlfir-reject-polymorphism`` surface it as
        // a clear error rather than a miscompile.  No reachable dispatch
        // observed so far depends on such a binding.
        if (auto c = mlir::dyn_cast<fir::CallOp>(op)) {
          if (auto callee = c.getCallee()) enqueue(funcForRef(*callee));
        } else if (auto fc = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          auto it = bySym.find(fc.getCallee());
          if (it != bySym.end()) enqueue(it->second);
        } else if (auto d = mlir::dyn_cast<fir::DispatchOp>(op)) {
          reachedMethods.insert(d.getMethod());
        }
      });
    }

    // Erase dispatch-table bindings whose method no reachable dispatch invokes:
    // that removes the hold on the unreachable bound procedures so the
    // following symbol-dce can drop the dead clusters.  Keeping the data layout
    // of every ``fir.type_info`` intact -- only its dispatch entries shrink.
    llvm::SmallVector<fir::DTEntryOp, 16> dead;
    module.walk([&](fir::DTEntryOp e) {
      if (!reachedMethods.contains(e.getMethod())) dead.push_back(e);
    });
    for (auto e : dead) e.erase();

    // Privatise the compiler-generated ``linkonce_odr`` RTTI globals (per-type
    // binding-table vtables ``E.v.<type>``, descriptors ``E.dt.``, components
    // ``E.c.``, ...).  Their ``fir.address_of`` elements are the *other* hold
    // on the unreachable bound procedures, and being non-private they are
    // symbol-dce roots.  ``linkonce_odr`` is flang's "discard if unused"
    // linkage, so privatising lets the following ``symbol-dce`` drop the ones
    // no reachable code references (and only those -- a descriptor the entry's
    // live data uses stays referenced and survives).
    module.walk([&](fir::GlobalOp g) {
      std::optional<llvm::StringRef> link = g.getLinkName();
      if (link && *link == "linkonce_odr")
        mlir::SymbolTable::setSymbolVisibility(g, mlir::SymbolTable::Visibility::Private);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createPruneUnreachablePass() { return std::make_unique<PruneUnreachablePass>(); }

}  // namespace hlfir_bridge
