// ============================================================================
// PruneNeverAllocatedMemberDeref.cpp  --  erase a structured ``fir.if`` guard
// whose body unconditionally dereferences an ALLOCATABLE / POINTER record
// member that the whole module never allocates and never lets escape.
// ============================================================================
//
// Merging a large USE-closure into one translation unit pulls in
// infrastructure whose object graph the entry only reads, never constructs.
// The ICON ocean ``solve_free_sfc_ab_mimetic`` entry, for instance, contains
//
//     IF (createsolvermatrix) CALL ocean_solve_dump_matrix(free_sfc_solver, ...)
//
// where ``free_sfc_solver % act`` is a ``CLASS(t_ocean_solve_backend),
// ALLOCATABLE`` member.  After full inlining the guarded body is the fused
// dump chain, and its very first act is to read members THROUGH ``% act``
// (``ASSOCIATED(this%trans)``, then the ``nblk_loc`` / ``nidx_loc`` /
// ``SIZE(coef_l_wp,3)`` loop bounds).  ``% act`` is never ALLOCATEd anywhere in
// the module (its constructor is an empty stub) and never escapes, so the
// descriptor's base address is null on every reachable path.  What survives to
// SDFG extraction is loop scaffolding indexed by members of a never-constructed
// object -- a cluster of free symbols (``..._act_lhs_nblk_loc``, ...) the
// bindings layer has nothing to bind, because the storage they name does not
// exist.
//
// The sound, static justification for dropping the body is NOT the runtime
// namelist flag ``createsolvermatrix`` (a runtime value flang may not fold, and
// neither may we).  It is that the guarded body UNCONDITIONALLY references an
// allocatable/pointer member that is PROVABLY never allocated and never escapes
// module-wide.  Referencing an unallocated allocatable (or dereferencing a
// disassociated pointer) is Fortran-standard undefined behaviour, so under the
// bridge's standing well-defined-input assumption ANY execution that enters the
// guard is already source-UB -- a correct compilation may do anything on that
// path, including delete it.  In a well-defined program the guard is therefore
// never entered, so erasing it (including any observable I/O that would follow
// the UB deref) changes nothing.
//
// The prune is keyed on that invariant, never on the branch condition or a
// member name:
//
//   For each ALLOCATABLE / POINTER, record-typed struct-member slot type M
//   reachable in the module (keyed by parent RecordType + member name):
//     * allocated(M) := some designate of M has its descriptor ref written
//                       (``fir.store`` into the slot) -- covers explicit
//                       ``ALLOCATE`` / pointer ``=>`` rebind / nullify.
//     * escapes(M)   := some designate of M has its descriptor ref (or a
//                       convert / declare alias of it) passed as an operand to
//                       a ``fir.call`` / ``func.call`` / ``fir.dispatch`` --
//                       an opaque callee could allocate it (this is also how
//                       the allocatable runtime intrinsics take the slot).
//     if NOT allocated(M) and NOT escapes(M):
//       for every designate of M that sits in the ENTRY block of a result-less
//       ``fir.if`` with no ``else`` region, erase that whole ``fir.if``.
//
// The escape trace stops at ``fir.load`` of the descriptor ref: the loaded box
// is a read snapshot, and passing it (or values read out of the pointee) to a
// callee cannot re-allocate the caller's slot -- only a write through the slot
// address can.  Any unrecognised consumer of the descriptor ref is treated
// conservatively as an escape, so a slot is pruned only when every use of it
// module-wide is a plain read.  Restricting M to record-typed members keeps the
// pass to the "opaque, never-constructed object" shape and away from ordinary
// numeric allocatables.
//
// Requiring the designate to sit in the ENTRY block of the guard is what makes
// "the body unconditionally dereferences M" a syntactic check: the entry block
// runs whenever the branch is taken, so reaching the guard implies reaching the
// UB deref.  A designate nested behind an inner conditional, or a guard with a
// non-trivial ``else``, or one that yields results, is left untouched (zero
// mutation) rather than risk deleting a live path.  Erasing a result-less
// ``fir.if`` is structurally safe: it has no results and MLIR region scoping
// guarantees nothing defined inside it is used outside.
//
// Runs AFTER ``hlfir-inline-all`` + ``symbol-dce`` (so the whole reachable
// program, and every allocation / escape site, is visible) and BEFORE
// ``hlfir-flatten-structs`` (so the flatten never mints companions for the
// dead chain).  A no-op on any input without a never-allocated, never-escaping
// record allocatable/pointer member read inside such a guard.
// ============================================================================

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// Peel fir.box / fir.class / fir.ref / fir.heap / fir.pointer down to the
/// innermost element type.
static mlir::Type unwrapAllWrappers(mlir::Type t) {
  for (;;) {
    mlir::Type inner = t;
    if (auto x = mlir::dyn_cast<fir::BaseBoxType>(t))
      inner = x.getEleTy();
    else if (auto x = mlir::dyn_cast<fir::ReferenceType>(t))
      inner = x.getEleTy();
    else if (auto x = mlir::dyn_cast<fir::HeapType>(t))
      inner = x.getEleTy();
    else if (auto x = mlir::dyn_cast<fir::PointerType>(t))
      inner = x.getEleTy();
    if (inner == t) return t;
    t = inner;
  }
}

/// True when the designate reads an ALLOCATABLE or POINTER member.
static bool isAllocatableOrPointer(hlfir::DesignateOp dg) {
  auto attrs = dg.getFortranAttrs();
  if (!attrs) return false;
  return bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::allocatable) ||
         bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer);
}

/// Trace the descriptor ref forward.  Returns true if the slot could be
/// written (``fir.store`` into it) or could escape (its address reaches a
/// call / dispatch), or is consumed by any op we do not recognise as a pure
/// read.  A plain ``fir.load`` is a read snapshot and terminates the walk.
static bool slotWrittenOrEscaped(mlir::Value ref) {
  llvm::SmallVector<mlir::Value, 8> work{ref};
  llvm::SmallPtrSet<mlir::Value, 8> seen{ref};
  while (!work.empty()) {
    mlir::Value v = work.pop_back_val();
    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* op = use.getOwner();
      // Read snapshot of the descriptor -- safe, and the loaded box carries no
      // capability to re-allocate the caller's slot.
      if (mlir::isa<fir::LoadOp>(op)) continue;
      // Any write anywhere near the descriptor ref: allocate / rebind / the
      // address itself leaking into storage.
      if (mlir::isa<fir::StoreOp>(op)) return true;
      // The descriptor ref (possibly converted to ``!fir.ref<!fir.box<none>>``)
      // reaching an opaque callee: the callee -- or an allocatable runtime
      // intrinsic -- could allocate it.
      if (mlir::isa<fir::CallOp, mlir::func::CallOp, fir::DispatchOp>(op)) return true;
      // Address-preserving forwards: keep tracing the aliased ref.
      if (mlir::isa<fir::ConvertOp, hlfir::DeclareOp>(op)) {
        for (mlir::Value r : op->getResults())
          if (seen.insert(r).second) work.push_back(r);
        continue;
      }
      // Unrecognised consumer of the descriptor ref -- refuse conservatively.
      return true;
    }
  }
  return false;
}

/// The result-less ``fir.if`` (no ``else``) that directly guards ``dg`` in its
/// entry block, or null when ``dg`` is not so guarded.
static fir::IfOp enclosingUnconditionalGuard(hlfir::DesignateOp dg) {
  auto guard = mlir::dyn_cast_or_null<fir::IfOp>(dg->getParentOp());
  if (!guard) return {};
  if (guard.getNumResults() != 0) return {};
  if (!guard.getElseRegion().empty()) return {};
  if (guard.getThenRegion().empty()) return {};
  // Unconditional on entry: the designate must live in the block that runs
  // whenever the branch is taken (the then-region entry block).
  if (dg->getBlock() != &guard.getThenRegion().front()) return {};
  return guard;
}

struct PruneNeverAllocatedMemberDerefPass
    : public mlir::PassWrapper<PruneNeverAllocatedMemberDerefPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PruneNeverAllocatedMemberDerefPass)

  llvm::StringRef getArgument() const final { return "hlfir-prune-never-allocated-member-deref"; }
  llvm::StringRef getDescription() const final {
    return "Erase a result-less fir.if guard whose entry block unconditionally "
           "dereferences an allocatable/pointer record member that the module "
           "never allocates and never lets escape (Fortran-standard UB under "
           "the well-defined-input assumption).";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Group every record-member designate by (parent RecordType, member name)
    // so allocation / escape is judged over the whole slot type, module-wide.
    // A member is a prune CANDIDATE only when it is an allocatable / pointer
    // record-typed slot; a member is only prunable when NO designate of that
    // slot type is written or escapes.
    llvm::StringMap<llvm::SmallVector<hlfir::DesignateOp, 2>> designatesBySlot;
    llvm::StringMap<bool> candidate;

    module.walk([&](hlfir::DesignateOp dg) {
      auto compAttr = dg.getComponentAttr();
      if (!compAttr) return;
      mlir::Type parentRec = unwrapAllWrappers(dg.getMemref().getType());
      auto rec = mlir::dyn_cast<fir::RecordType>(parentRec);
      if (!rec) return;
      std::string key = (rec.getName() + "::" + compAttr.getValue()).str();
      designatesBySlot[key].push_back(dg);
      // Candidate = allocatable/pointer member whose pointee is itself a record
      // (the "opaque object" shape); plain numeric allocatables are out of
      // scope for this prune.
      if (isAllocatableOrPointer(dg) && mlir::isa<fir::RecordType>(unwrapAllWrappers(dg.getResult().getType())))
        candidate[key] = true;
    });

    // Guards to erase, deduplicated (act and trans of the same object both
    // resolve to the same enclosing fir.if).
    llvm::SmallPtrSet<mlir::Operation*, 4> guardSet;
    llvm::SmallVector<fir::IfOp, 4> guards;

    for (auto& kv : designatesBySlot) {
      if (!candidate.lookup(kv.getKey())) continue;
      bool unsafe = false;
      for (hlfir::DesignateOp dg : kv.second)
        if (slotWrittenOrEscaped(dg.getResult())) {
          unsafe = true;
          break;
        }
      if (unsafe) continue;
      for (hlfir::DesignateOp dg : kv.second)
        if (fir::IfOp guard = enclosingUnconditionalGuard(dg))
          if (guardSet.insert(guard.getOperation()).second) guards.push_back(guard);
    }

    if (guards.empty()) return;

    // Erase outermost-first is unnecessary: skip any guard nested inside
    // another queued guard (the outer erase takes it), then erase the rest.
    // Each erase is atomic on a result-less op, so a refused shape never leaves
    // a half-mutated region.
    for (fir::IfOp guard : guards) {
      bool nestedInOther = false;
      for (mlir::Operation* p = guard->getParentOp(); p; p = p->getParentOp())
        if (guardSet.contains(p)) {
          nestedInOther = true;
          break;
        }
      if (!nestedInOther) guard.erase();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createPruneNeverAllocatedMemberDerefPass() {
  return std::make_unique<PruneNeverAllocatedMemberDerefPass>();
}

}  // namespace hlfir_bridge
