// ============================================================================
// PreserveMutableGlobals.cpp  --  Classify ``fir.global`` ops as INPUT vs
//                                  MUTABLE based on whether the IR writes
//                                  them, and clear init bodies of INPUTs so
//                                  ``sccp`` cannot fold their loads.
// ============================================================================
// Problem:
//     Fortran module-level scalars compile to ``fir.global`` ops with an
//     initializer body.  When the kernel's body never writes a global
//     (the caller is expected to mutate it from OUTSIDE via the binding
//     layer), ``sccp`` reads the in-IR initializer and folds every load
//     to that constant.  ``symbol-dce`` then eliminates the now-unused
//     ``fir.address_of`` chain and the AST extractor never sees a read.
//     The caller's pre-set value flows through the SDFG arglist (after
//     the prune-non-transient fix) but the kernel body uses the
//     constant-folded initializer, so the value is ignored.
//
//     NPB LU surfaces this on ``dt`` (the SSOR time step): the gfortran
//     reference behaviour is "user sets ``dt`` before calling
//     ``dolu()``", but the SDFG bakes ``dt=0`` and the SSOR sweep is a
//     no-op.  Every module-scalar input the caller pre-sets has the
//     same shape.
//
// Approach:
//     Walk the module once and tally, for every ``fir.global`` symbol,
//     how many ``fir.store`` / ``hlfir.assign`` ops target that
//     global's address.  The trace walks through the chain
//     ``hlfir.declare`` / ``fir.convert`` / ``hlfir.designate`` /
//     ``fir.box_addr`` back to the originating ``fir.address_of``.
//
//     Classify each ``fir.global``:
//       * ``constant``-attributed (PARAMETER) -- bake; leave alone.
//       * Writes detected anywhere in the IR -- MUTABLE; leave body
//         alone (the runtime needs to know the BSS initial value as
//         a possible reaching definition for the first read).
//       * No writes -- INPUT; clear the init body region and demote
//         linkage to ``common``.  ``sccp`` now has no reaching value
//         to fold against; subsequent loads survive into the AST.
//
//     The bridge's ``add_descriptors`` (after the prune-non-transient
//     fix) already registers every classified scalar as a non-
//     transient ``(1,)``-Array on the SDFG, so an INPUT global flows
//     through the arglist with the rest.  The bindings emitter is
//     responsible for marshalling caller-supplied values into the
//     length-1 buffer slot at call time.
//
// Why the "writes anywhere" rule is the right shape:
//     ``hlfir-inline-all`` runs BEFORE us (see pipeline ordering); by
//     the time we run, the entry function's body has every callee's
//     stores spliced in.  Counting writes in the whole module is
//     therefore the same as counting writes in the entry's transitive
//     closure -- and is robust to whatever the inliner did or didn't
//     manage to consume.
//
// Pre-requisites:
//     Runs AFTER ``hlfir-inline-all`` (so we see the inlined stores)
//     and BEFORE ``sccp,canonicalize,cse`` (so the body clearing
//     stops the fold).  See ``builder/__init__.py`` pipeline.
//
// What this pass does NOT do:
//     - Touch the bindings emitter: marshalling caller values into
//       the (1,)-Array slot is a Python-side concern and lives in
//       ``dace_fortran/bindings/``.
//     - Touch parameter (PARAMETER) globals: those bake correctly.
//     - Strip the global itself: ``symbol-dce`` (later in the
//       pipeline) takes care of any global that ends up truly dead
//       after the classification, the same way it always has.
// ============================================================================

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

#define DEBUG_TYPE "preserve-mutable-globals"

namespace hlfir_bridge {

namespace {

/// Maximum depth for the address-tracing chain walk.  Plenty of
/// headroom for ``fir.address_of -> hlfir.declare -> fir.convert ->
/// hlfir.designate`` cascades the inliner can splice together.
static constexpr unsigned kTraceDepth = 16;

/// Walk ``v``'s defining chain looking for the ``fir.address_of @sym``
/// it originated from.  Returns the symbol name on success, empty
/// ``llvm::StringRef()`` otherwise.
///
/// Recognised passthroughs:
///   * ``hlfir.declare``         -- the bridge's primary variable handle;
///   * ``fir.convert``           -- type laundering inserted by the
///     inliner;
///   * ``hlfir.designate``       -- field / element access; an
///     assignment to a sub-element of a global counts as a write to
///     the global as a whole;
///   * ``fir.box_addr``          -- box-wrapped declares.
///
/// Anything else (a fresh ``fir.alloca``, a function argument, ...)
/// breaks the chain -- the write doesn't reach a global.
static llvm::StringRef traceToGlobalSym(mlir::Value v) {
  for (unsigned d = 0; d < kTraceDepth && v; ++d) {
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto addrOf = mlir::dyn_cast<fir::AddrOfOp>(def)) {
      return addrOf.getSymbol().getRootReference().getValue();
    }
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      v = decl.getMemref();
      continue;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = conv.getValue();
      continue;
    }
    if (auto desig = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      v = desig.getMemref();
      continue;
    }
    if (auto boxAddr = mlir::dyn_cast<fir::BoxAddrOp>(def)) {
      v = boxAddr.getVal();
      continue;
    }
    return {};
  }
  return {};
}

struct PreserveMutableGlobalsPass
    : public mlir::PassWrapper<PreserveMutableGlobalsPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreserveMutableGlobalsPass)

  llvm::StringRef getArgument() const final { return "hlfir-preserve-mutable-globals"; }
  llvm::StringRef getDescription() const final {
    return "Classify fir.global ops as INPUT vs MUTABLE based on whether "
           "the IR writes them; clear init bodies of INPUTs so sccp "
           "cannot fold their loads to the BSS initializer.";
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Pass 1: collect the set of global symbol names the IR writes.
    // Both ``fir.store`` and ``hlfir.assign`` count as writes -- the
    // inliner emits both depending on which lowering path it
    // followed for the source-level assignment.
    llvm::StringSet<> writtenSyms;
    module.walk([&](mlir::Operation* op) {
      mlir::Value target;
      if (auto store = mlir::dyn_cast<fir::StoreOp>(op)) {
        target = store.getMemref();
      } else if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
        target = assign.getLhs();
      } else {
        return;
      }
      llvm::StringRef sym = traceToGlobalSym(target);
      if (!sym.empty()) writtenSyms.insert(sym);
    });

    // Pass 2: classify each ``fir.global`` and clear bodies of the
    // INPUT bucket.
    unsigned clearedInputs = 0;
    unsigned mutableKept = 0;
    unsigned constantKept = 0;
    module.walk([&](fir::GlobalOp g) {
      if (g.getConstant()) {
        ++constantKept;
        return;
      }
      llvm::StringRef sym = g.getSymName();
      // Function-scope globals are a flang lowering of routine-local
      // ``SAVE``-semantic variables with source-level initialisers
      // (``real :: bob = 1`` inside a subroutine).  They are NOT
      // caller inputs -- the caller has no symbol to bind -- so the
      // initialiser must bake.  Two name shapes appear:
      //
      //   _QF<func>E<var>                 -- subroutine outside any module
      //   _QM<mod>F<func>E<var>           -- subroutine inside a module
      //   _QM<mod>F<func>F<inner>E<var>   -- nested CONTAINS, etc.
      //
      // Module-level globals are ``_QM<mod>E<var>`` only -- there is
      // never an ``F`` segment between the leading ``_QM`` and the
      // final ``E``.  Flang lowercases every Fortran identifier
      // (module names, function names, variable names), so an
      // uppercase ``F`` after the leading ``_Q`` is always a scope-
      // marker: no module / function / variable name carries one.
      // Scanning for any uppercase ``F`` after position 2 is therefore
      // an exact match for "this global lives inside a function
      // scope".
      if (sym.drop_front(2).contains('F')) {
        ++constantKept;
        return;
      }
      if (writtenSyms.contains(sym)) {
        ++mutableKept;
        return;
      }
      // INPUT: no writes anywhere.  Clear the init body so sccp has
      // no reaching definition to propagate; demote to ``common``
      // linkage so any downstream lowering still finds a defined
      // symbol (the bridge codegen doesn't honour the body for
      // runtime layout -- DaCe's allocator owns that -- but
      // ``common`` keeps the global a definition rather than an
      // external declaration).
      mlir::Region& body = g.getRegion();
      if (!body.empty()) body.getBlocks().clear();
      g.setLinkName(llvm::StringRef("common"));
      ++clearedInputs;
    });

    LLVM_DEBUG(llvm::dbgs() << "PreserveMutableGlobals: cleared " << clearedInputs << " INPUT body(ies); kept "
                            << mutableKept << " MUTABLE + " << constantKept << " constant\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createPreserveMutableGlobalsPass() {
  return std::make_unique<PreserveMutableGlobalsPass>();
}

}  // namespace hlfir_bridge
