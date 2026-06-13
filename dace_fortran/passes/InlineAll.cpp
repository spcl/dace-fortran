// ============================================================================
// InlineAll.cpp  --  Aggressive whole-program inlining.
// ============================================================================
// Problem:
//     For deployment-time specialisation (namelist constant injection +
//     SCCP), we need interprocedural constant propagation across call
//     boundaries.  Rather than building a full context-sensitive SCCP,
//     we flatten the call tree: inline every callee into its caller
//     until only external/intrinsic declarations remain.
//
// Approach:
//     Fixed-point iteration that flattens the call tree into the public
//     (entry) root functions only.  Each sweep inlines every fir.call that
//     currently lives inside a root; a call buried in a not-yet-inlined
//     private helper is pulled in transitively on a later round, once that
//     helper has itself been inlined into a root.  After each sweep, private
//     helpers whose last caller was just absorbed are erased.
//
//     Restricting the inline targets to the roots and pruning absorbed
//     helpers each round keeps the working set to (the growing root bodies +
//     the still-un-duplicated helper library) instead of ballooning every
//     function in a large merged USE-closure at once -- the difference
//     between a few hundred MB and tens of GB for the ICON dynamical core,
//     whose entry pulls in a ~500-function closure.  When the module has no
//     public function (an ordinary module with no designated entry), the pass
//     falls back to inlining into every function so behaviour is unchanged.
//
// Assumptions:
//     - No recursive functions.  The pass does not detect cycles; if
//       recursion exists, it will hit the iteration cap and bail out.
//     - Code size explosion is acceptable  --  the result is meant for
//       specialisation, not direct compilation.
//
// After this pass, the root function(s) contain the full flattened program
// body; any remaining dead callees are removed with --symbol-dce.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"

#include <cstdlib>
#include "passes/Passes.h"

#define DEBUG_TYPE "inline-all"

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Inliner interface  --  accept everything.
// ---------------------------------------------------------------------------

/// Permissive inliner interface that allows inlining any callable into any
/// call site.  We override the legality hooks to always return true, and
/// defer the transformation hooks (``handleTerminator`` / ``handleArgument``
/// / ``handleResult``) to the per-dialect ``DialectInlinerInterface`` that
/// Flang registers for FIR and the core dialects  --  overriding them here
/// would short-circuit the correct per-op behaviour and can corrupt the IR.
struct AggressiveInlinerInterface : public mlir::InlinerInterface {
  using mlir::InlinerInterface::InlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }

  /// Always permit the single-block fast path.  The base
  /// ``InlinerInterface`` resolves this by querying the
  /// ``DialectInlinerInterface`` registered for the parent op of the
  /// inlined block (``getInterfaceFor(...->getParentOp())``).  After
  /// ``lift-cf-to-scf`` a call can sit inside an ``scf.if`` / ``scf.for``
  /// body whose dialect has no inliner interface in the bridge context, so
  /// that lookup returns null and the base method dereferences it (the
  /// guarding assert is compiled out of release MLIR).  Returning ``true``
  /// both avoids the crash and is the only correct answer there: an ``scf``
  /// region body must stay a single block, so the inliner cannot fall back
  /// to the block-splitting path.
  bool allowSingleBlockOptimization(
      llvm::iterator_range<mlir::Region::iterator> inlinedBlocks) const final {
    return true;
  }
};

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

struct InlineAllPass
    : public mlir::PassWrapper<InlineAllPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineAllPass)

  llvm::StringRef getArgument() const final { return "hlfir-inline-all"; }
  llvm::StringRef getDescription() const final {
    return "Aggressively inline all non-external callees to produce "
           "a flat, single-function representation.";
  }

  /// One sweep over the module.  Returns the number of call sites inlined.
  ///
  /// When ``rootsOnly`` is set, only calls whose enclosing function is a
  /// public root are inlined; calls inside private helpers are left for a
  /// later round, after the helper is inlined into a root.
  unsigned sweep(mlir::ModuleOp module, mlir::SymbolTable &symTab,
                 bool rootsOnly) {
    unsigned inlined = 0;
    AggressiveInlinerInterface interface(module.getContext());

    // Collect call ops first  --  we'll be mutating the IR during inlining,
    // so we cannot walk and inline simultaneously.
    llvm::SmallVector<fir::CallOp, 64> calls;
    module.walk([&](fir::CallOp call) {
      if (rootsOnly) {
        auto parent = call->getParentOfType<mlir::func::FuncOp>();
        if (!parent || parent.isPrivate()) return;
      }
      calls.push_back(call);
    });

    for (auto call : calls) {
      // Skip if the op was erased by a previous inlining in this sweep.
      if (!call->getParentOp()) continue;

      auto sym = call.getCallee();
      if (!sym) continue;  // indirect call

      auto callee = symTab.lookup<mlir::func::FuncOp>(sym->getLeafReference());
      if (!callee || callee.isDeclaration()) continue;  // external

      // Refuse multi-block callees -- splicing their CFG into a
      // structured ``scf.if`` / ``scf.for`` region around the call
      // site corrupts the region's single-block invariant and
      // crashes inside ``mlir::inlineCall`` (observed on QE's
      // ``errore`` early-return-then-STOP shape, ICON ``finish``,
      // and similar terminal error helpers that ``lift-cf-to-scf``
      // can't structurize because of the unreachable / noreturn
      // tail).  Leaving the call as a plain ``fir.call`` is safe:
      // the downstream bridge handles it the same way it handles
      // any external call site, and the multi-block callee stays
      // as its own (private, externally callable) function in the
      // module.
      if (callee.getBody().getBlocks().size() > 1) {
        if (std::getenv("HLFIR_INLINE_TRACE")) {
          llvm::errs() << "InlineAll: SKIP multi-block "
                       << callee.getSymName() << " ("
                       << callee.getBody().getBlocks().size()
                       << " blocks)\n";
          llvm::errs().flush();
        }
        continue;
      }

      // TRACE: emit per-call attempt so a downstream crash inside
      // ``mlir::inlineCall`` reveals which callee tripped it.  This
      // print is gated by the ``HLFIR_INLINE_TRACE`` env var so it
      // stays silent for ordinary runs.
      if (std::getenv("HLFIR_INLINE_TRACE")) {
        llvm::errs() << "InlineAll: inlining " << callee.getSymName()
                     << " into "
                     << call->getParentOfType<mlir::func::FuncOp>().getSymName()
                     << "\n";
        llvm::errs().flush();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "InlineAll: inlining " << callee.getSymName() << " into "
                 << call->getParentOfType<mlir::func::FuncOp>().getSymName()
                 << "\n");

      // Perform the inlining.
      auto callIface =
          mlir::dyn_cast<mlir::CallOpInterface>(call.getOperation());
      auto callableIface =
          mlir::dyn_cast<mlir::CallableOpInterface>(callee.getOperation());
      if (!callIface || !callableIface) continue;

      // Clone callback for inlineCall: insert cloned blocks BEFORE
      // ``postInsertBlock`` so the layout becomes
      //     [inlineBlock (inlined-into), cloned..., postInsertBlock].
      // Inserting at ``inlineBlock`` instead demotes the caller's
      // original entry block and drops its block-argument list,
      // which then trips func.func's signature verifier.
      auto cloneCallback = [](mlir::OpBuilder &builder, mlir::Region *src,
                              mlir::Block *inlineBlock, mlir::Block *postBlock,
                              mlir::IRMapping &mapper, bool shouldClone) {
        if (shouldClone) {
          src->cloneInto(inlineBlock->getParent(), postBlock->getIterator(),
                         mapper);
        } else {
          src->getBlocks().splice(postBlock->getIterator(), src->getBlocks());
        }
      };

      auto result = mlir::inlineCall(interface, cloneCallback, callIface,
                                     callableIface, &callee.getBody(),
                                     /*shouldCloneInlinedRegion=*/true);

      if (mlir::succeeded(result)) {
        // The call op is replaced by the inlined body.
        call->erase();
        ++inlined;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "InlineAll: FAILED to inline "
                                << callee.getSymName() << "\n");
      }
    }

    return inlined;
  }

  /// Erase private functions with no remaining symbol uses.  This is the
  /// function-level effect of ``--symbol-dce`` run between inlining rounds, so
  /// a helper body is freed as soon as its last caller has absorbed it rather
  /// than lingering until the end of the pipeline.  Returns the number erased.
  unsigned pruneDeadPrivateFuncs(mlir::ModuleOp module,
                                 mlir::SymbolTable &symTab) {
    mlir::SymbolTableCollection collection;
    mlir::SymbolUserMap users(collection, module);
    llvm::SmallVector<mlir::func::FuncOp, 64> dead;
    for (auto f : module.getOps<mlir::func::FuncOp>())
      if (f.isPrivate() && users.useEmpty(f)) dead.push_back(f);
    for (auto f : dead) symTab.erase(f);  // erases the op and the table entry
    return dead.size();
  }

  void runOnOperation() override {
    auto module = getOperation();
    mlir::SymbolTable symTab(module);

    // Inline into the public (entry) roots when the module designates one;
    // fall back to inlining into every function when nothing is public so an
    // ordinary entry-less module behaves as before.
    bool rootsOnly = false;
    for (auto f : module.getOps<mlir::func::FuncOp>())
      if (!f.isPrivate()) { rootsOnly = true; break; }

    // Fixed-point iteration.  Each round inlines one level of calls into the
    // roots and frees the helpers it just absorbed; repeat until no more
    // inlining is possible.  Cap at 128 rounds  --  without recursion this is
    // the max call-tree depth; in practice convergence is much faster.
    unsigned totalInlined = 0;
    for (int round = 0; round < 128; ++round) {
      unsigned n = sweep(module, symTab, rootsOnly);
      if (n == 0) break;
      totalInlined += n;
      unsigned freed = rootsOnly ? pruneDeadPrivateFuncs(module, symTab) : 0;

      LLVM_DEBUG(llvm::dbgs()
                 << "InlineAll: round " << round << " inlined " << n
                 << " call sites, freed " << freed << " helpers\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "InlineAll: total " << totalInlined
                            << " call sites inlined\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createInlineAllPass() {
  return std::make_unique<InlineAllPass>();
}

}  // namespace hlfir_bridge
