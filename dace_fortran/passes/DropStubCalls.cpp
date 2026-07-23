// ============================================================================
// DropStubCalls.cpp  --  erase calls to ``do_not_emit`` (stub) procedures.
// ============================================================================
//
// Motivation:
//     ``apply_external_functions(do_not_emit=[...])`` registers a procedure as
//     an IGNORED external: the bridge drops its call at SDFG emission and emits
//     no node.  ``externalize_symbols`` strips the body to a declaration, but
//     the ``fir.call`` itself survives the whole MLIR pipeline.
//
//     A call op that is going to be dropped anyway is not inert while it sits
//     there -- it is an opaque USE of whatever it was passed.  ICON-O
//     ``solve_free_sfc`` hit exactly that: a dropped ``dbg_print_3d(...)``
//     holding a copy-in temp made ``hlfir-fold-copy-in-out`` bail (it must not
//     reparent element accesses while some call still reads the old temp), so
//     the copy_in / copy_out pair survived, became a zero-filled phantom SDFG
//     argument, and every write to ``p_diag%veloc_adv_vert`` was dropped.  A
//     debug print with no numerical meaning silently corrupted the result.
//
//     ``builder/__init__.py`` already states the contract -- "a do_not_emit
//     (stub) callee ... its call never survives" -- so this pass just makes the
//     IR match it early, instead of leaving every later pass to reason around a
//     call that is already known to be dead.
//
// What the pass does:
//     Erases every ``fir.call`` whose callee matches a name in the module's
//     ``hlfir.stub_symbols`` attribute (set by ``set_stub_symbols``).  Runs
//     first in the pipeline so nothing downstream sees the dead calls.
//
// A stub call whose RESULT is consumed is skipped anyway, not rejected:
//     the call is still dropped, but each live result is replaced with a typed
//     zero so no consumer reads an undefined value.  This is the ``new_timer``
//     shape -- ``h = new_timer(...)`` feeding a no-op ``timer_start(h)``, where
//     the handle is inert and zero is as good as any id.  Because a zeroed
//     result COULD instead have fed live arithmetic, every drop emits a warning
//     (deduped per procedure), loudest where a result was zeroed, so a genuine
//     policy mistake is visible rather than silent.  Prefer an ExternalFunction
//     (emitted, callable) when the value is real.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct DropStubCallsPass : public mlir::PassWrapper<DropStubCallsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropStubCallsPass)

  llvm::StringRef getArgument() const final { return "hlfir-drop-stub-calls"; }
  llvm::StringRef getDescription() const final {
    return "Erase calls to do_not_emit (stub) procedures, whose calls are dropped at "
           "SDFG emission anyway; a consumed result is zeroed and warned, not rejected.";
  }

  /// Same matching as ``HLFIRModule::externalize_symbols``: bare symbol, or the
  /// ``...P<name>`` / ``_QP<name>`` Fortran manglings.
  static bool matches(llvm::StringRef sym, const llvm::SmallVectorImpl<std::string>& names) {
    for (const std::string& n : names)
      if (sym == n || sym.ends_with("P" + n) || sym.ends_with("_QP" + n)) return true;
    return false;
  }

  void runOnOperation() override {
    auto attr = getOperation()->getAttrOfType<mlir::ArrayAttr>("hlfir.stub_symbols");
    if (!attr || attr.empty()) return;
    llvm::SmallVector<std::string, 8> names;
    for (mlir::Attribute a : attr)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(a)) names.push_back(s.getValue().str());
    if (names.empty()) return;

    llvm::SmallVector<fir::CallOp, 16> dead;
    llvm::StringMap<unsigned> siteCount;    // callee -> total dropped call sites
    llvm::StringMap<unsigned> zeroedCount;  // callee -> live results replaced with zero
    getOperation().walk([&](fir::CallOp call) {
      auto callee = call.getCallee();
      if (!callee || !matches(callee->getRootReference().getValue(), names)) return;
      llvm::StringRef const name = callee->getRootReference().getValue();
      // The call is dropped whether or not it returns a value.  A live result
      // (``h = new_timer(...)`` feeding a no-op ``timer_start(h)``) is replaced
      // with a typed zero so no consumer reads an undefined value.
      mlir::OpBuilder b(call);
      for (mlir::Value res : call.getResults()) {
        if (res.use_empty()) continue;
        auto zero = b.create<fir::ZeroOp>(call.getLoc(), res.getType());
        res.replaceAllUsesWith(zero.getResult());
        zeroedCount[name]++;
      }
      siteCount[name]++;
      dead.push_back(call);
    });
    for (auto call : dead) call.erase();

    // A dropped stub is a silent change to the program, so make every drop
    // observable -- one deduped line per distinct procedure (not per call site:
    // ICON inlines thousands), loudest where a live result was zeroed.  Printed
    // via llvm::errs (as InlineAll / MarkBoundsRemapViews do) so it surfaces
    // regardless of the MLIR diagnostic handler.
    for (auto const& kv : siteCount) {
      unsigned const zeroed = zeroedCount.lookup(kv.first());
      llvm::errs() << "hlfir-drop-stub-calls: skipped do_not_emit procedure `" << kv.first() << "` (" << kv.second
                   << " call site(s))";
      if (zeroed)
        llvm::errs() << "; its result was consumed at " << zeroed
                     << " site(s) and replaced with zero -- verify this is an inert handle (e.g. a timer id), "
                        "not a value feeding live arithmetic";
      llvm::errs() << ".\n";
    }
    llvm::errs().flush();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createDropStubCallsPass() { return std::make_unique<DropStubCallsPass>(); }

}  // namespace hlfir_bridge
