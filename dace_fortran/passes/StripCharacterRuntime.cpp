// ============================================================================
// StripCharacterRuntime.cpp  --  Delete calls to flang's Fortran character
//                                runtime that don't carry numerical content.
// ============================================================================
// Problem:
//     Fortran string operations (``CHARACTER`` comparisons, ``TRIM``,
//     ``ADJUSTL`` / ``ADJUSTR``, etc.) lower to opaque runtime calls
//     into ``_FortranACharacter*`` library entries:
//
//       %res = fir.call @_FortranACharacterCompareScalar1(
//                  %a, %b, %la, %lb) : (... ) -> i32
//
//     Flang inlines these from helper routines like ``start_clock(name)``
//     that dispatch on a name string by scanning a clock-name table.
//     The bridge's SDFG is a numerical-equivalence model -- string-keyed
//     dispatch into diagnostic / timing code is orthogonal to that
//     contract.  Worse, the runtime call's i32 result is consumed by
//     ``arith.cmpi`` chains the AST builder traces with ``leafExpr``,
//     and the runtime callee is not recognised: the renderer falls
//     through to ``?``, then ``emit_scalar_assign`` emits
//     ``_out = (? == 0)`` and DaCe's ``ast.parse`` rejects it.
//
// Approach:
//     Walk the module for every ``fir.call`` whose callee starts with
//     ``_FortranACharacter``.  Replace each call's SSA result(s) with
//     a benign constant matching the result type (``i32`` -> ``0``,
//     i.e. "strings compare equal", which collapses the downstream
//     ``cmpi eq %res, 0`` to constant true and lets the canonicalizer
//     fold the dispatch into the matching arm).  Calls with no result
//     (``Trim``) get erased outright -- the destination box stays
//     uninitialised, but the bridge doesn't model character data
//     anyway and the downstream chain dies in the AST builder's
//     character handler.
//
// Safety:
//     - Pure deletion + result replacement; never synthesises an
//       abort or a return.  Matches the bridge's existing
//       ``hlfir-strip-error-helpers`` / ``hlfir-strip-runtime-io``
//       contract: stay in the no-error, no-output path of the source.
//     - A non-character ``fir.call`` is untouched (the match is on
//       the callee symbol prefix, not the signature).
//     - Calls with result types the pass doesn't recognise are
//       skipped with a debug log message; downstream uses are left
//       alone.
//
// Pre-requisites:
//     Runs BEFORE ``hlfir-inline-all`` (alongside ``strip-error-
//     helpers`` and ``strip-runtime-io``) so the inliner never has
//     to walk the runtime-character chains, and so the AST builder
//     never sees them.
// ============================================================================

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

#define DEBUG_TYPE "strip-character-runtime"

namespace hlfir_bridge {

namespace {

/// ``_FortranACharacter`` is the prefix flang uses for every
/// character-domain runtime entry point.  No non-character symbol
/// shares this prefix.
static constexpr llvm::StringLiteral kFortranCharPrefix = "_FortranACharacter";

/// Build a benign constant of ``ty`` immediately before ``call`` to
/// replace one of its results.  Returns ``nullptr`` if the result
/// type isn't one of the integer types the Fortran character
/// runtime is known to produce (currently always ``i32``).
static mlir::Value makeReplacement(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type ty) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(intTy, 0));
  }
  return {};
}

struct StripCharacterRuntimePass
    : public mlir::PassWrapper<StripCharacterRuntimePass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripCharacterRuntimePass)

  llvm::StringRef getArgument() const final { return "hlfir-strip-character-runtime"; }
  llvm::StringRef getDescription() const final {
    return "Delete fir.call ops to flang's _FortranACharacter* runtime "
           "(string compare / Trim / Adjust / ...) -- the bridge's "
           "numerical-equivalence contract does not model character data.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::SmallVector<fir::CallOp, 32> toErase;

    module.walk([&](fir::CallOp call) {
      auto sym = call.getCallee();
      if (!sym) return;
      llvm::StringRef name = sym->getLeafReference().getValue();
      if (!name.starts_with(kFortranCharPrefix)) return;

      mlir::OpBuilder builder(call);
      bool allReplaced = true;
      for (auto res : call.getResults()) {
        mlir::Value repl = makeReplacement(builder, call.getLoc(), res.getType());
        if (!repl) {
          LLVM_DEBUG(llvm::dbgs() << "StripCharacterRuntime: refusing to strip " << name
                                  << " -- result type unsupported\n");
          allReplaced = false;
          break;
        }
        res.replaceAllUsesWith(repl);
      }
      if (allReplaced) toErase.push_back(call);
    });

    for (auto call : toErase) call->erase();

    LLVM_DEBUG(llvm::dbgs() << "StripCharacterRuntime: erased " << toErase.size() << " _FortranACharacter* call(s)\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createStripCharacterRuntimePass() { return std::make_unique<StripCharacterRuntimePass>(); }

}  // namespace hlfir_bridge
