// ============================================================================
// StripRuntimeIo.cpp  --  Delete calls to flang's Fortran I/O runtime.
// ============================================================================
// Problem:
//     Every Fortran ``WRITE`` / ``PRINT`` / ``FLUSH`` / ``OPEN`` / ``CLOSE``
//     statement is lowered by flang to a sequence of opaque runtime calls
//     into ``_FortranAio*`` library entries:
//
//       %cookie = fir.call @_FortranAioBeginExternalListOutput(...)
//       %ok1 = fir.call @_FortranAioOutputAscii(%cookie, %str_ptr, %len)
//       %ok2 = fir.call @_FortranAioOutputInteger32(%cookie, %val)
//       %iostat = fir.call @_FortranAioEndIoStatement(%cookie)
//
//     The bridge's SDFG is a numerical-equivalence model -- the SDFG runs
//     a kernel and compares its output arrays to a gfortran reference.
//     Diagnostic prints to ``stdout`` / ``stderr`` / a crash log are
//     orthogonal to that contract.  They aren't side-effects the SDFG can
//     model anyway (no DaCe primitive for "print from inside a parallel
//     map"), and inlining their cookie-threading chains into the kernel
//     bloats the IR with unreachable C-string format literals + opaque
//     library symbols that ``hlfir-inline-all`` then has to walk.
//
//     The same is true of ICON's stop_clock / start_clock when their
//     bodies survive past the error-helper strip pass: those bodies
//     reduce to a ``WRITE(stdout, ...) "no clock for ... found !"`` style
//     diagnostic, which is exactly what this pass deletes.
//
// Approach:
//     Walk the module for every ``fir.call`` whose callee starts with
//     ``_FortranAio``; replace each call's result(s) with a benign
//     constant matching the result type (``i1`` -> false, ``i32`` -> 0,
//     ``!fir.ref<i8>`` -> ``fir.zero_bits``), then erase the call.  After
//     the walk every Fortran I/O statement has been collapsed to dead
//     constants which the trailing ``canonicalize`` + ``symbol-dce``
//     sweep up.
//
//     Three runtime result types cover every ``_FortranAio*`` entry:
//       - ``i1``                : per-item OutputXxx / InputXxx status
//       - ``i32``               : final IoStatement return / iostat code
//       - ``!fir.ref<i8>``      : the IO cookie (BeginExternal* / etc.)
//
//     Users that read ``IOSTAT=`` into a variable get 0 (success), which
//     is the only correct answer for "this IO statement is now a no-op".
//
// Safety:
//     - Pure deletion + result replacement; never synthesises an abort
//       or a return.  Matches the bridge's existing
//       ``hlfir-strip-error-helpers`` contract: stay in the no-error,
//       no-output path of the source.
//     - A non-IO ``fir.call`` -- even one that happens to take an
//       ``!fir.ref<i8>`` argument or returns ``i32`` -- is untouched
//       (the match is on the callee symbol, not the signature).
//     - Calls whose callee starts with ``_FortranAio`` but happen to
//       have an SSA result type we don't recognise are left alone with
//       a debug log message; the caller's downstream uses are unmodified
//       and the rest of the pipeline surfaces any incompatibility.
//
// Pre-requisites:
//     Runs BEFORE ``hlfir-inline-all`` (same slot as
//     ``hlfir-strip-error-helpers``), so the inliner never has to walk
//     the cookie-threading IO chains.  Pre-pipeline state doesn't
//     matter -- the symbol-prefix match works on any IR.
//
// What this pass does NOT do:
//     - Reroute IO: there's no SDFG-side ``print`` tasklet.  If you
//       need the SDFG to emit diagnostics, this is the wrong tool.
//     - Match by signature shape: only the ``_FortranAio`` prefix is
//       checked.  flang's own runtime symbols are the only consumers
//       of that prefix.
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

#define DEBUG_TYPE "strip-runtime-io"

namespace hlfir_bridge {

namespace {

/// ``_FortranAio`` is the prefix flang uses for every Fortran I/O
/// runtime entry point.  No non-IO symbol shares this prefix, so a
/// pure-prefix match is unambiguous.
static constexpr llvm::StringLiteral kFortranIoPrefix = "_FortranAio";

/// Build a benign constant of ``ty`` immediately before ``call`` to
/// replace one of its results.  Returns ``nullptr`` if the result type
/// isn't one of the three the Fortran I/O runtime is known to produce.
static mlir::Value makeReplacement(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Type ty) {
  // ``i1`` (per-item ``Output*`` / ``Input*`` status) and ``i32``
  // (final ``EndIoStatement`` iostat code) -> a plain ``arith.constant``
  // of zero.  Zero is also the canonical "no error" iostat value, so
  // any user that reads ``WRITE(..., IOSTAT=ios)`` into a variable sees
  // the success path.
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    return builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIntegerAttr(intTy, 0));
  }
  // ``!fir.ref<i8>`` (the IO cookie threaded between ``Begin*`` and
  // subsequent ``Output*`` / ``End*`` calls) -> ``fir.zero_bits``,
  // which is the standard FIR null pointer.  Every downstream
  // consumer of this cookie is itself a ``_FortranAio*`` call we'll
  // also erase, so the null never escapes the stripped IO sequence.
  if (mlir::isa<fir::ReferenceType>(ty)) {
    return builder.create<fir::ZeroOp>(loc, ty);
  }
  return {};
}

// ---------------------------------------------------------------------------
// The pass.
// ---------------------------------------------------------------------------

struct StripRuntimeIoPass
    : public mlir::PassWrapper<StripRuntimeIoPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripRuntimeIoPass)

  llvm::StringRef getArgument() const final { return "hlfir-strip-runtime-io"; }
  llvm::StringRef getDescription() const final {
    return "Delete fir.call ops to flang's _FortranAio* I/O runtime "
           "(WRITE / PRINT / FLUSH / OPEN / CLOSE).  Diagnostic output is "
           "orthogonal to the SDFG's numerical-equivalence contract.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::SmallVector<fir::CallOp, 32> toErase;

    module.walk([&](fir::CallOp call) {
      auto sym = call.getCallee();
      if (!sym) return;  // indirect call
      llvm::StringRef name = sym->getLeafReference().getValue();
      if (!name.starts_with(kFortranIoPrefix)) return;

      // For each SSA result, replace uses with a benign constant.
      // The replacement op is inserted right before ``call`` so it
      // dominates every use the call's result currently has.
      mlir::OpBuilder builder(call);
      bool allReplaced = true;
      for (auto res : call.getResults()) {
        mlir::Value repl = makeReplacement(builder, call.getLoc(), res.getType());
        if (!repl) {
          LLVM_DEBUG(llvm::dbgs()
                     << "StripRuntimeIo: refusing to strip " << name
                     << " -- result type unsupported\n");
          allReplaced = false;
          break;
        }
        res.replaceAllUsesWith(repl);
      }
      if (allReplaced) toErase.push_back(call);
    });

    for (auto call : toErase) call->erase();

    LLVM_DEBUG(llvm::dbgs()
               << "StripRuntimeIo: erased " << toErase.size()
               << " _FortranAio* call(s)\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createStripRuntimeIoPass() {
  return std::make_unique<StripRuntimeIoPass>();
}

}  // namespace hlfir_bridge
