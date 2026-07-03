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
//     FILE-bound I/O is a separate story: ``OPEN(newunit=u, file='cfg.nml')
//     ... READ (u, *) y`` lowers to a SetFile + BeginExternalListInput +
//     InputDescriptor + EndIoStatement chain that the AST-extraction-
//     time recognizer (``bridge/ast/dispatch.cpp::recognizeIoCall``)
//     maps to ``dace.libraries.fortran_io`` library nodes -- those
//     transfers ARE part of the kernel's contract.  Stripping them
//     would silently drop the data load, so a ``y = [1.5, 2.5]`` test
//     reads ``[0.0, 0.0]`` instead.  The strip pass must therefore
//     preserve file-bound chains and only erase stdout / stderr ones.
//
// Approach:
//     Walk the module for every ``fir.call`` whose callee starts with
//     ``_FortranAio``.  Walk each ``Begin*`` call's cookie SSA value
//     forward to gather the whole chain (everything that uses the
//     cookie transitively up to ``EndIoStatement``).  Mark the chain
//     as file-bound if ANY member is a ``SetFile`` call -- the
//     SetFile pattern is the universal flang marker for "this IO
//     targets a named file" (both OPEN's chain and any subsequent
//     read/write chain that uses an opened unit go through a
//     SetFile in the OPEN sequence).
//
//     Then for every IO call, replace its SSA result(s) with a benign
//     constant matching the result type (``i1`` -> false, ``i32`` ->
//     0, ``!fir.ref<i8>`` -> ``fir.zero_bits``) and erase, BUT ONLY
//     when the call is NOT part of any file-bound chain.  After the
//     walk every stdout / stderr IO statement has been collapsed to
//     dead constants which the trailing ``canonicalize`` +
//     ``symbol-dce`` sweep up; file-bound chains pass through to the
//     AST-extraction-time recognizer untouched.
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
static mlir::Value makeReplacement(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type ty) {
  // ``i1`` (per-item ``Output*`` / ``Input*`` status) and ``i32``
  // (final ``EndIoStatement`` iostat code) -> a plain ``arith.constant``
  // of zero.  Zero is also the canonical "no error" iostat value, so
  // any user that reads ``WRITE(..., IOSTAT=ios)`` into a variable sees
  // the success path.
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(intTy, 0));
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

struct StripRuntimeIoPass : public mlir::PassWrapper<StripRuntimeIoPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripRuntimeIoPass)

  llvm::StringRef getArgument() const final { return "hlfir-strip-runtime-io"; }
  llvm::StringRef getDescription() const final {
    return "Delete fir.call ops to flang's _FortranAio* I/O runtime that "
           "target "
           "stdout / stderr (WRITE * / PRINT / stop_clock diagnostic).  File-"
           "bound IO chains (anything reaching a SetFile call) are preserved "
           "for the AST-extraction-time recognizer.";
  }

  /// True iff ``func`` contains a ``_FortranAioSetFile`` call.  The
  /// SetFile runtime entry is emitted by every ``OPEN (..., FILE=
  /// '...')`` lowering, so its presence in a function is a tight
  /// proxy for "this function does file-bound I/O somewhere".  Per-
  /// function scope works because each subroutine's IO chains are
  /// self-contained: an ``OPEN`` and the ``READ`` / ``WRITE`` /
  /// ``CLOSE`` that use its unit all live inside the same
  /// ``func.func``.
  static bool funcTouchesFile(mlir::func::FuncOp func) {
    bool found = false;
    func.walk([&](fir::CallOp call) {
      if (found) return mlir::WalkResult::interrupt();
      auto sym = call.getCallee();
      if (sym && sym->getLeafReference().getValue().contains("SetFile")) {
        found = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return found;
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::SmallVector<fir::CallOp, 32> toErase;

    module.walk([&](mlir::func::FuncOp func) {
      // Functions that touch file IO (any ``SetFile`` call in scope)
      // keep their entire ``_FortranAio*`` chain intact -- the AST-
      // extraction-time recognizer needs to walk the SetFile +
      // BeginExternal* + EndIoStatement sequence to map ``OPEN`` /
      // ``READ`` / ``WRITE`` / ``CLOSE`` to library nodes.
      // Functions that don't touch file IO are diagnostic-only
      // (stdout writes, PRINT, stop_clock chatter); we strip
      // everything in them so ``hlfir-inline-all`` doesn't walk
      // their cookie chains.
      if (funcTouchesFile(func)) return;
      func.walk([&](fir::CallOp call) {
        auto sym = call.getCallee();
        if (!sym) return;
        llvm::StringRef name = sym->getLeafReference().getValue();
        if (!name.starts_with(kFortranIoPrefix)) return;
        mlir::OpBuilder builder(call);
        bool allReplaced = true;
        for (auto res : call.getResults()) {
          mlir::Value repl = makeReplacement(builder, call.getLoc(), res.getType());
          if (!repl) {
            LLVM_DEBUG(llvm::dbgs() << "StripRuntimeIo: refusing to strip " << name << " -- result type unsupported\n");
            allReplaced = false;
            break;
          }
          res.replaceAllUsesWith(repl);
        }
        if (allReplaced) toErase.push_back(call);
      });
    });

    for (auto call : toErase) call->erase();

    LLVM_DEBUG(llvm::dbgs() << "StripRuntimeIo: erased " << toErase.size() << " stdout-bound _FortranAio* call(s)\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createStripRuntimeIoPass() { return std::make_unique<StripRuntimeIoPass>(); }

}  // namespace hlfir_bridge
