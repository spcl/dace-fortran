// ============================================================================
// StripErrorHelpers.cpp  --  Delete calls to abort-style error helpers.
// ============================================================================
// Problem:
//     Climate / NWP / quantum-chemistry Fortran codebases route invariant
//     violations through a single "fatal error" helper:
//
//       SUBROUTINE errore(routine, message, ierr)        ! QE
//         IF (ierr <= 0) RETURN
//         WRITE(*, ...) routine, message
//         STOP 1
//       END SUBROUTINE
//
//     ICON calls ``finish``, ECRAD calls ``radiation_abort`` /
//     ``dwarning``, NPB calls ``timer_stop`` -- the shape is the same:
//     early ``RETURN`` on the no-error branch, ``STOP`` / unreachable
//     terminator on the error branch.  Every kernel call site looks like
//
//       CALL errore("routine_name", "human message", ierr)
//
//     two of these typically guarding allocation results or MPI status.
//
//     The bridge's SDFG is a numerical-equivalence model: it compares
//     output arrays for inputs that don't trigger error paths.  When the
//     test harness supplies valid inputs the error branch is dead code;
//     ``STOP 1`` semantics cannot round-trip through DaCe anyway (there
//     is no clean "panic from inside the SDFG and propagate to Python"
//     primitive).  Faithfully inlining the helper buys nothing.
//
//     Worse, ``lift-cf-to-scf`` cannot structurize these callees because
//     ``STOP`` is a noreturn terminator the ``scf`` dialect doesn't
//     model, so the callee stays multi-block.  Splicing a multi-block
//     CFG into the caller's ``scf.if`` / ``scf.for`` region violates
//     ``scf``'s single-block invariant; flang's ``mlir::inlineCall``
//     walks the broken IR and crashes (observed: SIGSEGV inside
//     ``Region::cloneInto`` on QE's ``vexx_bp_k_gpu``).
//
// Approach:
//     Match callees by name against a known-error-helper list, then
//     delete every ``fir.call`` to them.  The orphan callee is left for
//     ``symbol-dce`` to clean up.  Default match list covers the names
//     observed in the climate / NWP / QE ecosystems (lowercase, mangled
//     and unmangled forms): ``errore``, ``error``, ``finish``,
//     ``abor1``, ``upf_error``, ``radiation_abort``, ``dwarning``.
//
//     The list is also reachable via the ``HLFIR_ERROR_HELPERS``
//     environment variable (comma-separated extra names appended to the
//     default list); future Python-API plumbing can pass a per-call
//     list through the same channel.  A name in the list that isn't
//     present in the module is a silent no-op.
//
// Safety:
//     - The pass deletes ONLY the call op, not the callee body itself.
//       ``symbol-dce`` removes the body if every caller was stripped;
//       any remaining caller keeps the body alive.
//     - Stripping a no-result call ("CALL errore(...)") is unambiguous;
//       the call has no SSA result to thread through downstream uses.
//       Defensively, the pass refuses to strip ``fir.call`` ops with
//       non-empty results -- a function-style error helper (``ierr =
//       check_err(...)``) needs a more surgical rewrite.
//     - Operands the call op consumed but never produced (the routine
//       name string, the message string) become dead arguments; they
//       were already dead in the source's no-error path and will be
//       DCE'd by ``canonicalize`` later in the pipeline.
//
// Pre-requisites:
//     Runs BEFORE ``hlfir-inline-all`` so the inliner never sees the
//     multi-block error helpers.  Pre-pipeline (``hlfir-prune-
//     unreachable`` / ``lift-cf-to-scf``) doesn't matter -- the
//     name-based match works on any IR.
//
// What this pass does NOT do:
//     - Reroute error semantics: there is no early-return-on-error
//       rewrite, no SDFG-level abort tasklet.  If you need the SDFG to
//       fail loudly on invariant violations, this pass is the wrong
//       tool (and DaCe is the wrong runtime).
//     - Match by signature shape: a project-specific helper that isn't
//       in the default list must be added via ``HLFIR_ERROR_HELPERS``.
// ============================================================================

#include <cstdlib>
#include <string>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

#define DEBUG_TYPE "strip-error-helpers"

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Default error-helper names.
// ---------------------------------------------------------------------------
//
// Covers names observed in CLOUDSC / ECRAD / ICON / QE / NPB.  Each entry is
// the *Fortran* name (lowercase); the symbol-name matcher below compares the
// last ``P``-delimited segment of the mangled name (``_QMfooPerrore`` ->
// ``errore``, ``_QPerrore`` -> ``errore``) so a single entry catches both
// module-procedure and free-subroutine forms.
//
// Add a project-specific helper at runtime via the ``HLFIR_ERROR_HELPERS``
// environment variable: a comma-separated list of additional lowercase
// names appended to this default set.
const char* const kDefaultErrorHelpers[] = {
    "errore",           // Quantum ESPRESSO
    "error",            // generic; many codes
    "finish",           // ICON
    "abort_mpi",        // ICON -- wraps MPI_Abort; noreturn error-abort path
    "mpi_abort",        // bare MPI_Abort (if the abort_mpi wrapper was inlined first)
    "abor1",            // ECMWF IFS
    "abor1_sfx",        // ECMWF IFS SURFEX
    "upf_error",        // Quantum ESPRESSO UPF
    "radiation_abort",  // ECRAD
    "dwarning",         // ECRAD
    // ---- Diagnostic timing / annotation subroutines.  These are not
    // error abort paths but they're equally orthogonal to the bridge's
    // numerical-equivalence contract: they wrap timing instrumentation
    // and NVTX range markers that the SDFG does not model.  Stripping
    // the CALL keeps their inlined bodies (which would otherwise drag
    // in unresolvable runtime functions like ``f_tcpu`` / ``f_wall``)
    // out of ``hlfir-inline-all``.  Same trailing-``P``-segment
    // demangling as the error helpers, so module-procedure forms
    // (``_QMtiming_modPstart_clock``) match the same entries.
    "start_clock",     // Quantum ESPRESSO timing
    "stop_clock",      // Quantum ESPRESSO timing
    "nvtxstartrange",  // NVIDIA NVTX range marker (QE GPU port)
    "nvtxendrange",    // NVIDIA NVTX range marker (QE GPU port)
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the trailing ``P``-delimited segment of a flang-mangled name.
///
///   ``_QMmodPname`` -> ``name``
///   ``_QPname``     -> ``name``
///   ``name``        -> ``name``
///
/// Lowercased so the comparison against the default name table is
/// case-insensitive.
std::string demangleTail(llvm::StringRef sym) {
  auto last_p = sym.rfind('P');
  llvm::StringRef const tail = last_p == llvm::StringRef::npos ? sym : sym.substr(last_p + 1);
  // ``_QQ`` / numeric mangled suffixes never carry a tail ``P`` in the
  // function-name segment, so the rfind('P') is safe for the cases we
  // care about.  Lowercase manually (StringRef has no .lower()).
  std::string out;
  out.reserve(tail.size());
  for (char const c : tail) out.push_back(static_cast<char>(std::tolower(c)));
  return out;
}

/// Build the active match set: defaults plus anything appended via the
/// ``HLFIR_ERROR_HELPERS`` env var (comma-separated, whitespace-trimmed,
/// lowercased).
llvm::StringSet<> buildMatchSet() {
  llvm::StringSet<> names;
  for (const char* d : kDefaultErrorHelpers) names.insert(d);
  if (const char* extra = std::getenv("HLFIR_ERROR_HELPERS")) {
    llvm::StringRef src(extra);
    while (!src.empty()) {
      auto split = src.split(',');
      llvm::StringRef const tok = split.first.trim();
      if (!tok.empty()) {
        std::string lower;
        lower.reserve(tok.size());
        for (char const c : tok) lower.push_back(static_cast<char>(std::tolower(c)));
        names.insert(lower);
      }
      src = split.second;
    }
  }
  return names;
}

// ---------------------------------------------------------------------------
// The pass.
// ---------------------------------------------------------------------------

struct StripErrorHelpersPass : public mlir::PassWrapper<StripErrorHelpersPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripErrorHelpersPass)

  llvm::StringRef getArgument() const final { return "hlfir-strip-error-helpers"; }
  llvm::StringRef getDescription() const final {
    return "Delete fir.call ops whose callee matches a known abort-style "
           "error helper name (errore, finish, abor1, ...).  Runs before "
           "hlfir-inline-all to keep multi-block error helpers out of the "
           "inliner's hands.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    llvm::StringSet<> names = buildMatchSet();

    llvm::SmallVector<fir::CallOp, 16> toErase;
    module.walk([&](fir::CallOp call) {
      auto sym = call.getCallee();
      if (!sym) return;  // indirect call
      std::string const tail = demangleTail(sym->getLeafReference());
      if (!names.contains(tail)) return;

      // Defensive: only strip CALL-style (no SSA result) sites.
      // A function-style error helper that returns a status code would
      // need a more surgical rewrite (replace the call with a constant
      // zero / success value) -- we don't speculate on the return type.
      if (call->getNumResults() != 0) {
        LLVM_DEBUG(llvm::dbgs() << "StripErrorHelpers: refusing to strip " << *sym << " -- call has "
                                << call->getNumResults() << " result(s); needs explicit rewrite\n");
        return;
      }

      toErase.push_back(call);
    });

    for (auto call : toErase) call->erase();

    LLVM_DEBUG(llvm::dbgs() << "StripErrorHelpers: erased " << toErase.size() << " error-helper call site(s)\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createStripErrorHelpersPass() { return std::make_unique<StripErrorHelpersPass>(); }

}  // namespace hlfir_bridge
