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
//     ``_FortranACharacter`` (LLVM 22: ``hlfir.cmpchar`` too).  A COMPARE
//     against a character literal folds per literal: the first literal a
//     given entity is tested against is the one that reads as equal, every
//     other literal reads as not-equal.  The canonicalizer then bakes the
//     first dispatch arm AND drops the sibling arms -- including the
//     separate input-validation IFs that reference them, which under the
//     old fold-everything-equal rule stayed live and kept the dead arms'
//     optionals on the SDFG's call surface (QE addusxx_g / newdxx_g).
//     Every other character call keeps the benign ``0``.  Calls with no result
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
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
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
constexpr llvm::StringLiteral kFortranCharPrefix = "_FortranACharacter";

/// Strip ``fir.convert`` / ``hlfir.declare`` wrappers so two references to the
/// same character entity compare identical.
mlir::Value charRoot(mlir::Value v) {
  for (int hop = 0; hop < 8 && v; ++hop) {
    auto* def = v.getDefiningOp();
    if (!def) break;
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = conv.getValue();
      continue;
    }
    if (auto emb = mlir::dyn_cast<fir::EmboxCharOp>(def)) {
      v = emb.getMemref();
      continue;
    }
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      // A PARAMETER declare over a literal global is the literal itself; any
      // other declare IS the entity, so stop there.
      auto attrs = decl.getFortranAttrs();
      if (!attrs || !bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::parameter)) break;
      v = decl.getMemref();
      continue;
    }
    break;
  }
  return v;
}

/// The character literal ``v`` refers to, or empty when it is not a literal.
/// Flang materialises every literal as a ``fir.address_of`` of a constant
/// ``fir.global`` whose body is a single ``fir.string_lit``.
std::string literalText(mlir::Value v, mlir::ModuleOp module) {
  auto addr = mlir::dyn_cast_or_null<fir::AddrOfOp>(charRoot(v).getDefiningOp());
  if (!addr) return {};
  auto global = module.lookupSymbol<fir::GlobalOp>(addr.getSymbol().getLeafReference());
  if (!global || !global.getConstant().value_or(false)) return {};
  std::string text;
  global.walk([&](fir::StringLitOp lit) {
    if (auto str = mlir::dyn_cast<mlir::StringAttr>(lit.getValue())) text = str.getValue().str();
  });
  return text;
}

bool equalIgnoringCase(llvm::StringRef a, llvm::StringRef b) { return a.equals_insensitive(b); }

/// Per-dispatch-variable record of the ONE literal its compares are folded
/// against: the first literal the routine tests it against, which is the arm the
/// canonicalizer then bakes.
///
/// Folding EVERY compare to "equal" (the pass's original behaviour) makes a
/// ``flag=='c'/'r'/'i'`` dispatch bake its first arm, but leaves the routine's
/// SEPARATE input-validation IFs consulting the dead arms -- QE's addusxx_g then
/// demanded the 'r' arm's optionals be present while its 'c' arm computed, and
/// the dead arms' arrays stayed on the SDFG's call surface.  Folding against ONE
/// literal instead is consistent everywhere the flag appears: the dead arms'
/// validation clauses go FALSE and canonicalize away with their operands.
class LiteralSelection {
 public:
  /// Whether a compare of ``var`` against ``lit`` is the selected arm. The first
  /// literal seen for a variable wins, so the baked arm is the source's first --
  /// the contract the sample pipelines already document.
  bool selects(mlir::Value var, llvm::StringRef lit) {
    auto it = selected.find(var);
    if (it == selected.end()) {
      selected.try_emplace(var, lit.str());
      return true;
    }
    return equalIgnoringCase(it->second, lit);
  }

 private:
  llvm::DenseMap<mlir::Value, std::string> selected;
};

/// Classify one character compare: ``(matched, isLiteralCompare)``. A compare with
/// no literal operand keeps the old "strings compare equal" answer.
std::pair<bool, bool> foldCharCompare(mlir::Value lhs, mlir::Value rhs, mlir::ModuleOp module,
                                      LiteralSelection& selection) {
  std::string const lhsLit = literalText(lhs, module);
  std::string const rhsLit = literalText(rhs, module);
  if (lhsLit.empty() == rhsLit.empty()) return {true, false};  // both or neither literal
  mlir::Value const var = charRoot(lhsLit.empty() ? lhs : rhs);
  llvm::StringRef const lit = lhsLit.empty() ? rhsLit : lhsLit;
  return {selection.selects(var, lit), true};
}

/// Build a benign constant of ``ty`` immediately before ``call`` to
/// replace one of its results.  Returns ``nullptr`` if the result
/// type isn't one of the integer types the Fortran character
/// runtime is known to produce (currently always ``i32``).
mlir::Value makeReplacement(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type ty, int value) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty)) {
    return builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(intTy, value));
  }
  return {};
}

#if LLVM_VERSION_MAJOR >= 22
/// Value of ``pred`` when both operands compare equal, matching the LLVM 21 call path.
bool cmpCharEqualResult(mlir::arith::CmpIPredicate pred) {
  using P = mlir::arith::CmpIPredicate;
  switch (pred) {
    case P::eq:
    case P::sle:
    case P::ule:
    case P::sge:
    case P::uge:
      return true;
    default:
      return false;
  }
}

/// Result of ``pred`` given whether the two strings are equal. Ordering compares
/// carry no per-literal information, so they keep the equal-strings answer.
bool comparePredicateResult(mlir::arith::CmpIPredicate pred, bool equal) {
  using P = mlir::arith::CmpIPredicate;
  if (pred == P::eq) return equal;
  if (pred == P::ne) return !equal;
  return cmpCharEqualResult(pred);
}

void stripCmpChar(mlir::ModuleOp module, LiteralSelection& selection) {
  llvm::SmallVector<hlfir::CmpCharOp, 8> toErase;
  module.walk([&](hlfir::CmpCharOp cmp) {
    auto const [matched, isLiteral] = foldCharCompare(cmp.getLchr(), cmp.getRchr(), module, selection);
    bool const result =
        isLiteral ? comparePredicateResult(cmp.getPredicate(), matched) : cmpCharEqualResult(cmp.getPredicate());
    mlir::OpBuilder builder(cmp);
    auto repl = builder.create<mlir::arith::ConstantOp>(cmp.getLoc(), builder.getBoolAttr(result));
    cmp.getResult().replaceAllUsesWith(repl);
    toErase.push_back(cmp);
  });
  for (auto cmp : toErase) cmp->erase();
  LLVM_DEBUG(llvm::dbgs() << "StripCharacterRuntime: folded " << toErase.size() << " hlfir.cmpchar op(s)\n");
}
#endif

struct StripCharacterRuntimePass
    : public mlir::PassWrapper<StripCharacterRuntimePass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
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
    LiteralSelection selection;

    module.walk([&](fir::CallOp call) {
      auto sym = call.getCallee();
      if (!sym) return;
      llvm::StringRef const name = sym->getLeafReference().getValue();
      if (!name.starts_with(kFortranCharPrefix)) return;

      // ``CompareScalar<kind>(%a, %b, %la, %lb) -> i32`` returns 0 for equal.
      // Fold it per literal, so only the arm the dispatch bakes reads as a match.
      int compareResult = 0;
      if (name.contains("Compare") && call.getArgs().size() >= 2) {
        auto const [matched, isLiteral] = foldCharCompare(call.getArgs()[0], call.getArgs()[1], module, selection);
        if (isLiteral && !matched) compareResult = 1;
      }

      mlir::OpBuilder builder(call);
      bool allReplaced = true;
      for (auto res : call.getResults()) {
        mlir::Value const repl = makeReplacement(builder, call.getLoc(), res.getType(), compareResult);
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

#if LLVM_VERSION_MAJOR >= 22
    stripCmpChar(module, selection);
#endif
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createStripCharacterRuntimePass() { return std::make_unique<StripCharacterRuntimePass>(); }

}  // namespace hlfir_bridge
