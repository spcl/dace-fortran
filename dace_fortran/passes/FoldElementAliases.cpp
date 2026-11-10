// ============================================================================
// FoldElementAliases.cpp  --  erase declare-over-designate aliases.
// ============================================================================
//
// Motivation:
//     After ``hlfir-inline-all`` splices an elemental (or any scalar-arg
//     procedure) into a caller whose actual arguments are array elements,
//     Flang leaves an ``hlfir.declare`` per callee dummy whose memref is
//     the outer array's ``hlfir.designate`` at the current loop index:
//
//         %e = hlfir.designate %outer (%i) : ref<array<Nxf64>>, index ->
//         ref<f64> %d:2 = hlfir.declare %e {uniq_name = "_QMmodFcalleeEx"} :
//         ... hlfir.assign %val to %d#0 : f64, ref<f64>
//
//     The declare exists only to carry the callee's Fortran name into the
//     inlined body  --  it aliases one element of the outer array and adds
//     no semantics the caller didn't already have.
//
//     For the SDFG builder this second declare is active poison:
//       * ``extract_vars`` sees a second VarInfo with the callee's name
//         over a scalar ref, and the SDFG arglist grows a stray ``x`` /
//         ``od`` / ``g`` / ... scalar that the caller never needs to supply.
//       * The write ``hlfir.assign %val to %d#0`` looks like a plain
//         scalar-store, so our frontend never connects the write back
//         to the outer array at the outer's loop index.
//
// What the pass does:
//     For each ``hlfir.declare`` with no shape operand and whose memref
//     is ``hlfir.designate`` of *another* declare, replace every use of
//     the alias's results with the designate itself and erase the
//     alias.  After this, ``hlfir.assign %val to %d#0`` collapses to
//     ``hlfir.assign %val to %designate``  --  a regular indexed store the
//     SDFG builder already handles.
//
// Scope:
//     Only element-alias declares are folded.  Assumed-shape aliases
//     (memref via ``fir.convert`` of a box) are left to the bridge's
//     runtime rebase path (``trace_utils::asAssumedShapeAlias`` +
//     ``extract_ast::buildDesignateIndexExpr``)  --  their semantics
//     require an index rewrite, not a straight replacement.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// True iff ``v`` is computed from a ``fir.box_dims`` of ``box``.
bool derivesFromBoxDims(mlir::Value v, mlir::Value box, int depth = 0) {
  auto* d = v.getDefiningOp();
  if (!d || depth > 4) return false;
  if (auto dims = mlir::dyn_cast<fir::BoxDimsOp>(d)) return dims.getVal() == box;
  return llvm::any_of(d->getOperands(), [&](mlir::Value o) { return derivesFromBoxDims(o, box, depth + 1); });
}

/// True iff a triplet of section ``dg`` starts past its parent's first element, or strides.
bool shiftsParent(hlfir::DesignateOp dg) {
  auto isOne = [](mlir::Value v) {
    auto c = fir::getIntIfConstant(v);
    return c && *c == 1;
  };
  auto indices = dg.getIndices();
  unsigned cursor = 0;
  for (bool const triplet : dg.getIsTriplet()) {
    if (!triplet) {
      ++cursor;
      continue;
    }
    mlir::Value const lo = indices[cursor];
    mlir::Value const stride = indices[cursor + 2];
    cursor += 3;
    if (!isOne(stride) || !(isOne(lo) || derivesFromBoxDims(lo, dg.getMemref()))) return true;
  }
  return false;
}

struct FoldElementAliasesPass : public mlir::PassWrapper<FoldElementAliasesPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldElementAliasesPass)

  llvm::StringRef getArgument() const final { return "hlfir-fold-element-aliases"; }
  llvm::StringRef getDescription() const final {
    return "Erase hlfir.declare ops whose memref is an hlfir.designate of "
           "another declare (element-scoped aliases left behind by "
           "hlfir-inline-all on elemental / scalar-arg calls).";
  }

  /// The declare whose explicit-shape array ``c`` reinterprets as one element of it: an array actual of a scalar
  /// dummy (QE ``gk_sort(..., ngk, ...)``), which is sequence association with its first element.
  static hlfir::DeclareOp matchArrayToScalar(fir::ConvertOp c) {
    auto to = mlir::dyn_cast<fir::ReferenceType>(c.getType());
    auto from = mlir::dyn_cast<fir::ReferenceType>(c.getValue().getType());
    auto seq = from ? mlir::dyn_cast<fir::SequenceType>(from.getEleTy()) : nullptr;
    if (!to || !seq || seq.getEleTy() != to.getEleTy()) return {};
    auto decl = c.getValue().getDefiningOp<hlfir::DeclareOp>();
    if (!decl || c.getValue() != decl.getResult(0) || !decl.getShape()) return {};
    if (!mlir::isa<fir::ShapeOp, fir::ShapeShiftOp>(decl.getShape().getDefiningOp())) return {};
    return decl;
  }

  static void rewriteToFirstElement(fir::ConvertOp c) {
    auto decl = matchArrayToScalar(c);
    mlir::OpBuilder b(c);
    llvm::SmallVector<mlir::Value, 4> first;
    if (auto ss = decl.getShape().getDefiningOp<fir::ShapeShiftOp>()) {
      for (mlir::Value const lb : ss.getOrigins()) first.push_back(lb);
    } else {
      auto rank = decl.getShape().getDefiningOp<fir::ShapeOp>().getExtents().size();
      first.assign(rank, b.create<mlir::arith::ConstantIndexOp>(c.getLoc(), 1));
    }
    auto element = b.create<hlfir::DesignateOp>(c.getLoc(), c.getType(), decl.getResult(0), mlir::ValueRange{first});
    c.replaceAllUsesWith(element.getResult());
    c.erase();
  }

  void runOnOperation() override {
    llvm::SmallVector<fir::ConvertOp, 8> toScalar;
    getOperation().walk([&](fir::ConvertOp c) {
      if (matchArrayToScalar(c)) toScalar.push_back(c);
    });
    for (auto c : toScalar) rewriteToFirstElement(c);

    llvm::SmallVector<hlfir::DeclareOp, 16> toErase;

    getOperation().walk([&](hlfir::DeclareOp decl) {
      if (decl.getShape()) return;  // has shape  --  not an alias
      auto memref = decl.getMemref();
      auto* mrd = memref.getDefiningOp();
      if (!mrd) return;
      auto designate = mlir::dyn_cast<hlfir::DesignateOp>(mrd);
      if (!designate) return;
      // A section starting past its parent's first element, or striding, needs the alias: its elements are not the
      // parent's at the same index.
      if (shiftsParent(designate)) return;

      // Confirm the designate's base ultimately resolves to a
      // declare  --  without that anchor we don't have a "real"
      // outer storage to point uses at.  (Designates nested
      // inside designates would be handled transitively as each
      // outer round erases another layer.)
      //
      // ELEMENTAL / scalar-arg inlining over a section slice
      // adds wrapper layers: Flang section-designates the
      // caller's slice (and may embox / rebox / convert it on
      // top) before passing to the inlined dummy.  Walk through
      // hlfir.designate (section slices) and fir.embox /
      // fir.rebox / fir.convert (box wrappers) between the
      // element designate and the underlying declare so the
      // fold also applies to:
      //
      //     %slice = hlfir.designate %parent#0 (%lo:%hi:%step)
      //     %elem  = hlfir.designate %slice (%loop_iv) -> ref<scalar>
      //     %alias = hlfir.declare %elem ...
      //
      // We only need to confirm a declare exists at the chain
      // root  --  the replacement target stays the element
      // designate ``memref``, regardless of intermediate hops.
      auto* base = designate.getMemref().getDefiningOp();
      for (int hop = 0; hop < 8 && base; ++hop) {
        if (mlir::isa<hlfir::DeclareOp>(base)) break;
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(base)) {
          base = dg.getMemref().getDefiningOp();
          continue;
        }
        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(base)) {
          base = eb.getMemref().getDefiningOp();
          continue;
        }
        if (auto rb = mlir::dyn_cast<fir::ReboxOp>(base)) {
          base = rb.getBox().getDefiningOp();
          continue;
        }
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(base)) {
          base = cv.getValue().getDefiningOp();
          continue;
        }
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(base)) {
          // ALLOCATABLE / POINTER parent: walk through the loaded box so the
          // chain reaches the declare root (otherwise it stops at the load).
          base = ld.getMemref().getDefiningOp();
          continue;
        }
        base = nullptr;
        break;
      }
      if (!base || !mlir::isa<hlfir::DeclareOp>(base)) return;

      decl.getResult(0).replaceAllUsesWith(memref);
      decl.getResult(1).replaceAllUsesWith(memref);
      toErase.push_back(decl);
    });

    for (auto d : toErase) d.erase();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldElementAliasesPass() { return std::make_unique<FoldElementAliasesPass>(); }

}  // namespace hlfir_bridge
