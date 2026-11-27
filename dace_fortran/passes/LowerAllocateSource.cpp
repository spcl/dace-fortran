// Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
// SPDX-License-Identifier: GPL-3.0-or-later
// ============================================================================
// LowerAllocateSource.cpp  --  ALLOCATE(x, SOURCE=y) as allocmem + x = y.
// ============================================================================
// Problem:
//     flang lowers a plain ``ALLOCATE(x(n, m))`` inline (fir.allocmem + embox + store), but
//     ``ALLOCATE(x, SOURCE=y)`` to runtime calls:
//
//       fir.call @_FortranAAllocatableSetBounds(x, dim, lb, ub)     ! once per dim
//       fir.call @_FortranAAllocatableAllocateSource(x, y, ...)
//
//     The bridge models neither call, so ``x`` kept free shape symbols and never received
//     ``y``'s values (QE vexx_bp_k_gpu: ``ALLOCATE(psi_d, SOURCE = psi)`` gave an empty psi_d,
//     and the whole exchange term came out zero).
//
// What the pass does:
//     Rewrites each such pair into the IR flang emits for ``ALLOCATE(x(lb:ub, ...)); x = y``:
//     fir.allocmem with the SetBounds extents, fir.shape + fir.embox + fir.store into x's
//     descriptor, then ``hlfir.assign y to x realloc``.  The runtime's stat result becomes 0.
//
//     A scalar allocatable gets the same without a shape (ICON's ``ALLOCATE(x, SOURCE=y)`` of a scalar).
//
// Limits (rejected with an error, never lowered silently wrong):
//     a lower bound other than the literal 1, a missing SetBounds for some dim.  A polymorphic
//     allocatable keeps its runtime call, with a warning: the whole module runs this pass, and ICON's
//     polymorphic SOURCE= sit outside the kernels the bridge builds.
// ============================================================================

#include <optional>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

mlir::Value peelConverts(mlir::Value v) {
  while (auto cv = v.getDefiningOp<fir::ConvertOp>()) v = cv.getValue();
  return v;
}

std::optional<int64_t> constInt(mlir::Value v) {
  if (auto c = peelConverts(v).getDefiningOp<mlir::arith::ConstantOp>())
    if (auto attr = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return attr.getInt();
  return std::nullopt;
}

/// The source's declared extent of ``dim`` when its descriptor is built from a fir.shape (an explicit-shape dummy:
/// the extent the rest of the IR names), else null. The SetBounds bound is the same value read back through
/// fir.box_dims, which the bridge does not trace to a symbol.
mlir::Value declaredExtent(mlir::Value source, unsigned dim) {
  mlir::Value shape;
  if (auto embox = source.getDefiningOp<fir::EmboxOp>())
    shape = embox.getShape();
  else if (auto decl = source.getDefiningOp<hlfir::DeclareOp>())
    shape = decl.getShape();
  auto shapeOp = shape ? shape.getDefiningOp<fir::ShapeOp>() : fir::ShapeOp{};
  return shapeOp && dim < shapeOp.getExtents().size() ? shapeOp.getExtents()[dim] : mlir::Value{};
}

llvm::StringRef calleeName(fir::CallOp call) {
  auto callee = call.getCallee();
  return callee ? callee->getRootReference().getValue() : llvm::StringRef{};
}

struct LowerAllocateSourcePass
    : public mlir::PassWrapper<LowerAllocateSourcePass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerAllocateSourcePass)

  llvm::StringRef getArgument() const final { return "hlfir-lower-allocate-source"; }
  llvm::StringRef getDescription() const final {
    return "Rewrite ALLOCATE(x, SOURCE=y) runtime calls into fir.allocmem + hlfir.assign y to x.";
  }

  void runOnOperation() override {
    llvm::SmallVector<fir::CallOp, 8> sources;
    getOperation().walk([&](fir::CallOp call) {
      if (calleeName(call) == "_FortranAAllocatableAllocateSource") sources.push_back(call);
    });
    for (fir::CallOp call : sources)
      if (mlir::failed(lower(call))) return signalPassFailure();
  }

  // ``ALLOCATE(x, SOURCE=y)`` of a scalar: ``x`` gets a fresh scalar, then ``x = y``.
  static mlir::LogicalResult lowerScalar(fir::CallOp call, mlir::Value boxRef, mlir::Value source, fir::BoxType boxTy,
                                         fir::HeapType heapTy) {
    auto decl = boxRef.getDefiningOp<hlfir::DeclareOp>();
    if (!decl) return call.emitError("ALLOCATE(..., SOURCE=...) of an allocatable without an hlfir.declare");
    mlir::OpBuilder b(call);
    mlir::Location const loc = call.getLoc();
    mlir::Value const heap =
        b.create<fir::AllocMemOp>(loc, heapTy.getEleTy(), (decl.getUniqName().getValue() + ".alloc").str(),
                                  mlir::ValueRange{}, mlir::ValueRange{});
    b.create<fir::StoreOp>(loc, b.create<fir::EmboxOp>(loc, boxTy, heap), boxRef);
    b.create<hlfir::AssignOp>(loc, source, boxRef, /*realloc=*/true);
    for (mlir::Value res : call.getResults())
      res.replaceAllUsesWith(b.create<mlir::arith::ConstantOp>(loc, b.getIntegerAttr(res.getType(), 0)));
    call.erase();
    return mlir::success();
  }

  static mlir::LogicalResult lower(fir::CallOp call) {
    mlir::Value const boxRef = peelConverts(call.getArgs()[0]);
    mlir::Value const source = peelConverts(call.getArgs()[1]);
    auto boxTy = mlir::dyn_cast<fir::BoxType>(fir::dyn_cast_ptrEleTy(boxRef.getType()));
    auto heapTy = boxTy ? mlir::dyn_cast<fir::HeapType>(boxTy.getEleTy()) : fir::HeapType{};
    if (!heapTy) {
      llvm::errs() << "hlfir-lower-allocate-source: ALLOCATE(..., SOURCE=...) of a polymorphic allocatable keeps its "
                      "runtime call\n";
      return mlir::success();
    }
    auto seqTy = mlir::dyn_cast<fir::SequenceType>(heapTy.getEleTy());
    if (!seqTy) return lowerScalar(call, boxRef, source, boxTy, heapTy);

    // The SetBounds calls on this descriptor before the AllocateSource, one per dim.
    unsigned const rank = seqTy.getDimension();
    llvm::SmallVector<fir::CallOp, 4> setBounds(rank);
    for (mlir::Operation* op = call->getPrevNode(); op; op = op->getPrevNode()) {
      auto sb = mlir::dyn_cast<fir::CallOp>(op);
      if (!sb || calleeName(sb) != "_FortranAAllocatableSetBounds" || peelConverts(sb.getArgs()[0]) != boxRef) continue;
      std::optional<int64_t> const dim = constInt(sb.getArgs()[1]);
      if (!dim || *dim < 0 || *dim >= rank || setBounds[*dim]) continue;
      setBounds[*dim] = sb;
    }
    mlir::OpBuilder b(call);
    mlir::Location const loc = call.getLoc();
    mlir::Type const idx = b.getIndexType();
    llvm::SmallVector<mlir::Value, 4> extents;
    for (unsigned dim = 0; dim < rank; ++dim) {
      fir::CallOp sb = setBounds[dim];  // NOLINT(misc-const-correctness): getArgs is non-const
      if (!sb) return call.emitError("ALLOCATE(..., SOURCE=...) without a SetBounds call for every dim");
      if (constInt(sb.getArgs()[2]) != 1)
        return call.emitError("ALLOCATE(..., SOURCE=...) with a lower bound other than 1 is not supported");
      mlir::Value const declared = declaredExtent(source, dim);
      extents.push_back(declared ? declared : b.create<fir::ConvertOp>(loc, idx, sb.getArgs()[3]));  // ub, lb == 1
    }
    llvm::SmallVector<mlir::Value, 4> dynamicExtents;
    for (unsigned dim = 0; dim < rank; ++dim)
      if (seqTy.getShape()[dim] == fir::SequenceType::getUnknownExtent()) dynamicExtents.push_back(extents[dim]);
    // ``<decl>.alloc``, as flang names a plain ALLOCATE's allocmem: the bridge finds a variable's ALLOCATEs by it.
    auto decl = boxRef.getDefiningOp<hlfir::DeclareOp>();
    if (!decl) return call.emitError("ALLOCATE(..., SOURCE=...) of an allocatable without an hlfir.declare");
    mlir::Value const heap = b.create<fir::AllocMemOp>(loc, seqTy, (decl.getUniqName().getValue() + ".alloc").str(),
                                                       mlir::ValueRange{}, dynamicExtents);
    mlir::Value const shape = b.create<fir::ShapeOp>(loc, extents);
    mlir::Value const box = b.create<fir::EmboxOp>(loc, boxTy, heap, shape);
    b.create<fir::StoreOp>(loc, box, boxRef);
    b.create<hlfir::AssignOp>(loc, source, boxRef, /*realloc=*/true);

    for (mlir::Value res : call.getResults())
      res.replaceAllUsesWith(b.create<mlir::arith::ConstantOp>(loc, b.getIntegerAttr(res.getType(), 0)));
    call.erase();
    for (fir::CallOp sb : setBounds) sb.erase();
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLowerAllocateSourcePass() { return std::make_unique<LowerAllocateSourcePass>(); }

}  // namespace hlfir_bridge
