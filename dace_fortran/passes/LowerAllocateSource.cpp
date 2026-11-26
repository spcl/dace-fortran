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
//     fir.allocmem with the SetBounds extents, fir.shape (every lower bound the literal 1) or
//     fir.shape_shift (the literal lower bound every ALLOCATE of an allocatable source gives it; QE h_psi's
//     iexx_istart reads its bound off the source's descriptor) +
//     fir.embox + fir.store into x's descriptor, then ``hlfir.assign y to x realloc``.  The
//     runtime's stat result becomes 0.
//
// Limits (left as runtime calls, never lowered silently wrong):
//     a scalar or polymorphic allocatable, a missing SetBounds for some dim, a lower bound that is no literal at
//     some ALLOCATE of the source. The bridge refuses such a call where a
//     translated procedure makes it; a module procedure the kernel never calls (ICON's CLASS(*) key-value store) builds
//     as before.
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

/// The lower bound of dim ``dim`` that every ALLOCATE of the allocatable ``lb`` is read off (``fir.box_dims`` of its
/// loaded descriptor) gives it, as a literal: 1 for a plain ``fir.shape``, the ``fir.shape_shift`` literal otherwise.
/// Null when ``lb`` is no such read, a site's bound is no literal, or two sites disagree: the bridge could not name
/// the copy's lower bound then.
std::optional<int64_t> allocatedLowerBound(mlir::Value lb) {
  if (std::optional<int64_t> const c = constInt(lb)) return c;
  auto dims = peelConverts(lb).getDefiningOp<fir::BoxDimsOp>();
  if (!dims || peelConverts(lb) != dims.getResult(0)) return std::nullopt;
  std::optional<int64_t> const dim = constInt(dims.getDim());
  auto load = dims.getVal().getDefiningOp<fir::LoadOp>();
  auto decl = load ? load.getMemref().getDefiningOp<hlfir::DeclareOp>() : hlfir::DeclareOp{};
  auto module = decl ? decl->getParentOfType<mlir::ModuleOp>() : mlir::ModuleOp{};
  if (!dim || !module) return std::nullopt;
  std::string const name = (decl.getUniqName().getValue() + ".alloc").str();
  std::optional<int64_t> found;
  bool agree = true;
  // Over the allocations of the source: each one's lower bound of ``dim``
  module.walk([&](fir::AllocMemOp alloc) {
    auto uniq = alloc.getUniqName();
    if (!uniq || *uniq != name) return;
    std::optional<int64_t> bound;
    // Over the descriptors built on the allocation: the shape they carry
    for (mlir::Operation* user : alloc->getUsers()) {
      auto embox = mlir::dyn_cast<fir::EmboxOp>(user);
      mlir::Operation* shape = embox && embox.getShape() ? embox.getShape().getDefiningOp() : nullptr;
      if (mlir::isa_and_nonnull<fir::ShapeOp>(shape)) bound = 1;
      if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape))
        if (2 * *dim < ss.getPairs().size()) bound = constInt(ss.getPairs()[2 * *dim]);
    }
    agree = agree && bound && (!found || *found == *bound);
    found = bound;
  });
  return agree ? found : std::nullopt;
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
    // Over the calls: each one this pass can model is rewritten, the rest stay for the bridge to refuse if reached
    for (fir::CallOp call : sources) (void)lower(call);
  }

  static mlir::LogicalResult lower(fir::CallOp call) {
    mlir::Value const boxRef = peelConverts(call.getArgs()[0]);
    mlir::Value const source = peelConverts(call.getArgs()[1]);
    auto boxTy = mlir::dyn_cast<fir::BoxType>(fir::dyn_cast_ptrEleTy(boxRef.getType()));
    auto heapTy = boxTy ? mlir::dyn_cast<fir::HeapType>(boxTy.getEleTy()) : fir::HeapType{};
    auto seqTy = heapTy ? mlir::dyn_cast<fir::SequenceType>(heapTy.getEleTy()) : fir::SequenceType{};
    if (!seqTy) return mlir::failure();  // a scalar or polymorphic allocatable
    // ``<decl>.alloc``, as flang names a plain ALLOCATE's allocmem: the bridge finds a variable's ALLOCATEs by it.
    auto decl = boxRef.getDefiningOp<hlfir::DeclareOp>();
    if (!decl) return mlir::failure();

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
    // Over the dims: every one needs its SetBounds and a lower bound the bridge can name, checked before any op is
    // created
    llvm::SmallVector<int64_t, 4> lowBounds;
    for (fir::CallOp sb : setBounds) {
      std::optional<int64_t> const low = sb ? allocatedLowerBound(sb.getArgs()[2]) : std::nullopt;
      if (!low) return mlir::failure();
      lowBounds.push_back(*low);
    }
    mlir::OpBuilder b(call);
    mlir::Location const loc = call.getLoc();
    mlir::Type const idx = b.getIndexType();
    llvm::SmallVector<mlir::Value, 4> extents, lowers;
    bool unitLowers = true;
    // Over the dims: the extent (the source's declared one, else ub - lb + 1) and the lower bound
    for (unsigned dim = 0; dim < rank; ++dim) {
      fir::CallOp sb = setBounds[dim];  // NOLINT(misc-const-correctness): getArgs is non-const
      bool const unit = lowBounds[dim] == 1;
      unitLowers = unitLowers && unit;
      mlir::Value lb;
      if (!unit) lb = b.create<mlir::arith::ConstantIndexOp>(loc, lowBounds[dim]);
      mlir::Value extent = declaredExtent(source, dim);
      if (!extent) {
        mlir::Value const ub = b.create<fir::ConvertOp>(loc, idx, sb.getArgs()[3]);
        extent = ub;
        if (!unit)
          extent = b.create<mlir::arith::AddIOp>(loc, b.create<mlir::arith::SubIOp>(loc, ub, lb),
                                                 b.create<mlir::arith::ConstantIndexOp>(loc, 1));
      }
      extents.push_back(extent);
      lowers.push_back(lb);
    }
    llvm::SmallVector<mlir::Value, 4> dynamicExtents;
    for (unsigned dim = 0; dim < rank; ++dim)
      if (seqTy.getShape()[dim] == fir::SequenceType::getUnknownExtent()) dynamicExtents.push_back(extents[dim]);
    mlir::Value const heap = b.create<fir::AllocMemOp>(loc, seqTy, (decl.getUniqName().getValue() + ".alloc").str(),
                                                       mlir::ValueRange{}, dynamicExtents);
    llvm::SmallVector<mlir::Value, 8> pairs;
    // Over the dims: (lower bound, extent) pairs of the shape_shift
    for (unsigned dim = 0; dim < rank && !unitLowers; ++dim)
      pairs.append(
          {lowers[dim] ? lowers[dim] : mlir::Value(b.create<mlir::arith::ConstantIndexOp>(loc, 1)), extents[dim]});
    mlir::Value shape;
    if (unitLowers)
      shape = b.create<fir::ShapeOp>(loc, extents);
    else
      shape = b.create<fir::ShapeShiftOp>(loc, fir::ShapeShiftType::get(b.getContext(), rank), pairs);
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
