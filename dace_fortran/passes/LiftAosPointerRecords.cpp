// ============================================================================
// LiftAosPointerRecords.cpp  --  Lower AoS-of-pointer-records to flat
// per-member concatenation arrays.
// ============================================================================
//
// Motivating pattern (Graupel ``q``):
//
//     TYPE t_qx_ptr
//       REAL(wp), POINTER :: p(:), x(:,:)
//     END TYPE
//     TYPE(t_qx_ptr) :: q(N)
//
//     q(c1)%x => qv      ! N rebinds at function entry, ``c`` constant
//     q(c2)%x => qc
//     ...
//
//     ... = q(idx)%x(i, j)        ! ``idx`` may be a runtime value
//     q(idx)%x(i, j) = ...
//
// ``hlfir-flatten-structs`` rejects this shape (its docstring lists
// ``AoS-with-pointer-members`` as out of scope) and ``hlfir-rewrite-
// pointer-assigns`` rebinds only top-level pointer declares.  The
// result is the bridge emits ``q[(_i-1), (_j-1)]`` against a scalar-
// shape descriptor.
//
// Transformation
// --------------
// For each AoS-of-pointer-records ``q`` of static outer extent ``N``,
// per pointer-typed member ``m`` allocate a flat top-level transient
// ``q_<m>`` of shape ``(N, target_inner_shape...)``:
//
//   * Each rebind ``q(c)%m => target_c`` becomes an element-wise copy
//     loop into ``q_<m>(c, ...)`` at the rebind site.
//   * Each access ``q(idx)%m(i, j, ...)`` is rewritten to a direct
//     ``hlfir.designate`` over ``q_<m>(idx, i, j, ...)``.
//   * Before each ``func.return`` the matching copy-out loop writes
//     ``q_<m>(c, ...)`` back to ``target_c`` so the original sees any
//     in-body updates the alias made.
//
// Pipeline placement: BEFORE ``hlfir-flatten-structs`` so flatten
// never sees the unsupported shape, and BEFORE ``hlfir-rewrite-
// pointer-assigns`` because the rebinds we eliminate here are not
// top-level pointer rebinds the sibling pass would handle anyway.
// ============================================================================

#include <optional>
#include <string>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct MemberSpec {
  std::string name;
  fir::BoxType boxTy;
};

/// Resolved shape of a rebind target's storage.  ``shape`` carries the
/// per-dim static extent (the literal int from a static-shape target's
/// sequence type or from a fir.shape with constant operands);
/// ``extentVals`` carries the per-dim SSA value (for dynamic shapes,
/// from the target declare's shape operand).  Exactly one of the two
/// is populated per dim -- a static dim leaves the corresponding
/// ``extentVals`` slot null; a dynamic dim leaves the corresponding
/// ``shape`` slot equal to ``fir::SequenceType::getUnknownExtent()``.
struct InnerShape {
  mlir::Type elemTy;
  llvm::SmallVector<int64_t, 4> shape;
  llvm::SmallVector<mlir::Value, 4> extentVals;
  /// Box value to query at use-site via ``fir.box_dims`` when no
  /// extent is statically known and no ``fir.shape`` op exposes the
  /// SSA extents.  Always null when one of the above already provides
  /// the per-dim extents.
  mlir::Value boxSource;

  /// Returns true when every dim has a static int extent (so the
  /// concat array can be allocated as a fully-typed static alloca).
  bool allStatic() const {
    for (auto d : shape)
      if (d == fir::SequenceType::getUnknownExtent()) return false;
    return !shape.empty();
  }
};

struct Candidate {
  hlfir::DeclareOp aosDecl;
  fir::SequenceType seqTy;
  fir::RecordType recordTy;
  int64_t N;
  llvm::SmallVector<MemberSpec, 4> members;
};

struct Rebind {
  int64_t outerIdx;  ///< 1-based, captured raw
  std::string memberName;
  hlfir::DeclareOp targetDecl;
  fir::StoreOp store;
};

static std::optional<int64_t> constInt(mlir::Value v) {
  if (!v) return std::nullopt;
  auto* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// Element type stripped of the box-of-pointer wrappers (the pointer's
/// shape itself is assumed-shape ``?x?...`` for a generic
/// ``REAL, POINTER :: x(:,:)`` declaration, so the EXTENTS come from a
/// rebind target instead -- see ``innerShapeFromTargetDecl``).
static mlir::Type elemTypeOfPtrMember(fir::BoxType boxTy) {
  auto inner = boxTy.getEleTy();
  if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) inner = p.getEleTy();
  if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) inner = h.getEleTy();
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner))
    return seq.getEleTy();
  return inner;
}

/// Resolve a rebind target's static inner shape.  The target is the
/// declare that owns the original storage (e.g. ``qa(n, k)``); its
/// memref type is a ``ref<array<? x ? x f64>>`` (assumed-shape from
/// the dummy-arg ABI).  When the target is a dummy, walk to its
/// ``hlfir.declare`` ``shape`` operand and extract per-dim extents from
/// the produced ``fir.shape`` op.  When the target is local with a
/// static-shape declaration the sequence type itself carries the
/// extents.
/// Look at ``targetDecl``'s memref / result types and extract the
/// element type + per-dim static extents.  ``shape`` slots with
/// ``getUnknownExtent`` are dynamic; ``extentVals`` slots with a
/// non-null Value carry an SSA extent that already dominates the
/// declare.  Slots where the dim is dynamic AND no SSA extent is
/// available leave ``extentVals[d]`` null -- the caller emits a
/// ``fir.box_dims`` at the insertion point to recover it.
static std::optional<InnerShape> innerShapeFromTargetDecl(
    hlfir::DeclareOp targetDecl) {
  mlir::Type eleTy;
  unsigned rank = 0;
  // Find the element type + rank from whichever typed view of the
  // declare carries them (the memref or the box-typed result).
  auto memrefTy = targetDecl.getMemref().getType();
  if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(memrefTy)) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy())) {
      eleTy = seq.getEleTy();
      rank = seq.getShape().size();
    }
  } else if (auto boxTy = mlir::dyn_cast<fir::BoxType>(memrefTy)) {
    auto inner = boxTy.getEleTy();
    if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) inner = p.getEleTy();
    if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) inner = h.getEleTy();
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner)) {
      eleTy = seq.getEleTy();
      rank = seq.getShape().size();
    }
  }
  if (!eleTy || rank == 0) return std::nullopt;

  InnerShape s;
  s.elemTy = eleTy;

  // First try the declare's ``shape`` operand: when present it gives
  // ready-to-use SSA per-dim extents.
  if (auto shapeOper = targetDecl.getShape()) {
    if (auto shapeOp =
            mlir::dyn_cast_or_null<fir::ShapeOp>(shapeOper.getDefiningOp())) {
      for (auto ext : shapeOp.getExtents()) {
        if (auto c = constInt(ext)) {
          s.shape.push_back(*c);
          s.extentVals.push_back(mlir::Value{});
        } else {
          s.shape.push_back(fir::SequenceType::getUnknownExtent());
          s.extentVals.push_back(ext);
        }
      }
      return s;
    }
  }

  // Fallback: pure-static seq type, no shape operand needed.
  if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(memrefTy)) {
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy())) {
      bool allStatic = true;
      for (auto d : seq.getShape()) {
        if (d == fir::SequenceType::getUnknownExtent()) {
          allStatic = false;
          break;
        }
      }
      if (allStatic) {
        for (auto d : seq.getShape()) {
          s.shape.push_back(d);
          s.extentVals.push_back(mlir::Value{});
        }
        return s;
      }
    }
  }

  // Last resort: assumed-shape box-of-array dummy.  Record the box
  // value as the "source" so ``emitBoxDims`` can extract extents at
  // the insertion point.
  s.shape.assign(rank, fir::SequenceType::getUnknownExtent());
  s.extentVals.assign(rank, mlir::Value{});
  s.boxSource = targetDecl.getResult(0);
  return s;
}

static std::optional<Candidate> matchCandidate(hlfir::DeclareOp d) {
  auto refTy = mlir::dyn_cast<fir::ReferenceType>(d.getMemref().getType());
  if (!refTy) return std::nullopt;
  auto seqTy = mlir::dyn_cast<fir::SequenceType>(refTy.getEleTy());
  if (!seqTy) return std::nullopt;
  if (seqTy.getShape().size() != 1) return std::nullopt;
  auto N = seqTy.getShape()[0];
  if (N == fir::SequenceType::getUnknownExtent() || N <= 0) return std::nullopt;
  auto recordTy = mlir::dyn_cast<fir::RecordType>(seqTy.getEleTy());
  if (!recordTy) return std::nullopt;
  Candidate c;
  c.aosDecl = d;
  c.seqTy = seqTy;
  c.recordTy = recordTy;
  c.N = N;
  for (auto& member : recordTy.getTypeList()) {
    auto boxTy = mlir::dyn_cast<fir::BoxType>(member.second);
    if (!boxTy) return std::nullopt;
    auto inner = boxTy.getEleTy();
    if (!mlir::isa<fir::PointerType>(inner) && !mlir::isa<fir::HeapType>(inner))
      return std::nullopt;
    c.members.push_back({member.first, boxTy});
  }
  if (c.members.empty()) return std::nullopt;
  return c;
}

static hlfir::DeclareOp traceTarget(mlir::Value v) {
  for (int hop = 0; hop < 128 && v; ++hop) {
    auto* def = v.getDefiningOp();
    if (!def) return {};
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
      v = rb.getBox();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
      v = eb.getMemref();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      v = dg.getMemref();
      continue;
    }
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      return decl;
    }
    return {};
  }
  return {};
}

static std::optional<Rebind> matchRebindStore(fir::StoreOp store,
                                              const Candidate& cand) {
  auto memberSlot = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
      store.getMemref().getDefiningOp());
  if (!memberSlot) return std::nullopt;
  auto memberOpt = memberSlot.getComponent();
  if (!memberOpt.has_value()) return std::nullopt;
  auto memberName = memberOpt->getValue();
  if (memberName.empty()) return std::nullopt;
  auto elemRef = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
      memberSlot.getMemref().getDefiningOp());
  if (!elemRef) return std::nullopt;
  auto srcDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
      elemRef.getMemref().getDefiningOp());
  if (!srcDecl || srcDecl != cand.aosDecl) return std::nullopt;
  auto idxs = elemRef.getIndices();
  if (idxs.size() != 1) return std::nullopt;
  auto outerC = constInt(idxs[0]);
  if (!outerC) return std::nullopt;
  Rebind r;
  r.outerIdx = *outerC;
  r.memberName = memberName.str();
  r.store = store;
  r.targetDecl = traceTarget(store.getValue());
  if (!r.targetDecl) return std::nullopt;
  return r;
}

// --------------------------------------------------------------------------
// Materialisation helpers.
// --------------------------------------------------------------------------

/// Create a top-level static-shape concat transient ``q_<member>`` at the
/// start of ``func``.  Returns the resulting ``hlfir.declare`` (whose
/// ``result(0)`` is the storage view used by all designates).
static hlfir::DeclareOp createConcatStorage(
    mlir::OpBuilder& b, mlir::Location loc, mlir::func::FuncOp func,
    Candidate& cand, const MemberSpec& member, const InnerShape& innerShape,
    mlir::Operation* insertAfter, int64_t allocId) {
  mlir::OpBuilder::InsertionGuard g(b);
  if (insertAfter)
    b.setInsertionPointAfter(insertAfter);
  else
    b.setInsertionPointToStart(&func.getBody().front());

  llvm::SmallVector<int64_t, 4> typeDims;
  typeDims.push_back(cand.N);
  for (auto d : innerShape.shape) typeDims.push_back(d);
  auto seqTy = fir::SequenceType::get(typeDims, innerShape.elemTy);

  // Per-dim extents as index Values.  Static dims synth an index
  // constant; dynamic dims re-use the captured SSA extent, or fall
  // back to ``fir.box_dims`` on the captured ``boxSource``.
  llvm::SmallVector<mlir::Value, 4> extents;
  auto outerC = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(cand.N))
                    .getResult();
  extents.push_back(outerC);
  auto idxTy = b.getIndexType();
  for (size_t d = 0; d < innerShape.shape.size(); ++d) {
    if (innerShape.shape[d] != fir::SequenceType::getUnknownExtent()) {
      auto cst = b.create<mlir::arith::ConstantOp>(
          loc, b.getIndexAttr(innerShape.shape[d]));
      extents.push_back(cst.getResult());
    } else if (innerShape.extentVals[d]) {
      mlir::Value ext = innerShape.extentVals[d];
      if (ext.getType() != idxTy)
        ext = b.create<fir::ConvertOp>(loc, idxTy, ext).getResult();
      extents.push_back(ext);
    } else if (innerShape.boxSource) {
      auto dimC =
          b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(d)).getResult();
      auto bd = b.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                         innerShape.boxSource, dimC);
      // bd.getResult(1) is the extent of dim ``d``.
      extents.push_back(bd.getResult(1));
    } else {
      return {};
    }
  }
  // Alloca shape: static when fully-static, dynamic-shape signature
  // otherwise.
  fir::AllocaOp alloca;
  if (innerShape.allStatic()) {
    alloca = b.create<fir::AllocaOp>(loc, seqTy);
  } else {
    alloca = b.create<fir::AllocaOp>(loc, seqTy, /*uniqName=*/llvm::StringRef{},
                                     /*bindcName=*/llvm::StringRef{},
                                     /*typeparams=*/mlir::ValueRange{},
                                     /*shape=*/mlir::ValueRange{extents});
  }
  auto shapeOper = b.create<fir::ShapeOp>(loc, extents).getResult();
  auto uniqName = "_QXaos_lift_" + cand.aosDecl.getUniqName().str() + "_" +
                  member.name + "_" + std::to_string(allocId);
  auto decl =
      b.create<hlfir::DeclareOp>(loc, alloca.getResult(), uniqName, shapeOper);
  return decl;
}

/// Emit a nested ``fir.do_loop`` tree that copies between two arrays of
/// shape ``innerShape``, with an additional fixed outer index for one
/// side.
///
/// ``directionCopyIn = true``  copies ``target(i...)`` into ``concat(c, i...)``
/// ``directionCopyIn = false`` copies ``concat(c, i...)`` into ``target(i...)``
static void emitCopyLoop(mlir::OpBuilder& b, mlir::Location loc,
                         mlir::Operation* insertBefore,
                         hlfir::DeclareOp concatDecl, int64_t outerC,
                         hlfir::DeclareOp targetDecl,
                         const InnerShape& innerShape, bool directionCopyIn) {
  mlir::OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(insertBefore);
  auto idxTy = b.getIndexType();
  auto c1 =
      b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(1)).getResult();
  // Outer index as a 1-based literal (hlfir.designate uses Fortran 1-based
  // subscripts).
  auto outerVal = b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(outerC))
                      .getResult();

  llvm::SmallVector<mlir::Value, 4> ivs;
  for (size_t d = 0; d < innerShape.shape.size(); ++d) {
    mlir::Value hi;
    if (innerShape.shape[d] != fir::SequenceType::getUnknownExtent()) {
      hi = b.create<mlir::arith::ConstantOp>(
                loc, b.getIndexAttr(innerShape.shape[d]))
               .getResult();
    } else if (innerShape.extentVals[d]) {
      hi = innerShape.extentVals[d];
      if (hi.getType() != idxTy)
        hi = b.create<fir::ConvertOp>(loc, idxTy, hi).getResult();
    } else if (innerShape.boxSource) {
      auto dimC =
          b.create<mlir::arith::ConstantOp>(loc, b.getIndexAttr(d)).getResult();
      auto bd = b.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                         innerShape.boxSource, dimC);
      hi = bd.getResult(1);
    } else {
      return;
    }
    auto loop = b.create<fir::DoLoopOp>(loc, c1, hi, c1,
                                        /*unordered=*/false,
                                        /*finalCountValue=*/false);
    ivs.push_back(loop.getInductionVar());
    b.setInsertionPointToStart(loop.getBody());
  }

  // Designates: full-rank element accesses.  target uses ``ivs`` only;
  // concat prepends the outer literal.
  llvm::SmallVector<mlir::Value, 5> concatIdxs;
  concatIdxs.push_back(outerVal);
  for (auto v : ivs) concatIdxs.push_back(v);

  auto elemRefTy = fir::ReferenceType::get(innerShape.elemTy);
  mlir::Value srcRef =
      directionCopyIn ? targetDecl.getResult(0) : concatDecl.getResult(0);
  mlir::Value dstRef =
      directionCopyIn ? concatDecl.getResult(0) : targetDecl.getResult(0);
  llvm::SmallVector<mlir::Value, 5> srcIdxs;
  llvm::SmallVector<mlir::Value, 5> dstIdxs;
  if (directionCopyIn) {
    for (auto v : ivs) srcIdxs.push_back(v);
    for (auto v : concatIdxs) dstIdxs.push_back(v);
  } else {
    for (auto v : concatIdxs) srcIdxs.push_back(v);
    for (auto v : ivs) dstIdxs.push_back(v);
  }

  auto srcDg = b.create<hlfir::DesignateOp>(loc, elemRefTy, srcRef,
                                            mlir::ValueRange{srcIdxs});
  auto loaded = b.create<fir::LoadOp>(loc, srcDg.getResult());
  auto dstDg = b.create<hlfir::DesignateOp>(loc, elemRefTy, dstRef,
                                            mlir::ValueRange{dstIdxs});
  // ``hlfir.assign %loaded_scalar to %dst_element_ref`` -- a scalar
  // store into one element.  The bridge's ``buildAssignNode`` routes
  // this through the normal SDFG-emit path; a raw ``fir.store``
  // surfaces an access node whose data field resolves to the loop
  // iterator symbol instead of the destination array (a pre-existing
  // bridge gap), so the assign form is the right shape.
  b.create<hlfir::AssignOp>(loc, loaded.getResult(), dstDg.getResult(),
                            /*realloc=*/false,
                            /*keep_lhs_length_if_realloc=*/false,
                            /*temporary_lhs=*/false);
  (void)idxTy;
}

/// Rewrite every access through ``q(idx)%member`` to designate over
/// ``q_<member>(idx, ...)``.  The shape we recognise:
///
///   %elem = hlfir.designate %aosDecl (idx)
///   %slot = hlfir.designate %elem{member} {pointer}
///   %box  = fir.load %slot          -- target box
///   %addr = fir.box_addr %box       -- raw data pointer (sometimes)
///   %dg   = hlfir.designate %box (i, j, ...)  -- element ref
///                          OR
///           hlfir.designate %addr (i, j, ...)
///
/// We replace each ``%dg`` with a fresh designate over the matching
/// concat declare and let later canonicalisation prune the now-dead
/// load / box_addr / slot chain.
static void rewriteAccess(mlir::OpBuilder& b, hlfir::DesignateOp innerDg,
                          mlir::Value outerIdxVal,
                          hlfir::DeclareOp concatDecl) {
  mlir::OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(innerDg);
  auto idxTy = b.getIndexType();
  mlir::Value outerCast = outerIdxVal;
  if (outerCast.getType() != idxTy)
    outerCast = b.create<fir::ConvertOp>(innerDg.getLoc(), idxTy, outerCast)
                    .getResult();
  llvm::SmallVector<mlir::Value, 5> newIdxs;
  newIdxs.push_back(outerCast);
  for (auto idx : innerDg.getIndices()) newIdxs.push_back(idx);
  auto newDg = b.create<hlfir::DesignateOp>(
      innerDg.getLoc(), innerDg.getResult().getType(), concatDecl.getResult(0),
      mlir::ValueRange{newIdxs});
  innerDg.getResult().replaceAllUsesWith(newDg.getResult());
}

/// Walk all uses of ``q``'s declare and rewrite every access chain
/// through ``q(idx)%member`` to a direct designate over the matching
/// concat declare.  Returns the count of rewritten access chains.
static unsigned rewriteAllAccesses(
    mlir::OpBuilder& b, Candidate& cand,
    const llvm::DenseMap<llvm::StringRef, hlfir::DeclareOp>& concatByMember) {
  unsigned rewritten = 0;
  // Snapshot users -- replaceAllUsesWith below would invalidate a live
  // user iterator.
  llvm::SmallVector<hlfir::DesignateOp, 8> elemDesignates;
  for (auto* u : cand.aosDecl.getResult(0).getUsers())
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u))
      if (dg.getIndices().size() == 1) elemDesignates.push_back(dg);
  for (auto elemDg : elemDesignates) {
    auto outerIdx = elemDg.getIndices()[0];
    llvm::SmallVector<hlfir::DesignateOp, 4> memberDgs;
    for (auto* u : elemDg.getResult().getUsers())
      if (auto md = mlir::dyn_cast<hlfir::DesignateOp>(u))
        memberDgs.push_back(md);
    for (auto memberDg : memberDgs) {
      auto memberOpt = memberDg.getComponent();
      if (!memberOpt.has_value()) continue;
      auto memberName = memberOpt->getValue();
      auto it = concatByMember.find(memberName);
      if (it == concatByMember.end()) continue;
      auto concatDecl = it->second;
      // Each box load gives a box value; element designates on that
      // box are the access leaves we rewrite.
      llvm::SmallVector<fir::LoadOp, 4> loads;
      for (auto* u : memberDg.getResult().getUsers())
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) loads.push_back(ld);
      for (auto load : loads) {
        llvm::SmallVector<hlfir::DesignateOp, 4> innerDgs;
        for (auto* u : load.getResult().getUsers())
          if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u))
            innerDgs.push_back(dg);
        // box_addr -> element designate chain (some flang paths).
        for (auto* u : load.getResult().getUsers())
          if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(u))
            for (auto* uu : ba.getResult().getUsers())
              if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(uu))
                innerDgs.push_back(dg);
        for (auto innerDg : innerDgs) {
          rewriteAccess(b, innerDg, outerIdx, concatDecl);
          rewritten++;
        }
      }
    }
  }
  return rewritten;
}

// --------------------------------------------------------------------------
// Pass.
// --------------------------------------------------------------------------

struct LiftAosPointerRecordsPass
    : public mlir::PassWrapper<LiftAosPointerRecordsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftAosPointerRecordsPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-lift-aos-pointer-records";
  }
  llvm::StringRef getDescription() const final {
    return "Lift AoS-of-records-with-pointer-only members to flat per-"
           "member concatenation transients with copy-in / copy-out.";
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto func :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      processFunction(func);
    }
  }

  void processFunction(mlir::func::FuncOp func) {
    llvm::SmallVector<Candidate, 2> cands;
    func.walk([&](hlfir::DeclareOp d) {
      if (auto c = matchCandidate(d)) cands.push_back(*c);
    });
    int allocId = 0;
    for (auto& c : cands) processCandidate(func, c, allocId);
  }

  void processCandidate(mlir::func::FuncOp func, Candidate& cand,
                        int& allocId) {
    llvm::SmallVector<Rebind, 8> rebinds;
    func.walk([&](fir::StoreOp store) {
      if (auto r = matchRebindStore(store, cand)) rebinds.push_back(*r);
    });
    if (rebinds.empty()) return;

    // Stamp the recognition attribute on the function unconditionally
    // -- this surfaces the matched candidates to debug tools and the
    // test suite regardless of whether materialisation runs (the
    // materialiser bails when shapes can't fold to static integers).
    {
      auto* ctx = func.getContext();
      std::string key =
          ("hlfir.aos_ptr_records." + cand.aosDecl.getUniqName().str());
      llvm::SmallVector<mlir::Attribute, 8> entries;
      for (auto& r : rebinds) {
        llvm::SmallVector<mlir::NamedAttribute, 3> kv;
        kv.push_back({mlir::StringAttr::get(ctx, "outer"),
                      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                             r.outerIdx)});
        kv.push_back({mlir::StringAttr::get(ctx, "member"),
                      mlir::StringAttr::get(ctx, r.memberName)});
        kv.push_back(
            {mlir::StringAttr::get(ctx, "target"),
             mlir::StringAttr::get(ctx, r.targetDecl.getUniqName().str())});
        entries.push_back(mlir::DictionaryAttr::get(ctx, kv));
      }
      func->setAttr(key, mlir::ArrayAttr::get(ctx, entries));
    }

    // Per-member inner shape -- resolve from the FIRST rebind's
    // target.  Subsequent rebinds for the same member are required to
    // share the same shape (so the concat array's inner dims are
    // unambiguous); we cross-check by comparing the resolved shapes
    // before deciding the materialisation is safe.
    llvm::DenseMap<llvm::StringRef, InnerShape> shapeByMember;
    for (auto& r : rebinds) {
      auto memberKey = llvm::StringRef(r.memberName);
      if (shapeByMember.count(memberKey)) continue;
      auto s = innerShapeFromTargetDecl(r.targetDecl);
      if (!s) return;
      shapeByMember.try_emplace(memberKey, *s);
    }
    // Validate cross-rebind shape consistency.
    for (auto& r : rebinds) {
      auto memberKey = llvm::StringRef(r.memberName);
      auto resolved = innerShapeFromTargetDecl(r.targetDecl);
      if (!resolved) return;
      auto& recorded = shapeByMember.find(memberKey)->second;
      if (resolved->shape != recorded.shape) return;
    }

    mlir::OpBuilder b(func.getContext());
    // Find the latest declare that contributes a shape extent we depend
    // on, so the concat declare is inserted in a position that sees
    // every required SSA value.  At minimum the AoS declare itself
    // must be visible (so its uniq_name is in scope for our derived
    // name).  Targets dominated by their hlfir.declare ops feed the
    // concat's per-dim extents.
    mlir::Operation* insertAfter = cand.aosDecl.getOperation();
    auto pushDep = [&](mlir::Value v) {
      if (!v) return;
      auto* def = v.getDefiningOp();
      if (!def) return;
      if (insertAfter->isBeforeInBlock(def)) insertAfter = def;
    };
    for (auto& kv : shapeByMember) {
      for (auto v : kv.second.extentVals) pushDep(v);
      pushDep(kv.second.boxSource);
    }
    // Per-rebind: also account for the target declare itself, which is
    // the SSA producer of any box-source the copy loops will read.
    for (auto& r : rebinds) {
      pushDep(r.targetDecl.getResult(0));
      pushDep(r.targetDecl.getResult(1));
    }
    llvm::DenseMap<llvm::StringRef, hlfir::DeclareOp> concatByMember;
    for (auto& spec : cand.members) {
      auto memberKey = llvm::StringRef(spec.name);
      auto sIt = shapeByMember.find(memberKey);
      if (sIt == shapeByMember.end()) continue;  // member never rebound
      auto declare = createConcatStorage(b, func.getLoc(), func, cand, spec,
                                         sIt->second, insertAfter, allocId++);
      if (!declare) return;
      concatByMember[memberKey] = declare;
      insertAfter = declare.getOperation();
    }

    // Copy-in: at each rebind store's location, emit the loop nest;
    // erase the store so the pointer slot is no longer written.
    for (auto& r : rebinds) {
      auto it = concatByMember.find(r.memberName);
      if (it == concatByMember.end()) continue;
      auto& shape = shapeByMember.find(r.memberName)->second;
      emitCopyLoop(b, r.store.getLoc(), r.store, it->second, r.outerIdx,
                   r.targetDecl, shape, /*directionCopyIn=*/true);
      r.store.erase();
    }

    // Rewrite access chains.
    rewriteAllAccesses(b, cand, concatByMember);

    // Copy-out: before each func.return, mirror copy-in for each rebind.
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    func.walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
    for (auto ret : returns) {
      for (auto& r : rebinds) {
        auto it = concatByMember.find(r.memberName);
        if (it == concatByMember.end()) continue;
        auto& shape = shapeByMember.find(r.memberName)->second;
        emitCopyLoop(b, ret.getLoc(), ret, it->second, r.outerIdx, r.targetDecl,
                     shape, /*directionCopyIn=*/false);
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftAosPointerRecordsPass() {
  return std::make_unique<LiftAosPointerRecordsPass>();
}

}  // namespace hlfir_bridge
