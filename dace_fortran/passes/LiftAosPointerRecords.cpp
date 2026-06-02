// ============================================================================
// LiftAosPointerRecords.cpp  --  Detect AoS-of-pointer-records candidates;
// scaffolding for the upcoming lower-to-concat-array transform.
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
// bridge currently emits ``q[(_i-1), (_j-1)]`` against a scalar-shape
// descriptor, which Python's ``ast.parse`` later rejects.
//
// Strategy (planned)
// ------------------
// For each AoS-of-pointer-records ``q`` of static outer extent ``N``,
// per pointer-typed member ``m`` allocate a flat top-level transient
// ``q_<m>`` of shape ``(N, target_inner_shape...)``.  Then:
//
//   * Each rebind ``q(c)%m => target_c`` becomes a copy of
//     ``target_c`` into ``q_<m>(c, :, ...)`` (at the rebind site).
//   * Each access ``q(idx)%m(i, j, ...)`` is replaced with a direct
//     ``hlfir.designate`` over ``q_<m>(idx, i, j, ...)``.
//   * Before each function exit, copy ``q_<m>(c, :, ...)`` back to
//     each ``target_c`` so the originals see writes the body made
//     through the alias.
//
// This commit lands the matcher / collector only -- the bridge can
// already identify exactly which functions need the transform, and a
// downstream commit owns the IR materialisation (the triplet-
// designate + ``hlfir.assign`` for the copy-in / out + the access
// chain rewrite).  Splitting that way keeps each commit reviewable
// and lets QE parse + e2e correctness land in parallel.
//
// Scope
// -----
// Targeted at AoS-of-record-with-pointer-only-members where every
// rebind's outer index folds to a compile-time constant and every
// rebind target traces back to an ``hlfir.declare`` with a known
// static shape.  Mixed-member-kind records (some pointer, some flat
// / allocatable) deliberately fall through -- ``hlfir-flatten-
// structs`` already handles the flat-member parts.
//
// Pipeline placement: BEFORE ``hlfir-flatten-structs`` so flatten
// never sees the unsupported shape, and BEFORE ``hlfir-rewrite-
// pointer-assigns`` because the rebinds we will eliminate here are
// not top-level pointer rebinds the sibling pass would handle anyway.
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

/// One pointer-record member of an AoS candidate.  ``name`` is the
/// declared component name (``x`` / ``p`` in the Graupel example);
/// ``boxTy`` is the box-of-pointer type that wraps the target.
struct MemberSpec {
  std::string name;
  fir::BoxType boxTy;
};

/// A matched AoS-of-pointer-records candidate.
struct Candidate {
  hlfir::DeclareOp aosDecl;
  fir::SequenceType seqTy;
  fir::RecordType recordTy;
  int64_t N;
  llvm::SmallVector<MemberSpec, 4> members;
};

/// One rebind site ``q(outerIdx)%memberName => targetDecl``.
struct Rebind {
  int64_t outerIdx;  ///< 1-based, captured raw
  std::string memberName;
  hlfir::DeclareOp targetDecl;
  fir::StoreOp store;
};

static std::optional<int64_t> constInt(mlir::Value v) {
  if (!v) return std::nullopt;
  auto *def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return ia.getInt();
  return std::nullopt;
}

/// Match an ``hlfir.declare`` whose memref type is
/// ``fir.array<N x record<...>>`` where every record member is a
/// ``fir.box<fir.ptr<...>>`` / ``fir.box<fir.heap<...>>``.
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
  for (auto &member : recordTy.getTypeList()) {
    auto boxTy = mlir::dyn_cast<fir::BoxType>(member.second);
    if (!boxTy) return std::nullopt;
    auto inner = boxTy.getEleTy();
    if (!mlir::isa<fir::PointerType>(inner) &&
        !mlir::isa<fir::HeapType>(inner))
      return std::nullopt;
    c.members.push_back({member.first, boxTy});
  }
  if (c.members.empty()) return std::nullopt;
  return c;
}

/// Walk a rebind value back through ``rebox`` / ``embox`` / ``convert``
/// / ``designate`` to the originating declare; null when the chain
/// doesn't terminate at a declare.
static hlfir::DeclareOp traceTarget(mlir::Value v) {
  for (int hop = 0; hop < 128 && v; ++hop) {
    auto *def = v.getDefiningOp();
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

/// Match a rebind store ``fir.store %box to %slot_ref`` where ``slot_ref``
/// is ``hlfir.designate %elem{member} {pointer}`` and ``elem`` is
/// ``hlfir.designate %aosDecl (outer_const)``.
static std::optional<Rebind> matchRebindStore(fir::StoreOp store,
                                              const Candidate &cand) {
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

struct LiftAosPointerRecordsPass
    : public mlir::PassWrapper<LiftAosPointerRecordsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftAosPointerRecordsPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-lift-aos-pointer-records";
  }
  llvm::StringRef getDescription() const final {
    return "Detect AoS-of-records-with-pointer-only members (matcher / "
           "collector only -- materialisation lands in a follow-up "
           "commit).";
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
    for (auto &c : cands) {
      llvm::SmallVector<Rebind, 8> rebinds;
      func.walk([&](fir::StoreOp store) {
        if (auto r = matchRebindStore(store, c)) rebinds.push_back(*r);
      });
      // Matcher-only: record the discovery via a func-level attribute so
      // downstream passes / debug tooling can confirm the recognition
      // landed without the materialisation having to be on the critical
      // path yet.  Attribute name is one per candidate declare so multiple
      // AoS allocas in one function don't collide.
      if (!rebinds.empty()) {
        auto *ctx = func.getContext();
        std::string key =
            ("hlfir.aos_ptr_records." + c.aosDecl.getUniqName().str());
        llvm::SmallVector<mlir::Attribute, 8> entries;
        for (auto &r : rebinds) {
          llvm::SmallVector<mlir::NamedAttribute, 3> kv;
          kv.push_back({mlir::StringAttr::get(ctx, "outer"),
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get(ctx, 64), r.outerIdx)});
          kv.push_back({mlir::StringAttr::get(ctx, "member"),
                        mlir::StringAttr::get(ctx, r.memberName)});
          kv.push_back({mlir::StringAttr::get(ctx, "target"),
                        mlir::StringAttr::get(
                            ctx, r.targetDecl.getUniqName().str())});
          entries.push_back(mlir::DictionaryAttr::get(ctx, kv));
        }
        func->setAttr(key, mlir::ArrayAttr::get(ctx, entries));
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLiftAosPointerRecordsPass() {
  return std::make_unique<LiftAosPointerRecordsPass>();
}

}  // namespace hlfir_bridge
