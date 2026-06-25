// ============================================================================
// VersionShapeScalars.cpp  --  SSA-version a straight-line reassigned scalar
// that is used as an array shape.
// ============================================================================
// Problem:
//     A local integer scalar used as an array extent (``ALLOCATE(x(m))``) may
//     be REASSIGNED before another array is sized from it:
//
//         m = base * 2
//         ALLOCATE(x(m))        ! x's extent is m's FIRST value
//         m = m + 3
//         ALLOCATE(y(m))        ! y's extent is m's SECOND value
//         x = x * 2.0           ! whole-array op over x's shape
//
//     The bridge resolves BOTH ``x`` and ``y`` extents to the bare name ``m``
//     (``traceExtentExpr`` -> ``traceToDecl`` drops the version), so both get
//     ``shape = (m,)``.  ``m`` is then a MUTABLE SDFG symbol: after ``m = m+3``
//     a whole-array op over ``x``'s shape maps ``m = base*2+3`` iterations over
//     an array allocated to ``base*2`` -> heap corruption / OOB.
//
// The precise hazard:
//     The corruption needs a reassignment ORDERED AFTER an array was allocated
//     from the scalar (so the array's mutable extent symbol diverges from its
//     real extent while the array is live).  Two cases are therefore NOT
//     hazards and are left untouched:
//       * accumulate-then-allocate-once -- ``nij = 0; do .. nij = nij + ..;
//         ALLOCATE(qgm(.., nij))`` -- every reassignment precedes the single
//         allocation, so the extent is frozen for the array's lifetime; and
//       * data-access scalars -- a loop bound (``do jb = .., i_endblk``) or a
//         subscript (``z(1:jb)``) mints a trip-count / section ``fir.shape``
//         but never an ``fir.allocmem`` extent, so it is not a shape at all.
//
// What this pass does (straight line only):
//     For a scalar whose value feeds an ``fir.allocmem`` (ALLOCATE) extent and
//     that is reassigned AFTER that allocation in a single straight-line block,
//     it splits the scalar into one immutable version per store (``m``,
//     ``m_2``, ...).  Each store writes its own version; each load binds to the
//     version live at that point.  Downstream, ``x`` sizes from ``m`` and ``y``
//     from ``m_2`` -- both immutable -- and the hazard is gone.
//
// What it REFUSES (with a clear message):
//     If the post-allocation reassignment is NOT in a straight line -- a store
//     inside a loop or a conditional branch -- the value live at a downstream
//     array is ambiguous (which iteration / which branch?), and SSA-versioning
//     cannot statically name it.  Rather than silently emit a mutable shape
//     symbol (the corruption above), the pass emits an error and fails: the
//     user must hoist the size to a single assignment before the allocation or
//     pass an explicit dimension.
// ============================================================================

#include <algorithm>
#include <limits>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// True iff ``v`` (a scalar load result) feeds the EXTENT operand of a
/// ``fir.allocmem`` -- i.e. the scalar is the size in an ``ALLOCATE``
/// statement, threaded through kind conversions, extent arithmetic
/// (``m * 2``), and Flang's ``max(ext, 0)`` clamp (``arith.select``).
///
/// We key on ``fir.allocmem`` ALONE -- the size of an *extent/size* use -- and
/// deliberately NOT on ``fir.shape`` / ``fir.shape_shift``.  Those latter ops
/// are also minted for *data-access* uses that have nothing to do with a
/// declared array's extent: a loop bound (``do jb = i_startblk, i_endblk``)
/// produces a trip-count ``fir.shape`` for an internal temporary, and a
/// section subscript (``z(1:jb)``) produces a length ``fir.shape``.  Keying on
/// those falsely flags loop iterators and loop bounds as shape scalars (and
/// then refuses ICON kernels).  ``fir.allocmem`` is the only path by which a
/// *reassigned* scalar becomes a *mutable* array extent (an automatic /
/// explicit-shape array freezes its bound at entry, so it is never a
/// reassignment hazard), so it is both the precise and the sufficient anchor.
/// Bounded forward walk -- an extent expression is shallow.
static bool feedsAllocateExtent(mlir::Value v) {
  llvm::SmallVector<mlir::Value, 8> work{v};
  llvm::SmallPtrSet<mlir::Operation*, 16> seen;
  int budget = 256;
  while (!work.empty() && budget-- > 0) {
    mlir::Value cur = work.pop_back_val();
    for (auto* u : cur.getUsers()) {
      if (mlir::isa<fir::AllocMemOp>(u)) return true;
      // Arithmetic / convert / clamp that an extent expression threads through
      // (``m * 2``, kind conversions, Flang's ``max(ext, 0)`` =
      // ``arith.select(cmpi, ext, 0)`` guard).  Follow the result.
      if (mlir::isa<fir::ConvertOp, mlir::arith::MulIOp, mlir::arith::AddIOp,
                    mlir::arith::SubIOp, mlir::arith::DivSIOp,
                    mlir::arith::SelectOp>(u)) {
        if (seen.insert(u).second)
          for (auto r : u->getResults()) work.push_back(r);
      }
    }
  }
  return false;
}

/// The memref/target value a write op stores to: ``fir.store``'s memref or a
/// scalar ``hlfir.assign``'s LHS (Fortran ``m = ...`` lowers to assign, not
/// store).  Null if ``op`` is not a write.
static mlir::Value writeTargetOf(mlir::Operation* op) {
  if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) return st.getMemref();
  if (auto as = mlir::dyn_cast<hlfir::AssignOp>(op)) return as.getLhs();
  return {};
}

/// The mutable target operand of a write op (for redirection).
static mlir::OpOperand* writeTargetOperand(mlir::Operation* op) {
  if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) return &st.getMemrefMutable();
  if (auto as = mlir::dyn_cast<hlfir::AssignOp>(op)) return &as.getLhsMutable();
  return nullptr;
}

/// The innermost loop op (``fir.do_loop`` / ``scf.for`` / ``scf.while``)
/// enclosing ``op``, or null if ``op`` is not inside a loop.
static mlir::Operation* enclosingLoop(mlir::Operation* op) {
  for (mlir::Operation* p = op->getParentOp(); p; p = p->getParentOp())
    if (mlir::isa<fir::DoLoopOp, mlir::scf::ForOp, mlir::scf::WhileOp>(p))
      return p;
  return nullptr;
}

struct VersionShapeScalarsPass
    : public mlir::PassWrapper<VersionShapeScalarsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VersionShapeScalarsPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-version-shape-scalars";
  }
  llvm::StringRef getDescription() const final {
    return "SSA-version a scalar reassigned (in a straight line) AFTER an "
           "array "
           "was allocated from it, so each array binds the extent live at its "
           "own allocation; refuse (clear error) when that post-allocation "
           "reassignment is in a loop / branch.  Accumulate-then-allocate-once "
           "scalars and data-access (loop bound / subscript) scalars are left "
           "untouched.";
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::func::FuncOp func) {
      if (func.isExternal()) return;
      // Pre-order index = program order, used to ask "is this store ordered
      // after that allocation?".  Built once per function.
      llvm::DenseMap<mlir::Operation*, unsigned> order;
      unsigned idx = 0;
      func.walk<mlir::WalkOrder::PreOrder>(
          [&](mlir::Operation* op) { order[op] = idx++; });
      // Snapshot declares first: we mutate the IR (insert new declares) while
      // iterating, so collect candidates before rewriting.
      llvm::SmallVector<hlfir::DeclareOp, 16> declares;
      func.walk([&](hlfir::DeclareOp d) { declares.push_back(d); });
      for (auto decl : declares)
        if (failed(versionOne(decl, order))) {
          signalPassFailure();
          return;
        }
    });
  }

  /// Version ``decl`` if it is a shape scalar reassigned AFTER an array was
  /// allocated from it in a straight line; refuse (``failure``) if that
  /// post-allocation reassignment is in a loop / branch; otherwise leave it
  /// untouched (``success``).
  mlir::LogicalResult versionOne(
      hlfir::DeclareOp decl,
      const llvm::DenseMap<mlir::Operation*, unsigned>& order) {
    // Scalar integer local only.  ``hlfir.declare`` result #0 is the entity;
    // a scalar integer is ``!fir.ref<iN>`` (no box / sequence).
    auto refTy =
        mlir::dyn_cast<fir::ReferenceType>(decl.getResult(0).getType());
    if (!refTy || !mlir::isa<mlir::IntegerType>(refTy.getEleTy()))
      return mlir::success();

    // Collect writes / loads against EITHER declare result (#0 entity, #1 raw
    // memref -- a scalar uses them interchangeably).  A Fortran scalar
    // assignment ``m = ...`` lowers to ``hlfir.assign %v to %m`` (NOT
    // ``fir.store``), so both forms count as writes.
    llvm::SmallVector<mlir::Operation*, 4> stores;
    llvm::SmallVector<fir::LoadOp, 8> loads;
    for (auto res : {decl.getResult(0), decl.getResult(1)})
      for (auto* u : res.getUsers()) {
        if (writeTargetOf(u) == res)
          stores.push_back(u);
        else if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) {
          if (ld.getMemref() == res) loads.push_back(ld);
        }
      }
    if (stores.size() < 2) return mlir::success();  // single version, fine

    // Shape scalar?  Only act when a load actually feeds an ALLOCATE extent (an
    // *extent/size* use).  A reassigned loop counter / accumulator / loop bound
    // / subscript (a *data-access* use) does not feed an ``fir.allocmem`` and
    // is irrelevant -- this keeps loop iterators and loop bounds (which mint
    // trip-count / section ``fir.shape`` ops) from being misread as shapes.
    // ``earliestAlloc`` = program position of the first array allocated from
    // this scalar.
    unsigned earliestAlloc = std::numeric_limits<unsigned>::max();
    for (auto ld : loads)
      if (feedsAllocateExtent(ld.getResult()))
        earliestAlloc =
            std::min(earliestAlloc, order.lookup(ld.getOperation()));
    if (earliestAlloc == std::numeric_limits<unsigned>::max())
      return mlir::success();  // not used as an array extent

    // Hazard = a reassignment ordered AFTER an array was allocated from the
    // scalar.  Only then does the array's (mutable) extent symbol diverge from
    // its real extent.  The dominant idiom -- accumulate a size in a loop, then
    // ALLOCATE once with the final value (``nij = 0; do .. nij = nij + ..;
    // ALLOCATE(qgm(.., nij))``) -- reassigns ONLY before the single allocation,
    // so the extent is immutable for the array's lifetime and is left alone.
    llvm::SmallVector<mlir::Operation*, 4> hazardStores;
    for (auto* st : stores)
      if (order.lookup(st) > earliestAlloc) hazardStores.push_back(st);
    if (hazardStores.empty()) return mlir::success();  // size frozen at alloc

    // A post-allocation reassignment exists.  If every store is straight-line
    // (one block, no enclosing loop) we can SSA-version so each array binds the
    // extent live at its own allocation.  If any store -- in particular the
    // hazardous one -- sits in a loop or a conditional branch, the value live
    // at a downstream array is ambiguous and cannot be statically named.
    mlir::Block* blk = stores.front()->getBlock();
    for (auto* st : stores) {
      if (st->getBlock() != blk || enclosingLoop(st)) {
        return decl.emitError()
               << "shape variable '" << extractShortName(decl)
               << "' is reassigned inside a loop or conditional branch AFTER "
                  "an "
                  "array was allocated from it, so the array's extent symbol "
                  "mutates while the array is live and cannot be SSA-versioned "
                  "in a straight line.  Hoist the size to a single assignment "
                  "before the allocation, or pass it as an explicit dimension.";
      }
    }

    // Order stores by position in the block (program order).
    llvm::sort(stores, [&](mlir::Operation* a, mlir::Operation* b) {
      return a->isBeforeInBlock(b);
    });

    // Forward pass: the first store keeps ``decl``; each subsequent store
    // starts a fresh version.  A load reads the version of the most recent
    // prior store; the store's own RHS load (positioned before the store)
    // still reads the previous version -- so ``m_2 := m + 3`` is exact.
    mlir::OpBuilder builder(decl.getContext());
    hlfir::DeclareOp current = decl;
    unsigned version = 1;
    // Walk the block once; switch ``current`` at each non-first store and
    // redirect loads/stores of the original declare to ``current``.
    for (mlir::Operation& op : llvm::make_early_inc_range(*blk)) {
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(&op)) {
        redirectMemref(ld.getMemrefMutable(), decl, current);
        continue;
      }
      if (!writeTargetOf(&op)) continue;  // not a write to any scalar
      bool isOurStore = false;
      for (auto* s : stores)
        if (s == &op) {
          isOurStore = true;
          break;
        }
      if (!isOurStore) continue;
      if (&op == stores.front()) continue;  // first store keeps ``decl``
      // New version: clone the alloca + declare with a ``_<n>`` suffix.
      ++version;
      current = makeVersion(builder, decl, version);
      // The store WRITES the new version; its value operand (the RHS load)
      // was already redirected to the prior version above.
      if (auto* operand = writeTargetOperand(&op))
        redirectMemref(*operand, decl, current);
    }
    return mlir::success();
  }

  /// Short Fortran name of ``decl`` (the entity after the final scope letter
  /// in its ``uniq_name``), for diagnostics.
  static std::string extractShortName(hlfir::DeclareOp decl) {
    llvm::StringRef u = decl.getUniqName();
    auto e = u.rfind('E');
    return (e == llvm::StringRef::npos) ? u.str() : u.substr(e + 1).str();
  }

  /// Point a load/store memref operand that currently references ``from``'s
  /// result at the matching result of ``to`` (entity #0 / raw #1 preserved).
  static void redirectMemref(mlir::OpOperand& operand, hlfir::DeclareOp from,
                             hlfir::DeclareOp to) {
    if (from == to) return;
    mlir::Value v = operand.get();
    if (v == from.getResult(0))
      operand.set(to.getResult(0));
    else if (v == from.getResult(1))
      operand.set(to.getResult(1));
  }

  /// Clone ``orig``'s ``fir.alloca`` + ``hlfir.declare`` with a ``_<version>``
  /// suffix on the ``uniq_name`` (so ``traceToDecl`` names it ``m_<version>``),
  /// inserted right after the original declare so it dominates every later use.
  static hlfir::DeclareOp makeVersion(mlir::OpBuilder& builder,
                                      hlfir::DeclareOp orig, unsigned version) {
    builder.setInsertionPointAfter(orig);
    mlir::Location loc = orig.getLoc();
    std::string newUniq =
        orig.getUniqName().str() + "_" + std::to_string(version);

    // Fresh storage of the same scalar type, declared under the versioned name
    // (``traceToDecl`` reads the entity after the final ``E`` -> ``m_<n>``).
    auto memTy = mlir::cast<fir::ReferenceType>(orig.getResult(1).getType());
    auto alloca = builder.create<fir::AllocaOp>(loc, memTy.getEleTy());
    auto decl = builder.create<hlfir::DeclareOp>(
        loc, alloca.getResult(), newUniq, /*shape=*/mlir::Value{});
    return decl;
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createVersionShapeScalarsPass() {
  return std::make_unique<VersionShapeScalarsPass>();
}

}  // namespace hlfir_bridge
