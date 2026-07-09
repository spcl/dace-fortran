// ============================================================================
// EliminateDoubleBufferToggle.cpp  --  unroll a constant-trip loop that
// reassigns a double-buffer time-level toggle, substituting the toggle away.
// ============================================================================
// Problem (ICON ``mo_solve_nonhydro::solve_nh``):
//
//     DO istep = 1, 2
//       IF (istep == 1) THEN ; nvar = nnow ; ELSE ; nvar = nnew ; END IF
//       ... p_nh%prog(nnow)%rho(..) ... p_nh%prog(nvar)%rho(..) ...
//     END DO
//
//     ``p_nh%prog(:)`` is an alloc array-of-records double buffer.  The bridge
//     splits ``prog(nnow)`` / ``prog(nnew)`` into one static per-symbol dummy
//     lane bound at call time (``splitDoubleBufferMembers``).  ``nvar`` is a
//     THIRD index symbol, but it is REASSIGNED in-kernel (``nvar = nnow``
//     predictor / ``nvar = nnew`` corrector), so a static lane cannot be
//     re-pointed mid-kernel and the split rejects the function.
//
// Key observation: ``nvar`` is a pure alias of the time-level symbols -- its
// only definitions are ``nvar = nnow`` and ``nvar = nnew`` under a branch on
// the (constant-trip) ``istep`` loop.  Fully unrolling that loop makes the
// branch condition constant per copy, so ``nvar`` resolves to ``nnow`` in the
// predictor copy and ``nnew`` in the corrector copy.  After substitution only
// the stable symbols ``nnow`` / ``nnew`` index ``prog``, and the existing
// static-lane split handles them with no runtime-indexed companion.
//
// What this pass does (narrow by construction):
//   * Detects a reassigned toggle: a scalar slot whose loaded value indexes an
//     alloc/pointer array-of-records element AND that is stored to in-kernel.
//   * Finds the enclosing ``fir.do_loop`` of the toggle store; requires a
//     constant trip count.
//   * Fully unrolls that one loop, substituting the induction constant per
//     copy, folding the ``fir.if`` whose condition becomes constant, and
//     forwarding the toggle store to its loads -- so the toggle slot is dead
//     and every ``prog(toggle)`` reads ``prog(nnow)`` / ``prog(nnew)``.
//
// What it does NOT touch: any loop that does not assign a double-buffer toggle
// (it is a no-op there).  Runs BEFORE ``hlfir-split-aor-dummies``.
// ============================================================================

#include <algorithm>
#include <map>
#include <optional>

#include "Passes.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

namespace hlfir_bridge {
namespace {

/// Peel ``fir.convert`` chains and return the underlying value.
static mlir::Value peelConvert(mlir::Value v) {
  while (auto c = v.getDefiningOp<fir::ConvertOp>()) v = c.getValue();
  return v;
}

/// Fold a value to a constant integer through ``arith.constant`` /
/// ``arith.addi`` / ``arith.subi`` / ``arith.muli`` / ``fir.convert`` over
/// constant operands.  Returns nullopt if it does not reduce to a constant.
static std::optional<int64_t> foldConstInt(mlir::Value v) {
  if (!v) return std::nullopt;
  llvm::APInt ap;
  if (mlir::matchPattern(v, mlir::m_ConstantInt(&ap))) return ap.getSExtValue();
  auto* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) return foldConstInt(cv.getValue());
  if (auto add = mlir::dyn_cast<mlir::arith::AddIOp>(def)) {
    auto a = foldConstInt(add.getLhs()), b = foldConstInt(add.getRhs());
    if (a && b) return *a + *b;
  }
  if (auto sub = mlir::dyn_cast<mlir::arith::SubIOp>(def)) {
    auto a = foldConstInt(sub.getLhs()), b = foldConstInt(sub.getRhs());
    if (a && b) return *a - *b;
  }
  if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(def)) {
    auto a = foldConstInt(mul.getLhs()), b = foldConstInt(mul.getRhs());
    if (a && b) return *a * *b;
  }
  return std::nullopt;
}

/// Evaluate an ``arith.cmpi`` whose operands fold to constants.
static std::optional<bool> evalCmpi(mlir::Value cond) {
  auto cmp = cond.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp) {
    // A bare i1 constant condition.
    llvm::APInt ap;
    if (mlir::matchPattern(cond, mlir::m_ConstantInt(&ap))) return ap != 0;
    return std::nullopt;
  }
  auto l = foldConstInt(cmp.getLhs()), r = foldConstInt(cmp.getRhs());
  if (!l || !r) return std::nullopt;
  using P = mlir::arith::CmpIPredicate;
  switch (cmp.getPredicate()) {
    case P::eq:
      return *l == *r;
    case P::ne:
      return *l != *r;
    case P::slt:
      return *l < *r;
    case P::sle:
      return *l <= *r;
    case P::sgt:
      return *l > *r;
    case P::sge:
      return *l >= *r;
    case P::ult:
      return (uint64_t)*l < (uint64_t)*r;
    case P::ule:
      return (uint64_t)*l <= (uint64_t)*r;
    case P::ugt:
      return (uint64_t)*l > (uint64_t)*r;
    case P::uge:
      return (uint64_t)*l >= (uint64_t)*r;
  }
  return std::nullopt;
}

/// Cloner that unrolls one constant-trip ``fir.do_loop`` while forwarding the
/// tracked scalar slots (the induction mirror + the toggle) to their loads and
/// folding constant-condition ``fir.if`` ops.  ``mem`` maps a tracked slot
/// (original address SSA value, defined outside the loop) to its current value.
struct UnrollCloner {
  mlir::OpBuilder& b;
  mlir::IRMapping& vmap;
  llvm::DenseMap<mlir::Value, mlir::Value>& mem;
  const llvm::DenseSet<mlir::Value>& tracked;
  bool failed = false;

  bool isTracked(mlir::Value slot) const { return tracked.count(slot); }

  void processBlock(mlir::Block& body) {
    for (mlir::Operation& op : llvm::make_early_inc_range(body)) {
      if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
      processOp(&op);
      if (failed) return;
    }
  }

  void processOp(mlir::Operation* op) {
    // --- tracked-slot store: record value, drop the store ---
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
      if (isTracked(st.getMemref())) {
        mem[st.getMemref()] = vmap.lookupOrDefault(st.getValue());
        return;
      }
    }
    if (auto as = mlir::dyn_cast<hlfir::AssignOp>(op)) {
      if (isTracked(as.getLhs())) {
        mem[as.getLhs()] = vmap.lookupOrDefault(as.getRhs());
        return;
      }
    }
    // --- tracked-slot load: forward the recorded value ---
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(op)) {
      auto it = mem.find(ld.getMemref());
      if (it != mem.end()) {
        vmap.map(ld.getResult(), it->second);
        return;
      }
    }
    // --- constant-condition fir.if: keep only the taken branch ---
    if (auto iff = mlir::dyn_cast<fir::IfOp>(op)) {
      auto cond = vmap.lookupOrDefault(iff.getCondition());
      if (auto taken = evalCmpi(cond)) {
        mlir::Region& reg = *taken ? iff.getThenRegion() : iff.getElseRegion();
        if (!reg.empty()) {
          processBlock(reg.front());
          if (failed) return;
          // Map the fir.if results to the taken region's yields.
          if (iff.getNumResults() > 0) {
            auto* term = reg.front().getTerminator();
            for (auto [res, y] : llvm::zip(iff.getResults(), term->getOperands()))
              vmap.map(res, vmap.lookupOrDefault(y));
          }
        }
        return;
      }
      // --- non-constant fir.if ---
      // A tracked slot's store is DROPPED (its value lives only in ``mem``), and
      // its loads are forwarded only when the cloner walks them directly.  A
      // structural clone of a non-constant ``fir.if`` copies the whole nested
      // subtree in bulk, so a ``fir.load`` of a tracked slot inside it is copied
      // verbatim -- reading a slot this pass never writes, which surfaces as an
      // unpopulated free symbol (e.g. ICON ``solve_nh``'s ``istep`` inside the
      // data-dependent ``ELSE IF (istep == 2 .AND. idyn == ndyn_substeps_var(jg))``
      // exner_dyn_incr guard).  Spill the substituted tracked values back into
      // their slots first so those bulk-cloned nested loads read the right value.
      spillTrackedSlots(op->getLoc());
      // fall through to the structural clone below.
    }
    // --- nested fir.do_loop: rebuild so tracked loads inside resolve ---
    if (auto loop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      cloneNestedLoop(loop);
      return;
    }
    // --- default: structural clone ---
    b.clone(*op, vmap);
  }

  void cloneNestedLoop(fir::DoLoopOp loop) {
    auto lb = vmap.lookupOrDefault(loop.getLowerBound());
    auto ub = vmap.lookupOrDefault(loop.getUpperBound());
    auto st = vmap.lookupOrDefault(loop.getStep());
    llvm::SmallVector<mlir::Value> init;
    for (auto v : loop.getInitArgs()) init.push_back(vmap.lookupOrDefault(v));
    bool unordered = loop.getUnordered().value_or(false);
    bool finalVal = loop.getFinalValue().has_value();
    auto nl = b.create<fir::DoLoopOp>(loop.getLoc(), lb, ub, st, unordered, finalVal, init);
    auto& ob = loop.getRegion().front();
    auto& nb = nl.getRegion().front();
    for (auto [o, n] : llvm::zip(ob.getArguments(), nb.getArguments())) vmap.map(o, n);
    mlir::OpBuilder::InsertionGuard g(b);
    // The builder gave the new loop body a default ``fir.result`` terminator;
    // emit the cloned body before it, then re-point its operands.
    mlir::Operation* nterm = nb.empty() ? nullptr : nb.getTerminator();
    if (nterm)
      b.setInsertionPoint(nterm);
    else
      b.setInsertionPointToEnd(&nb);
    processBlock(ob);
    if (failed) return;
    llvm::SmallVector<mlir::Value> res;
    for (auto v : ob.getTerminator()->getOperands()) res.push_back(vmap.lookupOrDefault(v));
    if (nterm)
      nterm->setOperands(res);
    else
      b.create<fir::ResultOp>(loop.getLoc(), res);
    for (auto [o, n] : llvm::zip(loop.getResults(), nl.getResults())) vmap.map(o, n);
  }

  /// Materialise the current substituted value of every tracked slot back into
  /// its memref, so that a subsequent bulk structural clone (of a non-constant
  /// ``fir.if``) whose nested ``fir.load``s read those slots directly observe
  /// the substituted value instead of a never-written slot.  ``mem`` already
  /// holds the cloned (correctly-typed, induction-substituted) stored value for
  /// each tracked slot, so storing it back is type-preserving.
  void spillTrackedSlots(mlir::Location loc) {
    for (auto& kv : mem)
      if (kv.second) b.create<fir::StoreOp>(loc, kv.second, kv.first);
  }
};

struct EliminateDoubleBufferTogglePass
    : public mlir::PassWrapper<EliminateDoubleBufferTogglePass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateDoubleBufferTogglePass)

  llvm::StringRef getArgument() const final { return "hlfir-eliminate-double-buffer-toggle"; }
  llvm::StringRef getDescription() const final {
    return "Unroll the constant-trip loop that reassigns a double-buffer "
           "time-level toggle (ICON solve_nh's `nvar = nnow`/`nvar = nnew`) and "
           "substitute the toggle away, leaving only the stable time-level "
           "symbols for the static double-buffer split.";
  }

  /// Does the address ``slot`` get stored to anywhere (directly, or through an
  /// element/component designate)?
  static bool addressIsStored(mlir::Value slot) {
    for (auto* u : slot.getUsers()) {
      if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) {
        if (st.getMemref() == slot) return true;
      } else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u)) {
        if (as.getLhs() == slot) return true;
      } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
        if (dg.getMemref() == slot && addressIsStored(dg.getResult())) return true;
      }
    }
    return false;
  }

  /// Trace an array-of-records element index back to its declared scalar slot.
  static mlir::Value indexSlot(mlir::Value idx) {
    idx = peelConvert(idx);
    if (auto ld = idx.getDefiningOp<fir::LoadOp>()) {
      mlir::Value m = ld.getMemref();
      while (auto dg = m.getDefiningOp<hlfir::DesignateOp>()) m = dg.getMemref();
      return m;
    }
    return {};
  }

  void runOnOperation() override {
    getOperation().walk([&](mlir::func::FuncOp func) {
      if (func.isExternal()) return;
      runOnFunc(func);
    });
  }

  /// Trace an array-of-records element designate back to its root declare and
  /// the member path above the array (e.g. ``p%prog(idx)`` -> root ``p``,
  /// path ``["prog"]``).  Returns a null root if the chain is not an
  /// AoR-member access.
  static std::pair<mlir::Value, llvm::SmallVector<std::string>> traceRootPath(hlfir::DesignateOp elemDg) {
    auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(elemDg.getMemref().getDefiningOp());
    if (!ld) return {{}, {}};
    llvm::SmallVector<std::string> path;
    mlir::Value v = ld.getMemref();
    while (auto md = v.getDefiningOp<hlfir::DesignateOp>()) {
      auto comp = md.getComponentAttr();
      if (!comp) break;
      path.push_back(comp.getValue().str());
      v = md.getMemref();
    }
    std::reverse(path.begin(), path.end());
    return {v, path};
  }

  void runOnFunc(mlir::func::FuncOp func) {
    // 1. Group array-of-records element accesses by (root, member-path) and
    //    collect the distinct index slots per group.  Only a group indexed by
    //    >= 2 distinct symbols is a double buffer (mirrors the split's gate);
    //    a single runtime index is an ordinary AoR access, NOT a toggle.
    std::map<std::pair<void*, llvm::SmallVector<std::string>>, llvm::DenseSet<mlir::Value>> slotsPerGroup;
    func.walk([&](hlfir::DesignateOp dg) {
      if (dg.getComponentAttr()) return;
      if (dg.getIndices().size() != 1) return;
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(dg.getResult().getType());
      if (!refTy || !mlir::isa<fir::RecordType>(refTy.getEleTy())) return;
      mlir::Value slot = indexSlot(dg.getIndices()[0]);
      if (!slot) return;
      auto [root, path] = traceRootPath(dg);
      if (!root) return;
      slotsPerGroup[{root.getAsOpaquePointer(), path}].insert(slot);
    });

    // 2. A toggle is a slot in a >= 2-symbol group that is reassigned in-kernel.
    llvm::DenseSet<mlir::Value> toggleSlots;
    for (auto& [key, slots] : slotsPerGroup) {
      if (slots.size() < 2) continue;  // single index -> ordinary AoR access
      for (mlir::Value slot : slots)
        if (addressIsStored(slot)) toggleSlots.insert(slot);
    }
    if (toggleSlots.empty()) return;

    // 2. For each toggle slot, find the enclosing fir.do_loop of a store to it
    //    and unroll that loop (each loop unrolled at most once).
    llvm::DenseSet<mlir::Operation*> doneLoops;
    for (mlir::Value slot : toggleSlots) {
      for (auto* u : llvm::to_vector(slot.getUsers())) {
        bool isStore = false;
        if (auto st = mlir::dyn_cast<fir::StoreOp>(u))
          isStore = st.getMemref() == slot;
        else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u))
          isStore = as.getLhs() == slot;
        if (!isStore) continue;
        auto loop = u->getParentOfType<fir::DoLoopOp>();
        if (!loop || doneLoops.count(loop)) continue;
        doneLoops.insert(loop);
        unrollLoop(loop, toggleSlots);
      }
    }
  }

  void unrollLoop(fir::DoLoopOp loop, const llvm::DenseSet<mlir::Value>& toggleSlots) {
    auto lo = foldConstInt(loop.getLowerBound());
    auto hi = foldConstInt(loop.getUpperBound());
    auto step = foldConstInt(loop.getStep());
    if (!lo || !hi || !step || *step == 0) {
      loop.emitError(
          "double-buffer toggle is reassigned inside a loop whose "
          "trip count is not a compile-time constant; cannot unroll "
          "to substitute the toggle.  Keep the time-level rotation "
          "in the driver, outside the extracted kernel.");
      signalPassFailure();
      return;
    }
    int64_t trips = (*step > 0) ? ((*hi - *lo) / *step + 1) : ((*lo - *hi) / (-*step) + 1);
    if (trips < 1 || trips > 64) return;  // degenerate / runaway guard

    // Tracked slots = the toggle slots + this loop's induction-mirror slots
    // (a slot that receives a store of one of the loop's block args).
    llvm::DenseSet<mlir::Value> tracked = toggleSlots;
    auto& body = loop.getRegion().front();
    for (mlir::Operation& op : body) {
      if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
        mlir::Value v = peelConvert(st.getValue());
        for (auto ba : body.getArguments())
          if (v == ba) tracked.insert(st.getMemref());
      }
    }

    // Block args are [induction, iterArgs...]; the loop body terminator (and
    // the loop results) are [finalValue? nextInduction : (), nextIterArgs...].
    // Thread only the iter args between trips; compute the induction directly.
    unsigned numIters = loop.getInitArgs().size();
    mlir::OpBuilder b(loop);
    llvm::SmallVector<mlir::Value> iterVals(loop.getInitArgs().begin(), loop.getInitArgs().end());
    llvm::SmallVector<mlir::Value> finalResults;
    for (int64_t t = 0; t < trips; ++t) {
      int64_t ivVal = *lo + t * *step;
      mlir::IRMapping vmap;
      auto ivConst = b.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), ivVal);
      vmap.map(body.getArgument(0), ivConst);
      for (unsigned k = 0; k < numIters; ++k) vmap.map(body.getArgument(k + 1), iterVals[k]);

      llvm::DenseMap<mlir::Value, mlir::Value> mem;
      UnrollCloner cloner{b, vmap, mem, tracked};
      cloner.processBlock(body);
      if (cloner.failed) {
        signalPassFailure();
        return;
      }
      // Map this trip's terminator operands (= loop results order); the trailing
      // ``numIters`` are the iter args carried into the next trip.
      auto* term = body.getTerminator();
      llvm::SmallVector<mlir::Value> yields;
      for (auto v : term->getOperands()) yields.push_back(vmap.lookupOrDefault(v));
      iterVals.assign(yields.end() - numIters, yields.end());
      finalResults = yields;
    }
    // Re-point the loop's results to the last trip's yields and drop the loop.
    for (auto [res, val] : llvm::zip(loop.getResults(), finalResults)) res.replaceAllUsesWith(val);
    loop.erase();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createEliminateDoubleBufferTogglePass() {
  return std::make_unique<EliminateDoubleBufferTogglePass>();
}

}  // namespace hlfir_bridge
