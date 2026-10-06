// ============================================================================
// RaggedAos.cpp  --  runtime-sized ALLOCATABLE members of an array of records.
// ============================================================================
// See ``ragged_aos.h`` for the representation.  The work splits into
//
//   1. *Sites*: every ``fir.store embox(fir.allocmem) to a(i...) % m`` (an
//      ALLOCATE) and every store of a null box (the descriptor reset after a
//      DEALLOCATE) on an element of the array.
//   2. *Bounds*: before the top-level operation holding the first site, the
//      loops enclosing each site are re-created (bounds only, conditions are
//      ignored -- an over-approximation is still a bound) and each extent is
//      re-evaluated into a running maximum.  ``Bounder::pure`` clones
//      side-effect-free arithmetic over values available there, the loop
//      indices of the re-created loops included; ``Bounder::upper`` also
//      bounds a monotone expression of a counter of the
//      ``x = 0; DO ...; [IF (c)] x = x + 1; END DO`` idiom by the counting
//      loop's trip count.
//   3. *Rewrite*: the companion is allocated once from the maxima; every
//      ALLOCATE records its live extents in the ``_len`` table, every access
//      of the member goes to the companion (element and section designates
//      gain the element's indices in front, whole-member uses become the live
//      section, ``SIZE`` and ``ALLOCATED`` read the table).
// ============================================================================

#include <optional>
#include <tuple>
#include <variant>

#include "bridge/trace_utils.h"  // traceConstInt
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm_compat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "passes/ragged_aos.h"

namespace hlfir_bridge {

namespace {

constexpr int kRaggedMaxDepth = 32;

/// Numbers the DO variables of the re-created loops (module-wide unique names).
thread_local unsigned raggedLoopCount = 0;

/// The storage a memory reference ultimately addresses: follows declares, designates and box reinterprets.
mlir::Value storageRoot(mlir::Value v) {
  for (int i = 0; i < kRaggedMaxDepth && v; ++i) {
    mlir::Operation* d = v.getDefiningOp();
    if (!d) return v;
    if (auto x = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      v = x.getMemref();
    } else if (auto x = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      v = x.getMemref();
    } else if (auto x = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = x.getValue();
    } else if (auto x = mlir::dyn_cast<fir::LoadOp>(d)) {
      v = x.getMemref();
    } else if (auto x = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
      v = x.getVal();
    } else if (auto x = mlir::dyn_cast<fir::EmboxOp>(d)) {
      v = x.getMemref();
    } else if (auto x = mlir::dyn_cast<fir::ReboxOp>(d)) {
      v = x.getBox();
    } else {
      return v;
    }
  }
  return v;
}

/// Identifies a storage across aliases: a module variable by its symbol (every inlined use takes its own
/// ``fir.address_of``), anything else by its root value.
using RootKey = const void*;

RootKey rootKey(mlir::Value v) {
  mlir::Value const root = storageRoot(v);
  if (auto global = root.getDefiningOp<fir::AddrOfOp>()) return global.getSymbol().getAsOpaquePointer();
  return root.getAsOpaquePointer();
}

bool isElementDesignate(hlfir::DesignateOp dg) {
  if (dg.getComponentAttr() || dg.getIndices().empty()) return false;
  for (bool const t : dg.getIsTriplet())
    if (t) return false;
  return true;
}

/// ``a(i...) % memName`` designates of ``decl`` (or of an inlined alias of it).
llvm::SmallVector<hlfir::DesignateOp, 8> memberDesignates(hlfir::DeclareOp decl, llvm::StringRef memName) {
  llvm::SmallVector<hlfir::DesignateOp, 8> out;
  RootKey const root = rootKey(decl.getResult(0));
  auto func = decl->getParentOfType<mlir::func::FuncOp>();
  if (!func) return out;
  func.walk([&](hlfir::DesignateOp dg) {
    if (!dg.getComponentAttr() || dg.getComponentAttr().getValue() != memName) return;
    auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(dg.getMemref().getDefiningOp());
    if (parent && isElementDesignate(parent) && rootKey(parent.getMemref()) == root) out.push_back(dg);
  });
  return out;
}

/// The ``fir.allocmem`` a stored box comes from, if any.
fir::AllocMemOp allocationOf(mlir::Value box) {
  for (int i = 0; i < kRaggedMaxDepth && box; ++i) {
    mlir::Operation* d = box.getDefiningOp();
    if (!d) return {};
    if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) return am;
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      box = eb.getMemref();
    } else if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      box = cv.getValue();
    } else {
      return {};
    }
  }
  return {};
}

/// The ancestor of ``op`` that sits directly in ``block``, or null.
mlir::Operation* ancestorIn(mlir::Block* block, mlir::Operation* op) { return block->findAncestorOpInBlock(*op); }

/// One ALLOCATE of the member.
struct Site {
  fir::StoreOp store;
  fir::AllocMemOp alloc;
  hlfir::DesignateOp parent;
};

/// Everything the member's flattening needs, found before anything is changed.
struct Plan {
  mlir::Operation* point = nullptr;  ///< caps are computed and the companion allocated before this op
  llvm::SmallVector<Site, 4> sites;
  llvm::SmallVector<fir::StoreOp, 4> releases;   ///< null-box stores (descriptor reset after DEALLOCATE)
  llvm::SmallVector<mlir::Operation*, 4> frees;  ///< ``fir.freemem`` of the member
  llvm::SmallVector<hlfir::DesignateOp, 8> designates;
  unsigned rank = 0;
  mlir::Type eleTy;
  llvm::SmallVector<int64_t, 4> outerShape;        ///< unknown extents come from ``outerExtents``
  llvm::SmallVector<mlir::Value, 4> outerExtents;  ///< the runtime outer extents, from the array's own ALLOCATE
  /// A module array of records the kernel does not ALLOCATE: its runtime outer extents are read from its descriptor.
  bool outerFromDescriptor = false;
};

/// A write to a storage root between the plan's point and its last site.
struct Write {
  mlir::Operation* op;
  mlir::Value value;  ///< the stored value, null for an opaque write (a call)
};

/// Clones the computation of an extent before the plan's point (``b`` null: only checks that it can).
class Bounder {
 public:
  Bounder(mlir::Operation* point, mlir::Operation* last, mlir::DominanceInfo& dom) : point_(point), dom_(dom) {
    for (auto it = point->getIterator(); it != std::next(last->getIterator()); ++it) {
      it->walk([&](mlir::Operation* op) {
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
          writes_[rootKey(st.getMemref())].push_back({op, st.getValue()});
        } else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(op)) {
          writes_[rootKey(as.getLhs())].push_back({op, as.getRhs()});
        } else if (mlir::isa<fir::CallOp>(op)) {
          for (mlir::Value const arg : op->getOperands())
            if (mlir::isa<fir::ReferenceType, fir::BaseBoxType>(arg.getType()))
              writes_[rootKey(arg)].push_back({op, {}});
        }
        if (auto loop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
          // Flang keeps the DO variable in memory: the body first stores the iteration count into it.
          auto args = loop.getRegionIterArgs();
          if (!args.empty())
            if (auto st = mlir::dyn_cast<fir::StoreOp>(loop.getBody()->front()); st && st.getValue() == args.front())
              ivLoops_[rootKey(st.getMemref())].push_back(loop);
        }
      });
    }
  }

  /// Opens a copy of every loop enclosing ``site`` up to the point's level; returns false if a bound is not pure.
  bool openLoops(mlir::Operation* site, mlir::OpBuilder* b, mlir::IRMapping& map) {
    llvm::SmallVector<mlir::Operation*, 4> scopes;
    for (mlir::Operation* op = site->getParentOp(); op && op != point_->getParentOp(); op = op->getParentOp()) {
      if (!mlir::isa<fir::DoLoopOp, fir::IfOp>(op)) return false;  // a WHILE or other region: no trip range
      scopes.push_back(op);
    }
    for (mlir::Operation* scope : llvm::reverse(scopes)) {
      if (auto guard = mlir::dyn_cast<fir::IfOp>(scope)) {
        // A guard over values available here is kept, so the extents are only evaluated where the ALLOCATE can
        // run; any other guard is dropped (the bound then covers both branches).
        mlir::Value cond = pure(guard.getCondition(), b, map);
        if (!cond || !b) continue;
        auto loc = guard.getLoc();
        if (!guard.getThenRegion().isAncestor(site->getParentRegion())) {
          mlir::Value const yes = b->create<mlir::arith::ConstantIntOp>(loc, b->getI1Type(), 1);
          cond = b->create<mlir::arith::XOrIOp>(loc, cond, yes);
        }
        auto copy = b->create<fir::IfOp>(loc, cond, /*withElseRegion=*/false);
        mlir::Block& then = copy.getThenRegion().front();
        if (then.empty() || !then.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          b->setInsertionPointToEnd(&then);
          b->create<fir::ResultOp>(loc);
        }
        b->setInsertionPoint(then.getTerminator());
        continue;
      }
      auto loop = mlir::cast<fir::DoLoopOp>(scope);
      mlir::Value const lb = pure(loop.getLowerBound(), b, map);
      mlir::Value const ub = pure(loop.getUpperBound(), b, map);
      mlir::Value const step = pure(loop.getStep(), b, map);
      if (!lb || !ub || !step) return false;
      if (!b) {
        map.map(loop.getInductionVar(), loop.getInductionVar());
        continue;
      }
      // Shaped like Flang's DO: the counter lives in a variable the body stores first and reads back.
      auto loc = loop.getLoc();
      auto i64 = b->getI64Type();
      mlir::OpBuilder top(point_);
      auto slot = top.create<fir::AllocaOp>(loc, i64);
      auto var = top.create<hlfir::DeclareOp>(loc, slot.getResult(), "_ragged_it" + std::to_string(raggedLoopCount++));
      mlir::Value const init = b->create<fir::ConvertOp>(loc, i64, lb);
      auto copy = b->create<fir::DoLoopOp>(loc, lb, ub, step, false, false, mlir::ValueRange{init});
      mlir::Block* body = copy.getBody();
      if (!body->empty() && body->back().hasTrait<mlir::OpTrait::IsTerminator>()) body->back().erase();
      b->setInsertionPointToEnd(body);
      b->create<fir::StoreOp>(loc, copy.getRegionIterArgs().front(), var.getResult(0));
      mlir::Value const index =
          b->create<fir::ConvertOp>(loc, b->getIndexType(), b->create<fir::LoadOp>(loc, var.getResult(0)));
      auto current = b->create<fir::LoadOp>(loc, var.getResult(0));
      mlir::Value const next =
          b->create<mlir::arith::AddIOp>(loc, current, b->create<fir::ConvertOp>(loc, i64, step).getResult());
      b->create<fir::ResultOp>(loc, next);
      map.map(loop.getInductionVar(), index);
      b->setInsertionPoint(current);
    }
    return true;
  }

  /// ``v`` itself, cloned before the point (null if it is not side-effect-free arithmetic over values available
  /// there or over the copied loop indices).
  mlir::Value pure(mlir::Value v, mlir::OpBuilder* b, mlir::IRMapping& map) {
    if (map.contains(v)) return map.lookup(v);
    if (dom_.properlyDominates(v, point_)) return v;
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto load = mlir::dyn_cast<fir::LoadOp>(def)) {
      RootKey const root = rootKey(load.getMemref());
      if (writes_.contains(root)) {
        mlir::Value const index = loopIndex(load, root, b, map);
        if (index) map.map(v, index);
        return index;
      }
    } else if (!(mlir::isa<mlir::arith::ArithDialect>(def->getDialect()) ||
                 mlir::isa<fir::ConvertOp, hlfir::DeclareOp, hlfir::DesignateOp, fir::DummyScopeOp, fir::ShapeOp,
                           fir::ShapeShiftOp>(def)) ||
               def->getNumRegions() != 0) {
      return {};
    }
    for (mlir::Value const operand : def->getOperands())
      if (!pure(operand, b, map)) return {};
    if (!b) {
      for (mlir::Value const r : def->getResults()) map.map(r, r);
      return v;
    }
    mlir::Operation* copy = b->clone(*def, map);
    return copy->getResult(mlir::cast<mlir::OpResult>(v).getResultNumber());
  }

  /// An upper bound of ``v`` executed at ``site``: ``pure`` where possible, else a monotone expression of an idiom
  /// counter bounded by its counting loop's trip count.
  mlir::Value upper(mlir::Value v, mlir::Operation* site, mlir::OpBuilder* b, mlir::IRMapping& map) {
    if (mlir::Value const p = pure(v, b, map)) return p;
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      mlir::Value const x = upper(cv.getValue(), site, b, map);
      return x && b ? b->create<fir::ConvertOp>(cv.getLoc(), cv.getType(), x).getResult() : x;
    }
    if (mlir::isa<mlir::arith::AddIOp, mlir::arith::MaxSIOp>(def)) {
      mlir::Value const x = upper(def->getOperand(0), site, b, map);
      mlir::Value const y = x ? upper(def->getOperand(1), site, b, map) : mlir::Value{};
      if (!y || !b) return y;
      mlir::IRMapping operands;
      operands.map(def->getOperand(0), x);
      operands.map(def->getOperand(1), y);
      return b->clone(*def, operands)->getResult(0);
    }
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
      // Flang's extent clamp ``x > 0 ? x : 0`` is monotone in ``x``
      auto cmp = sel.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
      auto zero = traceConstInt(sel.getFalseValue());
      if (!cmp || cmp.getPredicate() != mlir::arith::CmpIPredicate::sgt || cmp.getLhs() != sel.getTrueValue() ||
          !zero || *zero != 0 || traceConstInt(cmp.getRhs()) != 0)
        return {};
      mlir::Value const x = upper(sel.getTrueValue(), site, b, map);
      if (!x || !b) return x;
      return b->create<mlir::arith::MaxSIOp>(sel.getLoc(), x, pure(sel.getFalseValue(), b, map)).getResult();
    }
    if (auto load = mlir::dyn_cast<fir::LoadOp>(def)) return counterBound(load, site, b, map);
    return {};
  }

 private:
  /// The DO variable ``root`` read inside one of the copied loops: that loop's copied index.
  mlir::Value loopIndex(fir::LoadOp load, RootKey root, mlir::OpBuilder* b, mlir::IRMapping& map) {
    auto it = ivLoops_.find(root);
    if (it == ivLoops_.end()) return {};
    for (fir::DoLoopOp loop : it->second) {
      if (!map.contains(loop.getInductionVar()) || !loop->isProperAncestor(load)) continue;
      // Every write is the iteration store or the final store after the loop; nothing else may write it.
      for (const Write& w : writes_.lookup(root))
        if (!w.value || !(llvm::is_contained(loop.getRegionIterArgs(), w.value) ||
                          mlir::isa_and_nonnull<fir::DoLoopOp>(w.value.getDefiningOp())))
          return {};
      mlir::Value const index = map.lookup(loop.getInductionVar());
      return b ? b->create<fir::ConvertOp>(load.getLoc(), load.getType(), index).getResult() : load.getResult();
    }
    return {};
  }

  /// Whether ``op`` (recursively) writes ``root``.
  bool writesTo(mlir::Operation* op, RootKey root) {
    for (const Write& w : writes_.lookup(root))
      if (op == w.op || op->isProperAncestor(w.op)) return true;
    return false;
  }

  /// Whether ``value`` is ``root`` plus one.
  static bool isIncrement(mlir::Value value, RootKey root) {
    auto add = value.getDefiningOp<mlir::arith::AddIOp>();
    if (!add) return false;
    for (auto [x, y] : {std::pair{add.getLhs(), add.getRhs()}, std::pair{add.getRhs(), add.getLhs()}}) {
      auto load = x.getDefiningOp<fir::LoadOp>();
      if (load && rootKey(load.getMemref()) == root && traceConstInt(y) == 1) return true;
    }
    return false;
  }

  /// ``x`` of the counting idiom, read at ``site``: bounded by the trip count of the counting loop before it.
  mlir::Value counterBound(fir::LoadOp load, mlir::Operation* site, mlir::OpBuilder* b, mlir::IRMapping& map) {
    RootKey const root = rootKey(load.getMemref());
    for (const Write& w : writes_.lookup(root))
      if (!w.value || !(traceConstInt(w.value) == 0 || isIncrement(w.value, root))) return {};
    // The nearest earlier write at any enclosing level must be a counting loop, preceded by the reset.
    for (mlir::Operation* op = site; op && op != point_->getParentOp(); op = op->getParentOp()) {
      for (mlir::Operation* prev = op->getPrevNode(); prev; prev = prev->getPrevNode()) {
        if (!writesTo(prev, root)) continue;
        auto loop = mlir::dyn_cast<fir::DoLoopOp>(prev);
        if (!loop || traceConstInt(loop.getStep()) != 1) return {};
        for (const Write& w : writes_.lookup(root))
          if (loop->isProperAncestor(w.op) && !isIncrement(w.value, root)) return {};
        mlir::Operation* reset = loop->getPrevNode();
        while (reset && !writesTo(reset, root)) reset = reset->getPrevNode();
        if (!reset || !llvm::any_of(writes_.lookup(root),
                                    [&](const Write& w) { return w.op == reset && traceConstInt(w.value) == 0; }))
          return {};
        mlir::Value const lb = pure(loop.getLowerBound(), b, map);
        mlir::Value const ub = pure(loop.getUpperBound(), b, map);
        if (!lb || !ub || !b) return lb && ub ? load.getResult() : mlir::Value{};
        auto loc = load.getLoc();
        auto idxTy = b->getIndexType();
        auto toIndex = [&](mlir::Value x) { return b->create<fir::ConvertOp>(loc, idxTy, x).getResult(); };
        mlir::Value const one = b->create<mlir::arith::ConstantIndexOp>(loc, 1);
        mlir::Value const zero = b->create<mlir::arith::ConstantIndexOp>(loc, 0);
        mlir::Value const span = b->create<mlir::arith::SubIOp>(loc, toIndex(ub), toIndex(lb));
        mlir::Value const trips = b->create<mlir::arith::AddIOp>(loc, span, one);
        mlir::Value const count = b->create<mlir::arith::MaxSIOp>(loc, trips, zero);
        return b->create<fir::ConvertOp>(loc, load.getType(), count).getResult();
      }
    }
    return {};
  }

  mlir::Operation* point_;
  mlir::DominanceInfo& dom_;
  llvm::DenseMap<RootKey, llvm::SmallVector<Write, 2>> writes_;
  llvm::DenseMap<RootKey, llvm::SmallVector<fir::DoLoopOp, 2>> ivLoops_;
};

/// The member's element type and rank, from the record field.
bool memberType(hlfir::DeclareOp decl, llvm::StringRef memName, Plan& plan) {
  mlir::Type t = decl.getResult(0).getType();
  for (int i = 0; i < 4; ++i) {
    if (auto r = mlir::dyn_cast<fir::ReferenceType>(t)) {
      t = r.getEleTy();
    } else if (auto bx = mlir::dyn_cast<fir::BaseBoxType>(t)) {
      t = bx.getEleTy();
    } else if (auto h = mlir::dyn_cast<fir::HeapType>(t)) {
      t = h.getEleTy();
    }
  }
  auto outer = mlir::dyn_cast<fir::SequenceType>(t);
  if (!outer) return false;
  plan.outerShape.assign(outer.getShape().begin(), outer.getShape().end());
  auto rec = mlir::dyn_cast<fir::RecordType>(outer.getEleTy());
  if (!rec) return false;
  auto box = mlir::dyn_cast_or_null<fir::BoxType>(rec.getType(memName));
  if (!box) return false;
  auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy());
  auto seq = heap ? mlir::dyn_cast<fir::SequenceType>(heap.getEleTy()) : fir::SequenceType{};
  if (!seq) return false;
  plan.rank = seq.getDimension();
  plan.eleTy = seq.getEleTy();
  return true;
}

/// Whether ``value`` only feeds ``ALLOCATED`` (``box_addr -> convert -> cmpi eq|ne 0``).
bool isAllocatedQuery(fir::BoxAddrOp addr) {
  for (mlir::Operation* u : addr->getUsers()) {
    auto cv = mlir::dyn_cast<fir::ConvertOp>(u);
    if (!cv) return false;
    for (mlir::Operation* cu : cv->getUsers()) {
      auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(cu);
      if (!cmp || traceConstInt(cmp.getRhs()) != 0 ||
          (cmp.getPredicate() != mlir::arith::CmpIPredicate::ne &&
           cmp.getPredicate() != mlir::arith::CmpIPredicate::eq))
        return false;
    }
  }
  return true;
}

/// The plan, or nullopt when some site, extent or access is outside what the flattening handles.
std::optional<Plan> planMember(hlfir::DeclareOp decl, llvm::StringRef memName) {
  Plan plan;
  if (!memberType(decl, memName, plan)) return std::nullopt;
  plan.designates = memberDesignates(decl, memName);
  for (hlfir::DesignateOp dg : plan.designates) {
    auto parent = mlir::cast<hlfir::DesignateOp>(dg.getMemref().getDefiningOp());
    for (mlir::Operation* u : dg->getUsers()) {
      if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) {
        if (st.getMemref() != dg.getResult()) return std::nullopt;
        if (auto am = allocationOf(st.getValue())) {
          if (am.getShape().size() != plan.rank) return std::nullopt;
          plan.sites.push_back({st, am, parent});
        } else {
          plan.releases.push_back(st);
        }
      } else if (auto load = mlir::dyn_cast<fir::LoadOp>(u)) {
        for (mlir::Operation* lu : load->getUsers()) {
          if (auto addr = mlir::dyn_cast<fir::BoxAddrOp>(lu)) {
            bool const freed =
                llvm::all_of(addr->getUsers(), [](mlir::Operation* au) { return mlir::isa<fir::FreeMemOp>(au); });
            if (freed) {
              for (mlir::Operation* au : addr->getUsers()) plan.frees.push_back(au);
            } else if (!isAllocatedQuery(addr)) {
              return std::nullopt;
            }
          } else if (!mlir::isa<hlfir::DesignateOp, fir::BoxDimsOp>(lu) && lu->getDialect() != dg->getDialect()) {
            return std::nullopt;  // only hlfir consumers take the live section in place of the descriptor
          }
        }
      } else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u)) {
        if (as.getLhs() != dg.getResult()) return std::nullopt;
      } else {
        return std::nullopt;
      }
    }
  }
  if (plan.sites.empty()) return std::nullopt;

  // The innermost block holding every site and every access: the companion is allocated there.
  mlir::Block* block = plan.sites.front().store->getBlock();
  auto widen = [&](mlir::Operation* op) {
    while (block && !block->findAncestorOpInBlock(*op))
      block = block->getParentOp() ? block->getParentOp()->getBlock() : nullptr;
  };
  for (Site s : plan.sites) widen(s.store);
  for (hlfir::DesignateOp dg : plan.designates) widen(dg);
  if (!block) return std::nullopt;
  mlir::Operation* last = nullptr;
  for (Site s : plan.sites) {
    mlir::Operation* top = ancestorIn(block, s.store);
    if (!plan.point || top->isBeforeInBlock(plan.point)) plan.point = top;
    if (!last || last->isBeforeInBlock(top)) last = top;
  }
  mlir::DominanceInfo dom(decl->getParentOfType<mlir::func::FuncOp>());
  for (hlfir::DesignateOp dg : plan.designates)
    if (!dom.dominates(plan.point, dg.getOperation())) return std::nullopt;

  // Runtime outer extents: those of the array's single ALLOCATE, which must have run by the point.  A record type
  // with allocatable components is allocated through the runtime: ``AllocatableSetBounds(box, dim, 1, ub)``.
  if (llvm::is_contained(plan.outerShape, fir::SequenceType::getUnknownExtent())) {
    RootKey const root = rootKey(decl.getResult(0));
    llvm::SmallVector<mlir::Value, 4> upper(plan.outerShape.size());
    unsigned allocations = 0;
    decl->getParentOfType<mlir::func::FuncOp>().walk([&](fir::CallOp call) {
      auto callee = call.getCallee();
      if (!callee || call.getArgs().empty() || rootKey(call.getArgs()[0]) != root) return;
      llvm::StringRef const name = callee->getRootReference().getValue();
      if (name == "_FortranAAllocatableAllocate") ++allocations;
      if (name != "_FortranAAllocatableSetBounds" || call.getArgs().size() < 4) return;
      auto dim = traceConstInt(call.getArgs()[1]);
      if (dim && *dim >= 0 && static_cast<size_t>(*dim) < upper.size() && traceConstInt(call.getArgs()[2]) == 1)
        upper[*dim] = call.getArgs()[3];
    });
    // A module array the kernel never ALLOCATEs was allocated by the host: its descriptor holds the extents.
    plan.outerFromDescriptor =
        allocations == 0 && mlir::isa_and_nonnull<fir::AddrOfOp>(decl.getMemref().getDefiningOp());
    if (!plan.outerFromDescriptor) {
      if (allocations != 1) return std::nullopt;
      for (auto [extent, ub] : llvm::zip(plan.outerShape, upper)) {
        if (extent != fir::SequenceType::getUnknownExtent()) continue;
        if (!ub || !dom.properlyDominates(ub, plan.point)) return std::nullopt;
        plan.outerExtents.push_back(ub);
      }
    }
  }

  Bounder bounder(plan.point, last, dom);
  for (Site s : plan.sites) {
    mlir::IRMapping map;
    if (!bounder.openLoops(s.store, nullptr, map)) return std::nullopt;
    for (mlir::Value const extent : s.alloc.getShape())
      if (!bounder.upper(extent, s.store, nullptr, map)) return std::nullopt;
  }
  return plan;
}

}  // namespace

void dropUnusedRecordArray(hlfir::DeclareOp decl) {
  RootKey const root = rootKey(decl.getResult(0));
  auto func = decl->getParentOfType<mlir::func::FuncOp>();
  bool used = false;
  func.walk([&](hlfir::DesignateOp dg) { used = used || rootKey(dg.getMemref()) == root; });
  if (used) return;
  // The runtime allocation calls on the descriptor go, then the ``ALLOCATED`` guards they leave empty.
  llvm::SmallVector<mlir::Operation*, 8> dead;
  func.walk([&](fir::CallOp call) {
    auto callee = call.getCallee();
    if (!callee || call.getArgs().empty() || rootKey(call.getArgs()[0]) != root || !call->use_empty()) return;
    llvm::StringRef const name = callee->getRootReference().getValue();
    if (name.starts_with("_FortranAAllocatable") || name == "_FortranAInitialize" || name == "_FortranADestroy")
      dead.push_back(call);
  });
  for (mlir::Operation* op : dead) op->erase();
  func.walk([&](fir::IfOp guard) {
    auto empty = [](mlir::Region& r) { return r.empty() || r.front().getOperations().size() <= 1; };
    if (guard->getNumResults() == 0 && empty(guard.getThenRegion()) && empty(guard.getElseRegion())) guard.erase();
  });
  // Whatever only computed the descriptor's address or arguments is dead now.
  for (bool changed = true; changed;) {
    changed = false;
    func.walk([&](mlir::Operation* op) {
      if (op->use_empty() && op->getNumResults() > 0 && mlir::isMemoryEffectFree(op)) {
        op->erase();
        changed = true;
      } else if (auto load = mlir::dyn_cast<fir::LoadOp>(op); load && load->use_empty()) {
        load.erase();
        changed = true;
      }
    });
  }
}

bool raggedAosMemberFlattenable(hlfir::DeclareOp decl, llvm::StringRef memName) {
  return planMember(decl, memName).has_value();
}

/// ``flatName`` minted as a local of ``func`` instead (``_QM<mod>F<proc>E<name>``), unless a declare already uses it.
static std::string kernelLocalName(mlir::func::FuncOp func, llvm::StringRef flatName) {
  auto [kind, proc] = fir::NameUniquer::deconstruct(func.getSymName());
  auto [varKind, var] = fir::NameUniquer::deconstruct(flatName);
  llvm::SmallVector<llvm::StringRef, 4> modules(proc.modules.begin(), proc.modules.end());
  llvm::SmallVector<llvm::StringRef, 4> procs(proc.procs.begin(), proc.procs.end());
  procs.push_back(proc.name);
  llvm::StringSet<> taken;
  func.walk([&](hlfir::DeclareOp d) { taken.insert(d.getUniqName()); });
  std::string name = fir::NameUniquer::doVariable(modules, procs, proc.blockId, var.name);
  for (int n = 1; taken.contains(name); ++n)
    name = fir::NameUniquer::doVariable(modules, procs, proc.blockId, var.name + "_cc" + std::to_string(n));
  return name;
}

void flattenRaggedAosMember(hlfir::DeclareOp decl, llvm::StringRef memName, const std::string& moduleFlatName) {
  std::optional<Plan> planned = planMember(decl, memName);
  if (!planned) return;
  Plan& plan = *planned;
  // A host-allocated module array's companions hold nothing across calls (no binding marshals them), so they are
  // the kernel's own transients: named in its scope, not as module state the caller would have to pass.
  std::string const flatName = plan.outerFromDescriptor
                                   ? kernelLocalName(decl->getParentOfType<mlir::func::FuncOp>(), moduleFlatName)
                                   : moduleFlatName;
  auto* ctx = decl->getContext();
  auto loc = decl.getLoc();
  mlir::OpBuilder b(plan.point);
  auto idxTy = b.getIndexType();
  auto i64 = b.getI64Type();
  auto cIndex = [&](int64_t v) { return b.create<mlir::arith::ConstantIndexOp>(loc, v).getResult(); };
  auto allocatable = fir::FortranVariableFlagsAttr::get(ctx, fir::FortranVariableFlagsEnum::allocatable);

  // The outer extents as values: the static ones as constants, the runtime ones from the array's ALLOCATE or, for a
  // module array the host allocated, from its descriptor.
  llvm::SmallVector<mlir::Value, 4> outerExtents;
  {
    mlir::Value const descriptor =
        plan.outerFromDescriptor ? b.create<fir::LoadOp>(loc, decl.getResult(0)).getResult() : mlir::Value{};
    auto dynamic = plan.outerExtents.begin();
    for (auto [dim, e] : llvm::enumerate(plan.outerShape)) {
      if (e != fir::SequenceType::getUnknownExtent())
        outerExtents.push_back(cIndex(e));
      else if (descriptor)
        outerExtents.push_back(
            b.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, descriptor, cIndex(static_cast<int64_t>(dim)))
                .getResult(1));
      else
        outerExtents.push_back(b.create<fir::ConvertOp>(loc, idxTy, *dynamic++).getResult());
    }
  }
  llvm::SmallVector<mlir::Value, 4> dynamicOuter;
  for (auto [e, v] : llvm::zip(plan.outerShape, outerExtents))
    if (e == fir::SequenceType::getUnknownExtent()) dynamicOuter.push_back(v);

  /// An allocatable ``name`` of ``extents`` (``dims`` with its runtime ones listed in ``runtime``), allocated here.
  auto allocate = [&](const std::string& name, llvm::ArrayRef<int64_t> dims, mlir::Type eleTy,
                      llvm::ArrayRef<mlir::Value> runtime, llvm::ArrayRef<mlir::Value> extents) {
    auto seq = fir::SequenceType::get(dims, eleTy);
    auto boxTy = fir::BoxType::get(fir::HeapType::get(seq));
    auto slot = b.create<fir::AllocaOp>(loc, boxTy);
    auto var = hlfir_bridge::createDeclare(b, loc, slot.getType(), slot.getType(), slot.getResult(),
                                           /*shape=*/mlir::Value{}, /*typeparams=*/mlir::ValueRange{},
                                           b.getStringAttr(name), allocatable);
    auto mem = b.create<fir::AllocMemOp>(loc, seq, name + ".alloc", mlir::ValueRange{}, runtime);
    auto shape = b.create<fir::ShapeOp>(loc, extents);
    b.create<fir::StoreOp>(loc, b.create<fir::EmboxOp>(loc, boxTy, mem.getResult(), shape.getResult()).getResult(),
                           var.getResult(0));
    return b.create<fir::LoadOp>(loc, var.getResult(0)).getResult();
  };

  // The live-extent table; entries hold extent + 1, so its zero fill reads as "unallocated".
  llvm::SmallVector<int64_t, 6> lenDims(plan.outerShape.begin(), plan.outerShape.end());
  llvm::SmallVector<mlir::Value, 6> lenExtents(outerExtents.begin(), outerExtents.end());
  if (plan.rank > 1) {  // a rank-1 member needs no extent index
    lenDims.push_back(plan.rank);
    lenExtents.push_back(cIndex(plan.rank));
  }
  mlir::Value const lenTable = allocate(flatName + "_len", lenDims, i64, dynamicOuter, lenExtents);
  b.create<hlfir::AssignOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, i64, 0).getResult(), lenTable);

  // Caps: the running maximum of every site's extents over its loops.  Fortran ``integer(8)`` updated through
  // ``hlfir.assign``, like source code: an extent read from an array element (``cnt(i)``) then carries its subscript
  // into the update -- a raw ``fir.store`` of an ``index`` slot reached the SDFG as ``max(cap, cnt)``.
  llvm::SmallVector<mlir::Value, 4> capSlots;
  for (unsigned d = 0; d < plan.rank; ++d) {
    auto slot = b.create<fir::AllocaOp>(loc, i64);
    auto capDecl = b.create<hlfir::DeclareOp>(loc, slot.getResult(), flatName + "_cap" + std::to_string(d));
    b.create<hlfir::AssignOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, i64, 0).getResult(), capDecl.getResult(0));
    capSlots.push_back(capDecl.getResult(0));
  }
  mlir::func::FuncOp func = decl->getParentOfType<mlir::func::FuncOp>();
  mlir::DominanceInfo dom(func);
  mlir::Operation* last = plan.point;
  for (Site s : plan.sites)
    if (mlir::Operation* top = ancestorIn(plan.point->getBlock(), s.store); last->isBeforeInBlock(top)) last = top;
  Bounder bounder(plan.point, last, dom);
  for (Site s : plan.sites) {
    mlir::OpBuilder::InsertionGuard guard(b);
    mlir::IRMapping map;
    bounder.openLoops(s.store, &b, map);
    for (unsigned d = 0; d < plan.rank; ++d) {
      mlir::Value const extent = bounder.upper(s.alloc.getShape()[d], s.store, &b, map);
      mlir::Value const asI64 = b.create<fir::ConvertOp>(loc, i64, extent);
      mlir::Value const seen = b.create<fir::LoadOp>(loc, capSlots[d]);
      b.create<hlfir::AssignOp>(loc, b.create<mlir::arith::MaxSIOp>(loc, seen, asI64).getResult(), capSlots[d]);
    }
  }
  llvm::SmallVector<mlir::Value, 4> caps;
  for (mlir::Value const slot : capSlots)
    caps.push_back(b.create<fir::ConvertOp>(loc, idxTy, b.create<fir::LoadOp>(loc, slot)).getResult());

  // The companion, padded to the caps.
  llvm::SmallVector<int64_t, 6> dims(plan.outerShape.begin(), plan.outerShape.end());
  dims.append(plan.rank, fir::SequenceType::getUnknownExtent());
  llvm::SmallVector<mlir::Value, 6> runtime(dynamicOuter.begin(), dynamicOuter.end());
  runtime.append(caps.begin(), caps.end());
  llvm::SmallVector<mlir::Value, 6> extents(outerExtents.begin(), outerExtents.end());
  extents.append(caps.begin(), caps.end());
  mlir::Value const flat = allocate(flatName, dims, plan.eleTy, runtime, extents);

  // The element's indices, and its live extent ``d`` (0-based).
  auto outerIndices = [&](hlfir::DesignateOp parent) {
    llvm::SmallVector<mlir::Value, 4> out;
    for (mlir::Value const v : parent.getIndices())
      out.push_back(v.getType() == idxTy ? v : b.create<fir::ConvertOp>(loc, idxTy, v).getResult());
    return out;
  };
  auto lenRef = [&](hlfir::DesignateOp parent, unsigned d) {
    llvm::SmallVector<mlir::Value, 6> idx = outerIndices(parent);
    if (plan.rank > 1) idx.push_back(cIndex(d + 1));
    return b.create<hlfir::DesignateOp>(loc, fir::ReferenceType::get(i64), lenTable, idx).getResult();
  };
  auto liveExtent = [&](hlfir::DesignateOp parent, unsigned d) {
    mlir::Value const entry = b.create<fir::ConvertOp>(loc, idxTy, b.create<fir::LoadOp>(loc, lenRef(parent, d)));
    return b.create<mlir::arith::SubIOp>(loc, entry, cIndex(1)).getResult();
  };
  auto liveSection = [&](hlfir::DesignateOp parent) {
    llvm::SmallVector<std::variant<mlir::Value, std::tuple<mlir::Value, mlir::Value, mlir::Value>>, 6> subs;
    for (mlir::Value const v : outerIndices(parent)) subs.push_back(v);
    llvm::SmallVector<mlir::Value, 4> live;
    for (unsigned d = 0; d < plan.rank; ++d) {
      live.push_back(liveExtent(parent, d));
      subs.push_back(std::tuple<mlir::Value, mlir::Value, mlir::Value>{cIndex(1), live.back(), cIndex(1)});
    }
    llvm::SmallVector<int64_t, 4> unknown(plan.rank, fir::SequenceType::getUnknownExtent());
    auto sectionTy = fir::BoxType::get(fir::SequenceType::get(unknown, plan.eleTy));
    auto shape = b.create<fir::ShapeOp>(loc, live);
    return b
        .create<hlfir::DesignateOp>(loc, sectionTy, flat, "", mlir::Value{}, subs, mlir::ValueRange{}, std::nullopt,
                                    shape.getResult())
        .getResult();
  };

  llvm::SmallVector<mlir::Operation*, 16> dead;
  for (mlir::Operation* free : plan.frees) dead.push_back(free);
  for (Site s : plan.sites) {
    b.setInsertionPoint(s.store);
    for (unsigned d = 0; d < plan.rank; ++d) {
      mlir::Value const extent = b.create<fir::ConvertOp>(loc, i64, s.alloc.getShape()[d]);
      mlir::Value const entry =
          b.create<mlir::arith::AddIOp>(loc, extent, b.create<mlir::arith::ConstantIntOp>(loc, i64, 1).getResult());
      b.create<hlfir::AssignOp>(loc, entry, lenRef(s.parent, d));
    }
    dead.push_back(s.store);
  }
  for (fir::StoreOp release : plan.releases) {
    b.setInsertionPoint(release);
    auto parent = mlir::cast<hlfir::DesignateOp>(
        mlir::cast<hlfir::DesignateOp>(release.getMemref().getDefiningOp()).getMemref().getDefiningOp());
    b.create<hlfir::AssignOp>(loc, b.create<mlir::arith::ConstantIntOp>(loc, i64, 0).getResult(), lenRef(parent, 0));
    dead.push_back(release);
  }
  for (mlir::Operation* op : dead) op->erase();

  for (hlfir::DesignateOp dg : plan.designates) {
    auto parent = mlir::cast<hlfir::DesignateOp>(dg.getMemref().getDefiningOp());
    for (mlir::Operation* u : llvm::make_early_inc_range(dg->getUsers())) {
      b.setInsertionPoint(u);
      if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u)) {
        b.create<hlfir::AssignOp>(as.getLoc(), as.getRhs(), liveSection(parent));
        as.erase();
        continue;
      }
      auto load = mlir::cast<fir::LoadOp>(u);
      for (mlir::Operation* lu : llvm::make_early_inc_range(load->getUsers())) {
        b.setInsertionPoint(lu);
        if (auto inner = mlir::dyn_cast<hlfir::DesignateOp>(lu); inner && !inner.getComponentAttr()) {
          llvm::SmallVector<mlir::Value, 8> idx = outerIndices(parent);
          llvm::SmallVector<bool, 8> triplet(idx.size(), false);
          idx.append(inner.getIndices().begin(), inner.getIndices().end());
          llvm::ArrayRef<bool> const innerTriplet = inner.getIsTriplet();
          triplet.append(innerTriplet.begin(), innerTriplet.end());
          auto merged = b.create<hlfir::DesignateOp>(inner.getLoc(), inner.getResult().getType(), flat,
                                                     mlir::StringAttr{}, mlir::Value{}, idx, triplet,
                                                     inner.getSubstring(), inner.getComplexPartAttr(), inner.getShape(),
                                                     inner.getTypeparams(), inner.getFortranAttrsAttr());
          inner.getResult().replaceAllUsesWith(merged.getResult());
          inner.erase();
        } else if (auto dims = mlir::dyn_cast<fir::BoxDimsOp>(lu)) {
          auto d = traceConstInt(dims.getDim());
          dims.getResult(1).replaceAllUsesWith(liveExtent(parent, static_cast<unsigned>(d.value_or(0))));
          dims.getResult(0).replaceAllUsesWith(cIndex(1));
          dims.getResult(2).replaceAllUsesWith(cIndex(1));
          dims.erase();
        } else if (auto addr = mlir::dyn_cast<fir::BoxAddrOp>(lu)) {
          for (mlir::Operation* cv : llvm::make_early_inc_range(addr->getUsers())) {
            for (mlir::Operation* cu : llvm::make_early_inc_range(cv->getUsers())) {
              auto cmp = mlir::cast<mlir::arith::CmpIOp>(cu);
              b.setInsertionPoint(cmp);
              auto pred = cmp.getPredicate() == mlir::arith::CmpIPredicate::ne ? mlir::arith::CmpIPredicate::sgt
                                                                               : mlir::arith::CmpIPredicate::sle;
              mlir::Value const len = b.create<fir::LoadOp>(loc, lenRef(parent, 0));
              auto allocated = b.create<mlir::arith::CmpIOp>(cmp.getLoc(), pred, len,
                                                             b.create<mlir::arith::ConstantIntOp>(loc, i64, 0));
              cmp.getResult().replaceAllUsesWith(allocated.getResult());
              cmp.erase();
            }
            cv->erase();
          }
          addr.erase();
        } else {
          lu->replaceUsesOfWith(load.getResult(), liveSection(parent));
        }
      }
      load.erase();
    }
    dg.erase();
    if (parent->use_empty()) parent.erase();
  }
  for (Site s : plan.sites) {
    for (mlir::Operation* user : llvm::make_early_inc_range(s.alloc->getUsers()))
      if (user->use_empty()) user->erase();
    if (s.alloc->use_empty()) s.alloc.erase();
  }
}

}  // namespace hlfir_bridge
