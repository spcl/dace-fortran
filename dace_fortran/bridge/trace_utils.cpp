// ============================================================================
// trace_utils.cpp  --  Shared SSA tracing utilities
// ============================================================================

#include "bridge/trace_utils.h"

#include <algorithm>
#include <unordered_map>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace hlfir_bridge {

// Disambiguation overrides for Fortran short-name collisions across
// inlined scopes.  When ``hlfir-inline-all`` splices a callee's body
// into the caller, both the caller's argument declare
// (``_QFmainEinp``) and the callee's dummy declare
// (``_QFinner_loopsEinp``) end up in one function with the same
// trailing short name (``inp``).  Without disambiguation,
// ``builder.arrays`` keys collide and view-alias linking edges
// self-loop.  ``extract_vars`` populates this map with
// ``mangled -> unique_short_name`` for the colliding entries; every
// subsequent ``extractName`` call resolves to the unique form.
static thread_local std::unordered_map<std::string, std::string>
    kManglingOverride;

void setManglingOverride(const std::string &mangled,
                         const std::string &shortName) {
  kManglingOverride[mangled] = shortName;
}

void clearManglingOverrides() { kManglingOverride.clear(); }

std::string extractName(const std::string &m) {
  auto it = kManglingOverride.find(m);
  if (it != kManglingOverride.end()) return it->second;
  auto p = m.rfind('E');
  std::string name = p != std::string::npos ? m.substr(p + 1) : m;
  // Sanitize dots  --  flang emits compiler-generated globals like
  // ``_QQro.4xi4.0`` (read-only constant pool for array literals)
  // whose names contain ``.``.  DaCe's ``NestedDict`` reserves
  // ``.`` as a nested-key separator and rejects dotted keys
  // outright.  Fortran identifiers can't contain ``.``, so
  // replacing every ``.`` with ``_`` is collision-free w.r.t.
  // user names.  Done at the boundary (extractName is the
  // canonical "MLIR mangled -> Python-side name" helper) so the
  // raw mangled names in the IR stay intact.
  std::replace(name.begin(), name.end(), '.', '_');
  return name;
}

// Allocatable re-allocation alias map.  Keyed by the raw Fortran name
// (what the declare chain alone would resolve to).  Updated as the
// bridge's IR walker passes ``fir.allocmem``-bound ``fir.store`` ops
// (see extract_ast.cpp); read by ``traceToDecl`` so every downstream
// access lands on the currently-live SDFG transient.
static thread_local std::unordered_map<std::string, std::string> kAllocAlias;

std::string allocAliasFor(const std::string &raw) {
  auto it = kAllocAlias.find(raw);
  return it == kAllocAlias.end() ? raw : it->second;
}

void setAllocAlias(const std::string &raw, const std::string &alias) {
  if (alias == raw)
    kAllocAlias.erase(raw);
  else
    kAllocAlias[raw] = alias;
}

void clearAllocAliases() { kAllocAlias.clear(); }

std::string traceToDecl(mlir::Value val, int max) {
  for (int i = 0; i < max && val; ++i) {
    auto *d = val.getDefiningOp();
    if (!d) break;
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      // Walk through inlined assumed-shape aliases to the outer
      // caller declare so downstream SDFG emission references
      // the real storage by its caller-side name.
      if (auto outer = asAssumedShapeAlias(dc)) {
        val = outer.getResult(0);
        continue;
      }
      return allocAliasFor(extractName(dc.getUniqName().str()));
    }
    if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d))
      return allocAliasFor(extractName(dc.getUniqName().str()));
    if (auto c = mlir::dyn_cast<fir::ConvertOp>(d)) {
      val = c.getValue();
      continue;
    }
    if (auto l = mlir::dyn_cast<fir::LoadOp>(d)) {
      val = l.getMemref();
      continue;
    }
    if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d)) {
      val = co.getRef();
      continue;
    }
    // ``fir.rebox`` retypes an existing box (e.g. section view box
    // -> ``box<ptr<...>>`` for a Fortran ``ptr => slice`` rebind);
    // it doesn't change the underlying storage.  Walk through so a
    // downstream designate over the reboxed value still resolves
    // to the parent's name.  Same role as the
    // ``hlfir-rewrite-pointer-assigns`` slice-target forwarding:
    // pointer reads land back on the parent array.
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      val = rb.getBox();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      val = eb.getMemref();
      continue;
    }
    // ``fir.box_addr`` extracts the data pointer from a descriptor
    // (heap / ptr underlying the box).  Flang emits it for every
    // allocatable / pointer dereference; the underlying storage
    // name is the box's source declare, so we walk through.
    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
      val = ba.getVal();
      continue;
    }
    // Section / element designates (``a(lo:hi)``, ``a(i)``)  --  walk
    // through to the underlying memref so a reduce over an
    // ``hlfir.any %levmask(i_startblk:i_endblk, jk)`` resolves its
    // source array to ``levmask``.
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      val = dg.getMemref();
      continue;
    }
    if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      val = s.getTrueValue();
      continue;
    }
    break;
  }
  return "";
}

std::optional<int64_t> traceConstInt(mlir::Value v) {
  for (int i = 0; i < limits::kTraceConstIntMax; ++i) {
    auto *d = v.getDefiningOp();
    if (!d) break;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(d))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
        return ia.getInt();
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    // Flang wraps each static extent in a `select extent>0, extent, 0`
    // clamp; follow the true branch to reach the original value.
    // Restrict to that exact shape -- the false value must be the
    // constant 0 -- so we don't accidentally follow Fortran ``MAX``
    // / ``MIN`` (also lowered as ``arith.select`` over a cmp) and
    // collapse a non-constant bound to its first operand.
    if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      auto *fdef = s.getFalseValue().getDefiningOp();
      bool false_is_zero = false;
      if (fdef) {
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(fdef))
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
            false_is_zero = (ia.getInt() == 0);
      }
      if (false_is_zero) {
        v = s.getTrueValue();
        continue;
      }
    }
    break;
  }
  return std::nullopt;
}

std::string posSymbolName(const std::string &array, int64_t one_based_idx) {
  // Keep in lockstep with ``internPosSymbol`` (ast/expressions.cpp): the
  // descriptor-shape side mints the name here, the AST builder mints the
  // matching ``symbol_init`` there, and they must agree.
  return "__sym_" + array + "_" + std::to_string(one_based_idx);
}

std::optional<std::pair<std::string, int64_t>>
constIndexedElementLoad(mlir::Value v) {
  if (!v) return std::nullopt;
  for (int i = 0; i < limits::kConvertChainDepth && v; ++i) {
    auto *d = v.getDefiningOp();
    if (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    break;
  }
  auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(v.getDefiningOp());
  if (!ld) return std::nullopt;
  auto dg =
      mlir::dyn_cast_or_null<hlfir::DesignateOp>(ld.getMemref().getDefiningOp());
  if (!dg) return std::nullopt;
  auto idxs = dg.getIndices();
  if (idxs.size() != 1) return std::nullopt;  // single 1-D element only
  auto triplets = dg.getIsTriplet();
  if (!triplets.empty() && triplets[0]) return std::nullopt;  // not a section
  auto c = traceConstInt(idxs[0]);
  if (!c) return std::nullopt;
  auto arr = traceToDecl(dg.getMemref());
  if (arr.empty()) return std::nullopt;
  return std::make_pair(arr, *c);
}

static void forEachConstIndexedElementImpl(
    mlir::Value v, const std::function<void(const std::string &, int64_t)> &fn,
    int depth) {
  if (depth > limits::kTraceToDeclMax || !v) return;
  if (auto e = constIndexedElementLoad(v)) {
    fn(e->first, e->second);
    return;
  }
  auto *def = v.getDefiningOp();
  if (!def) return;
  // Recurse through the same wrapper / arithmetic / max-min / select op
  // set ``traceExtentExpr`` renders, so every element leaf is reached.
  if (mlir::isa<fir::ConvertOp, mlir::arith::SelectOp, mlir::arith::CmpIOp,
                mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
                mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                mlir::arith::MaxSIOp, mlir::arith::MaxUIOp,
                mlir::arith::MinSIOp, mlir::arith::MinUIOp>(def)) {
    for (auto op : def->getOperands())
      forEachConstIndexedElementImpl(op, fn, depth + 1);
  }
}

void forEachConstIndexedElement(
    mlir::Value v, const std::function<void(const std::string &, int64_t)> &fn) {
  forEachConstIndexedElementImpl(v, fn, 0);
}

std::string traceExtentExpr(mlir::Value v) {
  if (!v) return "";
  auto *def = v.getDefiningOp();
  if (!def) return "";

  // Transparent peels.
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
    return traceExtentExpr(cv.getValue());

  // Constant integer literal.
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return std::to_string(ia.getInt());

  // Load of a Fortran scalar -- render as its short name.  A load of a
  // constant-indexed array element (``dims(1)``) becomes its position
  // symbol so the shape stays symbolic; promoting the whole array would
  // collide it with its own data descriptor.
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    if (auto e = constIndexedElementLoad(v))
      return posSymbolName(e->first, e->second);
    auto mem = ld.getMemref();
    auto *md = mem.getDefiningOp();
    if (!md) return "";
    if (mlir::isa<hlfir::DeclareOp>(md) || mlir::isa<fir::DeclareOp>(md)) {
      return traceToDecl(mem);
    }
    return "";
  }

  // ``arith.select`` over an ``arith.cmpi`` is BOTH Flang's
  // non-negativity clamp on an extent AND a genuine Fortran
  // ``MAX``/``MIN`` -- they must be told apart:
  //
  //   * Clamp ``max(ext, 0)``: ``select(ext sgt 0, ext, 0)`` -- the
  //     false arm is the constant ``0``.  Array extents are
  //     non-negative by construction, so this is dead defensive code;
  //     drop the wrap and return the underlying extent (keeps shapes
  //     readable -- ``klon`` not ``max(klon, 0)`` -- and lets sympy
  //     fold).
  //   * Genuine ``MAX(a, b)`` / ``MIN(a, b)``:
  //     ``select(a sgt b, a, b)`` / ``select(a slt b, a, b)`` -- the
  //     two arms are the operands.  Render ``max(a, b)`` / ``min(a, b)``
  //     so a real two-operand bound (``allocate(x(max(n, 1)))``)
  //     survives instead of being dropped.
  //
  // The cmp operands must match the select arms; otherwise it is some
  // other conditional we don't model.
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    auto *cdef = sel.getCondition().getDefiningOp();
    auto cmp = cdef ? mlir::dyn_cast<mlir::arith::CmpIOp>(cdef) : nullptr;
    if (!cmp || cmp.getLhs() != sel.getTrueValue() ||
        cmp.getRhs() != sel.getFalseValue())
      return "";
    using P = mlir::arith::CmpIPredicate;
    auto pred = cmp.getPredicate();
    bool falseIsZero = false;
    if (auto *fdef = sel.getFalseValue().getDefiningOp())
      if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(fdef))
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
          falseIsZero = (ia.getInt() == 0);
    if (falseIsZero && (pred == P::sgt || pred == P::sge))
      return traceExtentExpr(sel.getTrueValue());  // non-negativity clamp
    auto a = traceExtentExpr(sel.getTrueValue());
    auto b = traceExtentExpr(sel.getFalseValue());
    if (a.empty() || b.empty()) return "";
    if (pred == P::sgt || pred == P::sge || pred == P::ugt || pred == P::uge)
      return "max(" + a + ", " + b + ")";
    if (pred == P::slt || pred == P::sle || pred == P::ult || pred == P::ule)
      return "min(" + a + ", " + b + ")";
    return "";
  }

  // Binary integer arithmetic.  Render parenthesised so the result
  // composes cleanly when nested.  ``arith.max*i`` / ``arith.min*i`` are
  // the direct MAX/MIN lowering (vs the select-over-cmp form above).
  auto nm = def->getName().getStringRef();
  static const std::map<llvm::StringRef, std::string> bin = {
      {"arith.addi", " + "},   {"arith.subi", " - "},   {"arith.muli", " * "},
      {"arith.divsi", " // "}, {"arith.divui", " // "},
  };
  if (auto it = bin.find(nm); it != bin.end() && def->getNumOperands() == 2) {
    auto l = traceExtentExpr(def->getOperand(0));
    auto r = traceExtentExpr(def->getOperand(1));
    if (l.empty() || r.empty()) return "";
    return "(" + l + it->second + r + ")";
  }
  if ((nm == "arith.maxsi" || nm == "arith.maxui" || nm == "arith.minsi" ||
       nm == "arith.minui") &&
      def->getNumOperands() == 2) {
    auto l = traceExtentExpr(def->getOperand(0));
    auto r = traceExtentExpr(def->getOperand(1));
    if (l.empty() || r.empty()) return "";
    const char *fn = (nm == "arith.maxsi" || nm == "arith.maxui") ? "max" : "min";
    return std::string(fn) + "(" + l + ", " + r + ")";
  }

  return "";
}

void collectExtentExprScalars(mlir::Value v, std::set<std::string> &out) {
  if (!v) return;
  auto *def = v.getDefiningOp();
  if (!def) return;
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
    collectExtentExprScalars(cv.getValue(), out);
    return;
  }
  if (mlir::isa<mlir::arith::ConstantOp>(def)) return;
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    auto mem = ld.getMemref();
    if (auto *md = mem.getDefiningOp())
      if (mlir::isa<hlfir::DeclareOp>(md) || mlir::isa<fir::DeclareOp>(md)) {
        auto n = traceToDecl(mem);
        if (!n.empty()) out.insert(n);
      }
    return;
  }
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    // Walk both branches; cmp condition leaves are already
    // covered by the operands themselves.
    collectExtentExprScalars(sel.getTrueValue(), out);
    collectExtentExprScalars(sel.getFalseValue(), out);
    return;
  }
  auto nm = def->getName().getStringRef();
  if ((nm == "arith.addi" || nm == "arith.subi" || nm == "arith.muli" ||
       nm == "arith.divsi" || nm == "arith.divui" || nm == "arith.cmpi") &&
      def->getNumOperands() == 2) {
    collectExtentExprScalars(def->getOperand(0), out);
    collectExtentExprScalars(def->getOperand(1), out);
    return;
  }
}

hlfir::DeclareOp asAssumedShapeAlias(hlfir::DeclareOp decl) {
  // Signature: memref produced by another ``hlfir.declare`` (possibly
  // behind ``fir.convert`` rebox ops).  This is precisely what Flang
  // emits for the callee's dummy_scope declare after
  // ``hlfir-inline-all`` splices the callee's body into the caller  --
  // the callee declare aliases the caller's outer declare for
  // both assumed-shape (no shape operand on the inner declare) and
  // fixed-shape (the inner declare carries its own copy of the
  // callee-side shape) callees, the only difference being whether
  // the inner declare reissues a shape.  Either way the storage is
  // shared and downstream tracing should walk to the outer declare.
  auto mr = decl.getMemref();
  for (int i = 0; i < limits::kAliasMemrefWalkDepth && mr; ++i) {
    auto *d = mr.getDefiningOp();
    if (!d) break;
    if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) return outer;
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      mr = conv.getValue();
      continue;
    }
    // Internal-subprogram inlining wraps the outer fixed-shape array
    // in a ``fir.embox`` so the inlined assumed-shape callee sees a
    // ``fir.box``.  Peel through to the underlying declare.
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      mr = eb.getMemref();
      continue;
    }
    // Inlined-callee aliases on CLASS-allocatable / box-typed
    // dummies: the alias's memref comes from a ``fir.load`` of the
    // caller's box-slot declare, possibly preceded by ``fir.rebox``
    // peels (CLASS<heap<T>> -> CLASS<T>).  Walking through both is
    // what catches monomorphic CLASS subroutine dummies as
    // aliases of their caller-side allocatable.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      mr = ld.getMemref();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      mr = rb.getBox();
      continue;
    }
    break;
  }
  return {};
}

ShapeOperandInfo classifyShapeOperand(mlir::Value shape) {
  ShapeOperandInfo si;
  if (!shape) return si;
  auto *def = shape.getDefiningOp();
  if (auto sh = mlir::dyn_cast_or_null<fir::ShapeOp>(def)) {
    si.kind = ShapeOperandInfo::Shape;
    for (auto e : sh.getExtents()) si.extents.push_back(e);
    si.rank = si.extents.size();
  } else if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(def)) {
    si.kind = ShapeOperandInfo::ShapeShift;
    auto ops = ss->getOperands();
    for (unsigned i = 0; i + 1 < ops.size(); i += 2) {
      si.lbs.push_back(ops[i]);
      si.extents.push_back(ops[i + 1]);
    }
    si.rank = si.lbs.size();
  } else if (auto sf = mlir::dyn_cast_or_null<fir::ShiftOp>(def)) {
    si.kind = ShapeOperandInfo::Shift;
    for (auto lb : sf->getOperands()) si.lbs.push_back(lb);
    si.rank = si.lbs.size();
  }
  return si;
}

std::vector<std::optional<int64_t>> declareLowerBounds(hlfir::DeclareOp decl) {
  std::vector<std::optional<int64_t>> lbs;
  auto si = classifyShapeOperand(decl.getShape());
  if (si.kind == ShapeOperandInfo::Shape) {
    // Plain fir.shape: every dim defaults to lbound=1.
    lbs.assign(si.rank, std::optional<int64_t>(1));
  } else if (si.kind == ShapeOperandInfo::ShapeShift) {
    for (auto lb : si.lbs) lbs.push_back(traceConstInt(lb));
  }
  // fir.shift / no shape: leave empty (caller default), as before.
  return lbs;
}

llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape) {
  auto ext = classifyShapeOperand(shape).extents;
  return {ext.begin(), ext.end()};
}

fir::RecordType pointerToRecordMember(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return {};
  mlir::Type inner;
  if (auto h = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
    inner = h.getEleTy();
  else if (auto p = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
    inner = p.getEleTy();
  else
    return {};
  return mlir::dyn_cast<fir::RecordType>(inner);
}

fir::RecordType allocOrPtrArrayOfRecordsMember(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return {};
  mlir::Type inner;
  if (auto h = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
    inner = h.getEleTy();
  else if (auto p = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
    inner = p.getEleTy();
  else
    return {};
  auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
  if (!seq) return {};
  return mlir::dyn_cast<fir::RecordType>(seq.getEleTy());
}

}  // namespace hlfir_bridge
