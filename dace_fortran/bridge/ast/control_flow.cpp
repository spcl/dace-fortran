// Translation-unit headers; ast_helpers.h carries the cross-TU API + thread-local state shared with the other ast/*.cpp
// files.
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "bridge/ast/ast_helpers.h"
#include "bridge/ast/ast_internal.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace hlfir_bridge {

//
// MERGE-libcall + buildElementalAssign + comparison primitives + scf.if helpers. Included verbatim from extract_ast.cpp
// (#include "bridge/ast/control_flow.cpp"), sharing that TU's namespace/includes/file-statics; MUST NOT be added to the
// build's compile list -- CMakeLists.txt deliberately omits it.
std::vector<ASTNode> buildMergeLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
  auto& region = elem.getRegion();
  if (region.empty()) return {};
  auto& block = region.front();

  // Find the yield_element and confirm its operand is an arith.select.
  mlir::Value yielded;
  for (auto& op : block)
    if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
      yielded = y.getElementValue();
      break;
    }
  if (!yielded) return {};
  auto sel = mlir::dyn_cast_or_null<mlir::arith::SelectOp>(yielded.getDefiningOp());
  if (!sel) return {};

  // Each operand must trace to a fir.load of an hlfir.designate/hlfir.declare (fir.convert wrappers transparent);
  // anything else bails to buildElementalAssign's generic per-element tasklet path. Resolves to a declared array/scalar
  // name; broadcast is decided later from the memlet subset.
  auto traceLoadSource = [](mlir::Value v) -> std::string {
    // Walk through any fir.convert wrappers at the top.
    for (int i = 0; i < 8; ++i) {
      auto* op = v.getDefiningOp();
      if (!op) return "";
      auto cv = mlir::dyn_cast<fir::ConvertOp>(op);
      if (!cv) break;
      v = cv.getValue();
    }
    auto* op = v.getDefiningOp();
    if (!op) return "";
    auto ld = mlir::dyn_cast<fir::LoadOp>(op);
    if (!ld) return "";
    auto* md = ld.getMemref().getDefiningOp();
    if (!md) return "";
    if (mlir::isa<hlfir::DesignateOp>(md) || mlir::isa<hlfir::DeclareOp>(md)) return traceToDecl(ld.getMemref());
    return "";
  };

  std::string const mask_name = traceLoadSource(sel.getCondition());
  std::string const t_name = traceLoadSource(sel.getTrueValue());
  std::string const f_name = traceLoadSource(sel.getFalseValue());
  if (mask_name.empty() || t_name.empty() || f_name.empty()) return {};

  ASTNode lib;
  lib.kind = "libcall";
  lib.callee = "merge";
  auto dest = assign.getOperand(1);
  if (auto dd = dest.getDefiningOp())
    if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(dd)) lib.target = extractName(declOp.getUniqName().str());
  if (lib.target.empty()) lib.target = traceToDecl(dest);
  lib.target_is_array = isArrayRef(dest.getType());
  // MergeLibraryNode connector order: ``_t``, ``_f``, ``_mask``.
  lib.call_args.push_back(t_name);
  lib.call_args.push_back(f_name);
  lib.call_args.push_back(mask_name);
  return {std::move(lib)};
}

/// b = elementwise_expr(a) where the assign's source is an elemental: synthesises one kind="loop" ASTNode per shape dim
/// (synthetic iters ei0, ei1, ...) wrapping a kind="assign" child; buildExpr resolves block args to these iters via
/// indexStack().
std::vector<ASTNode> buildElementalAssign(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
  // Target array (LHS of the assign).
  ASTNode inner;
  inner.kind = "assign";
  auto dest = assign.getOperand(1);
  if (auto dd = dest.getDefiningOp())
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
      inner.target = allocAliasFor(extractName(decl.getUniqName().str()));
  if (inner.target.empty()) inner.target = traceToDecl(dest);
  inner.target_is_array = true;

  // Synthetic iter names for each shape dimension.
  auto shape = elem.getShape();
  auto& region = elem.getRegion();
  if (region.empty()) return {};
  auto& block = region.front();
  unsigned const rank = block.getNumArguments();

  std::vector<std::string> iter_names;
  iter_names.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) iter_names.push_back("ei" + std::to_string(i));

  // Push block-arg -> synthetic-name pairs so resolveIndex sees them everywhere we walk the body.
  unsigned pushed = 0;
  for (unsigned i = 0; i < rank; ++i) {
    indexStack().emplace_back(block.getArgument(i), iter_names[i]);
    ++pushed;
  }

  // Write index = target[per-array-dim index]; section designates add (lo-1) per triplet-iter, scalar dims carry their
  // own Fortran-1-based expr (elemental rank = triplet count, not the underlying array's rank). LowerAdj: expr
  // non-empty -> (iter+expr-1); value!=0 -> integer offset; else bare iter.
  struct LowerAdj {
    int64_t value = 0;
    std::string expr;
  };
  AccessInfo wa;
  wa.array_name = inner.target;
  wa.is_write = true;
  if (auto dd = dest.getDefiningOp()) {
    // Whole-member array-component LHS (`arr(jc,jk) % x`): %x has a component selector + whole-member shape but no
    // triplet subscript, so the generic triplet loop below would emit a rank-0 write memlet -> pointer-vs-scalar
    // compile error. Reuse expandDesignateChain for the leading record dims, then append elemental iters for the
    // trailing member-array dims.
    auto dstDgArrComp = mlir::dyn_cast<hlfir::DesignateOp>(dd);
    if (dstDgArrComp && dstDgArrComp.getComponentAttr() && dstDgArrComp.getIsTriplet().empty() &&
        dstDgArrComp.getIndices().empty() && isArrayRef(dstDgArrComp.getResult().getType())) {
      auto [arr, recDims] = expandDesignateChain(dstDgArrComp);
      if (!arr.empty()) {
        inner.target = arr;
        wa.array_name = arr;
      }
      for (auto& de : recDims) {
        wa.index_vars.push_back(de.var);
        wa.index_exprs.push_back(de.expr);
      }
      for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
      }
    } else if (auto dstDg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
      auto triplets = dstDg.getIsTriplet();
      auto idxOps = dstDg.getIndices();
      unsigned cursor = 0;
      unsigned tDim = 0;
      for (bool const isT : triplets) {
        if (isT && tDim < rank && cursor + 3 <= idxOps.size()) {
          LowerAdj adj;
          if (auto lo = traceConstInt(idxOps[cursor])) {
            adj.value = *lo - 1;
          } else {
            auto loExpr = buildIndexExpr(idxOps[cursor], 0);
            if (!loExpr.empty() && loExpr != "?") adj.expr = std::move(loExpr);
          }
          std::string ix = iter_names[tDim];
          if (!adj.expr.empty()) {
            std::string adjusted = "(";
            adjusted += ix;
            adjusted += " + ";
            adjusted += adj.expr;
            adjusted += " - 1)";
            ix = std::move(adjusted);
          } else if (adj.value > 0) {
            std::string adjusted = "(";
            adjusted += ix;
            adjusted += " + ";
            adjusted += std::to_string(adj.value);
            adjusted += ")";
            ix = std::move(adjusted);
          } else if (adj.value < 0) {
            std::string adjusted = "(";
            adjusted += ix;
            adjusted += " - ";
            adjusted += std::to_string(-adj.value);
            adjusted += ")";
            ix = std::move(adjusted);
          }
          wa.index_vars.push_back(iter_names[tDim]);
          wa.index_exprs.push_back(std::move(ix));
          cursor += 3;
          tDim++;
        } else if (!isT && cursor < idxOps.size()) {
          // Scalar dim -- thread its Fortran-1-based index expr directly into the write memlet so the memlet rank
          // matches the underlying array.
          auto sc = buildIndexExpr(idxOps[cursor], 0);
          if (sc.empty() || sc == "?") sc = "?";
          wa.index_vars.push_back(sc);
          wa.index_exprs.push_back(std::move(sc));
          cursor += 1;
        } else {
          // Defensive: skip bad cursor advance to avoid an infinite loop on malformed input.
          cursor += isT ? 3 : 1;
          if (isT) tDim++;
        }
      }
    } else {
      // Bare hlfir.declare -- write across the elemental's full rank (every dim is a triplet covering the array's
      // extent).
      for (unsigned i = 0; i < rank; ++i) {
        wa.index_vars.push_back(iter_names[i]);
        wa.index_exprs.push_back(iter_names[i]);
      }
    }
  } else {
    for (unsigned i = 0; i < rank; ++i) {
      wa.index_vars.push_back(iter_names[i]);
      wa.index_exprs.push_back(iter_names[i]);
    }
  }
  inner.accesses.push_back(std::move(wa));

  // Walk the body's yield_element to produce the RHS string.
  mlir::Value yielded;
  for (auto& op : block)
    if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
      yielded = y.getElementValue();
      break;
    }

  // Pre-walk: any hlfir.apply <libcall_expr> in the elemental body must be materialised into a transient BEFORE the
  // elemental runs, else buildExpr returns "?" (e.g. tasklet body "2 - ?"). Each libcall op gets a unique _libtmp_<gid>
  // name + declare_transient/libcall AST node pair; buildExpr then reads the transient like a normal Fortran name.
  std::vector<ASTNode> preNodes;
  if (yielded) {
    std::function<void(mlir::Value, int)> findApplies = [&](mlir::Value v, int depth) {
      if (depth > 40 || !v) return;
      auto* op = v.getDefiningOp();
      if (!op) return;
      if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
        auto src = apply.getExpr();
        if (auto* srcOp = src.getDefiningOp()) {
          // Inner elemental: walk its yielded value directly (not findApplies(src,...), which would only recurse on the
          // elemental's shape operands and miss the body's libcall expr-producers) -- QE's vcut_get (nested
          // MATMUL(TRANSPOSE())/tpi elementals) is the surfacing case.
          if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(srcOp)) {
            auto& iregion = inner_elem.getRegion();
            if (!iregion.empty()) {
              for (auto& iop : iregion.front()) {
                if (auto iy = mlir::dyn_cast<hlfir::YieldElementOp>(iop)) {
                  findApplies(iy.getElementValue(), depth + 1);
                  break;
                }
              }
            }
            return;
          }
          // Recognised libcall expr-producer -> materialise.
          if (const char* callee = libcallNameForExprOp(srcOp)) {
            if (!kHlfirExprToTransient.count(srcOp)) {
              std::string const tmp = "_libtmp_" + std::to_string(kLibTmpCounter++);
              kHlfirExprToTransient[srcOp] = tmp;

              mlir::Type const rty = srcOp->getResult(0).getType();
              auto shape = exprResultShape(rty);

              ASTNode decl;
              decl.kind = "declare_transient";
              decl.target = tmp;
              decl.expr = exprDtypeString(rty);
              decl.target_is_array = !shape.empty();
              AccessInfo shapeInfo;
              shapeInfo.array_name = tmp;
              for (auto& s : shape) shapeInfo.index_exprs.push_back(s);
              decl.accesses.push_back(std::move(shapeInfo));
              preNodes.push_back(std::move(decl));

              ASTNode lib;
              lib.kind = "libcall";
              lib.target = tmp;
              lib.target_is_array = !shape.empty();
              lib.callee = callee;
              // CSHIFT as an expr-producer (`2.0 - CSHIFT(arr,1)`): the shift is a scalar for
              // options["shift"]/reduce_axes, NOT a call_args operand (mirrors buildLibCallNode's whole-array form) --
              // else emit_library builds shift=None and the __shift symbol leaks unassigned (KeyError: '__shift').
              if (auto cshOp = mlir::dyn_cast<hlfir::CShiftOp>(srcOp)) {
                auto arrName = traceToDecl(cshOp.getArray());
                if (arrName.empty())
                  if (auto* ad = cshOp.getArray().getDefiningOp())
                    if (auto ae = mlir::dyn_cast<hlfir::ElementalOp>(ad)) {
                      auto [trName, mat] = materialiseElementalForLibcall(ae);
                      if (!trName.empty()) {
                        for (auto& mn : mat) preNodes.push_back(std::move(mn));
                        arrName = std::move(trName);
                      }
                    }
                lib.call_args.push_back(arrName);
                auto shiftVal = cshOp.getShift();
                if (auto c = traceConstInt(shiftVal))
                  lib.options["shift"] = std::to_string(*c);
                else {
                  auto sExpr = buildIndexExpr(shiftVal, 0);
                  if (!sExpr.empty() && sExpr != "?") lib.options["shift"] = sExpr;
                }
                if (auto dim = cshOp.getDim())
                  if (auto c = traceConstInt(dim)) lib.reduce_axes.push_back(*c - 1);
                preNodes.push_back(std::move(lib));
                return;
              }
              for (auto operand : srcOp->getOperands()) {
                auto n = traceToDecl(operand);
                if (n.empty()) {
                  // Same fix shape as dispatch.cpp's libcall-over-elemental: an inline hlfir.elemental operand (e.g.
                  // transpose(<inner gather>)) has no backing declare, so traceToDecl returns ""; materialise it into a
                  // synthetic transient and pass that name instead.
                  if (auto* od = operand.getDefiningOp()) {
                    if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(od)) {
                      auto [trName, mat_nodes] = materialiseElementalForLibcall(inner_elem);
                      if (!trName.empty()) {
                        for (auto& mn : mat_nodes) preNodes.push_back(std::move(mn));
                        n = std::move(trName);
                      }
                    }
                    // Nested libcall expr-producer (e.g. MATMUL(TRANSPOSE(a), q): the transpose result has no backing
                    // declare either) -- recursively materialise via the same _libtmp_<gid> mechanism so the outer
                    // libcall's source arg is a real name. QE's vcut_get was the surfacing case.
                    else if (n.empty() && libcallNameForExprOp(od)) {
                      // Memoise per (op, transient name) so a transpose result shared between two consumers reuses the
                      // same transient.
                      auto it = kHlfirExprToTransient.find(od);
                      std::string innerTr;
                      if (it != kHlfirExprToTransient.end()) {
                        innerTr = it->second;
                      } else {
                        innerTr = "_libtmp_" + std::to_string(kLibTmpCounter++);
                        kHlfirExprToTransient[od] = innerTr;
                        mlir::Type const irty = od->getResult(0).getType();
                        auto ishape = exprResultShape(irty);

                        ASTNode idecl;
                        idecl.kind = "declare_transient";
                        idecl.target = innerTr;
                        idecl.expr = exprDtypeString(irty);
                        idecl.target_is_array = !ishape.empty();
                        AccessInfo ishapeInfo;
                        ishapeInfo.array_name = innerTr;
                        for (auto& s : ishape) ishapeInfo.index_exprs.push_back(s);
                        idecl.accesses.push_back(std::move(ishapeInfo));
                        preNodes.push_back(std::move(idecl));

                        ASTNode ilib;
                        ilib.kind = "libcall";
                        ilib.target = innerTr;
                        ilib.target_is_array = !ishape.empty();
                        ilib.callee = libcallNameForExprOp(od);
                        for (auto iop : od->getOperands()) {
                          auto in = traceToDecl(iop);
                          // Only one-level nesting handled here; deeper would need full recursion (uncommon shape).
                          ilib.call_args.push_back(in);
                        }
                        preNodes.push_back(std::move(ilib));
                      }
                      n = innerTr;
                    }
                  }
                }
                lib.call_args.push_back(n);
              }
              preNodes.push_back(std::move(lib));
            }
          }
        }
        return;
      }
      for (auto operand : op->getOperands()) findApplies(operand, depth + 1);
    };
    findApplies(yielded, 0);
  }

  // Walk in tasklet-body mode: embedded comparisons (e.g. inside a MERGE mask) must produce bare names matching
  // emit_tasklet's connector wiring, else the bool path emits a[ei0-1] while the bare-name path emits a for the same
  // array, leaking ei0 as a free symbol.
  {
    NoSubscriptGuard const g;
    inner.expr = yielded ? buildExpr(yielded, 0) : "?";
  }

  // Read accesses: unlike plain assigns, must follow hlfir.apply into the referenced elemental's body (where the
  // designate lives), pushing its index mapping onto indexStack() so it sees the outer elemental's synthetic iter
  // names.
  if (yielded) {
    // Per-occurrence AccessInfo (depth-limited, no op-identity dedup): emit_tasklet counts array-name regex occurrences
    // in the RHS string, so shared SSA values (`x * x`) must yield a matching AccessInfo count or downstream wiring
    // strands a connector.
    collectReadAccesses(yielded, inner.accesses, 0);
  }

  // Pop the stack frames we pushed.
  for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();

  // Wrap the inner assign in one ASTNode kind="loop" per rank; the outermost loop is the result, deeper loops live as
  // its sole child.
  ASTNode current;
  current.kind = "assign";
  current = inner;
  for (int i = rank - 1; i >= 0; --i) {
    ASTNode wrap;
    wrap.kind = "loop";
    wrap.loop_iter = iter_names[i];
    wrap.loop_lower = 1;
    wrap.loop_bound = resolveExtent(shape, i);
    wrap.children.push_back(current);
    current = wrap;
  }
  // Prepend any libcall-result materialisations so the transient is declared and populated before the elemental body
  // reads it.
  if (!preNodes.empty()) {
    preNodes.push_back(std::move(current));
    return preNodes;
  }
  return {current};
}

/// Render an arith::cmpi predicate as a Python comparison operator; empty string for signed/unsigned variants not wired
/// up yet.
std::string cmpiPredStr(mlir::arith::CmpIPredicate p) {
  using P = mlir::arith::CmpIPredicate;
  switch (p) {
    case P::slt:
    case P::ult:
      return "<";
    case P::sle:
    case P::ule:
      return "<=";
    case P::sgt:
    case P::ugt:
      return ">";
    case P::sge:
    case P::uge:
      return ">=";
    case P::eq:
      return "==";
    case P::ne:
      return "!=";
  }
  return "";
}

/// Render an arith::cmpf predicate as a Python comparison operator; ordered/unordered predicates collapse to the same
/// operator (NaN handling can't be expressed in a Python condition string, so the lossy mapping is accepted).
std::string cmpfPredStr(mlir::arith::CmpFPredicate p) {
  using P = mlir::arith::CmpFPredicate;
  switch (p) {
    case P::OLT:
    case P::ULT:
      return "<";
    case P::OLE:
    case P::ULE:
      return "<=";
    case P::OGT:
    case P::UGT:
      return ">";
    case P::OGE:
    case P::UGE:
      return ">=";
    case P::OEQ:
    case P::UEQ:
      return "==";
    case P::ONE:
    case P::UNE:
      return "!=";
    default:
      return "";
  }
}

/// Like buildExpr but keeps explicit subscripts (a[(i)-1]) for a fir.load of an hlfir.designate; used by buildBoolExpr
/// since interstate-edge conditions evaluate in the caller's frame and can't rely on memlet-wired connectors.
std::string buildExprWithSubscripts(mlir::Value val, int d) {
  if (d > limits::kBuildExprDepth || !val) return "?";
  // Output lands in an interstate-edge/ConditionalBlock condition parsed by DaCe's symbolic engine, which can't handle
  // the dace.float32(...) wrap (treats dace as a free symbol); suppress it for this walk since f32-vs-f64 doesn't
  // change comparison outcomes.
  SuppressFloatCastGuard const floatCastGuard;
  auto* def = val.getDefiningOp();
  if (!def) return "?";
  // A reduction op materialised into a scalar by materialiseCondReductions (Reduce lib-node before the branch) renders
  // as the bare scalar, bypassing the inline-unroll reduction table below.
  if (auto it = kCondReductionScalars.find(def); it != kCondReductionScalars.end()) return it->second;

  if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) return buildExprWithSubscripts(conv.getValue(), d + 1);
  // hlfir.no_reassoc is Flang's reassociation-blocker wrapper around order-preserved expressions (e.g. `(1.0 -
  // ZA(JL,JK))`); pure structural metadata, so peel through to keep the inner chain subscript-aware -- without this it
  // bottoms out to buildExpr and emits bare `za`, which C++ codegen renders as `int - double*`.
  auto _nm = def->getName().getStringRef();
  if (_nm == "hlfir.no_reassoc" && def->getNumOperands() == 1)
    return buildExprWithSubscripts(def->getOperand(0), d + 1);

  // hlfir.minval/maxval/sum/product over a constant-extent array section: unfold inline (capped at product-of-extents
  // <= 64) so the reduction stays parseable in an interstate-edge condition -- else e.g. `IF (k >= MINVAL(kmin(iv,:)))`
  // (ICON aes_graupel l341) emerges as `(k >= ?)`, rejected by DaCe's symbolic engine at SDFG-build time.
  // Larger/runtime-extent sections fall through to "?".
  {
    auto opName = def->getName().getStringRef();
    struct RedSpec {
      llvm::StringRef op;
      llvm::StringRef pyOp;  // ``min`` / ``max`` (callable) or ``+`` / ``*``
                             // (binary infix)
      bool isBinop;
    };
    static const RedSpec kRedTbl[] = {
        {"hlfir.minval", "min", false},
        {"hlfir.maxval", "max", false},
        {"hlfir.sum", "+", true},
        {"hlfir.product", "*", true},
        // NOTE: ALL/ANY are deliberately NOT unfolded inline -- they lower to the AllNode/AnyNode library node (bool
        // scalar), materialised before the branch and read, not inlined. See the hlfir.all/any condition
        // materialisation in the dispatch IF handlers.
    };
    for (auto& e : kRedTbl) {
      if (opName != e.op) continue;
      if (def->getNumOperands() == 0) return "?";
      auto src = def->getOperand(0);
      while (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(src.getDefiningOp())) src = cv.getValue();
      auto* sd = src.getDefiningOp();
      if (!sd) return "?";

      // hlfir.elemental source: unfold the body per index combination and combine with the reduction op
      // (constant-extent shape only, same totalExtent <= 64 budget as the designate branch). Surfaces in QE's `IF
      // (SUM((i - i_real) ** 2) > eps6)`.
      if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
        auto shapeVal = elem.getShape();
        auto* shDef = shapeVal.getDefiningOp();
        auto shapeOp = mlir::dyn_cast_or_null<fir::ShapeOp>(shDef);
        if (!shapeOp) return "?";
        std::vector<int64_t> extents;
        int64_t total = 1;
        for (auto extVal : shapeOp.getExtents()) {
          auto ce = traceConstInt(extVal);
          if (!ce) return "?";
          extents.push_back(*ce);
          total *= *ce;
          if (total > 64) return "?";
        }
        auto& region = elem.getRegion();
        if (region.empty()) return "?";
        auto& block = region.front();
        if (block.getNumArguments() != extents.size()) return "?";
        // Find the yield_element op in the body once.
        mlir::Value yielded;
        for (auto& op : block) {
          if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
            yielded = y.getElementValue();
            break;
          }
        }
        if (!yielded) return "?";
        // Enumerate index combinations as literal Fortran-1-based integer strings; buildExprWithSubscripts converts to
        // 0-based at the access sites.
        std::vector<int64_t> cur(extents.size(), 1);
        auto incCur = [&]() -> bool {
          for (int i = static_cast<int>(extents.size()) - 1; i >= 0; --i) {
            if (cur[i] < extents[i]) {
              cur[i]++;
              return true;
            }
            cur[i] = 1;
          }
          return false;
        };
        std::vector<std::string> elems;
        do {
          // Push iter -> value bindings for each block arg.
          unsigned pushed = 0;
          for (unsigned i = 0; i < block.getNumArguments(); ++i) {
            indexStack().emplace_back(block.getArgument(i), std::to_string(cur[i]));
            ++pushed;
          }
          std::string es = buildExprWithSubscripts(yielded, d + 1);
          for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();
          if (es.empty() || es.find('?') != std::string::npos) return "?";
          elems.push_back(std::move(es));
        } while (incCur());
        if (elems.empty()) return "?";
        std::string acc = elems[0];
        for (size_t i = 1; i < elems.size(); ++i) {
          if (e.isBinop) {
            std::string next = "(";
            next += acc;
            next += " ";
            next += e.pyOp.str();
            next += " ";
            next += elems[i];
            next += ")";
            acc = std::move(next);
          } else {
            std::string next = e.pyOp.str();
            next += "(";
            next += acc;
            next += ", ";
            next += elems[i];
            next += ")";
            acc = std::move(next);
          }
        }
        return acc;
      }

      auto dg = mlir::dyn_cast<hlfir::DesignateOp>(sd);
      if (!dg) return "?";
      auto arr = traceToDecl(dg.getMemref());
      if (arr.empty()) return "?";
      auto triplets = dg.getIsTriplet();
      auto allIdx = dg.getIndices();
      if (triplets.empty()) return "?";
      struct DimSpec {
        bool isTriplet;
        std::vector<int64_t> values;  // triplet  --  enumerated 1-based values
        std::string scalarExpr;       // non-triplet  --  Fortran 1-based expr
      };
      std::vector<DimSpec> dims;
      unsigned cursor = 0;
      int64_t totalExtent = 1;
      bool ok = true;
      for (bool const isT : triplets) {
        DimSpec ds;
        ds.isTriplet = isT;
        if (isT) {
          if (cursor + 3 > allIdx.size()) {
            ok = false;
            break;
          }
          auto cLo = traceConstInt(allIdx[cursor]);
          auto cHi = traceConstInt(allIdx[cursor + 1]);
          auto cSt = traceConstInt(allIdx[cursor + 2]);
          if (!cLo || !cHi || !cSt || *cSt == 0) {
            ok = false;
            break;
          }
          int64_t const lo = *cLo;
          int64_t const hi = *cHi;
          int64_t const st = *cSt;
          if (st > 0)
            for (int64_t v = lo; v <= hi; v += st) ds.values.push_back(v);
          else
            for (int64_t v = lo; v >= hi; v += st) ds.values.push_back(v);
          if (ds.values.empty()) {
            ok = false;
            break;
          }
          totalExtent *= static_cast<int64_t>(ds.values.size());
          if (totalExtent > 64) {
            ok = false;
            break;
          }
          cursor += 3;
        } else {
          if (cursor + 1 > allIdx.size()) {
            ok = false;
            break;
          }
          ds.scalarExpr = buildIndexExpr(allIdx[cursor], d + 1);
          if (ds.scalarExpr.empty() || ds.scalarExpr == "?") {
            ok = false;
            break;
          }
          cursor += 1;
        }
        dims.push_back(std::move(ds));
      }
      if (!ok) return "?";
      std::vector<size_t> cur(dims.size(), 0);
      auto incCounters = [&dims](std::vector<size_t>& c) -> bool {
        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
          if (!dims[i].isTriplet) continue;
          if (c[i] + 1 < dims[i].values.size()) {
            c[i]++;
            return true;
          }
          c[i] = 0;
        }
        return false;
      };
      std::vector<std::string> elems;
      do {
        std::string s = arr + "[";
        bool first = true;
        for (size_t i = 0; i < dims.size(); ++i) {
          if (!first) s += ", ";
          if (dims[i].isTriplet)
            s += std::to_string(dims[i].values[cur[i]] - 1);
          else
            s += "(" + dims[i].scalarExpr + ") - 1";
          first = false;
        }
        s += "]";
        elems.push_back(std::move(s));
      } while (incCounters(cur));
      if (elems.empty()) return "?";
      std::string acc = elems[0];
      for (size_t i = 1; i < elems.size(); ++i) {
        if (e.isBinop) {
          std::string next = "(";
          next += acc;
          next += " ";
          next += e.pyOp.str();
          next += " ";
          next += elems[i];
          next += ")";
          acc = std::move(next);
        } else {
          std::string next = e.pyOp.str();
          next += "(";
          next += acc;
          next += ", ";
          next += elems[i];
          next += ")";
          acc = std::move(next);
        }
      }
      return acc;
    }
  }

  // fir.load of hlfir.designate: emit 0-based subscripts, peeling any fir.convert between load and designate -- without
  // this a chain like `fir.convert %designate; fir.load` falls out of the designate branch, bottoms out to buildExpr,
  // and strips the subscript (cloudsc line 2140's `(1.0 - ZA(JL,JK)) < ZEPSEC` hits this via arith.subf -> load ->
  // convert -> designate).
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    mlir::Value mem = ld.getMemref();
    for (int i = 0; i < 128 && mem; ++i) {
      auto* md = mem.getDefiningOp();
      if (!md) break;
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(md)) {
        mem = cv.getValue();
        continue;
      }
      break;
    }
    if (auto md = mem.getDefiningOp())
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md)) {
        // Pass the designate itself to traceToDecl so the component-aware walk fires for struct-field designates (`g %
        // threshold`); traceToDecl(dg.getMemref()) previously bypassed it and leaked the struct base `g` as a free
        // symbol for `if (x > g % threshold)`.
        auto arr = traceToDecl(dg.getResult());
        auto indices = dg.getIndices();
        if (arr.empty()) return "?";
        if (indices.empty()) return arr;
        std::string s = arr + "[";
        bool first = true;
        for (auto idx : indices) {
          if (!first) s += ", ";
          s += "(" + buildIndexExpr(idx, d + 1) + ") - 1";
          first = false;
        }
        s += "]";
        return s;
      }
  }

  // Unary math intrinsics: recurse so the inner array ref keeps its subscript -- else `ABS(a(i)) > eps` renders as
  // `abs(a) > eps` (bare pointer) in the lifted condition, which fails to compile.
  static const std::map<llvm::StringRef, std::string> unary_intrinsics = {
      {"math.absf", "abs"},    {"math.absi", "abs"},    {"math.sqrt", "sqrt"}, {"math.exp", "exp"},
      {"math.exp2", "exp2"},   {"math.log", "log"},     {"math.log2", "log2"}, {"math.log10", "log10"},
      {"math.sin", "sin"},     {"math.cos", "cos"},     {"math.tan", "tan"},   {"math.asin", "asin"},
      {"math.acos", "acos"},   {"math.atan", "atan"},   {"math.sinh", "sinh"}, {"math.cosh", "cosh"},
      {"math.tanh", "tanh"},   {"math.floor", "floor"}, {"math.ceil", "ceil"}, {"math.round", "round"},
      {"math.trunc", "trunc"},
  };
  auto unm = def->getName().getStringRef();
  if (auto it = unary_intrinsics.find(unm); it != unary_intrinsics.end() && def->getNumOperands() == 1)
    return it->second + "(" + buildExprWithSubscripts(def->getOperand(0), d + 1) + ")";

  // Binary arith  --  recurse through the subscript-aware builder.
  static const std::map<llvm::StringRef, std::string> bin_ops = {
      {"arith.mulf", " * "}, {"arith.addf", " + "},   {"arith.subf", " - "},
      {"arith.divf", " / "}, {"arith.muli", " * "},   {"arith.addi", " + "},
      {"arith.subi", " - "}, {"arith.divsi", " // "}, {"arith.divui", " // "},
  };
  auto nm = def->getName().getStringRef();
  if (auto it = bin_ops.find(nm); it != bin_ops.end() && def->getNumOperands() == 2)
    return "(" + buildExprWithSubscripts(def->getOperand(0), d + 1) + it->second +
           buildExprWithSubscripts(def->getOperand(1), d + 1) + ")";
  if (nm == "arith.negf" && def->getNumOperands() == 1)
    return "(-" + buildExprWithSubscripts(def->getOperand(0), d + 1) + ")";

  // Exponentiation a**b (math.fpowi/powf/powi/ipowi): recurse through the subscript-aware builder so a squared
  // per-element term keeps its subscript -- else it falls to buildExpr below and strips it, and QE's `IF (SUM((i -
  // i_real) ** 2) > eps6)` emits whole-array operands that C++ codegen rejects as `int* - double*`.
  if ((nm == "math.fpowi" || nm == "math.powf" || nm == "math.powi" || nm == "math.ipowi") &&
      def->getNumOperands() == 2) {
    return "(" + buildExprWithSubscripts(def->getOperand(0), d + 1) + " ** " +
           buildExprWithSubscripts(def->getOperand(1), d + 1) + ")";
  }

  // Generic fall-through: render any other op via buildExpr with kForceSubscripts set, so element-read leaves keep
  // their subscripts -- the single source of truth retiring the per-op handlers above and the "bare pointer in a
  // condition" bug class.
  ForceSubscriptsGuard const _force;
  return buildExpr(val, d + 1);
}

/// Build a Python-syntax boolean expression for an i1 SSA value (cmpf/cmpi predicates, andi/ori/xori chains incl. the
/// `xori %x, true` not-pattern, constant booleans); opaque inputs fall back to buildExpr (possibly "?").
std::string buildBoolExpr(mlir::Value val, int d) {
  if (d > limits::kBuildExprDepth) return "?";
  auto* def = val.getDefiningOp();
  if (!def) return "?";

  // Synthetic scalars for scf.if results: the emitted assignments write 0/1 into them, so reading the name as-is is
  // semantically a bool.
  {
    auto it = kScfValueMap.find(val);
    if (it != kScfValueMap.end()) return it->second;
  }

  // fir.is_present %x : (!fir.ref<T>) -> i1 is Flang's runtime query for present(x) on an OPTIONAL dummy;
  // lowerIsPresent walks operand back through declare aliases to fir.absent->0, an OPTIONAL dummy->its <name>_present
  // symbol, or a mandatory root->1.
  if (auto isp = mlir::dyn_cast<fir::IsPresentOp>(def)) {
    auto e = lowerIsPresent(isp.getVal());
    if (!e.empty()) return e;
    return "?";
  }

  // fir.convert (i1<->i1 kind, i8->i1, ...) and arith.trunci/extui are transparent here -- DaCe codegen treats any
  // non-zero integer as True in a Python condition, so the cast is a no-op.
  if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) return buildBoolExpr(conv.getValue(), d + 1);
  auto nm2 = def->getName().getStringRef();
  if (nm2 == "arith.trunci" || nm2 == "arith.extui" || nm2 == "arith.extsi") {
    if (def->getNumOperands() == 1) return buildBoolExpr(def->getOperand(0), d + 1);
  }

  // Pick the operand-renderer once for every leaf: tasklet-body context (kBoolExprNoSubscripts via NoSubscriptGuard)
  // wants bare identifiers (emit_tasklet wires subscripts via memlets), interstate-edge/IF contexts want explicit
  // arr[idx] (consumer is an expression parser). leafExpr is reused by the cmp branches and the fall-through so every
  // leaf agrees.
  bool const bareNames = kBoolExprNoSubscripts;
  auto leafExpr = [bareNames](mlir::Value v, int d) -> std::string {
    return bareNames ? buildExpr(v, d) : buildExprWithSubscripts(v, d);
  };

  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
    auto pred = cmpfPredStr(cmp.getPredicate());
    if (pred.empty()) return "?";
    return "(" + leafExpr(cmp.getLhs(), d + 1) + " " + pred + " " + leafExpr(cmp.getRhs(), d + 1) + ")";
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
    // ALLOCATED(arr)/ASSOCIATED(ptr) idiom (`cmpi ne, convert(box_addr(load %decl)), 0`): render as <decl>_allocated
    // instead of decomposing to lhs != rhs (lhs would resolve to "?"). Same matchAssociatedStatusBoxRef as buildExpr in
    // expressions.cpp; re-checked here since boolean contexts decompose before a whole-shape match.
    if (mlir::Value const src = matchAssociatedStatusBoxRef(cmp))
      if (auto arrName = traceToDecl(src); !arrName.empty()) return arrName + "_allocated";
    auto pred = cmpiPredStr(cmp.getPredicate());
    if (pred.empty()) return "?";
    return "(" + leafExpr(cmp.getLhs(), d + 1) + " " + pred + " " + leafExpr(cmp.getRhs(), d + 1) + ")";
  }
  auto nm = def->getName().getStringRef();
  if (nm == "arith.andi" && def->getNumOperands() == 2)
    return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " and " + buildBoolExpr(def->getOperand(1), d + 1) + ")";
  if (nm == "arith.ori" && def->getNumOperands() == 2)
    return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " or " + buildBoolExpr(def->getOperand(1), d + 1) + ")";
  // `xori %x, true` is Flang's lowering of `.not. x`; otherwise boolean xor, rendered as Python != (no xor operator).
  if (nm == "arith.xori" && def->getNumOperands() == 2) {
    auto* rhsDef = def->getOperand(1).getDefiningOp();
    if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(rhsDef))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
        if (ia.getInt() == 1) return "(not " + buildBoolExpr(def->getOperand(0), d + 1) + ")";
    return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " != " + buildBoolExpr(def->getOperand(1), d + 1) + ")";
  }
  if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return ia.getInt() ? "True" : "False";

  // Bool-tree leaf: any non-bool op at the bottom of the recursion (typically fir.load of an i1/fir.logical) goes
  // through leafExpr so the subscripted-vs-bare choice stays consistent across every leaf.
  return leafExpr(val, d + 1);
}

/// scf.if(void) -> kind=conditional; scf.if -> T additionally allocates a __sc_<id> synthetic scalar per result (each
/// arm assigns its yielded value into it, so downstream reads see a real SDFG descriptor); scf.condition(%c) -> if not
/// (%c): break. Pure-value ops (cmp*, load, xori, trunci, extui, convert) don't get AST nodes -- inlined by
/// buildExpr/buildBoolExpr on read. Synthetic scalar name for one scf.if result value; allocated on first reference,
/// memoised after. DaCe's side auto-declares names starting with __sc_.
std::string scfSynthName(mlir::Value v) {
  auto it = kScfValueMap.find(v);
  if (it != kScfValueMap.end()) return it->second;
  std::string s = "__sc_" + std::to_string(kScfValueCounter++);
  kScfValueMap[v] = s;
  return s;
}

static bool isScfIfResult(mlir::Value v) {
  auto* def = v.getDefiningOp();
  return def && mlir::isa<mlir::scf::IfOp>(def);
}

/// Convert a yielded value to a string for writing into a synthetic scalar; reuses buildExpr, which traces through
/// arith ops and cast chains.
std::string yieldedExpr(mlir::Value v) {
  // The yielded value (e.g. arith.andi of i1 cmps for a multi-element AND convergence check) must render WITH
  // subscripts via buildBoolExpr, not through NoSubscriptGuard's bare-name path -- else `rsdnm < tolrsd` renders as a
  // POINTER comparison in C++ (both double* params), silently breaking early-return semantics (NPB LU ssor istep bug;
  // see tests/lu_two_call_convergence_repro_test.py).
  auto b = buildBoolExpr(v, 0);
  if (b != "?") return b;
  return buildExpr(v, 0);
}

}  // namespace hlfir_bridge
