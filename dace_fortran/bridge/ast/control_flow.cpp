// Translation-unit headers.  ``ast_helpers.h`` carries the cross-TU
// API + thread-local state shared with the other ``ast/*.cpp`` files.
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
// MERGE-libcall + buildElementalAssign + comparison primitives + scf.if
// helpers.  Owns:
//   * buildMergeLibcall  --  recognises Flang's hlfir.elemental shape
//     for MERGE(t, f, mask) and routes to MergeLibraryNode.
//   * buildElementalAssign  --  the general elemental walker (where
//     non-MERGE elementals land).
//   * cmpiPredStr / cmpfPredStr  --  Python-syntax predicate strings.
//   * buildExprWithSubscripts  --  like buildExpr but keeps explicit
//     a[i-1] subscripts (interstate-edge condition mode).
//   * buildBoolExpr  --  Python bool expression for arith.cmp* /
//     andi/ori/xori chains, used by both elemental walks and conditionals.
//   * scfSynthName / isScfIfResult / yieldedExpr  --  helpers for
//     the synthetic-scalar scf.if-result machinery.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/control_flow.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list  --  CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
std::vector<ASTNode> buildMergeLibcall(hlfir::AssignOp assign,
                                       hlfir::ElementalOp elem) {
  auto &region = elem.getRegion();
  if (region.empty()) return {};
  auto &block = region.front();

  // Find the yield_element and confirm its operand is an arith.select.
  mlir::Value yielded;
  for (auto &op : block)
    if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
      yielded = y.getElementValue();
      break;
    }
  if (!yielded) return {};
  auto sel =
      mlir::dyn_cast_or_null<mlir::arith::SelectOp>(yielded.getDefiningOp());
  if (!sel) return {};

  // Each of the three operands must trace back to a fir.load of an
  // hlfir.designate of an hlfir.declare.  ``fir.convert`` wrappers
  // (e.g. ``logical<4> -> i1`` for the mask) are transparent.  Bail
  // on anything more elaborate (those go through the generic
  // per-element tasklet path via ``buildElementalAssign``).
  // Operands can be:
  //   * ``fir.load %designate``  --  array element (Flang's array path)
  //   * ``fir.load %declare``    --  scalar dummy (Flang hoists scalar
  //                                loads outside the elemental for
  //                                broadcast variants 3, 4, 5)
  // Either form resolves to a declared array / scalar by name; the
  // library node's expansion later introspects the incoming memlet's
  // subset to decide per-operand whether to broadcast.
  auto traceLoadSource = [](mlir::Value v) -> std::string {
    // Walk through any fir.convert wrappers at the top.
    for (int i = 0; i < 8; ++i) {
      auto *op = v.getDefiningOp();
      if (!op) return "";
      auto cv = mlir::dyn_cast<fir::ConvertOp>(op);
      if (!cv) break;
      v = cv.getValue();
    }
    auto *op = v.getDefiningOp();
    if (!op) return "";
    auto ld = mlir::dyn_cast<fir::LoadOp>(op);
    if (!ld) return "";
    auto *md = ld.getMemref().getDefiningOp();
    if (!md) return "";
    if (mlir::isa<hlfir::DesignateOp>(md) || mlir::isa<hlfir::DeclareOp>(md))
      return traceToDecl(ld.getMemref());
    return "";
  };

  std::string mask_name = traceLoadSource(sel.getCondition());
  std::string t_name = traceLoadSource(sel.getTrueValue());
  std::string f_name = traceLoadSource(sel.getFalseValue());
  if (mask_name.empty() || t_name.empty() || f_name.empty()) return {};

  ASTNode lib;
  lib.kind = "libcall";
  lib.callee = "merge";
  auto dest = assign.getOperand(1);
  if (auto dd = dest.getDefiningOp())
    if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(dd))
      lib.target = extractName(declOp.getUniqName().str());
  if (lib.target.empty()) lib.target = traceToDecl(dest);
  lib.target_is_array = isArrayRef(dest.getType());
  // MergeLibraryNode connector order: ``_t``, ``_f``, ``_mask``.
  lib.call_args.push_back(t_name);
  lib.call_args.push_back(f_name);
  lib.call_args.push_back(mask_name);
  return {std::move(lib)};
}

/// ``b = elementwise_expr(a)``  --  the ``hlfir.assign``'s source is an
/// ``hlfir.elemental``.  Synthesise one ``kind="loop"`` ASTNode per shape
/// dimension (synthetic iter names ``ei0``, ``ei1``, ...) wrapping a single
/// ``kind="assign"`` child whose RHS is the elemental's body expression
/// with the block args replaced by the synthetic iter names.
///
/// ``buildExpr`` consults ``indexStack()`` to resolve an elemental block
/// arg to its synthetic name, so the inner ``buildAssignNode``-style walk
/// sees ``a[ei0]`` etc. as a normal array read with a normal iter var.
std::vector<ASTNode> buildElementalAssign(hlfir::AssignOp assign,
                                          hlfir::ElementalOp elem) {
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
  auto &region = elem.getRegion();
  if (region.empty()) return {};
  auto &block = region.front();
  unsigned rank = block.getNumArguments();

  std::vector<std::string> iter_names;
  iter_names.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    iter_names.push_back("ei" + std::to_string(i));

  // Push block-arg -> synthetic-name pairs so resolveIndex sees them
  // everywhere we walk the body.
  unsigned pushed = 0;
  for (unsigned i = 0; i < rank; ++i) {
    indexStack().push_back({block.getArgument(i), iter_names[i]});
    ++pushed;
  }

  // Inner write access: target[<per-array-dim index>].  When the
  // destination is a section designate ``a(lo:hi)`` we need to add
  // ``(lo - 1)`` to each triplet-iter so the write lands on the right
  // element of the root array (mirrors the same logic in
  // ``buildDesignateIndexExpr`` for reads through nested section
  // designates).  When the designate mixes triplet + scalar dims
  // (e.g. ``res(:, pos(1)+2) = input1 + input2``) the per-array-dim
  // index list is one-per-dim of the underlying array  --  triplet dims
  // contribute their ``ei_<tDim>`` iter, scalar dims contribute the
  // scalar's Fortran 1-based index expression.  The elemental's rank
  // matches the triplet count, NOT the underlying array's rank.
  //
  // ``LowerAdj`` keeps the constant-fold and symbolic-fallback paths
  // in one place: ``expr`` non-empty -> ``(iter + expr - 1)`` form;
  // ``value != 0`` -> integer offset; both empty / zero -> bare iter.
  struct LowerAdj {
    int64_t value = 0;
    std::string expr;
  };
  AccessInfo wa;
  wa.array_name = inner.target;
  wa.is_write = true;
  if (auto dd = dest.getDefiningOp()) {
    if (auto dstDg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
      auto triplets = dstDg.getIsTriplet();
      auto idxOps = dstDg.getIndices();
      unsigned cursor = 0;
      unsigned tDim = 0;
      for (bool isT : triplets) {
        if (isT && tDim < rank && cursor + 3 <= idxOps.size()) {
          LowerAdj adj;
          if (auto lo = traceConstInt(idxOps[cursor])) {
            adj.value = *lo - 1;
          } else {
            auto loExpr = buildIndexExpr(idxOps[cursor], 0);
            if (!loExpr.empty() && loExpr != "?") adj.expr = std::move(loExpr);
          }
          std::string ix = iter_names[tDim];
          if (!adj.expr.empty())
            ix = "(" + ix + " + " + adj.expr + " - 1)";
          else if (adj.value > 0)
            ix = "(" + ix + " + " + std::to_string(adj.value) + ")";
          else if (adj.value < 0)
            ix = "(" + ix + " - " + std::to_string(-adj.value) + ")";
          wa.index_vars.push_back(iter_names[tDim]);
          wa.index_exprs.push_back(std::move(ix));
          cursor += 3;
          tDim++;
        } else if (!isT && cursor < idxOps.size()) {
          // Scalar dim  --  thread its (Fortran 1-based) index
          // expression directly into the write memlet so the
          // memlet rank matches the underlying array.
          auto sc = buildIndexExpr(idxOps[cursor], 0);
          if (sc.empty() || sc == "?") sc = "?";
          wa.index_vars.push_back(sc);
          wa.index_exprs.push_back(std::move(sc));
          cursor += 1;
        } else {
          // Defensive: skip bad cursor advance to avoid an
          // infinite loop on malformed input.
          cursor += isT ? 3 : 1;
          if (isT) tDim++;
        }
      }
    } else {
      // Bare ``hlfir.declare``  --  write across the elemental's full
      // rank (every dim is a triplet covering the array's extent).
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
  for (auto &op : block)
    if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
      yielded = y.getElementValue();
      break;
    }

  // Pre-walk: any ``hlfir.apply <libcall_expr>`` we encounter inside
  // the elemental body needs the libcall (``hlfir.matmul`` /
  // ``hlfir.transpose`` / ...) materialised into a real transient
  // BEFORE the elemental itself runs.  Without this, ``buildExpr``
  // sees an apply whose source isn't an inner elemental and returns
  // ``?``, producing tasklet bodies like ``2 - ?``.  Each libcall op
  // gets a unique ``_libtmp_<gid>`` name + a pair of pre-emitted AST
  // nodes (``declare_transient`` for the descriptor, ``libcall`` for
  // the runtime computation).  ``buildExpr`` then renders the apply
  // as a regular Fortran-style read of the transient.
  std::vector<ASTNode> preNodes;
  if (yielded) {
    std::function<void(mlir::Value, int)> findApplies = [&](mlir::Value v,
                                                            int depth) {
      if (depth > 40 || !v) return;
      auto *op = v.getDefiningOp();
      if (!op) return;
      if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
        auto src = apply.getExpr();
        if (auto *srcOp = src.getDefiningOp()) {
          // Inner elemental -> existing path inlines the body.
          // Walk the body's yielded value (where the real applies
          // live) -- ``findApplies(src, ...)`` would just recurse on
          // the elemental's top-level operands (its shape), missing
          // every libcall expr-producer the body's apply reads.  QE's
          // ``vcut_get`` (``MATMUL(TRANSPOSE(...)) / tpi``) flang-
          // lowers as TWO nested elementals: outer divides by tpi,
          // inner wraps matmul + ``hlfir.no_reassoc``.  Without this
          // body-walk, the matmul never lands in
          // ``kHlfirExprToTransient`` and the inner apply renders as
          // ``?`` in the consuming tasklet body.
          if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(srcOp)) {
            auto &iregion = inner_elem.getRegion();
            if (!iregion.empty()) {
              for (auto &iop : iregion.front()) {
                if (auto iy = mlir::dyn_cast<hlfir::YieldElementOp>(iop)) {
                  findApplies(iy.getElementValue(), depth + 1);
                  break;
                }
              }
            }
            return;
          }
          // Recognised libcall expr-producer -> materialise.
          if (const char *callee = libcallNameForExprOp(srcOp)) {
            if (!kHlfirExprToTransient.count(srcOp)) {
              std::string tmp = "_libtmp_" + std::to_string(kLibTmpCounter++);
              kHlfirExprToTransient[srcOp] = tmp;

              mlir::Type rty = srcOp->getResult(0).getType();
              auto shape = exprResultShape(rty);

              ASTNode decl;
              decl.kind = "declare_transient";
              decl.target = tmp;
              decl.expr = exprDtypeString(rty);
              decl.target_is_array = !shape.empty();
              AccessInfo shapeInfo;
              shapeInfo.array_name = tmp;
              for (auto &s : shape) shapeInfo.index_exprs.push_back(s);
              decl.accesses.push_back(std::move(shapeInfo));
              preNodes.push_back(std::move(decl));

              ASTNode lib;
              lib.kind = "libcall";
              lib.target = tmp;
              lib.target_is_array = !shape.empty();
              lib.callee = callee;
              // CSHIFT as an EXPR-PRODUCER (``2.0 - CSHIFT(arr, 1)``):
              // the shift is a SCALAR that belongs in ``options["shift"]``
              // (the dim in ``reduce_axes``), NOT a ``call_args`` array
              // operand.  ``buildLibCallNode`` already does this for the
              // whole-array assign form; mirror it here.  Without this,
              // ``options["shift"]`` stays empty, ``emit_library`` builds
              // the ``CShift`` node with ``shift=None``, and the pure
              // expansion falls back to the ``__shift`` symbol -- which
              // is referenced in the memlet subset but never assigned, so
              // it leaks as a free symbol (``KeyError: '__shift'``).
              if (auto cshOp = mlir::dyn_cast<hlfir::CShiftOp>(srcOp)) {
                auto arrName = traceToDecl(cshOp.getArray());
                if (arrName.empty())
                  if (auto *ad = cshOp.getArray().getDefiningOp())
                    if (auto ae = mlir::dyn_cast<hlfir::ElementalOp>(ad)) {
                      auto [trName, mat] = materialiseElementalForLibcall(ae);
                      if (!trName.empty()) {
                        for (auto &mn : mat) preNodes.push_back(std::move(mn));
                        arrName = std::move(trName);
                      }
                    }
                lib.call_args.push_back(arrName);
                auto shiftVal = cshOp.getShift();
                if (auto c = traceConstInt(shiftVal))
                  lib.options["shift"] = std::to_string(*c);
                else {
                  auto sExpr = buildIndexExpr(shiftVal, 0);
                  if (!sExpr.empty() && sExpr != "?")
                    lib.options["shift"] = sExpr;
                }
                if (auto dim = cshOp.getDim())
                  if (auto c = traceConstInt(dim))
                    lib.reduce_axes.push_back(*c - 1);
                preNodes.push_back(std::move(lib));
                return;
              }
              for (auto operand : srcOp->getOperands()) {
                auto n = traceToDecl(operand);
                if (n.empty()) {
                  // Same fix shape as dispatch.cpp's
                  // libcall-over-elemental: when the
                  // operand is an inline
                  // ``hlfir.elemental`` (e.g.
                  // ``transpose(<inner gather>)``),
                  // ``traceToDecl`` returns "" because
                  // the elemental has no backing
                  // declare.  Materialise the elemental
                  // into a synthetic transient and pass
                  // its name as the libcall arg.
                  if (auto *od = operand.getDefiningOp()) {
                    if (auto inner_elem =
                            mlir::dyn_cast<hlfir::ElementalOp>(od)) {
                      auto [trName, mat_nodes] =
                          materialiseElementalForLibcall(inner_elem);
                      if (!trName.empty()) {
                        for (auto &mn : mat_nodes)
                          preNodes.push_back(std::move(mn));
                        n = std::move(trName);
                      }
                    }
                    // Nested libcall expr-producer (e.g.
                    // ``MATMUL(TRANSPOSE(a), q)`` -- the
                    // matmul's first operand is the
                    // transpose result, which is also a
                    // libcall expr-producer with no
                    // backing declare).  Recursively
                    // materialise it via the same
                    // ``_libtmp_<gid>`` mechanism so the
                    // outer libcall's source arg is a
                    // real array name, not the empty
                    // string ``traceToDecl`` returned.
                    // QE's ``vcut_get`` (``i_real =
                    // MATMUL(TRANSPOSE(vcut % a), q)
                    // / tpi``) was the surfacing case.
                    else if (n.empty() && libcallNameForExprOp(od)) {
                      // Memoise per (op, transient name)
                      // so a transpose result shared
                      // between two consumers reuses the
                      // same transient.
                      auto it = kHlfirExprToTransient.find(od);
                      std::string innerTr;
                      if (it != kHlfirExprToTransient.end()) {
                        innerTr = it->second;
                      } else {
                        innerTr = "_libtmp_" +
                                  std::to_string(kLibTmpCounter++);
                        kHlfirExprToTransient[od] = innerTr;
                        mlir::Type irty = od->getResult(0).getType();
                        auto ishape = exprResultShape(irty);

                        ASTNode idecl;
                        idecl.kind = "declare_transient";
                        idecl.target = innerTr;
                        idecl.expr = exprDtypeString(irty);
                        idecl.target_is_array = !ishape.empty();
                        AccessInfo ishapeInfo;
                        ishapeInfo.array_name = innerTr;
                        for (auto &s : ishape)
                          ishapeInfo.index_exprs.push_back(s);
                        idecl.accesses.push_back(std::move(ishapeInfo));
                        preNodes.push_back(std::move(idecl));

                        ASTNode ilib;
                        ilib.kind = "libcall";
                        ilib.target = innerTr;
                        ilib.target_is_array = !ishape.empty();
                        ilib.callee = libcallNameForExprOp(od);
                        for (auto iop : od->getOperands()) {
                          auto in = traceToDecl(iop);
                          // Only one-level nesting handled
                          // here; deeper would need full
                          // recursion (uncommon shape).
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

  // Walk the yielded expression in tasklet-body mode: any embedded
  // comparisons (``a .eq. b`` inside a MERGE mask, etc.) must produce
  // bare names so emit_tasklet's per-occurrence connector wiring
  // matches.  Without this the bool path emits ``a[ei0-1] == b`` and
  // the bare-name path emits just ``a``  --  same array, two different
  // forms in one tasklet body, which leaks ``ei0`` as a free symbol.
  {
    NoSubscriptGuard g;
    inner.expr = yielded ? buildExpr(yielded, 0) : "?";
  }

  // Read accesses.  Unlike plain assigns we must follow hlfir.apply into
  // the referenced hlfir.elemental's body (where the real designate
  // lives)  --  pushing the apply's index mapping onto indexStack() so the
  // designate sees the same synthetic iter names as the outer elemental.
  if (yielded) {
    // Per-occurrence AccessInfo (depth-limited, no op-identity dedup).
    // emit_tasklet counts array-name regex occurrences in the RHS
    // string; shared SSA values (``x * x``) must yield matching
    // AccessInfo count or downstream wiring strands a connector.
    collectReadAccesses(yielded, inner.accesses, 0);
  }

  // Pop the stack frames we pushed.
  for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();

  // Wrap the inner assign in one ASTNode kind="loop" per rank.  The
  // outermost loop is the result; deeper loops live as its sole child.
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
  // Prepend any libcall-result materialisations so the transient is
  // declared and populated before the elemental body reads it.
  if (!preNodes.empty()) {
    preNodes.push_back(std::move(current));
    return preNodes;
  }
  return {current};
}

/// Render an arith::cmpi predicate as a Python comparison operator.  Returns
/// an empty string for signed/unsigned variants we haven't wired up yet.
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

/// Render an arith::cmpf predicate as a Python comparison operator.  Ordered
/// and unordered predicates both collapse to the same Python operator; NaN
/// handling is beyond what a Python condition string can express, so we
/// accept the lossy mapping.
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

/// Like ``buildExpr`` but keeps explicit array subscripts (``a[(i) - 1]``)
/// when the value is a ``fir.load`` of a ``hlfir.designate``.  Used by
/// ``buildBoolExpr`` so interstate-edge conditions can reference array
/// elements directly  --  they're evaluated in the caller's frame, not by a
/// tasklet, so they can't rely on memlet-wired connectors.
std::string buildExprWithSubscripts(mlir::Value val, int d) {
  if (d > limits::kBuildExprDepth || !val) return "?";
  // Output lands in an interstate-edge / ConditionalBlock condition,
  // parsed by DaCe's symbolic engine which can't handle the
  // ``dace.float32(...)`` precision wrap (treats ``dace`` as a free
  // symbol).  Suppress the wrap for the duration of this walk; the
  // f32-vs-f64 distinction doesn't change comparison outcomes.
  SuppressFloatCastGuard floatCastGuard;
  auto *def = val.getDefiningOp();
  if (!def) return "?";
  // A reduction op materialised into a scalar by ``materialiseCondReductions``
  // (Reduce lib-node before the branch) renders as the bare scalar -- the
  // inline-unroll reduction table below is bypassed for it.
  if (auto it = kCondReductionScalars.find(def);
      it != kCondReductionScalars.end())
    return it->second;

  if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
    return buildExprWithSubscripts(conv.getValue(), d + 1);
  // ``hlfir.no_reassoc`` is Fortran's reassociation-blocker wrapper
  // Flang inserts around expressions whose order the standard says
  // must be preserved (parenthesised expressions like
  // ``(1.0 - ZA(JL, JK))``).  It's pure structural metadata  --  peel
  // through so the inner subf / load / designate chain stays
  // subscript-aware.  Without this peel, the bridge bottoms out to
  // ``buildExpr`` and emits bare ``za`` (no subscript) in the cond,
  // which C++ codegen then renders as ``int - double*``.
  auto _nm = def->getName().getStringRef();
  if (_nm == "hlfir.no_reassoc" && def->getNumOperands() == 1)
    return buildExprWithSubscripts(def->getOperand(0), d + 1);

  // ``hlfir.minval`` / ``maxval`` / ``sum`` / ``product`` over a
  // CONSTANT-extent array section  --  unfold inline so the reduction
  // stays parseable in an interstate-edge condition.  Without this,
  // ``IF (k >= MINVAL(kmin(iv, :)))`` (ICON aes_graupel l341) hits
  // the ``buildExpr`` fall-through and emerges as ``(k >= ?)``, which
  // DaCe's symbolic engine then rejects when specialising symbols
  // at SDFG-build time.  Capped at product-of-extents <= 64 to keep
  // the unfolded expression bounded; larger or runtime-extent
  // sections fall through to ``?`` (and are reported as such by
  // the downstream emitter).
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
        // NOTE: ALL / ANY are deliberately NOT unfolded inline here.
        // Per design they lower to the ``AllNode`` / ``AnyNode`` library
        // node (returning a boolean scalar) -- the IF-condition path
        // materialises that scalar BEFORE the branch and reads it, rather
        // than inlining ``(odg[0] and odg[1] and ...)`` into the
        // condition.  See the ``hlfir.all`` / ``hlfir.any`` condition
        // materialisation in the dispatch IF handlers.
    };
    for (auto& e : kRedTbl) {
      if (opName != e.op) continue;
      if (def->getNumOperands() == 0) return "?";
      auto src = def->getOperand(0);
      while (auto cv =
                 mlir::dyn_cast_or_null<fir::ConvertOp>(src.getDefiningOp()))
        src = cv.getValue();
      auto* sd = src.getDefiningOp();
      if (!sd) return "?";

      // ``hlfir.elemental`` source -- unfold the elemental body per
      // index combination and combine with the reduction op.
      // Surfaces in QE's ``IF (SUM((i - i_real) ** 2) > eps6)`` where
      // the elemental computes ``(i[k] - i_real[k])**2`` for k=0..2
      // and SUM reduces.  Constant-extent shape only (matches the
      // designate branch's ``totalExtent <= 64`` budget).
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
        auto &region = elem.getRegion();
        if (region.empty()) return "?";
        auto &block = region.front();
        if (block.getNumArguments() != extents.size()) return "?";
        // Find the yield_element op in the body once.
        mlir::Value yielded;
        for (auto &op : block) {
          if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
            yielded = y.getElementValue();
            break;
          }
        }
        if (!yielded) return "?";
        // Enumerate the index combinations (Fortran 1-based -> the
        // ``buildExprWithSubscripts`` walk converts to 0-based at
        // the access sites; here we push the index name as a literal
        // Fortran-1-based integer string).
        std::vector<int64_t> cur(extents.size(), 1);
        auto incCur = [&]() -> bool {
          for (int i = static_cast<int>(extents.size()) - 1; i >= 0; --i) {
            if (cur[i] < extents[i]) { cur[i]++; return true; }
            cur[i] = 1;
          }
          return false;
        };
        std::vector<std::string> elems;
        do {
          // Push iter -> value bindings for each block arg.
          unsigned pushed = 0;
          for (unsigned i = 0; i < block.getNumArguments(); ++i) {
            indexStack().push_back(
                {block.getArgument(i), std::to_string(cur[i])});
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
          if (e.isBinop)
            acc = "(" + acc + " " + e.pyOp.str() + " " + elems[i] + ")";
          else
            acc = e.pyOp.str() + "(" + acc + ", " + elems[i] + ")";
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
      for (bool isT : triplets) {
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
          int64_t lo = *cLo, hi = *cHi, st = *cSt;
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
        if (e.isBinop)
          acc = "(" + acc + " " + e.pyOp.str() + " " + elems[i] + ")";
        else
          acc = e.pyOp.str() + "(" + acc + ", " + elems[i] + ")";
      }
      return acc;
    }
  }

  // fir.load of hlfir.designate: emit 0-based subscripts.  Peel
  // through any ``fir.convert`` (kind coercion, ref-shape rebox)
  // between the load and the designate  --  without this peel a chain
  // like ``%2 = fir.convert %designate ; %v = fir.load %2`` falls
  // out of the designate branch and ends up bottoming out to
  // ``buildExpr``, which strips the subscript and leaves a bare
  // array name in the interstate-edge / cond expression.  cloudsc
  // line 2140 (``(1.0 - ZA(JL, JK)) < ZEPSEC``) hits this through
  // the arith.subf -> load -> convert -> designate chain Flang
  // emits for the per-element load.
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    mlir::Value mem = ld.getMemref();
    for (int i = 0; i < 128 && mem; ++i) {
      auto *md = mem.getDefiningOp();
      if (!md) break;
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(md)) {
        mem = cv.getValue();
        continue;
      }
      break;
    }
    if (auto md = mem.getDefiningOp())
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(md)) {
        // Pass the designate itself to ``traceToDecl`` so the
        // component-aware walk fires for struct-field designates
        // (``g % threshold``); element / section designates fall
        // through to the parent name unchanged.  Previously
        // ``traceToDecl(dg.getMemref())`` bypassed the component
        // branch and returned the struct base ``g`` for the
        // ``if (x > g % threshold)`` shape, leaking ``g`` as a
        // free symbol into the interstate-edge condition.
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

  // Unary math intrinsics  --  recurse so the inner array reference
  // keeps its subscript.  Without this, ``ABS(a(i)) > eps`` in an
  // IF condition gets lifted to an interstate-edge expression that
  // C++ renders as ``abs(a) > eps`` (bare array pointer), which
  // fails to compile.  Handles the math.* and complex.* ops Flang
  // emits for Fortran's intrinsic library.
  static const std::map<llvm::StringRef, std::string> unary_intrinsics = {
      {"math.absf", "abs"},  {"math.absi", "abs"},    {"math.sqrt", "sqrt"},
      {"math.exp", "exp"},   {"math.exp2", "exp2"},   {"math.log", "log"},
      {"math.log2", "log2"}, {"math.log10", "log10"}, {"math.sin", "sin"},
      {"math.cos", "cos"},   {"math.tan", "tan"},     {"math.asin", "asin"},
      {"math.acos", "acos"}, {"math.atan", "atan"},   {"math.sinh", "sinh"},
      {"math.cosh", "cosh"}, {"math.tanh", "tanh"},   {"math.floor", "floor"},
      {"math.ceil", "ceil"}, {"math.round", "round"}, {"math.trunc", "trunc"},
  };
  auto unm = def->getName().getStringRef();
  if (auto it = unary_intrinsics.find(unm);
      it != unary_intrinsics.end() && def->getNumOperands() == 1)
    return it->second + "(" +
           buildExprWithSubscripts(def->getOperand(0), d + 1) + ")";

  // Binary arith  --  recurse through the subscript-aware builder.
  static const std::map<llvm::StringRef, std::string> bin_ops = {
      {"arith.mulf", " * "}, {"arith.addf", " + "},   {"arith.subf", " - "},
      {"arith.divf", " / "}, {"arith.muli", " * "},   {"arith.addi", " + "},
      {"arith.subi", " - "}, {"arith.divsi", " // "}, {"arith.divui", " // "},
  };
  auto nm = def->getName().getStringRef();
  if (auto it = bin_ops.find(nm);
      it != bin_ops.end() && def->getNumOperands() == 2)
    return "(" + buildExprWithSubscripts(def->getOperand(0), d + 1) +
           it->second + buildExprWithSubscripts(def->getOperand(1), d + 1) +
           ")";
  if (nm == "arith.negf" && def->getNumOperands() == 1)
    return "(-" + buildExprWithSubscripts(def->getOperand(0), d + 1) + ")";

  // Exponentiation ``a ** b`` (Flang's ``math.fpowi`` float**int,
  // ``math.powf`` float**float, ``math.powi`` / ``math.ipowi`` int**int).
  // Recurse the base (and exponent) through the subscript-aware builder so a
  // squared per-element term keeps its subscript.  Without this the power op
  // falls through to ``buildExpr`` below, which strips subscripts and leaves a
  // bare array name -- QE's ``IF (SUM((i - i_real) ** 2) > eps6)`` (the
  // elemental-unfold above renders each term via this builder) then emits
  // ``(float64(i) - i_real) ** 2`` with whole-array ``i`` / ``i_real``
  // operands, which C++ codegen rejects as ``int* - double*``.
  if ((nm == "math.fpowi" || nm == "math.powf" || nm == "math.powi" || nm == "math.ipowi") &&
      def->getNumOperands() == 2) {
    return "(" + buildExprWithSubscripts(def->getOperand(0), d + 1) + " ** " +
           buildExprWithSubscripts(def->getOperand(1), d + 1) + ")";
  }

  // GENERIC fall-through: render any other op (min / max / abs / select /
  // unhandled intrinsic / ...) via ``buildExpr`` BUT with ``kForceSubscripts``
  // set, so its element-read leaves keep their subscripts.  This is the single
  // source of truth that retires the per-op subscript handlers above and the
  // whole "bare array pointer in a condition" class of bug -- ``buildExpr``
  // already knows how to spell every op; the only thing it would otherwise get
  // wrong in a condition context is stripping the leaf subscript.
  ForceSubscriptsGuard _force;
  return buildExpr(val, d + 1);
}

/// Build a Python-syntax boolean expression for an ``i1`` SSA value.
/// Recognises ``arith.cmpf``, ``arith.cmpi``, ``arith.andi``, ``arith.ori``
/// (used as boolean ops on i1), ``arith.xori`` (boolean xor / ``not`` pattern
/// ``xori %x, true``), and constant booleans.  Opaque inputs fall back to
/// ``buildExpr`` (which may still produce a usable Python expression for the
/// condition, or ``"?"`` when the shape isn't understood).
std::string buildBoolExpr(mlir::Value val, int d) {
  if (d > limits::kBuildExprDepth) return "?";
  auto *def = val.getDefiningOp();
  if (!def) return "?";

  // Synthetic scalars for scf.if results.  The assignments we emit for
  // yielded values write 0/1 into them, so reading the name as-is is
  // semantically a bool.
  {
    auto it = kScfValueMap.find(val);
    if (it != kScfValueMap.end()) return it->second;
  }

  // ``fir.is_present %x : (!fir.ref<T>) -> i1``  --  the runtime query
  // Flang emits for Fortran's ``present(x)`` on an OPTIONAL dummy.
  // ``lowerIsPresent`` (in expressions.inc) walks the operand back
  // through inlined declare aliases to one of: ``fir.absent`` -> 0,
  // a host OPTIONAL dummy -> its companion ``<name>_present`` symbol,
  // or a mandatory root -> 1.
  if (auto isp = mlir::dyn_cast<fir::IsPresentOp>(def)) {
    auto e = lowerIsPresent(isp.getVal());
    if (!e.empty()) return e;
    return "?";
  }

  // fir.convert (i1 <-> i1 kind, i8 -> i1, ...) and arith.trunci / extui
  // are transparent here  --  DaCe codegen treats any non-zero integer as
  // True inside a Python condition, so the cast is a no-op.
  if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def))
    return buildBoolExpr(conv.getValue(), d + 1);
  auto nm2 = def->getName().getStringRef();
  if (nm2 == "arith.trunci" || nm2 == "arith.extui" || nm2 == "arith.extsi") {
    if (def->getNumOperands() == 1)
      return buildBoolExpr(def->getOperand(0), d + 1);
  }

  // Pick the operand-renderer once for every leaf in this bool tree:
  // tasklet-body context (``kBoolExprNoSubscripts`` set via
  // ``NoSubscriptGuard`` by elemental walks, MERGE-of-scalars, or
  // the i1 ``andi`` / ``ori`` chain handler) wants bare identifiers
  // because emit_tasklet's regex rewrite later turns them into
  // ``_in_a_0`` connectors and wires subscripts through memlets.
  // Interstate-edge / IF-condition contexts (the default) want the
  // explicit ``arr[idx]`` form because the consumer is an expression
  // parser, not a tasklet rewrite.  ``leafExpr`` is reused by the
  // cmp branches AND the last-resort fall-through so every leaf
  // threads through the same rendering decision.
  bool bareNames = kBoolExprNoSubscripts;
  auto leafExpr = [bareNames](mlir::Value v, int d) -> std::string {
    return bareNames ? buildExpr(v, d) : buildExprWithSubscripts(v, d);
  };

  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpFOp>(def)) {
    auto pred = cmpfPredStr(cmp.getPredicate());
    if (pred.empty()) return "?";
    return "(" + leafExpr(cmp.getLhs(), d + 1) + " " + pred + " " +
           leafExpr(cmp.getRhs(), d + 1) + ")";
  }
  if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(def)) {
    // ALLOCATED(arr) idiom: ``cmpi ne, convert(box_addr(load %decl)), 0``.
    // Render as ``<decl>_allocated`` (the bridge-synthesised
    // tracker symbol) instead of decomposing the cmp into
    // ``<lhs> != <rhs>`` where the LHS resolves to ``?``.
    // Same recognition as buildExpr's path in expressions.cpp;
    // duplicated here because boolean contexts decompose before
    // the LHS gets a whole-shape pattern match.
    if (cmp.getPredicate() == mlir::arith::CmpIPredicate::ne) {
      bool rhsZero = false;
      if (auto c = traceConstInt(cmp.getRhs())) rhsZero = (*c == 0);
      if (rhsZero) {
        mlir::Value cur = cmp.getLhs();
        for (int i = 0; i < limits::kConvertChainDepth && cur; ++i) {
          auto *cd = cur.getDefiningOp();
          if (!cd) break;
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(cd)) {
            cur = cv.getValue();
            continue;
          }
          break;
        }
        if (cur) {
          if (auto *cd = cur.getDefiningOp()) {
            if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(cd)) {
              auto src = ba.getVal();
              if (auto *sd = src.getDefiningOp())
                if (auto ld = mlir::dyn_cast<fir::LoadOp>(sd))
                  src = ld.getMemref();
              auto arrName = traceToDecl(src);
              if (!arrName.empty()) return arrName + "_allocated";
            }
          }
        }
      }
    }
    auto pred = cmpiPredStr(cmp.getPredicate());
    if (pred.empty()) return "?";
    return "(" + leafExpr(cmp.getLhs(), d + 1) + " " + pred + " " +
           leafExpr(cmp.getRhs(), d + 1) + ")";
  }
  auto nm = def->getName().getStringRef();
  if (nm == "arith.andi" && def->getNumOperands() == 2)
    return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " and " +
           buildBoolExpr(def->getOperand(1), d + 1) + ")";
  if (nm == "arith.ori" && def->getNumOperands() == 2)
    return "(" + buildBoolExpr(def->getOperand(0), d + 1) + " or " +
           buildBoolExpr(def->getOperand(1), d + 1) + ")";
  // ``xori %x, true`` is Flang's lowering of ``.not. x``.  Otherwise
  // boolean xor  --  Python has no operator, use ``!=``.
  if (nm == "arith.xori" && def->getNumOperands() == 2) {
    auto *rhsDef = def->getOperand(1).getDefiningOp();
    if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(rhsDef))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
        if (ia.getInt() == 1)
          return "(not " + buildBoolExpr(def->getOperand(0), d + 1) + ")";
    return "(" + buildBoolExpr(def->getOperand(0), d + 1) +
           " != " + buildBoolExpr(def->getOperand(1), d + 1) + ")";
  }
  if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt() ? "True" : "False";

  // Bool-tree leaf: any non-bool op reached at the bottom of the
  // recursion (typically a ``fir.load`` of an i1 / fir.logical) goes
  // through ``leafExpr`` so the operand-renderer choice (subscripted
  // vs bare) stays consistent across every leaf in this tree.
  return leafExpr(val, d + 1);
}

/// Faithful ``scf.while`` translator.
///
/// Rather than pattern-matching the shape ``lift-cf-to-scf`` produces, we
/// copy every structural op in the before-region into the AST one-for-one:
///
///   * ``scf.if`` (void)     -> ``kind="conditional"`` with recursively walked
///   arms.
///   * ``scf.if -> T``       -> same, but we allocate a ``__sc_<id>`` synthetic
///                             int scalar per result; each arm ends with a
///                             ``kind="assign"`` writing the yielded value to
///                             that scalar so downstream reads of the result
///                             find a real SDFG data descriptor.
///   * ``scf.condition(%c)`` -> ``if not (%c): break``.
///   * ``hlfir.assign``       -> existing ``buildAssignNode`` path.
///
/// Pure-value ops (``arith.cmp*``, ``fir.load``, ``arith.xori``,
/// ``arith.trunci``,
/// ``arith.extui``, ``fir.convert``, ...) don't become AST nodes  --  their
/// values are inlined by ``buildExpr`` / ``buildBoolExpr`` when downstream ops
/// read them.
///
/// The synthetic-scalar trick means the whole translation is compositional:
/// every MLIR op maps to one SDFG primitive, no special cases for EXIT or
/// value-yielding scf.if nestings.  DaCe's IR-level simplification can
/// re-flatten the result if it wants to.
/// Synthetic scalar name for one scf.if result value.  Allocated on first
/// reference; subsequent references return the same name.  DaCe's side
/// auto-declares names starting with ``__sc_``.
std::string scfSynthName(mlir::Value v) {
  auto it = kScfValueMap.find(v);
  if (it != kScfValueMap.end()) return it->second;
  std::string s = "__sc_" + std::to_string(kScfValueCounter++);
  kScfValueMap[v] = s;
  return s;
}

static bool isScfIfResult(mlir::Value v) {
  auto *def = v.getDefiningOp();
  return def && mlir::isa<mlir::scf::IfOp>(def);
}

std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block &block);

/// Helper: convert a yielded value to a string for writing into a synthetic
/// scalar.  scf.yield of an i32 constant / boolean / computed expression  --
/// just reuse buildExpr, which traces through arith ops and cast chains.
std::string yieldedExpr(mlir::Value v) {
  // The yielded value lands in an ``__sc_<N>`` interstate-edge
  // assignment / scalar tasklet for a downstream conditional check
  // (lift-cf-to-scf encodes Fortran ``if (...) early-return`` as
  // scf.if yielding an i32 then trunci-tested at the surrounding
  // scf.condition).  When the yielded value is an ``arith.andi`` of
  // i1 cmp results -- the multi-element AND convergence check
  // ``rsdnm(1) < tolrsd(1) .and. rsdnm(2) < tolrsd(2)`` -- buildExpr
  // at expressions.cpp:1280-1288 sets ``NoSubscriptGuard`` and routes
  // through ``buildBoolExpr``, stripping the per-cmp ``rsdnm[0]``
  // subscripts from the assumption that emit_tasklet wires them via
  // memlets.  But the snapshot assign here ends up in a tasklet
  // body where the regex rewrite only touches BARE names -- the
  // ``rsdnm < tolrsd`` cmpf renders as POINTER comparison in C++
  // (rsdnm and tolrsd are ``double*`` arglist params), which is
  // deterministic at runtime based on memory layout and silently
  // breaks the early-return semantics (NPB LU's ssor istep loop:
  // EVEN iter counts wrap to 1 iter, ODD counts iterate fully --
  // see tests/lu_two_call_convergence_repro_test.py for the
  // distilled repro).  Render with subscripts directly via
  // ``buildBoolExpr`` so the leaf cmpf operands keep their
  // ``arr[idx]`` form.
  auto b = buildBoolExpr(v, 0);
  if (b != "?") return b;
  return buildExpr(v, 0);
}

}  // namespace hlfir_bridge
