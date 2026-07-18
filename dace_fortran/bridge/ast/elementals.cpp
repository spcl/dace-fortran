// ast_helpers.h carries the cross-TU API + thread-local state shared with the other ast/*.cpp files.
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

// Reduction + elemental-libcall builders (buildReduceNode, buildElementalCountLibcall, buildSelectCaseChain,
// resolveExtent + helpers). Included verbatim into extract_ast.cpp -- must NOT be added to the build's compile list
// (CMakeLists.txt omits it).
ASTNode buildReduceNode(hlfir::AssignOp assign, mlir::Operation* redOp, std::string_view wcr,
                        std::string_view identity) {
  ASTNode n;
  n.kind = "reduce";

  captureElementDesignateWrite(assign.getOperand(1), n);

  // Source array  --  operand 0 of the reduction op.
  if (redOp->getNumOperands() > 0) n.reduce_src = traceToDecl(redOp->getOperand(0));
  n.reduce_wcr = wcr;
  n.reduce_identity = identity;

  // ``hlfir.sum %arr dim %d``: trace operand 1; absent (whole-array reduction) leaves reduce_axes empty.
  if (redOp->getNumOperands() >= 2) {
    auto d = redOp->getOperand(1);
    if (auto c = traceConstInt(d))
      // Fortran ``dim`` is 1-based; DaCe axes are 0-based.
      n.reduce_axes.push_back(*c - 1);
  }
  return n;
}

/// Forward declare  --  called from buildWhileNode to recurse into the body.

/// SELECT CASE has no direct DaCe equivalent: folds each fir.select_case label into a boolean guard, nesting into else;
/// cases sharing a destination block (``case (2, 3, 5)``) are OR-joined into one guard.
ASTNode buildSelectCaseChain(fir::SelectCaseOp sel) {
  auto operands = sel.getOperands();
  std::string const xExpr = buildExprWithSubscripts(sel.getSelector(operands), 0);

  auto cases = sel.getCases();
  unsigned const numCases = cases.size();

  // Per-case metadata for a first pass.
  struct CaseInfo {
    bool isDefault = false;
    std::string guard;
    mlir::Block* dest = nullptr;
  };
  std::vector<CaseInfo> infos;
  infos.reserve(numCases);
  for (unsigned i = 0; i < numCases; ++i) {
    CaseInfo ci;
    ci.dest = sel.getSuccessor(i);
    auto tag = cases[i];
    auto cmpOps = sel.getCompareOperands(operands, i);
    if (mlir::isa<mlir::UnitAttr>(tag)) {
      ci.isDefault = true;
    } else if (mlir::isa<fir::PointIntervalAttr>(tag) && cmpOps && !cmpOps->empty()) {
      ci.guard = "(" + xExpr + " == " + buildExprWithSubscripts((*cmpOps)[0], 0) + ")";
    } else if (mlir::isa<fir::ClosedIntervalAttr>(tag) && cmpOps && cmpOps->size() >= 2) {
      auto lo = buildExprWithSubscripts((*cmpOps)[0], 0);
      auto hi = buildExprWithSubscripts((*cmpOps)[1], 0);
      ci.guard = "((";
      ci.guard += xExpr;
      ci.guard += " >= ";
      ci.guard += lo;
      ci.guard += ") and (";
      ci.guard += xExpr;
      ci.guard += " <= ";
      ci.guard += hi;
      ci.guard += "))";
    } else if (mlir::isa<fir::LowerBoundAttr>(tag) && cmpOps && !cmpOps->empty()) {
      ci.guard = "(" + xExpr + " >= " + buildExprWithSubscripts((*cmpOps)[0], 0) + ")";
    } else if (mlir::isa<fir::UpperBoundAttr>(tag) && cmpOps && !cmpOps->empty()) {
      ci.guard = "(" + xExpr + " <= " + buildExprWithSubscripts((*cmpOps)[0], 0) + ")";
    } else {
      // Unknown shape: emit ``False`` so the case is never taken, keeping the chain well-formed.
      ci.guard = "False";
    }
    infos.push_back(std::move(ci));
  }

  // Merge runs of non-default cases sharing the same destination block (Fortran ``case (2, 3, 5)`` -> three fir.point
  // cases targeting the same successor).
  struct Group {
    std::string guard;  // OR-joined guards
    mlir::Block* dest = nullptr;
  };
  std::vector<Group> groups;
  std::vector<ASTNode> defaultBody;
  for (auto& ci : infos) {
    if (ci.isDefault) {
      if (ci.dest) defaultBody = buildAST(*ci.dest);
      continue;
    }
    if (!groups.empty() && groups.back().dest == ci.dest) {
      groups.back().guard += " or " + ci.guard;
    } else {
      Group g;
      g.guard = ci.guard;
      g.dest = ci.dest;
      groups.push_back(std::move(g));
    }
  }

  // Build the nested conditional chain backwards from the last non-default group, folding each into the next one's
  // else.
  ASTNode chain;
  bool first = true;
  for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
    ASTNode node;
    node.kind = "conditional";
    node.condition = "(" + it->guard + ")";
    if (it->dest) node.children = buildAST(*it->dest);
    if (first) {
      node.else_children = defaultBody;
      first = false;
    } else {
      node.else_children.push_back(std::move(chain));
    }
    chain = std::move(node);
  }
  // If every case was defaulted away, fall back to the default body wrapped in a trivial ``if True``.
  if (first) {
    chain.kind = "conditional";
    chain.condition = "True";
    chain.children = defaultBody;
  }
  return chain;
}

/// Extent of a fir.shape/fir.shape_shift operand at dim `d`: prefers a traced declare name, then a literal constant,
/// else `"?"`.
std::string resolveExtent(mlir::Value shape, unsigned d) {
  if (!shape) return "?";
  auto* def = shape.getDefiningOp();
  if (!def) return "?";
  mlir::Value ext;
  if (auto sh = mlir::dyn_cast<fir::ShapeOp>(def)) {
    if (d >= sh.getExtents().size()) return "?";
    ext = sh.getExtents()[d];
  } else if (auto ss = mlir::dyn_cast<fir::ShapeShiftOp>(def)) {
    auto ops = ss->getOperands();
    unsigned const idx = (2 * d) + 1;
    if (idx >= ops.size()) return "?";
    ext = ops[idx];
  } else {
    return "?";
  }
  auto n = traceToDecl(ext);
  if (!n.empty()) return n;
  if (auto c = traceConstInt(ext)) return std::to_string(*c);
  // Flang lowers a section's extent as ``select(sgt(hi-lo+1,0), hi-lo+1, 0)``; peel the clamp to recover the closed
  // form ``(hi - lo + 1)``.
  if (auto* eDef = ext.getDefiningOp())
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(eDef)) ext = sel.getTrueValue();
  auto idx = buildIndexExpr(ext, 0);
  if (!idx.empty() && idx != "?") return "(" + idx + ")";
  return "?";
}

/// Walks the designate parent chain to a (var, expr) list keyed to the underlying array's rank, not the innermost
/// designate's rank -- Flang's intermediate fixed-shape designates otherwise drop the parent's scalar dims.
std::pair<std::string, std::vector<DimEntry>> expandDesignateChain(hlfir::DesignateOp innermost) {
  std::vector<DimEntry> entries;
  auto inIdxs = innermost.getIndices();
  for (unsigned d = 0; d < inIdxs.size(); ++d) {
    auto idx = inIdxs[d];
    auto n = resolveIndex(idx);
    entries.push_back({n.empty() ? "?" : n, buildDesignateIndexExpr(innermost, d, idx, 0)});
  }

  // Set when a parent designate carries the component attr (e.g. ``vcut % corrected(i,j,k)``):
  // innermost.getComponentAttr() is empty even though the access is a flattened struct member, so the flat name needs
  // traceToDecl on the innermost RESULT instead.
  bool sawComponentParent = false;
  // Per-block AoR section hop (e.g. ``vec_in(jc, jk) % x(k)`` over ``p_vn(:, :, blockno)``): composes the section
  // positionally into [jc, jk, blockno, k]. Do NOT use the generic triplet loop below -- it drops the member tail.
  auto applySectionHop = [](hlfir::DesignateOp sec, std::vector<DimEntry>& entries) {
    auto trips = sec.getIsTriplet();
    unsigned T = 0;
    for (bool const t : trips)
      if (t) ++T;
    size_t const split = std::min<size_t>(T, entries.size());
    std::vector<DimEntry> head(entries.begin(), entries.begin() + split);
    std::vector<DimEntry> tail(entries.begin() + split, entries.end());
    std::vector<DimEntry> newRec;
    size_t hi = 0;
    size_t cur = 0;
    auto sidx = sec.getIndices();
    for (bool const trip : trips) {
      if (trip) {
        newRec.push_back(hi < head.size() ? head[hi] : DimEntry{"?", "?"});
        ++hi;
        cur += 3;  // triplet subscript = (lo, hi, step)
      } else {
        mlir::Value const s = cur < sidx.size() ? sidx[cur] : mlir::Value{};
        std::string const n = s ? resolveIndex(s) : std::string{};
        newRec.push_back({n.empty() ? "?" : n, s ? buildIndexExpr(s, 0) : "?"});
        cur += 1;  // scalar subscript = (idx)
      }
    }
    entries = std::move(newRec);
    entries.insert(entries.end(), tail.begin(), tail.end());
  };
  mlir::Value parent_val = innermost.getMemref();
  for (int level = 0; level < limits::kAliasMemrefWalkDepth; ++level) {
    if (!parent_val) break;
    auto* def = parent_val.getDefiningOp();
    if (!def) break;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      parent_val = cv.getValue();
      continue;
    }
    // Inlined AoR-section dummy parent (``vec_in``, a copy_in declare): the parent-walk would otherwise BREAK at this
    // DeclareOp and drop ``blockno`` -- fire the section hop and continue root-ward.
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      if (auto sec = asInlinedSectionOverComponent(dc)) {
        applySectionHop(sec, entries);
        parent_val = sec.getMemref();
        continue;
      }
      // Inlined-call dummy on a struct-member actual (gate #12): traceToDecl/walkMemberChain hop through to the caller
      // chain via leadsToComponentDesignate, so the parent-walk must hop too or the record subscript is dropped and the
      // memlet under-ranks.
      if (dc.getDummyScope() && leadsToComponentDesignate(dc.getMemref())) {
        parent_val = dc.getMemref();
        continue;
      }
      break;
    }
    // Pointer/allocatable deref: ``fir.load`` between two designates (e.g. ``arr(ia) % box(ir)`` where ``box`` is
    // allocatable). Walk through the load or the chain breaks and record-index ``ia`` is lost.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      parent_val = ld.getMemref();
      continue;
    }
    auto parent = mlir::dyn_cast<hlfir::DesignateOp>(def);
    if (!parent) break;
    // Only a whole-member selector with no index of its own (e.g. ``vcut % corrected``) needs sawComponentParent; a
    // component designate that also carries a record index (``arr(ia) % box(ir)``) keeps the existing parent_val
    // resolution -- widening this to every component parent breaks that case's memlet subset.
    if (parent.getComponentAttr() && parent.getIndices().empty()) sawComponentParent = true;
    // Direct AoR section parent (fully inlined, no ``vec_in`` declare): same positional compose as the copy_in-declare
    // case above; the generic triplet loop below would drop the member tail.
    if (auto sec = asSectionOverComponent(parent.getResult())) {
      applySectionHop(sec, entries);
      parent_val = sec.getMemref();
      continue;
    }
    auto triplets = parent.getIsTriplet();
    if (triplets.empty()) {
      // Element designate (every dim scalar): plain element access overwrites with the inner indices; AoR access
      // (``arr(i) % x(2)``) PREPENDs the parent's RECORD index ahead of the inner's FIELD index, matching the flatten
      // pass's ``[N_records, N_field_elems]`` layout.
      std::vector<DimEntry> parent_entries;
      auto pidxOps = parent.getIndices();
      for (auto idx : pidxOps) {
        auto n = resolveIndex(idx);
        parent_entries.push_back({n.empty() ? "?" : n, buildIndexExpr(idx, 0)});
      }
      // PREPEND the parent's record indices for a struct-field chain (innermost.getComponentAttr() or
      // sawComponentParent); otherwise overwrite, for plain nested-element access on a non-struct array.
      if (innermost.getComponentAttr() || sawComponentParent) {
        std::vector<DimEntry> combined = std::move(parent_entries);
        for (auto& e : entries) combined.push_back(std::move(e));
        entries = std::move(combined);
      } else {
        entries = std::move(parent_entries);
      }
      parent_val = parent.getMemref();
      continue;
    }
    std::vector<DimEntry> new_entries;
    size_t inner_i = 0;
    unsigned cursor = 0;
    auto pidxOps = parent.getIndices();
    for (bool const triplet : triplets) {
      if (triplet) {
        if (inner_i < entries.size())
          new_entries.push_back(entries[inner_i]);
        else
          new_entries.push_back({"?", "?"});
        ++inner_i;
        cursor += 3;
      } else {
        if (cursor < pidxOps.size()) {
          auto s = pidxOps[cursor];
          auto n = resolveIndex(s);
          new_entries.push_back({n.empty() ? "?" : n, buildIndexExpr(s, 0)});
        } else {
          new_entries.push_back({"?", "?"});
        }
        cursor += 1;
      }
    }
    // AoR shape with parent triplets all-false (Flang's runtime-indexed element-designate): concatenate record-first,
    // mirroring the empty-triplet AoR branch above; sawComponentParent extends this to the pointer/allocatable member
    // case.
    bool allScalar = true;
    for (bool const triplet : triplets)
      if (triplet) {
        allScalar = false;
        break;
      }
    if ((innermost.getComponentAttr() || sawComponentParent) && allScalar) {
      std::vector<DimEntry> combined = std::move(new_entries);
      for (auto& e : entries) combined.push_back(std::move(e));
      entries = std::move(combined);
    } else {
      entries = std::move(new_entries);
    }
    parent_val = parent.getMemref();
  }

  // For AoR/nested component chains, prefer traceToDecl on the innermost RESULT (accumulates the full flat name); falls
  // back to parent_val when no component is involved. sawComponentParent extends this to a component sitting on a
  // PARENT designate, not the innermost.
  std::string array_name;
  if (innermost.getComponentAttr() || sawComponentParent) {
    array_name = traceToDecl(innermost.getResult());
  }
  if (array_name.empty()) array_name = traceToDecl(parent_val);
  if (array_name.empty()) array_name = traceToDecl(innermost.getMemref());
  return {std::move(array_name), std::move(entries)};
}

/// Appends one ``AccessInfo`` per data read in a yielded SSA tree; shared by
/// buildElementalAssign/CountLibcall/AnyAllReduce so all three use the same expandDesignateChain-based index expansion
/// (parent-section scalar dims aren't silently dropped).
void collectReadAccesses(mlir::Value v, std::vector<AccessInfo>& accesses, int depth) {
  if (depth > 40) return;
  auto* op = v.getDefiningOp();
  if (!op) return;
  // ``fir.box_dims`` reads only shape/bounds metadata (already rendered as symbols); recursing into its operand would
  // record a spurious rank-1 whole-array read that fails sdfg.validate. Stop the walk here.
  if (op->getName().getStringRef() == "fir.box_dims") return;
  // A reduction materialised into a scalar by materialiseCondReductions is owned by its Reduce lib-node, not the
  // condition -- skip it here.
  if (kCondReductionScalars.find(op) != kCondReductionScalars.end()) return;
  if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
    auto [arr, dims] = expandDesignateChain(dg);
    AccessInfo ra;
    ra.array_name = arr;
    ra.is_read = true;
    for (auto& de : dims) {
      ra.index_vars.push_back(de.var);
      ra.index_exprs.push_back(de.expr);
    }
    accesses.push_back(std::move(ra));
    for (auto idx : dg.getIndices()) collectReadAccesses(idx, accesses, depth + 1);
    return;
  }
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(op)) {
    auto mem = ld.getMemref();
    if (auto* md = mem.getDefiningOp())
      if (mlir::isa<hlfir::DeclareOp>(md)) {
        AccessInfo ra;
        ra.array_name = traceToDecl(mem);
        ra.is_read = true;
        accesses.push_back(std::move(ra));
        return;
      }
  }
  if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
    auto src = apply.getExpr();
    if (auto* sd = src.getDefiningOp()) {
      auto it = kHlfirExprToTransient.find(sd);
      if (it != kHlfirExprToTransient.end()) {
        AccessInfo ra;
        ra.array_name = it->second;
        ra.is_read = true;
        for (auto idx : apply.getIndices()) {
          auto n = resolveIndex(idx);
          std::string const s = n.empty() ? std::string("?") : n;
          ra.index_vars.push_back(s);
          ra.index_exprs.push_back(s);
        }
        accesses.push_back(std::move(ra));
        return;
      }
      if (auto inner_elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
        auto& ireg = inner_elem.getRegion();
        if (!ireg.empty()) {
          auto& iblock = ireg.front();
          auto apply_idxs = apply.getIndices();
          unsigned pushed = 0;
          for (unsigned i = 0; i < iblock.getNumArguments() && i < apply_idxs.size(); ++i) {
            auto name = resolveIndex(apply_idxs[i]);
            indexStack().emplace_back(iblock.getArgument(i), name);
            ++pushed;
          }
          for (auto& iop : iblock)
            if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(iop))
              collectReadAccesses(y.getElementValue(), accesses, depth + 1);
          for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();
        }
      }
    }
    return;
  }
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
    // scf.if-valued expr reads live in the THEN/ELSE scf.yield values, not reachable via the generic operand recursion
    // below; mirror buildBoolExpr's scf.if descent (see tests/lu_two_call_convergence_repro_test.py).
    collectReadAccesses(ifOp.getCondition(), accesses, depth + 1);
    for (auto* reg : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (reg->empty()) continue;
      for (auto& iop : reg->front())
        if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(iop))
          for (auto yv : y.getResults()) collectReadAccesses(yv, accesses, depth + 1);
    }
    return;
  }
  for (auto operand : op->getOperands()) collectReadAccesses(operand, accesses, depth + 1);
}

/// Maps an hlfir expr-producing op to its FaCe libcall callee tag (matmul/transpose/dot_product); nullptr if unknown.
const char* libcallNameForExprOp(mlir::Operation* op) {
  if (!op) return nullptr;
  auto name = op->getName().getStringRef();
  // Must stay in sync with the libcall dispatcher's kLibTable (dispatch.cpp:2082); a missing entry here makes buildExpr
  // fall through and emit ``?`` for the apply.
  if (name == "hlfir.matmul") return "matmul";
  if (name == "hlfir.transpose") return "transpose";
  if (name == "hlfir.dot_product") return "dot_product";
  // ``hlfir.matmul_transpose`` is the fused MATMUL(TRANSPOSE(A), B) op that hlfir-optimized-bufferization synthesises
  // from separate matmul + transpose.
  if (name == "hlfir.matmul_transpose") return "matmul_transpose";
  // Fortran COUNT(mask [, dim]) -> CountLibraryNode; this entry covers COUNT as an inline operand
  // (dispatch.cpp::buildElementalCountLibcall handles the whole-assign case).
  if (name == "hlfir.count") return "count";
  // ``MINLOC`` / ``MAXLOC`` -- ArgMin / ArgMax library nodes.
  if (name == "hlfir.minloc") return "argmin";
  if (name == "hlfir.maxloc") return "argmax";
  // ``CSHIFT(array, shift [, dim])`` -- circular shift.
  if (name == "hlfir.cshift") return "cshift";
  return nullptr;
}

/// Per-dim shape strings from an hlfir.expr type: static dims -> decimal literal, ``?`` extents stay ``?`` for
/// descriptors.py to fill from a synthetic ``<name>_d<i>`` symbol.
std::vector<std::string> exprResultShape(mlir::Type ty) {
  std::vector<std::string> out;
  if (auto e = mlir::dyn_cast<hlfir::ExprType>(ty)) {
    for (int64_t const d : e.getShape()) {
      if (d == hlfir::ExprType::getUnknownExtent())
        out.emplace_back("?");
      else
        out.push_back(std::to_string(d));
    }
  }
  return out;
}

/// Maps an hlfir.expr element type to FaCe's dtype string, default float64; i1 (COUNT/ANY/ALL boolean masks) maps to
/// int32 since CountLibraryNode/Reduce.atomic expect 0/1 ints.
std::string exprDtypeString(mlir::Type ty) {
  if (auto e = mlir::dyn_cast<hlfir::ExprType>(ty)) {
    auto elt = e.getElementType();
    if (elt.isF64()) return "float64";
    if (elt.isF32()) return "float32";
    if (elt.isInteger(1)) return "int32";
    if (elt.isInteger(32)) return "int32";
    if (elt.isInteger(64)) return "int64";
  }
  return "float64";
}

/// Lowers an hlfir.elemental body (per-element predicate/expression) into a (declare_transient,
/// loop-chain-filling-transient) pair; used by buildElementalCountLibcall (terminal: count libcall) and
/// buildElementalAnyAllReduce (terminal: Reduce node) as the source for a terminal reduction/libcall. Caller picks the
/// transient-name prefix and the terminal node.
static std::pair<std::string, std::vector<ASTNode>> materialiseElementalToTransient(hlfir::ElementalOp elem,
                                                                                    std::string_view prefix) {
  auto& region = elem.getRegion();
  if (region.empty()) return {{}, {}};
  auto& block = region.front();
  unsigned const rank = block.getNumArguments();
  auto shape = elem.getShape();

  std::string trName = std::string(prefix) + std::to_string(kSynthTransientCounter++);

  // Dtype follows the elemental's result type via exprDtypeString (i1 boolean masks -> int32).
  std::string const dtype = exprDtypeString(elem.getType());

  ASTNode decl;
  decl.kind = "declare_transient";
  decl.target = trName;
  decl.expr = dtype;
  AccessInfo shape_info;
  shape_info.array_name = trName;
  for (unsigned i = 0; i < rank; ++i) shape_info.index_exprs.push_back(resolveExtent(shape, i));
  decl.accesses.push_back(std::move(shape_info));

  std::vector<std::string> iter_names;
  iter_names.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) iter_names.push_back("ei" + std::to_string(i));
  unsigned pushed = 0;
  for (unsigned i = 0; i < rank; ++i) {
    indexStack().emplace_back(block.getArgument(i), iter_names[i]);
    ++pushed;
  }

  ASTNode inner;
  inner.kind = "assign";
  inner.target = trName;
  inner.target_is_array = true;
  AccessInfo wa;
  wa.array_name = trName;
  wa.is_write = true;
  for (unsigned i = 0; i < rank; ++i) {
    wa.index_vars.push_back(iter_names[i]);
    wa.index_exprs.push_back(iter_names[i]);
  }
  inner.accesses.push_back(std::move(wa));

  mlir::Value yielded;
  for (auto& op : block)
    if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
      yielded = y.getElementValue();
      break;
    }

  // Pre-walk for dim-reductions on the apply chain (``hlfir.apply %sum_result`` where ``%sum_result = hlfir.sum
  // %inner_elem dim %k``): without pre-materialisation buildExpr returns ``?``. Mirrors findApplies
  // (control_flow.cpp:289+): materialise into a ``_libtmp_`` transient and register it in kHlfirExprToTransient.
  std::vector<ASTNode> reductionPreNodes;
  auto reduceWcrIdentity = [](mlir::Operation* op, std::string& wcr, std::string& identity) -> bool {
    auto nm = op->getName().getStringRef();
    if (nm == "hlfir.sum") {
      wcr = "lambda a, b: a + b";
      identity = "0";
      return true;
    }
    if (nm == "hlfir.product") {
      wcr = "lambda a, b: a * b";
      identity = "1";
      return true;
    }
    if (nm == "hlfir.minval") {
      wcr = "lambda a, b: min(a, b)";
      identity = "inf";
      return true;
    }
    if (nm == "hlfir.maxval") {
      wcr = "lambda a, b: max(a, b)";
      identity = "-inf";
      return true;
    }
    return false;
  };
  if (yielded) {
    std::function<void(mlir::Value, int)> walkForReductions = [&](mlir::Value v, int depth) {
      if (depth > 40 || !v) return;
      auto* op = v.getDefiningOp();
      if (!op) return;
      if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(op)) {
        auto src = apply.getExpr();
        auto* srcOp = src.getDefiningOp();
        if (!srcOp) return;
        if (kHlfirExprToTransient.count(srcOp)) return;
        std::string wcr;
        std::string identity;
        if (!reduceWcrIdentity(srcOp, wcr, identity)) {
          // Not a reduction: recurse into operands, AND descend into a nested elemental's body too -- e.g.
          // ``SUM(LOG(SUM(a,1)+1.0))`` has the inner sum-dim inside the ``+1.0`` elemental's body, not among this
          // apply's operands; without descending, buildExpr later renders it as ``?``.
          if (auto innerElem = mlir::dyn_cast<hlfir::ElementalOp>(srcOp)) {
            auto& ireg = innerElem.getRegion();
            if (!ireg.empty())
              for (auto& iop : ireg.front())
                if (auto iy = mlir::dyn_cast<hlfir::YieldElementOp>(iop))
                  walkForReductions(iy.getElementValue(), depth + 1);
          }
          for (auto operand : op->getOperands()) walkForReductions(operand, depth + 1);
          return;
        }
        // Materialise the inner source: elemental -> materialiseElementalForLibcall transient; named array ->
        // traceToDecl directly.
        mlir::Value const redSrc = srcOp->getOperand(0);
        std::string redSrcName;
        std::vector<ASTNode> srcMaterialNodes;
        if (auto* rsd = redSrc.getDefiningOp()) {
          if (auto innerElem = mlir::dyn_cast<hlfir::ElementalOp>(rsd)) {
            auto [trName, mat_nodes] = materialiseElementalForLibcall(innerElem);
            if (!trName.empty()) {
              redSrcName = std::move(trName);
              for (auto& mn : mat_nodes) srcMaterialNodes.push_back(std::move(mn));
            }
          }
        }
        if (redSrcName.empty()) redSrcName = traceToDecl(redSrc);
        if (redSrcName.empty()) return;  // can't materialise; skip
        // Mint the reduction's result transient.
        std::string const tmp = "_libtmp_" + std::to_string(kLibTmpCounter++);
        kHlfirExprToTransient[srcOp] = tmp;
        mlir::Type const rty = srcOp->getResult(0).getType();
        auto rshape = exprResultShape(rty);
        ASTNode decl;
        decl.kind = "declare_transient";
        decl.target = tmp;
        decl.expr = exprDtypeString(rty);
        decl.target_is_array = !rshape.empty();
        AccessInfo shapeInfo;
        shapeInfo.array_name = tmp;
        for (auto& s : rshape) shapeInfo.index_exprs.push_back(s);
        decl.accesses.push_back(std::move(shapeInfo));
        // Reduce AST node.
        ASTNode red;
        red.kind = "reduce";
        red.target = tmp;
        red.target_is_array = !rshape.empty();
        red.reduce_src = redSrcName;
        red.reduce_wcr = wcr;
        red.reduce_identity = identity;
        // ``dim`` is the second operand (1-based Fortran -> 0-based).
        if (srcOp->getNumOperands() >= 2) {
          auto dimV = srcOp->getOperand(1);
          if (auto c = traceConstInt(dimV)) red.reduce_axes.push_back(*c - 1);
        }
        // Pre-nodes ordering: source materialisation, then declare_transient, then the reduce that writes to it.
        for (auto& n : srcMaterialNodes) reductionPreNodes.push_back(std::move(n));
        reductionPreNodes.push_back(std::move(decl));
        reductionPreNodes.push_back(std::move(red));
        // Continue recursion in case of chained reductions deeper in the tree.
        for (auto operand : op->getOperands()) walkForReductions(operand, depth + 1);
        return;
      }
      // Plain non-apply op -- recurse into all operands.
      for (auto operand : op->getOperands()) walkForReductions(operand, depth + 1);
    };
    walkForReductions(yielded, 0);
  }

  std::string body = "?";
  if (yielded) {
    // Tasklet-body mode: comparisons/loads emit bare names so emit_tasklet's per-occurrence connector wiring picks them
    // up.
    NoSubscriptGuard const g;
    std::string b = buildBoolExpr(yielded, 0);
    if (b == "?") b = buildExpr(yielded, 0);
    body = b;
  }
  // Wrap in ``dace.int32(...)`` only when the elemental yields i1 (COUNT/ANY/ALL mask path); casting
  // SUM/PRODUCT/MINVAL/MAXVAL of real values would truncate and break the reduction.
  bool isI1Yield = false;
  if (yielded && yielded.getType().isInteger(1)) isI1Yield = true;
  inner.expr = isI1Yield ? ("int32(" + body + ")") : body;

  if (yielded) collectReadAccesses(yielded, inner.accesses, 0);

  for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();

  ASTNode current = inner;
  for (int i = (int)rank - 1; i >= 0; --i) {
    ASTNode wrap;
    wrap.kind = "loop";
    wrap.loop_iter = iter_names[i];
    wrap.loop_lower = 1;
    wrap.loop_bound = resolveExtent(shape, i);
    wrap.children.push_back(current);
    current = wrap;
  }

  std::vector<ASTNode> nodes;
  nodes.reserve(2 + reductionPreNodes.size());
  // Reduction pre-materialisation nodes go first: they declare the transients the elemental body's apply reads from.
  for (auto& n : reductionPreNodes) nodes.push_back(std::move(n));
  nodes.push_back(std::move(decl));
  nodes.push_back(std::move(current));
  return {std::move(trName), std::move(nodes)};
}

/// Materialises an hlfir.elemental into a synthetic transient (own dtype) for a libcall
/// (transpose/matmul/dot_product/...) to read, since emit_library needs a named SDFG array, not an unnamed hlfir.expr.
/// Returns empty AST_nodes on failure so the caller falls back to the original libcall.
std::pair<std::string, std::vector<ASTNode>> materialiseElementalForLibcall(hlfir::ElementalOp elem) {
  auto& region = elem.getRegion();
  if (region.empty()) return {{}, {}};
  auto& block = region.front();
  unsigned const rank = block.getNumArguments();
  auto shape = elem.getShape();
  if (!shape) return {{}, {}};

  std::string trName = "_libsrc_" + std::to_string(kSynthTransientCounter++);
  std::string const dtype = exprDtypeString(elem.getType());

  ASTNode decl;
  decl.kind = "declare_transient";
  decl.target = trName;
  decl.expr = dtype;
  AccessInfo shape_info;
  shape_info.array_name = trName;
  for (unsigned i = 0; i < rank; ++i) shape_info.index_exprs.push_back(resolveExtent(shape, i));
  decl.accesses.push_back(std::move(shape_info));

  std::vector<std::string> iter_names;
  iter_names.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) iter_names.push_back("li" + std::to_string(i));
  unsigned pushed = 0;
  for (unsigned i = 0; i < rank; ++i) {
    indexStack().emplace_back(block.getArgument(i), iter_names[i]);
    ++pushed;
  }

  ASTNode inner;
  inner.kind = "assign";
  inner.target = trName;
  inner.target_is_array = true;
  AccessInfo wa;
  wa.array_name = trName;
  wa.is_write = true;
  for (unsigned i = 0; i < rank; ++i) {
    wa.index_vars.push_back(iter_names[i]);
    wa.index_exprs.push_back(iter_names[i]);
  }
  inner.accesses.push_back(std::move(wa));

  mlir::Value yielded;
  for (auto& op : block)
    if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
      yielded = y.getElementValue();
      break;
    }
  if (!yielded) {
    for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();
    return {{}, {}};
  }

  std::string body;
  {
    NoSubscriptGuard const g;
    body = buildExpr(yielded, 0);
  }
  if (body == "?") {
    for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();
    return {{}, {}};
  }
  inner.expr = body;
  collectReadAccesses(yielded, inner.accesses, 0);

  for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();

  ASTNode current = inner;
  for (int i = (int)rank - 1; i >= 0; --i) {
    ASTNode wrap;
    wrap.kind = "loop";
    wrap.loop_iter = iter_names[i];
    wrap.loop_lower = 1;
    wrap.loop_bound = resolveExtent(shape, i);
    wrap.children.push_back(current);
    current = wrap;
  }

  std::vector<ASTNode> nodes;
  nodes.reserve(2);
  nodes.push_back(std::move(decl));
  nodes.push_back(std::move(current));
  return {std::move(trName), std::move(nodes)};
}

std::vector<ASTNode> buildElementalCountLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem) {
  auto [trName, nodes] = materialiseElementalToTransient(elem, "_count_mask_");
  if (nodes.empty()) return {};

  ASTNode lib;
  lib.kind = "libcall";
  lib.callee = "count";
  auto dest = assign.getOperand(1);
  captureElementDesignateWrite(dest, lib);
  if (!lib.target_is_array) lib.target_is_array = isArrayRef(dest.getType());
  lib.call_args.push_back(trName);

  nodes.push_back(std::move(lib));
  return nodes;
}

/// ANY(...)/ALL(...): same materialise-then-reduce shape as buildElementalCountLibcall, but the final node is a reduce
/// over the transient instead of the CountLibraryNode libcall.
std::vector<ASTNode> buildElementalAnyAllReduce(hlfir::AssignOp assign, hlfir::ElementalOp elem, std::string_view wcr,
                                                std::string_view identity) {
  auto [trName, nodes] = materialiseElementalToTransient(elem, "_mask_");
  if (nodes.empty()) return {};

  ASTNode red;
  red.kind = "reduce";
  captureElementDesignateWrite(assign.getOperand(1), red);
  red.reduce_src = trName;
  red.reduce_wcr = wcr;
  red.reduce_identity = identity;

  nodes.push_back(std::move(red));
  return nodes;
}

}  // namespace hlfir_bridge
