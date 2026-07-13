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
// Reduction + elemental-libcall builders.  Owns:
//   * buildReduceNode (sum / product / minval / maxval / any / all).
//   * buildElementalCountLibcall  --  the Mode-C COUNT path that
//     synthesises a transient mask from a comparison-as-elemental.
//   * buildSelectCaseChain  --  fir.select_case -> nested conditionals.
//   * resolveExtent, libcallNameForExprOp, exprResultShape,
//     exprDtypeString  --  small helpers used by reductions /
//     libcall-in-elemental materialisation.
//   * Thread-local counters: kSynthTransientCounter, kLibTmpCounter,
//     kBoolExprNoSubscripts.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/elementals.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list  --  CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
ASTNode buildReduceNode(hlfir::AssignOp assign, mlir::Operation* redOp, std::string_view wcr,
                        std::string_view identity) {
  ASTNode n;
  n.kind = "reduce";

  captureElementDesignateWrite(assign.getOperand(1), n);

  // Source array  --  operand 0 of the reduction op.
  if (redOp->getNumOperands() > 0) n.reduce_src = traceToDecl(redOp->getOperand(0));
  n.reduce_wcr = wcr;
  n.reduce_identity = identity;

  // ``hlfir.sum %arr dim %d``  --  trace the second operand.  When absent
  // (whole-array reduction) leave reduce_axes empty.
  if (redOp->getNumOperands() >= 2) {
    auto d = redOp->getOperand(1);
    if (auto c = traceConstInt(d))
      // Fortran ``dim`` is 1-based; DaCe axes are 0-based.
      n.reduce_axes.push_back(*c - 1);
  }
  return n;
}

/// Forward declare  --  called from buildWhileNode to recurse into the body.
std::vector<ASTNode> buildAST(mlir::Block& block);

/// Synthesise a chain of nested ``kind="conditional"`` AST nodes from a
/// ``fir.select_case`` terminator.  Fortran ``SELECT CASE`` has no direct
/// equivalent in DaCe's control-flow vocabulary, so we fold every case
/// label into a boolean guard and nest the rest in the ``else`` branch.
///
/// Case labels supported (from FIROps.td):
///   - ``#fir.point %v``       -> ``x == v``
///   - ``#fir.interval %l %h`` -> ``(x >= l) and (x <= h)``
///   - ``#fir.lower %l``       -> ``x >= l``
///   - ``#fir.upper %h``       -> ``x <= h``
///   - ``unit``                -> default (else at the innermost nesting)
///
/// Adjacent cases targeting the same successor block (``case (2, 3, 5)``
/// lowers to three ``fir.point`` cases all pointing at the same ``^bb``)
/// collapse into a single guard whose sub-predicates are OR-joined.
ASTNode buildSelectCaseChain(fir::SelectCaseOp sel) {
  auto operands = sel.getOperands();
  std::string xExpr = buildExprWithSubscripts(sel.getSelector(operands), 0);

  auto cases = sel.getCases();
  unsigned numCases = cases.size();

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
      ci.guard = "((" + xExpr + " >= " + lo + ") and (" + xExpr + " <= " + hi + "))";
    } else if (mlir::isa<fir::LowerBoundAttr>(tag) && cmpOps && !cmpOps->empty()) {
      ci.guard = "(" + xExpr + " >= " + buildExprWithSubscripts((*cmpOps)[0], 0) + ")";
    } else if (mlir::isa<fir::UpperBoundAttr>(tag) && cmpOps && !cmpOps->empty()) {
      ci.guard = "(" + xExpr + " <= " + buildExprWithSubscripts((*cmpOps)[0], 0) + ")";
    } else {
      // Unknown shape  --  emit ``False`` so the case is never taken,
      // keeping the rest of the chain well-formed.
      ci.guard = "False";
    }
    infos.push_back(std::move(ci));
  }

  // Merge runs of non-default cases sharing the same destination block
  // (Fortran ``case (2, 3, 5)`` -> three fir.point cases all targeting
  // the same successor).
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

  // Build the nested conditional chain from the last non-default group
  // backwards, folding each previous group into the next one's else.
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
  // If every case was defaulted away (no non-default labels), fall back
  // to the default body as-is wrapped in a trivial ``if True``.
  if (first) {
    chain.kind = "conditional";
    chain.condition = "True";
    chain.children = defaultBody;
  }
  return chain;
}

/// Resolve the extent of a fir.shape / fir.shape_shift operand at dim `d`,
/// preferring a traced declare name (`"nproma"`), then a literal constant
/// (`"10"`), and falling back to `"?"` if neither is available.
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
    unsigned idx = 2 * d + 1;
    if (idx >= ops.size()) return "?";
    ext = ops[idx];
  } else {
    return "?";
  }
  auto n = traceToDecl(ext);
  if (!n.empty()) return n;
  if (auto c = traceConstInt(ext)) return std::to_string(*c);
  // Flang lowers a section ``a(lo:hi)``'s materialised extent as
  // ``select(sgt(hi - lo + 1, 0), hi - lo + 1, 0)``  --  peel that
  // clamp so we recover the closed-form ``(hi - lo + 1)``, which
  // itself becomes a closed-form expression after ``buildIndexExpr``
  // promotes ``hi`` / ``lo`` to position symbols (``__sym_pos_N``).
  if (auto* eDef = ext.getDefiningOp())
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(eDef)) ext = sel.getTrueValue();
  auto idx = buildIndexExpr(ext, 0);
  if (!idx.empty() && idx != "?") return "(" + idx + ")";
  return "?";
}

/// Walk an innermost ``hlfir.designate``'s parent chain and produce
/// a per-original-dim (var, expr) list keyed to the underlying array's
/// rank  --  not to the innermost designate's rank.  Required when Flang
/// materialises a section as an intermediate fixed-shape designate
/// (``%inner = designate %m (%c1:%c7:%c1, %pos1)``) and an inner
/// elemental indexes only the surviving triplet dim
/// (``%elem_acc = designate %inner (%arg3)``): the access-collection
/// path otherwise produces a rank-1 access list (just ``[arg3]``)
/// while the underlying array (``m``) is rank 2 and needs the
/// parent's ``pos1`` scalar to occupy dim 1.
///
/// Per parent level:
///   * triplet dim -> consume one inner-iter entry (already rebased
///     by ``buildDesignateIndexExpr`` against the parent's lo);
///   * scalar dim  -> render the parent's scalar via ``buildIndexExpr``
///     and insert at the parent's dim position.
std::pair<std::string, std::vector<DimEntry>> expandDesignateChain(hlfir::DesignateOp innermost) {
  std::vector<DimEntry> entries;
  auto inIdxs = innermost.getIndices();
  for (unsigned d = 0; d < inIdxs.size(); ++d) {
    auto idx = inIdxs[d];
    auto n = resolveIndex(idx);
    entries.push_back({n.empty() ? "?" : n, buildDesignateIndexExpr(innermost, d, idx, 0)});
  }

  // Set when ANY parent designate in the walk carries a component
  // attribute (struct-field selector).  ``vcut % corrected(i,j,k)`` puts
  // the component on the PARENT (``designate{component=corrected}``) and
  // the element indices on the INNERMOST -- so ``innermost.getComponentAttr()``
  // is empty even though the access IS a flattened struct member.  The
  // flat name (``vcut_corrected``) only comes from ``traceToDecl`` on the
  // innermost RESULT, which walks the whole chain; record that we need it.
  bool sawComponentParent = false;
  // Per-block AoR section hop (``vec_in`` over ``p_vn(:, :, blockno)``, read
  // ``vec_in(jc, jk) % x(k)``): compose the section positionally into the
  // access.  ``entries`` so far = [record head (jc, jk) paired with the
  // section's TRIPLET dims] ++ [member tail (k)].  Each triplet dim is filled
  // by the next head entry (already rebased against the section lo by
  // ``buildDesignateIndexExpr``); each SCALAR dim (``blockno``) is inserted at
  // its record position from the section's own subscript -- yielding
  // [jc, jk, blockno, k] over the rank-4 companion.  Do NOT use the generic
  // triplet loop below (it consumes only the triplet count and drops the
  // member tail).
  auto applySectionHop = [](hlfir::DesignateOp sec, std::vector<DimEntry>& entries) {
    auto trips = sec.getIsTriplet();
    unsigned T = 0;
    for (bool t : trips)
      if (t) ++T;
    size_t split = std::min<size_t>(T, entries.size());
    std::vector<DimEntry> head(entries.begin(), entries.begin() + split);
    std::vector<DimEntry> tail(entries.begin() + split, entries.end());
    std::vector<DimEntry> newRec;
    size_t hi = 0, cur = 0;
    auto sidx = sec.getIndices();
    for (unsigned d = 0; d < trips.size(); ++d) {
      if (trips[d]) {
        newRec.push_back(hi < head.size() ? head[hi] : DimEntry{"?", "?"});
        ++hi;
        cur += 3;  // triplet subscript = (lo, hi, step)
      } else {
        mlir::Value s = cur < sidx.size() ? sidx[cur] : mlir::Value{};
        std::string n = s ? resolveIndex(s) : std::string{};
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
    // Inlined AoR-section dummy parent (``vec_in`` -- a copy_in declare):
    // the parent-walk otherwise BREAKS at this DeclareOp, dropping ``blockno``.
    // Fire the section hop and continue root-ward to the ``p_vn`` component.
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      if (auto sec = asInlinedSectionOverComponent(dc)) {
        applySectionHop(sec, entries);
        parent_val = sec.getMemref();
        continue;
      }
      // Inlined-call dummy bound to a struct-MEMBER actual (gate #12): after
      // ``hlfir-inline-all`` folds a caller ``ptr => patch_3d % p_patch_2d(1) %
      // cells % owned`` alias, the callee dummy (``in_subset``) declares straight
      // onto that caller designate chain -- which carries a record-array index
      // (``p_patch_2d(1)``) the dummy's own scope hides.  ``traceToDecl`` (flat
      // name) and ``walkMemberChain`` (companion registration) both hop THROUGH to
      // the caller chain via ``leadsToComponentDesignate``, so both count that
      // record index; the parent-walk must hop too, or the record subscript is
      // dropped and the memlet under-ranks its descriptor (the 2-D subset vs 3-D
      // ``patch_3d_p_patch_2d_cells_owned_vertical_levels`` mismatch).  A member
      // chain with no record-array index (``patch % edges % in_domain``) prepends
      // nothing, so this is inert for the ordinary gate-#12 cases.
      if (dc.getDummyScope() && leadsToComponentDesignate(dc.getMemref())) {
        parent_val = dc.getMemref();
        continue;
      }
      break;
    }
    // Pointer / allocatable dereference: ``fir.load`` between two
    // designates (QE L4 ``arr(ia) % box(ir)`` shape -- the inner
    // ``box`` member is allocatable, so its access path is
    // ``designate %component ; fir.load ; designate %element``).
    // Walk through the load to find the underlying designate so
    // the AoR chain stays connected.  Without this the walk
    // breaks at the load and the record-index ``ia`` is lost
    // from the AccessInfo.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      parent_val = ld.getMemref();
      continue;
    }
    auto parent = mlir::dyn_cast<hlfir::DesignateOp>(def);
    if (!parent) break;
    // Only a WHOLE-MEMBER component selector (``vcut % corrected`` -- the
    // pointer/allocatable member with no index of its own; the element
    // subscript lives on the innermost designate, reached through the
    // pointer-box ``fir.load``) needs the flat name recovered from the
    // innermost RESULT.  An AoR component designate that ALSO carries a
    // record index (``arr(ia) % box(ir)`` -- parent indices ``[ia]``)
    // keeps the existing ``parent_val`` name resolution; widening the
    // flag to every component parent reshapes that L4 access and breaks
    // its memlet subset.
    if (parent.getComponentAttr() && parent.getIndices().empty()) sawComponentParent = true;
    // DIRECT AoR section parent (worker fully inlined -- no ``vec_in`` declare;
    // ``p_vn(:, :, blockno)`` indexed straight off the ``pvn3d`` alias).  Same
    // positional compose as the copy_in-declare case above; the generic triplet
    // loop below would drop the member tail (``k``).
    if (auto sec = asSectionOverComponent(parent.getResult())) {
      applySectionHop(sec, entries);
      parent_val = sec.getMemref();
      continue;
    }
    auto triplets = parent.getIsTriplet();
    if (triplets.empty()) {
      // Element designate (every dim is a scalar).  Two shapes
      // bottom out here:
      //
      //   * Plain element access of an array (no further struct
      //     designate inside):  the inner designate's indices are
      //     dimensional element subscripts that supersede whatever
      //     the inner had -- overwrite.
      //
      //   * Array-of-Records access where the inner is a COMPONENT
      //     designate with its own field subscript
      //     (``arr(i) % x(2)``): the parent's indices are the
      //     RECORD-index (outer dim of the flat ``arr_x``), the
      //     inner's are the FIELD-index (inner dim).  In Fortran
      //     column-major, record runs fastest -- but the bridge's
      //     SDFG arrays for static AoR keep the source order
      //     (record on dim 0, field on dim 1, matching the
      //     flat layout the flatten pass produces with shape
      //     ``[N_records, N_field_elems]``).  PREPEND the parent's
      //     indices to the existing entries so the flat
      //     ``arr_x[i, 2]`` access carries both dims.
      std::vector<DimEntry> parent_entries;
      auto pidxOps = parent.getIndices();
      for (unsigned d = 0; d < pidxOps.size(); ++d) {
        auto idx = pidxOps[d];
        auto n = resolveIndex(idx);
        parent_entries.push_back({n.empty() ? "?" : n, buildIndexExpr(idx, 0)});
      }
      // AoR component-chain discriminator: PREPEND the parent's
      // (record) indices when this is a struct-field access chain.
      // ``innermost.getComponentAttr()`` covers the classic
      // ``arr(i) % x(j)`` shape (component on the innermost designate);
      // ``sawComponentParent`` extends it to the pointer/allocatable
      // member case (``vcut % corrected(i,j,k)`` / ``arr(ia) % box(ir)``)
      // where the component sits on an INTERMEDIATE whole-member
      // selector reached through a box ``fir.load`` -- there the element
      // subscript lives on the innermost designate and any outer record
      // index must PREPEND, not overwrite (overwriting drops the element
      // dims and the memlet rank no longer matches the flat array).
      // Otherwise overwrite (plain nested-element access on a non-struct
      // array, where the inner indices already describe the data view).
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
    for (unsigned d = 0; d < triplets.size(); ++d) {
      if (triplets[d]) {
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
    // AoR shape: when the access is a struct-field chain AND the parent
    // has only scalar indices (all triplets[d]==false), the parent's
    // indices are RECORD indices and the inner's are FIELD/element
    // indices.  Concatenate them (record-first) for the flat
    // ``arr_x[i, j]`` access.  Mirrors the empty-triplet AoR branch
    // above; this covers parents whose ``getIsTriplet()`` returns a
    // non-empty all-false array (Flang's runtime-indexed
    // element-designate shape).  ``sawComponentParent`` extends the
    // discriminator to the pointer/allocatable member case where the
    // component is on an intermediate whole-member selector
    // (``arr(ia) % box(ir)``: the record index ``ia`` arrives here and
    // must PREPEND onto the element ``[ir]``, not overwrite it).
    bool allScalar = true;
    for (unsigned d = 0; d < triplets.size(); ++d)
      if (triplets[d]) {
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

  // For AoR / nested component chains
  // (``arr(i) % x(j)``, ``arr(i) % inner % x(j)``, ...), prefer
  // ``traceToDecl`` on the innermost designate's RESULT -- that
  // recursion walks through every component designate and accumulates
  // the FULL flat name (``arr_x``, ``arr_inner_x``, ...) matching the
  // bridge's flatten convention.  Falls back to the parent-walk's
  // ``parent_val`` only when no component is involved (plain element
  // / section access on a non-struct array).
  //
  // ``sawComponentParent`` covers the pointer/allocatable-member case
  // where the component sits on a PARENT designate, not the innermost
  // (``vcut % corrected(i,j,k)``: innermost is the plain element
  // designate over the loaded box, the component ``corrected`` is on
  // its parent).  Without it ``traceToDecl(parent_val)`` returns the
  // bare struct base ``vcut`` and the read flattens to the wrong name.
  std::string array_name;
  if (innermost.getComponentAttr() || sawComponentParent) {
    array_name = traceToDecl(innermost.getResult());
  }
  if (array_name.empty()) array_name = traceToDecl(parent_val);
  if (array_name.empty()) array_name = traceToDecl(innermost.getMemref());
  return {std::move(array_name), std::move(entries)};
}

/// Walk a yielded SSA value tree and append one ``AccessInfo`` per
/// data read.  Shared between ``buildElementalAssign``,
/// ``buildElementalCountLibcall`` and ``buildElementalAnyAllReduce``
/// so all three see the same index expansion  --  in particular,
/// ``hlfir.designate`` reads always go through ``expandDesignateChain``
/// so a parent-section's scalar dim (``m(:, pos1)`` where ``pos1``
/// occupies dim 1) is not silently dropped.
///
/// Cases:
///   * ``hlfir.designate`` -> expandDesignateChain -> AccessInfo of
///     full underlying-array rank; recurse into the inner indices so
///     nested designates / loads register their own reads.
///   * ``fir.load %declare`` (scalar dummy without designate) ->
///     emit a no-subscript AccessInfo so ``build_memlet_index``
///     falls through to subset ``0`` for the 1-element-array dummy.
///   * ``hlfir.apply %elem, %i`` whose elemental was earlier
///     materialised into a transient (kHlfirExprToTransient hit) ->
///     register a read against the transient at the apply iters
///     (Fortran 1-based form, no ``+1`` adjustment).
///   * ``hlfir.apply %elem, %i`` whose source elemental is in scope ->
///     push the apply-iter mapping onto indexStack and recurse into
///     the inner elemental's yield.
///   * fallback: recurse on every operand.
void collectReadAccesses(mlir::Value v, std::vector<AccessInfo>& accesses, int depth) {
  if (depth > 40) return;
  auto* op = v.getDefiningOp();
  if (!op) return;
  // ``fir.box_dims`` reads only the descriptor's shape/bounds metadata, which
  // the index-expr builder already renders as shape/offset SYMBOLS -- never a
  // data element.  Recursing into its box operand re-hits the ARRAY's own
  // designate and records a spurious WHOLE-array read (the AoS outer index
  // only) -- a rank-1 memlet on the real rank-N array that fails
  // ``sdfg.validate`` (the ``patch_3d % p_patch_1d(1) % <member>``
  // offset-in-subscript shape broadcast through an ``hlfir.elemental``: one
  // bogus read per non-1-based dim).  Stop the walk here.  Matched by op name
  // (the same form the ``buildIndexExpr`` box_dims handler uses).
  if (op->getName().getStringRef() == "fir.box_dims") return;
  // A reduction materialised into a scalar by ``materialiseCondReductions`` is
  // owned by its Reduce lib-node; its source array reads belong to that node,
  // NOT the condition.  Skip it -- the condition reads the bare scalar
  // transient (added separately as a register read by the materialiser's reduce
  // node).
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
          std::string s = n.empty() ? std::string("?") : n;
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
            indexStack().push_back({iblock.getArgument(i), name});
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
    // The reads of an ``scf.if``-valued expression (e.g. NPB LU's break
    // continuation ``(not (rsdnm(i) < tolrsd(i))) if (rem > 0) else 0``)
    // live inside the THEN / ELSE regions' ``scf.yield`` values -- the
    // generic operand recursion below only reaches the ``scf.if``
    // CONDITION.  Mirror ``buildBoolExpr``'s scf.if descent so the yielded
    // array reads are enumerated; without this ``condAccesses`` comes back
    // empty and ``walkSCFBeforeRegion`` renders the break as a bare-pointer
    // compare on an interstate-edge symbol (see
    // tests/lu_two_call_convergence_repro_test.py).
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

/// Map an ``hlfir`` ``expr``-producing op to the FaCe libcall callee tag
/// (``matmul`` / ``transpose`` / ``dot_product``).  Returns nullptr if
/// the op isn't one of the materialisable libcalls we know about.
const char* libcallNameForExprOp(mlir::Operation* op) {
  if (!op) return nullptr;
  auto name = op->getName().getStringRef();
  // The set MUST stay in sync with the libcall dispatcher's
  // ``kLibTable`` at ``dispatch.cpp:2082`` -- every entry there that
  // has an HLFIR op-name maps to an inline-libcall name here so the
  // elemental + ``hlfir.apply`` materialisation in
  // ``control_flow.cpp::walkElementalBody`` can pre-emit a
  // ``_libtmp_<gid>`` transient for the result.  Without this entry,
  // ``buildExpr`` sees the apply, falls through, and emits ``?`` into
  // the tasklet body.  QE's ``vcut_get`` (inline matmul_transpose)
  // was the surfacing case for the recent additions.
  if (name == "hlfir.matmul") return "matmul";
  if (name == "hlfir.transpose") return "transpose";
  if (name == "hlfir.dot_product") return "dot_product";
  // ``hlfir.matmul_transpose`` is the fused ``MATMUL(TRANSPOSE(A), B)``
  // op that ``hlfir-optimized-bufferization`` synthesises from the
  // separate ``hlfir.matmul %T %B`` + ``%T = hlfir.transpose %A``
  // pair.
  if (name == "hlfir.matmul_transpose") return "matmul_transpose";
  // Fortran ``COUNT(mask [, dim])`` -- CountLibraryNode.  The
  // dispatcher path at ``dispatch.cpp::buildElementalCountLibcall``
  // handles the elemental-mask case at the WHOLE-assign level; this
  // entry covers the same op when it appears as an inline operand
  // (e.g. ``res = COUNT(arr > 0) + 1``).
  if (name == "hlfir.count") return "count";
  // ``MINLOC`` / ``MAXLOC`` -- ArgMin / ArgMax library nodes.
  if (name == "hlfir.minloc") return "argmin";
  if (name == "hlfir.maxloc") return "argmax";
  // ``CSHIFT(array, shift [, dim])`` -- circular shift.
  if (name == "hlfir.cshift") return "cshift";
  return nullptr;
}

/// Pull the per-dim shape strings out of an ``hlfir.expr`` type.
/// Static dims become decimal literals; ``?`` extents stay as ``?``
/// for descriptors.py to fill from a synthetic ``<name>_d<i>`` symbol.
std::vector<std::string> exprResultShape(mlir::Type ty) {
  std::vector<std::string> out;
  if (auto e = mlir::dyn_cast<hlfir::ExprType>(ty)) {
    for (int64_t d : e.getShape()) {
      if (d == hlfir::ExprType::getUnknownExtent())
        out.push_back("?");
      else
        out.push_back(std::to_string(d));
    }
  }
  return out;
}

/// Map an ``hlfir.expr<...>`` element type to FaCe's dtype string.
/// Defaults to ``float64`` to keep callers simple  --  the caller would
/// otherwise have to fall back to it anyway.
///
/// ``i1`` (boolean mask elements from COUNT / ANY / ALL elementals)
/// surfaces as ``int32`` -- DaCe's CountLibraryNode and the
/// ``Reduce.atomic`` over a boolean mask expect 0/1 integer
/// elements, and the bridge's materialise loop emits
/// ``dace.int32(<i1 yield>)`` to widen each element correctly.
/// Other integer widths (i8 / i16) fall through to float64 -- no
/// workload has surfaced those yet; if a future case needs them,
/// add the explicit mapping here.
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

/// Build the AST-node sequence for a Fortran reduction whose source is
/// an inline ``hlfir.elemental``  --  the "Mode C" path for COUNT (and the
/// shape that generalises to SUM / ANY / ALL on comparison sources).
///
/// Emits three ASTNodes in order:
///   1. ``kind="declare_transient"``  --  a fresh int32 transient sized to
///      the elemental's shape.  ``descriptors.emit_declare_transient``
///      registers the array on the SDFG and in ``builder.arrays``.
///   2. nested ``kind="loop"`` (rank-deep) wrapping a ``kind="assign"``
///      whose target is the transient and whose RHS is
///      ``dace.int32(<elemental yield expression>)``.  The bridge's
///      generic select / cmp / arith machinery walks the yield expr.
///   3. ``kind="libcall"`` to ``CountLibraryNode`` reading the transient
///      and writing the original ``hlfir.assign`` destination.
///
/// The for-loop body has no WCR  --  the reduction stays inside the
/// library node's expansion (which uses a ``Reduce`` library node, not
/// a WCR-on-tasklet).  When the user's elemental body is more elaborate
/// than a single comparison, the chain-of-tasklets shape still lands
/// inside the loop body as a normal assign; downstream loop-to-map
/// transformations can paralleise the synthesised loop without
/// modifying the rest of the SDFG.
///
/// Lower an ``hlfir.elemental`` whose body is a per-element predicate
/// (or boolean-valued expression) into a (transient declare, loop
/// chain that fills the transient) pair.  Returns the synthetic
/// transient name and a 2-element vector of AST nodes (declare +
/// outermost loop) ready for the caller to follow with a terminal
/// reduction / libcall over the transient.  The transient is always
/// int32; the body expression is always wrapped in ``dace.int32(...)``
/// so the in-tasklet cast normalises any logical / i1 result.
///
/// Used by ``buildElementalCountLibcall`` (terminal: count libcall)
/// and ``buildElementalAnyAllReduce`` (terminal: Reduce node).  The
/// only thing the caller chooses is the transient-name prefix
/// (so dump output identifies which reduction the mask serves) and
/// the terminal node.
static std::pair<std::string, std::vector<ASTNode>> materialiseElementalToTransient(hlfir::ElementalOp elem,
                                                                                    std::string_view prefix) {
  auto& region = elem.getRegion();
  if (region.empty()) return {{}, {}};
  auto& block = region.front();
  unsigned rank = block.getNumArguments();
  auto shape = elem.getShape();

  std::string trName = std::string(prefix) + std::to_string(kSynthTransientCounter++);

  // Dtype follows the elemental's result element type via
  // ``exprDtypeString`` (which now handles i1 -> int32 for the
  // boolean mask elements COUNT / ANY / ALL produce).  Previously
  // this was hardcoded ``int32`` because the helper was only used
  // for those mask elementals; routing SUM / PRODUCT / MINVAL /
  // MAXVAL of inline elementals (e.g. QE's
  // ``SUM((a - b) ** 2)``) through the same path needs the
  // elemental's real element type.
  std::string dtype = exprDtypeString(elem.getType());

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
    indexStack().push_back({block.getArgument(i), iter_names[i]});
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

  // Pre-walk for dim-reductions on the apply chain:
  // ``hlfir.apply %sum_result, %i`` where ``%sum_result =
  // hlfir.sum %inner_elem dim %k`` returns a vector that the outer
  // elemental's body applies element-wise.  Without pre-materialisation,
  // ``buildExpr(%apply)`` returns ``?`` because the ``hlfir.sum`` source
  // is neither an inner elemental nor a libcall whose result already
  // landed in ``kHlfirExprToTransient``.  QE's
  // ``MINVAL(SQRT(SUM(a ** 2, 1)))`` shape (in ``vcut_spheric_get``)
  // surfaces this as ``_out__mask_1 = sqrt(?)`` at
  // emit_tasklet validation time.
  //
  // Strategy mirrors ``findApplies`` for libcalls (control_flow.cpp:289+):
  // walk the body, find apply-of-reduction, materialise the inner
  // elemental into a ``_libtmp_<gid>`` transient, emit a
  // ``kind="reduce"`` AST node writing to a sibling transient, and
  // register the reduction op in ``kHlfirExprToTransient`` so
  // ``buildExpr``'s apply branch renders the apply as the transient's
  // name.
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
        std::string wcr, identity;
        if (!reduceWcrIdentity(srcOp, wcr, identity)) {
          // Not a reduction -- recurse into operands.  AND, when
          // the apply is over a nested ELEMENTAL, descend into
          // that elemental's BODY too: a chain like
          // ``SUM(LOG(SUM(a,1)+1.0))`` nests
          // ``outer-sum -> log-elem -> (+1.0)-elem -> inner-sum-dim``,
          // and the inner SUM-dim lives inside the ``+1.0``
          // elemental's body, NOT among this apply's operands.
          // Without descending, ``buildExpr`` later renders the
          // inner ``hlfir.apply %inner_sum`` as ``?`` (the inner
          // SUM was never materialised).  Each level the walk
          // reaches gets its own ``_libtmp_`` transient, so the
          // chain resolves transient-by-transient.
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
        // Materialise the inner source.  If it's an elemental,
        // run ``materialiseElementalForLibcall`` to get a
        // transient that the reduce reads from; if it's a
        // named array, use ``traceToDecl`` directly.
        mlir::Value redSrc = srcOp->getOperand(0);
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
        std::string tmp = "_libtmp_" + std::to_string(kLibTmpCounter++);
        kHlfirExprToTransient[srcOp] = tmp;
        mlir::Type rty = srcOp->getResult(0).getType();
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
        // Pre-nodes ordering: source materialisation first, then
        // declare_transient, then the reduce that writes to it.
        for (auto& n : srcMaterialNodes) reductionPreNodes.push_back(std::move(n));
        reductionPreNodes.push_back(std::move(decl));
        reductionPreNodes.push_back(std::move(red));
        // Continue recursion in case there's another reduction
        // deeper (chained reductions).
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
    // Tasklet-body mode: comparisons / loads emit bare names so
    // emit_tasklet's per-occurrence connector wiring picks them up.
    NoSubscriptGuard g;
    std::string b = buildBoolExpr(yielded, 0);
    if (b == "?") b = buildExpr(yielded, 0);
    body = b;
  }
  // Wrap the per-element value in ``dace.int32(...)`` ONLY when
  // the elemental yields a boolean (i1).  This is the COUNT /
  // ANY / ALL mask path -- the i1 value widens to int32 (0/1)
  // for the runtime mask buffer.  For SUM / PRODUCT / MINVAL /
  // MAXVAL of an inline elemental over real / integer values
  // (e.g. ``SUM((a - b) ** 2)``) the cast would silently truncate
  // every materialised value to int32, breaking the reduction.
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
  // Reduction pre-materialisation nodes go FIRST -- they declare the
  // transients the elemental body's apply now reads from.
  for (auto& n : reductionPreNodes) nodes.push_back(std::move(n));
  nodes.push_back(std::move(decl));
  nodes.push_back(std::move(current));
  return {std::move(trName), std::move(nodes)};
}

/// Materialise an ``hlfir.elemental`` into a synthetic transient with
/// the elemental's element dtype (NOT the int32 forced by
/// ``materialiseElementalToTransient``).  Used by the
/// libcall-over-elemental path (transpose / matmul / dot_product /
/// etc. fed an inline elemental expression like
/// ``transpose(1.0 - d)``): the bridge can't pass an unnamed
/// ``hlfir.expr<...>`` to ``emit_library`` (which keys the source
/// array on its SDFG name), so we stage the elemental into a transient
/// the libcall reads from.
///
/// Returns ``{transient_name, AST_nodes}``.  ``AST_nodes`` is empty on
/// any failure (caller should fall back to the original libcall and
/// let downstream surface the error).
std::pair<std::string, std::vector<ASTNode>> materialiseElementalForLibcall(hlfir::ElementalOp elem) {
  auto& region = elem.getRegion();
  if (region.empty()) return {{}, {}};
  auto& block = region.front();
  unsigned rank = block.getNumArguments();
  auto shape = elem.getShape();
  if (!shape) return {{}, {}};

  std::string trName = "_libsrc_" + std::to_string(kSynthTransientCounter++);
  std::string dtype = exprDtypeString(elem.getType());

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
    indexStack().push_back({block.getArgument(i), iter_names[i]});
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
    NoSubscriptGuard g;
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

/// ``ANY(arr1 .eq. arr2)`` / ``ALL(...)``  --  same materialise-then-reduce
/// shape as ``buildElementalCountLibcall``, but the final node is a
/// ``kind="reduce"`` over the transient rather than the
/// ``CountLibraryNode`` libcall.
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
