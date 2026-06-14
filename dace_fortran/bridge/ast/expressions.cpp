// Translation-unit headers.  ``ast_helpers.h`` carries the cross-TU
// API + thread-local state shared with the other ``ast/*.cpp`` files.
#include <cstdlib>
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
#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace hlfir_bridge {

//
// Expression-builder primitives.  Owns:
//   * buildExpr (recursive Python-syntax expression rewrite for arith,
//     math.*, fir.load, hlfir.designate, hlfir.apply, ...) + its forward
//     declarations.
//   * buildIndexExpr and buildDesignateIndexExpr (Fortran 1-based
//     index renderer with section-parent + assumed-shape rebase, 0).
//   * resolveIndex and indexStack() (elemental-iter substitution).
//   * allocaSynthName (synthetic names for bare fir.alloca scratch).
//   * Thread-local state used by buildExpr itself: kScfValueMap,
//     kAllocaMap, kHlfirExprToTransient.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/expressions.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list  --  CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.
std::vector<std::pair<mlir::Value, std::string>> &indexStack() {
  static thread_local std::vector<std::pair<mlir::Value, std::string>> s;
  return s;
}

std::string resolveIndex(mlir::Value idx) {
  // Look up through fir.convert chains since the index might be wrapped.
  mlir::Value cur = idx;
  for (int i = 0; i < limits::kConvertChainDepth; ++i) {
    for (auto it = indexStack().rbegin(); it != indexStack().rend(); ++it)
      if (it->first == cur) return it->second;
    if (auto *d = cur.getDefiningOp())
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
        cur = cv.getValue();
        continue;
      }
    break;
  }
  return traceToDecl(idx);
}

// Lower ``fir.is_present %v -> i1`` to a Python expression.  After
// ``hlfir-inline-all`` flattens an internal subprogram (Fortran's
// ``CONTAINS``), the operand of an inner-scope ``is_present`` walks
// back through one or more inlined ``hlfir.declare`` aliases until it
// roots at either:
//   * ``fir.absent``  --  the caller passed nothing -> constant ``0``;
//   * a host block-arg whose declare carries ``fortran_attrs<optional>``
//     -> emit the companion ``<name>_present`` symbol that
//     ``extract_vars`` registers alongside every host-scope OPTIONAL
//     dummy; the caller binds it to 0 / 1 at SDFG-call time;
//   * any other root (mandatory dummy, local alloca) -> constant ``1``,
//     since the storage is unconditionally bound.
// Only the host-scope declare (the one whose memref IS the block-arg)
// decides between ``_present`` and ``1``  --  inner-scope inlined
// aliases all carry ``optional`` from the callee's signature, but
// that's bookkeeping, not whether the caller actually passed storage.
// Returns ``""`` when the chain breaks before a recognisable root, so
// callers can fall back to ``?``.
std::string lowerIsPresent(mlir::Value operand) {
  mlir::Value cur = operand;
  hlfir::DeclareOp lastDecl;
  for (int i = 0; i < limits::kTraceToDeclMax && cur; ++i) {
    if (auto *d = cur.getDefiningOp()) {
      if (mlir::isa<fir::AbsentOp>(d)) return "0";
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
        cur = cv.getValue();
        continue;
      }
      if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
        lastDecl = dc;
        cur = dc.getMemref();
        continue;
      }
      // Descriptor-unwrapping wrappers are transparent for a presence
      // query: only ``fir.absent`` ever marks an argument absent, so a
      // box/ref that came from a real descriptor is present.  After
      // ``hlfir-inline-all`` splices an OPTIONAL dummy bound to a
      // PRESENT actual, the dummy's declare memref resolves through
      // ``fir.box_addr`` (and friends) onto the caller's storage --
      // without walking these we broke out and leaked ``?`` into the
      // guard (QE addusxx_g ``PRESENT(becphi_c)`` over a complex box).
      if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
        cur = ba.getVal();
        continue;
      }
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
        cur = rb.getBox();
        continue;
      }
      if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
        cur = eb.getMemref();
        continue;
      }
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
        cur = ld.getMemref();
        continue;
      }
      // A concrete data reference reached through the wrappers above is
      // a PRESENT actual argument.  After ``hlfir-inline-all`` splices
      // an OPTIONAL dummy bound to a present actual, the dummy's declare
      // memref resolves through ``box_addr`` onto the caller's value:
      // an ``hlfir.designate`` (array section / element, e.g. QE
      // ``becphi_c = becxx(ikq)%k(:, jbnd)``) or raw storage
      // (``alloca`` / ``allocmem`` / module ``addr_of``).  Absence only
      // ever appears as ``fir.absent`` (handled above), so any such
      // concrete root is present -> ``1``.
      if (mlir::isa<hlfir::DesignateOp, fir::AllocaOp, fir::AllocMemOp,
                    fir::AddrOfOp>(d))
        return "1";
      break;
    }
    // Block argument  --  that's the storage root.  The declare we
    // most recently walked through is the host-scope alias; its
    // OPTIONAL attribute tells us whether to emit the companion
    // symbol or fold to ``1``.
    if (mlir::isa<mlir::BlockArgument>(cur)) {
      bool isOpt = false;
      if (lastDecl)
        if (auto a = lastDecl.getFortranAttrs())
          isOpt =
              bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::optional);
      if (isOpt) {
        auto n = extractName(lastDecl.getUniqName().str());
        return allocAliasFor(n) + "_present";
      }
      return "1";
    }
    break;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Expression reconstruction
// ---------------------------------------------------------------------------

/// Recursively build a Python-syntax expression string from an SSA value.
/// Depth-limited to 30 to prevent infinite recursion on malformed IR.
///
/// Handles:
///   * binary / unary arith ops (addf, mulf, subf, divf, addi, muli,
///     negf, minimumf, maximumf)
///   * elementwise math.* ops (math.sin, math.cos, math.sqrt, math.exp,
///     math.log, math.log10, math.tan, math.sinh, math.cosh, math.tanh,
///     math.absf, math.floor, math.ceil, math.erf, math.erfc, math.powf,
///     math.atan, math.atan2, math.asin, math.acos)  --  emitted as a bare
///     Python call so DaCe's tasklet codegen can resolve the name.
///   * fir.load of hlfir.designate (named variable read)
///   * arith.constant integer / float literals
///   * fir.convert pass-through (numeric kind casts)
///   * hlfir.apply / hlfir.elemental composition (inlined at index)
// Cross-chunk helpers (signatures + docstrings live in
// ``bridge/ast/ast_helpers.h``).  Bodies appear later in this file
// or in ``assigns.inc`` / ``control_flow.inc``.

/// Build the index expression string for the ``dim``-th operand of a
/// ``hlfir.designate``, applying the assumed-shape rebase when the
/// designate's base is an inlined alias declare.  Rebase rule (Flang
/// convention: assumed-shape dummies implicitly carry lbound = 1):
///
///     outer_fortran_index = inner_fortran_index + outer_lbound - 1
///
/// so ``arr(i)`` with ``i`` in the callee's 1-based frame becomes
/// ``outer(i + outer_lbound - 1)``  --  downstream ``build_memlet_index``
/// then subtracts ``outer_lbound``, net result ``i - 1``, the same
/// Emit a Fortran 1-based index expression for one dim of a designate.
/// In the symbolic offset-symbol architecture, every memlet subtracts
/// ``offset_<arr>_d<dim>`` (declared by ``add_descriptors``).  The
/// array-level lower-bound rebase is therefore handled uniformly by
/// the offset symbol after ``sdfg.specialize``.
///
/// What this function still has to add to the raw index:
///   1. **Section-designate parent** (``hlfir.designate %inner (i)``
///      whose memref is ``hlfir.designate %a (lo:hi:stride)``): the
///      child's iter is local to the section, so we add ``(lo - 1)``
///      to map back to the root array's Fortran index.
///   2. **Assumed-shape alias** (``hlfir-inline-all`` splices a
///      callee's assumed-shape view into the caller; the callee's
///      view starts at lb=1 but the underlying storage uses the
///      caller's lb): the existing offset symbol on the resolved
///      OUTER array captures the caller's lb, so the callee's iter
///      ``i`` needs ``(lb_outer - 1)`` added so the final memlet
///      ``(i + lb_outer - 1) - offset_outer_d0`` collapses to
///      ``i - 1`` after specialise (correct for the callee's view).
std::string buildDesignateIndexExpr(hlfir::DesignateOp dg, unsigned dim,
                                    mlir::Value idx, int depth) {
  std::string raw = buildIndexExpr(idx, depth);
  auto memref = dg.getMemref();
  auto *defOp = memref.getDefiningOp();
  if (!defOp) return raw;

  // ``RewritePointerAssigns`` may have substituted the memref with
  // a slice-rebind value: ``fir.rebox(fir.embox(designate(parent,
  // slice)))``.  Walk through those wrappers to find the section
  // designate that carries the slice's lower bound, so the rebase
  // below fires the same way it does for an explicit
  // ``hlfir.designate %parent_section (%i)`` chain.  Without this,
  // a rebound slice ``ptr => arr(3:7)`` reads ``arr(i)`` instead of
  // ``arr(i + 2)`` for every ``ptr(i)`` access.
  {
    mlir::Value v = memref;
    for (int hop = 0; hop < limits::kConvertChainDepth && v; ++hop) {
      auto *d = v.getDefiningOp();
      if (!d) break;
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
        v = rb.getBox();
        continue;
      }
      if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
        v = eb.getMemref();
        continue;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
        v = cv.getValue();
        continue;
      }
      if (mlir::isa<hlfir::DesignateOp>(d)) {
        defOp = d;  // Found a section designate further up;
                    // fall into the existing parent path below.
      }
      break;
    }
  }

  // Section-designate parent contribution.
  if (auto parentDg = mlir::dyn_cast<hlfir::DesignateOp>(defOp)) {
    auto triplets = parentDg.getIsTriplet();
    if (!triplets.empty()) {
      // The inner designate indexes the section's (possibly RANK-REDUCED)
      // VIEW: its dim ``dim`` is the ``dim``-th TRIPLET dim of the parent
      // section, NOT the ``dim``-th original dim -- scalar-fixed dims
      // (the ``1`` in ``mill(1, offset+1:blk)``) are squeezed out of the
      // view and must be skipped.  Walk the parent dims to the ``dim``-th
      // triplet, then add THAT triplet's lower-bound rebase.  A
      // rank-preserving section (every dim a triplet) maps ``dim`` to
      // itself -- byte-identical to the previous ``triplets[dim]`` form.
      // (Without this, ``mill(1, off+1:blk)(k)`` dropped the ``+off``
      // rebase because ``triplets[dim=0]`` was the scalar dim, gathering
      // ``mill[1, k]`` instead of ``mill[1, off+k]``.)
      unsigned cursor = 0, tripletSeen = 0;
      bool found = false;
      for (unsigned k = 0; k < triplets.size(); ++k) {
        if (triplets[k]) {
          if (tripletSeen == dim) {
            found = true;
            break;
          }
          ++tripletSeen;
          cursor += 3;
        } else {
          cursor += 1;
        }
      }
      auto idxOps = parentDg.getIndices();
      if (found && cursor < idxOps.size()) {
        if (auto lo = traceConstInt(idxOps[cursor])) {
          int64_t adjust = *lo - 1;
          if (adjust > 0)
            raw = "(" + raw + " + " + std::to_string(adjust) + ")";
          else if (adjust < 0)
            raw = "(" + raw + " - " + std::to_string(-adjust) + ")";
        } else {
          // Section ``lo`` isn't a compile-time constant  --  typical
          // shape is ``a(pos(1):pos(2))`` where ``pos(1)`` minted
          // the symbol ``__sym_pos_1`` via ``buildIndexExpr``'s
          // load-of-designate path.  Use that closed-form so the
          // memlet stays expressible: rebase = ``+ (lo - 1)``.
          auto loExpr = buildIndexExpr(idxOps[cursor], depth + 1);
          if (!loExpr.empty() && loExpr != "?")
            raw = "(" + raw + " + " + loExpr + " - 1)";
        }
      }
    }
    return raw;
  }

  // Assumed-shape alias contribution: the callee's view-offset is 1
  // (Fortran default for assumed-shape) but the resolved outer
  // array's offset is the caller's lb.  Add ``(lb_outer - 1)`` so
  // the memlet form gives the right element after specialise.
  auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(defOp);
  if (!declOp) return raw;
  auto outer = asAssumedShapeAlias(declOp);
  if (!outer) return raw;
  auto lbs = declareLowerBounds(outer);
  if (dim >= lbs.size() || !lbs[dim]) return raw;
  int64_t adjust = *lbs[dim] - 1;
  if (adjust == 0) return raw;
  if (adjust > 0) return "(" + raw + " + " + std::to_string(adjust) + ")";
  return "(" + raw + " - " + std::to_string(-adjust) + ")";
}

/// Capture the LHS of an ``hlfir.assign`` whose destination is either a
/// bare ``hlfir.declare`` (whole-result variable, e.g. ``out = SUM(a)``)
/// or an ``hlfir.designate`` selecting one element of an array
/// (``res(2) = MINVAL(d)``).  Writes the resolved name into
/// ``node.target`` and, for the designate case, appends a per-dim
/// ``AccessInfo`` so the downstream emitter wires the output memlet to
/// that specific element.  Without this, every libcall / reduction in
/// the routine writes through the whole destination array and the last
/// one wins (or pytest fails with "memlet subset does not match node
/// dimension" when the destination has more dims than the libcall's
/// scalar output).
///
/// Shared by ``buildReduceNode``, ``buildElementalCountLibcall``,
/// ``buildElementalAnyAllReduce`` and ``buildLibCallNode``.
void captureElementDesignateWrite(mlir::Value dest, ASTNode &node) {
  if (auto dd = dest.getDefiningOp()) {
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd)) {
      node.target = allocAliasFor(extractName(decl.getUniqName().str()));
    } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
      // Struct field write target: ``g % y = ...`` -- the designate
      // has a component attribute but no element indices.  Use the
      // flattened ``<parent>_<member>`` name via ``traceToDecl`` on
      // the designate's result (lets the component branch fire)
      // and DON'T treat it as an array write (no AccessInfo
      // indexed loop is needed -- the target is a scalar / whole
      // field that downstream emit_assign treats by its own
      // descriptor classification).  Previously
      // ``traceToDecl(dg.getMemref())`` returned the struct base
      // ``g``, leaking ``g`` as the target name and forcing an
      // array-style write that ``KeyError``ed at arglist lookup.
      if (dg.getComponentAttr() && dg.getIndices().empty()) {
        node.target = traceToDecl(dg.getResult());
        // Whole-field write -- target_is_array reflects the field's
        // OWN shape, which downstream descriptor classification
        // recovers from the registered VarInfo.  Leaving it false
        // by default lets emit_assign pick the right path.
      } else {
        node.target = traceToDecl(dg.getMemref());
        node.target_is_array = true;
        AccessInfo wa;
        wa.array_name = node.target;
        wa.is_write = true;
        unsigned di = 0;
        for (auto idx : dg.getIndices()) {
          auto resolved = resolveIndex(idx);
          wa.index_vars.push_back(resolved.empty() ? "?" : resolved);
          wa.index_exprs.push_back(buildDesignateIndexExpr(dg, di, idx, 0));
          ++di;
        }
        node.accesses.push_back(std::move(wa));
      }
    }
  }
  if (node.target.empty()) node.target = traceToDecl(dest);
}

// ---------------------------------------------------------------------------
// Bridge-synthesised name conventions
// ---------------------------------------------------------------------------
//
// Synthetic names the bridge mints during AST extraction.  All start
// with ``_`` or ``__`` (reserved by the bridge) so they cannot collide
// with Flang-mangled Fortran names (which always start with ``_Q``
// followed by a Fortran-form identifier).
//
//   __sym_<arr>_<idx>   eager position-array symbol minted by
//                        ``buildIndexExpr`` for ``load(designate %arr
//                        (%const))``; load happens once at SDFG entry via a
//                        ``kind="symbol_init"`` AST node.
//                        Counter:  kPosSymbolRegistry (per-pair,
//                        deterministic).
//   __al_<n>            bare ``fir.alloca`` scratch (no surrounding declare),
//                        used as a synthetic scalar name so loads /
//                        stores of an unnamed alloca have something to
//                        reference in the AST.
//                        Counter:  kAllocaCounter.
//   __sc_<n>            scf.if synthetic scalar  --  a sink for the i-th
//                        ``scf.if`` result so downstream reads of the
//                        result Value resolve to a single name instead
//                        of recursing into both arms.
//                        Counter:  kScfValueCounter.
//   _count_mask_<n>     Mode-C COUNT mask transient (``COUNT(arr1.eq.arr2)``);
//                        per-element loop fills it, ``CountLibraryNode``
//                        reads it.
//                        Counter:  kSynthTransientCounter.
//   _mask_<n>           Mode-C ANY/ALL mask transient (``ANY(arr1.eq.arr2)``);
//                        same shape as count mask, terminated by a
//                        DaCe Reduce node instead of a libcall.
//                        Counter:  kSynthTransientCounter.
//   _libtmp_<n>         Libcall result transient inside an elemental  --
//                        ``hlfir.matmul`` / ``hlfir.transpose`` /
//                        ``hlfir.dot_product`` materialised ahead of
//                        the elemental that consumes it via
//                        ``hlfir.apply``.
//                        Counter:  kLibTmpCounter.
//
// All counters are thread-local and reset in ``extractAST`` (dispatch.inc)
// at module-walk start so two consecutive bridge calls don't inherit
// one another's numbering.
//
// The Python emitter pattern-matches on these prefixes in a few
// places (e.g. indirect-symbol detection); keep the prefixes stable
// or update both sides.
// ---------------------------------------------------------------------------

std::string allocaSynthName(mlir::Value memref) {
  auto *def = memref.getDefiningOp();
  if (!def) return "";
  auto it = kAllocaMap.find(def);
  if (it != kAllocaMap.end()) return it->second;
  std::string s = "__al_" + std::to_string(kAllocaCounter++);
  kAllocaMap[def] = s;
  return s;
}

/// Look up or mint the SDFG symbol name that stands in for
/// ``<array>(<i1>, <i2>, ...)`` (Fortran-side names / values).  Same key
/// always yields the same symbol  --  callers can safely use this
/// anywhere the load result was needed before.  ``posSymbolName`` is the
/// shared name format (also used by the descriptor-shape side).
std::string internPosSymbol(const std::string &array,
                            const std::vector<int64_t> &one_based_idxs) {
  auto k = std::make_pair(array, one_based_idxs);
  auto it = kPosSymbolRegistry.find(k);
  if (it != kPosSymbolRegistry.end()) return it->second;
  std::string s = posSymbolName(array, one_based_idxs);
  kPosSymbolRegistry[k] = s;
  return s;
}

std::string internPosSymbol(const std::string &array, int64_t one_based_idx) {
  return internPosSymbol(array, std::vector<int64_t>{one_based_idx});
}

std::string buildExpr(mlir::Value val, int d) {
  if (d > limits::kBuildExprDepth) return "?";
  // Synthetic scalars minted for scf.if results: every downstream read of
  // the result Value resolves to the scalar's name, not to walking into
  // the scf.if itself (which has no single defining expression  --  the
  // value comes from one of two arms).
  {
    auto it = kScfValueMap.find(val);
    if (it != kScfValueMap.end()) return it->second;
  }
  auto *def = val.getDefiningOp();
  if (!def) return "?";

  // ``fir.do_loop`` result: the loop is processed at the higher
  // ``kind="loop"`` AST level; downstream reads of the loop's
  // iter_args result resolve to the variable being accumulated.
  // Match the result index to the corresponding initial operand --
  // its source declare is the user-visible name.  Mirrors the
  // ``kScfValueMap`` mechanism for ``scf.if`` results (where the
  // synth-scalar map maintains the name); here the name is
  // recoverable directly from the iter_arg's init operand via
  // ``traceToDecl``.
  if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(def)) {
    // ``fir.do_loop`` returns:
    //   * result 0 -- the induction-variable's final value
    //                 (``index`` type), only present when
    //                 ``finalValue`` is requested.
    //   * results 1..N -- the iter_args' final values, in
    //                     declaration order.
    //
    // For the iter_args branch (where i > 0 OR the op has no
    // finalValue result), the i-th iter_arg's init operand carries
    // the user-visible source variable: trace through any
    // ``fir.load`` to reach the source declare, return its name.
    // Mirrors ``kScfValueMap`` for ``scf.if`` results.  The bridge
    // processes the loop body itself at the higher
    // ``kind="loop"`` AST level, so downstream reads of a loop
    // result resolve to "the accumulator after the loop ran".
    auto initArgs = doLoop.getInitArgs();
    unsigned resultIdx = mlir::cast<mlir::OpResult>(val).getResultNumber();
    unsigned iterIdx = resultIdx;
    // When the op has the finalValue result, results[0] is the
    // induction var and results[1..] are the iter_args.  Detect by
    // matching result count vs iter_args count.
    if (doLoop.getNumResults() == initArgs.size() + 1) {
      if (resultIdx == 0) {
        // Induction-variable final value -- name unknown to the
        // bridge (it's a loop iter, not a Fortran variable).  Fall
        // through and let the unhandled-op throw fire.
      } else {
        iterIdx = resultIdx - 1;
      }
    }
    if (iterIdx < initArgs.size()) {
      // Strategy 1: trace the iter_arg's INIT operand back through
      // a ``fir.load`` to the source declare.  Works for the
      // accumulator shape ``out = ... ; do ... ; out = out + ...``
      // where the iter_arg is initialised from a load of the
      // user variable.
      auto init = initArgs[iterIdx];
      if (auto *id = init.getDefiningOp())
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(id)) {
          auto n = traceToDecl(ld.getMemref());
          if (!n.empty()) return n;
        }
      // Strategy 2: walk the loop body for the FIRST
      // ``fir.store %arg_iter to %decl`` -- the iter_arg's stored
      // location is the user variable's declare.  Works for the
      // ``i`` shape where the iter_arg shadows the induction
      // counter via a convert (``%init = fir.convert %c1``, not a
      // load -- Strategy 1 doesn't fire).
      auto &body = doLoop.getRegion().front();
      mlir::Value iterArg = body.getArgument(iterIdx + 1);  // +1: skip induction
      for (auto &op : body) {
        if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
          if (st.getValue() == iterArg) {
            auto n = traceToDecl(st.getMemref());
            if (!n.empty()) return n;
            break;
          }
        }
      }
    }
  }

  // ``fir.embox`` wraps a memref into a Fortran box descriptor (adds
  // bounds + type-info around a raw pointer).  In an expression
  // context the descriptor and the underlying memref denote the
  // same Fortran value, so forward to the memref's expression.
  // Real-world ICON code hits this when an array dummy is passed to
  // a polymorphic helper via an assumed-shape interface  --  Flang
  // wraps the actual through ``fir.embox`` at the call site.
  if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def))
    return buildExpr(eb.getMemref(), d + 1);

  // ``hlfir.as_expr`` lifts a named variable / temporary into an
  // HLFIR-expr value (used when an op-class wants
  // ``hlfir.expr<...>`` rather than ``fir.ref<...>``).  Transparent
  // for expression building: the underlying variable IS the value.
  if (auto ae = mlir::dyn_cast<hlfir::AsExprOp>(def))
    return buildExpr(ae.getVar(), d + 1);

  // ``hlfir.declare`` registers a Fortran name on a memref.  When
  // referenced in an expression, the name itself is the read --
  // forward to the memref to pick up the canonical declaration
  // (``traceToDecl`` walks declare aliases) and return the name.
  if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
    auto name = traceToDecl(dc.getMemref());
    if (!name.empty()) return name;
    return buildExpr(dc.getMemref(), d + 1);
  }

  // ``fir.emboxchar`` packages ``(char_ptr, length)`` into a
  // CHARACTER descriptor.  The pointer half is the underlying
  // string; downstream Fortran arithmetic / concat works against
  // the character data itself, so forward to it.
  if (auto ebc = mlir::dyn_cast<fir::EmboxCharOp>(def))
    return buildExpr(ebc.getMemref(), d + 1);

  // ``fir.box_addr`` pulls the data pointer out of a box descriptor.
  // Forward through to the boxed value (peeling through any embox
  // we just installed above).
  if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(def))
    return buildExpr(ba.getVal(), d + 1);

  // ``fir.zero_bits`` (class name ``fir::ZeroOp``) produces an
  // all-zero value of any Fortran type -- used for uninitialized
  // scalars, null pointer sentinels, fresh boxes about to be
  // ``fir.embox``'d.  ``0`` is correct for every integer / real /
  // pointer typed RHS the expression layer surfaces.
  if (mlir::isa<fir::ZeroOp>(def))
    return "0";

  // ``fir.address_of`` takes a symbol reference and yields its
  // address.  In an expression context, the symbol's name is what
  // a downstream reader wants -- ``traceToDecl`` resolves it
  // through any subsequent declare aliases.
  if (auto ao = mlir::dyn_cast<fir::AddrOfOp>(def)) {
    auto name = traceToDecl(ao.getResult());
    if (!name.empty()) return name;
    // Strip the leading ``@`` from the symbol reference for a
    // human-readable fallback.  ``extractName`` peels Flang's
    // mangling decoration (``_QM<mod>E<var>`` -> ``var``) so the
    // tasklet code references the Fortran user name, not the raw
    // ``_QMmEarr1d`` form (which the SDFG arrays dict doesn't key
    // on -- the bridge stores each global under its short name).
    auto sym = ao.getSymbol().getRootReference().getValue().str();
    return extractName(sym);
  }

  // ``fir.alloca`` reserves storage on the stack.  ``traceToDecl``
  // walks past any subsequent ``hlfir.declare`` to pick up the
  // Fortran name the alloca participates in; if there is none
  // (synthetic temporary), fall through to the diagnostic / ``?``
  // path -- a raw alloca address has no spellable expression form.
  if (mlir::isa<fir::AllocaOp>(def)) {
    auto name = traceToDecl(def->getResult(0));
    if (!name.empty()) return name;
  }

  // ``fir.unboxchar`` extracts ``(char_ptr, length)`` from a
  // CHARACTER descriptor.  The char-data half (operand 0) is what
  // an expression context wants; the length is a separate use that
  // the bridge tracks via the box itself.
  if (auto uc = mlir::dyn_cast<fir::UnboxCharOp>(def))
    return buildExpr(uc.getOperand(), d + 1);

  // ``hlfir.concat`` is the Fortran ``//`` string concatenation
  // operator.  Map to Python ``+`` so the tasklet body uses the
  // string-concat semantics the DaCe codegen already understands.
  if (auto cc = mlir::dyn_cast<hlfir::ConcatOp>(def)) {
    std::string out;
    bool first = true;
    for (auto str : cc.getStrings()) {
      if (!first) out += " + ";
      out += buildExpr(str, d + 1);
      first = false;
    }
    return out;
  }

  auto nm = def->getName().getStringRef();

  // ``ALLOCATED(arr)``  --  Flang lowers as
  //   %addr = fir.box_addr (fir.load arr_box) -> heap<...>
  //   %i64  = fir.convert %addr : heap<...> -> i64
  //   %r    = arith.cmpi ne, %i64, %c0_i64
  // Recognise that exact shape (cmpi-ne against constant-zero of a
  // box_addr->convert chain) and read the per-allocatable companion
  // ``<arr>_allocated`` scalar that ``extract_vars`` registers and
  // the AST builder maintains at ALLOCATE / DEALLOCATE sites.
  if (nm == "arith.cmpi" && def->getNumOperands() == 2) {
    auto pred = def->getAttrOfType<mlir::IntegerAttr>("predicate");
    constexpr int64_t kPredNe = 1;  // mlir::arith::CmpIPredicate::ne
    if (pred && pred.getInt() == kPredNe) {
      // Operand 1 must be a constant int 0 (the null pointer
      // sentinel after the heap-addr->i64 cast).
      bool rhsZero = false;
      if (auto c = traceConstInt(def->getOperand(1))) rhsZero = (*c == 0);
      if (rhsZero) {
        // Operand 0: peel fir.convert back to find a fir.box_addr.
        mlir::Value cur = def->getOperand(0);
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
              // box_addr's operand is fir.load of a box
              // ref; trace through that to the declare.
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
  }

  // ``fir.box_dims %arr_decl, %dim``  --  Flang's lowering for SIZE /
  // LBOUND / UBOUND / SHAPE on assumed-shape (and other boxed)
  // arrays.  Produces a 3-tuple ``(lower_bound, extent, stride)``;
  // each result is read out via an OpResult index, so we map per
  // result number to the corresponding bridge-synthesised symbol.
  //
  // For the underlying array's K-th dim:
  //   * ``#0`` (lower bound) -> declared lb if present (``fir.shape_shift``),
  //                            otherwise Fortran-default ``1``.
  //   * ``#1`` (extent)      -> ``<arr>_d<K>`` symbol the bridge mints
  //                            for assumed-shape arrays in extract_vars
  //                            (line 426-429).  For explicit-shape arrays
  //                            (``dimension(N)``), the declare's ``fir.shape``
  //                            already carries the constant / symbol; we
  //                            recover it via ``buildIndexExpr`` on the
  //                            extent operand.
  //   * ``#2`` (stride)      -> ``1`` (assume contiguous; section
  //                            designates with non-1 stride don't reach
  //                            this path).
  if (nm == "fir.box_dims" && def->getNumOperands() >= 2) {
    auto resIdx = mlir::cast<mlir::OpResult>(val).getResultNumber();
    auto dimOp = def->getOperand(1);
    auto dimC = traceConstInt(dimOp);
    // Walk operand 0 back to the underlying ``hlfir.declare`` so we
    // can read its shape (for explicit-shape) or fall back to the
    // assumed-shape symbol form.
    mlir::Value arrayVal = def->getOperand(0);
    for (int hop = 0; hop < limits::kTraceToDeclMax && arrayVal; ++hop) {
      auto *adef = arrayVal.getDefiningOp();
      if (!adef) break;
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(adef)) {
        arrayVal = cv.getValue();
        continue;
      }
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(adef)) {
        arrayVal = ld.getMemref();
        continue;
      }
      break;
    }
    std::string arrName = traceToDecl(arrayVal);
    // ``fir.box_dims`` reads the *box* result (#1) of an
    // ``hlfir.declare``; ``traceToDecl`` keys on the addr result
    // and can come back empty for an assumed-shape / boxed dummy.
    // Recover the name from the declare's mangled uniq_name so the
    // bound resolves to its ``offset_<arr>_d<dim>`` /
    // ``<arr>_d<dim>`` symbol instead of leaking ``?`` into the
    // loop-bound expression (E10).
    if (arrName.empty()) {
      if (auto *adef = arrayVal.getDefiningOp())
        if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(adef))
          arrName = extractName(decl.getUniqName().str());
    }
    if (!dimC || arrName.empty()) return "?";
    unsigned dim = static_cast<unsigned>(*dimC);

    // Try to get the extent / lb from the declare's shape operand.
    // For inlined assumed-shape callee aliases (no shape on the
    // inner declare), walk to the outer declare via
    // ``asAssumedShapeAlias``  --  it shares storage with the caller
    // and carries the actual shape/lb info.
    mlir::Value shapeVal;
    if (auto *adef = arrayVal.getDefiningOp()) {
      if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(adef)) {
        shapeVal = decl.getShape();
        if (!shapeVal) {
          if (auto outer = asAssumedShapeAlias(decl))
            shapeVal = outer.getShape();
        }
      }
    }

    // Allocatable runtime shape: when the declare has no static
    // shape (assumed-shape signature backing an allocatable), the
    // shape lives on the embox emitted at the ALLOCATE site:
    //   %alloc = fir.allocmem ... {uniq_name = "<decl>.alloc"}
    //   %sh    = fir.shape_shift %lb0, %ext0, %lb1, %ext1, ...
    //   %box   = fir.embox %alloc(%sh)
    //   fir.store %box to <decl>#1
    // ``LBOUND(arr, dim)`` / ``UBOUND(arr, dim)`` on an allocatable
    // bound via ``allocate(arr(lo:hi))`` should resolve to the
    // user-supplied bounds, not the Fortran-default ``1`` /
    // synthetic ``<arr>_d<dim>`` symbol.  Walk the allocate site
    // (single-allocate case only  --  multi-allocate would need
    // per-access-site selection) and constant-fold the
    // shape_shift operands.
    auto resolveAllocShapeShift = [&]() -> fir::ShapeShiftOp {
      auto *adef = arrayVal.getDefiningOp();
      if (!adef) return {};
      auto decl = mlir::dyn_cast<hlfir::DeclareOp>(adef);
      if (!decl) return {};
      // Inlined-callee allocatable aliases: the inner declare's
      // memref points back at the outer (caller-scope) declare's
      // result.  The ``fir.allocmem`` uniq_name is keyed on the
      // outer's scope (``_QFouterEinput.alloc``), so walk through
      // the alias chain before looking up.
      if (auto outer = asAssumedShapeAlias(decl)) decl = outer;
      std::string allocName = decl.getUniqName().str() + ".alloc";
      auto mod = decl->getParentOfType<mlir::ModuleOp>();
      if (!mod) return {};
      fir::ShapeShiftOp out;
      mod.walk([&](fir::AllocMemOp a) {
        if (out) return;
        auto un = a.getUniqName();
        if (!un || un->str() != allocName) return;
        for (auto *user : a->getUsers()) {
          auto eb = mlir::dyn_cast<fir::EmboxOp>(user);
          if (!eb) continue;
          auto sh = eb.getShape();
          if (!sh) continue;
          if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(
                  sh.getDefiningOp())) {
            out = ss;
            break;
          }
        }
      });
      return out;
    };

    // Lower bound (#0):
    if (resIdx == 0) {
      if (shapeVal) {
        if (auto ss =
                mlir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp())) {
          auto ops = ss->getOperands();
          unsigned lbIdx = 2 * dim;
          if (lbIdx < ops.size()) {
            auto s = buildIndexExpr(ops[lbIdx], d + 1);
            if (!s.empty() && s != "?") return s;
          }
        }
        // ``fir.shift`` (assumed-shape / pointer dummy declared
        // with explicit local lower bounds, e.g. ``a(10:)`` ->
        // ``fir.shift %c10``).  Operands are lbs only -- the
        // K-th operand is dim K's lower bound.  Without this
        // ``LBOUND``/``UBOUND`` on such a dummy fell through to
        // the wrong default and leaked the ``?`` sentinel into
        // the loop-bound expression (E10).
        if (auto sf = mlir::dyn_cast<fir::ShiftOp>(shapeVal.getDefiningOp())) {
          if (dim < sf->getNumOperands()) {
            auto s = buildIndexExpr(sf->getOperand(dim), d + 1);
            if (!s.empty() && s != "?") return s;
          }
        }
      }
      if (auto ss = resolveAllocShapeShift()) {
        auto ops = ss->getOperands();
        unsigned lbIdx = 2 * dim;
        if (lbIdx < ops.size())
          if (auto c = traceConstInt(ops[lbIdx])) return std::to_string(*c);
      }
      return "1";
    }
    // Extent (#1):
    if (resIdx == 1) {
      if (shapeVal) {
        if (auto sh = mlir::dyn_cast<fir::ShapeOp>(shapeVal.getDefiningOp())) {
          if (dim < sh.getExtents().size()) {
            auto s = buildIndexExpr(sh.getExtents()[dim], d + 1);
            if (!s.empty() && s != "?") return s;
            // ``buildIndexExpr`` doesn't handle Flang's
            // ``max(ext, 0)`` extent-clamp idiom (an
            // ``arith.select`` over an ``arith.cmpi sgt``).
            // ``traceExtentExpr`` peels the clamp -- it's what
            // ``extract_vars`` already uses to derive
            // ``v.shape_symbols``.  Routes the box_dims extent
            // for an explicit-shape caller (``arr(n)``) directly
            // to ``n`` instead of falling through to the
            // synthetic ``<arr>_d<dim>`` symbol, which the
            // caller wouldn't know to bind.
            auto te = traceExtentExpr(sh.getExtents()[dim]);
            if (!te.empty() && te != "?") return te;
          }
        }
        if (auto ss =
                mlir::dyn_cast<fir::ShapeShiftOp>(shapeVal.getDefiningOp())) {
          auto ops = ss->getOperands();
          unsigned extIdx = 2 * dim + 1;
          if (extIdx < ops.size()) {
            auto s = buildIndexExpr(ops[extIdx], d + 1);
            if (!s.empty() && s != "?") return s;
            auto te = traceExtentExpr(ops[extIdx]);
            if (!te.empty() && te != "?") return te;
          }
        }
      }
      if (auto ss = resolveAllocShapeShift()) {
        auto ops = ss->getOperands();
        unsigned extIdx = 2 * dim + 1;
        if (extIdx < ops.size())
          if (auto c = traceConstInt(ops[extIdx])) return std::to_string(*c);
      }
      // Assumed-shape (no declare shape)  --  the bridge synthesised
      // ``<arr>_d<dim>`` in extract_vars.  Same string convention.
      return arrName + "_d" + std::to_string(dim);
    }
    // Stride (#2): contiguous default.
    if (resIdx == 2) return "1";
  }

  // Binary arithmetic.
  static const std::map<llvm::StringRef, std::string> bin_ops = {
      {"arith.mulf", " * "},
      {"arith.addf", " + "},
      {"arith.subf", " - "},
      {"arith.divf", " / "},
      {"arith.muli", " * "},
      {"arith.addi", " + "},
      {"arith.subi", " - "},
      {"arith.divsi", " // "},
      {"arith.divui", " // "},
      // Fortran COMPLEX arithmetic  --  flang emits dedicated ops on
      // ``complex<f32>`` / ``complex<f64>`` operands.
      {"fir.addc", " + "},
      {"fir.subc", " - "},
      {"fir.mulc", " * "},
      {"fir.divc", " / "},
  };
  if (auto it = bin_ops.find(nm);
      it != bin_ops.end() && def->getNumOperands() == 2) {
    return "(" + buildExpr(def->getOperand(0), d + 1) + it->second +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  if (nm == "arith.negf" && def->getNumOperands() == 1)
    return "(-" + buildExpr(def->getOperand(0), d + 1) + ")";

  // Fortran ``conjg(z)`` lowers to:
  //     %im  = fir.extract_value %z, [1] : complex<T> -> T
  //     %neg = arith.negf %im
  //     %r   = fir.insert_value %z, %neg, [1]
  // Recognise the full idiom and emit ``<z>.conjugate()`` so the
  // tasklet renders the Python complex method.  DaCe's tasklet
  // codegen lowers ``.conjugate()`` to ``std::conj`` on
  // ``std::complex<T>``.
  if (auto ins = mlir::dyn_cast<fir::InsertValueOp>(def)) {
    auto coords = ins.getCoor();
    // Fortran ``cmplx(re, im, kind=K)`` lowers to:
    //   %base = fir.undefined complex<T>
    //   %r0   = fir.insert_value %base, %re, [0]
    //   %r1   = fir.insert_value %r0, %im, [1]
    // Recognise the outermost insert at coord [1] whose adt is an
    // insert at coord [0] of an ``fir.undefined`` and emit
    // ``complex(<re>, <im>)``.
    if (coords.size() == 1) {
      if (auto coordAttr = mlir::dyn_cast<mlir::IntegerAttr>(coords[0]))
        if (coordAttr.getInt() == 1) {
          if (auto inner = mlir::dyn_cast_or_null<fir::InsertValueOp>(
                  ins.getAdt().getDefiningOp())) {
            auto innerCoords = inner.getCoor();
            if (innerCoords.size() == 1)
              if (auto a0 = mlir::dyn_cast<mlir::IntegerAttr>(innerCoords[0]))
                if (a0.getInt() == 0)
                  if (mlir::isa_and_nonnull<fir::UndefOp>(
                          inner.getAdt().getDefiningOp())) {
                    // Use the ``re + 1j*im`` form
                    // rather than ``complex(re, im)``:
                    // DaCe's tasklet C++ codegen
                    // doesn't lower a free
                    // ``complex(...)`` call to the
                    // ``std::complex`` constructor,
                    // but it does handle the
                    // ``1j`` literal arithmetic via
                    // its complex-arithmetic
                    // rewrites.
                    return "((" + buildExpr(inner.getVal(), d + 1) +
                           ") + 1j * (" + buildExpr(ins.getVal(), d + 1) + "))";
                  }
          }
        }
    }
    if (coords.size() == 1) {
      // Coord must be the literal index 1 (the imaginary slot).
      if (auto coordAttr = mlir::dyn_cast<mlir::IntegerAttr>(coords[0]))
        if (coordAttr.getInt() == 1) {
          auto val = ins.getVal();
          auto adt = ins.getAdt();
          if (auto neg = mlir::dyn_cast_or_null<mlir::arith::NegFOp>(
                  val.getDefiningOp())) {
            if (auto ext = mlir::dyn_cast_or_null<fir::ExtractValueOp>(
                    neg.getOperand().getDefiningOp())) {
              auto extCoords = ext.getCoor();
              bool extIsImag = false;
              if (extCoords.size() == 1)
                if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(extCoords[0]))
                  extIsImag = (a.getInt() == 1);
              if (extIsImag && ext.getAdt() == adt) {
                // Emit ``conj(<expr>)``  --  DaCe's tasklet
                // codegen routes the bare name through
                // ``dace::math::conj`` (defined in
                // ``runtime/include/dace/math.h``) which
                // forwards to ``std::conj`` for both
                // ``std::complex<float>`` and
                // ``std::complex<double>``.
                return "conj(" + buildExpr(adt, d + 1) + ")";
              }
            }
          }
        }
    }
  }

  // Standalone ``fir.extract_value`` on a complex value -- Fortran
  // ``real(z, kind=K)`` and ``aimag(z)`` lower to extract_value at
  // coord [0] / [1] respectively (the conjg / cmplx idiom recognizers
  // above match the embedded-extract case, but neither fires when
  // the extract is a top-level operand of a downstream arith op).
  // Emit ``<z>.real()`` / ``<z>.imag()`` -- method-call syntax so
  // ``cppunparse`` renders it as ``z.real()`` in the generated C++,
  // matching ``std::complex<T>::real()`` / ``::imag()`` member
  // functions.  The Python AST is well-formed (no runtime evaluation
  // happens on tasklet code), and DaCe's tasklet codegen routes it
  // through directly.
  if (auto ext = mlir::dyn_cast<fir::ExtractValueOp>(def)) {
    auto srcTy = ext.getAdt().getType();
    if (mlir::isa<mlir::ComplexType>(srcTy)) {
      auto coords = ext.getCoor();
      if (coords.size() == 1) {
        if (auto cAttr = mlir::dyn_cast<mlir::IntegerAttr>(coords[0])) {
          int64_t c = cAttr.getInt();
          if (c == 0) return "(" + buildExpr(ext.getAdt(), d + 1) + ".real())";
          if (c == 1) return "(" + buildExpr(ext.getAdt(), d + 1) + ".imag())";
        }
      }
    }
  }

  // Elementwise min / max  --  arith.minimumf / maximumf produce IEEE-min/max
  // (NaN-propagating); arith.minnumf / maxnumf are the numeric variants.
  static const std::map<llvm::StringRef, std::string> minmax_ops = {
      {"arith.minimumf", "min"}, {"arith.maximumf", "max"},
      {"arith.minnumf", "min"},  {"arith.maxnumf", "max"},
      {"arith.minsi", "min"},    {"arith.maxsi", "max"},
      {"arith.minui", "min"},    {"arith.maxui", "max"},
  };
  if (auto it = minmax_ops.find(nm);
      it != minmax_ops.end() && def->getNumOperands() == 2) {
    return it->second + "(" + buildExpr(def->getOperand(0), d + 1) + ", " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  // Elementwise math intrinsics -> bare Python names.  DaCe's tasklet
  // codegen maps ``sin``/``cos``/... to ``dace::math::sin`` etc. via
  // ``_ALLOWED_MODULES`` in ``dace/dtypes.py``.  The ``f`` suffix Flang
  // uses (absf / powf / ...) is stripped because the runtime wrappers
  // overload on the operand's type.
  static const std::map<llvm::StringRef, std::string> unary_math = {
      {"math.sin", "sin"},
      {"math.cos", "cos"},
      {"math.tan", "tan"},
      {"math.asin", "asin"},
      {"math.acos", "acos"},
      {"math.atan", "atan"},
      {"math.sinh", "sinh"},
      {"math.cosh", "cosh"},
      {"math.tanh", "tanh"},
      {"math.exp", "exp"},
      {"math.log", "log"},
      {"math.log10", "log10"},
      {"math.sqrt", "sqrt"},
      {"math.absf", "abs"},
      {"math.absi", "abs"},
      {"math.floor", "floor"},
      {"math.ceil", "ceil"},
      {"math.erf", "erf"},
      {"math.erfc", "erfc"},
      // ``llvm.intr.<op>``  --  LLVM-dialect intrinsic ops Flang uses
      // for some unary math (ANINT -> ``llvm.intr.round``, AINT
      // -> ``llvm.intr.trunc`` on some kinds, etc.).  These are
      // OPS, not function calls; the ``fir::CallOp`` table below
      // handles the ``fir.call @llvm.<op>.f{32,64}`` shape.
      {"llvm.intr.round", "round"},
      {"llvm.intr.trunc", "trunc"},
      {"llvm.intr.floor", "floor"},
      {"llvm.intr.ceil", "ceil"},
      {"llvm.intr.fabs", "abs"},
      {"llvm.intr.sqrt", "sqrt"},
      {"llvm.intr.exp", "exp"},
      {"llvm.intr.log", "log"},
      {"llvm.intr.sin", "sin"},
      {"llvm.intr.cos", "cos"},
  };
  if (auto it = unary_math.find(nm);
      it != unary_math.end() && def->getNumOperands() == 1) {
    return it->second + "(" + buildExpr(def->getOperand(0), d + 1) + ")";
  }

  // Exponentiation ``a ** b``.  Flang's four variants
  // (``math.fpowi`` float**int, ``math.powf`` float**float,
  // ``math.powi`` / ``math.ipowi`` int**int) all surface as the
  // Python ``**`` operator.  A downstream SDFG-level simplify pass
  // recognises ``**`` and rewrites it based on the tasklet's
  // input/output types  --  no variant marker needed at this layer.
  static const std::set<llvm::StringRef> pow_ops = {
      "math.fpowi",
      "math.powf",
      "math.powi",
      "math.ipowi",
  };
  if (pow_ops.count(nm) && def->getNumOperands() == 2) {
    return "(" + buildExpr(def->getOperand(0), d + 1) + " ** " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  // ``hlfir.no_reassoc`` is a transparency wrapper Flang emits around
  // parenthesised subexpressions to prevent the optimizer from
  // reassociating them across ``**`` / ``+`` boundaries.  For our
  // purposes it's a passthrough  --  recurse into its single operand so
  // we don't strand ``pow`` / ``addf`` results as ``?``.
  if (nm == "hlfir.no_reassoc" && def->getNumOperands() == 1) {
    return buildExpr(def->getOperand(0), d + 1);
  }

  static const std::map<llvm::StringRef, std::string> binary_math = {
      {"math.atan2", "atan2"},
      // Fortran ``SIGN(a, b)`` on float operands lowers to
      // ``math.copysign``; ``dace::math::copysign`` resolves at the
      // tasklet codegen layer.  Integer SIGN goes through the
      // generic ``arith.select`` ternary fallback (predicate-driven
      // min/max idiom shape).
      {"math.copysign", "copysign"},
  };
  if (auto it = binary_math.find(nm);
      it != binary_math.end() && def->getNumOperands() == 2) {
    return it->second + "(" + buildExpr(def->getOperand(0), d + 1) + ", " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  // Runtime / LLVM intrinsic calls that Flang sometimes emits for
  // intrinsics it doesn't lower to a ``math.*`` op.  Mapped to bare
  // Python names so DaCe's tasklet codegen routes them through
  // ``dace::math::*`` (or stdlib ``math.*``) the same way ``unary_math``
  // does for the math-dialect form.
  //
  // Notable cases:
  //   * ``math.sinh`` / ``math.cosh`` / ``math.tanh`` exist but Flang
  //     occasionally still emits ``fir.call @sinh``  --  recognise both.
  //   * Fortran ``MOD`` / ``MODULO`` lower to ``_FortranAMod*Real{4,8}``
  //     runtime calls; the Python ``math.fmod`` matches Fortran ``MOD``
  //     (truncated quotient) and a ``(a - b * floor(a/b))`` formula
  //     matches ``MODULO`` (floored quotient).
  //   * ``NINT(x)`` lowers to ``llvm.lround``; ``AINT(x)`` to
  //     ``llvm.trunc``; both are supported by DaCe's tasklet codegen
  //     when surfaced as ``round`` / ``trunc`` Python calls.
  if (auto call = mlir::dyn_cast<fir::CallOp>(def)) {
    auto callee = call.getCallee();
    if (callee) {
      llvm::StringRef cname = callee->getRootReference().getValue();
      // Single-arg pass-through to a Python identifier (math /
      // bare runtime calls).
      static const std::map<llvm::StringRef, std::string> unary_calls = {
          {"sinh", "sinh"},
          {"cosh", "cosh"},
          {"tanh", "tanh"},
          {"asinh", "asinh"},
          {"acosh", "acosh"},
          {"atanh", "atanh"},
          {"asin", "asin"},
          {"acos", "acos"},
          {"atan", "atan"},
          {"sin", "sin"},
          {"cos", "cos"},
          {"tan", "tan"},
          {"exp", "exp"},
          {"log", "log"},
          {"log10", "log10"},
          {"sqrt", "sqrt"},
          {"fabs", "abs"},
          // ``f``-suffixed f32 runtime variants Flang emits for
          // single-precision args: ``sinhf``, ``coshf``, etc.
          // Without these, ``real(4)`` SINH/COSH/TANH lowerings
          // hit the ``?`` fallback and the tasklet body fails to
          // parse.
          {"sinhf", "sinh"},
          {"coshf", "cosh"},
          {"tanhf", "tanh"},
          {"asinhf", "asinh"},
          {"acoshf", "acosh"},
          {"atanhf", "atanh"},
          {"asinf", "asin"},
          {"acosf", "acos"},
          {"atanf", "atan"},
          {"sinf", "sin"},
          {"cosf", "cos"},
          {"tanf", "tan"},
          {"expf", "exp"},
          {"logf", "log"},
          {"log10f", "log10"},
          {"sqrtf", "sqrt"},
          {"fabsf", "abs"},
          // C99 complex math runtime  --  flang lowers Fortran
          // SIN/COS/EXP/LOG/SQRT/ABS on COMPLEX(8) to ``c<func>``
          // and on COMPLEX(4) to ``c<func>f``.  DaCe's tasklet
          // codegen has Python ``cmath``-equivalent dispatch via
          // the same bare names.
          {"csin", "sin"},
          {"ccos", "cos"},
          {"ctan", "tan"},
          {"csinh", "sinh"},
          {"ccosh", "cosh"},
          {"ctanh", "tanh"},
          {"casin", "asin"},
          {"cacos", "acos"},
          {"catan", "atan"},
          {"cexp", "exp"},
          {"clog", "log"},
          {"csqrt", "sqrt"},
          {"cabs", "abs"},
          {"csinf", "sin"},
          {"ccosf", "cos"},
          {"ctanf", "tan"},
          {"csinhf", "sinh"},
          {"ccoshf", "cosh"},
          {"ctanhf", "tanh"},
          {"casinf", "asin"},
          {"cacosf", "acos"},
          {"catanf", "atan"},
          {"cexpf", "exp"},
          {"clogf", "log"},
          {"csqrtf", "sqrt"},
          {"cabsf", "abs"},
          // AINT / ANINT  --  same-kind real return, value-only round/trunc.
          {"llvm.trunc.f64", "trunc"},
          {"llvm.trunc.f32", "trunc"},
          {"llvm.floor.f64", "floor"},
          {"llvm.floor.f32", "floor"},
          {"llvm.ceil.f64", "ceil"},
          {"llvm.ceil.f32", "ceil"},
          {"llvm.round.f64", "round"},
          {"llvm.round.f32", "round"},
          {"llvm.fabs.f64", "abs"},
          {"llvm.fabs.f32", "abs"},
      };
      if (auto it = unary_calls.find(cname);
          it != unary_calls.end() && call.getNumOperands() >= 1) {
        return it->second + "(" + buildExpr(call.getOperand(0), d + 1) + ")";
      }
      // Type-converting casts  --  Fortran NINT(x) / INT(x).
      // Flang emits ``llvm.lround.i{32,64}.f{32,64}`` for NINT
      // (rounded-to-nearest, then truncating cast).  Render as
      // ``dace.int{32,64}(round(x))`` so the rounding stays
      // explicit and the cast lowers to ``static_cast<int{32,64}>``
      // in the C++ codegen.  Plain INT(x) lowers separately via
      // ``fir.convert`` (transparent here) and an integer cast on
      // the Python side; nothing extra needed for that.
      static const std::map<llvm::StringRef, std::string> cast_calls = {
          // ``NINT`` (and ``IDNINT``)  --  round-to-nearest then cast.
          {"llvm.lround.i32.f64", "int32"},
          {"llvm.lround.i32.f32", "int32"},
          {"llvm.lround.i64.f64", "int64"},
          {"llvm.lround.i64.f32", "int64"},
          // ``llvm.lrint`` (round-to-nearest under current rounding
          // mode) -- Fortran ``NINT`` may lower to this on some
          // targets.  Same Python rendering.
          {"llvm.lrint.i32.f64", "int32"},
          {"llvm.lrint.i32.f32", "int32"},
          {"llvm.lrint.i64.f64", "int64"},
          {"llvm.lrint.i64.f32", "int64"},
      };
      if (auto it = cast_calls.find(cname);
          it != cast_calls.end() && call.getNumOperands() >= 1) {
        return it->second + "(round(" + buildExpr(call.getOperand(0), d + 1) +
               "))";
      }
      // ``AINT(x)`` truncating to a float result -- LLVM emits
      // ``llvm.trunc.f{32,64}``.  Render as ``float{32,64}(int(x))``
      // so the integer cast truncates and the float widening
      // restores the kind.  Likewise for ``ANINT`` -> ``llvm.round``,
      // ``FLOOR`` -> ``llvm.floor``, ``CEILING`` -> ``llvm.ceil``.
      static const std::map<llvm::StringRef, std::string> float_round = {
          {"llvm.trunc.f64", "trunc"},
          {"llvm.trunc.f32", "trunc"},
          {"llvm.round.f64", "round"},
          {"llvm.round.f32", "round"},
          {"llvm.floor.f64", "floor"},
          {"llvm.floor.f32", "floor"},
          {"llvm.ceil.f64", "ceil"},
          {"llvm.ceil.f32", "ceil"},
          {"llvm.rint.f64", "round"},
          {"llvm.rint.f32", "round"},
          {"llvm.nearbyint.f64", "round"},
          {"llvm.nearbyint.f32", "round"},
      };
      if (auto it = float_round.find(cname);
          it != float_round.end() && call.getNumOperands() >= 1) {
        return it->second + "(" + buildExpr(call.getOperand(0), d + 1) + ")";
      }
      // Complex division  --  flang lowers ``a / b`` on COMPLEX(8) to
      // ``__divdc3(re_a, im_a, re_b, im_b)`` (and ``__divsc3`` for
      // COMPLEX(4)) for overflow-safe Smith's algorithm.  The 4
      // reals come from ``fir.extract_value`` ops on the loaded
      // complex operands; reconstruct the original complex
      // operand identities and emit ``(complex_a / complex_b)``
      // at the tasklet level.
      if ((cname == "__divdc3" || cname == "__divsc3") &&
          call.getNumOperands() == 4) {
        auto extractSource = [](mlir::Value re, mlir::Value im) -> mlir::Value {
          auto reOp =
              mlir::dyn_cast_or_null<fir::ExtractValueOp>(re.getDefiningOp());
          auto imOp =
              mlir::dyn_cast_or_null<fir::ExtractValueOp>(im.getDefiningOp());
          if (!reOp || !imOp) return {};
          if (reOp.getAdt() != imOp.getAdt()) return {};
          return reOp.getAdt();
        };
        auto srcA = extractSource(call.getOperand(0), call.getOperand(1));
        auto srcB = extractSource(call.getOperand(2), call.getOperand(3));
        if (srcA && srcB) {
          return "(" + buildExpr(srcA, d + 1) + " / " + buildExpr(srcB, d + 1) +
                 ")";
        }
      }
      // Two-arg ATAN2 runtime fallback.
      if (cname == "atan2" && call.getNumOperands() >= 2) {
        return "atan2(" + buildExpr(call.getOperand(0), d + 1) + ", " +
               buildExpr(call.getOperand(1), d + 1) + ")";
      }
      // Fortran MOD on real operands  --  truncated-quotient
      // remainder.  Maps directly to ``std::fmod`` (in ``<cmath>``,
      // pulled in via ``<dace/dace.h>``); integer MOD lowers to
      // ``arith.remsi`` and never reaches this fir.call branch.
      if ((cname == "_FortranAModReal4" || cname == "_FortranAModReal8") &&
          call.getNumOperands() >= 2) {
        return "fmod(" + buildExpr(call.getOperand(0), d + 1) + ", " +
               buildExpr(call.getOperand(1), d + 1) + ")";
      }
      // Fortran SCALE(x, n)  --  returns ``x * 2^n``.  Maps to
      // ``dace::math::ldexp`` (templated; ``std::ldexp``
      // internally).  Runtime-call signature is ``(x, n,
      // src_file_ptr, src_line)``  --  first two operands are
      // semantic.
      if ((cname == "_FortranAScale4" || cname == "_FortranAScale8") &&
          call.getNumOperands() >= 2) {
        return "ldexp(" + buildExpr(call.getOperand(0), d + 1) + ", " +
               buildExpr(call.getOperand(1), d + 1) + ")";
      }
      // Fortran EXPONENT(x)  --  returns ``e`` such that
      // ``x = mantissa * 2^e`` with ``0.5 <= |mantissa| < 1``.
      // ``dace::math::ilogb`` provides this via ``std::frexp``
      // (returns ``int`` directly so callers can use the result
      // in a tasklet-integer context).
      if ((cname == "_FortranAExponent4_4" || cname == "_FortranAExponent8_4" ||
           cname == "_FortranAExponent4_8" || cname == "_FortranAExponent8_8" ||
           cname == "_FortranAExponent4" || cname == "_FortranAExponent8") &&
          call.getNumOperands() >= 1) {
        return "ilogb(" + buildExpr(call.getOperand(0), d + 1) + ")";
      }
      // Fortran MODULO  --  floored-quotient remainder.
      // ``dace::math::floor_mod`` is the templated helper (uses
      // ``py_mod`` internally; ``floor`` for floats, sign-aware
      // ``((a%b)+b)%b`` for ints).  Required because Python's
      // ``%`` on int floors but C++'s ``%`` on int truncates.
      if ((cname == "_FortranAModuloReal4" || cname == "_FortranAModuloReal8" ||
           cname == "_FortranAModuloInteger4" ||
           cname == "_FortranAModuloInteger8") &&
          call.getNumOperands() >= 2) {
        return "floor_mod(" + buildExpr(call.getOperand(0), d + 1) + ", " +
               buildExpr(call.getOperand(1), d + 1) + ")";
      }
      // Fortran ``base ** exponent`` lowers to a runtime ``pow``
      // helper when the operands are typed combinations the IEEE
      // ``math.powf`` op can't cover  --  complex base, mixed
      // kinds, or any integer-exponent shape Flang opts to send
      // through the runtime.  The Fortran-runtime naming convention
      // is ``_FortranA<base-kind><exp-kind>``:
      //
      //   * ``z`` = complex<f64>, ``c`` = complex<f32>
      //   * ``d`` = f64,           ``s`` = f32
      //   * ``i`` = i32,           ``k`` = i64
      //
      // and pairs e.g. ``_FortranAzpowi`` = ``complex(8) ** int(4)``,
      // ``_FortranAdpowk`` = ``real(8) ** int(8)``.  All take
      // ``(base, exponent)`` and return the base's type.
      //
      // Python's ``**`` covers each shape because DaCe lowers it
      // through ``std::pow`` overloads at codegen time -- one handler
      // suffices for every variant.  QE's
      // ``(0.D0, -1.D0) ** nhtol(ih, nt)`` surfaces this as
      // ``_FortranAzpowi`` and previously yielded a ``?`` tasklet
      // body because the ``fir.call`` had no handler.
      if ((cname == "_FortranAzpowi" || cname == "_FortranAzpowk" ||
           cname == "_FortranAcpowi" || cname == "_FortranAcpowk" ||
           cname == "_FortranAdpowi" || cname == "_FortranAdpowk" ||
           cname == "_FortranAspowi" || cname == "_FortranAspowk") &&
          call.getNumOperands() >= 2) {
        return "(" + buildExpr(call.getOperand(0), d + 1) + " ** " +
               buildExpr(call.getOperand(1), d + 1) + ")";
      }
    }
  }

  // fir.convert: same-family kind casts (i32->i64, f32->f64, i64->f64)
  // are transparent  --  Fortran's KIND coercion semantics flow through
  // the tasklet's operand types so the C++ codegen widens for free.
  // Cross-family casts (float <-> int) are NOT transparent: Fortran's
  // INT(x) / NINT(x) / DBLE(x) / REAL(x) carry semantic intent (cast
  // truncates, NINT rounds, DBLE widens) that the bridge must
  // surface as an explicit ``dace.<ty>(...)`` call so the codegen
  // emits the right ``static_cast``.
  if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) {
    auto inT = conv.getValue().getType();
    auto outT = conv.getRes().getType();
    bool inIsInt = inT.isInteger(8) || inT.isInteger(16) || inT.isInteger(32) ||
                   inT.isInteger(64);
    bool outIsInt = outT.isInteger(8) || outT.isInteger(16) ||
                    outT.isInteger(32) || outT.isInteger(64);
    bool inIsFloat = mlir::isa<mlir::FloatType>(inT);
    bool outIsFloat = mlir::isa<mlir::FloatType>(outT);
    // Float -> integer: explicit truncating cast.  Use ``dace.intN``
    // so the C++ codegen lowers via ``static_cast<int{32,64}>``.
    if (inIsFloat && outIsInt) {
      const char *cast = outT.isInteger(64) ? "int64" : "int32";
      return std::string(cast) + "(" + buildExpr(conv.getValue(), d + 1) + ")";
    }
    // Integer -> float: same shape  --  codegen will widen at the
    // arithmetic site.  Tag with ``float64`` / ``float32`` so the
    // intent is explicit when the surrounding op is integer too.
    if (inIsInt && outIsFloat) {
      const char *cast = mlir::cast<mlir::FloatType>(outT).getWidth() == 32
                             ? "float32"
                             : "float64";
      return std::string(cast) + "(" + buildExpr(conv.getValue(), d + 1) + ")";
    }
    // Float -> wider float (f32 -> f64): wrap in an explicit
    // ``dace.float32(...)`` BEFORE the widening so the inner
    // expression's f32 arithmetic rounds at f32 precision.  In
    // C++ codegen, ``static_cast<float>(double_val)`` rounds to
    // the nearest f32, which matches Fortran's ``real(4)``
    // semantics  --  ``5.5 + epsilon(1.0)`` evaluates to ``5.5``
    // exactly because the epsilon is below f32's ulp at 5.5.
    // Without this wrap the C++ promotes both operands to double
    // and gives ``5.5 + 1.19e-7 = 5.5000001192...``.  Same-width
    // converts (f64 -> f64) stay transparent.
    if (inIsFloat && outIsFloat) {
      auto inW = mlir::cast<mlir::FloatType>(inT).getWidth();
      auto outW = mlir::cast<mlir::FloatType>(outT).getWidth();
      if (inW < outW && !kSuppressFloatCast) {
        const char *cast = inW == 32 ? "float32" : "float64";
        return std::string(cast) + "(" + buildExpr(conv.getValue(), d + 1) +
               ")";
      }
      if (inW < outW) return buildExpr(conv.getValue(), d + 1);
    }
    // Same width or float -> narrower float (truncating cast)  --
    // transparent (the underlying expression already has the
    // narrower type, or the narrowing is desired).
    return buildExpr(conv.getValue(), d + 1);
  }

  // MLIR bool/int casts used by lift-cf-to-scf when threading keep-going
  // flags through scf.if yields.  All transparent: the synthetic scalars
  // we mint for scf.if results already hold 0 / 1, and Python handles
  // ``0 != 0`` as False / ``1 != 0`` as True uniformly.
  if (nm == "arith.trunci" || nm == "arith.extui" || nm == "arith.extsi") {
    if (def->getNumOperands() == 1) return buildExpr(def->getOperand(0), d + 1);
  }

  // ``fir.is_present``  --  Fortran's ``present(x)`` on an OPTIONAL dummy.
  // Used as an integer (``res(1) = present(a)`` after the implicit
  // i1->i32 widening) as well as inside a guarding condition (handled
  // by buildBoolExpr).  Both sites trace through the same helper.
  if (auto isp = mlir::dyn_cast<fir::IsPresentOp>(def)) {
    auto e = lowerIsPresent(isp.getVal());
    if (!e.empty()) return e;
  }

  // Comparisons flowing into integer casts (``extui %cmp : i1 to i32``
  // yielded to a scf.if result) need to produce a usable expression,
  // not ``?``.  Defer to buildBoolExpr which understands cmpf / cmpi.
  if (nm == "arith.cmpf" || nm == "arith.cmpi") {
    auto b = buildBoolExpr(val, d + 1);
    if (b != "?") return b;
  }
  // ``xori %x, true`` -> logical NOT; any other i1 xori -> Python ``!=``.
  // For non-i1 operands, ``xori`` is the Fortran ``ieor(a,b)`` bitwise op
  // and lowers to Python ``^``.  MLIR's ``arith.constant true`` stores
  // the i1 value as -1 (all-bits set) on most targets, so match 1 / -1.
  if (nm == "arith.xori" && def->getNumOperands() == 2) {
    bool i1_operands = def->getOperand(0).getType().isInteger(1);
    auto *rhs = def->getOperand(1).getDefiningOp();
    if (i1_operands) {
      if (auto c = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(rhs))
        if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
          auto v = ia.getInt();
          if (v == 1 || v == -1)
            return "(not " + buildExpr(def->getOperand(0), d + 1) + ")";
        }
      return "(" + buildExpr(def->getOperand(0), d + 1) +
             " != " + buildExpr(def->getOperand(1), d + 1) + ")";
    }
    // Bitwise XOR: Fortran ``ieor(a, b)`` and the bitwise-NOT idiom
    // ``xori a, -1`` (Flang's lowering of ``ibclr``'s mask
    // construction step).
    return "(" + buildExpr(def->getOperand(0), d + 1) + " ^ " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  // Bitwise AND / OR  --  for non-i1 operands these are ``iand`` / ``ior``
  // (and the building blocks of ``ibclr`` / ``ibset`` / ``ibits`` /
  // ``btest``).  i1 versions are Fortran ``.AND.`` / ``.OR.`` chains;
  // route through ``buildBoolExpr`` so they render as Python
  // ``... and ...`` / ``... or ...`` -- the typical shape is a
  // ``LOGICAL :: llo1`` cached as ``llo1 = (a>b) .AND. (c>d) .AND.
  // ...``, where the surrounding ``fir.convert`` to ``!fir.logical<K>``
  // pulls us through ``buildExpr`` rather than ``buildBoolExpr``.
  // ``NoSubscriptGuard`` keeps array reads as bare identifiers -- we
  // are inside a ``buildExpr`` call destined for a tasklet body, so
  // the cmpf / cmpi operands must NOT carry ``arr[idx]`` subscripts
  // (emit_tasklet rewrites bare names into ``_in_arr_N`` connectors
  // and wires subscripts via memlets).
  if ((nm == "arith.andi" || nm == "arith.ori") && def->getNumOperands() == 2) {
    if (def->getOperand(0).getType().isInteger(1)) {
      // Bare-name mode for the cmp-leaf array reads (we're inside
      // ``buildExpr``, the tasklet renderer; emit_tasklet wires
      // ``a[i]`` subscripts through memlets and rewrites the
      // bare ``a`` to a ``_in_a_N`` connector).
      NoSubscriptGuard _g;
      auto b = buildBoolExpr(val, d + 1);
      if (b != "?") return b;
    } else {
      const char *op = (nm == "arith.andi") ? " & " : " | ";
      return "(" + buildExpr(def->getOperand(0), d + 1) + op +
             buildExpr(def->getOperand(1), d + 1) + ")";
    }
  }

  // Bit shifts  --  Fortran ``ishft`` (and the building blocks of
  // ``ibset`` / ``ibclr`` / ``ibits``).  ``arith.shrsi`` is the
  // arithmetic (sign-extending) right shift and maps to ``>>`` directly.
  // ``arith.shli`` / ``arith.shrui`` are the *logical* (bit-pattern,
  // zero-fill) shifts Flang emits for ``ISHFT``: a signed ``>>`` would
  // sign-extend instead of zero-filling -- wrong for a negative operand
  // (``ishft(-182, -2)``) -- so route them through the
  // ``logical_left_shift`` / ``logical_right_shift`` runtime helpers,
  // which shift via the unsigned type.
  if (nm == "arith.shrsi" && def->getNumOperands() == 2) {
    return "(" + buildExpr(def->getOperand(0), d + 1) + " >> " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }
  if ((nm == "arith.shli" || nm == "arith.shrui") &&
      def->getNumOperands() == 2) {
    const char *fn = (nm == "arith.shli") ? "logical_left_shift" : "logical_right_shift";
    return std::string(fn) + "(" + buildExpr(def->getOperand(0), d + 1) + ", " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  // Integer remainder  --  ``arith.remsi`` / ``arith.remui``  --  used by some
  // Fortran ``mod`` lowerings on integers.
  if ((nm == "arith.remsi" || nm == "arith.remui") &&
      def->getNumOperands() == 2) {
    return "(" + buildExpr(def->getOperand(0), d + 1) + " % " +
           buildExpr(def->getOperand(1), d + 1) + ")";
  }

  // Scalar min / max idiom: Flang lowers ``min(a, b)`` on f32/f64 to
  // ``arith.select(arith.cmpf olt, a, b)`` (and ``max`` via ``ogt``).
  // Recognise that shape so the tasklet code gets a bare min/max call.
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    auto *cdef = sel.getCondition().getDefiningOp();
    if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpFOp>(cdef)) {
      auto pred = cmp.getPredicate();
      using P = mlir::arith::CmpFPredicate;
      const char *fn = nullptr;
      if (pred == P::OLT || pred == P::ULT)
        fn = "min";
      else if (pred == P::OGT || pred == P::UGT)
        fn = "max";
      if (fn && cmp.getLhs() == sel.getTrueValue() &&
          cmp.getRhs() == sel.getFalseValue()) {
        return std::string(fn) + "(" + buildExpr(cmp.getLhs(), d + 1) + ", " +
               buildExpr(cmp.getRhs(), d + 1) + ")";
      }
    }
    // Same idiom for integer min / max via arith.cmpi.
    if (auto cmp = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(cdef)) {
      auto pred = cmp.getPredicate();
      using P = mlir::arith::CmpIPredicate;
      const char *fn = nullptr;
      if (pred == P::slt || pred == P::ult)
        fn = "min";
      else if (pred == P::sgt || pred == P::ugt)
        fn = "max";
      if (fn && cmp.getLhs() == sel.getTrueValue() &&
          cmp.getRhs() == sel.getFalseValue()) {
        return std::string(fn) + "(" + buildExpr(cmp.getLhs(), d + 1) + ", " +
               buildExpr(cmp.getRhs(), d + 1) + ")";
      }
    }
    // Inlined integer Fortran MODULO collapse:
    //
    //   r  = arith.remsi a, b              ; truncated remainder
    //   x  = arith.xori  a, b              ; signed-XOR (sign test)
    //   c1 = arith.cmpi slt, x, 0          ; (a^b) < 0  -> signs differ
    //   c2 = arith.cmpi ne, r, 0           ; r != 0
    //   c  = arith.andi c1, c2
    //   ab = arith.addi r, b               ; (r + b)
    //   r' = arith.select c, ab, r         ; floored result
    //
    // Flang inlines this for ``MODULO(int, int)`` instead of
    // emitting a runtime call.  Recognising the shape and emitting
    // a single ``floor_mod(a, b)`` keeps the tasklet expression
    // tight (one connector per operand instead of nine) and uses
    // the existing ``dace::math::floor_mod`` helper.
    do {
      auto trueOp = sel.getTrueValue().getDefiningOp();
      auto falseOp = sel.getFalseValue().getDefiningOp();
      auto condOp = sel.getCondition().getDefiningOp();
      auto add = mlir::dyn_cast_or_null<mlir::arith::AddIOp>(trueOp);
      auto rem = mlir::dyn_cast_or_null<mlir::arith::RemSIOp>(falseOp);
      auto andi = mlir::dyn_cast_or_null<mlir::arith::AndIOp>(condOp);
      if (!add || !rem || !andi) break;
      // add = (rem, b)
      auto add_lhs = add.getLhs().getDefiningOp();
      auto add_rem = mlir::dyn_cast_or_null<mlir::arith::RemSIOp>(add_lhs);
      if (!add_rem || add_rem.getResult() != rem.getResult()) break;
      mlir::Value a = rem.getLhs();
      mlir::Value b = rem.getRhs();
      if (add.getRhs() != b) break;
      // andi = (cmpi ne r 0, cmpi slt (xori a b) 0)  -- order-agnostic
      auto cm0 = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
          andi.getLhs().getDefiningOp());
      auto cm1 = mlir::dyn_cast_or_null<mlir::arith::CmpIOp>(
          andi.getRhs().getDefiningOp());
      if (!cm0 || !cm1) break;
      auto isNeR = [&](mlir::arith::CmpIOp c) {
        return c.getPredicate() == mlir::arith::CmpIPredicate::ne &&
               c.getLhs() == rem.getResult();
      };
      auto isSltXori = [&](mlir::arith::CmpIOp c) {
        if (c.getPredicate() != mlir::arith::CmpIPredicate::slt) return false;
        auto x = mlir::dyn_cast_or_null<mlir::arith::XOrIOp>(
            c.getLhs().getDefiningOp());
        return x && ((x.getLhs() == a && x.getRhs() == b) ||
                     (x.getLhs() == b && x.getRhs() == a));
      };
      if (!((isNeR(cm0) && isSltXori(cm1)) || (isNeR(cm1) && isSltXori(cm0))))
        break;
      return "floor_mod(" + buildExpr(a, d + 1) + ", " + buildExpr(b, d + 1) +
             ")";
    } while (false);
    // Generic ternary fallback  --  Fortran ``MERGE(t, f, mask)`` lowers
    // to a bare ``arith.select`` (and the SIZE/LBOUND/UBOUND clamps
    // Flang inlines as ``(0 > n) ? 0 : n`` use ``arith.select`` on a
    // cmpi whose operand order doesn't match the min/max idiom).
    // Render as Python ``(t if cond else f)``; the C++ codegen
    // accepts the conditional expression.
    //
    // ``buildExpr`` itself is the tasklet renderer (bare names),
    // so the cond's leaves must also be bare -- ``emit_tasklet``
    // will rewrite the connectors and wire subscripts via memlets.
    // Set ``NoSubscriptGuard`` for the ``buildBoolExpr`` call so
    // every leaf threads through bare-names mode, matching the
    // outer ``buildExpr`` calls for the select's true / false sides.
    std::string condExpr;
    {
      NoSubscriptGuard _g;
      condExpr = buildBoolExpr(sel.getCondition(), d + 1);
    }
    if (condExpr == "?") condExpr = buildExpr(sel.getCondition(), d + 1);
    return "(" + buildExpr(sel.getTrueValue(), d + 1) + " if " + condExpr +
           " else " + buildExpr(sel.getFalseValue(), d + 1) + ")";
  }

  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    auto mem = ld.getMemref();
    // Fortran 2008 complex-part accessor: ``z%re`` / ``z%im`` lower
    // to ``hlfir.designate %z {complex_part = false/true}`` -- a
    // SELECTOR on a COMPLEX value, NOT a struct-field component.
    // ``traceToDecl`` below would walk through it to the base ``z``
    // and silently drop the part extraction (the generated tasklet
    // became ``_out = _in_z`` instead of ``_out = _in_z.real()``,
    // a ``complex128 -> double`` codegen type error).  Render the
    // method-call form ``(<z>.real())`` / ``(<z>.imag())`` -- same
    // shape the ``fir.extract_value`` complex handler emits for
    // ``REAL(z)`` / ``AIMAG(z)``, which ``cppunparse`` maps to
    // ``std::complex<T>::real()`` / ``::imag()``.
    if (auto dg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(
            mem.getDefiningOp())) {
      if (auto cp = dg.getComplexPart()) {
        // ``dg.getMemref()`` is the COMPLEX value's declare addr (a
        // ``fir.ref<complex<T>>``) for the scalar case, or an
        // element/section designate for ``z(i)%re``.  ``traceToDecl``
        // resolves the addr to the Fortran name; for the array case
        // it walks the element designate to the base array, and the
        // index lives in the AccessInfo the read-collector emits.
        // Fall back to ``buildExpr`` only if ``traceToDecl`` can't
        // name it (e.g. a complex temporary).
        std::string base = traceToDecl(dg.getMemref());
        if (base.empty()) base = buildExpr(dg.getMemref(), d + 1);
        if (!base.empty() && base != "?")
          return "(" + base + (*cp ? ".imag()" : ".real()") + ")";
        return "?";
      }
    }
    // Pass the designate-or-load result directly to ``traceToDecl``.
    // It walks through ``hlfir.designate`` correctly: section /
    // element designates fall through to the parent name, struct-
    // field designates (component attr set) build the flattened
    // ``<parent>_<member>`` name (the fix at trace_utils.cpp from
    // commit 25f8e83).  Previously this branch short-circuited
    // with ``traceToDecl(dg.getMemref())`` which BYPASSED the
    // component-aware walk and returned the struct base name
    // ``g`` instead of ``g_c`` for ``g % c`` scalar reads --
    // leaking ``g`` as a free symbol into the generated tasklet.
    auto n = traceToDecl(mem);
    if (!n.empty()) return n;
    // Bare fir.alloca without a hlfir.declare  --  mint a synthetic
    // scalar name.  Flang uses these as scratch counters for
    // lift-cf-to-scf's lowered DO / DO-WHILE / DO+EXIT shapes.
    if (auto *md = mem.getDefiningOp())
      if (mlir::isa<fir::AllocaOp>(md)) return allocaSynthName(mem);
  }

  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    if (auto f = mlir::dyn_cast<mlir::FloatAttr>(cst.getValue())) {
      bool isF32 = false;
      if (auto ft = mlir::dyn_cast<mlir::FloatType>(cst.getType()))
        isF32 = ft.getWidth() == 32;
      std::string lit;
      if (isF32) {
        // The constant is genuinely f32.  Print the SHORTEST
        // decimal that round-trips to that f32 value (binary32
        // needs at most 9 significant digits) instead of the
        // f64-widened ``0.10000000149011612``.  Wrapped in
        // ``dace.float32(...)`` the two are bit-identical, but
        // the short form stays close to the Fortran source.
        float fv = static_cast<float>(f.getValueAsDouble());
        for (int prec = 1; prec <= 9; ++prec) {
          std::ostringstream o;
          o << std::setprecision(prec) << fv;
          if (std::strtof(o.str().c_str(), nullptr) == fv) {
            lit = o.str();
            break;
          }
        }
        if (lit.empty()) {  // non-finite or unexpected
          std::ostringstream o;
          o << std::setprecision(9) << fv;
          lit = o.str();
        }
      } else {
        // Print the SHORTEST decimal that round-trips to the same
        // binary64 value.  17 digits is the worst-case upper bound
        // (Steele & White) but ``1e-15`` only needs 1 significant
        // digit -- printing all 17 (``1.0000000000000001e-15``)
        // bloats the generated C++ and surfaced as noise in graupel
        // constants the user flagged.  Try shortest -> longer and
        // accept the first that round-trips.
        double dv = f.getValueAsDouble();
        // ``-0.0`` short-circuit: IEEE 754 says ``-0.0 == +0.0`` so
        // the round-trip loop below would accept ``"0"`` -> +0.0 and
        // silently drop the sign.  But the sign IS observable in
        // ``1.0/x`` (-> -inf vs +inf), ``ATAN2(x, -1.0)`` (-> -pi vs
        // +pi), ``SIGN(y, x)`` and complex branch cuts -- so
        // well-formed Fortran code can legitimately depend on it.
        // Emit ``"-0.0"`` directly when the sign bit is set on a
        // zero value.
        if (dv == 0.0 && std::signbit(dv)) {
          lit = "-0.0";
        } else {
          for (int prec = 1; prec <= 17; ++prec) {
            std::ostringstream o;
            o << std::setprecision(prec) << dv;
            if (std::strtod(o.str().c_str(), nullptr) == dv) {
              lit = o.str();
              break;
            }
          }
          if (lit.empty()) {  // non-finite (NaN; signed inf round-trips at prec=1)
            std::ostringstream o;
            o << std::setprecision(17) << dv;
            lit = o.str();
          }
        }
      }
      // ``ostringstream`` drops the decimal point for integer-valued
      // doubles (e.g. ``0.0`` -> ``"0"``).  That makes the C++ code
      // emit a plain ``int`` literal in a float context, so
      // ``max(0.0, double_expr)`` becomes ``max(0, expr)`` and
      // compiler-side overload resolution can pick the wrong ``max``.
      // Force a trailing ``.0`` so the literal is unambiguously
      // floating-point.
      if (lit.find('.') == std::string::npos &&
          lit.find('e') == std::string::npos &&
          lit.find('E') == std::string::npos &&
          lit.find("nan") == std::string::npos &&
          lit.find("inf") == std::string::npos)
        lit += ".0";
      // Wrap f32-typed constants in ``dace.float32(...)`` so the
      // C++ codegen emits ``static_cast<float>(literal)`` instead
      // of a double literal.  Without this, DaCe upgrades every
      // float constant to f64 and a Fortran ``real(4)`` chain
      // like ``5.5 + epsilon(1.0)`` evaluates in double precision
      //  --  producing ``5.5000001192...`` instead of the f32-rounded
      // ``5.5``.  Pairs with the ``fir.convert f32->f64`` wrap so
      // both the literal and the widening cast preserve the
      // intended precision.
      if (isF32 && !kSuppressFloatCast) return "float32(" + lit + ")";
      return lit;
    }
    if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
      return std::to_string(i.getInt());
  }

  // hlfir.apply %elem, %i  --  read one element of an hlfir.elemental expr
  // at a given index.  Inline the referenced elemental's body at the
  // apply site by mapping its block args to the apply's index operands
  // via indexStack(), then recursing into the yield_element operand.
  if (auto apply = mlir::dyn_cast<hlfir::ApplyOp>(def)) {
    auto src = apply.getExpr();
    // Materialised libcall result (``matmul`` / ``transpose`` / ...):
    // ``buildElementalAssign`` has already queued the libcall AST
    // node that writes the result to a synthetic transient; render
    // the apply as just the transient's bare name so emit_tasklet
    // rewrites it to an ``_in_<tmp>_<n>`` connector.  The indexing
    // lives entirely in the AccessInfo that ``collectReads`` adds
    // for this same apply (see the matching branch there).
    if (auto *srcDef = src.getDefiningOp()) {
      auto it = kHlfirExprToTransient.find(srcDef);
      if (it != kHlfirExprToTransient.end()) {
        return it->second;
      }
    }
    if (auto *srcDef = src.getDefiningOp())
      if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(srcDef)) {
        auto &region = elem.getRegion();
        if (!region.empty()) {
          auto &block = region.front();
          auto apply_idxs = apply.getIndices();
          unsigned pushed = 0;
          // Push the apply indices onto the index stack  --  as
          // synthetic names if we have them, otherwise pass the
          // Value through resolveIndex so callers see the same
          // iter names the outer elemental already set up.
          for (unsigned i = 0;
               i < block.getNumArguments() && i < apply_idxs.size(); ++i) {
            auto name = resolveIndex(apply_idxs[i]);
            indexStack().push_back({block.getArgument(i), name});
            ++pushed;
          }
          std::string result = "?";
          for (auto &op : block)
            if (auto y = mlir::dyn_cast<hlfir::YieldElementOp>(op)) {
              result = buildExpr(y.getElementValue(), d + 1);
              break;
            }
          for (unsigned i = 0; i < pushed; ++i) indexStack().pop_back();
          return result;
        }
      }
  }

  // ``fir.rebox`` -- pointer / box descriptor rebind.  Value-wise
  // the rebox is transparent: it builds a new descriptor over the
  // same data with adjusted bounds.  For ``buildExpr`` (value
  // lookup) just walk through to the input box; the indexing /
  // descriptor adjustment is handled at the access-info layer.
  // Surfaces in QE where ``MATMUL(TRANSPOSE(vcut % a), q) / tpi``
  // produces a rebox to assemble the slab descriptor before the
  // following ops read it.
  if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
    return buildExpr(rb.getBox(), d + 1);
  }

  // ``hlfir.designate`` reached in a value context (no enclosing
  // ``fir.load``).  Typical surface: a comparison like
  // ``SUM((i - i_real)**2) > eps6`` where the SUM result is left
  // as an ``hlfir.expr`` whose materialisation (when the bridge
  // hasn't run the elemental-to-transient pre-pass) lands on a
  // designate.  QE's ``vcut_get`` hits this -- the comparison
  // rendered as ``(? > 1e-06)`` because buildExpr fell through
  // to the unhandled-op log.
  //
  // Render strategy:
  //   * No indices and a component attr (``s%y``) -> flat
  //     ``<parent>_<member>`` name (same path used by traceToDecl
  //     for component designates).
  //   * Element designate with indices -> ``<name>[idx0, idx1, ...]``
  //     so ``emit_tasklet``'s ``_rewrite_read_connectors`` consumes
  //     the brackets and binds each occurrence to its memlet.
  //   * Section (triplet) designate -> bare name; the slice is
  //     captured by the AccessInfo path, the tasklet just reads
  //     the named slab.
  //
  // This pairs the AccessInfo ``collectReads`` already builds
  // (which DOES handle designate via ``expandDesignateChain``)
  // with a matching textual form so the emitter's per-occurrence
  // counts agree.
  if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
    auto comp = dg.getComponentAttr();
    if (comp && dg.getIndices().empty()) {
      auto parent = traceToDecl(dg.getMemref());
      if (!parent.empty()) return parent + "_" + comp.getValue().str();
    }
    auto name = traceToDecl(dg.getMemref());
    if (name.empty()) name = traceToDecl(dg.getResult());
    if (name.empty()) return "?";
    auto indices = dg.getIndices();
    if (indices.empty()) return name;
    auto triplets = dg.getIsTriplet();
    bool anyTriplet = false;
    for (bool t : triplets) {
      if (t) { anyTriplet = true; break; }
    }
    if (anyTriplet) return name;
    std::string out = name + "[";
    bool first = true;
    for (auto idx : indices) {
      if (!first) out += ", ";
      out += buildExpr(idx, d + 1);
      first = false;
    }
    out += "]";
    return out;
  }

  // ``scf.if`` -- structured-if as a value-yielding expression.
  // Each region ends in an ``scf.yield`` carrying the branch's
  // result; render as a Python ternary
  // ``(then_val if cond else else_val)`` so emit_tasklet picks it up
  // at codegen.  Memlet-subset uses of ``scf.if`` results would
  // need a separate sympify-safe rendering but no test currently
  // exercises that path -- the ternary suffices for tasklet bodies.
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(def)) {
    if (ifOp.getNumResults() == 0) return "?";
    unsigned resultIdx = 0;
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      if (ifOp.getResult(i) == val) { resultIdx = i; break; }
    auto extractYield = [&](mlir::Region &region) -> std::string {
      if (region.empty()) return "?";
      auto &block = region.front();
      for (auto &op : block) {
        if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
          if (resultIdx < y.getNumOperands())
            return buildExpr(y.getOperand(resultIdx), d + 1);
        }
      }
      return "?";
    };
    std::string thenVal = extractYield(ifOp.getThenRegion());
    std::string elseVal = extractYield(ifOp.getElseRegion());
    std::string condStr = buildExpr(ifOp.getCondition(), d + 1);
    return "(" + thenVal + " if " + condStr + " else " + elseVal + ")";
  }

  // ``hlfir.all`` / ``hlfir.any`` -- whole-array boolean reductions.
  // Lower to Python ``all(...)`` / ``any(...)`` over the input
  // expression.  Used by QE's ``IF (ALL(odg(:)))`` at line 286.
  if (auto allOp = mlir::dyn_cast<hlfir::AllOp>(def)) {
    return "all(" + buildExpr(allOp.getMask(), d + 1) + ")";
  }
  if (auto anyOp = mlir::dyn_cast<hlfir::AnyOp>(def)) {
    return "any(" + buildExpr(anyOp.getMask(), d + 1) + ")";
  }

  // Unhandled HLFIR op falls through to ``?``.  Logs the op-name +
  // location to stderr (always-on, previous DACE_FORTRAN_DEBUG_BUILDEXPR
  // gate is gone) so the missing case is visible without breaking the
  // ``?``-as-sentinel protocol many tests still rely on for legitimate
  // fallback paths.  Migration to explicit throws is captured in
  // ``tasks/audit_question_mark_emissions.md``.
  {
    std::string op_name = def->getName().getStringRef().str();
    std::string loc;
    llvm::raw_string_ostream os(loc);
    def->getLoc().print(os);
    llvm::errs() << "[buildExpr unhandled-op] op=" << op_name
                 << " at " << loc << "\n";
  }
  return "?";
}

}  // namespace hlfir_bridge
