// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
//
// Cross-file API for HLFIR AST extraction: thread-local state, cross-TU function declarations, and the NoSubscriptGuard
// helper, shared by every ast/*.cpp and bridge/extract_ast.cpp. Globals are inline thread_local so the C++17
// single-definition rule holds across TUs without a separate ast_state.cpp.
#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bridge/extract_ast.h"
#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace hlfir_bridge {

/// Throws with the unhandled op's name + MLIR location instead of returning "?" silently; pass __func__ as fn.
[[noreturn]] inline void throwUnhandled(mlir::Operation* op, const char* fn) {
  std::string const opName = op ? op->getName().getStringRef().str() : "<null>";
  std::string loc;
  if (op) {
    llvm::raw_string_ostream os(loc);
    op->getLoc().print(os);
  }
  throw std::runtime_error(std::string(fn) + ": unhandled HLFIR op '" + opName + "' at " +
                           (loc.empty() ? "<unknown loc>" : loc) +
                           ".  Add a handler in the corresponding "
                           "bridge/ast/*.cpp file (search for the helper "
                           "name) or update the op coverage in "
                           "tasks/audit_question_mark_emissions.md.");
}

// ============================================================================
// Thread-local state
// ============================================================================

/// Synthetic-scalar registry for scf.if results (name __sc_<N>); reset between SDFG builds by buildAST setup.
inline thread_local int kScfValueCounter = 0;
inline thread_local llvm::DenseMap<mlir::Value, std::string> kScfValueMap;

/// Synthetic-scalar registry for un-named fir.alloca scratch ops (e.g. DO WHILE counters lowered without a surrounding
/// hlfir.declare); names are __al_<N>.
inline thread_local int kAllocaCounter = 0;
inline thread_local llvm::DenseMap<mlir::Operation*, std::string> kAllocaMap;

/// Maps a libcall-producing hlfir op (matmul/transpose/dot_product) inlined inside an elemental body to the synthetic
/// transient buildElementalAssign materialises ahead of the loop; buildExpr consults this for hlfir.apply.
inline thread_local std::map<mlir::Operation*, std::string> kHlfirExprToTransient;

/// Maps a reduction op (sum/minval/maxval/product) in an IF/loop condition to the scalar transient a Reduce library
/// node writes before the branch; buildExpr/buildExprWithSubscripts/collectReadAccesses read the bare scalar instead of
/// inline-unrolling. See materialiseCondReductions in dispatch.cpp.
inline thread_local std::map<mlir::Operation*, std::string> kCondReductionScalars;

/// Position-array registry: __sym_<arr>_<i1>_<i2>... symbol minted for each arr(consts) read as an index/bound/extent,
/// keyed by (array, per-dim 1-based indices) so multi-dim elements get distinct symbols.
inline thread_local std::map<std::pair<std::string, std::vector<int64_t>>, std::string> kPosSymbolRegistry;

/// Synthetic-transient counter used by elemental walks (__libcall_tmp_<N>).
inline thread_local int kSynthTransientCounter = 0;

/// Counter for libcall-result transients minted alongside an hlfir.assign to a section/single element.
inline thread_local int kLibTmpCounter = 0;

// ----- buildBoolExpr context flags -----------------------------------------

/// When true, buildBoolExpr's cmp branches use bare array names (elemental-body/tasklet context, wired via memlets)
/// instead of subscripted form (interstate-edge condition default).
inline thread_local bool kBoolExprNoSubscripts = false;

/// RAII guard scoping kBoolExprNoSubscripts = true so the flag restores on any exit path (early return, thrown
/// exception).
struct NoSubscriptGuard {
  bool prev;
  NoSubscriptGuard() : prev(kBoolExprNoSubscripts) { kBoolExprNoSubscripts = true; }
  ~NoSubscriptGuard() { kBoolExprNoSubscripts = prev; }
};

/// When true, buildExpr's element-read leaf renders the explicit subscript instead of the bare name; the bare form is
/// only correct inside a tasklet (subscript rides the memlet), a condition/interstate-edge expression needs it inline.
/// Set by buildExprWithSubscripts's generic fall-through so any unhandled op still keeps operand subscripts in
/// conditions (graupel MAX(q%x,...) > qmin class of bug).
inline thread_local bool kForceSubscripts = false;

/// RAII guard scoping ``kForceSubscripts = true`` (see above).
struct ForceSubscriptsGuard {
  bool prev;
  ForceSubscriptsGuard() : prev(kForceSubscripts) { kForceSubscripts = true; }
  ~ForceSubscriptsGuard() { kForceSubscripts = prev; }
};

/// When true, suppress the dace.float32(...) wrap around f32 constants/converts. Set inside
/// buildBoolExpr/buildExprWithSubscripts, whose output lands in an interstate-edge/ConditionalBlock condition parsed by
/// DaCe's symbolic engine -- which treats dace.float32 as a free symbol and raises KeyError: 'dace'. Harmless inside
/// tasklet bodies; the f32-vs-f64 difference doesn't change comparison outcomes anyway.
inline thread_local bool kSuppressFloatCast = false;

/// RAII guard scoping ``kSuppressFloatCast = true``.
struct SuppressFloatCastGuard {
  bool prev;
  SuppressFloatCastGuard() : prev(kSuppressFloatCast) { kSuppressFloatCast = true; }
  ~SuppressFloatCastGuard() { kSuppressFloatCast = prev; }
};

// ============================================================================
// Cross-file API
// ============================================================================

// All inline (ODR-safe across TUs); each function's body lives in the .cpp chunk that owns it.

/// Build a Python-syntax expression string for val (bare-name form, tasklet contexts); depth-limited against runaway
/// recursion on malformed IR. See expressions.cpp.
std::string buildExpr(mlir::Value val, int d);

/// Render an SSA value as a Fortran-1-based index/loop-bound expression; recognises arith add/sub/mul/divsi/divui,
/// MAX/MIN (incl. cmp+select lowering), constant-indexed reads (via internPosSymbol), else "?". See assigns.cpp.
std::string buildIndexExpr(mlir::Value v, int d);

/// Like buildExpr but renders array reads with full arr[idx, ...] subscripts; used by buildBoolExpr for interstate-edge
/// condition contexts. See control_flow.cpp.
std::string buildExprWithSubscripts(mlir::Value val, int d);

/// Render an i1 SSA value as a Python boolean expression (cmpf/cmpi predicates, andi/ori/xori chains, fir.is_present
/// for OPTIONAL present(x)); honours context flags for bare-name vs subscripted operand rendering. See
/// control_flow.cpp.
std::string buildBoolExpr(mlir::Value val, int d);

/// Render the dim-th index expression of an hlfir.designate, applying section/assumed-shape lower-bound rebases. See
/// expressions.cpp.
std::string buildDesignateIndexExpr(hlfir::DesignateOp dg, unsigned dim, mlir::Value idx, int depth);

/// Per-dim AccessInfo entry produced by ``expandDesignateChain``.
struct DimEntry {
  std::string var;   // identifier for AccessInfo::index_vars
  std::string expr;  // 1-based expression for AccessInfo::index_exprs
};

/// Walk an innermost hlfir.designate's parent chain (declare aliases, fir.convert reshapes, section parents) into a
/// per-original-dim (var, expr) list keyed to the underlying array's full rank, so callers match even on a rank-reduced
/// view. See elementals.cpp.
std::pair<std::string, std::vector<DimEntry>> expandDesignateChain(hlfir::DesignateOp innermost);

/// Resolve an SSA index to its source name inside an elemental body (tracked synth-iter block arg); empty string on no
/// match. See expressions.cpp.
std::string resolveIndex(mlir::Value idx);

/// Lower fir.is_present %v -> i1 to a Python expression (<name>_present for OPTIONAL dummies, else constant 0/1). See
/// expressions.cpp.
std::string lowerIsPresent(mlir::Value operand);

/// Mint or look up the synthetic name for a bare fir.alloca scratch value (no surrounding hlfir.declare). See
/// expressions.cpp.
std::string allocaSynthName(mlir::Value memref);

/// Intern an arr(consts) element read as SDFG symbol __sym_<arr>_<i1>_<i2>..., attaching a symbol_init AST node so the
/// emitter loads it on an interstate edge at SDFG entry. N-D overload for multi-dim elements, 1-D for the common arr(7)
/// case. See expressions.cpp.
std::string internPosSymbol(const std::string& array, const std::vector<int64_t>& one_based_idxs);
std::string internPosSymbol(const std::string& array, int64_t one_based_idx);

/// Capture the LHS of an hlfir.assign (bare hlfir.declare or single-element hlfir.designate) into node.target + per-dim
/// AccessInfo. See expressions.cpp.
void captureElementDesignateWrite(mlir::Value dest, ASTNode& node);

/// Render an arith::cmpi/cmpf predicate as a Python comparison operator. See control_flow.cpp.
std::string cmpiPredStr(mlir::arith::CmpIPredicate p);
std::string cmpfPredStr(mlir::arith::CmpFPredicate p);

/// Walk a block of HLFIR ops into a list of ASTNodes -- the recursive backbone of AST extraction. See elementals.cpp.
std::vector<ASTNode> buildAST(mlir::Block& block);

/// Resolve the d-th extent of a fir.shape/fir.shape_shift to a name (constant, declared symbol, or synthetic); empty if
/// unrecognised. See elementals.cpp.
std::string resolveExtent(mlir::Value shape, unsigned d);

/// Walk an SSA expression tree and append every hlfir.designate read to accesses, so buildLibCallNode's tasklet picks
/// up every input array. See elementals.cpp.
void collectReadAccesses(mlir::Value v, std::vector<AccessInfo>& accesses, int depth);

/// Map an hlfir.matmul/transpose/dot_product op to the libcall name DaCe's runtime exposes. See elementals.cpp.
const char* libcallNameForExprOp(mlir::Operation* op);

/// Render the result-type shape of an hlfir.expr<...> value as per-dim extent strings; empty vector if not an
/// hlfir.expr. See elementals.cpp.
std::vector<std::string> exprResultShape(mlir::Type ty);

/// Render the result element-type of an hlfir.expr<...> value as a numpy-style dtype string. See elementals.cpp.
std::string exprDtypeString(mlir::Type ty);

/// Push/pop (blockArg, syntheticName) pairs on the elemental index-substitution stack used by resolveIndex. See
/// expressions.cpp.
std::vector<std::pair<mlir::Value, std::string>>& indexStack();

/// Peel fir.ref/box/heap/ptr wrappers off a type. See assigns.cpp.
mlir::Type peelWrappers(mlir::Type t);

/// True iff the type peels to a fir.array<...>, or is an hlfir.expr<...> with a non-empty shape. See assigns.cpp.
bool isArrayRef(mlir::Type t);

/// True iff the value is a constant 0 (any integer/index width). See assigns.cpp.
bool isConstantZero(mlir::Value v);

/// Trace a fir.do_loop lower-bound SSA value to a constant int; -1 if not constant. See assigns.cpp.
int64_t traceLB(mlir::Value v);

/// Walk through fir.convert to find the underlying section designate (hlfir.designate with triplet indices). See
/// assigns.cpp.
hlfir::DesignateOp asSectionDesignate(mlir::Value v);

}  // namespace hlfir_bridge
