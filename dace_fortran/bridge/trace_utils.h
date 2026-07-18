// trace_utils.h -- shared SSA tracing utilities. Walks Flang's def-use chains
// (fir.convert/fir.load/arith.select/hlfir.declare) to recover Fortran names and constants from SSA values; used by
// bridge extraction and MLIR passes.

#pragma once

#include <llvm/ADT/SmallVector.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"

namespace hlfir_bridge {

/// Recursion/walk-length budgets for SSA tracing; defensive guards against pathological IR -- bumping them never
/// changes semantics, only reduces false-``?`` fallbacks.
namespace limits {

/// Max recursion depth for buildExpr/buildExprWithSubscripts/buildBoolExpr; real ICON kernels nest 40+ deep, so keep
/// generous -- unused depth costs nothing.
inline constexpr int kBuildExprDepth = 1024;

/// Max recursion depth for buildIndexExpr; shallower than general expressions but inherits the same budget.
inline constexpr int kBuildIndexExprDepth = 1024;

/// Max fir.convert chain length walked in resolveIndex (Flang stacks several for kind coercions).
inline constexpr int kConvertChainDepth = 32;

/// Max wrapper-peel depth for type unwrapping (fir.box/ref/heap/pointer); nested chains like box<ref<heap<array<...>>>>
/// are rare but legal.
inline constexpr int kTypeWrapperPeelDepth = 32;

/// Default walk budget for traceToDecl/traceConstInt (long-running walks back to the originating declare/constant).
inline constexpr int kTraceToDeclMax = 1024;
inline constexpr int kTraceConstIntMax = 128;

/// Max memref-walk depth in asAssumedShapeAlias (peels fir.convert between the alias declare and the outer declare).
inline constexpr int kAliasMemrefWalkDepth = 32;

/// Default budget for an SSA back-walk peeling fir.convert/load/box/declare/designate to an originating declare or
/// memref; several passes hand-roll this as a bare ``< 128``, named here for one shared value.
inline constexpr int kSsaBackWalkDepth = 128;

/// Budget for the shallow shape-recovery walk in PropagateShapes (declare -> shape/hint within a few hops).
inline constexpr int kShapeWalkDepth = 20;

}  // namespace limits

/// Shape-hint attribute PropagateShapes attaches to assumed-shape dummy declares; ArrayAttr of StringAttrs, one per dim
/// (empty if that dim disagreed across callers).
inline constexpr const char* kShapeHintAttr = "hlfir_bridge.shape_hint";

/// Per-dim lower-bound hint stamped by hlfir-flatten-structs on a flat-companion declare, so resolveLowerBounds
/// recovers a nested member's non-default lb after its designate is rewritten away; ArrayAttr of StringAttrs, one per
/// flattened dim.
inline constexpr const char* kLbHintAttr = "hlfir_bridge.lb_hint";

/// Extract the short Fortran name from Flang's mangled name (e.g. "_QFcompute_z_v_grad_wEnproma" -> "nproma"); consults
/// a thread-local override map (set by extract_vars) to avoid inlined-dummy vs caller-scope short-name collisions.
std::string extractName(const std::string& mangled);

/// Register mangled -> shortName override for extractName, used by extract_vars to break short-name collisions between
/// caller and inlined-callee dummy declares. Per thread.
void setManglingOverride(const std::string& mangled, const std::string& shortName);

/// Drop every mangling-override binding; called at the start of each extractVariables/extractAST so prior-module
/// overrides don't leak.
void clearManglingOverrides();

/// Set the entry procedure's F-scope name; extractName consults it to scope-qualify non-entry-scope declares on demand.
/// Set once per build after set_entry_symbol; cleared with the override map each fresh extract. Per thread.
void setEntryScope(const std::string& scope);

/// Set of short names colliding across F-scopes; extractName qualifies as <scope>_<short> only for names in this set,
/// keeping unambiguous ones bare. Per thread, cleared each fresh extract.
void setShortNameCollisions(const std::set<std::string>& collisions);

/// Extract the F-segment (procedure scope between last F and last E) of a mangled name; "" if absent (module globals,
/// type-info metadata).
std::string getFScope(const std::string& uniq);

/// Trace an SSA value back to its hlfir.declare/fir.declare, peeling fir.convert/fir.load/arith.select; returns the
/// Fortran name or "" if the chain breaks. Consults the thread-local alloc-alias map (allocAliasFor/setAllocAlias) so
/// re-ALLOCATEd variables resolve to their current per-allocation transient (x, x_alloc1, ...).
std::string traceToDecl(mlir::Value val, int max = limits::kTraceToDeclMax);

/// Look up the active alias for a raw allocatable name; returns the raw name unchanged if none set.
std::string allocAliasFor(const std::string& raw);

/// Bind raw (allocatable base name) to alias for subsequent traceToDecl calls; alias == raw resets. Per thread.
void setAllocAlias(const std::string& raw, const std::string& alias);

/// Drop every alloc-alias binding (module-walk start, so each extractAST call sees a clean state).
void clearAllocAliases();

/// Trace an SSA value to a compile-time integer constant through any number of fir.convert wrappings; nullopt if not
/// constant-foldable.
std::optional<int64_t> traceConstInt(mlir::Value v);

/// Recognise Flang's ASSOCIATED(ptr)/ALLOCATED(arr) lowering (arith.cmpi ne, fir.convert*(fir.box_addr(fir.load?
/// %boxref)), 0) and return %boxref, to which callers append "_allocated"; null if cmp isn't that shape. Shared by
/// buildExpr/buildBoolExpr and extract_vars.
mlir::Value matchAssociatedStatusBoxRef(mlir::arith::CmpIOp cmp);

/// Render an integer SSA value as a Python expression of Fortran scalar names/literals/operators (e.g. Flang's clamped
/// ub-lb+1 -> "max((endcol - startcol) + 1, 0)") for a symbolic shape dim; leaf scalars must be separately promoted to
/// SDFG symbols (extract_vars::symbolNames). Empty string if unrecognised -- callers fall back to the "?"
/// synthetic-symbol path.
std::string traceExtentExpr(mlir::Value v);

/// Walk the same SSA chain traceExtentExpr recognises, appending every leaf scalar-declare name to out; used by
/// extract_vars Pass 2 to promote those scalars so traceExtentExpr's expression string doesn't reference undeclared
/// SDFG names.
void collectExtentExprScalars(mlir::Value v, std::set<std::string>& out);

/// Canonical SDFG symbol name for a constant-indexed element read <array>(<i1>,...); MUST match internPosSymbol (AST
/// builder) so descriptor-shape and symbol_init sides agree. E.g. dims(1) -> __sym_dims_1, shp(1,2,1) ->
/// __sym_shp_1_2_1.
std::string posSymbolName(const std::string& array, const std::vector<int64_t>& one_based_idxs);

/// If v is a fir.load of a hlfir.designate selecting a single compile-time-constant-indexed element, return {arr,
/// [i1,...]}; nullopt otherwise. Used to lift an array element feeding a shape extent to a position symbol instead of
/// the whole array's name.
std::optional<std::pair<std::string, std::vector<int64_t>>> constIndexedElementLoad(mlir::Value v);

/// Walk an extent SSA value (same op set as traceExtentExpr) and invoke fn(arr, [i1,...]) for every constant-indexed
/// element read; used at ALLOCATE sites to mint a position symbol for every shape element, not just loop-bound ones. fn
/// should be idempotent.
void forEachConstIndexedElement(mlir::Value v,
                                const std::function<void(const std::string&, const std::vector<int64_t>&)>& fn);

/// Decoded declare shape operand: fir.shape<N> (extents only, lb=1), fir.shape_shift<N> (interleaved lb,ext pairs), or
/// fir.shift<N> (lbs only, extents on the box).
struct ShapeOperandInfo {
  enum Kind : std::uint8_t { None, Shape, ShapeShift, Shift } kind = None;
  std::vector<mlir::Value> lbs;      // empty for Shape (implicit 1)
  std::vector<mlir::Value> extents;  // empty for Shift (box-carried)
  unsigned rank = 0;
};

/// Single decoder for the AnyShapeOrShiftType shape operand of hlfir.declare/fir.declare; the one source of truth
/// extent/lower-bound helpers share instead of re-matching ShapeOp/ShapeShiftOp by hand.
ShapeOperandInfo classifyShapeOperand(mlir::Value shape);

/// Extract extent SSA values from a fir.shape or fir.shape_shift; empty if neither (or null).
llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape);

/// Detects hlfir-inline-all's assumed-shape callee alias pattern (decl has no shape operand; its memref traces via
/// fir.convert to another hlfir.declare) and returns the outer declare, else null; callers skip registering the alias
/// and walk index exprs through to the outer frame.
hlfir::DeclareOp asAssumedShapeAlias(hlfir::DeclareOp decl);

/// Peel storage-transparent reinterpret ops (fir.convert/load/embox/rebox/box_addr,
/// hlfir.copy_in/coordinate_of/as_expr) off v to the underlying memref, stopping at the first non-transparent op
/// (designate/declare/alloca/block-arg/arith). Single source of truth for this peel set -- every access-path walker
/// must use it, not a hand-rolled subset. maxDepth bounds the walk.
mlir::Value peelBoxReinterpret(mlir::Value v, int maxDepth = limits::kAliasMemrefWalkDepth);

/// True iff mr (a declare's memref) peels to an hlfir.designate selecting a struct COMPONENT -- identifies an
/// inlined-call dummy bound to a caller struct member (dummy scope but memref is a component designate, not a
/// block-arg/alloca). Callers walk through to the caller-side member (gate #11 in traceToDecl; reused as gate #12 by
/// rootedAtStructDummy/walkMemberChain).
bool leadsToComponentDesignate(mlir::Value mr);

/// If v peels to a PURE array-of-records SECTION designate (triplet/scalar subscripts, no component selector,
/// RecordType element) whose base leads to a struct COMPONENT, return that section, else null. Narrow gate for the
/// per-block AoR-section-with-%member-leaf shape (e.g. vec_in bound to p_diag % p_vn(:,:,blockno)); the RecordType
/// requirement excludes plain-real member sections, which stay on the flattened-companion path.
hlfir::DesignateOp asSectionOverComponent(mlir::Value v);

/// asSectionOverComponent applied to decl's memref: the inlined AoR-section dummy (vec_in) whose box_addr/copy_in peels
/// to the section.
hlfir::DesignateOp asInlinedSectionOverComponent(hlfir::DeclareOp decl);

/// Number of SCALAR (non-triplet) subscript dims of a section designate -- the fixed record indices the AoR section
/// pins. vec_in over p_vn(:,:,blockno) -> 1.
unsigned countScalarSectionDims(hlfir::DesignateOp sec);

/// If decl is a LOCAL pointer to a whole derived-type object rebound once via ptr => <source>, return the source value
/// (peeled to its declare/component designate); null if not a local record-object pointer, rebind is ambiguous, or
/// pointee is scalar/array (a view, handled by view_alias). traceToDecl/rootedAtStructDummy/walkMemberChain follow this
/// hop so the pointer resolves to the caller-side flat name instead of its own. Mirrors
/// RewritePointerAssigns::traceRebindChain's peel set.
mlir::Value traceLocalPointerRebindSource(hlfir::DeclareOp decl);

/// Per-dim lower-bound constants for an hlfir.declare: from fir.shape_shift if present, else rank 1s (Fortran default);
/// a non-constant lb comes back as nullopt.
std::vector<std::optional<int64_t>> declareLowerBounds(hlfir::DeclareOp decl);

/// Recognise a scalar type(T), pointer/allocatable member (box<heap|ptr<record>>); the bridge can't navigate through it
/// but can safely ignore it if never read. Returns the pointed-to RecordType or null. Shared by hlfir-flatten-structs
/// and hlfir-lift-alloc-array-of-records (must agree, hence one definition).
fir::RecordType pointerToRecordMember(mlir::Type t);

/// Companion of pointerToRecordMember for the array-shaped case (box<heap|ptr<seq<? x record>>>); returns the inner
/// element RecordType or null. Treated as opaque by collectFlatLeaves (no runtime alloc-count info to pre-allocate a
/// flat companion); handled instead via the inlined-callee element-alias declare after hlfir-inline-all.
fir::RecordType allocOrPtrArrayOfRecordsMember(mlir::Type t);

/// Flang lowers PRESENT of a POINTER/ALLOCATABLE actual forwarded to an OPTIONAL dummy as a runtime select on the
/// pointer's association: %box = fir.if (box_addr(%p) != null) { result rebox(load %p) } else { result fir.absent }.
/// Given that select's result value, return the NON-absent (present) branch's yielded value; {} when the op isn't this
/// two-branch present/absent idiom. Lets memref walks (traceToDecl, the view-alias peel) reach the source declare.
mlir::Value presentBranchOfRuntimeOptional(fir::IfOp ifOp, mlir::Value result);

}  // namespace hlfir_bridge
