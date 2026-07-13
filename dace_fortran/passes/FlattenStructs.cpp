// ============================================================================
// FlattenStructs.cpp  --  Array-of-Structs -> Struct-of-Arrays at the HLFIR
// level.
// ============================================================================
//
// Goal
// ----
// Eliminate Fortran derived types from the IR before SDFG construction so the
// SDFG sees only flat per-member arrays.  Real-world ICON / ECRAD / QE code
// wraps many arrays in a struct (``type :: state_t; real(8) :: u(...), v(...)
// end type``) and passes the struct across subroutine boundaries.  DaCe's
// SDFG model handles flat arrays beautifully and structures awkwardly, so
// we rewrite at the HLFIR level where every struct access is a chain of
// ``hlfir.designate`` ops we can pattern-match.
//
// Design analogy: DaCe's ``StructToContainerGroups`` pass
// (``dace/transformation/passes/struct_to_container_group.py``) does the
// same job at the SDFG level.  Same recursive walk over record members,
// same SoA naming model, same outer-shape concatenation for array-of-
// struct.  The two passes are post-/pre-SDFG mirrors of one transform.
//
// Three flattening shapes
// -----------------------
// 1. **Scalar struct, all flat members.**  ``type t :: real
//    u(M); real v(N)`` produces ``<base>_u`` of shape ``(M)`` and
//    ``<base>_v`` of shape ``(N)``.  Single-level designate rewrite.
//
// 2. **Array-of-struct (AoS) with array members.**
//    ``type(t), dimension(K) :: A`` where each member is itself an
//    array shape concatenates outer x inner: ``A_u`` of shape
//    ``(K, M)``, ``A_v`` of shape ``(K, N)``.  ``A(i)%u(j)`` rewrites
//    to ``A_u(i, j)``  --  outer + inner indices merged in
//    ``rewriteDesignate``.  ``A(i)%u`` (whole-member access without
//    inner indices) rewrites to a triplet section ``A_u(i, 1:M:1)``.
//
// 3. **Nested record.**  Members that are themselves
//    record types unfold recursively; the leaf is whatever scalar /
//    array-of-scalar terminates the chain.  ``o%inner%x(j)`` rewrites
//    to a single flat ``o_inner_x(j)``.  ``collectFlatLeaves`` walks
//    every path to a flat leaf and ``rewriteDesignateChain`` walks
//    back through the designate chain to identify the matching
//    ``leafBase`` entry.
//
// Cross-subroutine handling
// -------------------------
// Struct dummy arguments get the same treatment.  ``replaceStructArg``
// inserts one block arg per member (or per leaf for nested) into the
// function signature and renames the function with ``_soa`` suffix.
// Inlined callee dummy declares that alias the outer struct via
// ``hlfir-inline-all`` are followed via ``collectFrom`` recursing
// through ``hlfir.declare`` users  --  the inlined alias chain is
// transparent to the rewrite.  ``recordStructArgEntry`` writes a
// ``hlfir.flatten_plan`` attribute the bindings emitter consumes to
// generate caller-side pack/unpack wrappers.
//
// Static-shape assumption
// -----------------------
// Every member shape and outer-array extent must fold to a
// compile-time constant  --  except for the allocatable-array member
// case (Phase 5a) below.  Pointer members and AoS-with-allocatable
// members are still out of scope and surface as
// loud-failure throws at ``extract_vars`` (``fir.RecordType``
// reaches a declare).
//
// Phase 5a  --  allocatable scalar-struct local member
// -------------------------------------------------
// ``type t :: real, allocatable :: w(:)`` paired with a LOCAL
// ``type(t) :: s`` instance flattens to a flat top-level
// allocatable ``s_w`` (declare carrying ``fortran_attrs =
// #fir.var_attrs<allocatable>``) plus per-allocate-site renames so
// flang's ``fir.allocmem`` op (originally named after the member's
// module scope, e.g. ``_QMlibEw.alloc``) appears under
// ``s_w.alloc``  --  the convention the bridge's ``collectAllocSites``
// walks.  Companion change: ``extract_vars.cpp`` pass 2b also walks
// every ``fir.allocmem``'s shape operands and promotes the traced
// declares to symbols, so ``allocate(s%w(n))`` (without any
// surrounding do-loop) doesn't leave ``n`` as a scalar that
// collides with the array-extent symbol downstream.
//
// Phase 5a is gated to: scalar-outer (no AoS) + allocatable / pointer
// array member.  Phase 5b extended to dummy-arg structs and pointer
// members.  AoS-with-allocatable, nested-struct-allocatables, and
// reallocation-inside-kernel for AoS members are still deferred to
// Phase 5c.
//
// Phase 5c  --  AoS + allocatable members
// ------------------------------------
// ``type t :: real, allocatable :: w(:); type(t) :: A(N)``  --  each
// batch instance ``A(i)`` owns its own runtime descriptor for
// ``A(i)%w``.  Two sub-cases share one logical contract
// (padding-to-max), but the IR shape and helpers differ.
//
// 5c-A  --  local instance, kernel-internal allocate (compile-time uniform)
//   When ``A`` is a local ``fir.alloca`` and every
//   ``allocate(A(i)%w(M))`` site uses the same compile-time constant
//   ``M``, ``aosAllocUniformConstSize`` returns ``M`` and we synthesise
//   a fully static companion ``A_w : ref<array<N x M x T>>``.  The
//   per-instance allocate / freemem chain becomes dead and is erased
//   by ``eraseAosAllocDeallocChain``.  Read-side pattern
//   ``fir.load + designate(loaded, j)`` is folded into a direct
//   2-index designate over the new flat declare by
//   ``collapseAosAllocReads``; whole-component assigns
//   (``A(i)%w = scalar``) are rewritten to row-section assigns
//   (``A_w(i, 1:M:1) = ...``) by ``rewriteAosWholeMemberAssign`` so
//   the existing concat path doesn't broadcast across all rows.
//
// 5c-B (inlined)  --  module-contained kernel after ``hlfir-inline-all``
//   When the AoS+allocatable struct is the dummy of a module-contained
//   subroutine, ``hlfir-inline-all`` splices the body in and the
//   inlined dummy becomes an alias declare carrying ``dummy_scope``.
//   ``collapseAosAllocReads`` follows the alias chain
//   (``hlfir.declare`` -> ``fir.embox`` / ``fir.convert``) back to the
//   original declare so reads inside the inlined body are still
//   collapsed.
//
// 5c-B (true SDFG-boundary)  --  ``intent(inout)`` AoS struct dummy
//   When the AoS+allocatable struct is the dummy of the SDFG entry
//   itself, the per-instance sizes are runtime-determined and
//   generally differ.  ``replaceStructArg`` inserts two block args
//   per allocatable member:
//     * ``cap_<base>_<m>`` of type ``ref<index>``  --  runtime cap
//     * ``<base>_<m>`` of type ``ref<array<N x ?xT>>``  --  2D buffer
//   It synthesises a declare for each, with ``uniq_name = "cap_..."``
//   on the cap declare so ``traceToDecl`` resolves the data declare's
//   inner extent to ``cap_<base>_<m>`` on the SDFG signature.
//   ``recordAosAllocEntry`` emits one ``aos_alloc=True`` FlattenEntry
//   per allocatable member; ``recordStructArgEntry`` takes an
//   exclude-set so non-allocatable siblings are still covered by a
//   separate aliasable entry (mixed structs are split into one
//   per-member aos_alloc entry plus one regular entry).
//
// Bindings-side contract for 5c-B (true boundary).  Stamped in the
// recipe's ``aos_alloc=True`` + ``cap_symbol`` fields and consumed
// by ``bindings/loop_copy.py``:
//   1. cap = max_i(merge(size(A(i)%w), 0, allocated(A(i)%w)))
//   2. allocate(A_w(N, cap)); zero-init.
//   3. Per i with allocated(A(i)%w): A_w(i, 1:size(A(i)%w)) = A(i)%w.
//   4. Call SDFG with the buffer + cap symbol.
//   5. On intent(out)/(inout) and per allocated row: copy back
//      A(i)%w = A_w(i, 1:size(A(i)%w)).
//   6. deallocate(A_w).
// Saved policy: NO runtime ``allocated()`` checks inside the SDFG  --
// the bindings handle every allocation query.  Mixed allocation
// states are allowed; unallocated rows stay zero-padded and the
// user's program logic must avoid reading them.  Empty-batch
// sentinel (``cap == 0 -> 1``) keeps the buffer non-degenerate.
//
// 5c-C  --  kernel-internal reallocation (NOT YET SUPPORTED)
//   When the kernel itself runs ``allocate(A(i)%w(N_i))`` (e.g. the
//   struct comes in ``intent(out)`` with no live data), the
//   bindings-time max is unknown.  Two follow-up directions:
//     TODO-1: HLFIR shape-discovery pre-pass that interprets each
//             ``allocate`` as size-discovery, collects all ``N_i``,
//             computes ``max(N_i)``, then re-runs normally.
//             Requires re-runnability of the discovery body.
//     TODO-2: F90 source-level rewrite that lifts each per-instance
//             allocate into a single max-sized pre-allocation.
//             Cleaner than the runtime two-pass approach but requires
//             understanding user code's scope / lifetime semantics.
//
// Things this pass deliberately does NOT do
// -----------------------------------------
// * Truly virtual polymorphic dispatch  --  handled separately by
//   ``fir-polymorphic-op`` (devirtualises) and
//   ``hlfir-reject-polymorphism`` (loud-fails on residuals).  This
//   pass peels ``fir.class<T>`` like ``fir.box<T>`` so monomorphic
//   CLASS receivers flatten through the same path as TYPE.
// * Nested struct with allocatable members at depth > 1
//   (``outer%inner%w(:)``)  --  needs the nested-record path to also
//   recognise allocatables on inner records.
// * Reallocation inside the kernel for AoS-allocatable companions
//   (Phase 5c-C TODO-1 / TODO-2 above).
//
// Naming caveat
// -------------
// Per-leaf names join the path with ``_``: ``base_member1_member2``.
// This is ambiguous if user code happens to name a struct field
// ``inner_x`` AND another field ``inner`` with subfield ``x``  --
// both would map to ``base_inner_x``.  Fortran style discourages
// underscores in field names so the collision risk is small in
// practice.  DaCe's container-groups pass uses delimited prefixes
// (``__CG_/__CA_/__m_``) to avoid this; we'd need to migrate the
// recipe consumers in lockstep to switch.
// ============================================================================

#include "bridge/trace_utils.h"  // traceConstInt for AoS+allocatable size resolution
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"
#include "passes/shallow_alias.h"

namespace hlfir_bridge {

namespace {

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Strip one layer of fir.box / fir.class / fir.ref / fir.heap /
/// fir.pointer.  ``fir.box<T>`` and ``fir.class<T>`` share the
/// ``fir::BaseBoxType`` base  --  peeling either via that common base
/// lets monomorphic CLASS declares flatten through the same rewrite
/// path as non-polymorphic TYPE declares.  Surviving virtual
/// dispatch is caught by ``hlfir-reject-polymorphism``, not here.
static mlir::Type unwrapOne(mlir::Type t) {
  if (auto x = mlir::dyn_cast<fir::BaseBoxType>(t)) return x.getEleTy();
  if (auto x = mlir::dyn_cast<fir::ReferenceType>(t)) return x.getEleTy();
  if (auto x = mlir::dyn_cast<fir::HeapType>(t)) return x.getEleTy();
  if (auto x = mlir::dyn_cast<fir::PointerType>(t)) return x.getEleTy();
  return t;
}

/// Walk through every wrapper until we hit a non-wrapper.
static mlir::Type unwrapAll(mlir::Type t) {
  for (;;) {
    auto inner = unwrapOne(t);
    if (inner == t) return t;
    t = inner;
  }
}

static bool isSimpleScalar(mlir::Type t) {
  if (t.isF32() || t.isF64()) return true;
  // Match every integer width that ``extract_vars.cpp`` knows how
  // to map to a DaCe dtype (int8/16/32/64) so the predicates agree.
  if (t.isInteger(8) || t.isInteger(16) || t.isInteger(32) || t.isInteger(64)) return true;
  // Fortran ``LOGICAL(KIND=N)`` lowers to ``fir.logical<N>``  --  a
  // distinct MLIR type from IntegerType.  Storage is N bytes (1, 2,
  // 4, 8); ``extract_vars.cpp`` maps each kind to the matching
  // ``int<N*8>`` dtype.  The kind-preserving mapping is required at
  // the SDFG layer because the flat companion's array stride /
  // total_size depend on element bytes; the bindings wrapper does
  // ``.TRUE.``/``.FALSE.`` <-> ``1``/``0`` conversion at the Fortran
  // caller boundary.
  if (mlir::isa<fir::LogicalType>(t)) return true;
  // Fortran ``COMPLEX(KIND=N)`` lowers to ``mlir::ComplexType`` over
  // an ``f32`` / ``f64`` element.  DaCe has native ``complex64`` /
  // ``complex128``; ``extract_vars.cpp`` maps each kind to the
  // matching DaCe complex dtype, so a struct's ``complex(c_double)``
  // member can flatten to a flat ``complex128`` companion (NOT split
  // into re/im flats -- DaCe handles complex arithmetic on the
  // single complex array natively).  The bindings emitter aliases
  // the complex flat with a normal ``c_f_pointer(c_loc(<outer>%z),
  // <outer>_z, [...])``  --  ABI between ``complex(c_double)`` and
  // ``complex128`` is bit-identical (16 bytes, two contiguous f64).
  if (auto ct = mlir::dyn_cast<mlir::ComplexType>(t)) {
    auto et = ct.getElementType();
    return et.isF32() || et.isF64();
  }
  return false;
}

/// Recognise an allocatable-array OR pointer-array struct member:
///   * ``real, allocatable :: w(:)``  -> ``fir.box<fir.heap<fir.array<?xT>>>``
///   * ``real, pointer     :: w(:)``  -> ``fir.box<fir.ptr<fir.array<?xT>>>``
///
/// Both share the same outer wrapper shape (a runtime descriptor on
/// the struct slot); only the inner indirection type differs (``heap``
/// vs ``ptr``).  Under the bridge's strict-no-aliasing assumption the
/// two are interchangeable for downstream lowering: each instance
/// holds a (data pointer + shape) descriptor, the static type of the
/// slot is the box, and the dynamic extent lives in the descriptor.
/// Allocate / freemem flow only fires on the heap variant; pointer
/// rebinds are collapsed by ``hlfir-rewrite-pointer-assigns`` (now
/// extended to handle the slice-target form for both top-level
/// pointers and pointer struct-member rebinds).
static bool isAllocatableArrayMember(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return false;
  mlir::Type inner;
  if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
    inner = heap.getEleTy();
  else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
    inner = ptr.getEleTy();
  else
    return false;
  auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
  if (!seq) return false;
  return isSimpleScalar(seq.getEleTy());
}

/// Recognise an allocatable-scalar OR pointer-scalar struct member:
///   * ``real, allocatable :: a``  -> ``fir.box<fir.heap<T>>``
///   * ``real, pointer     :: a``  -> ``fir.box<fir.ptr<T>>``
/// Sibling of ``isAllocatableArrayMember`` for rank-0 allocatables /
/// pointers.  These appear in nested struct hierarchies (e.g. an
/// inner record holds a scalar allocatable field); admitting them to
/// ``collectFlatLeaves`` lets ``replaceStructArgNested`` produce a
/// ``box<heap<T>>`` leaf with the appropriate FortranAttr.
static bool isAllocatableScalarMember(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return false;
  mlir::Type inner;
  if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
    inner = heap.getEleTy();
  else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
    inner = ptr.getEleTy();
  else
    return false;
  if (mlir::isa<fir::SequenceType>(inner)) return false;
  return isSimpleScalar(inner);
}

/// Scalar or array-of-scalar (or allocatable array-of-scalar).  Used
/// both for struct members (when the enclosing struct is a scalar)
/// and for the final companion pointee type.
static bool isFlatMemberType(mlir::Type t) {
  if (isSimpleScalar(t)) return true;
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) return isSimpleScalar(seq.getEleTy());
  if (isAllocatableArrayMember(t)) return true;
  return false;
}

/// Pull the enclosed RecordType out of a declared HLFIR type and report
/// whether it is wrapped in an outer fir.array (array-of-struct case).
/// Returns null if the peeled type is not a record.
static fir::RecordType peelToRecord(mlir::Type declaredTy, bool& outerIsArray,
                                    llvm::SmallVectorImpl<int64_t>& outerShape) {
  outerIsArray = false;
  outerShape.clear();
  auto peeled = unwrapAll(declaredTy);
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(peeled)) {
    outerIsArray = true;
    for (auto d : seq.getShape()) outerShape.push_back(d);
    peeled = seq.getEleTy();
  }
  return mlir::dyn_cast<fir::RecordType>(peeled);
}

/// Compute the companion pointee for a given (outer, member) pairing.
/// Returns null if unsupported.
///
/// Outer-is-array AND member-is-array case (``s(N)%w(M1, M2, ...)``):
/// concatenate the two shape vectors into a single fir.array<N, M1, M2,
/// ...>.  Fortran derived types have a SINGLE declared shape per member
/// that applies to every instance, so per-instance offset uniformity is
/// automatic  --  no per-element check needed.
static mlir::Type companionPointee(bool outerIsArray, llvm::ArrayRef<int64_t> outerShape, mlir::Type memberTy) {
  bool memberIsArray = mlir::isa<fir::SequenceType>(memberTy);
  if (outerIsArray && memberIsArray) {
    auto memSeq = mlir::cast<fir::SequenceType>(memberTy);
    llvm::SmallVector<int64_t, 6> concat(outerShape.begin(), outerShape.end());
    for (auto d : memSeq.getShape()) concat.push_back(d);
    return fir::SequenceType::get(concat, memSeq.getEleTy());
  }
  if (outerIsArray) return fir::SequenceType::get(outerShape, memberTy);
  return memberTy;  // scalar struct: pass the member through verbatim
}

/// Rebuild `shell`'s wrappers around a new inner type.  Used when we need
/// to mirror the original declare's result-0 wrapping (e.g.
/// fir.box<array<...>>) with the element type replaced.  ``fir.class<T>``
/// rebuilds as
/// ``fir.class<newT>`` to preserve the polymorphic tag (degrades
/// gracefully to ``fir.box`` only if explicit).
static mlir::Type rewrapWith(mlir::Type shell, mlir::Type newInner) {
  if (auto x = mlir::dyn_cast<fir::ClassType>(shell)) return fir::ClassType::get(rewrapWith(x.getEleTy(), newInner));
  if (auto x = mlir::dyn_cast<fir::BoxType>(shell)) return fir::BoxType::get(rewrapWith(x.getEleTy(), newInner));
  if (auto x = mlir::dyn_cast<fir::ReferenceType>(shell))
    return fir::ReferenceType::get(rewrapWith(x.getEleTy(), newInner));
  if (auto x = mlir::dyn_cast<fir::HeapType>(shell)) return fir::HeapType::get(rewrapWith(x.getEleTy(), newInner));
  if (auto x = mlir::dyn_cast<fir::PointerType>(shell))
    return fir::PointerType::get(rewrapWith(x.getEleTy(), newInner));
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(shell)) return fir::SequenceType::get(seq.getShape(), newInner);
  return newInner;
}

/// True when a struct-typed LOCAL declare is a rebindable runtime
/// DESCRIPTOR rather than OWNED storage.
///
/// An owned local struct lowers to plain storage: ``type(t) :: s`` ->
/// ``ref<record>`` and ``type(t) :: s(N)`` -> ``ref<array<record>>``.  A
/// ``POINTER`` / ``ALLOCATABLE`` derived-type local lowers to a box
/// descriptor: ``type(t), pointer :: p`` -> ``ref<box<ptr<record>>>``
/// (``allocatable`` -> ``ref<box<heap<record>>>``).  Its data aliases
/// another object's storage (set by ``p => x`` / ``allocate(p)``), so its
/// member reads run THROUGH the descriptor (a ``fir.load`` of the box,
/// then designate), not off the declare.  Splitting it into static
/// per-member companions is unsound -- the companions are disconnected
/// from the live pointee, and ``splitLocal``'s scalar path would even
/// mistype them (alloca ``i32`` vs ``rewrapWith``-ed ``box<ptr<i32>>``
/// declare).  ``hlfir-rewrite-pointer-assigns`` + the bridge's view path
/// own these; ``isLocallyFlattenable`` skips them.
///
/// Note: this gates on the OUTER indirection only.  A plain owned local
/// whose MEMBERS are pointer/allocatable arrays (``ref<record>`` outer)
/// still flattens via the Phase 5a/5b member companions -- those are
/// reached after peeling just ``ref`` (+ optional outer array), so the
/// remaining type is the record itself, not a box.
static bool isIndirectStructLocal(mlir::Type declaredTy) {
  mlir::Type inner = declaredTy;
  if (auto ref = mlir::dyn_cast<fir::ReferenceType>(inner)) inner = ref.getEleTy();
  // A runtime-sized PLAIN local array's ``hlfir.declare`` result #0 is a
  // ``fir.box<array<record>>`` -- still OWNED storage whose descriptor merely
  // carries the runtime extents, NOT a rebindable pointer/allocatable.  Peel
  // the box and judge by what is INSIDE it: only a ``ptr`` / ``heap`` (set by
  // ``p => x`` / ``allocate``) is a genuinely indirect, rebindable local.  A
  // static local keeps its ``ref<array<record>>`` shape and never hits the box
  // peel, so its classification is unchanged.
  if (auto box = mlir::dyn_cast<fir::BaseBoxType>(inner)) inner = box.getEleTy();
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner)) inner = seq.getEleTy();
  return mlir::isa<fir::PointerType, fir::HeapType>(inner);
}

/// Emit a fir.shape for a static extent list, inserting arith.constant ops
/// for each extent.  Returns the shape SSA value.  Empty extents -> null
/// (scalar  --  no shape needed).
static mlir::Value emitStaticShape(mlir::OpBuilder& b, mlir::Location loc, llvm::ArrayRef<int64_t> extents) {
  if (extents.empty()) return {};
  auto idxTy = b.getIndexType();
  llvm::SmallVector<mlir::Value, 4> dims;
  for (auto e : extents) {
    dims.push_back(b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(e)));
  }
  auto shapeTy = fir::ShapeType::get(b.getContext(), extents.size());
  return b.create<fir::ShapeOp>(loc, shapeTy, dims);
}

/// Recover the outer array's per-dim extent SSA values from a LOCAL declare's
/// ``fir.shape`` operand -- one Value per outer dim (a literal for a static
/// dim, the live runtime extent for a dynamic dim).  Empty when the declare
/// has no shape, or its shape isn't a plain ``fir.shape`` (a ``fir.shapeshift``
/// with non-default lower bounds is not threaded here).  This is what lets a
/// runtime-sized local AoS flatten: the companion alloca / shape reuse these
/// live extents instead of baking static literals.
static llvm::SmallVector<mlir::Value, 4> outerExtentValues(hlfir::DeclareOp decl) {
  llvm::SmallVector<mlir::Value, 4> out;
  auto sh = decl.getShape();
  if (!sh) return out;
  if (auto so = sh.getDefiningOp<fir::ShapeOp>()) out.assign(so.getExtents().begin(), so.getExtents().end());
  return out;
}

/// Build the companion ``fir.alloca`` for ``pointee``, threading a live extent
/// operand for every dim the type marks unknown (``fullExtVals`` is one Value
/// per pointee dim, leading-outer then trailing-member).  A fully-static
/// pointee gets a plain no-operand alloca -- byte-identical to the prior
/// behavior, so static locals are unaffected.
static fir::AllocaOp makeCompanionAlloca(mlir::OpBuilder& b, mlir::Location loc, mlir::Type pointee,
                                         llvm::ArrayRef<mlir::Value> fullExtVals) {
  auto seq = mlir::dyn_cast<fir::SequenceType>(pointee);
  if (!seq) return b.create<fir::AllocaOp>(loc, pointee);
  llvm::SmallVector<mlir::Value, 4> dynOps;
  auto shp = seq.getShape();
  for (unsigned i = 0; i < shp.size() && i < fullExtVals.size(); ++i)
    if (shp[i] == fir::SequenceType::getUnknownExtent()) dynOps.push_back(fullExtVals[i]);
  if (dynOps.empty()) return b.create<fir::AllocaOp>(loc, pointee);
  return b.create<fir::AllocaOp>(loc, pointee, /*typeparams=*/mlir::ValueRange{},
                                 /*shape=*/dynOps);
}

/// Extract the extents if `t` peels to a fir.array with all-static dims,
/// else return an empty vector.
static llvm::SmallVector<int64_t, 4> staticArrayExtents(mlir::Type t) {
  llvm::SmallVector<int64_t, 4> out;
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
    for (auto d : seq.getShape()) {
      if (d == fir::SequenceType::getUnknownExtent()) return {};
      out.push_back(d);
    }
  }
  return out;
}

/// Build the operandSegmentSizes attribute expected on hlfir.declare.
/// hlfir.declare has four operand segments in this order: memref, shape,
/// typeparams, dummy_scope.  We only ever construct declares with a memref
/// (and optionally a shape) in this pass  --  the remaining two segments are
/// always zero.
static mlir::NamedAttribute declareSegments(mlir::OpBuilder& b, bool hasShape) {
  llvm::SmallVector<int32_t, 4> sizes{1, hasShape ? 1 : 0, 0, 0};
  return b.getNamedAttr("operandSegmentSizes", b.getDenseI32ArrayAttr(sizes));
}

/// True if every member is flat (scalar or array-of-scalar) and we can
/// synthesise a companion pointee for every (outer, member) pair.  AoS
/// outers concatenate outer x inner extents in ``companionPointee``.
static bool allMembersFlattenable(fir::RecordType rec) {
  for (auto& pair : rec.getTypeList()) {
    if (!isFlatMemberType(pair.second)) return false;
  }
  return true;
}

/// Cap recursion through nested record members (both the shallow-alias
/// analysis and ``collectFlatLeaves``); Fortran nesting is realistically ~10.
static constexpr int kFlattenMaxDepth = 12;

/// Outcome of the shallow-alias analysis.  ``ok`` means the record is one
/// contiguous run of ``count`` elements of a single ``elem`` type, so it can
/// be represented as a ``(count)`` array that pointer-aliases the
/// array-of-structs layout an external call expects  --  no deep gather /
/// scatter.
struct ShallowAlias {
  bool ok = false;
  int64_t count = 0;
  mlir::Type elem;
};

/// Decide whether ``rec`` is shallow-aliasable: every member is INLINE
/// contiguous storage of one uniform scalar type, recursively  --
///   * a simple scalar contributes one element;
///   * a static-shape array-of-scalar contributes ``product(extents)``;
///   * a nested record contributes its own shallow-alias elements;
/// and every leaf shares the SAME element type.  An allocatable / pointer
/// member (a runtime descriptor, not inline bytes), a dynamic-extent array, a
/// non-scalar leaf, or a type mismatch fails the analysis: the record's storage
/// would then not be the contiguous uniform ``(count)`` block an array-of
/// -structs pointer addresses, so a shallow alias would be unsound and the
/// caller must deep-copy instead.
///
/// Deliberately stricter than :func:`allMembersFlattenable` (which admits
/// allocatable-array members and mixed member types for the SoA split): the
/// shallow path aliases raw storage, so it demands true inline contiguity and
/// a single element type.  The uniform-type requirement is what lets the whole
/// record fold to one typed ``(count)`` array; a member that is not itself
/// shallow-aliasable poisons the whole record (hence the recursion returns
/// failure up the chain).
static ShallowAlias analyzeShallowAlias(fir::RecordType rec, int depth = 0) {
  if (depth > kFlattenMaxDepth) return {};
  int64_t total = 0;
  mlir::Type elem;
  for (auto& pair : rec.getTypeList()) {
    mlir::Type mt = pair.second;
    mlir::Type leaf;
    int64_t n = 1;
    if (isSimpleScalar(mt)) {
      leaf = mt;
    } else if (mlir::isa<fir::SequenceType>(mt)) {
      auto ext = staticArrayExtents(mt);  // empty iff any extent is dynamic
      auto seq = mlir::cast<fir::SequenceType>(mt);
      if (ext.empty() || !isSimpleScalar(seq.getEleTy())) return {};
      for (int64_t d : ext) n *= d;
      leaf = seq.getEleTy();
    } else if (auto nrec = mlir::dyn_cast<fir::RecordType>(mt)) {
      ShallowAlias sub = analyzeShallowAlias(nrec, depth + 1);
      if (!sub.ok) return {};
      n = sub.count;
      leaf = sub.elem;
    } else {
      return {};  // box (allocatable / pointer), char, etc. -- not inline
    }
    if (!elem)
      elem = leaf;
    else if (elem != leaf)
      return {};  // non-uniform element type
    total += n;
  }
  if (!elem || total <= 0) return {};
  ShallowAlias r;
  r.ok = true;
  r.count = total;
  r.elem = elem;
  return r;
}

// ---------------------------------------------------------------------------
// Nested struct flattening helpers
// ---------------------------------------------------------------------------
//
// A nested record type (``type(outer_t) :: o`` whose member ``inner`` is
// itself ``type(inner_t)``) flattens by walking ALL paths from root to a
// flat leaf and synthesising one ``hlfir.declare`` per leaf.  Naming
// follows the path: ``o_inner_x``, ``o_inner_y``, etc.  The original
// member-by-member rewrite at the top level still works because each
// nested ``hlfir.designate`` chain unwinds through a sequence of
// component selectors that we walk in ``rewriteDesignateChain``.
//
// FlatLeaf records one such leaf:
//   * ``path``     --  successive component names from the outermost
//                   record down to the leaf.  Joined with ``_`` for
//                   the synthesised declare's uniq_name suffix.
//   * ``leafTy``   --  the leaf's type (scalar or fir.array<scalar>).
struct FlatLeaf {
  llvm::SmallVector<std::string, 4> path;
  mlir::Type leafTy;
};

/// Walk a ``RecordType`` recursively and append every flat leaf to
/// ``out``.  Returns false if any leaf is non-flat (i.e. cannot be
/// reached through a chain of pure-record + flat steps); on false the
/// pass falls back to its single-level path.  Limit recursion depth
/// to guard against unexpectedly deep nesting (Fortran allows up to a
/// realistic ~10).

/// Recursively walk a record type and append every reachable flat
/// leaf to ``out``.  Returns false if any path bottoms out at a
/// non-flat shape (allocatable / pointer member, dynamic-extent
/// inner array, etc.); on false the caller falls back to a
/// non-nested rewrite and the un-flattened struct surfaces a
/// loud failure downstream.
///
/// Three member shapes are recognised at each level:
///   * **flat member** (scalar / static-shape array of scalar)  --
///     contributes one leaf with its intrinsic shape preserved
///     (the ``outerDims`` accumulated above are prepended so
///     intermediate ``array<N x RecordType>`` levels concat into
///     the leaf's flat companion shape).
///   * **pure record** (``RecordType`` directly)  --  recurses with
///     no shape contribution.
///   * **array of records** (``array<N x RecordType>``)  --  recurses
///     into the inner record after pushing ``N`` onto
///     ``outerDims``; every leaf produced by that recursion
///     inherits ``N`` as a leading dim.  This is what enables
///     ``p_prog%pprog(i)%w(j, k)`` (where ``pprog: type(t)(10)``
///     is an array-of-struct member) to flatten to a 3D companion
///     ``p_prog_pprog_w`` of shape ``(10, 5, 5)``.
// ``pointerToRecordMember`` / ``allocOrPtrArrayOfRecordsMember`` moved
// to ``bridge/trace_utils`` so this pass and
// ``hlfir-lift-alloc-array-of-records`` share one definition (the two
// must agree on which members are treated as opaque).

/// Env-gated (``DACE_HLFIR_DEBUG_FLATTEN``) diagnostic: report the member path
/// and type at which ``collectFlatLeaves`` bails, so the disqualifying member
/// of a struct is visible.  Temporary aid for the ICON struct-flatten work.
static void logFlatBail(const llvm::SmallVectorImpl<std::string>& prefix, const char* reason, mlir::Type t) {
  if (!std::getenv("DACE_HLFIR_DEBUG_FLATTEN")) return;
  std::string path;
  for (auto& p : prefix) {
    path += "%";
    path += p;
  }
  llvm::errs() << "[flatten bail] " << reason << " at " << path << " : ";
  t.print(llvm::errs());
  llvm::errs() << "\n";
}

/// When ``partial`` is set, an unsupported member is SKIPPED (no leaf, keep
/// walking the siblings) instead of failing the whole record.  Used only on the
/// struct-dummy-argument path: a skipped member that is genuinely accessed is
/// caught later (the companion is missing), while a skipped member that is
/// never accessed simply costs nothing -- and ``replaceStructArgNested`` only
/// erases the struct dummy when no users remain, so a kept-alive accessed
/// member never dangles.  The local-instance / struct-assign callers keep
/// all-or-nothing.
static bool collectFlatLeaves(fir::RecordType rec, llvm::SmallVectorImpl<std::string>& prefix,
                              llvm::SmallVectorImpl<int64_t>& outerDims, llvm::SmallVectorImpl<FlatLeaf>& out,
                              llvm::SmallPtrSetImpl<mlir::Type>& visited, int depth = 0, bool partial = false) {
  if (depth > kFlattenMaxDepth) return false;
  // Mark this record as in-progress so a downstream pointer member
  // whose pointee re-enters the same type (mutual recursion: ``type a_t
  // { type(b_t), pointer :: b }; type b_t { type(a_t) :: a }``) is
  // recognised as a parent pointer rather than infinite-recursed
  // through.  Pointers to records that close a cycle through any
  // ancestor are treated as opaque  --  no leaf emitted, no failure
  // raised.  Code that actually navigates through such a pointer
  // (``s%b%a%w``) is out of scope for this admission path; the user
  // contract is that the pointer is either unused or points back to
  // the parent instance.
  visited.insert(rec);
  auto guard = llvm::make_scope_exit([&]() { visited.erase(rec); });
  for (auto& pair : rec.getTypeList()) {
    prefix.push_back(pair.first);
    // ``type(T), pointer :: f`` / ``type(T), allocatable :: f`` is
    // opaque to the leaf walker  --  we don't have a flat
    // representation for "all the records reachable through this
    // pointer".  Skip silently rather than fail the whole
    // flatten; downstream the only code paths that navigate
    // through such a pointer are the cycle-collapse rewrite (the
    // user-contract case ``s%b%a%w === s%w``) or genuine multi-
    // instance pointer chases (out of scope per the parent-
    // pointer contract).  Either way, the flat leaf set is what
    // matters here, and the pointer doesn't contribute one.
    if (pointerToRecordMember(pair.second)) {
      prefix.pop_back();
      continue;
    }
    // Admit allocatable/pointer scalars alongside the regular flat
    // shapes  --  ``replaceStructArgNested``'s BoxType leaf branch
    // already produces the right declare for either rank.
    if (isFlatMemberType(pair.second) || isAllocatableScalarMember(pair.second)) {
      FlatLeaf leaf;
      leaf.path.assign(prefix.begin(), prefix.end());
      // Compose the leaf's flat companion shape:
      //   outerDims (accumulated array-of-record dims walked
      //   on the way down) ++ memberDims (the leaf member's
      //   own intrinsic shape, if any).
      mlir::Type leafEle = pair.second;
      llvm::SmallVector<int64_t, 4> memberDims;
      if (auto seq = mlir::dyn_cast<fir::SequenceType>(leafEle)) {
        for (auto d : seq.getShape()) {
          if (d == fir::SequenceType::getUnknownExtent()) {
            logFlatBail(prefix, "dynamic-extent static array leaf", pair.second);
            prefix.pop_back();
            return false;  // dynamic extents in the
                           // leaf require a runtime
                           // shape we don't synthesise
                           // in this path.
          }
          memberDims.push_back(d);
        }
        leafEle = seq.getEleTy();
      }
      if (outerDims.empty() && memberDims.empty()) {
        // Pure scalar leaf  --  no array wrapper.
        leaf.leafTy = leafEle;
      } else {
        llvm::SmallVector<int64_t, 6> shape(outerDims.begin(), outerDims.end());
        shape.append(memberDims.begin(), memberDims.end());
        leaf.leafTy = fir::SequenceType::get(shape, leafEle);
      }
      out.push_back(std::move(leaf));
    } else if (auto innerRec = mlir::dyn_cast<fir::RecordType>(pair.second)) {
      if (!collectFlatLeaves(innerRec, prefix, outerDims, out, visited, depth + 1, partial)) {
        prefix.pop_back();
        return false;
      }
    } else if (auto seq = mlir::dyn_cast<fir::SequenceType>(pair.second)) {
      // Array-of-record member: recurse INTO the inner record
      // with the outer extents pushed on so each leaf inherits
      // them as leading dims.  Bail on dynamic extents  --  those
      // would need a runtime-shape companion the synth path
      // doesn't yet emit.
      auto innerRec = mlir::dyn_cast<fir::RecordType>(seq.getEleTy());
      if (!innerRec) {
        logFlatBail(prefix, "array member, element not a record", pair.second);
        prefix.pop_back();
        return false;
      }
      llvm::SmallVector<int64_t, 4> theseDims;
      for (auto d : seq.getShape()) {
        if (d == fir::SequenceType::getUnknownExtent()) {
          logFlatBail(prefix, "dynamic-extent array-of-records", pair.second);
          prefix.pop_back();
          return false;
        }
        theseDims.push_back(d);
      }
      for (auto d : theseDims) outerDims.push_back(d);
      bool ok = collectFlatLeaves(innerRec, prefix, outerDims, out, visited, depth + 1, partial);
      for (size_t i = 0; i < theseDims.size(); ++i) outerDims.pop_back();
      if (!ok) {
        prefix.pop_back();
        return false;
      }
    } else {
      // Member is e.g. allocatable / pointer  --  not flattenable
      // through this path.  Bail so the pass leaves the
      // struct untouched and the loud-failure throw in
      // extract_vars points at the right gap.
      logFlatBail(prefix, "unsupported member (box/char/alloc-array-of-records)", pair.second);
      prefix.pop_back();
      if (partial) continue;  // skip this member, keep flattening siblings
      return false;
    }
    prefix.pop_back();
  }
  return true;
}

/// Top-level entry point for the flat-leaf walker.  Internal
/// callers always start with empty ``outerDims``.  Forwards to
/// the recursive form above.
static bool collectFlatLeaves(fir::RecordType rec, llvm::SmallVectorImpl<std::string>& prefix,
                              llvm::SmallVectorImpl<FlatLeaf>& out, int depth = 0, bool partial = false) {
  llvm::SmallVector<int64_t, 4> outerDims;
  llvm::SmallPtrSet<mlir::Type, 4> visited;
  return collectFlatLeaves(rec, prefix, outerDims, out, visited, depth, partial);
}

/// Entry point that threads a caller-provided ``outerDims`` (used by
/// ``splitLocal`` / the AoS-allocatable pre-flatten check to seed the
/// outer record's array extents).
static bool collectFlatLeaves(fir::RecordType rec, llvm::SmallVectorImpl<std::string>& prefix,
                              llvm::SmallVectorImpl<int64_t>& outerDims, llvm::SmallVectorImpl<FlatLeaf>& out,
                              int depth = 0) {
  llvm::SmallPtrSet<mlir::Type, 4> visited;
  return collectFlatLeaves(rec, prefix, outerDims, out, visited, depth);
}

/// Detect a "jagged" scalar-struct: every member is a 1-D array of the same
/// scalar element type, and at least two members have different extents.
///
/// When true, the struct is packed into a single 2-D companion of shape
/// ``[numMembers x max(extents)]``  --  an ELLPACK-style padded representation
/// used when per-member flattening would produce differently-shaped siblings.
/// Scalar and non-1-D members are reported unsupported, so the caller falls
/// back to the per-member path.
static bool isJaggedScalarStruct(fir::RecordType rec, mlir::Type& eleTy, llvm::SmallVectorImpl<int64_t>& extents) {
  eleTy = nullptr;
  extents.clear();

  for (auto& pair : rec.getTypeList()) {
    auto seq = mlir::dyn_cast<fir::SequenceType>(pair.second);
    if (!seq) return false;
    auto shape = seq.getShape();
    if (shape.size() != 1) return false;
    if (shape[0] == fir::SequenceType::getUnknownExtent()) return false;
    auto se = seq.getEleTy();
    if (!isSimpleScalar(se)) return false;
    if (!eleTy)
      eleTy = se;
    else if (eleTy != se)
      return false;
    extents.push_back(shape[0]);
  }
  if (extents.size() < 2) return false;

  for (size_t i = 1; i < extents.size(); ++i)
    if (extents[i] != extents[0]) return true;
  return false;  // all uniform  --  per-member path handles it cleanly
}

// ---------------------------------------------------------------------------
// Fortran-name helpers  --  drive FlattenPlan construction in the pass
// ---------------------------------------------------------------------------

/// Extract the user-visible Fortran variable name from a Flang uniq_name.
///
/// Flang's mangled uniq_name carries the enclosing scope:
///     ``_QF<sub>E<var>``           --  dummy/local in subroutine ``<sub>``
///     ``_QM<mod>F<sub>E<var>``     --  in module ``<mod>``, subroutine
///     ``<sub>``
///     ``_QF<sub>E<var>_component``   --  nested cases exist but keep the ``E``
///                                    as the last separator for the outer
///                                    user name.
///
/// Grabbing everything after the *last* ``E`` gives the declared
/// Fortran name intact for the common case and degrades gracefully
/// (returns the full string) for unfamiliar mangling schemes.
static std::string demangleVarName(llvm::StringRef uniqName) {
  auto epos = uniqName.rfind('E');
  if (epos == llvm::StringRef::npos) return uniqName.str();
  return uniqName.substr(epos + 1).str();
}

/// Map a Flang intent flag to the writeback_intent string the binding
/// emitter expects (``in`` / ``out`` / ``inout`` / ``""``).  The
/// emitter uses ``inout`` and ``out`` to gate copy-out code  --  ``in``
/// and empty are both read-only.
static std::string extractIntent(std::optional<fir::FortranVariableFlagsEnum> flagsOpt) {
  if (!flagsOpt) return "";
  auto flags = *flagsOpt;
  auto has = [&](fir::FortranVariableFlagsEnum f) {
    return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(f)) != 0;
  };
  if (has(fir::FortranVariableFlagsEnum::intent_inout)) return "inout";
  if (has(fir::FortranVariableFlagsEnum::intent_out)) return "out";
  if (has(fir::FortranVariableFlagsEnum::intent_in)) return "in";
  return "";
}

/// Pretty-print a Flang element type as the Fortran scratch dtype the
/// Python ``FlattenRecipe`` carries (``float64`` / ``float32`` /
/// ``int32`` / ``int64`` / ``bool``).
///
/// LOGICAL of every KIND maps to ``bool`` -- the SDFG internal
/// storage for LOGICAL is always 1-byte boolean.  The Fortran-side
/// width conversion (1 / 2 / 4 / 8 bytes per the source LOGICAL's
/// KIND) is the binding-wrapper / bind_c_shim's job at the
/// boundary; the SDFG kernel itself never sees the wider Fortran
/// LOGICAL layout.
///
/// Returns an empty string for types we don't map; the caller
/// typically falls back to ``float64`` in that case.
static std::string dtypeName(mlir::Type t) {
  if (t.isF32()) return "float32";
  if (t.isF64()) return "float64";
  if (t.isInteger(8)) return "int8";
  if (t.isInteger(16)) return "int16";
  if (t.isInteger(32)) return "int32";
  if (t.isInteger(64)) return "int64";
  if (t.isInteger(1) || mlir::isa<fir::LogicalType>(t)) return "bool";
  if (auto ct = mlir::dyn_cast<mlir::ComplexType>(t)) {
    auto et = ct.getElementType();
    if (et.isF32()) return "complex64";
    if (et.isF64()) return "complex128";
  }
  return "";
}

/// Peel the descriptor wrappers a struct member's declared type may
/// carry before its array / scalar core.  A POINTER or ALLOCATABLE
/// array member is typed ``fir.box<fir.ptr<fir.array<...>>>`` (or
/// ``fir.heap`` for ALLOCATABLE); a CLASS member adds ``fir.class``.
/// ``memberRank`` / ``memberElementType`` need the core
/// ``fir.array<...>`` (or scalar) underneath, not the box  --  without
/// peeling, every deferred-shape struct member looks rank-0 and the
/// flatten plan emits a scalar ``c_f_pointer`` alias for what is
/// really a multidimensional array.
static mlir::Type peelMemberWrappers(mlir::Type t) {
  for (;;) {
    if (auto x = mlir::dyn_cast<fir::ClassType>(t)) {
      t = x.getEleTy();
    } else if (auto x = mlir::dyn_cast<fir::BoxType>(t)) {
      t = x.getEleTy();
    } else if (auto x = mlir::dyn_cast<fir::ReferenceType>(t)) {
      t = x.getEleTy();
    } else if (auto x = mlir::dyn_cast<fir::HeapType>(t)) {
      t = x.getEleTy();
    } else if (auto x = mlir::dyn_cast<fir::PointerType>(t)) {
      t = x.getEleTy();
    } else {
      return t;
    }
  }
}

/// Element type of a member  --  unwraps any descriptor wrappers and
/// fir.array to its element, or returns the scalar itself.  Used to
/// pick the recipe dtype.
static mlir::Type memberElementType(mlir::Type memTy) {
  mlir::Type core = peelMemberWrappers(memTy);
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(core)) return seq.getEleTy();
  return core;
}

/// Return the rank of a member type (0 for scalars).  Peels POINTER /
/// ALLOCATABLE / CLASS descriptor wrappers first so deferred-shape
/// array members report their true rank.
static int memberRank(mlir::Type memTy) {
  mlir::Type core = peelMemberWrappers(memTy);
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(core)) return seq.getShape().size();
  return 0;
}

/// Source-LOGICAL byte width for a struct member's element type, or
/// ``0`` when the member is not a ``LOGICAL`` (any other dtype --
/// real, integer, complex, character, nested record).  Used to set
/// the recipe's ``source_logical_kind`` attribute so the binding
/// emitter can bridge a Fortran ``LOGICAL(KIND=N)`` source slot
/// (1 / 2 / 4 / 8 bytes) through a 1-byte SDFG ``bool`` companion
/// without clobbering adjacent struct fields.  ``mlir::i1`` (the
/// HLFIR boolean) maps to KIND=1.
static int memberLogicalKind(mlir::Type memTy) {
  mlir::Type et = memberElementType(memTy);
  if (auto lt = mlir::dyn_cast<fir::LogicalType>(et)) return lt.getFKind();
  if (et.isInteger(1)) return 1;
  return 0;
}

// ---------------------------------------------------------------------------
// Shared designate rewrite
// ---------------------------------------------------------------------------

/// Redirect a single hlfir.designate whose base is `oldBase` onto the per-
/// member companion.  Works for bare field access (no indices) and for
/// indexed access; clones the original op for the latter so indices, shape,
/// and remaining attributes survive.
/// Walk back through a chain of ``hlfir.designate`` ops to collect the
/// component names from outermost to innermost.  The chain anchor is the
/// first non-designate operand (typically the original ``hlfir.declare``
/// of the struct root).  Returns the joined "_" path on success and an
/// empty string if the chain doesn't end in pure component selectors.
/// Walk a chain of ``hlfir.designate`` ops back from the leaf up
/// to the underlying ``hlfir.declare``, collecting:
///   * ``path``                   --  outer-first list of component names.
///   * ``intermediateIndices``    --  outer-first list of indices that
///                                 appeared on NON-LEAF designates
///                                 (i.e. on intermediate steps of
///                                 the chain).  Empty for the
///                                 simple case where only the leaf
///                                 carries indices.
///
/// Two chain shapes are handled by separate downstream paths:
///
///   1. **Leaf-only indices** (the original case): every
///      intermediate designate is a pure ``{component}`` selector,
///      and any indices live on the leaf itself.  Caller clones
///      the leaf and swaps its memref to the flat companion  --
///      preserving triplet sections, shape operands, and any
///      other leaf-side attributes.
///
///   2. **Intermediate indices** (array-of-record member): the
///      chain has a ``designate(idx)`` step between component
///      designates, e.g. ``p_prog%pprog(i)%w(j, k)``.  Caller
///      builds a fresh designate over the flat companion with
///      indices merged across all chain steps.  Triplet sections
///      on intermediate steps aren't in scope here (rare; would
///      need separate handling).
///
/// Returns the joined ``"a_b_c"`` path key on success (matching the
/// FlatLeaf naming the synth produces); empty string if the chain
/// has no component step at all, or if a triplet section appears
/// at a non-leaf level.
static std::string walkDesignateChain(hlfir::DesignateOp leaf,
                                      llvm::SmallVectorImpl<mlir::Value>& intermediateIndices) {
  llvm::SmallVector<std::string, 4> compsRev;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> intermediateIdxGroupsRev;
  hlfir::DesignateOp cur = leaf;
  for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
    mlir::StringAttr compAttr;
    for (auto nm : {"component_name", "component"})
      if (auto a = cur->getAttrOfType<mlir::StringAttr>(nm)) {
        compAttr = a;
        break;
      }
    if (compAttr) compsRev.push_back(compAttr.getValue().str());
    bool isLeaf = (cur == leaf);
    if (!isLeaf) {
      // Intermediate steps must be plain (no triplets).
      // Triplet sections on intermediate levels would mean a
      // non-uniform slice through the array-of-record path
      // (e.g. ``p_prog%pprog(2:5)%w(j)``); not in scope.
      //
      // ``getIsTriplet()`` returns a nullable
      // ``DenseBoolArrayAttr``  --  iterating a null attr (when
      // the designate carries no isTriplet, the common case
      // for component-only or scalar-index designates) is a
      // crash, so guard via the raw attr accessor first.
      if (auto trip = cur.getIsTripletAttr())
        for (bool t : trip.asArrayRef())
          if (t) return "";
      llvm::SmallVector<mlir::Value, 4> these(cur.getIndices().begin(), cur.getIndices().end());
      intermediateIdxGroupsRev.push_back(std::move(these));
    }
    // Walk to parent.
    auto memref = cur.getMemref();
    cur = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memref.getDefiningOp());
  }
  if (compsRev.empty()) return "";
  // Reverse to outer-first.  Components join with "_" to match
  // FlatLeaf.path's canonical form.
  std::string joined;
  for (auto it = compsRev.rbegin(); it != compsRev.rend(); ++it) {
    if (!joined.empty()) joined += "_";
    joined += *it;
  }
  for (auto it = intermediateIdxGroupsRev.rbegin(); it != intermediateIdxGroupsRev.rend(); ++it)
    intermediateIndices.append(it->begin(), it->end());
  return joined;
}

/// Backwards-compatible wrapper used by callers that only need the
/// path (no merged indices)  --  keeps the original entry point shape
/// while ``walkDesignateChain`` is the canonical implementation.
static std::string designateChainPath(hlfir::DesignateOp leaf, hlfir::DesignateOp& outAnchor) {
  llvm::SmallVector<mlir::Value, 4> ignored;
  auto joined = walkDesignateChain(leaf, ignored);
  outAnchor = leaf;
  return joined;
}

/// Walk the same ``hlfir.designate`` chain ``walkDesignateChain``
/// recognises (leaf -> root) and collect the per-dim lower bounds the
/// chain's indexed designates carry on their shape operands, returned
/// outer-first to match the flat companion's concatenated dim order
/// (outer array dims, then nested member dims).
///
/// A nested array member's non-default lower bound (``inner%v(0:3)``)
/// lives ONLY on the per-access designate's ``fir.shape_shift``; the
/// synthesised flat companion declare gets a plain ``fir.shape``
/// (extents only).  ``resolveLowerBounds`` would then default every
/// flattened dim to 1.  ``replaceStructArgNested`` calls this before
/// the chains are rewritten away and stamps the result as
/// ``kLbHintAttr`` so the resolver recovers the real bound (E8).
///
/// Each ``hlfir.designate`` with a component selector AND indices
/// contributes ``getIndices().size()`` dims; its shape operand gives
/// those dims' lbs (``fir.shape`` -> all 1, ``fir.shape_shift`` ->
/// lb,ext pairs, ``fir.shift`` -> lbs).  A non-constant lb comes back
/// as ``"?"``.  Returns the flat outer-first lb-string vector, or an
/// empty vector if the chain has no component step.
static llvm::SmallVector<std::string, 4> collectChainLowerBounds(hlfir::DesignateOp leaf) {
  // Per-designate lb groups, leaf-first (reversed at the end).
  llvm::SmallVector<llvm::SmallVector<std::string, 4>, 4> groupsRev;
  bool sawComponent = false;
  hlfir::DesignateOp cur = leaf;
  for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
    bool hasComponent = false;
    for (auto nm : {"component_name", "component"})
      if (cur->getAttrOfType<mlir::StringAttr>(nm)) {
        hasComponent = true;
        break;
      }
    if (hasComponent) sawComponent = true;

    unsigned nIdx = cur.getIndices().size();
    if (nIdx) {
      llvm::SmallVector<std::string, 4> lbs;
      // HLFIR puts the lb-bearing shape op in ``component_shape``
      // when the designate selects an array component (the
      // ``o%arr(i)`` / ``inner%v(j)`` case); ``getShape()``
      // (operand 4) is only the designate result's own section
      // shape and is null for scalar-element access.
      mlir::Value shp = hasComponent ? cur.getComponentShape() : cur.getShape();
      auto* sd = shp ? shp.getDefiningOp() : nullptr;
      if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(sd)) {
        auto ops = ss->getOperands();
        for (unsigned k = 0; k + 1 < ops.size(); k += 2) {
          auto c = traceConstInt(ops[k]);
          lbs.push_back(c ? std::to_string(*c) : "?");
        }
      } else if (auto sf = mlir::dyn_cast_or_null<fir::ShiftOp>(sd)) {
        for (auto lb : sf->getOperands()) {
          auto c = traceConstInt(lb);
          lbs.push_back(c ? std::to_string(*c) : "?");
        }
      } else {
        // fir.shape (extents only) or no shape operand:
        // Fortran-default lb 1 for every index this step
        // contributes.
        lbs.assign(nIdx, "1");
      }
      // Defensive: keep the group width == #indices so the
      // outer-first flatten stays dim-aligned even if a shape
      // op's rank disagrees (malformed / sliced IR).
      if (lbs.size() != nIdx) lbs.assign(nIdx, "1");
      groupsRev.push_back(std::move(lbs));
    }
    cur = mlir::dyn_cast_or_null<hlfir::DesignateOp>(cur.getMemref().getDefiningOp());
  }
  llvm::SmallVector<std::string, 4> flat;
  if (!sawComponent) return flat;
  for (auto it = groupsRev.rbegin(); it != groupsRev.rend(); ++it) flat.append(it->begin(), it->end());
  return flat;
}

/// Path-prefix accumulated while tracing an alias declare back to its
/// underlying decl: outermost component first.  Surfaces when an
/// inlined-callee dummy aliases ``decl`` through
/// ``hlfir.declare -> fir.convert -> fir.embox -> hlfir.designate*``
/// chains.  ``rewriteDesignateChain`` prepends this prefix so a leaf
/// rooted at the alias designs into the same flat companion as a
/// leaf rooted directly at ``decl``.
struct AliasPrefix {
  llvm::SmallVector<std::string, 4> path;
  llvm::SmallVector<mlir::Value, 4> indices;
};

/// Compose a Fortran array SECTION that an inlined-callee dummy was bound to,
/// folding it into one designate over the flat companion.
///
/// Shape handled (the ubiquitous ICON ``_onBlock`` idiom -- an outer routine
/// loops over blocks and passes a per-block 2-D slice of a 3-D field member to
/// a worker that indexes it element-wise):
///
///   leaf    = designate(section, L...)       ! dummy(i, j)      -- element/section
///   section = designate(aliasRoot, S...)     ! member(:, :, blk) -- leading
///                                             !   full-range unit-stride
///                                             !   triplets + trailing scalars
///   aliasRoot resolves via ``aliasPrefixes`` to a flat companion in ``leafBase``
///
/// ``walkDesignateChain`` / the alias-prefix walk both bail on the section's
/// triplets, which used to drop the fixed scalar dim (``blk``) entirely and
/// leave a rank-mismatched ``companion(i, j)`` designate (2 indices over the
/// rank-3 companion).  This composes positionally instead: each full-range
/// unit-stride triplet dim is filled by the next leaf index; each scalar dim is
/// kept -- yielding ``companion(i, j, blk)`` whose index count matches the
/// companion rank.
///
/// Returns true if it rewrote the leaf; false (no IR change) for any shape
/// outside the contract, so the caller's generic path runs unchanged.
static bool rewriteSectionedAliasLeaf(hlfir::DesignateOp leaf, const llvm::StringMap<mlir::Value>& leafBase,
                                      const llvm::DenseMap<mlir::Value, AliasPrefix>& aliasPrefixes) {
  // The leaf must read a SECTION: its memref is a designate carrying triplets.
  auto section = mlir::dyn_cast_or_null<hlfir::DesignateOp>(leaf.getMemref().getDefiningOp());
  if (!section) return false;
  auto secTripAttr = section.getIsTripletAttr();
  if (!secTripAttr) return false;
  auto trips = secTripAttr.asArrayRef();
  bool anyTrip = false;
  for (bool b : trips) anyTrip |= b;
  if (!anyTrip) return false;

  // The section's base must resolve to a flat companion via a WHOLE-member
  // alias prefix (the inlined-callee dummy declare bound to the whole member,
  // no extra scalar selectors) so the section dims map 1:1 onto the companion.
  auto pit = aliasPrefixes.find(section.getMemref());
  if (pit == aliasPrefixes.end()) return false;
  const AliasPrefix& pref = pit->second;
  if (!pref.indices.empty()) return false;
  std::string path;
  for (auto& c : pref.path) {
    if (!path.empty()) path += "_";
    path += c;
  }
  if (path.empty()) return false;
  auto lbIt = leafBase.find(path);
  if (lbIt == leafBase.end()) return false;
  mlir::Value newBase = lbIt->second;

  // Decode the section's per-dim selectors from its flat ``indices`` list (a
  // triplet dim consumes 3 entries: lb, ub, step; a scalar dim consumes 1) and
  // the leaf's selectors likewise.  The leaf supplies one selector per section
  // TRIPLET dim, in order.
  auto secIdx = section.getIndices();
  auto leafIdx = leaf.getIndices();
  auto leafTripAttr = leaf.getIsTripletAttr();
  llvm::ArrayRef<bool> leafTrips = leafTripAttr ? leafTripAttr.asArrayRef() : llvm::ArrayRef<bool>{};

  llvm::SmallVector<mlir::Value, 8> outIdx;
  llvm::SmallVector<bool, 4> outTrip;
  unsigned si = 0, li = 0, leafDim = 0;
  for (bool t : trips) {
    if (!t) {
      // Scalar section selector (``blk``): kept verbatim.
      if (si >= secIdx.size()) return false;
      outIdx.push_back(secIdx[si++]);
      outTrip.push_back(false);
      continue;
    }
    // Full-range unit-stride triplet required (lb == 1 && step == 1) so the
    // consuming leaf index maps directly onto this companion dim; bail
    // otherwise (the generic path leaves a valid nested chain).
    if (si + 2 >= secIdx.size()) return false;
    auto lbC = traceConstInt(secIdx[si]);
    auto stepC = traceConstInt(secIdx[si + 2]);
    si += 3;
    if (!lbC || *lbC != 1 || !stepC || *stepC != 1) return false;
    bool leafIsTrip = leafDim < leafTrips.size() && leafTrips[leafDim];
    if (leafIsTrip) {
      if (li + 2 >= leafIdx.size()) return false;
      outIdx.push_back(leafIdx[li]);
      outIdx.push_back(leafIdx[li + 1]);
      outIdx.push_back(leafIdx[li + 2]);
      outTrip.push_back(true);
      li += 3;
    } else {
      if (li >= leafIdx.size()) return false;
      outIdx.push_back(leafIdx[li++]);
      outTrip.push_back(false);
    }
    ++leafDim;
  }
  // Every leaf selector must have been consumed (one per section triplet dim).
  if (li != leafIdx.size()) return false;

  bool anyOutTrip = false;
  for (bool b : outTrip) anyOutTrip |= b;
  mlir::OpBuilder rb(leaf);
  auto newOp = rb.create<hlfir::DesignateOp>(leaf.getLoc(), /*result_type=*/leaf.getResult().getType(),
                                             /*memref=*/newBase,
                                             /*component=*/mlir::StringAttr{},
                                             /*component_shape=*/mlir::Value{},
                                             /*indices=*/mlir::ValueRange{outIdx},
                                             /*is_triplet=*/rb.getDenseBoolArrayAttr(outTrip),
                                             /*substring=*/mlir::ValueRange{},
                                             /*complex_part=*/mlir::BoolAttr{},
                                             /*shape=*/anyOutTrip ? leaf.getShape() : mlir::Value{},
                                             /*typeparams=*/mlir::ValueRange{},
                                             /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
  leaf.getResult().replaceAllUsesWith(newOp.getResult());
  leaf.erase();
  return true;
}

/// Rewrite a multi-level ``hlfir.designate`` chain ending at ``leaf``
/// (e.g. ``designate{"x"}.designate{"inner"} %o`` for ``o%inner%x``)
/// to read directly from the path-flattened declare named in
/// ``leafBase``.  ``leaf`` may carry indices (``a(i,j)``)  --  those are
/// preserved.  ``aliasPrefixes`` lets the rewriter prepend a buried
/// prefix when the chain bottoms out at an inlined-callee alias
/// declare whose source threads through embox/convert into a
/// designate chain rooted at ``decl`` (the type_arg2 / type_array
/// shape).  Returns true if the rewrite fired.
static bool rewriteDesignateChain(hlfir::DesignateOp leaf, const llvm::StringMap<mlir::Value>& leafBase,
                                  const llvm::DenseMap<mlir::Value, AliasPrefix>* aliasPrefixes = nullptr) {
  // An inlined-callee dummy bound to a per-block SECTION of a flattened member
  // (``member(:, :, blk)``) needs the section's scalar dim composed back in;
  // neither ``walkDesignateChain`` nor the alias-prefix walk below carry it
  // (both bail on the section's triplets).  Handle that shape first.
  if (aliasPrefixes && rewriteSectionedAliasLeaf(leaf, leafBase, *aliasPrefixes)) return true;

  llvm::SmallVector<mlir::Value, 4> intermediateIndices;
  std::string path = walkDesignateChain(leaf, intermediateIndices);

  // Augment with any alias-prefix attached to the chain's root
  // designate's memref (the declare that the innermost ``cur``
  // selects from).
  if (aliasPrefixes) {
    hlfir::DesignateOp cur = leaf;
    for (int i = 0; i < kFlattenMaxDepth && cur; ++i) {
      auto memref = cur.getMemref();
      auto nextDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memref.getDefiningOp());
      if (!nextDg) {
        auto pit = aliasPrefixes->find(memref);
        if (pit != aliasPrefixes->end()) {
          const auto& pref = pit->second;
          std::string joined;
          for (auto& c : pref.path) {
            if (!joined.empty()) joined += "_";
            joined += c;
          }
          if (!path.empty()) {
            if (!joined.empty()) joined += "_";
            joined += path;
          }
          path = std::move(joined);
          llvm::SmallVector<mlir::Value, 4> merged(pref.indices.begin(), pref.indices.end());
          merged.append(intermediateIndices.begin(), intermediateIndices.end());
          intermediateIndices = std::move(merged);
        }
        break;
      }
      cur = nextDg;
    }
  }

  if (path.empty()) return false;
  auto it = leafBase.find(path);
  if (it == leafBase.end()) return false;
  auto newBase = it->second;
  // E8: the nested member's non-default lower bound is recovered
  // and stamped as ``kLbHintAttr`` on the flat companion by
  // ``rewriteChainsRootedAt`` (just before it calls this), since
  // that's where the un-rewritten chain leaves are still visible
  // with their ``fir.shape_shift`` operands intact.

  // Leaf-only path (no intermediate indices).  Preserves the
  // leaf's full shape  --  including triplet sections, shape
  // operand, complex_part, etc.  --  by cloning and rewiring just
  // the memref + clearing the component attrs.  Whole-leaf
  // access (``base{"a"}{"b"}`` with no indices) just RAUWs.
  if (intermediateIndices.empty()) {
    if (leaf.getIndices().empty()) {
      leaf.getResult().replaceAllUsesWith(newBase);
      leaf.erase();
      return true;
    }
    mlir::OpBuilder rb(leaf);
    auto* clone = rb.clone(*leaf.getOperation());
    clone->setOperand(0, newBase);
    clone->removeAttr("component");
    clone->removeAttr("component_name");
    leaf.getResult().replaceAllUsesWith(clone->getResult(0));
    leaf.erase();
    return true;
  }

  // Intermediate-indices path (array-of-record member surfaced
  // by ``collectFlatLeaves``'s extra outerDims).  Build a fresh
  // designate over the flat companion with intermediate +
  // leaf indices merged in outer-first order.  No triplets at
  // intermediate levels (walker bails on that).  Whether the
  // leaf itself has triplets is rare in this shape  --  a section
  // on the innermost array of a record-of-record-of-...  --  and
  // is also out of scope; bail to keep the contract narrow.
  if (auto leafTrip = leaf.getIsTripletAttr())
    for (bool t : leafTrip.asArrayRef())
      if (t) return false;

  // Whole-component-array access surfaced through the chain
  // (``p_prog%pprog(i)%w`` with leaf having a ``shape`` operand
  // and no own indices, but result type ``ref<array<M1, M2, ...>>``).
  // The flat companion is a higher-rank array (intermediate dims
  // ++ inner dims).  Replacing the leaf with a plain N-index
  // designate where N = #intermediates only would crash the
  // verifier (rank mismatch)  --  instead emit a section designate
  // ``flat(idx_1, ..., 1:M_1:1, 1:M_2:1)`` so the result keeps
  // the leaf's array shape while the outer scalar indices pin
  // the record element.
  mlir::OpBuilder rb(leaf);
  auto loc = leaf.getLoc();
  if (leaf.getIndices().empty()) {
    if (auto memberSeqTy = mlir::dyn_cast<fir::SequenceType>(fir::unwrapRefType(leaf.getResult().getType()))) {
      auto idxTy = rb.getIndexType();
      auto c1 = rb.create<mlir::arith::ConstantOp>(loc, idxTy, rb.getIndexAttr(1));
      llvm::SmallVector<mlir::Value, 8> sliceIndices;
      llvm::SmallVector<bool, 4> isTriplet;
      for (auto idx : intermediateIndices) {
        sliceIndices.push_back(idx);
        isTriplet.push_back(false);
      }
      for (auto d : memberSeqTy.getShape()) {
        if (d == fir::SequenceType::getUnknownExtent()) return false;
        auto cN = rb.create<mlir::arith::ConstantOp>(loc, idxTy, rb.getIndexAttr(d));
        sliceIndices.push_back(c1.getResult());
        sliceIndices.push_back(cN.getResult());
        sliceIndices.push_back(c1.getResult());
        isTriplet.push_back(true);
      }
      auto newOp = rb.create<hlfir::DesignateOp>(loc,
                                                 /*result_type=*/leaf.getResult().getType(),
                                                 /*memref=*/newBase,
                                                 /*component=*/mlir::StringAttr{},
                                                 /*component_shape=*/mlir::Value{},
                                                 /*indices=*/mlir::ValueRange{sliceIndices},
                                                 /*is_triplet=*/rb.getDenseBoolArrayAttr(isTriplet),
                                                 /*substring=*/mlir::ValueRange{},
                                                 /*complex_part=*/mlir::BoolAttr{},
                                                 /*shape=*/leaf.getShape(),
                                                 /*typeparams=*/mlir::ValueRange{},
                                                 /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
      leaf.getResult().replaceAllUsesWith(newOp.getResult());
      leaf.erase();
      return true;
    }
  }

  llvm::SmallVector<mlir::Value, 6> merged(intermediateIndices.begin(), intermediateIndices.end());
  for (auto v : leaf.getIndices()) merged.push_back(v);
  auto newOp = rb.create<hlfir::DesignateOp>(loc, leaf.getResult().getType(), newBase, mlir::ValueRange{merged});
  leaf.getResult().replaceAllUsesWith(newOp.getResult());
  leaf.erase();
  return true;
}

/// Trace ``other``'s memref back through ``hlfir.declare`` /
/// ``fir.convert`` / ``fir.embox`` / ``hlfir.designate`` ops, building
/// the ``(path, indices)`` prefix that an alias root buries.  When an
/// inlined-callee dummy aliases ``decl`` via
/// ``convert(embox(designate{"w"}(designate{"pprog"}(i))))``, a leaf
/// rooted at the alias declare needs the ``("pprog", "w") + [i]``
/// prefix prepended so the rewrite designs into the right flat
/// companion.  Triplet sections on intermediate steps are out of scope.
static std::optional<AliasPrefix> traceAliasPrefixToDecl(hlfir::DeclareOp other, hlfir::DeclareOp decl) {
  llvm::SmallVector<std::string, 4> pathRev;
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> indexGroupsRev;
  mlir::Value mr = other.getMemref();
  for (int i = 0; i < kFlattenMaxDepth && mr; ++i) {
    auto* d = mr.getDefiningOp();
    if (!d) return std::nullopt;
    if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      if (outer == decl) {
        AliasPrefix info;
        for (auto it = pathRev.rbegin(); it != pathRev.rend(); ++it) info.path.push_back(*it);
        for (auto it = indexGroupsRev.rbegin(); it != indexGroupsRev.rend(); ++it)
          info.indices.append(it->begin(), it->end());
        return info;
      }
      // Intermediate declare (a previously-inlined alias).
      // Continue walking from its source  --  declares act as
      // identity wrappers around their memref operand for the
      // purposes of alias tracking.
      mr = outer.getMemref();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      mr = cv.getValue();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      mr = eb.getMemref();
      continue;
    }
    // Inlined-callee alias declares: the alias's memref is a
    // ``fir.load`` of the parent declare's address (possibly
    // through one or more ``fir.rebox`` reshapes for CLASS<heap<T>>
    // -> CLASS<T> peels in OOP code, or POINTER box reshapes).
    // Walking through both is the minimum to recognise these as
    // aliases of the parent.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      mr = ld.getMemref();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      mr = rb.getBox();
      continue;
    }
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      mlir::StringAttr compAttr;
      for (auto nm : {"component_name", "component"})
        if (auto a = dg->getAttrOfType<mlir::StringAttr>(nm)) {
          compAttr = a;
          break;
        }
      if (compAttr) pathRev.push_back(compAttr.getValue().str());
      if (auto trip = dg.getIsTripletAttr())
        for (bool t : trip.asArrayRef())
          if (t) return std::nullopt;
      llvm::SmallVector<mlir::Value, 4> these(dg.getIndices().begin(), dg.getIndices().end());
      indexGroupsRev.push_back(std::move(these));
      mr = dg.getMemref();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

/// Walk every ``hlfir.designate`` chain rooted at ``decl`` (or any
/// inlined-callee alias of it) and rewrite each leaf to the matching
/// flat companion in ``leafBase``.  Shared between the local-declare
/// path (``splitLocal``) and the dummy-arg path
/// (``replaceStructArgNested``)  --  both produce the same ``leafBase``
/// shape and need the same rewrite logic.
static void rewriteChainsRootedAt(hlfir::DeclareOp decl, const llvm::StringMap<mlir::Value>& leafBase) {
  auto func = decl->getParentOfType<mlir::func::FuncOp>();
  if (!func) return;

  // Discover declares that alias ``decl`` (inlined-callee dummies
  // whose memref chain leads back to it).  Each gets its buried
  // path + scalar prefix recorded for the chain rewriter.
  llvm::DenseSet<mlir::Value> equivalentRoots;
  llvm::DenseMap<mlir::Value, AliasPrefix> aliasPrefixes;
  equivalentRoots.insert(decl.getResult(0));
  equivalentRoots.insert(decl.getResult(1));
  func.walk([&](hlfir::DeclareOp other) {
    if (other == decl) return;
    if (auto info = traceAliasPrefixToDecl(other, decl)) {
      equivalentRoots.insert(other.getResult(0));
      equivalentRoots.insert(other.getResult(1));
      if (!info->path.empty()) {
        aliasPrefixes[other.getResult(0)] = *info;
        aliasPrefixes[other.getResult(1)] = *info;
      }
    }
  });

  // Find each chain's leaf  --  a designate whose users are NOT
  // themselves designates (otherwise we'd rewrite a parent and
  // lose the inner part of the chain)  --  then verify the chain
  // bottoms out at one of the equivalent roots.
  llvm::SmallVector<hlfir::DesignateOp, 16> chainLeaves;
  func.walk([&](hlfir::DesignateOp dg) {
    bool hasDesignateUser = false;
    for (auto* u : dg.getResult().getUsers())
      if (mlir::isa<hlfir::DesignateOp>(u)) {
        hasDesignateUser = true;
        break;
      }
    if (hasDesignateUser) return;
    // Walk the memref chain back to an equivalent root.  Peel
    // through ``hlfir.designate`` (intermediate component / index
    // selects), ``fir.load`` (loaded box from an allocatable /
    // pointer declare slot), ``fir.rebox`` (class<heap<T>> ->
    // class<T> and similar peels), and ``fir.convert``.  The
    // load + rebox peels are what catch direct-access reads on
    // a CLASS-allocatable in main scope:
    //   ``%load = fir.load %decl#0`` then
    //   ``%dg = hlfir.designate %load{"<field>"}``.
    // Without those peels, %dg's memref (the load result) is
    // not an equivalent root and the chain stays un-rewritten.
    mlir::Value v = dg.getMemref();
    for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
      if (equivalentRoots.contains(v)) {
        chainLeaves.push_back(dg);
        return;
      }
      auto* def = v.getDefiningOp();
      if (!def) break;
      if (auto dg2 = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
        v = dg2.getMemref();
        continue;
      }
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
        v = ld.getMemref();
        continue;
      }
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
        v = rb.getBox();
        continue;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
        v = cv.getValue();
        continue;
      }
      break;
    }
  });
  // E8: before the chains are rewritten away, recover each nested
  // array member's real per-dim lower bound from its designate's
  // ``fir.shape_shift`` and stamp it on the flat companion declare.
  // The synthesised companion carries only a plain ``fir.shape``
  // (extents), so ``resolveLowerBounds`` would otherwise default
  // every flattened dim to 1 -- losing ``inner%v(0:3)``'s lb 0.
  for (auto dg : chainLeaves) {
    llvm::SmallVector<mlir::Value, 4> ignored;
    std::string key = walkDesignateChain(dg, ignored);
    if (key.empty()) continue;
    auto it = leafBase.find(key);
    if (it == leafBase.end()) continue;
    auto lbs = collectChainLowerBounds(dg);
    if (lbs.empty()) continue;
    bool anyNonDefault = false;
    for (auto& s : lbs)
      if (s != "1") {
        anyNonDefault = true;
        break;
      }
    if (!anyNonDefault) continue;
    auto* declOp = it->second.getDefiningOp();
    auto decl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(declOp);
    if (!decl) continue;
    llvm::SmallVector<mlir::Attribute, 4> attrs;
    for (auto& s : lbs) attrs.push_back(mlir::StringAttr::get(decl.getContext(), s));
    decl->setAttr(hlfir_bridge::kLbHintAttr, mlir::ArrayAttr::get(decl.getContext(), attrs));
  }

  for (auto dg : chainLeaves) rewriteDesignateChain(dg, leafBase, &aliasPrefixes);
}

/// Append ``dg``'s subscripts onto ``outIdx`` / ``outTrip``, preserving the
/// triplet structure: a triplet subscript consumes three operands
/// (lb, ub, step) and contributes ONE ``true`` flag; a scalar subscript
/// consumes one operand and contributes ONE ``false`` flag.  A null
/// ``is_triplet`` attr (the common scalar-only case) means every index is a
/// scalar.  This lets the concat rewrite merge a SECTION parent
/// (``A(:,:,:)%x(1)``) without miscounting its triplet operands as scalar
/// indices.
static void appendDesignateSubscripts(hlfir::DesignateOp dg, llvm::SmallVectorImpl<mlir::Value>& outIdx,
                                      llvm::SmallVectorImpl<bool>& outTrip) {
  auto idx = dg.getIndices();
  auto tripAttr = dg.getIsTripletAttr();
  if (!tripAttr) {
    for (auto v : idx) {
      outIdx.push_back(v);
      outTrip.push_back(false);
    }
    return;
  }
  unsigned o = 0;
  for (bool t : tripAttr.asArrayRef()) {
    outIdx.push_back(idx[o++]);
    if (t) {
      outIdx.push_back(idx[o++]);
      outIdx.push_back(idx[o++]);
    }
    outTrip.push_back(t);
  }
}

static void rewriteDesignate(hlfir::DesignateOp dg, const llvm::StringMap<mlir::Value>& memberBase,
                             const llvm::StringSet<>& concatMembers = {}) {
  // The hlfir.designate op prints the component as ``{"name"}`` but stores
  // it under the attribute key ``component_name``  --  depending on the HLFIR
  // tablegen spelling.  Tolerate either key so we don't silently no-op.
  mlir::StringAttr compAttr;
  for (auto nm : {"component_name", "component"}) {
    if (auto a = dg->getAttrOfType<mlir::StringAttr>(nm)) {
      compAttr = a;
      break;
    }
  }
  if (!compAttr) return;

  auto it = memberBase.find(compAttr.getValue());
  if (it == memberBase.end()) return;
  auto newBase = it->second;

  // Concat case (``s(N)%w(M, ...)``): the parent op is an indexed
  // designate without component (the per-element access on the outer
  // array-of-struct).  Merge the parent's outer indices with this
  // designate's member indices so the new designate is a flat
  // multi-dim access on the concatenated companion.
  bool isConcat = concatMembers.count(compAttr.getValue());
  if (isConcat) {
    auto parentMemref = dg.getMemref();
    auto parentDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(parentMemref.getDefiningOp());
    if (parentDg) {
      // Verify parent is a pure indexed access (no component).
      bool parentHasComponent = false;
      for (auto nm : {"component_name", "component"})
        if (parentDg->getAttrOfType<mlir::StringAttr>(nm)) {
          parentHasComponent = true;
          break;
        }
      if (!parentHasComponent && !parentDg.getIndices().empty()) {
        mlir::OpBuilder rb(dg);
        if (!dg.getIndices().empty()) {
          // Merge the parent outer access with this member access into one
          // flat designate.  Element parent (``A(i)%w(j, k)`` -> ``A_w(i, j,
          // k)``) and SECTION parent (``A(:,:,:)%x(1)`` -> ``A_x(:,:,:,1)``)
          // are both handled by preserving each side's triplet flags -- else a
          // sectioned parent's lb/ub/step triplet operands get miscounted as
          // scalar indices and the rank check fails.
          llvm::SmallVector<mlir::Value, 12> mergedIndices;
          llvm::SmallVector<bool, 6> mergedTriplet;
          appendDesignateSubscripts(parentDg, mergedIndices, mergedTriplet);
          appendDesignateSubscripts(dg, mergedIndices, mergedTriplet);
          bool anyTriplet = llvm::any_of(mergedTriplet, [](bool t) { return t; });
          hlfir::DesignateOp newOp;
          if (anyTriplet) {
            // A sectioned access yields an array: build via the long-form
            // builder so we can pass ``is_triplet`` and the result ``shape``.
            newOp = rb.create<hlfir::DesignateOp>(dg.getLoc(),
                                                  /*result_type=*/dg.getResult().getType(),
                                                  /*memref=*/newBase,
                                                  /*component=*/mlir::StringAttr{},
                                                  /*component_shape=*/mlir::Value{},
                                                  /*indices=*/mlir::ValueRange{mergedIndices},
                                                  /*is_triplet=*/rb.getDenseBoolArrayAttr(mergedTriplet),
                                                  /*substring=*/mlir::ValueRange{},
                                                  /*complex_part=*/mlir::BoolAttr{},
                                                  /*shape=*/dg.getShape(),
                                                  /*typeparams=*/mlir::ValueRange{},
                                                  /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
          } else {
            // All-scalar element access  --  short-form builder, unchanged.
            newOp = rb.create<hlfir::DesignateOp>(dg.getLoc(), dg.getResult().getType(), newBase,
                                                  mlir::ValueRange{mergedIndices});
          }
          dg.getResult().replaceAllUsesWith(newOp.getResult());
          dg.erase();
          if (parentDg.getResult().use_empty()) parentDg.erase();
          return;
        }
        // Whole-component access: ``A(i)%w`` -> flat section
        // ``A_w(i, 1:M:1, 1:M:1, ...)``  --  scalar outer index,
        // triplet over every inner dim.  The result type
        // stays the original member type (``ref<array<M, ...>>``).
        auto memberSeqTy = mlir::dyn_cast<fir::SequenceType>(fir::unwrapRefType(dg.getResult().getType()));
        if (!memberSeqTy) {
          // Whole scalar component (rare)  --  same as
          // empty-indices old behaviour: replace use.
          dg.getResult().replaceAllUsesWith(newBase);
          dg.erase();
          if (parentDg.getResult().use_empty()) parentDg.erase();
          return;
        }
        auto loc = dg.getLoc();
        auto idxTy = rb.getIndexType();
        auto c1 = rb.create<mlir::arith::ConstantOp>(loc, idxTy, rb.getIndexAttr(1));
        llvm::SmallVector<mlir::Value, 8> sliceIndices;
        llvm::SmallVector<bool, 4> isTriplet;
        // Preserve the parent's triplet structure (a sectioned outer,
        // ``A(:,:,:)%w``, keeps its triplets; an element outer stays scalar).
        appendDesignateSubscripts(parentDg, sliceIndices, isTriplet);
        for (auto d : memberSeqTy.getShape()) {
          if (d == fir::SequenceType::getUnknownExtent()) {
            // Cannot construct a static-bound triplet  --
            // bail to the safe fallback.
            return;
          }
          auto cN = rb.create<mlir::arith::ConstantOp>(loc, idxTy, rb.getIndexAttr(d));
          sliceIndices.push_back(c1.getResult());
          sliceIndices.push_back(cN.getResult());
          sliceIndices.push_back(c1.getResult());
          isTriplet.push_back(true);
        }
        // Build via the long-form ``hlfir.designate`` builder
        // so we can pass ``is_triplet`` directly.
        auto newOp = rb.create<hlfir::DesignateOp>(loc,
                                                   /*result_type=*/dg.getResult().getType(),
                                                   /*memref=*/newBase,
                                                   /*component=*/mlir::StringAttr{},
                                                   /*component_shape=*/mlir::Value{},
                                                   /*indices=*/mlir::ValueRange{sliceIndices},
                                                   /*is_triplet=*/rb.getDenseBoolArrayAttr(isTriplet),
                                                   /*substring=*/mlir::ValueRange{},
                                                   /*complex_part=*/mlir::BoolAttr{},
                                                   /*shape=*/dg.getShape(),
                                                   /*typeparams=*/mlir::ValueRange{},
                                                   /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
        dg.getResult().replaceAllUsesWith(newOp.getResult());
        dg.erase();
        if (parentDg.getResult().use_empty()) parentDg.erase();
        return;
      }
    }
    // Parent isn't an indexed designate  --  fall through to the
    // single-rewrite path (probably a whole-array reference).
  }

  if (dg.getIndices().empty()) {
    // Scalar member access ``X%m``.  If the parent is an indexed access on
    // an array-of-struct (``X(i)%m``), the i-th element is ``X_m(i)`` -- the
    // parent's index must be applied to the flat companion.  Replacing with
    // the bare companion (the rank-0 ``st%m`` behaviour) would drop the
    // index, alias the WHOLE companion array, and leak the connector as a
    // free symbol.  Apply the parent indices only when the companion's rank
    // matches them (the plain per-element companion); a rank-0 struct (no
    // parent index) or a differently-shaped companion (e.g. the aos_alloc
    // packed form) keeps the bare alias.
    auto parentDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(dg.getMemref().getDefiningOp());
    bool parentHasComponent = false;
    if (parentDg)
      for (auto nm : {"component_name", "component"})
        if (parentDg->getAttrOfType<mlir::StringAttr>(nm)) {
          parentHasComponent = true;
          break;
        }
    if (parentDg && !parentHasComponent && !parentDg.getIndices().empty()) {
      // ``newBase`` is the companion's HLFIR variable form: ``ref<array<...>>``
      // for a static companion, ``box<array<...>>`` for a runtime-sized one.
      // Peel both so the per-element index application fires in either case --
      // otherwise a dynamic companion's box would alias the WHOLE array onto
      // the scalar use and a later ``fir.load`` faults on the box.
      mlir::Type compInner = fir::unwrapRefType(newBase.getType());
      if (auto box = mlir::dyn_cast<fir::BaseBoxType>(compInner)) compInner = box.getEleTy();
      auto compSeq = mlir::dyn_cast<fir::SequenceType>(compInner);
      if (compSeq && compSeq.getShape().size() == parentDg.getIndices().size()) {
        mlir::OpBuilder rb(dg);
        llvm::SmallVector<mlir::Value, 4> outerIdx(parentDg.getIndices().begin(), parentDg.getIndices().end());
        auto newOp =
            rb.create<hlfir::DesignateOp>(dg.getLoc(), dg.getResult().getType(), newBase, mlir::ValueRange{outerIdx});
        dg.getResult().replaceAllUsesWith(newOp.getResult());
        dg.erase();
        if (parentDg.getResult().use_empty()) parentDg.erase();
        return;
      }
    }
    dg.getResult().replaceAllUsesWith(newBase);
    dg.erase();
    return;
  }

  mlir::OpBuilder rb(dg);
  auto* clone = rb.clone(*dg.getOperation());
  clone->setOperand(0, newBase);
  clone->removeAttr("component");
  clone->removeAttr("component_name");
  dg.getResult().replaceAllUsesWith(clone->getResult(0));
  dg.erase();
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

struct FlattenStructsPass : public mlir::PassWrapper<FlattenStructsPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenStructsPass)

  /// When true, only the AoR-element-access splits run; the
  /// scalar-struct flatten (``planAndReplaceStructArgs``) is skipped.
  /// Used by the ``hlfir-split-aor-dummies`` pipeline entry to seed
  /// per-symbol / per-inner-member scalar-struct dummies BEFORE
  /// ``hlfir-marshal-external-structs`` runs, so marshal sees the new
  /// dummies as the call's struct args and can expand them into per-
  /// member designates on the new dummy's name.  The full
  /// ``hlfir-flatten-structs`` run later in the pipeline picks up the
  /// already-split state (the split helpers are idempotent) and does
  /// the scalar-struct flatten on top.
  bool splitOnly = false;

  llvm::StringRef getArgument() const final { return splitOnly ? "hlfir-split-aor-dummies" : "hlfir-flatten-structs"; }
  llvm::StringRef getDescription() const final {
    if (splitOnly)
      return "Run only the alloc-array-of-records dummy splits "
             "(splitMultiDimAoRScalarMembers + splitDoubleBufferMembers); "
             "skip the scalar-struct flatten.  Pre-stage for marshal-"
             "external-structs.";
    return "Flatten derived types with flat members into per-member "
           "companions (AoS -> SoA), rewriting struct-typed dummy "
           "arguments, renaming the function, and splitting local "
           "allocations.";
  }

  /// Collected FlattenEntry dicts  --  stamped on the module at the end
  /// of ``runOnOperation`` as the ``hlfir.flatten_plan`` attribute.
  llvm::SmallVector<mlir::Attribute, 4> planEntries;

  void runOnOperation() override {
    planEntries.clear();
    getOperation().walk([this](mlir::func::FuncOp f) { flattenFunc(f); });

    // DEBUG (env-gated): after flattening, run each op's own verifier and
    // print the ones that now fail -- the exact malformation the pass-manager
    // verifier rejects -- with the memref's defining op so the producing
    // rewrite can be located.
    if (std::getenv("DACE_FLATTEN_DEBUG_DESIGNATE")) {
      getOperation().walk([](mlir::Operation* op) {
        if (mlir::succeeded(mlir::verify(op))) return;
        llvm::errs() << "FLATTEN_BAD_OP loc=" << op->getLoc() << "\n";
        llvm::errs() << "  op: " << *op << "\n";
        for (auto operand : op->getOperands())
          if (auto* def = operand.getDefiningOp()) llvm::errs() << "  operand def: " << *def << "\n";
      });
    }

    // Collect flatten entries the AoR-scalar-inner split stamped on its
    // persistent companion declares (see ``splitMultiDimAoRScalarMembers``).
    // The split runs in the ``hlfir-split-aor-dummies`` pre-pass; stamping the
    // entry on the companion declare (not the transient ``planEntries``, which
    // this full pass clears + re-emits) lets THIS pass's plan carry the gather
    // entry that maps ``<flat>`` back to ``<base>%<path>(i)%<inner>``.
    // Collect ONE entry per flat companion.  The SAME value-record companion
    // (``p_patch_edges_primal_normal_cell_v1``) can be reached from more than one
    // function in the module -- a ``do_not_emit``'d external kept as a stub
    // alongside its caller both take ``p_patch`` and both carry the AoR split's
    // stamp -- and every stamp maps it back to the IDENTICAL AoS source, so a
    // module-wide walk would push the same recipe twice and the bindings would
    // double-declare the ``_v1`` / ``_v2`` companion.  Dedup by the companion's
    // flat name (the reconstruction recipe is a function of the source path, so
    // same name => same recipe).
    llvm::DenseSet<mlir::Attribute> seenAorFlat;
    getOperation().walk([&](hlfir::DeclareOp d) {
      auto e = d->getAttrOfType<mlir::DictionaryAttr>("hlfir_bridge.aor_flat_entry");
      if (!e) return;
      mlir::Attribute key = e;
      if (auto recipe = e.getAs<mlir::DictionaryAttr>("recipe"))
        if (auto flatNames = recipe.getAs<mlir::ArrayAttr>("flat_names"))
          if (!flatNames.empty()) key = flatNames[0];
      if (seenAorFlat.insert(key).second) planEntries.push_back(e);
    });

    if (planEntries.empty()) return;

    // Stamp the plan as ``hlfir.flatten_plan = {entries = [...]}``.
    // The binding emitter / bridge later reads this attribute back to
    // reconstruct the Python FlattenPlan object.
    auto* ctx = getOperation().getContext();
    mlir::Builder b(ctx);
    auto entries = b.getArrayAttr(planEntries);
    auto plan = b.getDictionaryAttr({b.getNamedAttr("entries", entries)});
    getOperation()->setAttr("hlfir.flatten_plan", plan);
  }

  /// Append one FlattenEntry dict to ``planEntries`` describing the
  /// just-performed struct-dummy split.  Covers the *per-member* path
  /// (``replaceStructArg``); the jagged-ELLPACK path is omitted from
  /// the plan  --  callers of that path fall back to the looped copy-in
  /// emission without plan metadata.
  ///
  /// ``excludeMembers`` lists member names already covered by a
  /// separate ``aos_alloc=True`` entry (see ``recordAosAllocEntry``);
  /// they're skipped here so the plan has exactly one recipe per flat
  /// companion.  When every member is excluded the function emits
  /// nothing.
  void recordStructArgEntry(hlfir::DeclareOp argDecl, fir::RecordType rec, llvm::StringRef intentStr,
                            bool outerIsArray = false, llvm::ArrayRef<int64_t> outerShape = {},
                            const llvm::StringSet<>& excludeMembers = {}) {
    auto* ctx = argDecl.getContext();
    mlir::Builder b(ctx);
    auto mkStr = [&](llvm::StringRef s) -> mlir::Attribute { return b.getStringAttr(s); };

    std::string outerName = demangleVarName(argDecl.getUniqName());
    // ``outerName`` names the SDFG flat companion (``p_prog_nnow_rho``).  The
    // SOURCE expression the binding reads from is normally the same identifier,
    // but a double-buffer per-symbol dummy (``p_prog_nnow``, minted by
    // ``splitDoubleBufferMembers``) is NOT a real Fortran-side variable -- it
    // stands for ``p%prog(nnow)``.  The split records that real source path on
    // the dummy declare as ``hlfir_bridge.dbuf_source``; use it for every
    // read / shape / outer expression so the binding aliases the actual storage
    // (``c_loc(p%prog(nnow)%rho)``), while the flat name keeps ``outerName``.
    std::string sourceBase = outerName;
    if (auto srcAttr = argDecl->getAttrOfType<mlir::StringAttr>("hlfir_bridge.dbuf_source"))
      sourceBase = srcAttr.getValue().str();
    // Outer type: dump the declared type as MLIR text  --  the Python
    // side uses it only for commentary, so round-tripping the MLIR
    // form is sufficient.
    std::string outerType;
    {
      llvm::raw_string_ostream os(outerType);
      argDecl.getResult(0).getType().print(os);
    }

    // For AoS dummy args the outer index dim(s) prepend the
    // member's index dims.  So for ``A(i)%w(j, k)``, the recipe's
    // total rank is outer_rank + member_rank, and the read expr
    // is ``A($i1)%w($i2, $i3)``.
    unsigned outerRank = outerIsArray ? (unsigned)outerShape.size() : 0u;

    // Emit one FlattenEntry per member.  Earlier the function
    // bundled every member into a single multi-flat recipe, which
    // worked while every struct member had the same dtype (e.g.
    // two ``real(c_double)`` arrays) but produced an incorrect
    // ``scratch_dtype`` when members differed -- a ``t_state``
    // with a ``complex(c_double)`` member alongside a ``real``
    // member would assign one shared ``float64`` dtype to both
    // flats, and the emitter would declare the complex flat as
    // ``real(c_double), pointer``.  Per-member entries keep each
    // flat's dtype isolated.
    for (auto& pair : rec.getTypeList()) {
      llvm::StringRef memName = pair.first;
      if (excludeMembers.count(memName)) continue;
      mlir::Type memTy = pair.second;
      int memRank = memberRank(memTy);
      int totalRank = (int)outerRank + memRank;

      std::string flat = (outerName + "_" + memName).str();

      // read_expr: ``<outer>($i1, ..., $iOR)%<member>($iOR+1, ..., $iTotal)``
      // Scalar outer + scalar member: just ``<outer>%<member>``.
      std::string read = sourceBase;
      if (outerRank > 0) {
        read += "(";
        for (unsigned i = 1; i <= outerRank; ++i) {
          if (i > 1) read += ", ";
          read += "$i" + std::to_string(i);
        }
        read += ")";
      }
      read += "%";
      read += memName.str();
      if (memRank > 0) {
        read += "(";
        for (int i = 1; i <= memRank; ++i) {
          if (i > 1) read += ", ";
          read += "$i" + std::to_string((int)outerRank + i);
        }
        read += ")";
      }

      std::string scratchDtype = dtypeName(memberElementType(memTy));
      if (scratchDtype.empty()) scratchDtype = "float64";

      // Shape exprs for this member: outer-array dims first
      // (sampled from ``outer``), then the member's own dims
      // (sampled from ``outer(1, ...)%member``).
      llvm::SmallVector<mlir::Attribute, 4> shapeExprs;
      if (totalRank > 0) {
        for (unsigned i = 1; i <= outerRank; ++i) {
          std::string s = "size(" + sourceBase + ", dim=" + std::to_string((int)i) + ")";
          shapeExprs.push_back(mkStr(s));
        }
        std::string sampleOuter = sourceBase;
        if (outerRank > 0) {
          sampleOuter += "(";
          for (unsigned i = 0; i < outerRank; ++i) {
            if (i) sampleOuter += ", ";
            sampleOuter += "1";
          }
          sampleOuter += ")";
        }
        for (int i = 1; i <= memRank; ++i) {
          std::string s = ("size(" + sampleOuter + "%" + memName.str() + ", dim=" + std::to_string(i) + ")");
          shapeExprs.push_back(mkStr(s));
        }
      }

      // A member is a zero-copy ``c_f_pointer`` alias only for a rank-0 struct,
      // where ``c_loc(st%m)`` is a single contiguous block whose shape IS the
      // member's own dims.  For an AoS (the outer is an array) the alias is
      // invalid even when the struct has ONE member: although ``outer(:)%m`` is
      // then contiguous in memory, that memory is member-FIRST
      // (``[m_dims..., outer_dims...]``) while the SoA companion's shape is
      // member-LAST (``[outer_dims..., m_dims...]``), so a flat reinterpret
      // would TRANSPOSE the data (``t_cartesian_coordinates%x`` over an AoS of
      // velocity vectors).  Every AoS member is therefore deep-copied (allocate
      // + gather loop), which materialises the member-last layout correctly.
      bool memberAliasable = (outerRank == 0);
      int logicalKind = memberLogicalKind(memTy);

      auto recipe = b.getDictionaryAttr({
          b.getNamedAttr("flat_names", b.getArrayAttr({mkStr(flat)})),
          b.getNamedAttr("read_exprs", b.getArrayAttr({mkStr(read)})),
          b.getNamedAttr("write_expr", mkStr("")),
          b.getNamedAttr("rank", b.getI64IntegerAttr(totalRank)),
          b.getNamedAttr("shape_exprs", b.getArrayAttr(shapeExprs)),
          b.getNamedAttr("aliasable", b.getBoolAttr(memberAliasable)),
          b.getNamedAttr("scratch_dtype", mkStr(scratchDtype)),
          b.getNamedAttr("aos_alloc", b.getBoolAttr(false)),
          b.getNamedAttr("cap_symbol", mkStr("")),
          b.getNamedAttr("source_logical_kind", b.getI64IntegerAttr(logicalKind)),
      });

      // Per-member outer_expr ``<source>%<member>`` so the
      // emitter renders one ``c_f_pointer(c_loc(<source>%<member>),
      // <flat>, [...])`` per entry without ambiguity.
      std::string outerExpr = sourceBase + "%" + memName.str();

      auto entry = b.getDictionaryAttr({
          b.getNamedAttr("outer_expr", mkStr(outerExpr)),
          b.getNamedAttr("outer_type", mkStr(outerType)),
          b.getNamedAttr("writeback_intent", mkStr(intentStr)),
          b.getNamedAttr("recipe", recipe),
      });
      planEntries.push_back(entry);
    }
  }

  /// Nested-DT companion of ``recordStructArgEntry``: emit one
  /// FlattenEntry per leaf, each carrying the leaf's own rank,
  /// shape_exprs, dtype and full dotted ``%``-path.
  ///
  /// Example  --  ``type(t_outer)`` whose ``a%v`` and ``b%v`` are
  /// scalar-record members containing ``v(NX, NY)`` produces two
  /// entries:
  ///     outer_expr = "st%a%v", recipe = {
  ///       flat_names  = ["st_a_v"],
  ///       read_exprs  = ["st%a%v($i1, $i2)"],
  ///       rank        = 2,
  ///       shape_exprs = ["size(st%a%v, dim=1)", "size(st%a%v, dim=2)"],
  ///       aliasable   = true, scratch_dtype = "float64" }
  ///     outer_expr = "st%b%v", recipe = { ... "st_b_v" ... }
  ///
  /// The bindings emitter then renders one ``c_f_pointer(c_loc(st%a%v),
  /// st_a_v, [...])`` alias per entry -- zero-copy, same shape as the
  /// non-nested ``recordStructArgEntry`` output.  Per-leaf entries (vs
  /// a single shared-tuple recipe) keep heterogeneous nested structs
  /// correct: a leaf's rank / dtype no longer leaks onto its
  /// siblings.  ``collectFlatLeaves`` rejects unsupported leaf shapes
  /// upstream.
  void recordNestedStructArgEntry(hlfir::DeclareOp argDecl, llvm::ArrayRef<FlatLeaf> leaves,
                                  llvm::StringRef intentStr) {
    auto* ctx = argDecl.getContext();
    mlir::Builder b(ctx);
    auto mkStr = [&](llvm::StringRef s) -> mlir::Attribute { return b.getStringAttr(s); };

    std::string outerName = demangleVarName(argDecl.getUniqName());
    std::string outerType;
    {
      llvm::raw_string_ostream os(outerType);
      argDecl.getResult(0).getType().print(os);
    }

    // Emit one FlattenEntry per leaf: a heterogeneous nested struct
    // (differing rank / dtype / deferred-shape per leaf) needs each
    // alias to carry its own rank, shape_exprs, dtype and dotted
    // outer_expr -- the same per-member shape ``recordStructArgEntry``
    // emits.  A single shared recipe only round-trips a homogeneous
    // struct.
    bool emittedAny = false;
    for (auto& leaf : leaves) {
      // Joined ``a_v`` style suffix (same convention as
      // ``replaceStructArgNested``'s declare uniq_name).
      std::string joined;
      for (unsigned i = 0; i < leaf.path.size(); ++i) {
        if (i) joined += "_";
        joined += leaf.path[i];
      }
      std::string flat = outerName + "_" + joined;

      int leafRank = memberRank(leaf.leafTy);

      // Dotted path ``st%a%v`` for the read_expr / shape_expr.
      std::string dotted = outerName;
      for (auto& p : leaf.path) {
        dotted += "%";
        dotted += p;
      }
      std::string read = dotted;
      if (leafRank > 0) {
        read += "(";
        for (int i = 1; i <= leafRank; ++i) {
          if (i > 1) read += ", ";
          read += "$i" + std::to_string(i);
        }
        read += ")";
      }

      std::string scratchDtype = dtypeName(memberElementType(leaf.leafTy));
      if (scratchDtype.empty()) scratchDtype = "float64";
      int logicalKind = memberLogicalKind(leaf.leafTy);

      llvm::SmallVector<mlir::Attribute, 4> shapeExprs;
      for (int i = 1; i <= leafRank; ++i) {
        std::string s = "size(" + dotted + ", dim=" + std::to_string(i) + ")";
        shapeExprs.push_back(mkStr(s));
      }

      auto recipe = b.getDictionaryAttr({
          b.getNamedAttr("flat_names", b.getArrayAttr({mkStr(flat)})),
          b.getNamedAttr("read_exprs", b.getArrayAttr({mkStr(read)})),
          b.getNamedAttr("write_expr", mkStr("")),
          b.getNamedAttr("rank", b.getI64IntegerAttr(leafRank)),
          b.getNamedAttr("shape_exprs", b.getArrayAttr(shapeExprs)),
          b.getNamedAttr("aliasable", b.getBoolAttr(true)),
          b.getNamedAttr("scratch_dtype", mkStr(scratchDtype)),
          b.getNamedAttr("aos_alloc", b.getBoolAttr(false)),
          b.getNamedAttr("cap_symbol", mkStr("")),
          b.getNamedAttr("source_logical_kind", b.getI64IntegerAttr(logicalKind)),
      });

      auto entry = b.getDictionaryAttr({
          b.getNamedAttr("outer_expr", mkStr(dotted)),
          b.getNamedAttr("outer_type", mkStr(outerType)),
          b.getNamedAttr("writeback_intent", mkStr(intentStr)),
          b.getNamedAttr("recipe", recipe),
      });
      planEntries.push_back(entry);
      emittedAny = true;
    }

    if (!emittedAny) return;
  }

  /// Phase 5c-B (true SDFG-boundary): emit one FlattenEntry per
  /// AoS+allocatable member.  The bindings layer pads to max
  /// per-instance size, populates ``cap_<base>_<member>`` from the
  /// pack-in loop, and ships a 2D buffer ``<base>_<member>(N, cap)``
  /// to the SDFG.  The bridge declares the data block-arg with
  /// type ``ref<array<N x ?xT>>`` so the inner extent surfaces as a
  /// runtime symbol (``cap_<base>_<member>``) on the SDFG signature.
  void recordAosAllocEntry(hlfir::DeclareOp argDecl, fir::RecordType rec, llvm::StringRef memName,
                           llvm::StringRef intentStr, llvm::ArrayRef<int64_t> outerShape) {
    auto* ctx = argDecl.getContext();
    mlir::Builder b(ctx);
    auto mkStr = [&](llvm::StringRef s) -> mlir::Attribute { return b.getStringAttr(s); };

    std::string outerName = demangleVarName(argDecl.getUniqName());
    std::string outerType;
    {
      llvm::raw_string_ostream os(outerType);
      argDecl.getResult(0).getType().print(os);
    }

    // Locate the member type so we can record its dtype.
    mlir::Type memTy;
    for (auto& pair : rec.getTypeList())
      if (pair.first == memName) {
        memTy = pair.second;
        break;
      }

    std::string scratchDtype = "float64";
    if (memTy)
      if (std::string dt = dtypeName(memberElementType(memTy)); !dt.empty()) scratchDtype = dt;
    int logicalKind = memTy ? memberLogicalKind(memTy) : 0;

    std::string flatName = outerName + "_" + memName.str();
    std::string capName = "cap_" + flatName;
    unsigned outerRank = (unsigned)outerShape.size();

    // read_expr: ``<outer>($i1, ..., $iOR)%<member>($i_OR+1)``.
    // We always treat the allocatable member as 1-D for now (the
    // inner extent is the cap symbol  --  runtime-determined).
    std::string read = outerName;
    read += "(";
    for (unsigned i = 1; i <= outerRank; ++i) {
      if (i > 1) read += ", ";
      read += "$i" + std::to_string(i);
    }
    read += ")%";
    read += memName.str();
    read += "($i" + std::to_string((int)outerRank + 1) + ")";

    llvm::SmallVector<mlir::Attribute, 1> flatNames{mkStr(flatName)};
    llvm::SmallVector<mlir::Attribute, 1> readExprs{mkStr(read)};

    // shape_exprs: ``size(<outer>, dim=i)`` for each outer dim,
    // then the cap symbol for the inner.  The bindings layer's
    // ``_build_symbol_assigns`` skips the cap symbol because the
    // pack-in code computes it directly.
    llvm::SmallVector<mlir::Attribute, 2> shapeExprs;
    for (unsigned i = 1; i <= outerRank; ++i) {
      std::string s = "size(" + outerName + ", dim=" + std::to_string((int)i) + ")";
      shapeExprs.push_back(mkStr(s));
    }
    shapeExprs.push_back(mkStr(capName));

    int64_t totalRank = (int64_t)outerRank + 1;

    auto recipe = b.getDictionaryAttr({
        b.getNamedAttr("flat_names", b.getArrayAttr(flatNames)),
        b.getNamedAttr("read_exprs", b.getArrayAttr(readExprs)),
        b.getNamedAttr("write_expr", mkStr("")),
        b.getNamedAttr("rank", b.getI64IntegerAttr(totalRank)),
        b.getNamedAttr("shape_exprs", b.getArrayAttr(shapeExprs)),
        b.getNamedAttr("aliasable", b.getBoolAttr(false)),
        b.getNamedAttr("scratch_dtype", mkStr(scratchDtype)),
        b.getNamedAttr("aos_alloc", b.getBoolAttr(true)),
        b.getNamedAttr("cap_symbol", mkStr(capName)),
        b.getNamedAttr("source_logical_kind", b.getI64IntegerAttr(logicalKind)),
    });

    auto entry = b.getDictionaryAttr({
        b.getNamedAttr("outer_expr", mkStr(outerName)),
        b.getNamedAttr("outer_type", mkStr(outerType)),
        b.getNamedAttr("writeback_intent", mkStr(intentStr)),
        b.getNamedAttr("recipe", recipe),
    });
    planEntries.push_back(entry);
  }

  // -------------------------------------------------------------------
  // Function-level orchestration
  // -------------------------------------------------------------------

  void flattenFunc(mlir::func::FuncOp func) {
    if (func.isExternal()) return;
    // Skip private functions.  The bridge always builds an SDFG for
    // the single public entry; callees have been inlined into it.
    // Private siblings (kept alive only by a dispatch_table after
    // ``fir-polymorphic-op`` resolved every call site) would
    // otherwise pollute the module-level flatten_plan with phantom
    // CLASS dummies whose flat names look like top-level program
    // args at extract time.
    if (func.isPrivate()) return;

    // Step 0: decompose struct-valued ``hlfir.assign`` ops into
    // per-leaf assigns BEFORE the per-member declare rewrite runs.
    // ``val%var = indices`` (where both sides are entire struct
    // values) becomes one ``hlfir.assign`` per leaf of the struct
    // type; the existing designate-rewrite path then folds each
    // leaf assign into a flat ``val_var_<leaf> = indices_<leaf>``.
    // SKIPPED in split-only mode: the full ``hlfir-flatten-structs``
    // run later in the pipeline handles it, and running it twice can
    // leave a transient designate state that the inter-pass verifier
    // rejects.
    if (!splitOnly) decomposeStructAssigns(func);

    // Step 0.4: (C) split a multi-dim array-of-records member with SCALAR
    // inner record members (ICON's ``s%edges%primal_normal_cell(i,j,k)%v1``
    // pattern -- ``t_tangent_vectors{v1: f64, v2: f64}``) into one
    // dynamic-shape companion array dummy per inner member.  Inserts the
    // companion dummies before Step 0.5 / Step 1 see them.
    bool splitMD = splitMultiDimAoRScalarMembers(func);
    // Diagnose the LOCAL counterpart of the alloc-array-of-records-with-
    // scalar-inner-members pattern: when the chain ends at a LOCAL
    // ``hlfir.declare`` of a struct rather than a function argument,
    // ``splitMultiDimAoRScalarMembers`` cannot rewrite it -- the dummy
    // case lifts the inner members to function-argument companions and
    // relies on the caller-side bindings to marshal strided views; the
    // local case has no caller-side hook, and rewriting the local case
    // also requires teaching the pass to thread synthesised
    // ``_FortranAAllocatable{SetBounds,Allocate,Deallocate}`` calls
    // through per-member companion allocations (different element type
    // -> different strides, no shared heap).  Emit a clear error so the
    // bridge fails loudly with a TODO marker instead of letting the
    // bridge stumble into an opaque ``KeyError`` later when its emitter
    // hits the unresolvable designate chain.
    //
    // First try the local-case rewrite (per-member plain local allocatable
    // companions with inline allocmem/freemem); only if it cannot fire does
    // the diagnostic below report the unsupported shape.
    splitLocalAoRScalarMembers(func);
    // The never-allocated LOCAL alloc-AoR variant (any leaf type, incl
    // pointer/allocatable arrays) -- structural access only, caller supplies
    // the assumed-shape extents.  Additive: fires only where the allocate-
    // driven scalar handler above bails.
    splitLocalNoAllocAoRMembers(func);
    diagnoseLocalAoRScalarInnerMembers(func);

    // Step 0.5: (B) split an allocatable array-of-records struct-dummy member
    // accessed only by stable index symbols (ICON's prog(nnow)/prog(nnew)
    // double buffer) into one scalar-struct dummy per symbol, so Step 1
    // flattens each via the scalar path.  Inserts the new dummies before Step 1
    // sees them.
    bool splitArgs = splitDoubleBufferMembers(func);

    if (splitOnly) {
      // ``hlfir-split-aor-dummies`` mode: only the splits run.  Update
      // the function type so the block argument-list change validates;
      // skip the ``_soa`` rename so the full ``hlfir-flatten-structs``
      // run later finds the function under its original (or
      // marshal-rewritten) name.
      if (splitMD || splitArgs) {
        auto& block = func.front();
        auto newInputs = llvm::to_vector(block.getArgumentTypes());
        func.setType(mlir::FunctionType::get(func.getContext(), newInputs, func.getFunctionType().getResults()));
      }
      return;
    }

    // Step 1: collect struct-typed dummy arguments, rewrite them in
    // one pass over the original index list so mutations (insertArgument /
    // eraseArgument) don't invalidate later iterations.
    bool rewroteArgs = planAndReplaceStructArgs(func);

    if (splitMD || splitArgs || rewroteArgs) {
      auto& block = func.front();
      auto newInputs = llvm::to_vector(block.getArgumentTypes());
      func.setType(mlir::FunctionType::get(func.getContext(), newInputs, func.getFunctionType().getResults()));
      mlir::SymbolTable::setSymbolName(func, (func.getName() + "_soa").str());
    }

    // Step 2: local allocations of struct types.
    llvm::SmallVector<hlfir::DeclareOp, 8> work;
    func.walk([&](hlfir::DeclareOp d) {
      if (isLocallyFlattenable(d)) work.push_back(d);
    });
    for (auto d : work) splitLocal(d);
  }

  /// Decompose every struct-valued ``hlfir.assign`` in ``func`` into
  /// per-leaf assigns.  Source pattern (e.g. Fortran ``val%var =
  /// indices`` where both sides are whole struct values):
  ///
  ///   hlfir.assign %indices_struct to %val_var_struct : type<T>
  ///
  /// This pass walks the leaf set of ``T`` (via ``collectFlatLeaves``)
  /// and emits one ``hlfir.designate``-and-``hlfir.assign`` chain per
  /// leaf, copying the matching path from src to dst:
  ///
  ///   %src_leaf = hlfir.designate %indices_struct {"path0"}{"path1"}
  ///   %dst_leaf = hlfir.designate %val_var_struct  {"path0"}{"path1"}
  ///   hlfir.assign %src_leaf to %dst_leaf : <leaf_ty>
  ///
  /// The downstream per-member designate rewrite (``rewriteDesignate``
  /// / ``rewriteDesignateChain``) then folds each leaf chain into the
  /// flat-name form ``val_var_<path0>_<path1> = indices_<path0>_<path1>``.
  /// Array leaves stay whole-array assigns; scalar leaves stay scalar
  /// assigns.
  ///
  /// Out of scope: array-of-struct copies (whole-AoS-to-AoS).  Those
  /// would need to wrap each per-leaf assign in an outer-dim DO loop
  ///  --  separate work.
  void decomposeStructAssigns(mlir::func::FuncOp func) {
    llvm::SmallVector<hlfir::AssignOp, 16> targets;
    func.walk([&](hlfir::AssignOp op) {
      auto src = op.getRhs();
      auto dst = op.getLhs();
      bool srcIsRec = mlir::isa<fir::RecordType>(unwrapAll(src.getType()));
      bool dstIsRec = mlir::isa<fir::RecordType>(unwrapAll(dst.getType()));
      if (srcIsRec || dstIsRec) targets.push_back(op);
    });
    for (auto op : targets) decomposeStructAssign(op);
  }

  void decomposeStructAssign(hlfir::AssignOp op) {
    auto src = op.getRhs();
    auto dst = op.getLhs();

    bool outerIsArray = false;
    llvm::SmallVector<int64_t, 4> outerShape;
    auto rec = peelToRecord(dst.getType(), outerIsArray, outerShape);
    if (!rec) {
      // Try src side instead.
      rec = peelToRecord(src.getType(), outerIsArray, outerShape);
      if (!rec) return;
    }
    // AoS -> AoS struct copy is out of scope (would need an outer
    // index loop wrapping each leaf assign).  Leave the assign
    // alone; downstream gates flag it.
    if (outerIsArray) return;

    llvm::SmallVector<std::string, 4> prefix;
    llvm::SmallVector<FlatLeaf, 8> leaves;
    if (!collectFlatLeaves(rec, prefix, leaves)) return;

    mlir::OpBuilder b(op);
    auto loc = op.getLoc();

    // Build a designate chain over ``base`` following the path
    // components in ``leaf.path``.  Resolves the per-step result
    // type by looking up each component in the running record
    // type's member list.
    auto buildLeafDesignate = [&](mlir::Value base, const FlatLeaf& leaf) -> mlir::Value {
      mlir::Value cur = base;
      for (auto& component : leaf.path) {
        auto curRec = mlir::dyn_cast<fir::RecordType>(unwrapAll(cur.getType()));
        if (!curRec) return {};
        mlir::Type fieldTy;
        for (auto& p : curRec.getTypeList()) {
          if (p.first == component) {
            fieldTy = p.second;
            break;
          }
        }
        if (!fieldTy) return {};
        auto refFieldTy = fir::ReferenceType::get(fieldTy);
        auto componentAttr = mlir::StringAttr::get(b.getContext(), component);
        // Array-valued field needs a fir.shape operand for the
        // hlfir.designate verifier ("shape must be provided if
        // and only if the result is an array that is not a box
        // address").  Static extents only  --  dynamic-extent
        // record members aren't reachable through
        // ``collectFlatLeaves`` anyway.
        mlir::Value fieldShape;
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(fieldTy)) {
          auto exts = staticArrayExtents(seq);
          if (exts.empty()) return {};
          fieldShape = emitStaticShape(b, loc, exts);
        }
        auto newOp = b.create<hlfir::DesignateOp>(loc,
                                                  /*resultType0=*/refFieldTy,
                                                  /*memref=*/cur,
                                                  /*component=*/componentAttr,
                                                  /*component_shape=*/fieldShape,
                                                  /*indices=*/mlir::ValueRange{},
                                                  /*is_triplet=*/mlir::DenseBoolArrayAttr{},
                                                  /*substring=*/mlir::ValueRange{},
                                                  /*complex_part=*/mlir::BoolAttr{},
                                                  /*shape=*/fieldShape,
                                                  /*typeparams=*/mlir::ValueRange{},
                                                  /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
        cur = newOp.getResult();
      }
      return cur;
    };

    for (auto& leaf : leaves) {
      mlir::Value lhsLeaf = buildLeafDesignate(dst, leaf);
      mlir::Value rhsLeaf = buildLeafDesignate(src, leaf);
      if (!lhsLeaf || !rhsLeaf) {
        // One of the chains failed to resolve a component;
        // bail out for this assign  --  leave it intact and let
        // downstream gates flag it loudly.
        return;
      }
      // For scalar leaves, load the source ref first so
      // ``hlfir.assign`` carries a value RHS (matching the
      // standard scalar-assign shape that downstream extract_ast
      // recognises).  Array leaves stay as ``ref<array>``-to-
      // ``ref<array>`` whole-array copy.
      mlir::Value rhsValue = rhsLeaf;
      if (isSimpleScalar(leaf.leafTy)) {
        rhsValue = b.create<fir::LoadOp>(loc, rhsLeaf).getResult();
      }
      b.create<hlfir::AssignOp>(loc, rhsValue, lhsLeaf);
    }
    op.erase();
  }

  /// (C) Multi-dim AoR with scalar inner members.  A function-argument-rooted
  /// alloc / pointer array-of-records (rank >= 1) whose element type is a
  /// record with ONLY scalar leaf members (no inner pointer / allocatable /
  /// array members), accessed as ``<chain>(<idx>...)%<inner>``, is split into
  /// one dynamic-shape companion-array dummy per inner member
  /// (``<base>[_<member path>]_<inner>`` with rank = AoR rank, dtype = inner
  /// scalar).  Each access chain is rewritten to a designate on the matching
  /// companion at the same indices.
  ///
  /// ICON canonical pattern (``t_patch%edges%primal_normal_cell(i,j,k)%v1``
  /// with ``t_tangent_vectors {v1: f64, v2: f64}``): splits into two
  /// rank-3 dynamic-shape ``f64`` companions
  /// ``p_patch_edges_primal_normal_cell_v1`` and ``...v2``.  The bindings
  /// layer is responsible for marshalling strided views into the companions
  /// from the original AoR's box descriptor at call time -- this pass only
  /// performs the structural rewrite so the SDFG builds.
  ///
  /// Bails (returns false, leaves the function untouched) if any access uses
  /// an inner record with non-scalar members.
  ///
  /// :param func: the function whose AoR-rooted dummies to split.
  /// :returns: true if a per-inner-member companion was inserted.
  bool splitMultiDimAoRScalarMembers(mlir::func::FuncOp func) {
    auto& block = func.front();
    auto* ctx = func.getContext();

    // Map: ``argDecl.getResult(0)`` -> demangled base + argDecl.
    llvm::DenseMap<mlir::Value, std::pair<std::string, hlfir::DeclareOp>> argDecls;
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      hlfir::DeclareOp argDecl;
      for (auto* u : block.getArgument(i).getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
          argDecl = d;
          break;
        }
      if (!argDecl) continue;
      argDecls.try_emplace(argDecl.getResult(0), demangleVarName(argDecl.getUniqName()), argDecl);
    }
    if (argDecls.empty()) return false;

    // Key: (root, access-order member path, inner-member name).
    struct Key {
      void* root;
      std::vector<std::string> path;
      std::string inner;
      bool operator<(const Key& o) const {
        if (root != o.root) return root < o.root;
        if (path != o.path) return path < o.path;
        return inner < o.inner;
      }
    };
    struct Site {
      hlfir::DesignateOp innerDg;  // the inner-member designate to replace
      hlfir::DesignateOp elemDg;   // the multi-dim element designate
    };
    std::map<Key, llvm::SmallVector<Site, 4>> sites;
    llvm::SmallPtrSet<mlir::Operation*, 32> deadOps;
    // Track per-Key inner-member scalar type (must be uniform across sites).
    std::map<Key, mlir::Type> innerScalarTy;
    // Track per-(root, path) the AoR's element record type so we can verify
    // it has only scalar leaves.
    std::map<std::pair<void*, std::vector<std::string>>, fir::RecordType> elemRecByPath;

    func.walk([&](hlfir::DesignateOp innerDg) {
      // Inner-member designate: has component, no subscripts.
      auto innerComp = innerDg.getComponentAttr();
      if (!innerComp) return;
      if (!innerDg.getIndices().empty()) return;
      auto innerName = innerComp.getValue().str();

      // Memref must be an element designate: subscripts, no component.
      auto elemDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(innerDg.getMemref().getDefiningOp());
      if (!elemDg) return;
      if (elemDg.getComponentAttr()) return;
      if (elemDg.getIndices().empty()) return;

      // Element designate's memref must be a loaded box.
      auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(elemDg.getMemref().getDefiningOp());
      if (!ld) return;

      // Walk back via the member-designate chain to a function-arg declare.
      llvm::SmallVector<std::string, 4> walkedPath;
      llvm::SmallPtrSet<mlir::Operation*, 8> chainDead;
      chainDead.insert(ld);
      chainDead.insert(elemDg);
      // NB: ``innerDg`` is deliberately NOT tracked here -- it is erased
      // INLINE in the rewrite loop (after ``replaceAllUsesWith``), so adding
      // it to ``deadOps`` would make the trailing ``use_empty()`` sweep
      // dereference the freed op (a non-deterministic SmallPtrSet iteration ->
      // flaky heap corruption).  ``elemDg`` / ``ld`` / the member-chain hops
      // are erased ONLY by that guarded sweep, so they stay.
      mlir::Value v = ld.getMemref();
      while (true) {
        auto* d = v.getDefiningOp();
        auto md = mlir::dyn_cast_or_null<hlfir::DesignateOp>(d);
        if (!md) break;
        auto comp = md.getComponentAttr();
        if (!comp) break;
        walkedPath.push_back(comp.getValue().str());
        chainDead.insert(md);
        v = md.getMemref();
      }
      // Follow inlined-callee alias declares (``%alias = hlfir.declare
      // %blockarg#N ...``) back to the block-arg declare: after
      // ``hlfir-inline-all`` the access roots at the callee's dummy alias, not
      // the outer block arg (e.g. ICON-O coriolis' ``operators_coefficients``
      // reached through the inlined ``rot_vertex_ocean_3d``'s ``p_op_coeff``).
      while (argDecls.find(v) == argDecls.end()) {
        auto adcl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(v.getDefiningOp());
        if (!adcl) break;
        v = adcl.getMemref();
      }
      auto it = argDecls.find(v);
      if (it == argDecls.end()) return;

      // Element record type comes from the elemDg result (ref<RecordType>).
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(elemDg.getResult().getType());
      if (!refTy) return;
      auto elemRec = mlir::dyn_cast<fir::RecordType>(refTy.getEleTy());
      if (!elemRec) return;

      // Verify the inner record has only FLAT leaves: a plain scalar OR a
      // STATIC array of a plain scalar (e.g. ICON-O's
      // ``t_cartesian_coordinates :: x(3)``).  Reject allocatable / pointer /
      // record / dynamic-array / character members -- those are other lanes
      // (e.g. ``LiftAllocArrayOfRecords``).  A static-array leaf flattens to a
      // companion whose trailing dims are the member's extents and whose
      // whole-member read (``%x``) becomes a trailing contiguous section.
      auto isFlatLeaf = [](mlir::Type t) -> bool {
        if (isSimpleScalar(t)) return true;
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
          for (auto d : seq.getShape())
            if (d == fir::SequenceType::getUnknownExtent()) return false;
          return isSimpleScalar(seq.getEleTy());
        }
        return false;
      };
      bool flatLeavesOnly = true;
      mlir::Type matchedInnerTy;
      for (auto& p : elemRec.getTypeList()) {
        if (!isFlatLeaf(p.second)) {
          flatLeavesOnly = false;
          break;
        }
        if (p.first == innerName) matchedInnerTy = p.second;
      }
      if (!flatLeavesOnly || !matchedInnerTy) return;

      Key key;
      key.root = v.getAsOpaquePointer();
      key.path.assign(walkedPath.rbegin(), walkedPath.rend());
      key.inner = innerName;

      auto recKey = std::make_pair(key.root, key.path);
      elemRecByPath[recKey] = elemRec;

      sites[key].push_back({innerDg, elemDg});
      innerScalarTy[key] = matchedInnerTy;
      deadOps.insert(chainDead.begin(), chainDead.end());
    });
    if (sites.empty()) return false;

    bool changed = false;
    for (auto& kv : sites) {
      auto root = mlir::Value::getFromOpaquePointer(kv.first.root);
      auto rootIt = argDecls.find(root);
      if (rootIt == argDecls.end()) continue;
      auto& [demangledBase, argDecl] = rootIt->second;
      auto innerTy = innerScalarTy[kv.first];

      // Determine the AoR rank from the first site's element designate.
      unsigned outerRank = kv.second[0].elemDg.getIndices().size();

      // Companion type: ``ref<box<heap<array<? x ... x ? x innerTy>>>>``.
      // Dynamic shape -- the bridge's allocatable-array path will
      // synth shape symbols (``<name>_d0``, ``_d1``, ...) at extract
      // time and the bindings layer is expected to populate them.
      llvm::SmallVector<int64_t, 4> dynShape(outerRank, fir::SequenceType::getUnknownExtent());
      // Array leaf (``x(3)``): append the member's STATIC extents as trailing
      // companion dims and flatten to its element type, so ``c%e(i,j)%x``
      // maps to a contiguous trailing section ``c_e_x(i, j, 1:3:1)``.
      mlir::Type leafEleTy = innerTy;
      if (auto memSeq = mlir::dyn_cast<fir::SequenceType>(innerTy)) {
        for (auto d : memSeq.getShape()) dynShape.push_back(d);
        leafEleTy = memSeq.getEleTy();
      }
      auto arrTy = fir::SequenceType::get(dynShape, leafEleTy);
      auto heapTy = fir::HeapType::get(arrTy);
      auto boxTy = fir::BoxType::get(heapTy);
      auto refBoxTy = fir::ReferenceType::get(boxTy);

      mlir::OpBuilder b(argDecl);
      b.setInsertionPointAfter(argDecl);
      unsigned newIdx = block.getNumArguments();
      block.insertArgument(newIdx, refBoxTy, argDecl.getLoc());
      auto newArg = block.getArgument(newIdx);

      std::string name = demangledBase;
      for (auto& p : kv.first.path) name += "_" + p;
      name += "_" + kv.first.inner;

      mlir::NamedAttrList attrs;
      attrs.append("uniq_name", mlir::StringAttr::get(ctx, name));
      attrs.append("fortran_attrs",
                   fir::FortranVariableFlagsAttr::get(ctx, fir::FortranVariableFlagsEnum::allocatable));
      attrs.append(declareSegments(b, /*hasShape=*/false));
      auto decl = b.create<hlfir::DeclareOp>(argDecl.getLoc(), mlir::TypeRange{refBoxTy, refBoxTy},
                                             mlir::ValueRange{newArg}, attrs);

      // Record the FlattenEntry for this companion so the bindings wrapper
      // GATHERS ``<base>%<path>(i)%<inner>`` into it (and scatters back for
      // intent(inout)) instead of zero-stubbing.  ``collectFlatLeaves`` skips
      // the value-record ARRAY member (dynamic box of records) on the nested
      // path, so without this entry the companion is a real SDFG-boundary arg
      // with no data source and the emitter allocates it size-1 zeroed -- the
      // kernel then reads garbage/zero for every ``pnc(i)%v1``.  Mirrors the
      // non-aliasable AoS form ``recordStructArgEntry`` emits (aliasable=false
      // => allocate + element gather loop from the ``$i``-indexed read expr).
      {
        mlir::Builder pb(ctx);
        auto mkStr = [&](llvm::StringRef s) -> mlir::Attribute { return pb.getStringAttr(s); };
        // AoR source path ``<base>%<path0>%<path1>...`` (the array member).
        std::string aorPath = demangledBase;
        for (auto& seg : kv.first.path) aorPath += "%" + seg;
        // read: ``<aorPath>($i1, ..., $iOuterRank)%<inner>[($iOR+1, ...)]``.
        std::string read = aorPath + "(";
        for (unsigned i = 1; i <= outerRank; ++i) {
          if (i > 1) read += ", ";
          read += "$i" + std::to_string((int)i);
        }
        read += ")%" + kv.first.inner;
        int memRank = memberRank(innerTy);  // 0 scalar inner, >0 static-array inner (``%x(3)``)
        int totalRank = (int)outerRank + memRank;
        if (memRank > 0) {
          read += "(";
          for (int i = 1; i <= memRank; ++i) {
            if (i > 1) read += ", ";
            read += "$i" + std::to_string((int)outerRank + i);
          }
          read += ")";
        }
        // shape: outer AoR extents, then the inner member's own (static) dims.
        llvm::SmallVector<mlir::Attribute, 4> shapeExprs;
        for (unsigned i = 1; i <= outerRank; ++i)
          shapeExprs.push_back(mkStr("size(" + aorPath + ", dim=" + std::to_string((int)i) + ")"));
        if (memRank > 0) {
          std::string sample = aorPath + "(";
          for (unsigned i = 0; i < outerRank; ++i) {
            if (i) sample += ", ";
            sample += "1";
          }
          sample += ")%" + kv.first.inner;
          for (int i = 1; i <= memRank; ++i)
            shapeExprs.push_back(mkStr("size(" + sample + ", dim=" + std::to_string(i) + ")"));
        }
        std::string dtype = dtypeName(memberElementType(innerTy));
        if (dtype.empty()) dtype = "float64";
        std::string intentStr = extractIntent(argDecl.getFortranAttrs());
        std::string outerType;
        {
          llvm::raw_string_ostream os(outerType);
          argDecl.getResult(0).getType().print(os);
        }
        auto recipe = pb.getDictionaryAttr({
            pb.getNamedAttr("flat_names", pb.getArrayAttr({mkStr(name)})),
            pb.getNamedAttr("read_exprs", pb.getArrayAttr({mkStr(read)})),
            pb.getNamedAttr("write_expr", mkStr("")),
            pb.getNamedAttr("rank", pb.getI64IntegerAttr(totalRank)),
            pb.getNamedAttr("shape_exprs", pb.getArrayAttr(shapeExprs)),
            pb.getNamedAttr("aliasable", pb.getBoolAttr(false)),
            pb.getNamedAttr("scratch_dtype", mkStr(dtype)),
            pb.getNamedAttr("aos_alloc", pb.getBoolAttr(false)),
            pb.getNamedAttr("cap_symbol", mkStr("")),
            pb.getNamedAttr("source_logical_kind", pb.getI64IntegerAttr(memberLogicalKind(innerTy))),
        });
        auto entry = pb.getDictionaryAttr({
            pb.getNamedAttr("outer_expr", mkStr(aorPath + "%" + kv.first.inner)),
            pb.getNamedAttr("outer_type", mkStr(outerType)),
            pb.getNamedAttr("writeback_intent", mkStr(intentStr)),
            pb.getNamedAttr("recipe", recipe),
        });
        // Stamp the entry on the PERSISTENT companion declare rather than
        // pushing to the transient ``planEntries``: this split runs in the
        // ``hlfir-split-aor-dummies`` pre-pass, and the full
        // ``hlfir-flatten-structs`` pass re-runs it as an idempotent no-op (the
        // companions already exist) then RE-EMITS ``hlfir.flatten_plan`` from a
        // freshly-cleared ``planEntries`` -- overwriting the pre-pass's plan.
        // ``runOnOperation`` collects these stamps so the final plan carries the
        // gather entry too (mirrors the ``hlfir_bridge.dbuf_source`` stamp).
        decl->setAttr("hlfir_bridge.aor_flat_entry", entry);
      }

      for (auto& site : kv.second) {
        mlir::OpBuilder sb(site.innerDg);
        auto loc = site.innerDg.getLoc();
        // Load the box, then designate over the companion at the original
        // element indices.
        auto loadedBox = sb.create<fir::LoadOp>(loc, decl.getResult(0));
        mlir::Value repl;
        if (auto memSeq = mlir::dyn_cast<fir::SequenceType>(innerTy)) {
          // Array leaf: the WHOLE-member read (``%x``) maps to a trailing
          // contiguous SECTION of the companion -- the original element
          // indices (scalar) followed by a ``1:extent:1`` triplet per member
          // dim.  Result type mirrors the original member designate's
          // (``ref<array<N x T>>``).
          auto idxTy = sb.getIndexType();
          auto toIndex = [&](mlir::Value v) -> mlir::Value {
            return v.getType() == idxTy ? v : sb.create<fir::ConvertOp>(loc, idxTy, v).getResult();
          };
          llvm::SmallVector<mlir::Value, 8> indices;
          llvm::SmallVector<bool, 4> trip;
          for (auto v : site.elemDg.getIndices()) {
            indices.push_back(toIndex(v));
            trip.push_back(false);
          }
          llvm::SmallVector<mlir::Value, 2> shapeDims;
          for (auto d : memSeq.getShape()) {
            auto c1 = sb.create<mlir::arith::ConstantOp>(loc, idxTy, sb.getIndexAttr(1));
            auto cN = sb.create<mlir::arith::ConstantOp>(loc, idxTy, sb.getIndexAttr(d));
            indices.push_back(c1.getResult());
            indices.push_back(cN.getResult());
            indices.push_back(c1.getResult());
            trip.push_back(true);
            shapeDims.push_back(cN.getResult());
          }
          auto shp = sb.create<fir::ShapeOp>(loc, mlir::ValueRange{shapeDims}).getResult();
          auto newDg = sb.create<hlfir::DesignateOp>(loc, /*resultType0=*/site.innerDg.getResult().getType(),
                                                     /*memref=*/loadedBox.getResult(),
                                                     /*component=*/mlir::StringAttr{},
                                                     /*component_shape=*/mlir::Value{},
                                                     /*indices=*/mlir::ValueRange{indices},
                                                     /*is_triplet=*/sb.getDenseBoolArrayAttr(trip),
                                                     /*substring=*/mlir::ValueRange{},
                                                     /*complex_part=*/mlir::BoolAttr{},
                                                     /*shape=*/shp,
                                                     /*typeparams=*/mlir::ValueRange{},
                                                     /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
          repl = newDg.getResult();
        } else {
          // Scalar leaf: element access (7-arg convenience build).
          auto elemRefTy = fir::ReferenceType::get(innerTy);
          llvm::SmallVector<mlir::Value, 4> idxs(site.elemDg.getIndices().begin(), site.elemDg.getIndices().end());
          auto newDg = sb.create<hlfir::DesignateOp>(loc, elemRefTy, loadedBox.getResult(),
                                                     /*indices=*/idxs,
                                                     /*typeparams=*/mlir::ValueRange{},
                                                     /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
          repl = newDg.getResult();
        }
        site.innerDg.getResult().replaceAllUsesWith(repl);
        site.innerDg.erase();
      }
      changed = true;
    }
    // Drop the dead element designates and member-chain ops.  The element
    // designate is unconditionally dead after the rewrite (its only user
    // was the innerDg we replaced); the member chain is dead only if no
    // other access through the same path remained.
    for (auto* op : deadOps)
      if (op->use_empty()) op->erase();
    return changed;
  }

  /// (A.1') Rewrite a LOCAL-rooted alloc-array-of-records-with-scalar-inner-
  /// members into one plain local allocatable companion per inner member --
  /// the local counterpart of :func:`splitMultiDimAoRScalarMembers` (which
  /// handles the function-argument case via per-member dummy companions the
  /// caller marshals).  Because the inner members are uniform scalars, each
  /// companion is a plain local allocatable the rest of the bridge already
  /// lowers, so we drive its storage with INLINE ``fir.allocmem`` /
  /// ``fir.freemem`` (using the extents captured from the original
  /// ``_FortranAAllocatableSetBounds`` calls).  The ``_FortranAAllocatable*``
  /// runtime calls flang emits for the derived-type AoR element -- which the
  /// bridge cannot lower for a SoA split -- are dropped.
  ///
  /// :returns: true if at least one local AoR member was rewritten.
  bool splitLocalAoRScalarMembers(mlir::func::FuncOp func) {
    auto& block = func.front();
    auto* ctx = func.getContext();
    auto idxTy = mlir::IndexType::get(ctx);
    auto i64Ty = mlir::IntegerType::get(ctx, 64);

    llvm::DenseSet<mlir::Value> argDeclResults;
    for (unsigned i = 0; i < block.getNumArguments(); ++i)
      for (auto* u : block.getArgument(i).getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) argDeclResults.insert(d.getResult(0));

    // (root declare result, member path incl. AoR member as last hop).
    using RPKey = std::pair<void*, std::vector<std::string>>;
    struct Site {
      hlfir::DesignateOp innerDg, elemDg;
    };
    struct PathInfo {
      hlfir::DeclareOp rootDecl;
      unsigned rank = 0;
      // inner member name -> (scalar type, access sites).
      std::map<std::string, std::pair<mlir::Type, llvm::SmallVector<Site, 8>>> members;
    };
    std::map<RPKey, PathInfo> byPath;

    func.walk([&](hlfir::DesignateOp innerDg) {
      auto innerComp = innerDg.getComponentAttr();
      if (!innerComp || !innerDg.getIndices().empty()) return;
      auto innerName = innerComp.getValue().str();
      auto elemDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(innerDg.getMemref().getDefiningOp());
      if (!elemDg || elemDg.getComponentAttr() || elemDg.getIndices().empty()) return;
      auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(elemDg.getMemref().getDefiningOp());
      if (!ld) return;
      llvm::SmallVector<std::string, 4> walkedPath;
      mlir::Value v = ld.getMemref();
      while (auto md = mlir::dyn_cast_or_null<hlfir::DesignateOp>(v.getDefiningOp())) {
        if (!md.getComponentAttr()) break;
        walkedPath.push_back(md.getComponentAttr().getValue().str());
        v = md.getMemref();
      }
      if (argDeclResults.contains(v)) return;  // dummy case
      auto rootDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(v.getDefiningOp());
      if (!rootDecl || !mlir::isa_and_nonnull<fir::AllocaOp>(rootDecl.getMemref().getDefiningOp())) return;
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(elemDg.getResult().getType());
      if (!refTy) return;
      auto elemRec = mlir::dyn_cast<fir::RecordType>(refTy.getEleTy());
      if (!elemRec) return;
      mlir::Type matchedTy;
      for (auto& p : elemRec.getTypeList()) {
        if (mlir::isa<fir::SequenceType, fir::BoxType, fir::ReferenceType, fir::HeapType, fir::PointerType,
                      fir::RecordType, fir::CharacterType>(p.second))
          return;  // not a pure-scalar element record
        if (p.first == innerName) matchedTy = p.second;
      }
      if (!matchedTy) return;
      RPKey k{v.getAsOpaquePointer(), std::vector<std::string>(walkedPath.rbegin(), walkedPath.rend())};
      auto& pi = byPath[k];
      pi.rootDecl = rootDecl;
      pi.rank = elemDg.getIndices().size();
      auto& mem = pi.members[innerName];
      mem.first = matchedTy;
      mem.second.push_back({innerDg, elemDg});
    });
    if (byPath.empty()) return false;

    // Match a runtime-call box operand back to its (root, path): peel
    // ``fir.convert``, then walk the member-designate chain to the root.
    auto boxPathRoot = [&](mlir::Value boxOperand) -> std::pair<void*, std::vector<std::string>> {
      mlir::Value v = boxOperand;
      while (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(v.getDefiningOp())) v = cv.getValue();
      llvm::SmallVector<std::string, 4> p;
      while (auto md = mlir::dyn_cast_or_null<hlfir::DesignateOp>(v.getDefiningOp())) {
        if (!md.getComponentAttr()) break;
        p.push_back(md.getComponentAttr().getValue().str());
        v = md.getMemref();
      }
      return {v.getAsOpaquePointer(), std::vector<std::string>(p.rbegin(), p.rend())};
    };

    // Collect the SetBounds / Allocate / Deallocate runtime calls per path.
    struct RTCalls {
      // dim -> (lo, hi) i64 values from SetBounds.
      std::map<unsigned, std::pair<mlir::Value, mlir::Value>> bounds;
      fir::CallOp allocate, deallocate;
      llvm::SmallVector<fir::CallOp, 6> setBounds;
    };
    std::map<RPKey, RTCalls> rtByPath;
    func.walk([&](fir::CallOp call) {
      auto callee = call.getCallee();
      if (!callee) return;
      llvm::StringRef name = callee->getRootReference().getValue();
      bool isSB = name == "_FortranAAllocatableSetBounds";
      bool isAl = name == "_FortranAAllocatableAllocate";
      bool isDe = name == "_FortranAAllocatableDeallocate";
      if (!isSB && !isAl && !isDe) return;
      if (call.getArgs().empty()) return;
      auto pr = boxPathRoot(call.getArgs()[0]);
      auto it = byPath.find(pr);
      if (it == byPath.end()) return;
      auto& rt = rtByPath[pr];
      if (isSB) {
        // SetBounds(box, dim_i32, lo_i64, hi_i64).
        if (call.getArgs().size() >= 4) {
          if (auto dimC = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(call.getArgs()[1].getDefiningOp())) {
            auto dimAttr = mlir::dyn_cast<mlir::IntegerAttr>(dimC.getValue());
            if (dimAttr) rt.bounds[dimAttr.getInt()] = {call.getArgs()[2], call.getArgs()[3]};
          }
        }
        rt.setBounds.push_back(call);
      } else if (isAl) {
        rt.allocate = call;
      } else {
        rt.deallocate = call;
      }
    });

    llvm::SmallPtrSet<mlir::Operation*, 32> deadOps;
    bool changed = false;

    for (auto& kv : byPath) {
      auto& pi = kv.second;
      auto rtIt = rtByPath.find(kv.first);
      if (rtIt == rtByPath.end() || !rtIt->second.allocate) continue;
      auto& rt = rtIt->second;
      unsigned rank = pi.rank;
      // Need every dim's bounds to size the companions.
      bool haveAllBounds = rt.bounds.size() >= rank;
      for (unsigned d = 0; d < rank && haveAllBounds; ++d)
        if (!rt.bounds.count(d)) haveAllBounds = false;
      if (!haveAllBounds) continue;

      auto demangledBase = demangleVarName(pi.rootDecl.getUniqName());

      // Build, per inner member, a local allocatable companion + its
      // (init, allocate, deallocate) inline storage.
      for (auto& m : pi.members) {
        const std::string& innerName = m.first;
        mlir::Type innerTy = m.second.first;
        auto& siteList = m.second.second;

        llvm::SmallVector<int64_t, 4> dynShape(rank, fir::SequenceType::getUnknownExtent());
        auto arrTy = fir::SequenceType::get(dynShape, innerTy);
        auto heapTy = fir::HeapType::get(arrTy);
        auto boxTy = fir::BoxType::get(heapTy);
        auto refBoxTy = fir::ReferenceType::get(boxTy);
        auto shapeTy = fir::ShapeType::get(ctx, rank);

        std::string name = demangledBase;
        for (auto& p : kv.first.second) name += "_" + p;
        name += "_" + innerName;
        // Mangle the companion's uniq_name with the root's Flang scope
        // prefix (``_QF<func>E``) so the bridge's variable extraction
        // classifies it as a function-LOCAL (transient), not a program
        // argument -- it keys local-vs-input off the ``_QF...E<var>`` form
        // (extract_vars.cpp).  demangle(uniqName) recovers the readable
        // array name unchanged.
        std::string rootUniq = pi.rootDecl.getUniqName().str();
        auto ePos = rootUniq.rfind('E');
        std::string scopePrefix = ePos == std::string::npos ? std::string() : rootUniq.substr(0, ePos + 1);
        std::string uniqName = scopePrefix + name;
        auto loc = pi.rootDecl.getLoc();

        // (a) descriptor init right after the root declare.
        mlir::OpBuilder b(pi.rootDecl);
        b.setInsertionPointAfter(pi.rootDecl);
        auto boxAlloca = b.create<fir::AllocaOp>(loc, boxTy);
        auto zero = b.create<fir::ZeroOp>(loc, heapTy);
        llvm::SmallVector<mlir::Value, 4> zeroDims(rank,
                                                   b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(0)));
        auto shape0 = b.create<fir::ShapeOp>(loc, shapeTy, zeroDims);
        auto embox0 = b.create<fir::EmboxOp>(loc, boxTy, zero, shape0);
        b.create<fir::StoreOp>(loc, embox0, boxAlloca);
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, uniqName));
        attrs.append("fortran_attrs",
                     fir::FortranVariableFlagsAttr::get(ctx, fir::FortranVariableFlagsEnum::allocatable));
        attrs.append(declareSegments(b, /*hasShape=*/false));
        auto decl =
            b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{refBoxTy, refBoxTy}, mlir::ValueRange{boxAlloca}, attrs);

        // (b) inline allocmem just before the original Allocate call.
        mlir::OpBuilder ab(rt.allocate);
        auto aloc = rt.allocate.getLoc();
        auto one64 = ab.create<mlir::arith::ConstantOp>(aloc, i64Ty, ab.getI64IntegerAttr(1));
        auto zeroIdx = ab.create<mlir::arith::ConstantOp>(aloc, idxTy, ab.getIndexAttr(0));
        llvm::SmallVector<mlir::Value, 4> extents;
        for (unsigned d = 0; d < rank; ++d) {
          auto [lo, hi] = rt.bounds[d];
          auto diff = ab.create<mlir::arith::SubIOp>(aloc, hi, lo);
          auto ext64 = ab.create<mlir::arith::AddIOp>(aloc, diff, one64);
          auto extIdx = ab.create<fir::ConvertOp>(aloc, idxTy, ext64);
          auto cmp = ab.create<mlir::arith::CmpIOp>(aloc, mlir::arith::CmpIPredicate::sgt, extIdx, zeroIdx);
          extents.push_back(ab.create<mlir::arith::SelectOp>(aloc, cmp, extIdx, zeroIdx));
        }
        // The bridge recognises an allocatable's storage by an allocmem
        // whose ``uniq_name`` is ``<decl-uniq_name>.alloc`` (see
        // bridge/ast/expressions.cpp); without it the allocmem falls
        // through to buildExpr as an unhandled op.
        std::string allocName = uniqName + ".alloc";
        auto am = ab.create<fir::AllocMemOp>(aloc, arrTy, /*uniq_name=*/llvm::StringRef(allocName), mlir::ValueRange{},
                                             extents);
        am->setAttr("fir.must_be_heap", ab.getBoolAttr(true));
        auto shapeA = ab.create<fir::ShapeOp>(aloc, shapeTy, extents);
        auto eboxA = ab.create<fir::EmboxOp>(aloc, boxTy, am, shapeA);
        ab.create<fir::StoreOp>(aloc, eboxA, decl.getResult(0));

        // (c) inline freemem just before the original Deallocate call (if
        // any -- a kernel that never deallocates simply leaks, as the
        // original would).
        if (rt.deallocate) {
          mlir::OpBuilder db(rt.deallocate);
          auto dloc = rt.deallocate.getLoc();
          auto ldBox = db.create<fir::LoadOp>(dloc, decl.getResult(0));
          auto heapVal = db.create<fir::BoxAddrOp>(dloc, heapTy, ldBox);
          db.create<fir::FreeMemOp>(dloc, heapVal);
        }

        // (d) rewrite every inner-member designate to a designate over the
        // companion box at the same indices (reads and writes alike).
        for (auto& site : siteList) {
          mlir::OpBuilder sb(site.innerDg);
          auto loadedBox = sb.create<fir::LoadOp>(site.innerDg.getLoc(), decl.getResult(0));
          auto elemRefTy = fir::ReferenceType::get(innerTy);
          llvm::SmallVector<mlir::Value, 4> idxs(site.elemDg.getIndices().begin(), site.elemDg.getIndices().end());
          auto newDg = sb.create<hlfir::DesignateOp>(site.innerDg.getLoc(), elemRefTy, loadedBox.getResult(), idxs,
                                                     /*typeparams=*/mlir::ValueRange{},
                                                     /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
          site.innerDg.getResult().replaceAllUsesWith(newDg.getResult());
          deadOps.insert(site.innerDg);
          if (site.elemDg) deadOps.insert(site.elemDg);
        }
        changed = true;
      }

      // Retire the original AoR runtime-call sequence; its operand chains
      // (member designates, converts, loads) become dead and are swept below.
      for (auto sb : rt.setBounds) deadOps.insert(sb);
      deadOps.insert(rt.allocate);
      if (rt.deallocate) deadOps.insert(rt.deallocate);
    }

    if (!changed) return false;
    // The Allocate/Deallocate calls return an i32 status that may be read
    // (an ``allocate(stat=)`` check); feed those uses a constant 0 (success)
    // so the calls become dead and erasable.
    for (auto* op : deadOps)
      for (auto res : op->getResults())
        if (!res.use_empty())
          if (auto it = mlir::dyn_cast<mlir::IntegerType>(res.getType())) {
            mlir::OpBuilder pb(op);
            res.replaceAllUsesWith(pb.create<mlir::arith::ConstantOp>(op->getLoc(), it, mlir::IntegerAttr::get(it, 0)));
          }
    for (auto* op : deadOps)
      if (op->use_empty()) op->erase();
    // Fixpoint-erase the now-dead operand chains the retired runtime calls /
    // accesses left behind (member designates, box loads, converts, the
    // descriptor-init embox/shape/zero_bits).  Collect-then-erase so we never
    // erase the op a walk is visiting.
    bool progress = true;
    while (progress) {
      progress = false;
      llvm::SmallVector<mlir::Operation*, 16> nowDead;
      func.walk([&](mlir::Operation* op) {
        if (mlir::isa<hlfir::DesignateOp, fir::ConvertOp, fir::LoadOp, fir::EmboxOp, fir::ShapeOp, fir::ZeroOp>(op) &&
            op->use_empty())
          nowDead.push_back(op);
      });
      for (auto* op : nowDead) {
        op->erase();
        progress = true;
      }
    }
    return true;
  }

  /// (A.1b) LOCAL-rooted alloc-array-of-records member that is NEVER allocated
  /// (purely structural access -- ``call f(p%items(1))`` / ``x = p%items(1)%w``
  /// with no ``allocate(p%items(...))``), whose leaf member may be ANY flat
  /// type: scalar, static array, OR a pointer/allocatable scalar/array
  /// (``real, pointer :: w(:,:)``).
  ///
  /// ``splitLocalAoRScalarMembers`` bails here -- it requires an ``Allocate``
  /// runtime call to size the companion and admits scalar leaves only; the
  /// dummy paths bail too (local root).  Flatten each
  /// ``<root>%<path>(idx...)%<member>`` to ONE local allocatable companion
  /// ``<root>_<path>_<member>`` of shape ``[AoR-dims..., member-dims...]`` (all
  /// runtime ``?`` -- no ``allocmem``; the bridge synthesises ``_d<i>`` extent
  /// symbols the caller supplies, exactly as for the never-allocated assumed-
  /// shape contract).  A SCALAR / static-array leaf rewrites its member
  /// designate to a companion element / trailing-section designate; a
  /// POINTER / ALLOCATABLE leaf (whose member designate yields a *box* that is
  /// then ``load``ed and indexed) instead rewrites that ``load`` to a box
  /// SECTION over the companion ``[idx..., :, ...]`` so both the read and any
  /// inlined dummy that aliased the pointer route to the one companion.
  ///
  /// :returns: true if at least one member was rewritten.
  bool splitLocalNoAllocAoRMembers(mlir::func::FuncOp func) {
    auto& block = func.front();
    auto* ctx = func.getContext();
    auto idxTy = mlir::IndexType::get(ctx);

    llvm::DenseSet<mlir::Value> argDeclResults;
    for (unsigned i = 0; i < block.getNumArguments(); ++i)
      for (auto* u : block.getArgument(i).getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) argDeclResults.insert(d.getResult(0));

    // Decompose a leaf member type into (scalar element type, extents).  Empty
    // extents = scalar leaf.  ``dynamic`` is set when the leaf is a pointer /
    // allocatable (box) whose member designate yields a box that must be
    // load-and-sectioned rather than directly replaced.  Returns null type if
    // the member is not a flat leaf (nested record / character / ...).
    auto leafEle = [](mlir::Type t, llvm::SmallVectorImpl<int64_t>& dims, bool& isBox) -> mlir::Type {
      isBox = false;
      if (isSimpleScalar(t)) return t;
      if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
        if (!isSimpleScalar(seq.getEleTy())) return nullptr;
        for (auto d : seq.getShape())
          if (d == fir::SequenceType::getUnknownExtent())
            return nullptr;
          else
            dims.push_back(d);
        return seq.getEleTy();
      }
      if (auto box = mlir::dyn_cast<fir::BaseBoxType>(t)) {
        mlir::Type inner = box.getEleTy();
        if (auto h = mlir::dyn_cast<fir::HeapType>(inner))
          inner = h.getEleTy();
        else if (auto p = mlir::dyn_cast<fir::PointerType>(inner))
          inner = p.getEleTy();
        else
          return nullptr;
        isBox = true;
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner)) {
          if (!isSimpleScalar(seq.getEleTy())) return nullptr;
          for (auto d : seq.getShape()) dims.push_back(d);  // ? allowed
          return seq.getEleTy();
        }
        if (isSimpleScalar(inner)) return inner;
      }
      return nullptr;
    };

    using RPKey = std::pair<void*, std::vector<std::string>>;
    struct Site {
      hlfir::DesignateOp innerDg, elemDg;
    };
    struct PathInfo {
      hlfir::DeclareOp rootDecl;
      unsigned rank = 0;
      std::map<std::string, std::pair<mlir::Type, llvm::SmallVector<Site, 8>>> members;
    };
    std::map<RPKey, PathInfo> byPath;

    func.walk([&](hlfir::DesignateOp innerDg) {
      auto innerComp = innerDg.getComponentAttr();
      if (!innerComp || !innerDg.getIndices().empty()) return;
      auto innerName = innerComp.getValue().str();
      auto* mdef = innerDg.getMemref().getDefiningOp();
      hlfir::DesignateOp elemDg;  // null in the rank-0 scalar-struct case.
      fir::LoadOp ld;
      fir::RecordType elemRec;
      if (auto ed = mlir::dyn_cast_or_null<hlfir::DesignateOp>(mdef)) {
        // (a) element designate of an alloc-array-of-records.
        if (ed.getComponentAttr() || ed.getIndices().empty()) return;
        elemDg = ed;
        ld = mlir::dyn_cast_or_null<fir::LoadOp>(ed.getMemref().getDefiningOp());
        if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(ed.getResult().getType()))
          elemRec = mlir::dyn_cast<fir::RecordType>(refTy.getEleTy());
      } else if (auto l = mlir::dyn_cast_or_null<fir::LoadOp>(mdef)) {
        // (b) direct load of a SCALAR allocatable / class struct box
        // (``class(t),allocatable :: p`` -> ``p%n``): rank 0, no element index.
        ld = l;
        elemRec = mlir::dyn_cast<fir::RecordType>(unwrapAll(l.getResult().getType()));
      } else {
        return;
      }
      if (!ld || !elemRec) return;
      llvm::SmallVector<std::string, 4> walkedPath;
      mlir::Value v = ld.getMemref();
      while (auto md = mlir::dyn_cast_or_null<hlfir::DesignateOp>(v.getDefiningOp())) {
        if (!md.getComponentAttr()) break;
        walkedPath.push_back(md.getComponentAttr().getValue().str());
        v = md.getMemref();
      }
      if (argDeclResults.contains(v)) return;  // dummy case
      auto rootDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(v.getDefiningOp());
      if (!rootDecl || !mlir::isa_and_nonnull<fir::AllocaOp>(rootDecl.getMemref().getDefiningOp())) return;
      // Per-member: only the ACCESSED member must be a flat leaf -- other
      // members (an EXTENDS parent record, an unaccessed allocatable, ...) are
      // irrelevant to THIS member's companion (each access is its own site).
      mlir::Type matchedTy;
      for (auto& p : elemRec.getTypeList())
        if (p.first == innerName) {
          matchedTy = p.second;
          break;
        }
      llvm::SmallVector<int64_t, 4> probeDims;
      bool probeBox;
      if (!matchedTy || !leafEle(matchedTy, probeDims, probeBox)) return;
      RPKey k{v.getAsOpaquePointer(), std::vector<std::string>(walkedPath.rbegin(), walkedPath.rend())};
      auto& pi = byPath[k];
      pi.rootDecl = rootDecl;
      pi.rank = elemDg ? elemDg.getIndices().size() : 0;
      auto& mem = pi.members[innerName];
      mem.first = matchedTy;
      mem.second.push_back({innerDg, elemDg});
    });
    if (byPath.empty()) return false;

    llvm::SmallPtrSet<mlir::Operation*, 32> deadOps;
    bool changed = false;
    for (auto& kv : byPath) {
      auto& pi = kv.second;
      unsigned rank = pi.rank;
      auto demangledBase = demangleVarName(pi.rootDecl.getUniqName());
      std::string rootUniq = pi.rootDecl.getUniqName().str();
      auto ePos = rootUniq.rfind('E');
      std::string scopePrefix = ePos == std::string::npos ? std::string() : rootUniq.substr(0, ePos + 1);

      for (auto& m : pi.members) {
        const std::string& innerName = m.first;
        mlir::Type innerTy = m.second.first;
        auto& siteList = m.second.second;
        llvm::SmallVector<int64_t, 4> leafDims;
        bool leafIsBox = false;
        mlir::Type leafEleTy = leafEle(innerTy, leafDims, leafIsBox);
        if (!leafEleTy) continue;

        // Rank-0 scalar struct with a SCALAR leaf (``class(t),allocatable ::
        // p``
        // -> ``p%n``): a plain scalar local transient (no box / array). Rewrite
        // every member designate -- read or assign target -- to its declare
        // ref.
        if (rank == 0 && leafDims.empty() && !leafIsBox) {
          std::string name0 = demangledBase;
          for (auto& p : kv.first.second) name0 += "_" + p;
          name0 += "_" + innerName;
          auto loc0 = pi.rootDecl.getLoc();
          mlir::OpBuilder b0(pi.rootDecl);
          b0.setInsertionPointAfter(pi.rootDecl);
          auto refEle = fir::ReferenceType::get(leafEleTy);
          auto sAlloca = b0.create<fir::AllocaOp>(loc0, leafEleTy);
          mlir::NamedAttrList attrs0;
          attrs0.append("uniq_name", mlir::StringAttr::get(ctx, scopePrefix + name0));
          attrs0.append(declareSegments(b0, /*hasShape=*/false));
          auto sDecl =
              b0.create<hlfir::DeclareOp>(loc0, mlir::TypeRange{refEle, refEle}, mlir::ValueRange{sAlloca}, attrs0);
          for (auto& site : siteList) {
            site.innerDg.getResult().replaceAllUsesWith(sDecl.getResult(0));
            deadOps.insert(site.innerDg);
          }
          changed = true;
          continue;
        }

        // Companion shape: AoR record dims (runtime ?) ++ leaf dims.
        llvm::SmallVector<int64_t, 6> compShape(rank, fir::SequenceType::getUnknownExtent());
        for (auto d : leafDims) compShape.push_back(d);
        auto arrTy = fir::SequenceType::get(compShape, leafEleTy);
        auto heapTy = fir::HeapType::get(arrTy);
        auto boxTy = fir::BoxType::get(heapTy);
        auto refBoxTy = fir::ReferenceType::get(boxTy);
        auto shapeTy = fir::ShapeType::get(ctx, compShape.size());

        std::string name = demangledBase;
        for (auto& p : kv.first.second) name += "_" + p;
        name += "_" + innerName;
        std::string uniqName = scopePrefix + name;
        auto loc = pi.rootDecl.getLoc();

        // Local allocatable companion, zero-init descriptor, NO allocmem -- the
        // bridge symbolises its ``?`` extents (caller-supplied), matching the
        // never-allocated assumed-shape contract.
        mlir::OpBuilder b(pi.rootDecl);
        b.setInsertionPointAfter(pi.rootDecl);
        auto boxAlloca = b.create<fir::AllocaOp>(loc, boxTy);
        auto zero = b.create<fir::ZeroOp>(loc, heapTy);
        llvm::SmallVector<mlir::Value, 6> zeroDims(compShape.size(),
                                                   b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(0)));
        auto shape0 = b.create<fir::ShapeOp>(loc, shapeTy, zeroDims);
        auto embox0 = b.create<fir::EmboxOp>(loc, boxTy, zero, shape0);
        b.create<fir::StoreOp>(loc, embox0, boxAlloca);
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, uniqName));
        attrs.append("fortran_attrs",
                     fir::FortranVariableFlagsAttr::get(ctx, fir::FortranVariableFlagsEnum::allocatable));
        attrs.append(declareSegments(b, /*hasShape=*/false));
        auto decl =
            b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{refBoxTy, refBoxTy}, mlir::ValueRange{boxAlloca}, attrs);

        for (auto& site : siteList) {
          auto sloc = site.innerDg.getLoc();
          llvm::SmallVector<mlir::Value, 4> elemIdx;
          if (site.elemDg)  // null for a rank-0 scalar-struct member
            elemIdx.assign(site.elemDg.getIndices().begin(), site.elemDg.getIndices().end());
          auto toIndex = [&](mlir::OpBuilder& sb, mlir::Value vv) -> mlir::Value {
            return vv.getType() == idxTy ? vv : sb.create<fir::ConvertOp>(sloc, idxTy, vv).getResult();
          };
          if (!leafIsBox) {
            // Scalar / static-array leaf: replace the member designate.
            mlir::OpBuilder sb(site.innerDg);
            auto loadedBox = sb.create<fir::LoadOp>(sloc, decl.getResult(0));
            mlir::Value repl;
            if (auto memSeq = mlir::dyn_cast<fir::SequenceType>(innerTy)) {
              llvm::SmallVector<mlir::Value, 8> indices;
              llvm::SmallVector<bool, 4> trip;
              for (auto vv : elemIdx) {
                indices.push_back(toIndex(sb, vv));
                trip.push_back(false);
              }
              llvm::SmallVector<mlir::Value, 2> shapeDims;
              auto c1 = sb.create<mlir::arith::ConstantOp>(sloc, idxTy, sb.getIndexAttr(1));
              for (auto d : memSeq.getShape()) {
                auto cN = sb.create<mlir::arith::ConstantOp>(sloc, idxTy, sb.getIndexAttr(d));
                indices.push_back(c1);
                indices.push_back(cN);
                indices.push_back(c1);
                trip.push_back(true);
                shapeDims.push_back(cN);
              }
              auto shp = sb.create<fir::ShapeOp>(sloc, mlir::ValueRange{shapeDims});
              repl = sb.create<hlfir::DesignateOp>(sloc, site.innerDg.getResult().getType(), loadedBox.getResult(),
                                                   mlir::StringAttr{}, mlir::Value{}, mlir::ValueRange{indices},
                                                   sb.getDenseBoolArrayAttr(trip), mlir::ValueRange{}, mlir::BoolAttr{},
                                                   shp, mlir::ValueRange{}, fir::FortranVariableFlagsAttr{})
                         .getResult();
            } else {
              auto elemRefTy = fir::ReferenceType::get(innerTy);
              repl = sb.create<hlfir::DesignateOp>(sloc, elemRefTy, loadedBox.getResult(), elemIdx, mlir::ValueRange{},
                                                   fir::FortranVariableFlagsAttr{})
                         .getResult();
            }
            site.innerDg.getResult().replaceAllUsesWith(repl);
            deadOps.insert(site.innerDg);
            if (site.elemDg) deadOps.insert(site.elemDg);
            changed = true;
            continue;
          }
          // Pointer / allocatable leaf: the member designate yields a box that
          // is ``load``ed, then indexed.  Replace that LOAD with a box SECTION
          // over the companion ``[idx..., :leaf-dims:]`` so the loaded box (and
          // anything aliasing it, e.g. an inlined dummy) points at the
          // companion's per-element slab.
          for (auto* u : llvm::to_vector(site.innerDg.getResult().getUsers())) {
            auto ldBox = mlir::dyn_cast<fir::LoadOp>(u);
            if (!ldBox) continue;
            mlir::OpBuilder sb(ldBox);
            auto loadedComp = sb.create<fir::LoadOp>(sloc, decl.getResult(0));
            llvm::SmallVector<mlir::Value, 8> indices;
            llvm::SmallVector<bool, 4> trip;
            for (auto vv : elemIdx) {
              indices.push_back(toIndex(sb, vv));
              trip.push_back(false);
            }
            auto c1 = sb.create<mlir::arith::ConstantOp>(sloc, idxTy, sb.getIndexAttr(1));
            llvm::SmallVector<mlir::Value, 2> shapeDims;
            for (unsigned d = 0; d < leafDims.size(); ++d) {
              auto cd = sb.create<mlir::arith::ConstantOp>(sloc, idxTy, sb.getIndexAttr(rank + d));
              auto bd = sb.create<fir::BoxDimsOp>(sloc, idxTy, idxTy, idxTy, loadedComp.getResult(), cd);
              indices.push_back(c1);
              indices.push_back(bd.getResult(1));  // extent
              indices.push_back(c1);
              trip.push_back(true);
              shapeDims.push_back(bd.getResult(1));
            }
            auto shp = sb.create<fir::ShapeOp>(sloc, mlir::ValueRange{shapeDims});
            // Section box result must match the original loaded pointer box so
            // downstream ``designate``s / aliases keep their type.
            auto sectionDg = sb.create<hlfir::DesignateOp>(
                sloc, ldBox.getResult().getType(), loadedComp.getResult(), mlir::StringAttr{}, mlir::Value{},
                mlir::ValueRange{indices}, sb.getDenseBoolArrayAttr(trip), mlir::ValueRange{}, mlir::BoolAttr{}, shp,
                mlir::ValueRange{}, fir::FortranVariableFlagsAttr{});
            ldBox.getResult().replaceAllUsesWith(sectionDg.getResult());
            deadOps.insert(ldBox);
          }
          deadOps.insert(site.innerDg);
          if (site.elemDg) deadOps.insert(site.elemDg);
          changed = true;
        }
      }
    }
    if (!changed) return false;
    for (auto* op : deadOps)
      if (op->use_empty()) op->erase();
    bool progress = true;
    while (progress) {
      progress = false;
      llvm::SmallVector<mlir::Operation*, 16> nowDead;
      func.walk([&](mlir::Operation* op) {
        if (mlir::isa<hlfir::DesignateOp, fir::ConvertOp, fir::LoadOp>(op) && op->use_empty()) nowDead.push_back(op);
      });
      for (auto* op : nowDead) {
        op->erase();
        progress = true;
      }
    }
    return true;
  }

  /// (A.1) Diagnose LOCAL-rooted instances of the alloc-array-of-records-
  /// with-scalar-inner-members pattern.  When the access chain ends at a
  /// LOCAL ``hlfir.declare`` of a struct (instead of a function-argument
  /// declare), :func:`splitMultiDimAoRScalarMembers` deliberately bails:
  /// the dummy-case rewrite synthesises per-inner-member function-argument
  /// companions and relies on the caller-side bindings layer to marshal
  /// strided views from the original AoR descriptor; the local case has
  /// no such caller hook and ALSO needs the synthesised
  /// ``_FortranAAllocatable{SetBounds,Allocate,Deallocate}`` runtime
  /// calls rewritten to drive per-member companion allocations (different
  /// element types -> different strides, no shared heap pointer).
  ///
  /// When the bridge encounters this shape today, the downstream emitter
  /// hits an unresolvable designate chain and surfaces as an opaque
  /// ``KeyError`` Python-side.  This walker preempts that with a clear
  /// MLIR diagnostic identifying the exact root + path + inner-member
  /// triple, plus a TODO marker pointing at the fix path.
  ///
  /// TODO[alloc-array-of-records LOCAL case]: extend
  /// :func:`splitMultiDimAoRScalarMembers` to (a) accept LOCAL declares
  /// as chain roots, (b) synthesise per-inner-member local
  /// ``fir.alloca`` companions of the right shape, and (c) rewrite the
  /// ``_FortranAAllocatableSetBounds`` + ``_FortranAAllocatableAllocate``
  /// + ``_FortranAAllocatableDeallocate`` call sequences to drive each
  /// companion's allocation independently.  See the project memory
  /// ``project_alloc_array_of_records_scalar_inner.md``.
  void diagnoseLocalAoRScalarInnerMembers(mlir::func::FuncOp func) {
    auto& block = func.front();
    // Function-arg declares -- the dummy case the existing pass handles.
    // Any declare NOT in this set, but rooted at a ``fir.alloca``, is a
    // local-case candidate.
    llvm::DenseSet<mlir::Value> argDeclResults;
    for (unsigned i = 0; i < block.getNumArguments(); ++i)
      for (auto* u : block.getArgument(i).getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) argDeclResults.insert(d.getResult(0));

    llvm::SmallVector<hlfir::DesignateOp, 4> hits;
    func.walk([&](hlfir::DesignateOp innerDg) {
      // Inner-member designate: has component, no subscripts.
      auto innerComp = innerDg.getComponentAttr();
      if (!innerComp) return;
      if (!innerDg.getIndices().empty()) return;
      // Memref must be an element designate.
      auto elemDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(innerDg.getMemref().getDefiningOp());
      if (!elemDg) return;
      if (elemDg.getComponentAttr()) return;
      if (elemDg.getIndices().empty()) return;
      // Element designate's memref must be a loaded box.
      auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(elemDg.getMemref().getDefiningOp());
      if (!ld) return;
      // Walk the member-chain to its root declare.
      mlir::Value v = ld.getMemref();
      while (auto md = mlir::dyn_cast_or_null<hlfir::DesignateOp>(v.getDefiningOp())) {
        if (!md.getComponentAttr()) break;
        v = md.getMemref();
      }
      if (argDeclResults.contains(v)) return;  // already handled by dummy case.
      // LOCAL-case candidate: the root is a hlfir.declare whose memref is
      // a ``fir.alloca`` (we deliberately ignore ``fir.allocmem`` here --
      // that lives in a different lane).
      auto rootDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(v.getDefiningOp());
      if (!rootDecl) return;
      if (!mlir::isa_and_nonnull<fir::AllocaOp>(rootDecl.getMemref().getDefiningOp())) return;
      // Verify the element record type holds only scalar leaves -- mirrors
      // the dummy-case scalarOnly check so we only fire on the genuine
      // ``t_tangent_vectors{v1: f64, v2: f64}``-style pattern.
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(elemDg.getResult().getType());
      if (!refTy) return;
      auto elemRec = mlir::dyn_cast<fir::RecordType>(refTy.getEleTy());
      if (!elemRec) return;
      for (auto& p : elemRec.getTypeList()) {
        mlir::Type mt = p.second;
        if (mlir::isa<fir::SequenceType, fir::BoxType, fir::ReferenceType, fir::HeapType, fir::PointerType,
                      fir::RecordType>(mt))
          return;
      }
      hits.push_back(innerDg);
    });

    if (hits.empty()) return;

    // Emit one diagnostic per (root, path, inner_member) triple at the
    // first hit and signal pass failure.  We don't try to enumerate
    // every site -- one location is enough for the user to find the
    // pattern.
    auto first = hits.front();
    first.emitError() << "hlfir-flatten-structs: LOCAL alloc-array-of-records with "
                      << "SCALAR inner members not yet supported.  "
                      << "Detected on inner-member designate of ``" << first.getComponentAttr().getValue()
                      << "``.  The dummy-case rewrite "
                      << "(``splitMultiDimAoRScalarMembers``) synthesises per-inner-"
                      << "member function-argument companions and relies on the "
                      << "caller-side bindings to marshal strided views; the local "
                      << "case additionally requires rewriting the synthesised "
                      << "``_FortranAAllocatable{SetBounds,Allocate,Deallocate}`` "
                      << "runtime calls to drive per-member companion allocations "
                      << "(different element types -> different strides, no shared "
                      << "heap).  "
                      << "TODO[project_alloc_array_of_records_scalar_inner]: extend "
                      << "splitMultiDimAoRScalarMembers to accept local-declare "
                      << "roots and rewrite the ``_FortranAAllocatable*`` runtime "
                      << "calls per-companion.  "
                      << "Workaround: hoist the local struct + its allocate/deallocate "
                      << "out to a caller (becomes the dummy case, which IS supported), "
                      << "or split the AoR field out of the struct into a top-level "
                      << "allocatable of the inner-member scalar type.";
    signalPassFailure();
  }

  /// (B) Double-buffer split.  A function-argument-rooted alloc / pointer
  /// array-of-records accessed only as ``<chain>(<idx>)`` for stable index
  /// symbols (ICON time-level buffering: ``prog(nnow)`` / ``prog(nnew)`` /
  /// ...) is split into one record-element dummy per (chain, index-symbol)
  /// pair (``<base>[_<member path>]_<sym>``), so the existing scalar-struct
  /// flatten in ``planAndReplaceStructArgs`` handles each element directly --
  /// no runtime-indexed companion.
  ///
  /// **Design: the pattern is purely structural -- no caller-side hint
  /// declares "this is a double buffer."**  The split fires when ALL of:
  ///
  ///   1. ``hlfir.designate %X(%idx)`` is a 1-D element-access designate
  ///      (single subscript, no component).
  ///   2. ``%X`` traces back through one ``fir.load`` of a box plus a
  ///      chain (possibly empty) of ``hlfir.designate{"<member>"}`` ops
  ///      to a function-argument ``hlfir.declare``.  The chain length
  ///      is unbounded; each hop's name joins the companion's prefix.
  ///   3. The terminal AoR-member type is
  ///      ``box<heap<array<? x record>>>`` (allocatable) or
  ///      ``box<ptr<array<? x record>>>`` (pointer) -- checked by
  ///      :cpp:func:`allocOrPtrArrayOfRecordsMember`.
  ///   4. The single index ``%idx`` traces (via ``traceToDecl``) to a
  ///      stable named symbol (any declared integer -- ``nnow``, ``nnew``,
  ///      ``nvar``, ``jg``, etc.).  Computed-index sites bail the entire
  ///      function back.
  ///
  /// "Double buffer" is the canonical ICON pattern (two stable symbols
  /// ``nnow`` / ``nnew``), but the same code handles single-buffer,
  /// triple-buffer, pointer-spine AoR, nested-struct chains, and the
  /// direct-AoR-dummy (``type(t), allocatable :: s(:)``) uniformly.  The
  /// number of resulting per-symbol dummies = the number of distinct
  /// symbols observed across all access sites for that (root, chain)
  /// pair.
  ///
  /// **Caller contract:** at every call site, the caller binds each
  /// per-symbol dummy to the array element corresponding to that
  /// symbol's runtime value (time-level rotation stays in the driver).
  /// The bridge does not encode the rotation -- it splits the IR into
  /// per-symbol lanes and leaves the lane-to-element mapping to the
  /// bindings layer.
  ///
  /// The binding resolves ``<chain>(nnow)`` /
  /// ``(nnew)`` into the per-symbol dummies at call time (the time-level
  /// rotation stays in the driver).
  ///
  /// Three chain shapes are supported:
  ///
  ///   * Top-level AoR member: ``s%prog(idx)`` (one member hop).
  ///   * Nested-struct AoR member: ``s%inner%prog(idx)`` (multiple
  ///     plain-struct hops above the AoR; the joined path becomes part
  ///     of the companion name).
  ///   * Direct-AoR dummy: ``s(idx)`` where ``s`` is itself the alloc /
  ///     pointer array-of-records (empty member path; the companion
  ///     name is ``<base>_<sym>``).
  ///
  /// Bails (returns false, leaves the function untouched) on any element
  /// access whose index doesn't trace to a single declared symbol -- the
  /// member is left for the generic array-of-records path or a downstream
  /// error.
  ///
  /// :param func: the function whose AoR-rooted dummies to split.
  /// :returns: true if a per-symbol dummy was inserted.
  bool splitDoubleBufferMembers(mlir::func::FuncOp func) {
    auto& block = func.front();
    auto* ctx = func.getContext();

    // Map: ``argDecl.getResult(0)`` -> demangled base name + argDecl.  Walked
    // back-edges from element-designate sites must terminate at one of these
    // root declares to be admissible.
    llvm::DenseMap<mlir::Value, std::pair<std::string, hlfir::DeclareOp>> argDecls;
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      hlfir::DeclareOp argDecl;
      for (auto* u : block.getArgument(i).getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
          argDecl = d;
          break;
        }
      if (!argDecl) continue;
      argDecls.try_emplace(argDecl.getResult(0), demangleVarName(argDecl.getUniqName()), argDecl);
    }
    if (argDecls.empty()) return false;

    // Resolve an access base through inline-alias ``hlfir.declare`` chains back
    // to the block-arg-rooted declare registered in ``argDecls``.
    //
    // ``hlfir-inline-all`` (run BEFORE this split, pipeline builder L146 vs
    // L238) re-declares a callee dummy over the caller-side declare result:
    // ``%alias = hlfir.declare %root#0 {...callee...}`` -- and can nest
    // (``%deep = hlfir.declare %alias#0``).  The back-walk above stops at the
    // outermost such alias, which is ABSENT from ``argDecls`` (built only from
    // the block-arg declares below), so an inlined-callee ``p%prog(idx)%m``
    // access would miss the split and leak a rank-conflated flat companion.
    // Follow the ``memref`` operand through the alias declares until the chain
    // reaches the block-arg declare -- canonicalised onto its result #0, the
    // ``argDecls`` key, regardless of which result an assumed-shape alias
    // re-declared -- or a genuine non-alias root (returns a null Value, so the
    // caller keeps the existing skip).  Mirrors the declare-over-declare hop in
    // ``trace_utils.cpp::sectionBaseReachesComponent`` and the forward
    // inline-alias follow in ``planAndReplaceStructArgs::hasComponentReachable``
    // (same file); keys purely on structure (a declare aliasing a block arg),
    // never on any variable name.
    auto resolveInlineAlias = [&argDecls](mlir::Value v) -> mlir::Value {
      for (int hop = 0; hop < limits::kAliasMemrefWalkDepth && v; ++hop) {
        if (argDecls.count(v)) return v;
        auto dc = mlir::dyn_cast_or_null<hlfir::DeclareOp>(v.getDefiningOp());
        if (!dc) break;  // genuine non-alias root
        // A block-arg-rooted declare re-boxes an entry-block argument; its
        // result #0 is the ``argDecls`` key.
        if (mlir::isa<mlir::BlockArgument>(dc.getMemref()) && argDecls.count(dc.getResult(0))) return dc.getResult(0);
        v = dc.getMemref();
      }
      return {};
    };

    // Key: (root, access-order member path, index-symbol name).
    struct Key {
      void* root;
      std::vector<std::string> path;
      std::string sym;
      bool operator<(const Key& o) const {
        if (root != o.root) return root < o.root;
        if (path != o.path) return path < o.path;
        return sym < o.sym;
      }
    };
    std::map<Key, llvm::SmallVector<hlfir::DesignateOp, 4>> sites;
    llvm::SmallPtrSet<mlir::Operation*, 32> deadOps;
    // Base address (declared-variable memref) of each buffer-index symbol, for
    // the in-kernel-mutation reject below.
    std::map<std::string, mlir::Value> symAddr;
    bool bail = false;

    func.walk([&](hlfir::DesignateOp elemDg) {
      if (bail) return;
      if (elemDg.getComponentAttr()) return;        // member access, not subscript
      if (elemDg.getIndices().size() != 1) return;  // 1-D AoR only
      // Walk back from the loaded box through zero or more
      // ``hlfir.designate{"<member>"}`` hops to a function-arg declare.
      auto* defOp = elemDg.getMemref().getDefiningOp();
      auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(defOp);
      if (!ld) return;  // not a box-load -> not an alloc / ptr AoR access
      llvm::SmallVector<std::string, 4> walkedPath;
      llvm::SmallPtrSet<mlir::Operation*, 8> chainDead;
      chainDead.insert(ld);
      mlir::Value v = ld.getMemref();
      while (true) {
        auto* d = v.getDefiningOp();
        auto md = mlir::dyn_cast_or_null<hlfir::DesignateOp>(d);
        if (!md) break;
        auto comp = md.getComponentAttr();
        if (!comp) break;
        walkedPath.push_back(comp.getValue().str());
        chainDead.insert(md);
        v = md.getMemref();
      }
      // Follow inline-alias declare chains (hlfir-inline-all) back to the
      // block-arg declare, so an inlined-callee ``p%prog(idx)%m`` access binds
      // to the SAME per-index companion as the entry's own block-arg-rooted
      // access instead of leaking a rank-conflated flat companion.
      v = resolveInlineAlias(v);
      if (!v) return;  // chain doesn't terminate at a func arg
      auto it = argDecls.find(v);
      if (it == argDecls.end()) return;  // chain doesn't terminate at a func arg
      // The element type at the AoR access must be a record so the new
      // dummy below has a meaningful elemRec to typestamp.
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(elemDg.getResult().getType());
      if (!refTy || !mlir::isa<fir::RecordType>(refTy.getEleTy())) return;
      std::string sym = traceToDecl(elemDg.getIndices()[0]);
      if (sym.empty()) {
        bail = true;
        return;
      }
      // Record the index symbol's declared base address (peel the subscript
      // i32->i64 convert, the load, and any element/component designates) so
      // the reject below can see if it is written inside the kernel.
      if (!symAddr.count(sym)) {
        mlir::Value idxVal = elemDg.getIndices()[0];
        while (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(idxVal.getDefiningOp())) idxVal = cv.getValue();
        if (auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(idxVal.getDefiningOp())) {
          mlir::Value m = ld.getMemref();
          while (auto dg2 = mlir::dyn_cast_or_null<hlfir::DesignateOp>(m.getDefiningOp())) m = dg2.getMemref();
          symAddr[sym] = m;
        }
      }
      Key key;
      key.root = v.getAsOpaquePointer();
      key.path.assign(walkedPath.rbegin(), walkedPath.rend());
      key.sym = std::move(sym);
      sites[std::move(key)].push_back(elemDg);
      deadOps.insert(chainDead.begin(), chainDead.end());
    });
    if (bail || sites.empty()) return false;

    // Multi-buffer-toggle gate: a double-buffer pattern requires
    // MULTIPLE distinct stable index symbols on the SAME
    // ``(root, member_path)`` -- e.g. ICON dycore's
    // ``prog(nnow) % w`` + ``prog(nnew) % w`` (two distinct symbols
    // ``nnow`` and ``nnew`` toggling between physical buffers).  A
    // single distinct symbol per ``(root, path)`` is just a regular
    // AoR access through one runtime index; splitting it mints a
    // false-positive per-symbol companion (QE's
    // ``tabxx(ia) % box(ir)`` -> ``arr_ia_box`` instead of
    // ``arr_box``).  Count distinct symbols per ``(root, path)`` and
    // skip sites that don't meet the >=2 threshold; those fall
    // through to the regular AoR flatten path in
    // ``planAndReplaceStructArgs``.
    std::map<std::pair<void*, std::vector<std::string>>, std::set<std::string>> symsPerPath;
    for (auto& kv : sites) {
      symsPerPath[{kv.first.root, kv.first.path}].insert(kv.first.sym);
    }

    // In-kernel time-level-swap REJECT.  The split binds each per-symbol lane
    // to one physical buffer element ONCE at call time and leaves the
    // time-level rotation to the driver (ICON: ``CALL swap(nnow, nnew)`` lives
    // in mo_nh_stepping, OUTSIDE solve_nonhydro).  If a toggle symbol is
    // instead REASSIGNED inside the kernel -- an in-kernel swap/rotation -- the
    // static lanes cannot be re-pointed mid-kernel, so the split would silently
    // miscompile.  Detect a write (``fir.store`` / ``hlfir.assign`` to the
    // symbol's address, directly or through an element/component designate) on
    // any toggling symbol and fail loudly instead.
    std::function<bool(mlir::Value)> addressIsStored = [&](mlir::Value addr) -> bool {
      if (!addr) return false;
      for (auto* u : addr.getUsers()) {
        if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) {
          if (st.getMemref() == addr) return true;
        } else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(u)) {
          if (as.getLhs() == addr) return true;
        } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
          if (dg.getMemref() == addr && addressIsStored(dg.getResult())) return true;
        }
      }
      return false;
    };
    for (auto& [path, syms] : symsPerPath) {
      if (syms.size() < 2) continue;  // not a double buffer -- regular AoR path
      for (auto& s : syms) {
        auto it = symAddr.find(s);
        if (it != symAddr.end() && addressIsStored(it->second)) {
          func.emitError("double-buffer time-level index `" + s +
                         "` is reassigned inside the kernel (an in-kernel swap "
                         "/ time-level "
                         "rotation): the static per-symbol buffer split binds "
                         "each lane to "
                         "one buffer at call time and cannot re-point it "
                         "mid-kernel.  Keep "
                         "the rotation (e.g. `swap(nnow, nnew)`) in the "
                         "driver, outside the "
                         "extracted kernel.");
          signalPassFailure();
          return false;
        }
      }
    }

    bool changed = false;
    for (auto& kv : sites) {
      auto root = mlir::Value::getFromOpaquePointer(kv.first.root);
      auto rootIt = argDecls.find(root);
      if (rootIt == argDecls.end()) continue;
      // Apply the multi-buffer-toggle gate.
      auto& symSet = symsPerPath[{kv.first.root, kv.first.path}];
      if (symSet.size() < 2) {
        // Single-symbol access on this ``(root, path)`` -- not a
        // double-buffer pattern.  Leave the chain intact for the
        // regular AoR flatten.
        continue;
      }
      auto& [demangledBase, argDecl] = rootIt->second;
      auto refTy = mlir::cast<fir::ReferenceType>(kv.second[0].getResult().getType());
      auto elemRec = mlir::cast<fir::RecordType>(refTy.getEleTy());
      auto refElem = fir::ReferenceType::get(elemRec);

      mlir::OpBuilder b(argDecl);
      b.setInsertionPointAfter(argDecl);
      unsigned newIdx = block.getNumArguments();
      block.insertArgument(newIdx, refElem, argDecl.getLoc());
      auto newArg = block.getArgument(newIdx);

      std::string name = demangledBase;
      for (auto& p : kv.first.path) name += "_" + p;
      name += "_" + kv.first.sym;

      // The real Fortran source this per-symbol lane stands for:
      // ``<base>%<path...>(<sym>)`` (e.g. ``p%prog(nnow)``).  The flatten-plan
      // recorder reads this off the declare so the binding aliases the actual
      // storage instead of the synthetic dummy name (which is not a real
      // Fortran-side variable).
      // Render the real Fortran source INDEX for the ``<base>(<idx>)`` caller
      // path.  The lane ``sym`` (``traceToDecl`` of the index) is the declared
      // NAME -- ``nnow`` for a scalar buffer symbol, or the array base ``nold``
      // for an array-element buffer index (ICON ``INTEGER :: nold(10)`` accessed
      // as ``p_prog(nold(1))``).  For the caller alias the index must keep any
      // constant element subscript, else the binding aliases the whole ``nold``
      // array -- an illegal vector-subscripted pointer component.  A plain
      // scalar symbol falls through to ``traceToDecl`` (renders to itself).
      auto renderBufferIndexExpr = [&](mlir::Value idx) -> std::string {
        mlir::Value v = idx;
        // The AoR index is ``convert*(load(designate(base, const...)))`` -- the
        // loaded integer is widened (i32->i64) by one or more ``fir.convert``s
        // before it subscripts ``p_prog``.  Peel the converts, then the load,
        // to reach the ``hlfir.designate`` that carries the element subscript
        // (mirrors the symAddr walk above).
        while (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(v.getDefiningOp())) v = cv.getValue();
        if (auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(v.getDefiningOp())) v = ld.getMemref();
        if (auto dg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(v.getDefiningOp())) {
          if (!dg.getComponentAttr() && !dg.getIndices().empty()) {
            std::string base = traceToDecl(dg.getMemref());
            std::string subs;
            for (auto s : dg.getIndices()) {
              std::optional<int64_t> c = traceConstInt(s);
              if (!c) {
                base.clear();  // non-constant subscript -- fall back to the bare symbol
                break;
              }
              subs += (subs.empty() ? "" : ", ") + std::to_string(*c);
            }
            if (!base.empty()) return base + "(" + subs + ")";
          }
        }
        return traceToDecl(idx);
      };
      std::string idxExpr = kv.second.empty() ? kv.first.sym : renderBufferIndexExpr(kv.second[0].getIndices()[0]);

      std::string sourceExpr = demangledBase;
      for (size_t i = 0; i < kv.first.path.size(); ++i) {
        sourceExpr += "%" + kv.first.path[i];
        if (i + 1 == kv.first.path.size()) sourceExpr += "(" + idxExpr + ")";
      }

      mlir::NamedAttrList attrs;
      attrs.append("uniq_name", mlir::StringAttr::get(ctx, name));
      attrs.append(declareSegments(b, /*hasShape=*/false));
      auto decl = b.create<hlfir::DeclareOp>(argDecl.getLoc(), mlir::TypeRange{refElem, refElem},
                                             mlir::ValueRange{newArg}, attrs);
      decl->setAttr("hlfir_bridge.dbuf_source", mlir::StringAttr::get(ctx, sourceExpr));

      for (auto elemDg : kv.second) elemDg.getResult().replaceAllUsesWith(decl.getResult(0));
      for (auto elemDg : kv.second) elemDg.erase();
      changed = true;
    }
    // Drop the now-dead member designates + box loads so the original
    // AoR-rooted dummy can be erased later (no lingering reference
    // keeping it alive).
    for (auto* op : deadOps)
      if (op->use_empty()) op->erase();
    return changed;
  }

  /// Returns true if any struct-typed dummy argument was rewritten.
  bool planAndReplaceStructArgs(mlir::func::FuncOp func) {
    auto& block = func.front();

    struct Plan {
      hlfir::DeclareOp argDecl;
      fir::RecordType rec;
      bool jagged = false;
      mlir::Type jaggedEleTy;
      llvm::SmallVector<int64_t, 4> jaggedExtents;
      bool outerIsArray = false;
      llvm::SmallVector<int64_t, 4> outerShape;
      // Members that must take the Phase 5c-B AoS+allocatable
      // path (cap+data block-arg pair, runtime inner extent).
      llvm::SmallVector<std::string, 2> aosAllocMembers;
      // Nested branch: when ``rec`` has any member that's itself
      // a record, ``allMembersFlattenable`` returns false and the
      // single-level flat path bails.  Instead, walk every leaf
      // path and replace the single struct dummy with one block
      // arg per leaf.  Static-shape leaves only at first cut.
      bool nested = false;
      llvm::SmallVector<FlatLeaf, 8> leaves;
    };

    // Keep plans sorted by ORIGINAL argument index.
    llvm::SmallVector<std::pair<unsigned, Plan>, 4> plans;
    for (unsigned i = 0, n = block.getNumArguments(); i < n; ++i) {
      auto arg = block.getArgument(i);
      hlfir::DeclareOp argDecl;
      for (auto* u : arg.getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
          argDecl = d;
          break;
        }
      if (!argDecl) continue;

      bool outerIsArray = false;
      llvm::SmallVector<int64_t, 4> outerShape;
      auto rec = peelToRecord(argDecl.getResult(0).getType(), outerIsArray, outerShape);
      if (!rec) continue;
      // AoS dummy args: dynamic-extent outer dims (assumed-shape
      // ``arr(n)``) are now supported -- ``replaceStructArg``'s
      // companion declare gets a box-wrapped result type when the
      // concat'd pointee has any unknown dim (descriptor carries
      // the extent at runtime), and the bindings layer's
      // assumed-shape marshalling picks up the actual extent.
      // Previously a hard bail.
      //
      // Guard: skip when ``splitDoubleBufferMembers`` (Step 0.5)
      // has already consumed this dummy by rewriting every component
      // designate user to a fresh per-buffer-index dummy.  Detect by
      // walking through the typical access shapes -- direct designate
      // users on argDecl, designate on a LOAD of the argDecl (the
      // pointer / allocatable dummy shape: ``fir.load %arg`` then
      // designate on the boxed value), and one level of nested
      // designate (element-then-component AoR chains) -- looking for
      // any user with a component attribute.  Without this guard
      // ``test_dbuf_split_direct_aor_dummy`` regressed because the
      // regular flatten minted a conflicting ``s_<member>`` companion
      // alongside the split's ``s_<sym>_<member>``.  Without the
      // LOAD-aware extension, pointer / allocatable dummies (QE L4
      // ``arr(ia) % box(ir)`` shape) would falsely be skipped because
      // their access path goes through a ``fir.load`` before any
      // designate user.
      if (outerIsArray) {
        std::function<bool(mlir::Value)> hasComponentReachable = [&](mlir::Value v) -> bool {
          for (auto* u : v.getUsers()) {
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
              if (dg.getComponentAttr()) return true;
              // Element designate -- recurse on its result for
              // the inner component designate.
              if (hasComponentReachable(dg.getResult())) return true;
            }
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) {
              // Pointer / allocatable dummy: load the box, then
              // designate on the boxed value.
              if (hasComponentReachable(ld.getResult())) return true;
            }
            if (auto dcl = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
              // Inlined-callee alias declare (``%alias = hlfir.declare
              // %arg#N ...``): after ``hlfir-inline-all`` the callee's
              // dummy reads designate over this alias, not the block arg
              // directly.  Follow it so an AoS struct dummy whose only
              // component accesses come through an inlined alias (e.g.
              // ICON-O coriolis' ``vn_dual(i,j,k)%x`` inside the inlined
              // ``rot_vertex_ocean_3d``) is still recognised as
              // component-accessed and flattened -- otherwise it is
              // silently skipped and the access leaks as an unregistered
              // flat-companion name (``p_vn_dual_x``) at the libcall.
              if (hasComponentReachable(dcl.getResult(0))) return true;
            }
          }
          return false;
        };
        // Seed from BOTH declare results: a static-shape inlined alias
        // re-declares result #0 (``declare %arg#0``), but a dynamic-extent
        // dummy's alias re-declares the raw memref result #1 with a fresh box
        // (``declare %arg#1(%shape)``).  Check both so the dynamic case is
        // not missed.
        if (!hasComponentReachable(argDecl.getResult(0)) && !hasComponentReachable(argDecl.getResult(1))) continue;
      }

      Plan p;
      p.argDecl = argDecl;
      p.rec = rec;
      p.outerIsArray = outerIsArray;
      p.outerShape.assign(outerShape.begin(), outerShape.end());
      if (!outerIsArray && isJaggedScalarStruct(rec, p.jaggedEleTy, p.jaggedExtents))
        p.jagged = true;
      else if (!allMembersFlattenable(rec)) {
        // Nested branch: the struct has at least one record-
        // typed member.  Walk every leaf path; if every leaf is
        // a flat type with static extents we can replace the
        // single struct dummy with one block arg per leaf.
        // Outer-array nested (``type(t)::s(N)`` where ``t`` is
        // nested): ``collectFlatLeaves`` already prepends ``outerDims``
        // onto each leaf's intrinsic type (line ~555 docs), so the
        // produced ``FlatLeaf::leafTy`` carries the outer extent and
        // ``replaceStructArgNested`` can mint block args of the
        // concat'd shape.  Without this, the user-flagged multi-level
        // AoS hierarchy (``arr(i) % inner % x(j)``) stays
        // unflattened.
        llvm::SmallVector<std::string, 4> prefix;
        llvm::SmallVector<FlatLeaf, 8> leaves;
        llvm::SmallVector<int64_t, 4> outerDimsForCollect;
        if (outerIsArray)
          for (auto d : outerShape) outerDimsForCollect.push_back(d);
        // ``partial=true``: skip unsupported members (CHARACTER,
        // alloc-array-of- records, ...) and flatten the rest, rather than
        // abandoning the whole struct on the first one.  Unaccessed skipped
        // members cost nothing; an accessed skipped member keeps the struct
        // dummy alive (no companion) and is filtered/handled downstream. When
        // ``outerIsArray``, use the outerDims-taking overload so leaves carry
        // the outer extent at the front of their flat shape; otherwise the
        // simpler overload covers the scalar-struct case.  ``partial`` only
        // applies to the simpler overload (the outerDims one doesn't support
        // partial-skip yet -- nested AoR with an unsupported
        // member would simply not flatten in this path).
        bool collected;
        if (outerIsArray) {
          collected = collectFlatLeaves(rec, prefix, outerDimsForCollect, leaves, /*depth=*/0);
        } else {
          collected = collectFlatLeaves(rec, prefix, leaves, /*depth=*/0,
                                        /*partial=*/true);
        }
        if (!collected) continue;
        p.nested = true;
        p.leaves = std::move(leaves);
      }
      // Phase 5b: dummy struct args with allocatable members
      // flatten the same way as local instances  --  each
      // allocatable member becomes a flat top-level allocatable
      // companion (``<base>_<member>``) and the bindings layer
      // marshals it across the call boundary.
      //
      // Phase 5c-B (true SDFG-boundary): AoS + allocatable
      // members get the padding-to-max contract.  Each such
      // member becomes a 2D buffer ``A_<member>(N, cap)`` plus
      // a runtime cap symbol; the bindings layer computes the
      // cap by max-ing per-instance ``size()`` values, packs
      // each allocated row's live region into the buffer, and
      // unpacks back on intent(out)/(inout).
      if (outerIsArray) {
        for (auto& pair : rec.getTypeList())
          if (isAllocatableArrayMember(pair.second)) p.aosAllocMembers.push_back(pair.first);
      }
      plans.push_back({i, p});
    }

    if (plans.empty()) return false;

    // Walk plans in reverse so lower indices aren't invalidated by
    // higher-index erases.  Each replace either mutates the argument
    // list in place (insert-then-erase) or bails out without changes.
    for (auto& entry : llvm::reverse(plans)) {
      auto idx = entry.first;
      auto& p = entry.second;
      if (p.jagged) {
        replaceStructArgJagged(func, idx, p.argDecl, p.rec, p.jaggedEleTy, p.jaggedExtents);
        // Jagged path is not represented in the plan yet.
        continue;
      }
      if (p.nested) {
        // Phase 2 dummy-arg extension: replace the nested struct
        // dummy with one block arg per leaf and record the
        // matching FlattenEntry so the bindings emitter can
        // alias every leaf via its full ``%``-path.
        std::string intentStrNested = extractIntent(p.argDecl.getFortranAttrs());
        recordNestedStructArgEntry(p.argDecl, p.leaves, intentStrNested);
        replaceStructArgNested(func, idx, p.argDecl, p.leaves);
        continue;
      }
      // Record the entry BEFORE the declare is erased.  If
      // ``replaceStructArg`` bails out (dangling users on the
      // old declare), the entry still describes the intended
      // recipe  --  but the SDFG won't carry the flat members so
      // the emitter will just skip it downstream.
      std::string intentStr = extractIntent(p.argDecl.getFortranAttrs());
      llvm::StringSet<> aosAllocSet;
      for (auto& m : p.aosAllocMembers) aosAllocSet.insert(m);
      // Phase 5c-C (NOT supported, raise loudly): an ``intent(out)`` AoS
      // dummy has its allocatable components auto-deallocated on entry
      // (F2003), so the kernel must perform the FIRST ``allocate`` itself.
      // There is then no caller-side data to size the padded companion
      // from -- the binding would compute ``cap = 0 -> 1`` and emit a
      // degenerate buffer that silently truncates the kernel's writes.
      // Fail rather than miscompile; pass the struct as ``intent(inout)``
      // with pre-allocated members instead.
      if (!p.aosAllocMembers.empty() && intentStr == "out") {
        p.argDecl->emitError(
            "AoS-allocatable member on an intent(out) struct dummy needs a "
            "kernel-internal first allocate, which is unsupported: no caller "
            "data to size the padded companion.  Pass the struct as "
            "intent(inout) with pre-allocated members.");
        signalPassFailure();
        continue;
      }
      // Phase 5c-B: emit one aos_alloc=True entry per AoS+
      // allocatable member.  Then emit the regular entry
      // covering the non-allocatable members (skipped via the
      // exclude set).
      for (auto& m : p.aosAllocMembers) recordAosAllocEntry(p.argDecl, p.rec, m, intentStr, p.outerShape);
      recordStructArgEntry(p.argDecl, p.rec, intentStr, p.outerIsArray, p.outerShape, aosAllocSet);
      replaceStructArg(func, idx, p.argDecl, p.rec, p.outerIsArray, p.outerShape, aosAllocSet);
    }
    return true;
  }

  // -------------------------------------------------------------------
  // Struct dummy arguments
  // -------------------------------------------------------------------

  void replaceStructArg(mlir::func::FuncOp func, unsigned argIdx, hlfir::DeclareOp argDecl, fir::RecordType rec,
                        bool outerIsArray = false, llvm::ArrayRef<int64_t> outerShape = {},
                        const llvm::StringSet<>& aosAllocMembers = {}) {
    auto& block = func.front();
    auto loc = argDecl.getLoc();
    auto* ctx = func.getContext();
    auto baseName = argDecl.getUniqName().str();
    auto demangledBase = demangleVarName(baseName);

    // Insert new block args right after the old one so the argument order
    // tracks the original member order.  Insertion shifts indices >= pos
    // by 1, so we insert sequentially at argIdx+1, argIdx+2, ...
    llvm::StringMap<mlir::Value> memberBase;
    llvm::StringMap<mlir::Value> aosAllocFlatBase;
    llvm::StringSet<> concatMembers;
    unsigned memberCount = 0;
    for (auto& pair : rec.getTypeList()) {
      auto memName = pair.first;
      auto memTy = pair.second;

      // Phase 5c-B (true SDFG boundary): AoS + allocatable member.
      // Insert two block args  --  the runtime cap (``index``) then a
      // 2D data buffer ``ref<array<N x ?xT>>``.  Build a declare
      // for each, with the cap declare's name = ``cap_<base>_<m>``
      // so ``traceToDecl`` resolves the inner extent to that
      // symbol on the SDFG signature.  ``collapseAosAllocReads``
      // afterwards rewrites every ``fir.load + designate``
      // chain on the original member box into a direct 2-index
      // designate over the new flat declare.
      if (aosAllocMembers.count(memName)) {
        auto box = mlir::cast<fir::BoxType>(memTy);
        mlir::Type eleTy;
        if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
          eleTy = mlir::cast<fir::SequenceType>(heap.getEleTy()).getEleTy();
        else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
          eleTy = mlir::cast<fir::SequenceType>(ptr.getEleTy()).getEleTy();
        else
          continue;

        // Pointee shape: outer extents (static) x {?} (cap, runtime).
        llvm::SmallVector<int64_t, 4> exts(outerShape.begin(), outerShape.end());
        exts.push_back(fir::SequenceType::getUnknownExtent());
        auto pointee = fir::SequenceType::get(exts, eleTy);
        auto refTy = fir::ReferenceType::get(pointee);
        // HLFIR's variable-form result for an array with any
        // dynamic extent must be a ``!fir.box``; flang itself
        // emits the same shape for explicit-shape dummies whose
        // last extent comes from a runtime ``n`` (see how
        // ``real(8), intent(inout) :: x(3, n)`` lowers  --  the
        // declare returns ``(!fir.box<!fir.array<3x?xf64>>,
        // !fir.ref<!fir.array<3x?xf64>>)``).  Match that pair so
        // the verifier accepts the synthesised declare.
        auto boxTy = fir::BoxType::get(pointee);

        auto idxTy = mlir::IndexType::get(ctx);
        auto idxRefTy = fir::ReferenceType::get(idxTy);

        std::string flatName = demangledBase + "_" + memName;
        std::string capName = "cap_" + flatName;

        // Insert cap arg (block.getArgument as an ``index``
        // value passed by reference so the bindings layer can
        // populate it from the wrapper).
        unsigned capArgIdx = argIdx + 1 + memberCount;
        block.insertArgument(capArgIdx, idxRefTy, loc);
        auto capArg = block.getArgument(capArgIdx);
        ++memberCount;

        unsigned dataArgIdx = argIdx + 1 + memberCount;
        block.insertArgument(dataArgIdx, refTy, loc);
        auto dataArg = block.getArgument(dataArgIdx);
        ++memberCount;

        mlir::OpBuilder b(&block, std::next(argDecl->getIterator()));
        b.setInsertionPoint(argDecl);

        // Cap declare: scalar ``ref<index>`` with uniq_name
        // ``cap_<base>_<member>``.  The bridge's
        // ``traceToDecl`` will resolve the data declare's
        // shape extent to this name.
        mlir::NamedAttrList capAttrs;
        capAttrs.append("uniq_name", mlir::StringAttr::get(ctx, capName));
        capAttrs.append(declareSegments(b, /*hasShape=*/false));
        auto capDecl =
            b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{idxRefTy, idxRefTy}, mlir::ValueRange{capArg}, capAttrs);

        // Load the cap value to use it in the data declare's
        // shape op.
        auto capVal = b.create<fir::LoadOp>(loc, capDecl.getResult(0)).getResult();

        // Build the shape op: outer (static) + cap (runtime).
        llvm::SmallVector<mlir::Value, 4> dims;
        for (auto e : outerShape)
          dims.push_back(b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(e)).getResult());
        dims.push_back(capVal);
        auto shapeTy = fir::ShapeType::get(ctx, dims.size());
        auto shape = b.create<fir::ShapeOp>(loc, shapeTy, dims).getResult();

        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, flatName));
        attrs.append(declareSegments(b, /*hasShape=*/true));

        auto newDecl =
            b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{boxTy, refTy}, mlir::ValueRange{dataArg, shape}, attrs);

        aosAllocFlatBase[memName] = newDecl.getResult(0);
        continue;
      }

      auto pointee = companionPointee(outerIsArray, outerShape, memTy);
      if (!pointee) continue;  // defensive; caller already checked
      auto refTy = fir::ReferenceType::get(pointee);

      // Dynamic-extent companion (assumed-shape outer ``arr(n)``):
      // ``hlfir.declare`` needs the variable-form ``(box, ref)``
      // result pair when any dim is unknown (Flang itself emits
      // the same shape for explicit-shape dummies whose extent
      // comes from a runtime ``n``).  Pair this with a
      // ``fir.shape`` operand whose dynamic-dim operands are
      // pulled from a fresh ``index``-typed value loaded from
      // the caller-supplied extent symbol -- the bindings layer
      // populates that symbol at the call boundary.
      auto extentsForShape = staticArrayExtents(pointee);
      bool hasDynExtent = mlir::isa<fir::SequenceType>(pointee) && extentsForShape.empty();

      bool memberIsArray = mlir::isa<fir::SequenceType>(memTy);
      bool concat = outerIsArray && memberIsArray;
      if (concat) concatMembers.insert(memName);

      unsigned newArgIdx = argIdx + 1 + memberCount;
      block.insertArgument(newArgIdx, refTy, loc);
      auto newArg = block.getArgument(newArgIdx);

      mlir::OpBuilder b(&block, std::next(argDecl->getIterator()));
      b.setInsertionPoint(argDecl);

      // Array members need a fir.shape operand for the declare to
      // verify.  For AoS+memberArray (concat), build the concat
      // shape from the outer dims followed by the member's static
      // extents.  For dynamic-extent companions, emit a runtime
      // ``fir.shape`` from a fresh ``index`` value per unknown dim
      // -- the bindings layer populates the extent symbol.
      mlir::Value shape;
      if (!hasDynExtent) {
        shape = emitStaticShape(b, loc, extentsForShape);
      } else {
        // Build a runtime shape op.  Each unknown dim becomes a
        // ``fir.alloca index`` + immediate ``fir.load`` -- the
        // bindings emitter populates the alloca at the call site
        // from the assumed-shape descriptor.  Static dims pass
        // through as ``arith.constant``.
        auto seq = mlir::cast<fir::SequenceType>(pointee);
        auto idxTy = mlir::IndexType::get(ctx);
        llvm::SmallVector<mlir::Value, 4> dims;
        for (auto d : seq.getShape()) {
          if (d == fir::SequenceType::getUnknownExtent()) {
            auto al = b.create<fir::AllocaOp>(loc, idxTy);
            auto ld = b.create<fir::LoadOp>(loc, al.getResult());
            dims.push_back(ld.getResult());
          } else {
            dims.push_back(b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(d)).getResult());
          }
        }
        auto shapeTy = fir::ShapeType::get(ctx, dims.size());
        shape = b.create<fir::ShapeOp>(loc, shapeTy, dims).getResult();
      }

      llvm::SmallVector<mlir::Value, 2> operands;
      operands.push_back(newArg);
      if (shape) operands.push_back(shape);

      mlir::NamedAttrList attrs;
      attrs.append("uniq_name", mlir::StringAttr::get(ctx, baseName + "_" + memName));
      // Pointer / allocatable struct member: the declare carries
      // the shape in its runtime box descriptor, not as a static
      // shape op.  Without the matching Fortran attr,
      // ``extract_vars`` won't peel through the box+ptr/heap
      // wrappers to find the inner SequenceType and the flat
      // companion ends up classified as a scalar of dtype
      // ``!fir.box<!fir.ptr<...>>`` -- yielding KeyError on the
      // synthesised ``<companion>_d<i>`` extent symbol when
      // ``arglist()`` later collects free symbols.
      if (auto box = mlir::dyn_cast<fir::BoxType>(memTy)) {
        if (mlir::isa<fir::PointerType, fir::HeapType>(box.getEleTy())) {
          bool isPointer = mlir::isa<fir::PointerType>(box.getEleTy());
          attrs.append("fortran_attrs",
                       fir::FortranVariableFlagsAttr::get(ctx, isPointer ? fir::FortranVariableFlagsEnum::pointer
                                                                         : fir::FortranVariableFlagsEnum::allocatable));
        }
      }
      attrs.append(declareSegments(b, /*hasShape=*/shape != nullptr));

      // For dynamic-extent companions, the HLFIR variable-form
      // first result is ``box<array<?xT>>`` (descriptor-aware) not
      // ``ref<array<?xT>>``.  Mirror the
      // ``real(8), intent(inout) :: x(3, n)`` lowering at
      // ``replaceStructArg``'s aosAlloc path (line ~3007).
      mlir::Type firstResultTy = refTy;
      if (hasDynExtent) {
        firstResultTy = fir::BoxType::get(pointee);
      }
      auto newDecl =
          b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{firstResultTy, refTy}, mlir::ValueRange(operands), attrs);

      memberBase[memName] = newDecl.getResult(0);
      ++memberCount;
    }

    // Phase 5c-B AoS+allocatable: rewrite every per-instance
    // ``fir.load + hlfir.designate`` chain into a 2-index
    // designate over the new flat declare.  Run BEFORE the plain-
    // member designate sweep so the alloc-member's parent designates
    // (``A(i)``) are still alive.
    for (auto& kv : aosAllocFlatBase) {
      stripReallocOnAosMember(argDecl, kv.first());
      collapseAosAllocReads(argDecl, kv.first(), kv.second);
    }

    // Rewrite designates on the struct declare.  AoS dummies (the
    // outer is an array of struct) carry a concat path: the direct
    // user is an indexed designate (no component) on the outer
    // array, and the component-designate is its child.  The simple
    // walk handles those.  For scalar struct dummies (no AoS), use
    // ``rewriteChainsRootedAt`` so designates rooted at an inlined-
    // callee's alias declare ( ``hlfir.declare %p_int#0 ... uniq_name
    // = "...ptr_int"`` in the body of an inlined ``rot_vertex_ri``
    // call) also fold to the flat companion -- without this, the
    // alias declare keeps the original struct declare alive and the
    // bail at "use_empty" leaves ``p_int`` as a dangling SDFG free
    // symbol later on.
    if (outerIsArray) {
      llvm::SmallVector<hlfir::DesignateOp, 16> designates;
      // Collect designates rooted at ``argDecl`` OR any inlined-callee ALIAS
      // declare of it (``%alias = hlfir.declare %argDecl#0 ...``).  After
      // ``hlfir-inline-all`` the callee's reads designate over the alias, not
      // the block arg directly, so without following aliases the AoS-member
      // accesses (``vn_dual(i,j,k)%x`` inside an inlined
      // ``rot_vertex_ocean_3d``) are never rewritten to the flat companion --
      // the access keeps its struct designate and the bridge emits a
      // WHOLE-array memlet on the companion (losing the element indices), which
      // fails downstream (e.g. a libcall over the un-indexed multi-dim
      // companion).  Mirrors the alias-following the scalar-struct
      // ``rewriteChainsRootedAt`` path does. Seed from BOTH declare results
      // (see the gate above): static aliases re-declare result #0,
      // dynamic-extent aliases re-declare the raw memref result #1.
      llvm::SmallVector<mlir::Value, 4> roots{argDecl.getResult(0), argDecl.getResult(1)};
      llvm::SmallVector<hlfir::DeclareOp, 4> aliasDecls;
      llvm::SmallPtrSet<mlir::Operation*, 8> seen;
      for (unsigned ri = 0; ri < roots.size(); ++ri) {
        for (auto* u : roots[ri].getUsers()) {
          if (auto adcl = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
            if (adcl != argDecl && seen.insert(adcl).second) {
              aliasDecls.push_back(adcl);
              roots.push_back(adcl.getResult(0));
            }
            continue;
          }
          auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u);
          if (!dg) continue;
          bool hasComponent = false;
          for (auto nm : {"component_name", "component"})
            if (dg->getAttrOfType<mlir::StringAttr>(nm)) {
              hasComponent = true;
              break;
            }
          if (hasComponent) {
            designates.push_back(dg);
            continue;
          }
          // Pure indexed designate (A(i) on AoS dummy)  --  collect children.
          for (auto* cu : dg.getResult().getUsers())
            if (auto cdg = mlir::dyn_cast<hlfir::DesignateOp>(cu)) designates.push_back(cdg);
        }
      }
      for (auto dg : designates) rewriteDesignate(dg, memberBase, concatMembers);
      // Drop now-dead inlined alias declares so ``argDecl`` becomes erasable
      // (the alias is the only remaining user keeping the block arg live).
      for (auto adcl : aliasDecls)
        if (adcl.getResult(0).use_empty() && adcl.getResult(1).use_empty()) adcl.erase();
    } else {
      rewriteChainsRootedAt(argDecl, memberBase);
    }

    // Erase the old declare; if other ops still reference its results
    // something went sideways and we leave the block arg in place rather
    // than breaking the IR.
    if (!argDecl.getResult(0).use_empty() || !argDecl.getResult(1).use_empty()) return;
    argDecl.erase();
    if (!block.getArgument(argIdx).use_empty()) return;
    block.eraseArgument(argIdx);
  }

  /// Replace a NESTED struct dummy arg with one block arg per flat
  /// leaf.  Mirrors ``replaceStructArg`` for the single-level case
  /// but consumes ``collectFlatLeaves`` output to handle arbitrary
  /// nesting depth.  Static-shape leaves only  --  dynamic-extent or
  /// allocatable leaves are left for the Phase 5b nested follow-up
  /// (the leaf walker bails on those upstream).
  ///
  /// For each leaf with path ``[a, b, c]`` and type ``leafTy``:
  ///   * Insert a block arg of type ``ref<leafTy>`` after the
  ///     original struct dummy.
  ///   * Synthesise an ``hlfir.declare`` with
  ///     ``uniq_name = <base>_a_b_c`` and a static shape operand
  ///     when ``leafTy`` is an array.
  ///
  /// Then walks every chain rooted at the original ``argDecl`` (or
  /// any inlined-callee alias of it) and rewrites it via the shared
  /// ``rewriteChainsRootedAt`` helper.  Erases the old declare and
  /// the original block arg if all uses cleared.
  void replaceStructArgNested(mlir::func::FuncOp func, unsigned argIdx, hlfir::DeclareOp argDecl,
                              llvm::ArrayRef<FlatLeaf> leaves) {
    auto& block = func.front();
    auto loc = argDecl.getLoc();
    auto* ctx = func.getContext();
    auto baseName = argDecl.getUniqName().str();
    auto demangledBase = demangleVarName(baseName);

    llvm::StringMap<mlir::Value> leafBase;
    unsigned leafCount = 0;
    for (auto& leaf : leaves) {
      // Build the joined-path key (matches ``rewriteDesignateChain``'s
      // ``walkDesignateChain`` output).
      std::string joinedKey;
      for (unsigned i = 0; i < leaf.path.size(); ++i) {
        if (i) joinedKey += "_";
        joinedKey += leaf.path[i];
      }
      std::string suffix = joinedKey;

      auto leafTy = leaf.leafTy;
      auto refTy = fir::ReferenceType::get(leafTy);

      unsigned newArgIdx = argIdx + 1 + leafCount;
      block.insertArgument(newArgIdx, refTy, loc);
      auto newArg = block.getArgument(newArgIdx);
      ++leafCount;

      mlir::OpBuilder b(argDecl);

      // Array leaves need a fir.shape operand on the declare;
      // dynamic-extent leaves were filtered upstream by
      // ``collectFlatLeaves``.  Allocatable / pointer leaves
      // (``box<heap<array<?>>>`` / ``box<ptr<array<?>>>``) carry
      // their shape in the descriptor at runtime  --  no explicit
      // shape op, but the Fortran ``allocatable`` / ``pointer``
      // attr must be set so ``extract_vars`` peels through every
      // wrapper to find the inner SequenceType (rank > 0
      // classification).
      mlir::Value leafShape;
      fir::FortranVariableFlagsAttr fortranAttrs;
      if (auto seq = mlir::dyn_cast<fir::SequenceType>(leafTy)) {
        llvm::SmallVector<int64_t, 4> exts;
        for (auto d : seq.getShape()) {
          if (d == fir::SequenceType::getUnknownExtent()) return;
          exts.push_back(d);
        }
        leafShape = emitStaticShape(b, loc, exts);
      } else if (auto box = mlir::dyn_cast<fir::BoxType>(leafTy)) {
        // Allocatable / pointer leaf  --  the box wraps a
        // ``fir.heap`` (allocatable) or ``fir.ptr`` (pointer).
        bool isPointer = mlir::isa<fir::PointerType>(box.getEleTy());
        fortranAttrs = fir::FortranVariableFlagsAttr::get(
            ctx, isPointer ? fir::FortranVariableFlagsEnum::pointer : fir::FortranVariableFlagsEnum::allocatable);
      }

      llvm::SmallVector<mlir::Value, 2> operands{newArg};
      if (leafShape) operands.push_back(leafShape);
      mlir::NamedAttrList attrs;
      attrs.append("uniq_name", mlir::StringAttr::get(ctx, demangledBase + "_" + suffix));
      if (fortranAttrs) attrs.append("fortran_attrs", fortranAttrs);
      attrs.append(declareSegments(b, /*hasShape=*/leafShape != nullptr));

      auto newDecl = b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{refTy, refTy}, mlir::ValueRange(operands), attrs);

      leafBase[joinedKey] = newDecl.getResult(0);
    }

    // Reuse the chain-rewrite machinery from the local-instance
    // path: walks every designate chain rooted at ``argDecl`` (or
    // any inlined-callee alias of it) and folds it down to the
    // matching flat companion.
    rewriteChainsRootedAt(argDecl, leafBase);

    // Erase the original struct declare + block arg if all uses
    // cleared.  If something still references the old declare,
    // leave both in place so the IR stays valid.
    if (!argDecl.getResult(0).use_empty() || !argDecl.getResult(1).use_empty()) return;
    argDecl.erase();
    if (!block.getArgument(argIdx).use_empty()) return;
    block.eraseArgument(argIdx);
  }

  /// Pack a jagged scalar-struct argument (1-D array members of same scalar
  /// type with differing extents) into a single 2-D companion of shape
  /// ``[numMembers x max(extents)]``.  Access to member `m` at index `j`
  /// becomes ``combined(rowIdx(m), j)``  --  an ELLPACK-style padded view.
  void replaceStructArgJagged(mlir::func::FuncOp func, unsigned argIdx, hlfir::DeclareOp argDecl, fir::RecordType rec,
                              mlir::Type eleTy, llvm::ArrayRef<int64_t> extents) {
    auto& block = func.front();
    auto loc = argDecl.getLoc();
    auto* ctx = func.getContext();
    auto baseName = argDecl.getUniqName().str();

    int64_t maxExt = 0;
    for (auto e : extents)
      if (e > maxExt) maxExt = e;
    int64_t rows = extents.size();

    auto combinedTy = fir::SequenceType::get({rows, maxExt}, eleTy);
    auto combinedRef = fir::ReferenceType::get(combinedTy);

    // New block argument right after the old struct arg.
    block.insertArgument(argIdx + 1, combinedRef, loc);
    auto combinedArg = block.getArgument(argIdx + 1);

    mlir::OpBuilder b(argDecl);
    // The packed companion is a static-shape array over a raw address base,
    // so its hlfir.declare must carry a fir.shape operand (the verifier
    // rejects a shapeless array declare); build ``[rows, maxExt]``.
    llvm::SmallVector<int64_t, 2> combinedExtents{rows, maxExt};
    mlir::Value combinedShape = emitStaticShape(b, loc, combinedExtents);
    mlir::NamedAttrList attrs;
    attrs.append("uniq_name", mlir::StringAttr::get(ctx, baseName + "_packed"));
    attrs.append(declareSegments(b, /*hasShape=*/true));
    auto combinedDecl = b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{combinedRef, combinedRef},
                                                   mlir::ValueRange{combinedArg, combinedShape}, attrs);

    // Per-member aliased view into a single row of the combined array.
    // fir.coordinate_of rank-reduces the 2-D combined ref to a 1-D row
    // ref; fir.convert then bridges the row's max-extent type to the
    // member's original extent type so downstream hlfir.designate uses
    // type-check unchanged.
    llvm::StringMap<mlir::Value> memberBase;
    int64_t rowIdx = 0;
    auto idxTy = b.getIndexType();
    for (auto& pair : rec.getTypeList()) {
      auto memName = pair.first;
      auto memSeq = mlir::cast<fir::SequenceType>(pair.second);
      int64_t ext = memSeq.getShape()[0];

      auto rowConst = b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(rowIdx));

      auto rowRefTy = fir::ReferenceType::get(fir::SequenceType::get({maxExt}, eleTy));
      auto rowPtr = b.create<fir::CoordinateOp>(loc, rowRefTy, combinedDecl.getResult(0), mlir::ValueRange{rowConst});

      auto memberRefTy = fir::ReferenceType::get(fir::SequenceType::get({ext}, eleTy));
      auto casted = b.create<fir::ConvertOp>(loc, memberRefTy, rowPtr.getResult());

      // Synthesise a per-member ``hlfir.declare`` so ``traceToDecl``
      // can stop at the member view rather than walking through
      // the ``fir.convert`` + ``fir.coordinate_of`` chain back to
      // the 2-D ``<base>_packed`` companion (which yields a 2-D
      // shape for a 1-D access subset and the memlet-dim validator
      // rejects ``g_packed[0]`` on a 2-D ``g_packed``).  The
      // ``hlfir.declare`` carries an explicit ``fir.shape`` operand
      // so the verifier accepts the static-extent member type.
      auto memberShape = emitStaticShape(b, loc, llvm::ArrayRef<int64_t>{ext});
      mlir::NamedAttrList memberAttrs;
      memberAttrs.append("uniq_name", mlir::StringAttr::get(ctx, baseName + "_" + memName));
      memberAttrs.append(declareSegments(b, /*hasShape=*/true));
      auto memberDecl = b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{memberRefTy, memberRefTy},
                                                   mlir::ValueRange{casted.getResult(), memberShape}, memberAttrs);

      memberBase[memName] = memberDecl.getResult(0);
      ++rowIdx;
    }

    // Rewrite each component-selecting designate to the member's view.
    llvm::SmallVector<hlfir::DesignateOp, 8> designates;
    for (auto* u : argDecl.getResult(0).getUsers())
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) designates.push_back(dg);
    for (auto dg : designates) rewriteDesignate(dg, memberBase);

    if (!argDecl.getResult(0).use_empty() || !argDecl.getResult(1).use_empty()) return;
    argDecl.erase();
    if (!block.getArgument(argIdx).use_empty()) return;
    block.eraseArgument(argIdx);
  }

  // -------------------------------------------------------------------
  // Local struct allocations
  // -------------------------------------------------------------------

  // -------------------------------------------------------------------
  // Allocatable struct member helpers (Phase 5a)
  // -------------------------------------------------------------------
  //
  // When we replace ``s%w`` (allocatable member) with a flat
  // top-level allocatable ``s_w``, the user's ``allocate(s%w(N))``
  // statement still lowers via flang to a ``fir.allocmem`` op whose
  // ``uniq_name`` attribute points at the MEMBER's namespace
  // (``_QMlibEw.alloc``)  --  independent of the enclosing struct's
  // declare scope.  The bridge's ``collectAllocSites`` matches
  // allocate sites by ``<declUniqName>.alloc``, so without renaming,
  // the flat declare's allocate site is invisible and the SDFG
  // ends up with an unbound runtime extent symbol.
  //
  // ``renameMemberAllocmems`` walks the original struct declare's
  // direct designate users with component name == ``memName``, and
  // for each, walks store users.  Any allocmem reaching that store
  // through ``fir.embox`` (the standard allocate lowering shape)
  // gets its ``uniq_name`` rewritten to ``<flatName>.alloc``.
  //
  // Caveats
  // -------
  // * Only the direct designate users of ``decl`` are walked.
  //   Aliases through inlined-callee declares would need the same
  //   alias-following machinery the main rewrite uses; we don't
  //   support cross-call ``allocate(s%w(...))`` yet.  This matches
  //   the AoS/parametric-dim phase boundaries documented above.
  // * Multiple allocate sites for the same member (an allocate +
  //   deallocate + re-allocate cycle, for example) all get the
  //   same flat name.  ``allocAliasName`` in extract_vars.cpp then
  //   mints ``<flat>_alloc1``, ``<flat>_alloc2``, ... per site.
  void renameMemberAllocmems(hlfir::DeclareOp decl, llvm::StringRef memName, llvm::StringRef flatName) {
    auto* ctx = decl.getContext();
    std::string newAlloc = (flatName + ".alloc").str();

    // Bug fix: Phase 5a only walked ``decl``'s direct users.  When
    // the ``allocate(s%w(n))`` call sits inside an internal
    // subprogram, ``hlfir-inline-all`` splices the callee body in
    // and the designate of ``s%w`` ends up rooted at an inlined
    // alias declare (its memref traces back through ``fir.embox``
    // / ``fir.convert`` chains to ``decl``), not at ``decl``
    // itself.  Walk both ``decl`` directly AND every aliasing
    // declare that resolves back to it, mirroring the same
    // alias-following machinery the main ``splitLocal``
    // designate-collector uses.
    llvm::SmallVector<hlfir::DeclareOp, 8> roots{decl};
    llvm::DenseSet<mlir::Operation*> seen{decl.getOperation()};
    std::function<bool(mlir::Value)> resolvesTo = [&](mlir::Value v) -> bool {
      for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
        auto* d = v.getDefiningOp();
        if (!d) return false;
        if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) return outer == decl;
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
          v = cv.getValue();
          continue;
        }
        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
          v = eb.getMemref();
          continue;
        }
        return false;
      }
      return false;
    };
    // Walk all hlfir.declare ops in the enclosing function and
    // collect those that alias ``decl`` (memref chains through
    // ``embox`` / ``convert`` / another ``declare`` back to it).
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      func.walk([&](hlfir::DeclareOp other) {
        if (other == decl) return;
        if (seen.count(other.getOperation())) return;
        if (resolvesTo(other.getMemref())) {
          seen.insert(other.getOperation());
          roots.push_back(other);
        }
      });
    }

    for (auto root : roots) {
      for (auto* u : root.getResult(0).getUsers()) {
        auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u);
        if (!dg) continue;
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = dg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) continue;
        for (auto* du : dg.getResult().getUsers()) {
          auto store = mlir::dyn_cast<fir::StoreOp>(du);
          if (!store) continue;
          // Trace the stored value back to its
          // ``fir.allocmem`` through the standard ``embox``
          // wrapping (and possibly a ``fir.convert`` for
          // box-shape canonicalisation).
          mlir::Value v = store.getValue();
          for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
            auto* d = v.getDefiningOp();
            if (!d) break;
            if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) {
              am->setAttr("uniq_name", mlir::StringAttr::get(ctx, newAlloc));
              break;
            }
            if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
              v = eb.getMemref();
              continue;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
              v = cv.getValue();
              continue;
            }
            break;
          }
        }
      }
    }
  }

  // -------------------------------------------------------------------
  // AoS + allocatable helpers (Phase 5c-A)
  // -------------------------------------------------------------------
  //
  // ``aosAllocUniformConstSize(decl, memName)`` walks every
  // ``fir.allocmem`` op whose target traces back through ``embox`` /
  // ``convert`` to a designate of ``decl(<i>){<memName>}`` and checks
  // that all such allocate sites use the SAME compile-time constant
  // size operand.  Returns the constant on uniform match, ``nullopt``
  // otherwise.
  //
  // The "uniform constant" gate is what makes Phase 5c-A's
  // padding-to-max trivial: every ``A(i)%<member>`` allocates the
  // same M elements, so the flat companion is statically
  // ``A_<member>(N, M)``, the bindings layer doesn't need to compute
  // ``max`` at runtime, and the kernel-internal ``allocate`` sites
  // become semantic no-ops over the pre-existing 2D buffer.
  //
  // Caveats
  // -------
  // * Element-form designate of ``decl`` (``A(i)`` with a SCALAR
  //   index per outer dim) is the only path we walk.  Section-form
  //   designates of the AoS outer (``A(1:N)``) wouldn't match  --
  //   they'd be compiler-generated whole-array assigns, not
  //   per-instance allocates.
  // * Sites whose size operand isn't a constant (e.g.
  //   ``allocate(A(i)%w(some_runtime_var))``) cause us to bail  --
  //   that's the variable-runtime-size case (5c-B / 5c-C).
  /// Collapse the ``fir.load (designate of A(i){memName}) ->
  /// hlfir.designate (loaded, j)`` read pattern into a direct
  /// 2-index ``hlfir.designate flatBase (i, j)`` over the Phase
  /// 5c-A companion.
  ///
  /// Why: the original IR threads every read of ``A(i)%w(j)``
  /// through the box descriptor  --  flang emits ``fir.load %ref``
  /// to fetch the descriptor, then ``hlfir.designate %loaded
  /// (j)`` to index inside the box.  After flatten replaces
  /// ``%ref`` with a plain ``ref<array<NxMxT>>``, ``fir.load``
  /// loads the *whole* 2D value rather than a box, and the
  /// inner ``designate (j)`` indexes it as if it were 1-D.
  /// We leapfrog by replacing the entire chain with a direct
  /// ``hlfir.designate flatBase (i, j)`` against the new ref.
  ///
  /// Mirrors the strategy in ``hlfir-rewrite-pointer-assigns``
  /// (forward-substitute a multi-step load chain into a single
  /// direct access against the rewrite target).
  ///
  /// Caveats
  /// -------
  /// * Only walks element-form parent designates (``A(i)``,
  ///   single scalar index per outer dim).  Section-form parent
  ///   designates (``A(1:N)%w``) are a different shape and not
  ///   handled here.
  /// * Only walks element-form INNER designates
  ///   (``loaded(j)``).  Whole-component reads
  ///   (``hlfir.assign x to <designate of A(i){w}>``) take a
  ///   separate section-rewrite path that's not yet wired.
  /// * If any reader is a ``fir.box_addr`` rather than a
  ///   designate (the path the existing pointer rewrite
  ///   handles), the chain is left alone  --  the bridge's
  ///   downstream handling for ``box_addr`` returns a
  ///   ``fir.ptr<...>`` that doesn't match the static 2D
  ///   companion.  Future TODO if real code hits this.
  void collapseAosAllocReads(hlfir::DeclareOp decl, llvm::StringRef memName, mlir::Value flatBase) {
    // Phase 5c-B: also recognise inlined-callee aliased declares.
    // When ``hlfir-inline-all`` splices a module-contained
    // ``call kernel(A)`` body into the caller, the kernel's
    // ``A`` dummy becomes a fresh ``hlfir.declare %caller_A_decl
    // dummy_scope %dsc {uniq_name="..."}`` aliasing the same
    // storage.  Designates inside the inlined kernel body are
    // rooted at the alias's results, not ``decl``'s  --  without
    // following alias chains we'd miss every read inside the
    // inlined call.
    auto isDeclOrAlias = [&](mlir::Value v) -> bool {
      for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
        if (v == decl.getResult(0) || v == decl.getResult(1)) return true;
        auto* d = v.getDefiningOp();
        if (!d) return false;
        if (auto inner = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
          v = inner.getMemref();
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
        return false;
      }
      return false;
    };
    // Resolve dimension ``dim`` of the packed companion to the extent
    // value baked into its declare's shape op.  Used to redirect a
    // ``size(A(i)%member)`` query onto the companion's matching dim --
    // the runtime ``cap_<base>_<member>`` symbol (true-boundary case)
    // or the static extent (uniform-const case).
    auto companionExtent = [](mlir::Value base, unsigned dim) -> mlir::Value {
      auto declOp = mlir::dyn_cast_or_null<hlfir::DeclareOp>(base.getDefiningOp());
      if (!declOp || !declOp.getShape()) return {};
      auto* shapeDef = declOp.getShape().getDefiningOp();
      if (auto sh = mlir::dyn_cast_or_null<fir::ShapeOp>(shapeDef)) {
        if (dim < sh.getExtents().size()) return sh.getExtents()[dim];
      }
      if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeDef)) {
        unsigned extIdx = 2 * dim + 1;
        if (extIdx < ss->getNumOperands()) return ss->getOperand(extIdx);
      }
      return {};
    };
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      // Two-stage erase so dependencies tear down cleanly:
      // (1) eraseInner  --  the rewritten inner designates
      //     (still hold a use on ``load`` until erased)
      // (2) eraseRest   --  load, memDg, parent (sweep after the
      //     inner designates are gone so the use_empty checks
      //     trigger)
      llvm::SmallVector<mlir::Operation*, 16> eraseInner;
      llvm::SmallVector<mlir::Operation*, 16> eraseRest;
      func.walk([&](fir::LoadOp load) {
        // Is this a load of a per-instance member designate?
        auto memDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(load.getMemref().getDefiningOp());
        if (!memDg) return;
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) return;
        auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
        if (!parent) return;
        if (!isDeclOrAlias(parent.getMemref())) return;
        // parent must be element-form (no triplets).
        for (bool t : parent.getIsTriplet())
          if (t) return;
        if (parent.getIndices().empty()) return;

        // Collect the outer indices (typically a single
        // scalar i for a 1-D AoS).
        llvm::SmallVector<mlir::Value, 4> outerIdx(parent.getIndices().begin(), parent.getIndices().end());

        // For each user of the loaded box: if it's an
        // element-form designate, rewrite to a direct
        // 2-index designate over flatBase.
        for (auto* u : load.getResult().getUsers()) {
          // ``size(A(i)%member)`` lowers to ``fir.box_dims`` on the
          // loaded per-instance member box.  That box no longer exists
          // after flattening, so redirect the queried extent to the
          // packed companion's matching dimension (member dim ``d``
          // maps to companion dim ``outerRank + d``).  Only the extent
          // result (#1) feeds loop bounds; lb (#0) / stride (#2), where
          // used, are the contiguous 1-based defaults of a packed row.
          if (auto bd = mlir::dyn_cast<fir::BoxDimsOp>(u)) {
            auto memberDimC = traceConstInt(bd.getDim());
            if (!memberDimC) continue;
            mlir::Value ext = companionExtent(flatBase, outerIdx.size() + static_cast<unsigned>(*memberDimC));
            if (!ext) continue;
            mlir::OpBuilder b(bd);
            bd.getResult(1).replaceAllUsesWith(ext);
            auto one = [&]() {
              return b.create<mlir::arith::ConstantOp>(bd.getLoc(), b.getIndexType(), b.getIndexAttr(1)).getResult();
            };
            if (!bd.getResult(0).use_empty()) bd.getResult(0).replaceAllUsesWith(one());
            if (!bd.getResult(2).use_empty()) bd.getResult(2).replaceAllUsesWith(one());
            eraseInner.push_back(bd);
            continue;
          }
          auto inner = mlir::dyn_cast<hlfir::DesignateOp>(u);
          if (!inner) continue;
          // Element form (no triplets, has indices).
          bool anyTrip = false;
          for (bool t : inner.getIsTriplet())
            if (t) {
              anyTrip = true;
              break;
            }
          if (anyTrip) continue;
          if (inner.getIndices().empty()) continue;

          mlir::OpBuilder b(inner);
          auto idxTy = b.getIndexType();
          auto toIndex = [&](mlir::Value v) {
            if (v.getType() == idxTy) return v;
            return b.create<fir::ConvertOp>(inner.getLoc(), idxTy, v).getResult();
          };
          llvm::SmallVector<mlir::Value, 4> mergedIdx;
          for (auto v : outerIdx) mergedIdx.push_back(toIndex(v));
          for (auto v : inner.getIndices()) mergedIdx.push_back(toIndex(v));

          // Build the new designate.  Result type stays
          // the inner designate's result type (element ref).
          auto newDg = b.create<hlfir::DesignateOp>(inner.getLoc(), inner.getResult().getType(), flatBase,
                                                    mlir::ValueRange{mergedIdx});
          inner.getResult().replaceAllUsesWith(newDg.getResult());
          eraseInner.push_back(inner);
        }
        // The load + the member/parent designate chain become
        // dead once the inner designate is erased.  Schedule
        // them for the second sweep.
        eraseRest.push_back(load);
        eraseRest.push_back(memDg);
        eraseRest.push_back(parent);
      });
      for (auto* op : eraseInner)
        if (op->use_empty()) op->erase();
      for (auto* op : eraseRest)
        if (op->use_empty()) op->erase();
    }
  }

  /// Rewrite whole-component ``hlfir.assign``s whose LHS is
  /// ``<designate of A(i){memName}>`` into row-section assigns on
  /// the flat 2D companion: ``A_<member>(i, 1:M:1) = rhs``.
  /// Without this, the existing concat path replaces the LHS with
  /// the bare flat declare (the whole 2D), and the scalar ``rhs``
  /// gets broadcast across ALL rows  --  silently corrupting
  /// previously-written rows.
  ///
  /// Element-form assigns (``A(i)%w(j) = ...``) are NOT in scope
  /// here  --  those go through the element designate + the
  /// existing concat path, which already merges parent + inner
  /// indices correctly.  Only whole-component assigns
  /// (``A(i)%w = scalar`` or ``A(i)%w = src(:)``) need the
  /// section-rewrite treatment.
  /// Look up the allocate-site size for a specific outer index of
  /// an AoS-allocatable member.  Returns the constant size used in
  /// ``allocate(A(i)%<member>(N_i))`` when ``i`` matches
  /// ``targetIdx`` and the size is a compile-time constant.
  /// Returns ``nullopt`` if the matching allocate isn't found or
  /// its size isn't constant  --  in which case the caller falls back
  /// to the global cap ``M``.
  static std::optional<int64_t> aosAllocSizeAt(hlfir::DeclareOp decl, llvm::StringRef memName, int64_t targetIdx) {
    std::optional<int64_t> found;
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      func.walk([&](hlfir::DesignateOp memDg) -> mlir::WalkResult {
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) return mlir::WalkResult::advance();
        auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
        if (!parent) return mlir::WalkResult::advance();
        if (parent.getMemref() != decl.getResult(0) && parent.getMemref() != decl.getResult(1))
          return mlir::WalkResult::advance();
        for (bool t : parent.getIsTriplet())
          if (t) return mlir::WalkResult::advance();
        if (parent.getIndices().size() != 1) return mlir::WalkResult::advance();
        auto pi = traceConstInt(parent.getIndices().front());
        if (!pi || *pi != targetIdx) return mlir::WalkResult::advance();
        for (auto* u : memDg.getResult().getUsers()) {
          auto store = mlir::dyn_cast<fir::StoreOp>(u);
          if (!store) continue;
          mlir::Value v = store.getValue();
          for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
            auto* d = v.getDefiningOp();
            if (!d) break;
            if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) {
              auto sizes = am.getShape();
              if (sizes.size() == 1) {
                if (auto sz = traceConstInt(sizes.front())) {
                  found = *sz;
                  return mlir::WalkResult::interrupt();
                }
              }
              return mlir::WalkResult::interrupt();
            }
            if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
              v = eb.getMemref();
              continue;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
              v = cv.getValue();
              continue;
            }
            break;
          }
        }
        return mlir::WalkResult::advance();
      });
    }
    return found;
  }

  void rewriteAosWholeMemberAssign(hlfir::DeclareOp decl, llvm::StringRef memName, mlir::Value flatBase, int64_t M) {
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      llvm::SmallVector<hlfir::AssignOp, 4> dead;
      func.walk([&](hlfir::AssignOp op) {
        auto memDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(op.getLhs().getDefiningOp());
        if (!memDg) return;
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) return;
        auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
        if (!parent) return;
        if (parent.getMemref() != decl.getResult(0) && parent.getMemref() != decl.getResult(1)) return;
        // parent must be element-form (no triplets).
        for (bool t : parent.getIsTriplet())
          if (t) return;
        if (parent.getIndices().empty()) return;

        // Per-instance section bound.  When the outer index
        // is a compile-time constant we can match it to a
        // specific allocate site and use that site's size as
        // the section bound  --  needed for the jagged case
        // (``A(1)%val(3)`` vs ``A(2)%val(4)``) so each row
        // assign writes only its live region instead of
        // splatting up to the global cap.  Falls back to the
        // cap when the index is symbolic or no matching
        // allocate is found.
        int64_t sectionBound = M;
        if (parent.getIndices().size() == 1) {
          if (auto pi = traceConstInt(parent.getIndices().front())) {
            if (auto specific = aosAllocSizeAt(decl, memName, *pi)) sectionBound = *specific;
          }
        }

        // Build A_w(parent_idx, 1:sectionBound:1) section designate.
        mlir::OpBuilder b(op);
        auto loc = op.getLoc();
        auto idxTy = b.getIndexType();
        auto toIndex = [&](mlir::Value v) {
          if (v.getType() == idxTy) return v;
          return b.create<fir::ConvertOp>(loc, idxTy, v).getResult();
        };
        auto c1 = b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(1));
        auto cBound = b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(sectionBound));

        llvm::SmallVector<mlir::Value, 6> indices;
        llvm::SmallVector<bool, 4> tripletFlags;
        for (auto v : parent.getIndices()) {
          indices.push_back(toIndex(v));
          tripletFlags.push_back(false);
        }
        indices.push_back(c1.getResult());
        indices.push_back(cBound.getResult());
        indices.push_back(c1.getResult());
        tripletFlags.push_back(true);

        // Result type: box<array<sectionBound x T>>  --  a row
        // view shaped to match the per-instance live region.
        auto flatTy = mlir::cast<fir::ReferenceType>(flatBase.getType()).getEleTy();
        auto flatSeq = mlir::cast<fir::SequenceType>(flatTy);
        auto eleTy = flatSeq.getEleTy();
        auto rowSeqTy = fir::SequenceType::get({sectionBound}, eleTy);
        auto boxTy = fir::BoxType::get(rowSeqTy);

        auto newShape = b.create<fir::ShapeOp>(loc, mlir::ValueRange{cBound.getResult()}).getResult();

        auto sectionDg = b.create<hlfir::DesignateOp>(loc,
                                                      /*resultType0=*/boxTy,
                                                      /*memref=*/flatBase,
                                                      /*component=*/mlir::StringAttr{},
                                                      /*component_shape=*/mlir::Value{},
                                                      /*indices=*/mlir::ValueRange{indices},
                                                      /*is_triplet=*/b.getDenseBoolArrayAttr(tripletFlags),
                                                      /*substring=*/mlir::ValueRange{},
                                                      /*complex_part=*/mlir::BoolAttr{},
                                                      /*shape=*/newShape,
                                                      /*typeparams=*/mlir::ValueRange{},
                                                      /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});

        // Build the new assign with the section LHS.  The RHS
        // stays as-is (scalar or src array of matching shape).
        b.create<hlfir::AssignOp>(loc, op.getRhs(), sectionDg.getResult());
        dead.push_back(op);
      });
      for (auto op : dead) op.erase();
    }
  }

  /// Strip the ``realloc`` attribute from ``hlfir.assign`` ops
  /// whose LHS designates an AoS-allocatable member after
  /// flattening.  Phase 5c-A turns the LHS into a static array
  /// section; ``realloc`` is only valid when the LHS is genuinely
  /// allocatable, and the op's verifier rejects it otherwise.
  void stripReallocOnAosMember(hlfir::DeclareOp decl, llvm::StringRef memName) {
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      func.walk([&](hlfir::AssignOp op) {
        mlir::Value lhs = op.getLhs();
        auto memDg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(lhs.getDefiningOp());
        if (!memDg) return;
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) return;
        auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
        if (!parent) return;
        if (parent.getMemref() != decl.getResult(0) && parent.getMemref() != decl.getResult(1)) return;
        op.setRealloc(false);
      });
    }
  }

  /// Erase the kernel-internal allocate / deallocate chain for an
  /// AoS allocatable member after Phase 5c-A flattening.  The 2D
  /// buffer is now pre-allocated at static shape, so each
  /// ``allocate(A(i)%<member>(M))`` becomes a no-op:
  ///   * ``fir.store (embox(allocmem)) to <designate>``  --  erase.
  ///   * ``fir.allocmem`` itself  --  erase if dead (no other users).
  ///   * ``fir.embox``  --  erase if dead.
  ///   * ``fir.freemem`` (matching deallocate)  --  erase.
  ///   * Any subsequent ``fir.zero_bits`` + ``fir.embox`` + store
  ///     pattern (the post-deallocate "set descriptor to null"
  ///     sequence flang inserts)  --  erase.
  void eraseAosAllocDeallocChain(hlfir::DeclareOp decl, llvm::StringRef memName) {
    llvm::SmallVector<mlir::Operation*, 16> deadOps;
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      func.walk([&](hlfir::DesignateOp memDg) {
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) return;
        auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
        if (!parent) return;
        if (parent.getMemref() != decl.getResult(0) && parent.getMemref() != decl.getResult(1)) return;
        for (auto* u : memDg.getResult().getUsers())
          if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) deadOps.push_back(st);
      });
      // Also collect ``fir.freemem`` ops whose source traces
      // back through ``fir.box_addr`` + ``fir.load`` of the
      // member designate.
      func.walk([&](fir::FreeMemOp fm) {
        mlir::Value v = fm.getHeapref();
        for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
          auto* d = v.getDefiningOp();
          if (!d) break;
          if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
            v = ba.getVal();
            continue;
          }
          if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
            v = ld.getMemref();
            continue;
          }
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            v = cv.getValue();
            continue;
          }
          if (auto memDg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
            mlir::StringAttr compAttr;
            for (auto nm : {"component_name", "component"})
              if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
                compAttr = a;
                break;
              }
            if (compAttr && compAttr.getValue() == memName) {
              auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
              if (parent && (parent.getMemref() == decl.getResult(0) || parent.getMemref() == decl.getResult(1))) {
                deadOps.push_back(fm);
              }
            }
            break;
          }
          break;
        }
      });
    }
    // Erase stores / freemem; the embox / allocmem / etc. become
    // dead and get swept by canonicalisation downstream.  The
    // wrapping do-loop's body (e.g. the per-instance
    // ``deallocate`` loop) becomes a stub of just iv bookkeeping
    // and dead box-load ops  --  but the IR-level loop op stays.
    // We don't try to erase the loop here (its result types
    // don't trivially substitute into init args).  Instead the
    // SDFGBuilder's post-gen sweep adds a single empty state to
    // any zero-block CFG region, so the resulting empty
    // ``LoopRegion`` validates cleanly.
    for (auto* op : deadOps) op->erase();
  }

  /// Result of scanning all ``allocate(A(i)%<m>(N))`` sites for a
  /// single member: the per-batch padding size (the max of all
  /// constant N values seen), and a uniform-flag set when every
  /// site uses the same constant.  Returns ``nullopt`` iff any
  /// site is non-constant or there are no sites.
  struct AosAllocConstSize {
    int64_t padTo;  ///< pad-to-max companion column count
    bool uniform;   ///< true iff every allocate uses the same constant
  };

  static std::optional<AosAllocConstSize> aosAllocMaxConstSize(hlfir::DeclareOp decl, llvm::StringRef memName) {
    std::optional<int64_t> maxSeen, minSeen;
    bool anySite = false;
    // Walk all hlfir.designate ops that index into ``decl`` via an
    // outer-element form (``A(i)``) and then component-select the
    // target member.  For each, follow store users and recover the
    // allocate site.
    //
    // Generalised from "uniform-only" to "max-of-constants": a
    // genuinely jagged AoS like batched CSR (``allocate(A(1)%val(3))``
    // and ``allocate(A(2)%val(4))``) flattens to a max-padded
    // companion ``A_val(N, max)``.  The kernel's element-wise
    // accesses (``A(i)%val(j)``) work uniformly because ``j``
    // never exceeds the per-instance live size by program logic;
    // the padding columns stay unread.  Whole-component assigns
    // (``A(i)%w = scalar``) still fire the section-rewrite path  --
    // see ``rewriteAosWholeMemberAssign``  --  but only when the
    // result is uniform (otherwise the per-instance live size
    // differs from the cap and the simple ``1:M:1`` triplet
    // would over-write padding with stale data).
    if (auto func = decl->getParentOfType<mlir::func::FuncOp>()) {
      mlir::WalkResult result = func.walk([&](hlfir::DesignateOp memDg) -> mlir::WalkResult {
        // Check this is the member designate (A(i){memName}).
        mlir::StringAttr compAttr;
        for (auto nm : {"component_name", "component"})
          if (auto a = memDg->getAttrOfType<mlir::StringAttr>(nm)) {
            compAttr = a;
            break;
          }
        if (!compAttr || compAttr.getValue() != memName) return mlir::WalkResult::advance();

        // Parent must be a per-instance designate of decl.
        auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(memDg.getMemref().getDefiningOp());
        if (!parent) return mlir::WalkResult::advance();
        if (parent.getMemref() != decl.getResult(0) && parent.getMemref() != decl.getResult(1))
          return mlir::WalkResult::advance();
        // parent should be element-form (no triplets).
        for (bool t : parent.getIsTriplet())
          if (t) return mlir::WalkResult::advance();

        // Look for an allocate-store chain on memDg.
        for (auto* u : memDg.getResult().getUsers()) {
          auto store = mlir::dyn_cast<fir::StoreOp>(u);
          if (!store) continue;
          mlir::Value v = store.getValue();
          for (int i = 0; i < kFlattenMaxDepth && v; ++i) {
            auto* d = v.getDefiningOp();
            if (!d) break;
            if (auto am = mlir::dyn_cast<fir::AllocMemOp>(d)) {
              // Recover the size: allocmem typically takes
              // a single ``%size`` operand for a 1-D array.
              auto sizes = am.getShape();
              if (sizes.size() != 1) {
                maxSeen.reset();
                return mlir::WalkResult::interrupt();
              }
              auto sz = traceConstInt(sizes.front());
              if (!sz) {
                maxSeen.reset();
                return mlir::WalkResult::interrupt();
              }
              anySite = true;
              if (!maxSeen || *sz > *maxSeen) maxSeen = *sz;
              if (!minSeen || *sz < *minSeen) minSeen = *sz;
              break;
            }
            if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
              v = eb.getMemref();
              continue;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
              v = cv.getValue();
              continue;
            }
            break;
          }
        }
        return mlir::WalkResult::advance();
      });
      if (result.wasInterrupted()) return std::nullopt;
    }
    if (!anySite || !maxSeen) return std::nullopt;
    return AosAllocConstSize{*maxSeen, *minSeen == *maxSeen};
  }

  static bool isLocallyFlattenable(hlfir::DeclareOp decl) {
    auto* def = decl.getMemref().getDefiningOp();
    auto alloca = mlir::dyn_cast_or_null<fir::AllocaOp>(def);
    if (!alloca) return false;

    bool outerIsArray = false;
    llvm::SmallVector<int64_t, 4> outerShape;
    auto rec = peelToRecord(decl.getResult(0).getType(), outerIsArray, outerShape);
    if (!rec) return false;
    // Only OWNED struct storage flattens locally.  ``peelToRecord`` peels
    // through box/ptr/heap, so it also reaches the record of a POINTER /
    // ALLOCATABLE derived-type local -- but that is a rebindable runtime
    // descriptor whose member reads run through the box, not the declare
    // (see isIndirectStructLocal).  Leave those to the pointer-rewrite /
    // view path; flattening them yields dead, mistyped companions.
    if (isIndirectStructLocal(decl.getResult(0).getType())) return false;
    // Runtime-sized outer array (dynamic alloca operands or an unknown type
    // extent): the companion ``fir.alloca`` threads the live shape operands
    // through (``outerExtentValues`` + ``makeCompanionAlloca``).  Supported
    // ONLY when (a) no member is an allocatable array -- the Phase-5c
    // alloc-member AoS path still needs a static outer extent to size its 2D
    // companion -- (b) the declare's shape is a plain ``fir.shape`` so the
    // extents are recoverable, and (c) every member is directly flattenable
    // (the single-level companion path threads the runtime extents; the
    // nested-leaf path still bakes static shapes, so a runtime-sized nested
    // struct stays deferred).  Otherwise fall through to the loud downstream
    // path rather than emit a mistyped companion.
    bool runtimeSized = alloca.getNumOperands() != 0;
    for (auto d : outerShape)
      if (d == fir::SequenceType::getUnknownExtent()) runtimeSized = true;
    if (runtimeSized) {
      for (auto& pair : rec.getTypeList())
        if (isAllocatableArrayMember(pair.second)) return false;
      if (!decl.getShape() || !decl.getShape().getDefiningOp<fir::ShapeOp>()) return false;
      if (!allMembersFlattenable(rec)) return false;
    }
    // Phase 5c-A: AoS + allocatable member.  Permitted when every
    // ``allocate(A(i)%<member>(M))`` site uses the SAME compile-
    // time constant ``M`` across all instances.  Then the flat
    // companion ``A_<member>(N, M)`` is fully static and the
    // alloc / dealloc / load / designate machinery becomes
    // semantic no-ops over the pre-allocated 2D buffer.  The
    // read-side rewrite (``collapseAosAllocReads``) leapfrogs
    // ``fir.load + hlfir.designate (loaded, j)`` into a direct
    // 2-index ``hlfir.designate flatBase (i, j)``.
    if (outerIsArray) {
      for (auto& pair : rec.getTypeList()) {
        if (!isAllocatableArrayMember(pair.second)) continue;
        if (!aosAllocMaxConstSize(decl, pair.first).has_value()) return false;
      }
    }
    if (allMembersFlattenable(rec)) return true;
    // Nested-struct fallback.  ``collectFlatLeaves`` recurses
    // through ``RecordType`` and ``array<N x RecordType>``
    // members, building each leaf's flat companion shape.  When
    // the outer declare is itself an array of a nested record
    // (``type(t) :: s(3)`` with ``t`` nested), the same walker
    // accepts: thread the outer extents in as the initial
    // ``outerDims`` so every leaf's flat companion concatenates
    // them as leading dims (matching what ``splitLocal`` does
    // when it builds the alloca below).
    llvm::SmallVector<std::string, 4> prefix;
    llvm::SmallVector<int64_t, 4> initialDims(outerShape.begin(), outerShape.end());
    llvm::SmallVector<FlatLeaf, 8> leaves;
    if (!collectFlatLeaves(rec, prefix, initialDims, leaves)) return false;
    // The LOCAL nested path (``splitLocal``) cannot own an
    // allocatable/pointer LEAF (``box<heap|ptr<...>>``): it has no caller
    // bindings to supply the descriptor (unlike the dummy-arg nested path,
    // which receives one per leaf) and no local allocate-flow synthesis, so
    // it would emit an attr-less box declare (verifier-invalid) and, even
    // with the attr, leave the companion box unallocated -- a silent
    // miscompile.  A box leaf is a rebindable descriptor, not owned static
    // storage (cf. ``isIndirectStructLocal``); defer the whole struct to the
    // loud downstream path rather than half-flatten it.
    for (auto& leaf : leaves)
      if (mlir::isa<fir::BaseBoxType>(leaf.leafTy)) return false;
    return true;
  }

  void splitLocal(hlfir::DeclareOp decl) {
    mlir::OpBuilder b(decl);
    auto* ctx = b.getContext();
    auto loc = decl.getLoc();

    bool outerIsArray = false;
    llvm::SmallVector<int64_t, 4> outerShape;
    auto rec = peelToRecord(decl.getResult(0).getType(), outerIsArray, outerShape);
    auto shape = decl.getShape();
    auto baseName = decl.getUniqName().str();

    // Companion uniq_name uniquer.  The flat companion for an
    // array-of-records member ``base % m`` naturally wants
    // ``<base>_<m>`` -- but a kernel may ALSO declare a REAL local of
    // that exact name (ICON ocean's ``p_nabla2_dual`` AoR of
    // ``t_cartesian_coordinates`` alongside a 3-D sync-scratch local
    // ``p_nabla2_dual_x``).  Reusing the name would give two declares
    // the same user-visible (demangled) name, so ``extract_vars`` binds
    // the member access to the WRONG (real-local, lower-rank)
    // descriptor and the member's trailing dim offset symbol
    // (``offset_p_nabla2_dual_x_d3``) never registers.  Collect the
    // demangled names of every existing declare in the enclosing
    // function once, then hand out companion names that avoid them
    // (and each other).  The no-collision case is preserved verbatim
    // -- the returned name equals ``<base>_<m>`` -- so only a genuine
    // clash pays the disambiguation ``_cc`` (cartesian-companion)
    // suffix.
    llvm::StringSet<> takenDemangled;
    if (auto parentFunc = decl->getParentOfType<mlir::func::FuncOp>())
      parentFunc.walk([&](hlfir::DeclareOp d) { takenDemangled.insert(demangleVarName(d.getUniqName())); });
    auto mintCompanionName = [&](std::string cand) -> std::string {
      if (takenDemangled.count(demangleVarName(cand))) {
        std::string base = cand;
        for (int n = 1; takenDemangled.count(demangleVarName(cand)); ++n) {
          std::string suffix = "_cc";
          if (n > 1) suffix += std::to_string(n);
          cand = base + suffix;
        }
      }
      takenDemangled.insert(demangleVarName(cand));
      return cand;
    };

    // Nested-record path: walk the path-leaf set and synthesise
    // one declare per leaf, indexed by the path-joined name
    // (``o_inner_x``).  The single-level path below stays the
    // hot path for non-nested structs.
    //
    // When the outer declare is itself an array of a nested
    // record (``type(t) :: s(3)`` where ``t`` is nested), the
    // outer extents thread into ``collectFlatLeaves`` as the
    // initial ``outerDims`` so each leaf's flat companion
    // concatenates them as leading dims (e.g. ``s(3)%w(5,5)``
    // collapses to ``s_w`` of shape ``(3, 5, 5)``).
    bool nested = !allMembersFlattenable(rec);
    if (nested) {
      llvm::SmallVector<std::string, 4> prefix;
      llvm::SmallVector<int64_t, 4> initialDims(outerShape.begin(), outerShape.end());
      llvm::SmallVector<FlatLeaf, 8> leaves;
      if (!collectFlatLeaves(rec, prefix, initialDims, leaves)) return;

      llvm::StringMap<mlir::Value> leafBase;
      for (auto& leaf : leaves) {
        auto memTy = leaf.leafTy;
        auto newAlloca = b.create<fir::AllocaOp>(loc, memTy);
        auto declTy = fir::ReferenceType::get(memTy);
        std::string suffix;
        std::string joinedKey;
        for (unsigned i = 0; i < leaf.path.size(); ++i) {
          if (i) {
            suffix += "_";
            joinedKey += "_";
          }
          suffix += leaf.path[i];
          joinedKey += leaf.path[i];
        }

        // Array leaves need a fir.shape operand on the declare.
        // ``hlfir.declare op of array entity with a raw address
        // base must have a shape operand``.  We derive extents
        // from the leaf type itself; static extents only.
        mlir::Value leafShape;
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(memTy)) {
          llvm::SmallVector<int64_t, 4> exts;
          bool allStatic = true;
          for (auto d : seq.getShape()) {
            if (d == fir::SequenceType::getUnknownExtent()) {
              allStatic = false;
              break;
            }
            exts.push_back(d);
          }
          if (!allStatic) return;  // dynamic extent unsupported
          leafShape = emitStaticShape(b, loc, exts);
        }

        llvm::SmallVector<mlir::Value, 2> operands{newAlloca};
        if (leafShape) operands.push_back(leafShape);
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, mintCompanionName(baseName + "_" + suffix)));
        attrs.append(declareSegments(b, /*hasShape=*/leafShape != nullptr));
        auto newDecl =
            b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{declTy, declTy}, mlir::ValueRange(operands), attrs);
        leafBase[joinedKey] = newDecl.getResult(0);
      }

      // Walk every hlfir.designate chain rooted at ``decl`` (or
      // any inlined-callee alias of it) and rewrite each leaf to
      // the matching flat companion in ``leafBase``.  Shared
      // helper with the dummy-arg nested path
      // (``replaceStructArgNested``)  --  both sides build the same
      // ``leafBase`` and need the same chain-rewrite logic.
      rewriteChainsRootedAt(decl, leafBase);

      if (decl.getResult(0).use_empty() && decl.getResult(1).use_empty()) {
        auto* allocaOp = decl.getMemref().getDefiningOp();
        decl.erase();
        if (allocaOp && allocaOp->use_empty()) allocaOp->erase();
      }
      return;
    }

    // Single-level path  --  every member is flat directly.
    llvm::StringMap<mlir::Value> memberBase;
    // Track which members are AoS-with-array-members so the
    // designate rewriter knows to merge outer + inner indices.
    llvm::StringSet<> concatMembers;
    for (auto& pair : rec.getTypeList()) {
      auto memName = pair.first;
      auto memTy = pair.second;

      // Phase 5c-A: AoS + allocatable member with uniform
      // constant allocate size.  Synth a fully static 2D
      // companion ``A_<member>(N, M)`` and erase the per-
      // instance ``fir.allocmem`` / ``fir.freemem`` chain
      // (the buffer is pre-allocated at the static N*M shape;
      // the kernel's ``allocate(A(i)%w(M))`` becomes a
      // semantic no-op).
      if (isAllocatableArrayMember(memTy) && outerIsArray) {
        auto sizeOpt = aosAllocMaxConstSize(decl, memName);
        if (!sizeOpt) continue;  // gate already verified; defensive
        int64_t M = sizeOpt->padTo;
        bool sizeUniform = sizeOpt->uniform;
        auto box = mlir::cast<fir::BoxType>(memTy);
        mlir::Type eleTy;
        if (auto heap = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
          eleTy = mlir::cast<fir::SequenceType>(heap.getEleTy()).getEleTy();
        else if (auto ptr = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
          eleTy = mlir::cast<fir::SequenceType>(ptr.getEleTy()).getEleTy();
        else
          continue;

        // Concat shape: outerShape x {M}.
        llvm::SmallVector<int64_t, 4> exts(outerShape.begin(), outerShape.end());
        exts.push_back(M);
        auto pointee = fir::SequenceType::get(exts, eleTy);
        auto refTy = fir::ReferenceType::get(pointee);

        auto newAlloca = b.create<fir::AllocaOp>(loc, pointee);
        mlir::Value memberShape = emitStaticShape(b, loc, exts);

        std::string flatName = baseName + "_" + memName;
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, flatName));
        attrs.append(declareSegments(b, /*hasShape=*/true));

        llvm::SmallVector<mlir::Value, 2> ops{newAlloca.getResult(), memberShape};
        auto newDecl = b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{refTy, refTy}, mlir::ValueRange{ops}, attrs);
        memberBase[memName] = newDecl.getResult(0);
        concatMembers.insert(memName);  // tells designate rewriter
                                        // to merge outer + inner indices

        // Strip ``realloc`` from any ``hlfir.assign`` whose
        // LHS targets this member: after flatten the LHS is a
        // static array section, not an allocatable, and the
        // assign op's verifier rejects ``realloc=true`` on
        // non-allocatable LHS.
        stripReallocOnAosMember(decl, memName);

        // Rewrite whole-component assigns (``A(i)%w = ...``)
        // into row-section assigns (``A_w(i, 1:N_i:1) = ...``).
        // ``rewriteAosWholeMemberAssign`` resolves the
        // section bound per-assign: if the parent's outer
        // index is a compile-time constant it looks up the
        // matching allocate's size (handles the jagged case
        // ``A(1)%val(3)`` / ``A(2)%val(4)`` where each row
        // wants its own live size); otherwise falls back to
        // the global cap ``M``.  Must run BEFORE
        // ``rewriteDesignate`` sweeps the parent chain,
        // otherwise the concat path's whole-component branch
        // would replace the LHS with the bare flat 2D ref and
        // the assign would broadcast across all rows.
        (void)sizeUniform;  // kept for potential future
                            // gating; the per-assign size
                            // resolution covers both cases.
        rewriteAosWholeMemberAssign(decl, memName, newDecl.getResult(0), M);

        // Collapse ``fir.load + hlfir.designate (loaded, j)``
        // chains into direct 2-index designates over the new
        // companion.  Must run BEFORE ``rewriteDesignate``
        // sweeps the parent designate chain, otherwise the
        // load + inner designate would be left dangling
        // against the rewritten (now plain ref) parent.
        collapseAosAllocReads(decl, memName, newDecl.getResult(0));

        // Erase the per-instance allocate / freemem chain.
        // Each ``allocate(A(i)%<member>(M))`` lowers to:
        //   %alloc = fir.allocmem !fir.array<?xT>, %M
        //   %box   = fir.embox %alloc(%shape)
        //   fir.store %box to <designate of A(i){memName}>
        // and each ``deallocate`` to ``fir.freemem``.  After
        // synth, the 2D buffer is already there; the chain
        // becomes dead.  Erase the stores first (the box
        // value has no other consumer), then sweep the
        // dangling allocmem / embox / freemem.
        eraseAosAllocDeallocChain(decl, memName);
        continue;
      }

      // Allocatable / pointer array members get a parallel
      // synthesis path (Phase 5a + 5b).  The companion is a
      // top-level allocatable / pointer: ``fir.alloca
      // <box<heap|ptr<array<?xT>>>>`` plus a declare carrying
      // the matching fortran_attr.  Skipped for AoS outers  --
      // those go through the Phase 5c-A path above.
      if (isAllocatableArrayMember(memTy) && !outerIsArray) {
        auto allocaTy = memTy;  // box<heap|ptr<array<?xT>>>
        auto refTy = fir::ReferenceType::get(allocaTy);
        auto newAlloca = b.create<fir::AllocaOp>(loc, allocaTy);

        // Pick the right fortran_attr based on the member's
        // inner indirection: ``fir.heap`` -> ALLOCATABLE,
        // ``fir.ptr`` -> POINTER.  Downstream queries
        // (extract_vars peel-through, the bridge's
        // allocatable / pointer handling) key on this flag.
        auto box = mlir::cast<fir::BoxType>(memTy);
        bool isPointer = mlir::isa<fir::PointerType>(box.getEleTy());
        auto attrFlag = isPointer ? fir::FortranVariableFlagsEnum::pointer : fir::FortranVariableFlagsEnum::allocatable;

        std::string flatName = baseName + "_" + memName;
        mlir::NamedAttrList attrs;
        attrs.append("uniq_name", mlir::StringAttr::get(ctx, flatName));
        attrs.append("fortran_attrs", fir::FortranVariableFlagsAttr::get(ctx, attrFlag));
        attrs.append(declareSegments(b, /*hasShape=*/false));

        auto newDecl = b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{refTy, refTy},
                                                  mlir::ValueRange{newAlloca.getResult()}, attrs);
        memberBase[memName] = newDecl.getResult(0);

        // Bug fix (Phase 5a): flang names the per-allocate
        // ``fir.allocmem`` op after the MEMBER's module scope
        // (e.g. ``_QMlibEw.alloc`` for ``module lib :: type t
        // :: real, allocatable :: w(:)``), not after the
        // enclosing struct's local declare scope.  The bridge
        // collects allocate sites by matching
        // ``<declUniqName>.alloc`` (see ``collectAllocSites``
        // in extract_vars.cpp).  Without renaming, the flat
        // ``_QFmainEs_w`` declare won't find ``_QMlibEw.alloc``
        // and the SDFG ends up with a free symbol
        // ``s_w_d0`` that nothing binds.
        //
        // Find every ``fir.allocmem`` reaching this member's
        // designate via embox + store and rename it to
        // ``<flat_uniq_name>.alloc``.  Walk pre-rewrite: at
        // this point the designate of ``%struct{"<memName>"}``
        // still exists; the store of the embox-of-allocmem
        // targets it.
        renameMemberAllocmems(decl, memName, flatName);
        continue;
      }

      auto pointee = companionPointee(outerIsArray, outerShape, memTy);
      if (!pointee) continue;

      bool memberIsArray = mlir::isa<fir::SequenceType>(memTy);
      bool concat = outerIsArray && memberIsArray;
      mlir::Type res1Ty = fir::ReferenceType::get(pointee);

      bool outerDynamic = false;
      for (auto d : outerShape)
        if (d == fir::SequenceType::getUnknownExtent()) outerDynamic = true;

      // Result #0 is the HLFIR "variable" form.  For the concat case it must be
      // the flat companion (``rewrapWith`` would produce the nested
      // ``ref<array<N x array<M, ...>>>`` the verifier rejects against the flat
      // alloca); a runtime-sized companion's variable form is a
      // ``box<array<?x...>>`` (descriptor-aware), a static one a plain
      // ``ref<array<...>>``.  The non-concat case keeps the outer declare's own
      // wrapping (already a box for a dynamic outer, a ref for a static one).
      mlir::Type res0Ty;
      if (concat)
        res0Ty = outerDynamic ? mlir::Type(fir::BoxType::get(pointee)) : res1Ty;
      else
        res0Ty = rewrapWith(decl.getResult(0).getType(), memTy);

      // The member's own extents must be static on the local path -- a dynamic
      // member dim is unsupported here (bail, as before).  Gathered once, up
      // front, since both the concat shape and the scalar-outer member shape
      // need them.
      llvm::SmallVector<int64_t, 4> memExts;
      if (memberIsArray) {
        bool allStatic = true;
        for (auto d : mlir::cast<fir::SequenceType>(memTy).getShape()) {
          if (d == fir::SequenceType::getUnknownExtent()) {
            allStatic = false;
            break;
          }
          memExts.push_back(d);
        }
        if (!allStatic) continue;
      }

      // Build the companion alloca and pick its shape operand.  A runtime-sized
      // outer array threads its live extents (one Value per outer dim, from the
      // declare's ``fir.shape``) into BOTH the alloca and a fresh ``fir.shape``:
      // a plain no-operand alloca over a dynamic pointee, or a static-literal
      // shape baking ``-1`` for the unknown dims, is verifier-invalid.  A fully
      // static outer keeps the prior plain-alloca + ``emitStaticShape`` path
      // verbatim (the hot path, byte-for-byte unchanged).
      mlir::Value newAlloca;
      mlir::Value memberShape = shape;
      if (outerDynamic) {
        // outerDynamic implies outerIsArray, so the declare carries the live
        // outer extents.  ``fullExtVals`` is aligned 1:1 with the pointee dims:
        // the outer extents, then (concat only) the static member extents.
        llvm::SmallVector<mlir::Value, 6> fullExtVals = outerExtentValues(decl);
        if (concat)
          for (auto e : memExts)
            fullExtVals.push_back(b.create<mlir::arith::ConstantOp>(loc, b.getIndexType(), b.getIndexAttr(e)));
        newAlloca = makeCompanionAlloca(b, loc, pointee, fullExtVals);
        if (concat) {
          // Fresh shape over outer+member  --  ``decl.getShape()`` only carries
          // the outer dim(s).
          auto shTy = fir::ShapeType::get(ctx, fullExtVals.size());
          memberShape = b.create<fir::ShapeOp>(loc, shTy, fullExtVals);
          concatMembers.insert(memName);
        }
        // outer-array + scalar member: pointee is ``array<outer x scalar>`` and
        // ``decl.getShape()`` already describes it  --  keep memberShape =
        // shape.
      } else {
        newAlloca = b.create<fir::AllocaOp>(loc, pointee);
        if (concat) {
          llvm::SmallVector<int64_t, 6> exts(outerShape.begin(), outerShape.end());
          for (auto e : memExts) exts.push_back(e);
          memberShape = emitStaticShape(b, loc, exts);
          concatMembers.insert(memName);
        } else if (!outerIsArray && memberIsArray) {
          // Scalar outer + array member: the new declare needs a shape over the
          // MEMBER's own extents.  The outer declare's ``shape`` is null for a
          // plain ``type(t) :: x``, so without a fresh ``fir.shape`` the
          // synthesised ``hlfir.declare`` for ``x_arr_field`` would have a raw
          // address base AND no shape operand  --  which the verifier rejects
          // with "must have a shape operand that is a shape or shapeshift".
          memberShape = emitStaticShape(b, loc, memExts);
        }
      }

      llvm::SmallVector<mlir::Value, 2> operands;
      operands.push_back(newAlloca);
      if (memberShape) operands.push_back(memberShape);

      mlir::NamedAttrList attrs;
      attrs.append("uniq_name", mlir::StringAttr::get(ctx, mintCompanionName(baseName + "_" + memName)));
      attrs.append(declareSegments(b, /*hasShape=*/memberShape != nullptr));

      auto newDecl =
          b.create<hlfir::DeclareOp>(loc, mlir::TypeRange{res0Ty, res1Ty}, mlir::ValueRange(operands), attrs);

      memberBase[memName] = newDecl.getResult(0);
    }

    // Collect designates to rewrite.  For non-concat members the
    // direct user is the component-designate.  For concat members
    // the direct user is an INDEXED designate (no component) on
    // the outer array; the actual component-designate is its
    // child.  We also walk transparently through:
    //   * ``hlfir.declare`` aliases  --  inlined-callee dummy declares
    //     that share the outer's storage.
    //   * ``fir.embox`` / ``fir.convert`` chains  --  the wrapping
    //     flang inserts when an inlined callee takes a
    //     ``CLASS(t)`` (or assumed-shape ``TYPE(t)``) dummy.  The
    //     outer concrete declare gets emboxed to ``fir.box<t>``,
    //     converted to ``fir.class<t>``, and the inlined declare
    //     is over the converted value.
    llvm::SmallVector<hlfir::DesignateOp, 16> designates;
    llvm::SmallVector<hlfir::DeclareOp, 8> aliasDecls;
    llvm::SmallVector<mlir::Operation*, 4> wrapperOps;
    std::function<void(mlir::Value)> collectFrom = [&](mlir::Value root) {
      for (auto* u : root.getUsers()) {
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
          bool hasComponent = false;
          for (auto nm : {"component_name", "component"})
            if (dg->getAttrOfType<mlir::StringAttr>(nm)) {
              hasComponent = true;
              break;
            }
          if (hasComponent) {
            designates.push_back(dg);
          } else {
            for (auto* cu : dg.getResult().getUsers())
              if (auto cdg = mlir::dyn_cast<hlfir::DesignateOp>(cu)) designates.push_back(cdg);
          }
        } else if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
          aliasDecls.push_back(dc);
          collectFrom(dc.getResult(0));
          if (dc.getResult(1) != dc.getResult(0)) collectFrom(dc.getResult(1));
        } else if (mlir::isa<fir::EmboxOp>(u) || mlir::isa<fir::ConvertOp>(u)) {
          wrapperOps.push_back(u);
          for (auto v : u->getResults()) collectFrom(v);
        }
      }
    };
    // Seed from BOTH declare results.  A runtime-sized outer array's
    // ``hlfir.declare`` returns a ``(box, ref)`` pair; some member designates
    // root at the raw ``ref`` (result #1) rather than the descriptor-aware
    // ``box`` (result #0) -- notably the section pack/unpack loop
    // ``p_nabla2_dual(:,:,blockno) % x(k)`` that copies an
    // array-of-cartesian member into a flat sync buffer.  Walking result #0
    // alone leaves those designates on the original struct, so they collapse
    // downstream to the bare ``<base>_x`` name (colliding with the real
    // 3-D local of that name) and the struct declare never clears.  Mirror
    // the nested path (``rewriteChainsRootedAt``), which already seeds both.
    collectFrom(decl.getResult(0));
    if (decl.getResult(1) != decl.getResult(0)) collectFrom(decl.getResult(1));
    for (auto dg : designates) rewriteDesignate(dg, memberBase, concatMembers);
    for (auto a : aliasDecls)
      if (a.getResult(0).use_empty() && a.getResult(1).use_empty()) a.erase();
    // Sweep wrapper ops in REVERSE so each step's only users
    // (the next op down the chain) are already gone before we
    // try to erase its source.
    for (auto* w : llvm::reverse(wrapperOps))
      if (llvm::all_of(w->getResults(), [](mlir::Value v) { return v.use_empty(); })) w->erase();

    if (decl.getResult(0).use_empty() && decl.getResult(1).use_empty()) {
      auto* allocaOp = decl.getMemref().getDefiningOp();
      decl.erase();
      if (allocaOp && allocaOp->use_empty()) allocaOp->erase();
    }
  }
};

}  // anonymous namespace

std::vector<ShallowAliasInfo> computeShallowAliasReport(mlir::ModuleOp module) {
  std::vector<ShallowAliasInfo> out;
  llvm::StringSet<> seen;
  auto consider = [&](mlir::Type t) {
    mlir::Type p = unwrapAll(t);
    while (auto seq = mlir::dyn_cast<fir::SequenceType>(p))
      p = unwrapAll(seq.getEleTy());  // peel array-of-record to the record
    auto rec = mlir::dyn_cast<fir::RecordType>(p);
    if (!rec || !seen.insert(rec.getName()).second) return;
    ShallowAlias sa = analyzeShallowAlias(rec);
    out.push_back({rec.getName().str(), sa.ok, sa.ok ? sa.count : 0, sa.ok ? dtypeName(sa.elem) : std::string()});
  };
  module.walk([&](mlir::Operation* op) {
    for (auto t : op->getResultTypes()) consider(t);
    for (auto t : op->getOperandTypes()) consider(t);
  });
  module.walk([&](mlir::func::FuncOp f) {
    for (auto t : f.getArgumentTypes()) consider(t);
  });
  return out;
}

std::unique_ptr<mlir::Pass> createFlattenStructsPass() { return std::make_unique<FlattenStructsPass>(); }

std::unique_ptr<mlir::Pass> createSplitAoRDummiesPass() {
  auto p = std::make_unique<FlattenStructsPass>();
  p->splitOnly = true;
  return p;
}

}  // namespace hlfir_bridge
