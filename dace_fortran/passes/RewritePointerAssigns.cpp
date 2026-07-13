// ============================================================================
// RewritePointerAssigns.cpp  --  collapse Fortran pointer rebinding under the
// strict-no-aliasing assumption.
// ============================================================================
//
// The bridge assumes distinct names never alias the same storage at runtime
// (a relaxation Fortran's strict semantics do NOT grant  --  the language
// allows ``POINTER`` rebinds to overlap with ``TARGET`` declarations and
// even with each other).  Under this relaxation, ``tmp => target`` is a
// pure rename: every read or write of ``tmp`` after the rebind is an
// access to ``target``'s storage.  We materialise that rename by
// rewriting all uses of the pointer declare to the target declare.
//
// Uniform "always rebase to parent" strategy (TARGET design)
// ===========================================================
// Every pointer rebind has the same logical shape:
//
//     ptr => <parent>(<chain-indices>)
//
// where ``<parent>`` is a single ``hlfir.declare`` (the original
// caller-side or local-side TARGET storage) and ``<chain-indices>`` is
// a possibly-empty list of designate steps (whole-array, section
// triplets, scalar element selection) flang lowered between the
// rebind value and the parent declare.
//
// The rewrite is the same for every shape: replace each access through
// the pointer with a direct designate over the PARENT, merging the
// chain's indices with the access (the Fortran source author's
// ``p(i, j)``):
//
//   ``ptr => x``               =>  ``p(i, j)``     ->  ``x(i, j)``
//   ``ptr => x(2:5)``          =>  ``p(i)``        ->  ``x(i + 1)``
//   ``ptr => x(:, j)``         =>  ``p(i)``        ->  ``x(i, j)``
//   ``ptr => arr(i)`` (scalar) =>  ``p`` (scalar)  ->  ``arr(i)``
//   ``ptr => s%a`` (after flatten) =>  ``p(i)``    ->  ``s_a(i)``
//   ``ptr => s%a(2:5)`` (after flatten) => ``p(i)`` ->  ``s_a(i + 1)``
//
// All of these reduce to the same operation:
//   1. Walk the rebind value through ``fir.embox`` / ``fir.rebox`` /
//      ``fir.convert`` and ``hlfir.designate`` ops to find the parent
//      declare and capture each designate's (indices, isTriplet)
//      record into a CHAIN.
//   2. For every downstream access through ``fir.load %ptrDecl#0``
//      (designate user OR box_addr user) build a fresh designate
//      over the parent with merged indices: triplet positions
//      consume one access index (rebased by ``lo - 1``); scalar
//      positions take the chain's literal value verbatim; whole-
//      array (no chain entry) passes the access indices through
//      untouched.
//   3. Erase the load, the rebind store, the alloca / init chain,
//      and the pointer declare.
//
// This collapses the per-variant special cases (plain target,
// slice target, element rebind) into one rewrite step.  The
// box_addr legacy fast path stays only for the empty-chain whole-
// scalar case where the user expects a raw ``!fir.ptr<T>`` (before
// any designate).  Inlined-callee pointer aliases get the same
// per-load designate-rewrite applied independently.
//
// Why now (before flatten-structs):
//   * Pointer declares carry ``fir.box<fir.ptr<...>>`` types.  Letting
//     them survive into flatten-structs would either inflate the
//     all-or-nothing flatten gate (every pointer member becomes a
//     non-flat member) or require treating a pointer slot as another
//     allocatable-style runtime-shape variable.  Collapsing the alias
//     here keeps flatten-structs's input clean: just declares + scalar
//     stores / loads, no ``fir.box`` indirection on what is effectively
//     a renamed reference.
//   * Downstream allocatable / pointer struct-member lowering only
//     needs to deal with TRUE runtime-shape members (POINTER /
//     ALLOCATABLE arrays as struct fields), not with name-aliasing
//     pointer locals.  Splitting these two concerns simplifies both
//     passes.
//
// ============================================================================
// I-level design  --  uniform "rebase to parent" rewrite
// ============================================================================
//
// Every Fortran pointer rebind has the same logical shape:
//
//     ptr => <parent>(<chain>)
//
// where ``<parent>`` is a single ``hlfir.declare`` (the original
// caller-side or local-side TARGET storage) and ``<chain>`` is a
// possibly-empty list of ``hlfir.designate`` steps (whole-array,
// section triplets, scalar element selection) flang lowered between
// the rebind value and the parent.  After this pass, every access
// through ``ptr`` lands on a direct ``hlfir.designate`` of the
// parent with indices merged from the chain and the access:
//
//   ``ptr => x``               =>  ``p(i, j)``     ->  ``x(i, j)``
//   ``ptr => x(2:5)``          =>  ``p(i)``        ->  ``x(i + 1)``
//   ``ptr => x(:, j)``         =>  ``p(i)``        ->  ``x(i, j)``
//   ``ptr => arr(i)`` (scalar) =>  ``p`` (scalar)  ->  ``arr(i)``
//   ``ptr => s%a`` (after flatten) =>  ``p(i)``    ->  ``s_a(i)``
//   ``ptr => s%a(2:5)`` (after flatten) => ``p(i)`` ->  ``s_a(i + 1)``
//
// All variants reduce to the same three-step rewrite.  Throughout
// this pass, ``access_indices`` means "the indices supplied by the
// Fortran source author's access through the pointer"  --  e.g.
// ``p(i, j)`` produces ``access_indices = [i, j]``.  Spelt out
// "access" rather than "user" because MLIR already has ``user`` as
// the SSA-downstream consumer of a value (``Op->getUsers()``); the
// two are unrelated and the bare term ``user`` would be
// ambiguous in a pass-level comment.
//
//   1. Walk the rebind value through ``fir.embox`` / ``fir.rebox`` /
//      ``fir.convert`` and ``hlfir.designate`` ops to find the
//      parent declare and capture each designate's
//      (indices, isTriplet) record into a CHAIN.
//   2. For every downstream access through ``fir.load %ptrDecl#0``
//       --  ``hlfir.designate`` users (array pointers) AND
//      ``fir.box_addr`` users (scalar pointers)  --  build a fresh
//      designate over the parent with merged indices.  Triplet
//      positions in the chain consume one access index (rebased
//      by ``lo - 1``); scalar positions take the chain's literal
//      value verbatim; whole-array (no chain entry) passes the
//      access indices through untouched.
//   3. Erase the load + chain (all dead after the rewrite),
//      the rebind store, the alloca / init chain, and the
//      pointer declare.
//
// Helper interface (defined below):
//
//   struct RebindChain {
//       hlfir::DeclareOp         parent;   // root TARGET declare
//       SmallVector<hlfir::DesignateOp, 2> chain;  // walks-back order:
//                                                  // outermost designate first
//   };
//
//   /// Trace a rebind value through embox/rebox/convert/designate
//   /// chains to the parent declare; returns ``parent == nullptr``
//   /// if the chain doesn't end at a declare.
//   static RebindChain traceRebindChain(mlir::Value rebindValue);
//
//   /// ``access_indices``  --  the indices supplied by the Fortran
//   /// source author's access through the pointer (e.g.
//   /// ``p(i, j)`` -> access_indices = [i, j]).  "Access"
//   /// disambiguates from MLIR's SSA ``user`` (Op->getUsers()).
//   /// Compose them with the chain's per-step indices into a
//   /// flat index list over the parent's storage.  The result
//   /// list has one entry per parent dim, ready to drop into a
//   /// new ``hlfir.designate %parent (...)`` op.  Emits any rebase
//   /// arithmetic (``access_idx + lo - 1``) at the supplied
//   /// builder / loc.  Returns false if the merge can't be
//   /// expressed (rare: section-of-section with overlapping
//   /// triplet / access-index counts that don't reconcile).
//   static bool mergeIndices(const RebindChain &c,
//                            mlir::ValueRange access_indices,
//                            mlir::OpBuilder &b, mlir::Location loc,
//                            SmallVectorImpl<mlir::Value> &out);
//
// Bail-loud guards (preflight, run BEFORE the rewrite):
//
//   * INTERLEAVED REBIND/READ  --  ``ptr => A; use; ptr => B; use``.
//     A read between two distinct rebinds observes the EARLIER
//     target; collapsing to one would lose that semantics.
//     Sequential dead-store rebinds (no reads between) are fine  --
//     the last rebind is the only observable one.
//   * BOUNDS REMAP  --  ``ptr(0:n-1) => src(1:n)``.  The user pointer's
//     lower bound differs from the section box's natural ``lo=1``.
//     Flang emits a ``fir.shift`` / ``fir.shape_shift`` operand on
//     the rebox to record the remap; forwarding silently would
//     shift every access by ``remap_lo - 1``.
//   * REBOX SLICE OPERAND  --  defensive reject (flang doesn't
//     typically emit this for pointer rebinds; would mean an
//     additional stride/section overlay we don't model).
//
// Each guard is independent of the rewrite: preflight scans the
// rebind value's chain BEFORE invoking ``traceRebindChain``.
//
// FIR/HLFIR box & shape primer (essential context for the chain
// walker)
// =============================================================
// Pointer rebinds operate on box-typed values.  The exact wrapper
// shape and shape-encoding choice flang makes determines whether
// the bridge can collapse the rebind safely.
//
//  Wrapper types (outer -> inner):
//   * ``fir.ref<T>``          --  plain pointer-to-T, no metadata.
//   * ``fir.box<T>``          --  descriptor: data pointer + shape /
//                              stride / type info.
//   * ``fir.ptr<T>``          --  Fortran POINTER indirection.
//   * ``fir.heap<T>``         --  Fortran ALLOCATABLE indirection.
//
//  Shape ops on the box/declare:
//   * ``fir.shape``        --  extents only; bounds default to 1.
//   * ``fir.shift``        --  lower bounds only (REMAP marker).
//   * ``fir.shape_shift``  --  both extents and bounds (also REMAP).
//
//  Rebind value forms collapsed into the unified path:
//   * ``embox(declare)``                           --  chain = []
//   * ``embox(designate(declare, indices))``       --  chain = [dg]
//   * ``rebox(embox(... designate ...))``          --  chain = [dg]
//                                                   (rebox is a
//                                                   metadata retag,
//                                                   bounds-preserving
//                                                   if its shape is a
//                                                   plain fir.shape)
//   * ``embox(designate(d, scalar_idx))``          --  chain = [dg]
//                                                   (element rebind:
//                                                   access_indices
//                                                   empty, chain
//                                                   provides all
//                                                   indices)
//   * ``embox(zero_bits)``                         --  initial nullify;
//                                                   skipped (no
//                                                   rebind store).
//
// Survivor declares (NOT loud failures)
// =====================================
// A pointer declare that survives this pass with live uses is
// passed through to ``extract_vars``, which gates pointer-attr
// peeling on use-emptiness  --  so a pointer declare with no live
// uses gets erased here, and one with live uses (cross-procedure
// pointer dummy, complex chained target the pass couldn't
// recognise, etc.) stays as a SCALAR passthrough downstream
// rather than a phantom rank>0 array on the SDFG signature.
// This keeps the pipeline smooth for cases the pass doesn't
// recognise without forcing them all to be loud-failures.
//
// Inlined-callee aliases
// ======================
// When ``hlfir-inline-all`` splices a module-contained call's
// body into the caller, the callee's pointer dummy declare
// becomes a fresh ``hlfir.declare %callerDecl#0 dummy_scope %dsc
// {pointer, uniq_name="..."}`` whose ``memref`` operand is the
// caller's ``ptrDecl#0``.  The unified rewrite walks each alias
// declare's loads in lockstep with the parent's, applying the
// same merge to every downstream user.
// ============================================================================

#include <optional>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// Static presence of an OPTIONAL dummy, resolved after ``hlfir-inline-all``.
/// Once a call is inlined, an omitted optional flows in as ``fir.absent`` and a
/// passed one as a concrete box/embox of the actual; only the ENTRY function's
/// own optionals stay genuinely runtime.
enum class OptPresence { Absent, Present, Unknown };

/// Trace a ``fir.is_present`` operand back through the declare / convert /
/// rebox / load / box_addr chain the inliner leaves, classifying it as
/// statically Absent (``fir.absent``), statically Present (boxed/declared from
/// a concrete address), or Unknown (a genuine runtime optional -- an entry-arg
/// with ``fir.optional``, or a value we can't prove).  Sound by construction:
/// only the two definite cases are ever returned non-Unknown.
static OptPresence traceOptionalPresence(mlir::Value v) {
  for (unsigned i = 0; i < 16 && v; ++i) {
    if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      auto* owner = ba.getOwner();
      auto func = mlir::dyn_cast_or_null<mlir::func::FuncOp>(owner->getParentOp());
      if (func && owner->isEntryBlock()) {
        // An entry-arg flagged ``fir.optional`` is the real runtime case; any
        // other entry arg is unconditionally present.
        if (func.getArgAttr(ba.getArgNumber(), "fir.optional")) return OptPresence::Unknown;
        return OptPresence::Present;
      }
      return OptPresence::Unknown;
    }
    auto* d = v.getDefiningOp();
    if (!d) return OptPresence::Unknown;
    if (mlir::isa<fir::AbsentOp>(d)) return OptPresence::Absent;
    // Boxing / addressing a concrete object proves presence.
    if (mlir::isa<fir::EmboxOp, fir::AllocaOp, fir::AllocMemOp, fir::AddrOfOp, hlfir::DesignateOp>(d))
      return OptPresence::Present;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      v = rb.getBox();
      continue;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      v = ld.getMemref();
      continue;
    }
    if (auto bx = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
      v = bx.getVal();
      continue;
    }
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      v = decl.getMemref();
      continue;
    }
    return OptPresence::Unknown;
  }
  return OptPresence::Unknown;
}

/// Fold every ``fir.if`` whose condition is a statically-resolvable
/// ``PRESENT(optional)`` down to its live branch, hoisting that branch's body
/// in front of the ``if`` and erasing the conditional.
///
/// ICON's ``_onBlock`` subset idiom rebinds a pointer in both arms of such a
/// guard (``IF (PRESENT(subset_range)) cells_subset => subset_range ELSE
/// cells_subset => patch%cells%in_domain``) -- a runtime selection between two
/// rebind targets, which the View model can't represent.  After
/// ``hlfir-inline-all`` each inlined copy's ``PRESENT`` is compile-time
/// constant, so exactly one branch is live; folding it here makes the rebind
/// straight-line (one store, its target computed BEFORE the reads it must
/// dominate) so the per-pointer rewrite below handles it normally.
///
/// Done explicitly rather than via ``canonicalize`` for two reasons: plain
/// canonicalize does NOT fold ``fir.is_present`` (the operand sits behind
/// ``hlfir.declare``, not a bare ``fir.absent``), and a general canonicalize
/// at this pipeline slot would run before ``hlfir-preserve-mutable-globals``
/// and prematurely fold mutable-global loads.  This touches only
/// ``is_present``-conditioned ``fir.if`` ops.  Fixpoint loop with a fresh walk
/// per fold so erasing a dead branch never leaves a stale ``IfOp`` handle.
static void foldPresenceGuardedIfs(mlir::func::FuncOp func) {
  while (true) {
    fir::IfOp target;
    OptPresence presence = OptPresence::Unknown;
    func.walk([&](fir::IfOp ifOp) {
      if (target) return;
      auto isPresent = mlir::dyn_cast_or_null<fir::IsPresentOp>(ifOp.getCondition().getDefiningOp());
      if (!isPresent) return;
      auto p = traceOptionalPresence(isPresent.getVal());
      if (p == OptPresence::Unknown) return;
      target = ifOp;
      presence = p;
    });
    if (!target) return;

    mlir::Region& live = (presence == OptPresence::Present) ? target.getThenRegion() : target.getElseRegion();
    if (!live.empty()) {
      mlir::Block& blk = live.front();
      if (auto* term = blk.getTerminator(); term && target.getNumResults())
        for (auto [res, val] : llvm::zip(target.getResults(), term->getOperands())) res.replaceAllUsesWith(val);
      // Hoist the live body (sans terminator) just before the ``if``, keeping
      // source order so each moved op's operands still dominate it.
      llvm::SmallVector<mlir::Operation*, 8> body;
      for (auto& o : blk.without_terminator()) body.push_back(&o);
      for (auto* o : body) o->moveBefore(target);
    }
    target.erase();
  }
}

/// One step of an ``hlfir.designate`` chain captured during a
/// rebind-value trace.  Records the indices and per-dim triplet
/// flags exactly as flang lowered them, so ``mergeIndices`` can
/// recombine them with the access (the Fortran source author's
/// ``p(i, j)``) without re-walking the IR.
struct ChainStep {
  /// The original designate op  --  kept for source-loc / verifier
  /// hints; not strictly required for the merge.
  hlfir::DesignateOp dg;
  /// Indices in HLFIR designate operand order: each triplet dim
  /// contributes 3 entries (lo, hi, step); each scalar dim
  /// contributes 1.  ``triplets[d]`` says which case ``d`` is.
  llvm::SmallVector<mlir::Value, 6> indices;
  llvm::SmallVector<bool, 4> triplets;
};

/// Output of ``traceRebindChain``: the parent declare and the chain
/// of designate steps that produced the rebind value.  ``parent``
/// is null when the trace doesn't terminate at an ``hlfir.declare``
/// (rare; the rewriter bails for those cases).
///
/// Chain order is walks-back: ``chain[0]`` is the OUTERMOST
/// designate (closest to the rebind value), ``chain.back()`` is
/// the INNERMOST (closest to the parent).  Since hlfir.designate
/// composes inside-out (``designate(designate(parent, A), B)``
/// applies B to the result of A), the OUTERMOST step is the one
/// the access (the Fortran source author's ``p(i, j)``) binds
/// against  --  its triplet positions consume access indices first.
/// ``mergeIndices`` walks the chain in outer-first order to apply
/// access indices at the right level.
struct RebindChain {
  hlfir::DeclareOp parent;
  llvm::SmallVector<ChainStep, 2> chain;
};

/// If ``t`` is a (ref / ptr / heap / box) wrapper around a DERIVED-TYPE
/// (``fir.type``) value -- and NOT an array of one -- return that record
/// type; otherwise null.  Used to distinguish a pointer rebound to a whole
/// derived-type member (``p => x%a%in_domain``, whose reads are component
/// selects ``p%c``) from a pointer rebound to an array section (whose reads
/// are index access and compose through ``mergeIndices``).  The component
/// case can't collapse to a flat index list -- it re-roots the user's
/// designate on the target ref directly.
static fir::RecordType recordRefOf(mlir::Type t) {
  mlir::Type inner = t;
  for (;;) {
    if (auto r = mlir::dyn_cast<fir::ReferenceType>(inner)) {
      inner = r.getEleTy();
      continue;
    }
    if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) {
      inner = p.getEleTy();
      continue;
    }
    if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) {
      inner = h.getEleTy();
      continue;
    }
    if (auto b = mlir::dyn_cast<fir::BaseBoxType>(inner)) {
      inner = b.getEleTy();
      continue;
    }
    break;
  }
  return mlir::dyn_cast<fir::RecordType>(inner);
}

/// True when ``slotRefTy`` is a component member SLOT holding a RECORD
/// POINTER/ALLOCATABLE: ``!fir.ref<!fir.box<!fir.ptr|heap<record>>>``.  This is
/// the ``this%member => target`` derived-type rebind shape whose reads are
/// component selects.  Array-data pointer members (``box<ptr<array<T>>>``) do
/// NOT match -- those are owned by the array / view path, never re-rooted here.
static bool isRecordPointerMemberSlot(mlir::Type slotRefTy) {
  auto ref = mlir::dyn_cast<fir::ReferenceType>(slotRefTy);
  if (!ref) return false;
  auto box = mlir::dyn_cast<fir::BaseBoxType>(ref.getEleTy());
  if (!box) return false;
  mlir::Type inner = box.getEleTy();
  if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) return mlir::isa<fir::RecordType>(p.getEleTy());
  if (auto h = mlir::dyn_cast<fir::HeapType>(inner)) return mlir::isa<fir::RecordType>(h.getEleTy());
  return false;
}

/// True when ``v`` roots at OWNED storage visible in this function -- a
/// ``fir.alloca`` / ``fir.allocmem`` or a module global (``fir.address_of``) --
/// rather than a genuine runtime dummy (a function block-argument).  Walks the
/// ``hlfir.declare`` / ``fir.convert`` / ``fir.embox`` / ``fir.box_addr`` /
/// ``fir.rebox`` chain.  After ``hlfir-inline-all`` an inlined callee's dummy
/// declare threads its ``memref`` to the caller's storage, so a ``this`` bound
/// to a LOCAL struct resolves to that local; a real entry-function dummy bottoms
/// out at a block-argument and is rejected.  Mirrors the intent of
/// ``FlattenStructs``'s ``isIndirectStructLocal`` (owned, not a rebindable
/// descriptor) but discriminates on the STORAGE ROOT rather than the outer type.
static bool rootsAtOwnedStorage(mlir::Value v) {
  for (int i = 0; i < 128 && v; ++i) {
    if (mlir::isa<mlir::BlockArgument>(v)) return false;
    auto* def = v.getDefiningOp();
    if (!def) return false;
    if (mlir::isa<fir::AllocaOp, fir::AllocMemOp, fir::AddrOfOp>(def)) return true;
    if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      v = d.getMemref();
      continue;
    }
    if (auto c = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = c.getValue();
      continue;
    }
    if (auto e = mlir::dyn_cast<fir::EmboxOp>(def)) {
      v = e.getMemref();
      continue;
    }
    if (auto b = mlir::dyn_cast<fir::BoxAddrOp>(def)) {
      v = b.getVal();
      continue;
    }
    if (auto r = mlir::dyn_cast<fir::ReboxOp>(def)) {
      v = r.getBox();
      continue;
    }
    return false;
  }
  return false;
}

/// Trace a rebind value back through ``fir.embox``/``fir.rebox``/
/// ``fir.convert``/``hlfir.designate`` ops to the originating
/// ``hlfir.declare``.  Each designate encountered is captured into
/// the chain so ``mergeIndices`` can compose them with the
/// access indices (see the term explanation on
/// ``mergeIndices``).  Returns ``parent == nullptr`` when the
/// chain hits something the rewriter doesn't model (e.g. an
/// ``hlfir.declare`` with no parent storage, or an unsupported
/// op shape).
static RebindChain traceRebindChain(mlir::Value v) {
  RebindChain out;
  for (int i = 0; i < 128 && v; ++i) {
    auto* def = v.getDefiningOp();
    if (!def) return out;
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
      v = rb.getBox();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
      v = eb.getMemref();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      // ALLOCATABLE / POINTER parent: its section designate reads a LOADED
      // box (``%b = fir.load %decl#0``).  Walk through the load to reach the
      // declare -- without this the chain stops at the load and the rebind
      // is reported as unsupported (parent == nullptr).  Same gap class as
      // the bounds-remap-view source trace in extract_vars.cpp.
      v = ld.getMemref();
      continue;
    }
    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(def)) {
      // POINTER-to-derived parent reached through ``fir.box_addr`` of a
      // loaded descriptor: ``designate(box_addr(load %ptr_decl)){comp}``.
      // The box_addr unwraps the box to its raw data pointer; walk through
      // it (like ``fir.load``) so a rebind whose target is a member of a
      // POINTER dummy (``p => patch%verts%in_domain``) reaches the parent
      // declare instead of bailing (parent == nullptr) at this op.
      v = ba.getVal();
      continue;
    }
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      ChainStep step;
      step.dg = dg;
      step.indices.assign(dg.getIndices().begin(), dg.getIndices().end());
      for (bool t : dg.getIsTriplet()) step.triplets.push_back(t);
      out.chain.push_back(std::move(step));
      v = dg.getMemref();
      continue;
    }
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      out.parent = dc;
      return out;
    }
    // Unsupported op in the chain  --  return with parent=null so
    // the caller bails.
    return out;
  }
  return out;
}

/// Compose ``access_indices`` with the chain's per-step indices
/// into a flat index list over the parent's storage, suitable for
/// ``hlfir.designate %parent (...result...)``.
///
/// ``access_indices``  --  the indices supplied by the Fortran source
/// author's access through the pointer (e.g. ``p(i, j)`` ->
/// ``access_indices = [i, j]``).  Spelt "access" rather than "user"
/// because MLIR already uses ``user`` for the SSA-downstream
/// consumer of a value (``Op->getUsers()``); the two are unrelated.
///
/// Algorithm: walk the chain INNER-first (the chain itself is
/// stored in walks-back / outer-first order, so we iterate
/// ``rbegin..rend``).  At each step:
///   * triplet positions consume one access index (with rebase);
///   * scalar  positions take the chain's literal value verbatim.
/// The output of one step becomes the access-index list for the
/// next-outer step.  After the outermost step's merge, the
/// resulting list has one entry per parent dim and is ready to
/// drop into ``hlfir.designate %parent (...)``.
///
/// In practice the bridge rarely sees chains of length > 1 because
/// ``hlfir-flatten-structs`` has already collapsed component
/// chains; the recursion is for completeness (section-of-section).
///
/// Empty chain -> ``access_indices`` pass through unchanged (the
/// whole-rebind case ``ptr => x``; ``p(i, j)`` |-> ``x(i, j)``).
///
/// Returns ``true`` when the merge is well-defined.  ``false`` when
/// triplet / access-index counts don't reconcile (rare; the
/// rewriter leaves the access alone in that case).
///
/// Index rebase: a triplet ``(lo, hi, step)`` with ``lo == 1``
/// passes the access index through unchanged.  Otherwise emits
/// ``access_idx + (lo - 1)`` as plain ``arith.addi`` over
/// ``index``.  The ``step`` and ``hi`` are not used in element
/// rebasing (they only shape extents, which DaCe gets from the
/// parent declare's own shape).
static bool mergeIndices(const RebindChain& c, mlir::ValueRange access_indices, mlir::OpBuilder& b, mlir::Location loc,
                         llvm::SmallVectorImpl<mlir::Value>& out) {
  if (c.chain.empty()) {
    for (auto v : access_indices) out.push_back(v);
    return true;
  }
  auto idxTy = b.getIndexType();
  auto toIndex = [&](mlir::Value v) {
    if (v.getType() == idxTy) return v;
    return b.create<fir::ConvertOp>(loc, idxTy, v).getResult();
  };
  auto rebase = [&](mlir::Value access_idx, mlir::Value lo) -> mlir::Value {
    // Constant-fold ``lo == 1`` to keep the IR clean.
    if (auto loCst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(lo.getDefiningOp())) {
      if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(loCst.getValue())) {
        if (a.getInt() == 1) return toIndex(access_idx);
      }
    }
    mlir::Value aIdx = toIndex(access_idx);
    mlir::Value loIdx = toIndex(lo);
    auto c1 = b.create<mlir::arith::ConstantOp>(loc, idxTy, b.getIndexAttr(1));
    auto adj = b.create<mlir::arith::SubIOp>(loc, loIdx, c1.getResult());
    return b.create<mlir::arith::AddIOp>(loc, aIdx, adj.getResult()).getResult();
  };

  // Apply each chain step in INNER-first order.  Walk
  // ``chain.back()`` (innermost) first; result becomes the input
  // for the next-outer step (towards chain[0]).  Final result is
  // the index list against the parent's storage.
  llvm::SmallVector<mlir::Value, 6> cur(access_indices.begin(), access_indices.end());
  for (auto it = c.chain.rbegin(); it != c.chain.rend(); ++it) {
    const ChainStep& s = *it;
    llvm::SmallVector<mlir::Value, 6> next;
    unsigned cursor = 0;        // walks s.indices
    unsigned accessCursor = 0;  // walks cur (the access-index list)
    for (unsigned d = 0; d < s.triplets.size(); ++d) {
      if (s.triplets[d]) {
        if (cursor + 2 >= s.indices.size() || accessCursor >= cur.size()) {
          return false;
        }
        next.push_back(rebase(cur[accessCursor], s.indices[cursor]));
        cursor += 3;  // lo, hi, step
        ++accessCursor;
      } else {
        if (cursor >= s.indices.size()) return false;
        next.push_back(toIndex(s.indices[cursor]));
        cursor += 1;
      }
    }
    if (accessCursor != cur.size()) return false;
    cur = std::move(next);
  }
  out.append(cur.begin(), cur.end());
  return true;
}

struct RewritePointerAssignsPass
    : public mlir::PassWrapper<RewritePointerAssignsPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewritePointerAssignsPass)

  llvm::StringRef getArgument() const final { return "hlfir-rewrite-pointer-assigns"; }
  llvm::StringRef getDescription() const final {
    return "Collapse Fortran ``ptr => target`` rebinds under the "
           "strict-no-aliasing assumption: every use of ``ptr`` after "
           "the rebind becomes a use of ``target``.  Pinned rule: the "
           "bridge assumes distinct names never alias.";
  }

  void runOnOperation() override {
    // First collapse any ``PRESENT(optional)``-guarded ``fir.if`` to its live
    // branch (statically known post-inline) so a pointer rebound in both arms
    // becomes a single straight-line rebind whose target dominates its reads.
    getOperation().walk([](mlir::func::FuncOp f) { foldPresenceGuardedIfs(f); });

    // Collect candidates first; rewriting mutates the IR and would
    // invalidate a fused walk.
    llvm::SmallVector<hlfir::DeclareOp, 8> ptrDecls;
    getOperation().walk([&](hlfir::DeclareOp d) {
      auto attrs = d.getFortranAttrs();
      if (!attrs) return;
      if (bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer)) ptrDecls.push_back(d);
    });

    for (auto ptrDecl : ptrDecls) rewrite(ptrDecl);

    // Second candidate class: struct-member RECORD pointer rebinds
    // (``this%member => target%...(...)``) whose base is an OWNED LOCAL struct.
    // The slot is a component-designate STORE target (not a pointer declare), so
    // neither the pointer-declare rewrite above, nor ``LiftAosPointerRecords``
    // (needs an array outer), nor the Python object-alias path (bare-identifier
    // RHS only) catches it.  Re-root each such read on the rebind target so it
    // is byte-identical to the dummy-rooted spelling the descriptor mint already
    // resolves (see rewriteMemberSlotRebind).  Collect first; rewriting mutates.
    llvm::SmallVector<fir::StoreOp, 8> memberSlotStores;
    getOperation().walk([&](fir::StoreOp st) {
      auto slot = mlir::dyn_cast_or_null<hlfir::DesignateOp>(st.getMemref().getDefiningOp());
      if (!slot || !slot.getComponent().has_value()) return;
      if (!isRecordPointerMemberSlot(slot.getResult().getType())) return;
      auto baseDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(slot.getMemref().getDefiningOp());
      if (!baseDecl || !rootsAtOwnedStorage(baseDecl.getResult(0))) return;
      // Skip the initial nullify store (``embox(zero_bits)``); the rewrite keys
      // off the real rebind and would then process the slot twice.
      auto* valDef = st.getValue().getDefiningOp();
      if (auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef))
        if (mlir::isa_and_nonnull<fir::ZeroOp>(embox.getMemref().getDefiningOp())) return;
      memberSlotStores.push_back(st);
    });
    for (auto st : memberSlotStores) rewriteMemberSlotRebind(st);

    // Sweep: any pointer declare with use_empty results after
    // the rewrites is dead  --  erase to keep extract_vars clean.
    // Pointer declares that survived with live uses are passed
    // through to extract_vars, which gates pointer-attr peeling
    // on use-emptiness so a never-collapsed pointer stays as a
    // scalar passthrough rather than a phantom rank>0 array.
    // (Cross-procedure pointer dummy rebinds, complex chained
    // targets, and other unsupported rebind shapes all flow
    // through this path; they surface as either a working
    // scalar passthrough or a clean downstream error rather than
    // a hard pass-failure here.)
    getOperation().walk([&](hlfir::DeclareOp d) {
      auto attrs = d.getFortranAttrs();
      if (!attrs) return;
      if (!bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer)) return;
      if (d.getResult(0).use_empty() && d.getResult(1).use_empty()) d.erase();
    });
  }

 private:
  void rewrite(hlfir::DeclareOp ptrDecl) {
    // Skip pointers that ``hlfir-mark-bounds-remap-views`` already
    // identified as Fortran 2003 bounds-remapping views
    // (``ptr(1:N*K) => target(:, slice)``).  Those are handled at
    // SDFG construction by emitting a DaCe ``View`` node aliasing
    // the parent array; the index-rewriting model below cannot
    // express a rank reshape and would either fail (rank-mismatch
    // in ``mergeIndices``) or, worse, silently produce wrong
    // accesses.  Leaving the rebind IR intact lets the bridge's
    // descriptors.py path consume it directly.
    if (ptrDecl->hasAttr("hlfir_bridge.bounds_remap_view")) return;

    // Find the rebind store(s): ``fir.store %targetBox to
    // %ptrDecl#0``.  Three forms:
    //   * Initial nullify (``embox(zero_bits)``): skipped.
    //   * Plain target  (``embox(declare)``):           collapse.
    //   * Slice target  (``rebox(designate(declare))``): forward.
    //
    // Loud-failure cases (we abort the pass with an emitError so
    // the bridge surfaces a clean unsupported message rather
    // than silently producing wrong code):
    //
    //   * Multiple non-nullify rebinds in scope (``ptr => A; ...;
    //     ptr => B``)  --  would silently bind every read to the
    //     FIRST rebind's target.  Same for conditional rebinds
    //     across branches.
    //   * Element-form designate target (``ptr => arr(i)``,
    //     scalar pointer rebound to one element)  --  different IR
    //     shape than the supported slice rebind.
    //   * Bounds remap (``ptr(0:n-1) => src(1:n)``)  --  flang adds
    //     a ``fir.shift`` operand on the rebox to record the
    //     remapped lower bound.  Forwarding the rebind value
    //     as-is would silently produce off-by-(remap_lo-1)
    //     indices on every read.
    // Collect non-nullify rebind stores in IR order.  Multiple
    // sequential stores before any read are fine  --  only the LAST
    // one is observable, all earlier ones are dead-store
    // rebinds.  Rebinds INTERLEAVED with reads (a read between
    // two stores) bail loudly because the bridge can't pick a
    // single coherent collapse target.
    llvm::SmallVector<fir::StoreOp, 4> nonNullifyStores;
    llvm::SmallVector<fir::LoadOp, 4> loads;
    for (auto* u : ptrDecl.getResult(0).getUsers()) {
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) loads.push_back(ld);
    }
    fir::StoreOp rebindStore;
    for (auto* u : ptrDecl.getResult(0).getUsers()) {
      auto st = mlir::dyn_cast<fir::StoreOp>(u);
      if (!st) continue;
      auto* valDef = st.getValue().getDefiningOp();

      // Skip the initial nullify on either rebind form.
      if (auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef))
        if (mlir::isa_and_nonnull<fir::ZeroOp>(embox.getMemref().getDefiningOp())) continue;

      nonNullifyStores.push_back(st);
    }

    // Order the non-nullify stores in IR-walk order so "last"
    // means last observable rebind.
    std::sort(nonNullifyStores.begin(), nonNullifyStores.end(),
              [](fir::StoreOp a, fir::StoreOp b) { return a->isBeforeInBlock(b); });

    // Interleaved-rebind detection: a read between two rebinds observes
    // the EARLIER target.  The bridge lowers a rebind as a View (or a
    // collapse) of ONE source, so this must bail loudly rather than bind
    // every read to the last target.  DOMINANCE-based, not
    // ``isBeforeInBlock``: the read may sit inside a nested ``scf`` region
    // (an ``IF`` body) where IR-order comparison is undefined and a purely
    // intra-block check silently misses it -- then the scalar/array View
    // path would wire both reads to the last source (wrong result, no
    // error).  If more than one rebind store exists and ANY load is not
    // dominated by the LAST (effective) store, that load may reach an
    // earlier rebind -> reject.  Dead-store rebinds (``p=>A; p=>B; use p``)
    // have every read after the last store, so all loads are dominated and
    // this does not fire.
    if (nonNullifyStores.size() > 1) {
      mlir::Operation* effective = nonNullifyStores.back().getOperation();
      mlir::DominanceInfo dom;
      for (auto ld : loads) {
        if (!dom.dominates(effective, ld.getOperation())) {
          ld.emitError("hlfir-rewrite-pointer-assigns: pointer ``" + ptrDecl.getUniqName().str() +
                       "`` is read between two rebind sites (interleaved "
                       "rebind)  --  the bridge lowers a rebind as a View of "
                       "a single source, so a read that may observe an "
                       "earlier target cannot be lowered.  Refactor to use "
                       "distinct pointer variables, or guard the single "
                       "rebind site behind a runtime selection.");
          signalPassFailure();
          return;
        }
      }
    }

    // Pick the LAST non-nullify store as the effective rebind.
    // Earlier stores are dead-store rebinds (no observable reads
    // between them); the alloca-store cleanup at the end of this
    // function will erase them.
    if (!nonNullifyStores.empty()) rebindStore = nonNullifyStores.back();
    if (!rebindStore) return;

    // Preflight bail-loud guards on the rebind value's chain.
    // Each guard is independent of the chain trace below so
    // unsupported shapes surface a clean error rather than a
    // miscompile.
    //   * BOUNDS REMAP  --  ``fir.rebox`` with ``fir.shift`` /
    //     ``fir.shape_shift`` operand encodes a remapped lower
    //     bound (``ptr(0:..) => src``).  Forwarding silently
    //     shifts every access by ``remap_lo - 1``.
    //   * REBOX SLICE OPERAND  --  defensive; flang doesn't emit
    //     this for pointer rebinds today, but it would mean an
    //     extra stride/section overlay we don't model.
    // Identity check for ``fir.shift`` / ``fir.shape_shift`` operands.
    // flang attaches a default ``fir.shift`` to every rebox even when the
    // user did NOT write a bounds remap (a plain ``ptr => src(..)`` with
    // no ``ptr(<lo>:..) =>`` reshape).  The bounds-remap guard below
    // must NOT fire for those: the shift carries every lower-bound as
    // either (a) the literal ``1`` (Fortran's default lb) or (b) the
    // same source box's own ``fir.box_dims %src, %i -> #0``  --  flang
    // emits the latter to re-assert the source's runtime lower bounds.
    // Both shapes are observationally identical to an unshifted rebind,
    // so the guard arms only when an lb operand is neither of those.
    auto isConstantOne = [](mlir::Value lb) {
      if (!lb) return false;
      auto* def = lb.getDefiningOp();
      if (!def) return false;
      if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
        if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return a.getInt() == 1;
      }
      return false;
    };
    auto isBoxDimsLowerBoundOfSource = [](mlir::Value lb, mlir::Value srcBox) {
      // Match the ``lb = fir.box_dims %srcBox, %i -> (#0, #1, #2)``
      // shape where ``lb`` is the result#0 (lower bound).  The middle
      // and stride results (#1, #2) are NOT lower bounds and should
      // not match -- we want the actual ``lb`` channel.
      if (!lb || !srcBox) return false;
      auto opRes = mlir::dyn_cast<mlir::OpResult>(lb);
      if (!opRes || opRes.getResultNumber() != 0) return false;
      auto bd = mlir::dyn_cast<fir::BoxDimsOp>(opRes.getOwner());
      if (!bd) return false;
      return bd.getVal() == srcBox;
    };
    auto isIdentityShift = [&](mlir::Operation* shapeDef, mlir::Value srcBox) {
      auto checkLb = [&](mlir::Value lb) { return isConstantOne(lb) || isBoxDimsLowerBoundOfSource(lb, srcBox); };
      if (auto s = mlir::dyn_cast_or_null<fir::ShiftOp>(shapeDef)) {
        for (mlir::Value lb : s.getOrigins())
          if (!checkLb(lb)) return false;
        return true;
      }
      if (auto s = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeDef)) {
        auto pairs = s.getPairs();
        for (size_t i = 0; i < pairs.size(); i += 2)
          if (!checkLb(pairs[i])) return false;
        return true;
      }
      return false;
    };

    // Extract every lower-bound operand of a ``fir.shift`` /
    // ``fir.shape_shift`` as a compile-time constant.  Returns false (leaving
    // ``out`` cleared) if any lb is non-constant -- a non-default lb we cannot
    // model as a View's access offset, so the rebind stays unsupported.
    auto constLbValue = [](mlir::Value lb) -> std::optional<int64_t> {
      if (!lb) return std::nullopt;
      auto* def = lb.getDefiningOp();
      if (!def) return std::nullopt;
      if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
        if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return a.getInt();
      return std::nullopt;
    };
    auto tryExtractConstLbs = [&](mlir::Operation* shapeDef, llvm::SmallVectorImpl<int64_t>& out) -> bool {
      out.clear();
      if (auto s = mlir::dyn_cast_or_null<fir::ShiftOp>(shapeDef)) {
        for (mlir::Value lb : s.getOrigins()) {
          auto cv = constLbValue(lb);
          if (!cv) return false;
          out.push_back(*cv);
        }
        return true;
      }
      if (auto s = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shapeDef)) {
        auto pairs = s.getPairs();
        for (size_t i = 0; i < pairs.size(); i += 2) {
          auto cv = constLbValue(pairs[i]);
          if (!cv) return false;
          out.push_back(*cv);
        }
        return true;
      }
      return false;
    };

    // Non-default constant lower bounds from a bounds-remap rebind
    // (``w(0:n-1) => src(1:n)``).  Captured during the value-chain walk
    // below; stamped as ``hlfir_bridge.pointer_view_lb`` when the rebind is
    // tagged as a View, or used to reject loudly if it falls to the rewrite.
    llvm::SmallVector<int64_t, 4> remapLbs;

    for (mlir::Value v = rebindStore.getValue(); v;) {
      auto* def = v.getDefiningOp();
      if (!def) break;
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
        if (mlir::Value shape = rb.getShape()) {
          auto* shapeDef = shape.getDefiningOp();
          bool isShift =
              mlir::isa_and_nonnull<fir::ShiftOp>(shapeDef) || mlir::isa_and_nonnull<fir::ShapeShiftOp>(shapeDef);
          if (isShift && !isIdentityShift(shapeDef, rb.getBox())) {
            // Bounds remap (``w(0:n-1) => src(1:n)``): the rebox shift
            // rebases the pointer's lower bound.  If every lb is a
            // compile-time constant we model it as a View whose access
            // offset is that lb -- capture the constants here; the tag
            // block stamps them on ``pointer_view_lb`` and descriptors.py
            // turns them into ``offset_<w>_d<d>``.  A non-constant /
            // box-derived shift can't be modelled this way -> reject.
            if (!tryExtractConstLbs(shapeDef, remapLbs)) {
              rebindStore.emitError(
                  "hlfir-rewrite-pointer-assigns: pointer "
                  "rebind with non-constant bounds remap "
                  "(``ptr(<lo>:..) => src(..)``) not supported  --  "
                  "flang encodes the remapped lower bound on the "
                  "rebox's shift operand and forwarding the rebound "
                  "box would silently shift every read by "
                  "``remap_lo - 1``.");
              signalPassFailure();
              return;
            }
          }
        }
        if (rb.getSlice()) {
          rebindStore.emitError(
              "hlfir-rewrite-pointer-assigns: pointer "
              "rebind with rebox slice operand not "
              "supported.");
          signalPassFailure();
          return;
        }
        v = rb.getBox();
        continue;
      }
      // Literal-extent section rebind (``w(0:9) => src(1:10)`` with a
      // compile-time ``src(10)``) lowers via ``fir.embox`` carrying a
      // ``fir.shape_shift`` rather than a ``fir.rebox``.  Same bounds-remap
      // capture as the rebox arm: a non-identity shift's constant lb(s)
      // become the view's access offset.  Embox is the leaf of the rebind
      // value, so stop after inspecting it.
      if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
        if (mlir::Value shape = eb.getShape()) {
          auto* shapeDef = shape.getDefiningOp();
          bool isShift =
              mlir::isa_and_nonnull<fir::ShiftOp>(shapeDef) || mlir::isa_and_nonnull<fir::ShapeShiftOp>(shapeDef);
          // A plain whole-array rebind onto an allocatable/pointer member
          // (``my_arr2 => my_arr%w``) is NOT a bounds remap: flang emboxes
          // ``fir.box_addr %srcBox`` and re-asserts the source's OWN runtime
          // lower bounds via ``fir.box_dims %srcBox`` on the shape_shift.
          // That shift is an IDENTITY (lb = source's lb), so recover the
          // source box behind the embox'd address and hand it to
          // isIdentityShift -- mirroring the rebox arm's ``rb.getBox()`` --
          // otherwise the box_dims-of-source lbs read as a non-constant
          // remap and the rebind is wrongly rejected.
          mlir::Value srcBox;
          for (mlir::Value mem = eb.getMemref(); mem;) {
            auto* mdef = mem.getDefiningOp();
            if (!mdef) break;
            if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(mdef)) {
              srcBox = ba.getVal();
              break;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(mdef)) {
              mem = cv.getValue();
              continue;
            }
            break;
          }
          if (isShift && !isIdentityShift(shapeDef, srcBox)) {
            if (!tryExtractConstLbs(shapeDef, remapLbs)) {
              rebindStore.emitError(
                  "hlfir-rewrite-pointer-assigns: pointer "
                  "rebind with non-constant bounds remap "
                  "(``ptr(<lo>:..) => src(..)``) not supported.");
              signalPassFailure();
              return;
            }
          }
        }
        break;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
        v = cv.getValue();
        continue;
      }
      break;
    }

    // Trace the rebind value to (parent, chain).  Bail out if the
    // chain doesn't terminate at an ``hlfir.declare``  --  leave the
    // pointer declare alive; downstream extract_vars treats it as
    // a scalar passthrough.
    RebindChain chain = traceRebindChain(rebindStore.getValue());
    if (!chain.parent) return;

    // Inlined-callee alias collapse: any other pointer declare in
    // the function whose memref traces back to ``ptrDecl`` (via
    // ``hlfir.declare`` chain) is an alias of the same storage  --
    // typically the inlined dummy of a module-contained call that
    // received our pointer as an argument.  Without redirecting
    // its uses, the alias's loads stay live, extract_vars surfaces
    // it as an independent rank>0 array, and the SDFG ends up
    // demanding extra ``<alias>_d0`` symbols.  Collect them now
    // so the rewrite below redirects their loads in lockstep.
    llvm::SmallVector<hlfir::DeclareOp, 4> aliasDecls;
    if (auto func = ptrDecl->getParentOfType<mlir::func::FuncOp>()) {
      func.walk([&](hlfir::DeclareOp other) {
        if (other == ptrDecl) return;
        auto attrs = other.getFortranAttrs();
        if (!attrs) return;
        if (!bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer)) return;
        // Walk other.getMemref() back through hlfir.declare /
        // fir.convert chain and check if it reaches ptrDecl's
        // results.
        mlir::Value mr = other.getMemref();
        for (int i = 0; i < 128 && mr; ++i) {
          if (mr == ptrDecl.getResult(0) || mr == ptrDecl.getResult(1)) {
            aliasDecls.push_back(other);
            return;
          }
          auto* d = mr.getDefiningOp();
          if (!d) return;
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            mr = cv.getValue();
            continue;
          }
          if (auto inner = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
            mr = inner.getMemref();
            continue;
          }
          return;
        }
      });
    }

    // ----- P3: lower a plain SECTION rebind as a DaCe View -----
    // A same-rank section rebind (``p => a(:, j)`` / ``p => store(2:5)``
    // / ``p => a(:,:,k)``) aliases a slab of the parent.  Rather than the
    // index-rewrite below, TAG the pointer declare so ``extract_vars``
    // emits a ``view_alias`` (reusing the bounds-remap-view trace + the
    // view_alias stride path, which gets NON-packed strides right).
    // Scoped to the single-designate, has-a-triplet, no-inlined-alias
    // shape; the other shapes (whole-array, scalar element, struct
    // member, inlined alias) still rewrite below and migrate next.
    if (aliasDecls.empty()) {
      bool tagAsView = false;
      bool tagScalarView = false;
      if (chain.chain.size() == 1) {
        // Section rebind (``p => a(:, j)``): a single designate with at
        // least one triplet dim.
        for (bool t : chain.chain[0].triplets)
          if (t) {
            tagAsView = true;
            break;
          }
      } else if (chain.chain.empty()) {
        // Whole-array rebind (``p => a`` / ``p => s%w``): no designate,
        // the view spans the entire parent.  Only for an ARRAY pointer --
        // a scalar pointer (``tmp => x`` / ``tmp => arr(i)``) is a
        // separate not-yet-migrated shape and still rewrites below.
        mlir::Type t = ptrDecl.getResult(0).getType();
        for (int i = 0; i < 6; ++i) {
          if (auto r = mlir::dyn_cast<fir::ReferenceType>(t)) {
            t = r.getEleTy();
            continue;
          }
          if (auto box = mlir::dyn_cast<fir::BoxType>(t)) {
            t = box.getEleTy();
            continue;
          }
          if (auto p = mlir::dyn_cast<fir::PointerType>(t)) {
            t = p.getElementType();
            continue;
          }
          if (auto h = mlir::dyn_cast<fir::HeapType>(t)) {
            t = h.getElementType();
            continue;
          }
          break;
        }
        if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
          if (seq.getDimension() > 0) tagAsView = true;
        } else if (!mlir::isa<fir::RecordType>(t)) {
          // Scalar pointer rebind (``tmp => x``, ``x`` a scalar TARGET):
          // the pointer peeled to a non-array element type.  Lower as a
          // length-1-array View of the target -- extract_vars emits ``tmp``
          // as a length-1 ``view_alias`` and re-classifies the target as a
          // length-1 Array (a View cannot alias a const Scalar source).
          tagScalarView = true;
        }
        // A whole-RECORD pointer rebind (``params_oce => v_params``) is neither
        // an array View nor a scalar View -- its reads/writes are ``p % member``
        // component selects.  Leave it UNTAGGED so the rewrite path's
        // ``recordTarget`` re-roots each ``p % m`` onto the target record
        // (the empty-chain case handled alongside ``p => x % a % member``).
      }
      if (tagAsView) {
        ptrDecl->setAttr("hlfir_bridge.pointer_view", mlir::UnitAttr::get(&getContext()));
        // Bounds-remap lower bound(s) captured during the value-chain
        // walk (``w(0:n-1) => src(1:n)``): forward them to extract_vars,
        // which surfaces them as the view's ``lower_bounds`` so
        // descriptors.py stamps ``offset_<w>_d<d> = lb``.  Absent for a
        // default-lb section rebind (identity shift -> remapLbs empty).
        if (!remapLbs.empty())
          ptrDecl->setAttr("hlfir_bridge.pointer_view_lb", mlir::DenseI64ArrayAttr::get(&getContext(), remapLbs));
        return;
      }
      if (tagScalarView) {
        ptrDecl->setAttr("hlfir_bridge.pointer_view_scalar", mlir::UnitAttr::get(&getContext()));
        return;
      }
    }

    // A non-default lower-bound rebind we captured but could NOT tag as a
    // View (scalar pointer, inlined-alias, or other not-yet-migrated shape)
    // cannot be index-rewritten without silently shifting every read by
    // ``lb - 1`` -- preserve the loud-failure contract.
    if (!remapLbs.empty()) {
      rebindStore.emitError(
          "hlfir-rewrite-pointer-assigns: pointer rebind with bounds "
          "remap (``ptr(<lo>:..) => src(..)``) not supported for this "
          "rebind shape (only array section / whole-array rebinds lower "
          "as Views).");
      signalPassFailure();
      return;
    }

    ptrDecl.emitWarning() << "hlfir-rewrite-pointer-assigns: collapsing pointer "
                          << "rebind ``" << ptrDecl.getUniqName().str() << " => " << chain.parent.getUniqName().str()
                          << "(...chain...)`` under the strict-no-aliasing "
                          << "assumption.  Every access through the pointer is "
                          << "rewritten to a direct designate of the parent's "
                          << "storage; if your program relies on alias semantics "
                          << "this rewrite is unsafe.";

    // Unified rewrite: for every ``fir.load %ptrDecl#0`` (and
    // every load through an aliased pointer declare  --  the
    // inlined-callee shape), walk its users and rewrite each:
    //
    //   * ``hlfir.designate %loaded (access_indices)``
    //     -> ``hlfir.designate %parent (mergeIndices(chain,
    //                                               access_indices))``
    //   * ``fir.box_addr %loaded``
    //     -> ``%parent.getResult(0)`` if chain is empty (whole
    //                                rebind), else a direct
    //                                designate over parent
    //                                using the chain's indices
    //                                with no access-index
    //                                contribution (element
    //                                rebind / scalar view).
    //                                Any type mismatch with
    //                                the box_addr's result
    //                                is bridged with a
    //                                ``fir.convert``.
    //
    // SSA dominance: load sites use the loaded box AFTER the
    // store, so substituting the store's input value is
    // dominance-correct for any load that comes after the
    // store.  Loads BEFORE the rebind would be reads of an
    // unbound pointer  --  undefined behaviour we don't model.
    llvm::SmallVector<mlir::Operation*, 8> deadReaders;

    // Helper closure: rewrites all users of one load.
    auto rewriteLoadUsers = [&](fir::LoadOp ld) {
      // Skip loads that happen BEFORE the rebind in the same
      // block (reads of an unbound pointer).  Loads in nested
      // blocks (typical: inside a do_loop body) report as
      // "different blocks"  --  treat them as "after" since they
      // can only execute after the enclosing block reaches
      // the loop.
      if (ld->getBlock() == rebindStore->getBlock() && ld->isBeforeInBlock(rebindStore)) return;
      // When the pointer aliases a whole DERIVED-TYPE member
      // (``p => x%a%in_domain``), the rebind target is a record ref, not an
      // array, and its reads are component selects (``p%c``) that can't
      // compose through ``mergeIndices`` (which collapses array sections to a
      // flat index list and would drop the component).  Capture the target
      // ref directly: the outermost chain designate's result is the
      // already-computed ``x%a%in_domain``, which dominates every read after
      // the rebind store and survives cleanup (``deadReaders`` is
      // use_empty-swept and we add a live user below).  Reads then re-root on
      // this target instead of on the parent's index storage.
      mlir::Value recordTarget;
      if (!chain.chain.empty() && recordRefOf(chain.chain.front().dg.getResult().getType()))
        recordTarget = chain.chain.front().dg.getResult();
      // Whole-RECORD pointer rebind (``params_oce => v_params``, empty chain):
      // the target IS the parent declare (a module global / dummy record), and
      // ``p % member`` reads/writes re-root onto it -- the member spelling then
      // matches the marshalled ``v_params % member`` used directly elsewhere.
      else if (chain.chain.empty() && chain.parent && recordRefOf(ptrDecl.getResult(0).getType()))
        recordTarget = chain.parent.getResult(0);
      // A pointer rebound to an ARRAY member reached through a
      // MULTI-designate struct-member chain (``p => patch_3d %
      // p_patch_1d(1) % dolic_c`` whole, or ``... % zdistance(:, :, blk)``
      // section).  ``mergeIndices`` + ``chain.parent`` can't handle these:
      // the parent is the ROOT struct, not an indexable array, so a flat
      // ``designate %patch_3d (...)`` is invalid.  Re-root reads on the
      // MEMBER instead -- the exact spelling the struct flatten + marshal
      // path already lowers for a direct ``s % m(...)`` access elsewhere in
      // the kernel -- so the pointer collapses with no leftover transient.
      // Two shapes, keyed on the outermost (section) designate:
      //   * whole member (no triplet): its result is the member's box SLOT
      //     (``!fir.ref<!fir.box<...>>``), identical in type to the pointer's
      //     own slot -- redirect the pointer LOAD to it; element designates
      //     downstream read the member box unchanged.
      //   * section (>=1 triplet): its MEMREF is the addressed member box;
      //     compose ONLY that section's indices with the access indices
      //     (the fixed ``blk`` / ``jc`` selectors ride along) so each read
      //     ``p(i, j)`` becomes a single ``designate(member, i, j, blk)``.
      // ``recordRefOf`` (whole-RECORD member) is taken above; only ARRAY
      // members reach here.  A plain single-designate section / whole-array
      // rebind is already a P3 View (tagged + returned before the rewrite),
      // so only STRUCT-MEMBER designate chains (``p => s % m`` /
      // ``p => s % m(:,:,blk)``, any depth incl. a lone member designate)
      // reach here and need re-rooting.
      mlir::Value sectionMemberBox;  // section arm: loaded member box (s % m)
      mlir::Value wholeMemberSlot;   // whole  arm: member box slot (ref<box>)
      if (!recordTarget && !chain.chain.empty()) {
        ChainStep& front = chain.chain.front();
        bool frontHasTriplet = false;
        for (bool t : front.triplets)
          if (t) {
            frontHasTriplet = true;
            break;
          }
        if (frontHasTriplet) {
          sectionMemberBox = front.dg.getMemref();
        } else if (auto ref = mlir::dyn_cast<fir::ReferenceType>(front.dg.getResult().getType())) {
          // Whole-member: the component designate yields the member's box
          // SLOT (``!fir.ref<!fir.box<...>>``).  Accept any box element
          // class (``ptr`` vs ``heap`` may differ from the pointer's own
          // ``ptr`` slot) -- the reads re-root on a fresh load of it below.
          // Restricted to an ARRAY member (the leak class); a scalar-member
          // pointer keeps its own (scalar-view / element-rebind) path.
          if (auto box = mlir::dyn_cast<fir::BaseBoxType>(ref.getEleTy())) {
            mlir::Type inner = box.getEleTy();
            if (auto p = mlir::dyn_cast<fir::PointerType>(inner)) inner = p.getElementType();
            else if (auto h = mlir::dyn_cast<fir::HeapType>(inner))
              inner = h.getElementType();
            if (mlir::isa<fir::SequenceType>(inner)) wholeMemberSlot = front.dg.getResult();
          }
        }
      }
      auto retagTo = [](mlir::OpBuilder& b, mlir::Location loc, mlir::Value v, mlir::Type want) -> mlir::Value {
        if (v.getType() == want) return v;
        return b.create<fir::ConvertOp>(loc, want, v).getResult();
      };

      // Whole-member rebind: load the member's box ONCE, right before this
      // pointer load (the member-slot designate at the rebind dominates it),
      // so every element designate below re-roots on the member box.  A fresh
      // load (not an operand redirect) so a ``heap`` member box binds cleanly
      // to a ``ptr`` pointer read -- the redirect would leave the load's
      // result typed for the wrong box class.
      mlir::Value wholeMemberBox;
      if (wholeMemberSlot) {
        mlir::OpBuilder b(ld);
        wholeMemberBox = b.create<fir::LoadOp>(ld.getLoc(), wholeMemberSlot).getResult();
      }

      // Snapshot users  --  we rewrite in place and the user
      // list mutates as we go.
      llvm::SmallVector<mlir::Operation*, 4> userOps;
      for (auto* uu : ld.getResult().getUsers()) userOps.push_back(uu);
      for (auto* uu : userOps) {
        if (auto userDg = mlir::dyn_cast<hlfir::DesignateOp>(uu)) {
          mlir::OpBuilder b(userDg);
          auto loc = userDg.getLoc();
          // Record-member target: never run the index-merge path (it drops
          // the component and emits an invalid designate).  Re-root the
          // designate on the target ref directly -- but only when the user's
          // memref type matches the target exactly, so the clone verifies
          // (the common record read goes load -> box_addr -> designate, where
          // the component designate is a user of box_addr, handled below; a
          // direct designate on the loaded box is left alone rather than
          // risk a mistyped op).
          if (recordTarget) {
            if (userDg.getMemref().getType() == recordTarget.getType()) {
              auto* cloned = b.clone(*userDg.getOperation());
              cloned->setOperand(0, recordTarget);
              userDg.getResult().replaceAllUsesWith(cloned->getResult(0));
              deadReaders.push_back(userDg);
            }
            continue;
          }
          // Whole-member target: re-root the read on the loaded member box
          // with the SAME indices -- ``p(i, j, k)`` becomes
          // ``designate(s % m, i, j, k)``.  No index composition (a whole
          // rebind preserves rank and bounds).
          if (wholeMemberBox) {
            auto wDg = b.create<hlfir::DesignateOp>(loc,
                                                    /*result_type=*/userDg.getResult().getType(),
                                                    /*memref=*/wholeMemberBox,
                                                    /*indices=*/userDg.getIndices());
            userDg.getResult().replaceAllUsesWith(wDg.getResult());
            deadReaders.push_back(userDg);
            continue;
          }
          // Array-section member target: compose ONLY the outermost section
          // with the access indices and re-root on the member box, so
          // ``p(i, j)`` becomes ``designate(s % m, i, j, blk)`` -- a single
          // designate on the already-flattenable member.
          if (sectionMemberBox) {
            RebindChain sectionOnly;
            sectionOnly.chain.push_back(chain.chain.front());
            llvm::SmallVector<mlir::Value, 6> secMerged;
            if (!mergeIndices(sectionOnly, userDg.getIndices(), b, loc, secMerged)) continue;
            auto secDg = b.create<hlfir::DesignateOp>(loc,
                                                      /*result_type=*/userDg.getResult().getType(),
                                                      /*memref=*/sectionMemberBox,
                                                      /*indices=*/mlir::ValueRange{secMerged});
            userDg.getResult().replaceAllUsesWith(secDg.getResult());
            deadReaders.push_back(userDg);
            continue;
          }
          llvm::SmallVector<mlir::Value, 6> merged;
          if (!mergeIndices(chain, userDg.getIndices(), b, loc, merged))
            continue;  // leave userDg alive; bail-loud
                       // path or downstream surfaces
                       // the unsupported shape
          auto newDg = b.create<hlfir::DesignateOp>(loc,
                                                    /*result_type=*/userDg.getResult().getType(),
                                                    /*memref=*/chain.parent.getResult(0),
                                                    /*indices=*/mlir::ValueRange{merged});
          userDg.getResult().replaceAllUsesWith(newDg.getResult());
          deadReaders.push_back(userDg);
          continue;
        }
        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(uu)) {
          // A section-member target composes indices per element designate;
          // a whole-array ``box_addr`` of it has no single flat rewrite, so
          // leave it (the surviving pointer keeps the read resolvable).  The
          // ``chain.parent`` path below is invalid for a struct-member chain
          // (parent is the root struct), so it must not run here.
          if (sectionMemberBox) continue;
          mlir::OpBuilder b(ba);
          auto loc = ba.getLoc();
          mlir::Value replacement;
          // Whole-member target: ``box_addr`` of the member box gives the
          // member's raw data address directly.
          if (wholeMemberBox) {
            replacement = retagTo(b, loc, b.create<fir::BoxAddrOp>(loc, wholeMemberBox).getResult(),
                                  ba.getResult().getType());
            ba.getResult().replaceAllUsesWith(replacement);
            deadReaders.push_back(ba);
            continue;
          }
          // Record-member target: box_addr of the loaded descriptor yields
          // the member's raw address = the target ref (retagged to the
          // box_addr's result type).  Downstream component designates then
          // read ``target%c`` directly.
          if (recordTarget) {
            replacement = retagTo(b, loc, recordTarget, ba.getResult().getType());
            ba.getResult().replaceAllUsesWith(replacement);
            deadReaders.push_back(ba);
            continue;
          }
          if (chain.chain.empty()) {
            // Whole rebind  --  box_addr resolves directly
            // to the parent's ref.
            replacement = chain.parent.getResult(0);
          } else {
            // Chained rebind  --  build a designate over
            // the parent using the chain's own indices
            // with no access-index contribution.  This
            // is the element-rebind shape
            // (``ptr => arr(i)``) and the rare
            // scalar-view-of-section.
            llvm::SmallVector<mlir::Value, 6> merged;
            if (!mergeIndices(chain, /*access_indices=*/{}, b, loc, merged)) continue;
            // Result type follows the original box_addr's
            // result; designate yields a ref into the
            // parent's storage at the chain's position.
            replacement = b.create<hlfir::DesignateOp>(loc, ba.getResult().getType(), chain.parent.getResult(0),
                                                       mlir::ValueRange{merged})
                              .getResult();
          }
          if (replacement.getType() != ba.getResult().getType()) {
            replacement = b.create<fir::ConvertOp>(loc, ba.getResult().getType(), replacement);
          }
          ba.getResult().replaceAllUsesWith(replacement);
          deadReaders.push_back(ba);
          continue;
        }
        if (auto cin = mlir::dyn_cast<hlfir::CopyInOp>(uu)) {
          // A pointer rebound to a CONTIGUOUS whole target (empty chain)
          // passed to a contiguous-dummy callee (e.g. a ``bind(c)`` external)
          // is wrapped by Flang in ``copy_in`` / ``copy_out`` around a temp.
          // The copy is unnecessary here: fold each ``box_addr`` of the copied
          // box straight to the parent's ref and drop the ``copy_in`` /
          // ``copy_out`` pair, so the callee reads / writes the target in
          // place (the external then connects to the target array, not an
          // unnameable copy buffer).  A chained / sectioned rebind keeps its
          // copy  --  that may be a genuine non-contiguous materialisation.
          llvm::SmallVector<mlir::Operation*, 2> boxUsers(cin.getResult(0).getUsers().begin(),
                                                          cin.getResult(0).getUsers().end());
          // Only fold when the rebind is a whole contiguous target AND every
          // use of the copied box is a box_addr (so eliding the copy is
          // complete -- a non-box_addr use would dangle once copy_in/out go).
          bool foldable = chain.chain.empty() && !boxUsers.empty();
          for (auto* cu : boxUsers)
            if (!mlir::isa<fir::BoxAddrOp>(cu)) {
              foldable = false;
              break;
            }
          if (foldable) {
            for (auto* cu : boxUsers) {
              auto ba = mlir::cast<fir::BoxAddrOp>(cu);
              mlir::OpBuilder b(ba);
              // box_addr wants the target's raw data address; the declare's
              // result #1 is that memref (result #0 may be a dynamic-extent
              // box, which does not fir.convert to a ref/ptr).
              mlir::Value replacement = chain.parent.getResult(1);
              if (replacement.getType() != ba.getResult().getType())
                replacement = b.create<fir::ConvertOp>(ba.getLoc(), ba.getResult().getType(), replacement);
              ba.getResult().replaceAllUsesWith(replacement);
              deadReaders.push_back(ba);
            }
            // Drop the matching copy_out (it consumes the copy_in flag).
            llvm::SmallVector<mlir::Operation*, 2> flagUsers(cin.getResult(1).getUsers().begin(),
                                                             cin.getResult(1).getUsers().end());
            for (auto* fu : flagUsers)
              if (mlir::isa<hlfir::CopyOutOp>(fu)) deadReaders.push_back(fu);
            deadReaders.push_back(cin);
          }
          continue;
        }
        // Other user shapes (rare)  --  leave alone.  The
        // surviving load + ptr declare keep them
        // resolvable downstream as a scalar passthrough.
      }
      // Always queue the load  --  use_empty is checked at
      // sweep time.  Without this, the load is checked here
      // while user ops still reference it and is never
      // pushed for erase, leaving the pointer declare with a
      // live user.
      deadReaders.push_back(ld);
    };

    // Snapshot loads of the primary pointer declare.
    llvm::SmallVector<fir::LoadOp, 4> snapshotLoads;
    for (auto* u : ptrDecl.getResult(0).getUsers())
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) snapshotLoads.push_back(ld);
    for (auto ld : snapshotLoads) rewriteLoadUsers(ld);

    // Same walk for each aliased pointer declare  --  every load
    // returns the same box value we just rewrote, so rewriting
    // its users via the same chain lands them on the rebound
    // parent too.
    llvm::SmallVector<hlfir::DeclareOp, 4> aliasesToErase;
    for (auto alias : aliasDecls) {
      llvm::SmallVector<fir::LoadOp, 4> aliasLoads;
      for (auto* u : alias.getResult(0).getUsers())
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) aliasLoads.push_back(ld);
      for (auto ld : aliasLoads) rewriteLoadUsers(ld);
      aliasesToErase.push_back(alias);
    }

    // Erase user ops first (they hold the only uses on each
    // load), then the loads themselves, in iteration order.
    // ``op->use_empty()`` at the moment of erase decides whether
    // each is safe to drop.
    for (auto* op : deadReaders)
      if (op->use_empty()) op->erase();

    // Erase use-empty alias declares.
    for (auto alias : aliasesToErase) {
      if (alias.getResult(0).use_empty() && alias.getResult(1).use_empty()) alias.erase();
    }

    // Erase the rebind store + the entire alloca/init chain feeding
    // ptrDecl.  The alloca's only remaining users at this point are
    // the ``hlfir.declare`` itself and the (now dead) initial
    // nullify chain.
    rebindStore.erase();

    mlir::Value ptrAlloca = ptrDecl.getMemref();
    // Sweep dead init ops in reverse so each erase's only users are
    // already gone.  Pattern:
    //   %a = fir.alloca
    //   %z = fir.zero_bits
    //   %e = fir.embox %z
    //   fir.store %e to %a   -- initial nullify
    //   hlfir.declare %a
    llvm::SmallVector<mlir::Operation*, 4> deadInit;
    for (auto* u : ptrAlloca.getUsers()) {
      if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) deadInit.push_back(st);
    }
    for (auto* op : deadInit) op->erase();

    if (ptrDecl.getResult(0).use_empty() && ptrDecl.getResult(1).use_empty()) ptrDecl.erase();

    // Sweep the dangling embox + zero_bits + alloca if no users
    // remain.
    if (auto* def = ptrAlloca.getDefiningOp())
      if (def->use_empty()) def->erase();
  }

  /// Re-root reads of a struct-member RECORD pointer rebind
  /// (``this%member => target%...(...)``) onto the rebind target.
  ///
  /// flang lowers ``this%member`` (a POINTER derived-type slot) as
  /// ``load(slot)`` -> ``box_addr`` -> component designates, all rooted at the
  /// LOCAL struct.  The bridge's descriptor mint keys on the DUMMY-rooted
  /// spelling (``patch_3d%p_patch_2d(1)%cells%max_connectivity``), so the
  /// local-rooted read (``free_sfc_solver_lhs%patch_2d%cells%max_connectivity``)
  /// never matches and its leaf scalar surfaces as a free symbol.  The rebind
  /// target designate (``chain.front``) is the SAME storage as the dummy-rooted
  /// reads and, being computed at the rebind site, dominates every read after
  /// it.  Re-rooting each read on that target makes it byte-for-byte the
  /// already-resolved spelling, so the existing mint fires unchanged and binds
  /// the REAL value (bit-exact -- no new value is introduced).
  ///
  /// Conservative by construction: only the single-static-rebind, RECORD-target,
  /// fully-dominated shape is re-rooted; every other shape (array-section /
  /// whole-pointer-dummy / interleaved / non-dominated / unmodelled reader) is
  /// left EXACTLY as-is.  This path never emits a new loud failure, so a
  /// currently-building program cannot regress into a build error.
  void rewriteMemberSlotRebind(fir::StoreOp store) {
    auto slot0 = mlir::dyn_cast_or_null<hlfir::DesignateOp>(store.getMemref().getDefiningOp());
    if (!slot0 || !slot0.getComponent().has_value()) return;
    llvm::StringRef comp = slot0.getComponent()->getValue();

    // Gather every designate of the same (base, component).  flang usually CSEs
    // the store target and the read into one designate, but tolerate duplicates
    // so no read is left dangling on an unbound slot after the store is erased.
    mlir::Value base = slot0.getMemref();
    llvm::SmallVector<hlfir::DesignateOp, 2> slotDesignates;
    for (auto* u : base.getUsers())
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u)) {
        auto c = dg.getComponent();
        if (c.has_value() && c->getValue() == comp) slotDesignates.push_back(dg);
      }

    llvm::SmallVector<fir::StoreOp, 4> slotStores;
    llvm::SmallVector<fir::StoreOp, 4> nonNullifyStores;
    llvm::SmallVector<fir::LoadOp, 4> loads;
    for (auto dg : slotDesignates)
      for (auto* u : dg.getResult().getUsers()) {
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(u)) {
          loads.push_back(ld);
        } else if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) {
          slotStores.push_back(st);
          auto* valDef = st.getValue().getDefiningOp();
          if (auto embox = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef))
            if (mlir::isa_and_nonnull<fir::ZeroOp>(embox.getMemref().getDefiningOp())) continue;  // nullify
          nonNullifyStores.push_back(st);
        } else {
          return;  // unmodelled slot user -- decline, leave the rebind untouched
        }
      }

    // Only the single observable rebind is handled.  A second observable rebind
    // would need the interleaved-read analysis the pointer-declare path runs;
    // here we simply decline (leaving the IR as it currently builds).
    if (nonNullifyStores.size() != 1) return;
    fir::StoreOp rebindStore = nonNullifyStores.front();

    // Every read must be dominated by the rebind store, so re-rooting each is
    // sound and the store is safe to erase.  A read not dominated (an earlier
    // block, or a read of the still-unbound slot) -> decline, no mutation.
    mlir::DominanceInfo dom;
    for (auto ld : loads)
      if (!dom.dominates(rebindStore.getOperation(), ld.getOperation())) return;

    // Trace the stored box to (parent, chain).  Only a RECORD-target rebind --
    // whose outermost chain designate yields a derived-type ref (e.g.
    // ``patch_3d%p_patch_2d(1)``) -- re-roots here.  A whole-pointer-dummy
    // rebind (empty chain) or an array-section target is declined: those reads
    // are not plain component selects and need the array / view path.
    RebindChain chain = traceRebindChain(rebindStore.getValue());
    if (!chain.parent || chain.chain.empty()) return;
    mlir::Value recordTarget = chain.chain.front().dg.getResult();
    if (!recordRefOf(recordTarget.getType())) return;

    // Validate BEFORE mutating: every read reaches the record either through a
    // ``fir.box_addr`` of the loaded slot, or (rarely) a component designate
    // directly on the loaded box whose memref type matches the target.
    for (auto ld : loads)
      for (auto* uu : ld.getResult().getUsers()) {
        if (mlir::isa<fir::BoxAddrOp>(uu)) continue;
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(uu))
          if (dg.getMemref().getType() == recordTarget.getType()) continue;
        return;  // unmodelled read shape -- decline (nothing mutated yet)
      }

    // Commit.  Re-root each read on the target designate; this mirrors the
    // record-member re-root the pointer-declare path applies in its load-user
    // rewrite (box_addr -> target ref; direct component designate -> cloned
    // onto the target).
    llvm::SmallVector<mlir::Operation*, 8> deadReaders;
    for (auto ld : loads) {
      llvm::SmallVector<mlir::Operation*, 4> userOps(ld.getResult().getUsers().begin(),
                                                     ld.getResult().getUsers().end());
      for (auto* uu : userOps) {
        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(uu)) {
          mlir::OpBuilder b(ba);
          mlir::Value repl = recordTarget;
          if (repl.getType() != ba.getResult().getType())
            repl = b.create<fir::ConvertOp>(ba.getLoc(), ba.getResult().getType(), repl).getResult();
          ba.getResult().replaceAllUsesWith(repl);
          deadReaders.push_back(ba);
        } else if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(uu)) {
          mlir::OpBuilder b(dg);
          auto* cloned = b.clone(*dg.getOperation());
          cloned->setOperand(0, recordTarget);
          dg.getResult().replaceAllUsesWith(cloned->getResult(0));
          deadReaders.push_back(dg);
        }
      }
      deadReaders.push_back(ld);
    }
    for (auto* op : deadReaders)
      if (op->use_empty()) op->erase();

    // Erase the rebind + nullify store(s) and their now-dead value chain, then
    // the dead slot designates.  The target designate (``recordTarget``) stays
    // live -- the re-rooted reads reference it.
    for (auto st : slotStores) {
      mlir::Value storedVal = st.getValue();
      st.erase();
      if (auto* def = storedVal.getDefiningOp())
        if (def->use_empty()) def->erase();
    }
    for (auto dg : slotDesignates)
      if (dg.getResult().use_empty()) dg.erase();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createRewritePointerAssignsPass() { return std::make_unique<RewritePointerAssignsPass>(); }

}  // namespace hlfir_bridge
