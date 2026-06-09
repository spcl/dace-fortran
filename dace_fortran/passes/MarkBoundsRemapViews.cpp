// ============================================================================
// MarkBoundsRemapViews.cpp  --  Tag Fortran 2003 bounds-remapping pointer
//                              assignments as SDFG-level Views.
// ============================================================================
// Problem:
//     Fortran 2003 allows a pointer assignment to bundle a *rank reshape*
//     with an *explicit lower bound*:
//
//       COMPLEX(8), POINTER :: prhoc_d(:)
//       prhoc_d(1 : N*K) => rhoc_d(:, slice_lo : slice_lo+K-1)
//       CALL fwfft('Rho', prhoc_d, dfftt, howmany=K)
//
//     The right-hand side is a rank-2 slice of ``rhoc_d``; the left-hand
//     side reinterprets it as a rank-1 flat view of the same memory.
//     This is the "reshape-as-pointer" trick QE uses to feed a 2-D
//     buffer into a 1-D FFT routine without copying.  Two operations
//     are bundled:
//
//       1. Rank reshape (rank-N source -> rank-1 LHS view).
//       2. Bounds remap (LHS gets the explicit ``(1 : N*K)`` bounds).
//
//     Crucially this is a *view*, not a copy: writes through ``prhoc_d``
//     must propagate back to ``rhoc_d``.  Fortran's ``=>`` is alias
//     semantics, unambiguous in the source.
//
//     The existing ``hlfir-rewrite-pointer-assigns`` pass rejects this
//     case because its index-rewriting model can't recompose a rank-1
//     access into a rank-N parent access.  The view IR is correct;
//     the rewriter just doesn't know how to consume it.
//
// Approach:
//     This pass DETECTS the bounds-remap-view shape and TAGS the LHS
//     pointer's ``hlfir.declare`` with a unit attribute
//     ``hlfir_bridge.bounds_remap_view``.  It does NOT transform the
//     IR -- the rebox chain stays intact, and downstream passes that
//     already understand pointer-typed boxes continue to work.
//
//     Two consumers of the tag (in this commit + follow-ups):
//
//     1. ``hlfir-rewrite-pointer-assigns`` skips marked declares so
//        its rank-mismatch error doesn't fire.
//     2. The bridge's SDFG-build path (descriptors.py) reads the tag,
//        traces the rebox chain to find the parent array, and emits
//        a DaCe ``View`` node with shape ``[total_extent]``,
//        ``strides=[1]``, and a fresh offset symbol --
//        ``offset_<ptr_name>_d0`` following the existing convention --
//        that an interstate edge binds per surrounding loop
//        iteration.
//
// Detection criteria (all must hold, pinned by
// ``tests/bounds_remap_view/test_view_vs_copy_distinguishable.py``):
//
//   1. The op is ``fir.rebox``.
//   2. The input box's element type has rank R_in (counted by
//      walking ``!fir.array<?x?x...x T>``).
//   3. The output box's element type has rank R_out, and R_out !=
//      R_in (the rank change).
//   4. The output box type wraps ``!fir.ptr<...>``  --  it is a
//      pointer-typed box.
//   5. The shape operand is a ``fir.shape_shift`` op (explicit
//      lower bound) -- not a plain ``fir.shape``.
//   6. The lower-bound operand of the ``shape_shift`` evaluates
//      (possibly through ``fir.convert``) to the literal integer
//      ``1`` -- Fortran's natural default LB.  A non-1 LB would
//      mean a *true* index shift the bridge cannot model; the
//      pass leaves those alone and the existing
//      ``hlfir-rewrite-pointer-assigns`` rejects them loudly.
//   7. The rebox flows (possibly through ``fir.convert``) into a
//      ``fir.store`` whose memref operand is a pointer's
//      ``hlfir.declare``.
//
//     Other op shapes do not match:
//       * ``RESHAPE`` intrinsic copy: lowers to ``hlfir.reshape`` /
//         a temp ``!hlfir.expr``  --  no ``fir.rebox`` involved.
//       * Plain pointer assign (no remap): ``fir.embox`` (not
//         rebox), same input/output rank.
//       * Plain slice assignment: ``hlfir.assign`` between same-
//         rank boxes; no pointer involved.
//
// Out of scope:
//     * The actual SDFG-side View emission (will live in
//       ``descriptors.py`` -- a follow-up commit).
//     * Bounds remaps where LHS lb != 1 (rejected upstream).
//     * Rebox chains that traverse a function-call boundary.
//
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

#include <cstdlib>

#define DEBUG_TYPE "mark-bounds-remap-views"

namespace hlfir_bridge {

namespace {

/// The unit attribute name attached to a pointer's ``hlfir.declare``
/// when its rebind is a bounds-remapping (rank-reshaping) view of a
/// parent array.  Downstream consumers read this attribute to know to
/// (a) skip index-rewriting and (b) emit an SDFG ``View`` node.
static constexpr llvm::StringLiteral kBoundsRemapViewAttr =
    "hlfir_bridge.bounds_remap_view";

/// Walk ``v`` through any number of ``fir.convert`` ops and return
/// the underlying value.  flang routes ``arith.constant 1 : i64``
/// through ``fir.convert : (i64) -> index`` before passing it to
/// ``shape_shift`` operands; the LB-equals-1 check must see through
/// this otherwise valid rebinds look like shifted ones.
static mlir::Value peelConverts(mlir::Value v) {
  for (int hops = 0; v && hops < 8; ++hops) {
    auto *def = v.getDefiningOp();
    if (!def) return v;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    return v;
  }
  return v;
}

/// Return ``true`` if ``v`` resolves (through any ``fir.convert``
/// chain) to a constant integer equal to ``1``.  ``1`` is the
/// natural Fortran default lower bound; rebinds whose explicit LB
/// reduces to ``1`` represent no actual index shift and are safe
/// to treat as views.
static bool isConstantOne(mlir::Value v) {
  v = peelConverts(v);
  if (!v) return false;
  auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(v.getDefiningOp());
  if (!cst) return false;
  if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
    return i.getInt() == 1;
  return false;
}

/// Return the rank of the array element type inside a
/// ``!fir.box<...>`` / ``!fir.box<!fir.ptr<...>>`` type, or
/// ``-1`` when ``ty`` isn't a box-of-array.
static int boxedArrayRank(mlir::Type ty) {
  auto box = mlir::dyn_cast<fir::BoxType>(ty);
  if (!box) return -1;
  mlir::Type elt = box.getEleTy();
  // Peel one ``!fir.ptr`` layer  --  pointer boxes wrap the array
  // type in ``!fir.ptr<...>``.
  if (auto p = mlir::dyn_cast<fir::PointerType>(elt)) elt = p.getElementType();
  if (auto h = mlir::dyn_cast<fir::HeapType>(elt)) elt = h.getElementType();
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(elt))
    return static_cast<int>(seq.getDimension());
  return -1;
}

/// Return ``true`` if ``ty`` is ``!fir.box<!fir.ptr<...>>`` (pointer
/// box, the LHS shape of a Fortran POINTER variable's box value).
static bool isPointerBox(mlir::Type ty) {
  auto box = mlir::dyn_cast<fir::BoxType>(ty);
  if (!box) return false;
  return mlir::isa<fir::PointerType>(box.getEleTy());
}

/// Walk ``v`` through ``fir.convert`` and ``fir.rebox`` ops to find
/// the originating ``hlfir.declare``.  Stops on anything else.
static hlfir::DeclareOp findDeclareThroughChain(mlir::Value v) {
  for (int hops = 0; v && hops < 16; ++hops) {
    auto *def = v.getDefiningOp();
    if (!def) return {};
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = cv.getValue();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
      v = rb.getBox();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
      v = eb.getMemref();
      continue;
    }
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      v = dg.getMemref();
      continue;
    }
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def)) return dc;
    return {};
  }
  return {};
}

/// Walk ``rebox``'s users (through ``fir.convert``) to find a
/// ``fir.store`` and return its memref operand.  Returns null if no
/// such store exists in this lexical scope.
static mlir::Value findStoreTargetForRebox(fir::ReboxOp rebox) {
  llvm::SmallVector<mlir::Value, 4> work{rebox.getResult()};
  for (int hops = 0; !work.empty() && hops < 16; ++hops) {
    mlir::Value v = work.pop_back_val();
    for (auto *user : v.getUsers()) {
      if (auto st = mlir::dyn_cast<fir::StoreOp>(user)) return st.getMemref();
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(user)) work.push_back(cv.getResult());
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// The pass.
// ---------------------------------------------------------------------------

struct MarkBoundsRemapViewsPass
    : public mlir::PassWrapper<MarkBoundsRemapViewsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkBoundsRemapViewsPass)

  llvm::StringRef getArgument() const final { return "hlfir-mark-bounds-remap-views"; }
  llvm::StringRef getDescription() const final {
    return "Tag the LHS pointer declare of every Fortran 2003 bounds-"
           "remapping pointer assignment (``ptr(1:N*K) => target(...)``) "
           "with ``hlfir_bridge.bounds_remap_view``.  Pure detect-and-"
           "mark -- the IR isn't transformed.  Downstream consumers "
           "(hlfir-rewrite-pointer-assigns; descriptors.py) read the "
           "tag to skip the index-rewriting model and emit an SDFG "
           "View node instead.";
  }

  void runOnOperation() override {
    unsigned tagged = 0;
    auto module = getOperation();

    // ``fir.embox`` form: ``p(1:M, 1:K) => arr1d`` (1D target -> 2D
    // pointer view) lowers to ``embox %arr1d(%shape_shift_2D)`` with
    // a rank-laundering ``fir.convert`` from ``<NxT>`` to ``<?x?xT>``
    // sitting between the parent's declare and the embox.  Same
    // bounds-remap-view semantics as the rebox form -- just produced
    // through a different IR shape because the parent is a plain
    // ref-array (not yet wrapped in a box).
    module.walk([&](fir::EmboxOp embox) {
      if (!isPointerBox(embox.getType())) return;
      int outRank = boxedArrayRank(embox.getType());
      if (outRank <= 0) return;

      // The embox's memref carries the (possibly rank-laundered)
      // target's ref type.  Walk through any ``fir.convert`` to
      // find the source rank.
      mlir::Value mem = embox.getMemref();
      mlir::Type sourceTy = mem.getType();
      for (int hop = 0; hop < 8; ++hop) {
        if (auto *def = mem.getDefiningOp()) {
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
            mem = cv.getValue();
            sourceTy = mem.getType();
            continue;
          }
        }
        break;
      }
      // Strip ref/heap/ptr off sourceTy to read the underlying
      // sequence type's static rank.
      int inRank = -1;
      for (int p = 0; p < 8; ++p) {
        if (auto r = mlir::dyn_cast<fir::ReferenceType>(sourceTy)) {
          sourceTy = r.getEleTy();
          continue;
        }
        if (auto h = mlir::dyn_cast<fir::HeapType>(sourceTy)) {
          sourceTy = h.getEleTy();
          continue;
        }
        if (auto pt = mlir::dyn_cast<fir::PointerType>(sourceTy)) {
          sourceTy = pt.getEleTy();
          continue;
        }
        break;
      }
      if (auto seq = mlir::dyn_cast<fir::SequenceType>(sourceTy))
        inRank = static_cast<int>(seq.getDimension());
      if (inRank <= 0 || inRank == outRank) return;

      // Shape operand must be ``shape_shift`` with all LB == 1.
      mlir::Value shape = embox.getShape();
      if (!shape) return;
      auto shiftOp = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp());
      if (!shiftOp) return;
      auto pairs = shiftOp.getPairs();
      bool allLbOne = true;
      for (size_t i = 0; i + 1 < pairs.size(); i += 2)
        if (!isConstantOne(pairs[i])) { allLbOne = false; break; }
      if (!allLbOne) return;

      // Locate the store target and trace to the pointer declare.
      mlir::Value storeTarget;
      for (auto *u : embox.getResult().getUsers()) {
        if (auto st = mlir::dyn_cast<fir::StoreOp>(u)) {
          storeTarget = st.getMemref();
          break;
        }
      }
      if (!storeTarget) return;
      auto ptrDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
          storeTarget.getDefiningOp());
      if (!ptrDecl) ptrDecl = findDeclareThroughChain(storeTarget);
      if (!ptrDecl) return;

      ptrDecl->setAttr(kBoundsRemapViewAttr,
                        mlir::UnitAttr::get(&getContext()));
      ++tagged;
    });

    module.walk([&](fir::ReboxOp rebox) {
      // (1) Output type must be a pointer-typed box.
      if (!isPointerBox(rebox.getType())) return;
      int outRank = boxedArrayRank(rebox.getType());
      if (outRank <= 0) return;

      // (2) Input rank must differ from output rank.
      int inRank = boxedArrayRank(rebox.getBox().getType());
      if (inRank <= 0 || inRank == outRank) return;

      // (3) Shape operand must be a ``fir.shape_shift`` (explicit LB)
      //     not a plain ``fir.shape`` (no LB).
      mlir::Value shape = rebox.getShape();
      if (!shape) return;
      auto shiftOp = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp());
      if (!shiftOp) return;

      // (4) Every lb in the shape_shift must resolve to constant 1.
      //     ``shape_shift`` operand layout: lb0, ext0, lb1, ext1, ...
      auto pairs = shiftOp.getPairs();
      bool allLbOne = true;
      for (size_t i = 0; i + 1 < pairs.size(); i += 2)
        if (!isConstantOne(pairs[i])) { allLbOne = false; break; }
      if (!allLbOne) return;

      // (5) Locate the store target -- the pointer's box-ref.
      mlir::Value storeTarget = findStoreTargetForRebox(rebox);
      if (!storeTarget) return;

      // (6) Walk back from the store target to the pointer's declare.
      auto ptrDecl = mlir::dyn_cast_or_null<hlfir::DeclareOp>(
          storeTarget.getDefiningOp());
      if (!ptrDecl) {
        // Sometimes there's a ``fir.convert`` between -- walk through.
        ptrDecl = findDeclareThroughChain(storeTarget);
      }
      if (!ptrDecl) return;

      // Tag the LHS pointer's declare.  ``UnitAttr`` is enough: the
      // downstream SDFG-build path re-traces the rebox chain itself
      // to discover the parent array and the per-rebind offset
      // arithmetic.  Idempotent  --  re-running the pass is a no-op
      // because ``setAttr`` overwrites with the same unit.
      ptrDecl->setAttr(kBoundsRemapViewAttr,
                        mlir::UnitAttr::get(&getContext()));
      ++tagged;

      if (std::getenv("HLFIR_BOUNDS_REMAP_TRACE")) {
        llvm::errs() << "MarkBoundsRemapViews: tagged "
                     << ptrDecl.getUniqName() << " (rebox rank "
                     << inRank << " -> " << outRank << ")\n";
        llvm::errs().flush();
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "MarkBoundsRemapViews: tagged " << tagged
               << " bounds-remap-view pointer declare(s)\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createMarkBoundsRemapViewsPass() {
  return std::make_unique<MarkBoundsRemapViewsPass>();
}

}  // namespace hlfir_bridge
