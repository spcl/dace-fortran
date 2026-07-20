// ============================================================================
// FoldCopyInOut.cpp  --  fold ``hlfir.copy_in`` / ``hlfir.copy_out`` pairs.
// ============================================================================
//
// Motivation:
//     Flang materialises a non-contiguous slice argument (e.g.
//     ``call inner_loops(INP(I, :), OUT(I, :))`` where ``INP`` is column-
//     major) into a heap-allocated copy via:
//
//         %src   = hlfir.designate %parent (i, lo:hi:1) shape ...
//                : -> !fir.box<!fir.array<NxT>>
//         %cpy:2 = hlfir.copy_in %src to %tempBox
//                : (!fir.box<!fir.array<NxT>>, !fir.ref<!fir.box<...>>)
//                  -> (!fir.box<!fir.array<NxT>>, i1)
//         %addr  = fir.box_addr %cpy#0 : ... -> !fir.ref<!fir.array<NxT>>
//         %alias:2 = hlfir.declare %addr ... { uniq_name = "_QFcalleeEarg" }
//                                              : ...
//         ... uses of %alias#0 / %alias#1 ...
//         hlfir.copy_out %tempBox, %cpy#1 to %src : ...
//
//     The bridge does not model ``hlfir.copy_in`` / ``hlfir.copy_out``  --
//     ``%alias`` becomes an uninitialised transient and writes via the
//     callee never propagate back to ``%parent``.  Tests with this
//     pattern (``memlet_in_map_test``, ``type_array_slice``, the
//     ``noncontiguous_*`` cluster) silently produce wrong values.
//
// What the pass does (stride-1 section scope):
//     For each ``hlfir.copy_in`` whose source is a ``hlfir.designate``
//     section with ANY NUMBER of stride-1 triplets and any number of
//     scalar dims, fold the alias-side accesses back to the parent.
//     The alias carries one dimension per SOURCE TRIPLET (scalar source
//     dims are collapsed away), so the rebuild walks source dims in
//     order and consumes one alias dim per triplet:
//
//         arr(i, lo:hi)  ->  %alias(j)     -> %parent(i, j + lo - 1)
//         arr(:, :, blk) ->  %alias(j, k)  -> %parent(j, k, blk)
//         arr(:, 1, :)   ->  %alias(j, k)  -> %parent(j, 1, k)
//
//     The last one is why dims are walked in order rather than as a
//     scalar prefix plus a trailing triplet: the scalar sits in the
//     MIDDLE of the rebuilt index list.
//
//     ``copy_in`` / ``copy_out`` and the heap buffer alloca then erase
//     because nothing references them.  The chain below the alias
//     declare reads / writes ``%parent`` directly.
//
// Unfoldable pairs are REJECTED, not left alone (``rejectSurvivors``):
//     the bridge does not model copy_in / copy_out, so a surviving pair
//     becomes a zero-filled phantom SDFG argument whose writes are
//     dropped.  Emitting that is a silent wrong answer, so the pass
//     fails the pipeline instead -- EXCEPT a record-element section (see
//     below), which only warns: it is a known-unfoldable case that
//     historically lowered as a phantom, and failing it would block
//     programs that never depended on it.
//
// Out of scope (left for follow-ups):
//     * Non-stride-1 triplets (``arr(1:N:2)``).  Would need an index
//       multiply ``+ (j-1)*stride`` instead of the current ``+ (lo-1)``.
//     * Record-element sections (``p_diag%p_vn(:,:,blk)`` where p_vn is a
//       ``t_cartesian_coordinates`` array).  Reparenting emits a record-level
//       ``offset_p_vn_d*`` that struct-flatten later leaves unresolved (it
//       renames the data to the leaf submember ``p_vn_x``).  The fold would
//       need to emit leaf-submember accesses; until then it bails and warns.
//     * Non-section sources are handled separately: a POINTER *variable*
//       view alias-folds (``tryFoldViewSource``); a POINTER / ALLOCATABLE
//       *component* reparents its alias accesses onto the member box
//       (``reparentMemberCopy``) -- designate reads the box strides, so it is
//       correct even when the component target is non-contiguous.  A bare
//       assumed-shape dummy box (a local forwarded to a deeper explicit-shape
//       callee) reparents onto the source box itself.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// One dimension of a source designate: either a scalar index (``lo`` holds it)
/// or a ``lo:hi:stride`` triplet.
struct DimIdx {
  bool triplet;
  mlir::Value lo;
  mlir::Value hi;
  mlir::Value stride;
};

/// A section source, one entry per source dimension in declaration order.
/// Dimension order is what makes ``arr(:, 1, :)`` work: the scalar has to stay
/// in the middle of the rebuilt index list, so the dims cannot be split into a
/// "scalar prefix" plus a trailing triplet.
struct SectionShape {
  llvm::SmallVector<DimIdx, 4> dims;
  unsigned tripletCount = 0;
};

/// Walk a designate's per-dim ``isTriplet`` flags + flat index list (3 operands
/// per triplet dim, 1 per scalar dim).  Returns false when the operand layout
/// disagrees with the flags, or when the source is not a section at all.
bool parseSection(hlfir::DesignateOp dg, SectionShape& out) {
  auto trip = dg.getIsTripletAttr();
  if (!trip) return false;
  auto tripFlags = trip.asArrayRef();
  if (tripFlags.empty()) return false;

  auto idxRange = dg.getIndices();
  unsigned cursor = 0;
  for (bool const isT : tripFlags) {
    if (isT) {
      if (cursor + 3 > idxRange.size()) return false;
      out.dims.push_back({true, idxRange[cursor], idxRange[cursor + 1], idxRange[cursor + 2]});
      cursor += 3;
      out.tripletCount++;
    } else {
      if (cursor + 1 > idxRange.size()) return false;
      out.dims.push_back({false, idxRange[cursor], {}, {}});
      cursor += 1;
    }
  }
  if (cursor != idxRange.size()) return false;
  return out.tripletCount > 0;
}

/// Trace ``v`` to a constant ``index``-typed integer if possible.
/// Walks ``arith.constant`` and ``fir.convert`` shims.  Returns
/// ``std::nullopt`` if the value isn't a constant.
std::optional<int64_t> traceConstIndex(mlir::Value v) {
  if (!v) return std::nullopt;
  auto* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) return ia.getInt();
    return std::nullopt;
  }
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) return traceConstIndex(cv.getValue());
  return std::nullopt;
}

/// True when ``v`` is the integer constant ``1`` (any int width / index).
bool isConstOne(mlir::Value v) {
  auto c = traceConstIndex(v);
  return c.has_value() && *c == 1;
}

/// True when ``t`` (a box / ref / array of) has a derived-type element.  Folding
/// a record-element section reparents onto the member box, but struct-flatten
/// then renames the data to its leaf submember (``p_vn`` -> ``p_vn_x``) with a
/// submember offset symbol, leaving the record-level ``offset_p_vn_d*`` the
/// reparented designate emitted unresolved.  Out of scope until the fold emits
/// leaf-submember accesses; bail so the pair is left for the survivor policy.
bool isRecordElement(mlir::Type t) {
  if (auto bx = mlir::dyn_cast<fir::BaseBoxType>(t)) t = bx.getEleTy();
  if (auto rt = mlir::dyn_cast<fir::ReferenceType>(t)) t = rt.getEleTy();
  if (auto heap = mlir::dyn_cast<fir::HeapType>(t)) t = heap.getEleTy();
  if (auto ptr = mlir::dyn_cast<fir::PointerType>(t)) t = ptr.getEleTy();
  if (auto sq = mlir::dyn_cast<fir::SequenceType>(t)) t = sq.getEleTy();
  return mlir::isa<fir::RecordType>(t);
}

struct FoldCopyInOutPass : public mlir::PassWrapper<FoldCopyInOutPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldCopyInOutPass)

  llvm::StringRef getArgument() const final { return "hlfir-fold-copy-in-out"; }
  llvm::StringRef getDescription() const final {
    return "Fold hlfir.copy_in / hlfir.copy_out pairs around inlined-callee "
           "alias declares (stride-1 sections); reject pairs that cannot fold.";
  }

  void runOnOperation() override {
    llvm::SmallVector<hlfir::CopyInOp, 16> copies;
    getOperation().walk([&](hlfir::CopyInOp op) { copies.push_back(op); });
    for (auto cin : copies) tryFold(cin);
    rejectSurvivors();
  }

  // A pair the fold could not match is NOT benign.  The bridge does not model
  // copy_in / copy_out, so the temporary surfaces as an SDFG argument that the
  // binding shim allocates and zero-fills: writes through it are dropped and
  // reads see zeros.  That is a silent wrong answer, so refuse the program
  // instead of emitting it -- same contract as ``hlfir-reject-polymorphism``.
  void rejectSurvivors() {
    bool failed = false;
    getOperation().walk([&](hlfir::CopyInOp op) {
      // Record-element sections are a known-unfoldable case (isRecordElement):
      // the reparent would emit an unresolved record-level offset.  These
      // historically lowered as a phantom argument that this program's compared
      // outputs did not depend on, so WARN rather than fail the whole pipeline --
      // downgrading loudly, not silently.  Non-record survivors are a genuine
      // silent miscompile (dropped writes / zero reads), so those still fail.
      if (isRecordElement(op.getVar().getType())) {
        op.emitWarning("hlfir-fold-copy-in-out: record-element ``hlfir.copy_in`` for ")
            << sourceName(op)
            << " left unfolded (out of scope: reparent would emit an unresolved record offset). "
               "It lowers as a phantom SDFG argument -- reads see zeros -- so any result that "
               "depends on it is wrong.  Fold needs leaf-submember accesses for record sections.";
        return;
      }
      op.emitError("hlfir-fold-copy-in-out: ``hlfir.copy_in`` survived the fold for ")
          << sourceName(op)
          << ".  The bridge cannot model a copy-in/copy-out pair: the temporary "
             "would become an uninitialised SDFG argument, so writes through it "
             "are dropped and reads see zeros.  Pass a contiguous whole array (or "
             "a stride-1 section the fold covers) instead of this actual argument.";
      failed = true;
    });
    if (failed) signalPassFailure();
  }

  // Best-effort Fortran name behind a copy_in source, for the diagnostic.
  static std::string sourceName(hlfir::CopyInOp cin) {
    mlir::Value v = cin.getVar();
    if (auto dg = v.getDefiningOp<hlfir::DesignateOp>()) v = dg.getMemref();
    if (auto ld = v.getDefiningOp<fir::LoadOp>()) v = ld.getMemref();
    if (auto decl = v.getDefiningOp<hlfir::DeclareOp>())
      return (llvm::Twine("``") + decl.getUniqName().getValue() + "``").str();
    return "<unnamed actual argument>";
  }

  // Collect the terminal designate uses of an alias declare, transparently
  // following the ladder of per-level dummy re-declares (``%inner =
  // hlfir.declare %outer#1``) that inlining a forwarding chain leaves -- the
  // section source, like the member source, can be forwarded through a deeper
  // inlined callee (``z_vn_ab(:,1,:)`` -> map_edges2edges -> inner) before it is
  // used.  ``chain`` is filled with the whole ladder so the caller erases it
  // innermost-first.  Returns false (bail, leave the fold undone) on any foreign
  // use -- a non-review declare, a whole-array assign, or a surviving call.
  static bool collectAliasUses(hlfir::DeclareOp aliasDecl, llvm::SmallVectorImpl<hlfir::DeclareOp>& chain,
                               llvm::SmallVectorImpl<hlfir::DesignateOp>& uses) {
    chain.push_back(aliasDecl);
    for (size_t ci = 0; ci < chain.size(); ++ci)
      for (mlir::Value const res : {chain[ci].getResult(0), chain[ci].getResult(1)})
        for (auto* u : res.getUsers()) {
          if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(u))
            uses.push_back(dg);
          else if (auto rd = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
            if (rd.getMemref() != res) return false;  // consumes the alias as non-memref -- out of scope
            chain.push_back(rd);
          } else
            return false;
        }
    return true;
  }

  void tryFold(hlfir::CopyInOp cin) {
    // 1) Source must be a section ``hlfir.designate``.
    auto srcDg = cin.getVar().getDefiningOp<hlfir::DesignateOp>();
    if (!srcDg) {
      dispatchNonSectionSource(cin);
      return;
    }
    SectionShape sec;
    if (!parseSection(srcDg, sec)) return;
    // Stride-1 only: a strided triplet needs ``+ (j-1)*stride``, not ``+ (lo-1)``.
    for (auto const& d : sec.dims)
      if (d.triplet && !isConstOne(d.stride)) return;
    // Record-element sections are out of scope (see isRecordElement): folding one
    // emits a record-level offset that struct-flatten later leaves unresolved.
    if (isRecordElement(srcDg.getResult().getType())) return;

    // 2) Walk users of ``cin#0`` (the box copy) for the
    // ``fir.box_addr`` and from there for the alias declare.
    fir::BoxAddrOp boxAddr;
    for (auto* u : cin.getResult(0).getUsers()) {
      if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(u)) {
        boxAddr = ba;
        break;
      }
    }
    if (!boxAddr) return;

    hlfir::DeclareOp aliasDecl;
    for (auto* u : boxAddr.getResult().getUsers()) {
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        aliasDecl = d;
        break;
      }
    }
    if (!aliasDecl) return;

    // 3) Source's parent  --  the array we're slicing.  This is
    // the memref of the source designate.
    mlir::Value const parent = srcDg.getMemref();
    mlir::OpBuilder b(aliasDecl);

    // 4) Rewrite uses of the alias declare's results.  Each designate
    // ``%alias #X (j)`` becomes ``%parent (scalars..., j + lo - 1)``.  The alias
    // may be re-declared through a deeper inlined callee, so walk the ladder;
    // each level re-views the SAME storage with the SAME index frame, so the
    // section shift maps identically at every depth.  Bail on a whole-result use
    // (``hlfir.assign %v to %alias`` or a surviving call) -- out of this scope.
    llvm::SmallVector<hlfir::DeclareOp, 4> chain;
    llvm::SmallVector<hlfir::DesignateOp, 8> aliasUseDgs;
    if (!collectAliasUses(aliasDecl, chain, aliasUseDgs)) return;

    for (auto useDg : aliasUseDgs) rewriteAccess(useDg, parent, sec, b);

    // 5) Erase the chain: re-declare ladder (innermost-first), box_addr,
    // copy_in, any copy_out targeting this copy_in, the alloca for the temp box.
    // Order matters  --  erase users first, defs last.
    // copy_out has the form ``copy_out %tempBox, %cin#1 to %var``.
    llvm::SmallVector<hlfir::CopyOutOp, 2> copyOuts;
    getOperation().walk([&](hlfir::CopyOutOp op) {
      if (op.getOperand(1) == cin.getResult(1)) copyOuts.push_back(op);
    });

    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
      if (it->getResult(0).use_empty() && it->getResult(1).use_empty()) it->erase();

    if (boxAddr.getResult().use_empty()) boxAddr.erase();

    for (auto co : copyOuts) co.erase();

    if (cin.getResult(0).use_empty() && cin.getResult(1).use_empty()) {
      mlir::Value const temp = cin.getTempBox();
      cin.erase();
      // The temp box is typically a ``fir.alloca`` whose only
      // users were the copy_in / copy_out pair.  Erase if dead.
      if (auto* def = temp.getDefiningOp())
        if (def->use_empty()) def->erase();
    }
  }

  /// Route a non-section ``copy_in`` (source is a load of a POINTER /
  /// ALLOCATABLE box) to the right handler:
  ///   * ``load(declare)``   -- a POINTER *variable* (bounds-remap view or
  ///     plain rebind).  Its target is contiguous, so the copy is element-
  ///     wise-equivalent to the view: alias-fold it (``tryFoldViewSource``).
  ///   * ``load(designate)`` -- a POINTER / ALLOCATABLE *component*
  ///     (``st%p_diag%vort``).  The component target may be strided, so
  ///     box_addr aliasing would let the contiguous explicit-shape callee
  ///     stride-walk a frame it never got = heap OOB.  Reparent the alias
  ///     accesses onto the member box (``reparentMemberCopy``) -- designate
  ///     reads the box strides, so it is correct for a strided target.
  ///   * ``declare(rebox(load(designate)))`` -- the same component forwarded
  ///     through an assumed-shape dummy (``e(:,:,:)``) before the explicit-shape
  ///     callee.  Trace past the rebox to the member and reparent identically.
  static bool isPtrOrAllocDesignate(hlfir::DesignateOp dg) {
    auto attrs = dg.getFortranAttrs();
    return attrs && bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer |
                                                   fir::FortranVariableFlagsEnum::allocatable);
  }

  /// True when ``v`` is defined by a POINTER / ALLOCATABLE component designate or
  /// by a POINTER / ALLOCATABLE variable declare -- the two shapes that put a
  /// stride-carrying box behind a ``fir.load``.
  static bool isPtrOrAllocSource(mlir::Value v) {
    if (auto dg = v.getDefiningOp<hlfir::DesignateOp>()) return isPtrOrAllocDesignate(dg);
    if (auto decl = v.getDefiningOp<hlfir::DeclareOp>()) {
      auto attrs = decl.getFortranAttrs();
      return attrs && bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer |
                                                     fir::FortranVariableFlagsEnum::allocatable);
    }
    return false;
  }

  void dispatchNonSectionSource(hlfir::CopyInOp cin) {
    mlir::Value const src = cin.getVar();
    if (auto ld = src.getDefiningOp<fir::LoadOp>()) {
      mlir::Value const memref = ld.getMemref();
      if (memref.getDefiningOp<hlfir::DeclareOp>()) {
        tryFoldViewSource(cin);
        return;
      }
      if (auto memDg = memref.getDefiningOp<hlfir::DesignateOp>())
        if (isPtrOrAllocDesignate(memDg))
          reparentMemberCopy(cin);
      return;
    }
    // Assumed-shape forward: an explicit-shape callee reached through an
    // assumed-shape dummy (veloc_adv_vert: p_diag%veloc_adv_vert -> dispatch
    // e(:,:,:) -> writer e(n1,n2,nb)).  Flang reboxes the pointer member into an
    // assumed-shape box, re-declares it, and copy_in's THAT -- so the source is a
    // declare, not a load.  Trace declare -> rebox -> load -> member designate;
    // reparent onto the rebox's input, the member pointer box that carries the
    // component strides (same box the direct path uses).
    if (auto decl = src.getDefiningOp<hlfir::DeclareOp>()) {
      if (auto rb = decl.getMemref().getDefiningOp<fir::ReboxOp>()) {
        if (auto ld = rb.getBox().getDefiningOp<fir::LoadOp>())
          // Under the load sits either the pointer/allocatable COMPONENT
          // (``st%p_diag%vort``) or -- once alias collapse has hoisted that
          // component to a top-level pointer dummy -- the pointer VARIABLE
          // declare for it.  Either way ``ld`` is the stride-carrying box that
          // names the real destination, so both reparent identically.
          if (isPtrOrAllocSource(ld.getMemref())) {
            reparentMemberCopy(cin, ld.getResult());
            // The reboxed assumed-shape dummy declare + rebox are now dead
            // (their only uses were the folded copy_in/out); drop them so they
            // don't leak as a phantom array.  Users already gone via reparent.
            if (decl.getResult(0).use_empty() && decl.getResult(1).use_empty()) decl.erase();
            if (rb.getResult().use_empty()) rb.erase();
          }
        return;
      }
      // Bare assumed-shape dummy box forwarded to a deeper explicit-shape callee:
      // a local array (or any array) passed to grad_fd_norm_oce_3d's assumed-shape
      // psi_c, which forwards it to grad_..._onblock via copy_in.  The source is
      // just ``%box`` = declare (no rebox, no load).  ``%box`` itself carries the
      // strides, so reparent the alias designates onto it -- designate reads the
      // strides, so it is sound whether the source is contiguous or not.
      if (mlir::isa<fir::BaseBoxType>(src.getType())) reparentMemberCopy(cin);
    }
  }

  /// Fold ``copy_in`` / ``copy_out`` around an inlined-callee dummy bound to a
  /// POINTER / ALLOCATABLE struct component (``st%p_diag%vort``).  The bridge
  /// surfaces the copy_in temp as a phantom argument ``v`` and drops the copies,
  /// so writes through the dummy never reach the component.  Rewrite every alias
  /// access ``%v (idx...)`` to designate the MEMBER BOX directly:
  ///
  ///     %e = hlfir.designate %memberBox (idx...)
  ///
  /// The member box carries the component's strides, so designate hits the right
  /// element even when the target is non-contiguous (box_addr aliasing would
  /// lose the strides = heap OOB).  Erasing the alias declare also drops the
  /// phantom ``v`` from the signature.  Follows the ladder of per-level dummy
  /// re-declares that inlining a forwarding chain leaves (the actual reaches the
  /// stores through one ``hlfir.declare`` re-view per callee level), reparenting
  /// the terminal designates and erasing the whole ladder.  Verbatim reparent
  /// makes this sound at any depth (no section offset to compose).  Bails on a
  /// non-review declare or any other non-designate use (whole-array assign /
  /// surviving call), leaving the drop in place rather than risk a wrong rewrite.
  void reparentMemberCopy(hlfir::CopyInOp cin) { reparentMemberCopy(cin, cin.getVar()); }

  /// ``memberBox`` is the stride-carrying box the alias accesses reparent onto:
  /// the loaded pointer/allocatable box.  Usually ``cin.getVar()`` (member passed
  /// straight to the explicit-shape callee), but for an assumed-shape forward it
  /// is the member box UNDER the rebox that produced the assumed-shape dummy.
  void reparentMemberCopy(hlfir::CopyInOp cin, mlir::Value memberBox) {  // box<ptr<array>> -- carries strides

    // copy_in#0 -> fir.box_addr -> (fir.convert)? -> hlfir.declare (alias ``v``).
    fir::BoxAddrOp boxAddr;
    for (auto* u : cin.getResult(0).getUsers())
      if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(u)) {
        boxAddr = ba;
        break;
      }
    if (!boxAddr) return;

    fir::ConvertOp convert;
    hlfir::DeclareOp aliasDecl;
    for (auto* u : boxAddr.getResult().getUsers()) {
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        aliasDecl = d;
        break;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(u)) {
        for (auto* uu : cv.getResult().getUsers())
          if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(uu)) {
            aliasDecl = d;
            convert = cv;
            break;
          }
        if (aliasDecl) break;
      }
    }
    if (!aliasDecl) return;

    // Collect alias-use designates, following the ladder of per-level dummy
    // re-declares that inlining leaves.  An actual forwarded through N callee
    // levels (veloc_adv_vert -> _mimetic -> _rot) reaches the element
    // designates through N chained ``%inner = hlfir.declare %outer#1`` re-views
    // of the SAME copy_in storage, so the terminal stores sit under the
    // innermost declare, not aliasDecl.
    llvm::SmallVector<hlfir::DesignateOp, 8> uses;
    llvm::SmallVector<hlfir::DeclareOp, 4> chain;
    if (!collectAliasUses(aliasDecl, chain, uses)) return;

    // Reparent each ``%alias (idx...)`` onto the member box verbatim.
    for (auto useDg : uses) {
      mlir::OpBuilder b(useDg);
      auto newOp = b.create<hlfir::DesignateOp>(
          useDg.getLoc(),
          /*result_type=*/useDg.getResult().getType(),
          /*memref=*/memberBox,
          /*component=*/mlir::StringAttr{},
          /*component_shape=*/mlir::Value{},
          /*indices=*/useDg.getIndices(),
          /*is_triplet=*/useDg.getIsTripletAttr(),
          /*substring=*/mlir::ValueRange{},
          /*complex_part=*/mlir::BoolAttr{},
          /*shape=*/useDg.getShape(),
          /*typeparams=*/useDg.getTypeparams(),
          /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
      useDg.getResult().replaceAllUsesWith(newOp.getResult());
      useDg.erase();
    }

    // Erase the now-dead chain: re-declare ladder, convert, box_addr,
    // copy_out(s), copy_in, temp box.  Users first, defs last.
    llvm::SmallVector<hlfir::CopyOutOp, 2> copyOuts;
    getOperation().walk([&](hlfir::CopyOutOp op) {
      if (op.getOperand(1) == cin.getResult(1)) copyOuts.push_back(op);
    });
    // Erase the re-declare ladder innermost-first: the terminal designates are
    // gone, so the inner declare is dead, which frees the outer, up to aliasDecl.
    for (auto it = chain.rbegin(); it != chain.rend(); ++it)
      if (it->getResult(0).use_empty() && it->getResult(1).use_empty()) it->erase();
    if (convert && convert.getResult().use_empty()) convert.erase();
    if (boxAddr.getResult().use_empty()) boxAddr.erase();
    for (auto co : copyOuts) co.erase();
    if (cin.getResult(0).use_empty() && cin.getResult(1).use_empty()) {
      mlir::Value const temp = cin.getTempBox();
      cin.erase();
      if (auto* def = temp.getDefiningOp())
        if (def->use_empty()) def->erase();
    }
  }

  /// Fold ``copy_in`` whose source is a load of a POINTER / VIEW box.
  /// Replaces the copy box (``cin#0``) with the source box (``load(p_box)``)
  /// so the inlined alias declare reads ``box_addr(load(p_box))`` directly,
  /// then erases the now-dead copy_in / copy_out pair + temp buffer.
  /// ``asAssumedShapeAlias`` (box_addr peel) resolves the alias to ``p``,
  /// and ``p``'s bounds-remap view folds every ``v(i)`` to the parent.
  void tryFoldViewSource(hlfir::CopyInOp cin) {
    auto ld = cin.getVar().getDefiningOp<fir::LoadOp>();
    if (!ld) return;
    auto srcDecl = ld.getMemref().getDefiningOp<hlfir::DeclareOp>();
    if (!srcDecl) return;
    // Scope to POINTER actuals (covers bounds-remap views + plain pointer
    // rebinds).  A copy_in of a plain contiguous array is left to the
    // section path / untouched.
    auto attrs = srcDecl.getFortranAttrs();
    if (!attrs || !bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer)) return;
    // The copy box and the source box share the same type
    // (``box<ptr<array<?>>>``), so the replacement is type-safe and the
    // downstream ``fir.box_addr`` now extracts the source view's data.
    if (cin.getResult(0).getType() != cin.getVar().getType()) return;
    cin.getResult(0).replaceAllUsesWith(cin.getVar());

    llvm::SmallVector<hlfir::CopyOutOp, 2> copyOuts;
    getOperation().walk([&](hlfir::CopyOutOp op) {
      if (op.getOperand(1) == cin.getResult(1)) copyOuts.push_back(op);
    });
    for (auto co : copyOuts) co.erase();

    if (cin.getResult(0).use_empty() && cin.getResult(1).use_empty()) {
      mlir::Value const temp = cin.getTempBox();
      cin.erase();
      if (auto* def = temp.getDefiningOp())
        if (def->use_empty()) def->erase();
    }
  }

  /// Rewrite a single ``hlfir.designate %alias (j_1, ..., j_K)`` use
  /// to the equivalent designate on ``parent`` with the section
  /// indices folded in.  Preserves triplets / shape on the alias-
  /// access side (so ``%alias(1:N:1)`` whole-array becomes
  /// ``%parent(scalars..., 1:N:1)``).
  static void rewriteAccess(hlfir::DesignateOp useDg, mlir::Value parent, const SectionShape& sec, mlir::OpBuilder& b) {
    b.setInsertionPoint(useDg);
    auto loc = useDg.getLoc();

    // The alias has one dimension per SOURCE TRIPLET (scalar source dims are
    // already collapsed away), so walk the source dims in order and consume one
    // alias dim per triplet.  Scalar source dims pass through in place, which is
    // what keeps ``arr(:, 1, :)`` correct.
    auto aliasTripAttr = useDg.getIsTripletAttr();
    auto aliasIdx = useDg.getIndices();

    // Split the alias's own flat index list into per-dim entries.
    llvm::SmallVector<DimIdx, 4> aliasDims;
    if (!aliasTripAttr || aliasTripAttr.asArrayRef().empty()) {
      for (auto idx : aliasIdx) aliasDims.push_back({false, idx, {}, {}});
    } else {
      unsigned cursor = 0;
      for (bool const isT : aliasTripAttr.asArrayRef()) {
        if (isT) {
          if (cursor + 3 > aliasIdx.size()) return;
          aliasDims.push_back({true, aliasIdx[cursor], aliasIdx[cursor + 1], aliasIdx[cursor + 2]});
          cursor += 3;
        } else {
          if (cursor + 1 > aliasIdx.size()) return;
          aliasDims.push_back({false, aliasIdx[cursor], {}, {}});
          cursor += 1;
        }
      }
      if (cursor != aliasIdx.size()) return;
    }
    if (aliasDims.size() != sec.tripletCount) return;

    // ``parent_idx = alias_idx + (lo - 1)`` for the dim's own lo.  Stride-1 only
    // -- the caller filters strided triplets out before reaching here.
    auto lowerShift = [&](mlir::Value lo) -> mlir::Value {
      auto loConst = traceConstIndex(lo);
      if (loConst && *loConst == 1) return {};  // ``arr(:)`` default lo = 1: no shift
      mlir::Value l = lo;
      if (!l.getType().isIndex()) l = b.create<fir::ConvertOp>(loc, b.getIndexType(), l);
      mlir::Value const one = b.create<mlir::arith::ConstantOp>(loc, b.getIndexType(), b.getIndexAttr(1));
      return b.create<mlir::arith::SubIOp>(loc, l, one);
    };
    auto shift = [&](mlir::Value v, mlir::Value by) -> mlir::Value {
      if (!by) return v;
      mlir::Value vc = v;
      if (!v.getType().isIndex()) vc = b.create<fir::ConvertOp>(loc, b.getIndexType(), v);
      return b.create<mlir::arith::AddIOp>(loc, vc, by);
    };

    llvm::SmallVector<mlir::Value, 6> newIndices;
    llvm::SmallVector<bool, 4> newTripFlags;
    unsigned aliasCursor = 0;
    for (auto const& d : sec.dims) {
      if (!d.triplet) {  // scalar source dim: pass through, holds its position
        newIndices.push_back(d.lo);
        newTripFlags.push_back(false);
        continue;
      }
      auto const& a = aliasDims[aliasCursor++];
      mlir::Value const by = lowerShift(d.lo);
      if (a.triplet) {
        newIndices.push_back(shift(a.lo, by));
        newIndices.push_back(shift(a.hi, by));
        newIndices.push_back(a.stride);
        newTripFlags.push_back(true);
      } else {
        newIndices.push_back(shift(a.lo, by));
        newTripFlags.push_back(false);
      }
    }

    auto newOp = b.create<hlfir::DesignateOp>(
        loc,
        /*result_type=*/useDg.getResult().getType(),
        /*memref=*/parent,
        /*component=*/mlir::StringAttr{},
        /*component_shape=*/mlir::Value{},
        /*indices=*/mlir::ValueRange{newIndices},
        /*is_triplet=*/
        (newTripFlags.empty() ? mlir::DenseBoolArrayAttr{} : b.getDenseBoolArrayAttr(newTripFlags)),
        /*substring=*/mlir::ValueRange{},
        /*complex_part=*/mlir::BoolAttr{},
        /*shape=*/useDg.getShape(),
        /*typeparams=*/mlir::ValueRange{},
        /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
    useDg.getResult().replaceAllUsesWith(newOp.getResult());
    useDg.erase();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createFoldCopyInOutPass() { return std::make_unique<FoldCopyInOutPass>(); }

}  // namespace hlfir_bridge
