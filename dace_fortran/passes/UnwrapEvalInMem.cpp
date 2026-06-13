// ============================================================================
// UnwrapEvalInMem.cpp  --  lower hlfir.eval_in_mem to alloca + body + reads.
// ============================================================================
//
// Motivation:
//     Flang's HLFIR wraps any array-valued expression that has to be
//     evaluated into pre-allocated memory in an ``hlfir.eval_in_mem``
//     op.  The op's body takes a single block argument -- a
//     ``fir.ref<array<...>>`` "result buffer" -- and writes the value
//     into it; the op then yields an SSA ``!hlfir.expr<...>`` token
//     that downstream code consumes via ``hlfir.assign`` (or other
//     HLFIR consumers).  The canonical caller shape after
//     ``hlfir-inline-all`` for ``tmp = make3(x)`` is:
//
//         %38 = hlfir.eval_in_mem shape %37 : (!fir.shape<1>)
//                 -> !hlfir.expr<3xf64> {
//         ^bb0(%buf: !fir.ref<!fir.array<3xf64>>):
//            %47 = fir.call @make3(%x) ...
//            fir.save_result %47 to %buf(%37) ...
//         }
//         hlfir.assign %38 to %target : !hlfir.expr<3xf64>, ...
//
//     The bridge's expression resolver cannot read an ``!hlfir.expr``
//     value -- it has no concrete memref behind it -- and gives up
//     with the ``?`` placeholder for the assignment's RHS.  Flang's
//     own ``bufferize-hlfir`` lowers eval_in_mem (along with EVERY
//     other hlfir.expr-typed value), but its scope is too broad: it
//     reshapes IR that other tests depend on at the HLFIR level and
//     regresses them.
//
// What the pass does:
//     A targeted, eval_in_mem-only rewrite.  For each op:
//
//       1. Insert a ``fir.alloca`` of the eval_in_mem's result
//          element type and shape immediately before the op.
//       2. Wrap the alloca in a fresh ``hlfir.declare`` so downstream
//          designate / box ops route through it cleanly.
//       3. Splice every op from the eval_in_mem body into the
//          caller's block right before the eval_in_mem, remapping the
//          body's block argument (the result buffer) to the new
//          declare's reference result.
//       4. Replace every use of the eval_in_mem's SSA result (the
//          ``!hlfir.expr`` token) with the declare's box result;
//          callers reading the expression are now reading the
//          allocated buffer directly.
//       5. Erase the eval_in_mem.
//
//     The result IR has the same shape a stack-local Fortran array
//     would have (``fir.alloca`` + ``hlfir.declare`` + writes +
//     reads), so the bridge's existing extract_vars + AST emitter
//     paths handle it without any ``.tmp.*`` filter widening, no
//     synthetic ``_allocated`` companions, no convert-walker hacks.
//
// Naming:
//     The fresh declare gets a unique ``uniq_name`` derived from a
//     counter so multiple eval_in_mem unwraps in the same function
//     do not collide.  The name lives in the bridge's namespace (no
//     dotted ``.tmp.`` prefix) so extract_vars surfaces it as a
//     normal local transient.
//
// Idempotent:
//     A second invocation finds no eval_in_mem ops to rewrite and is
//     a no-op.
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

struct UnwrapEvalInMemPass
    : public mlir::PassWrapper<UnwrapEvalInMemPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnwrapEvalInMemPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-unwrap-eval-in-mem";
  }
  llvm::StringRef getDescription() const final {
    return "Lower hlfir.eval_in_mem to fir.alloca + body + reads so the "
           "bridge's existing emitter handles array-valued expression "
           "results without going through flang's broader bufferize-hlfir.";
  }

  /// Trace back through ``fir.convert`` to find a memref source declare.
  /// Returns the declare's first result (the box result), or a null Value
  /// if the chain doesn't end at a declare we can use.
  static mlir::Value traceToDeclareResult(mlir::Value v) {
    for (int i = 0; i < 4 && v; ++i) {
      auto* def = v.getDefiningOp();
      if (!def) return {};
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
        v = cv.getValue();
        continue;
      }
      if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def))
        return dc.getResult(0);
      return {};
    }
    return {};
  }

  /// Rewrite a single ``hlfir.eval_in_mem`` op.  Returns true on success.
  ///
  /// Strategy: look for the canonical post-``hlfir-inline-all`` body shape
  ///
  ///     ... callee body assigns to its own ``r`` declare ...
  ///     %loaded = fir.load %r#0 : !fir.ref<!fir.array<NxT>>
  ///     fir.save_result %loaded to %buf(%shape) : ...
  ///
  /// and bypass the intermediate buffer entirely: replace uses of the
  /// eval_in_mem's ``!hlfir.expr`` result with the inlined ``r`` declare's
  /// box result, splice the body's writes-into-``r`` into the caller, and
  /// erase the eval_in_mem + the load + save_result chain.  The bridge's
  /// AST emitter then sees ``hlfir.assign %r#0 to %target`` as a regular
  /// whole-array copy from a stack-local array -- the same shape any
  /// Fortran local would have.
  ///
  /// When the body doesn't end in the load+save_result canonical shape
  /// (i.e. the result is computed in-place into the buffer, not into a
  /// separate ``r``), fall back to a fresh alloca + body splice; the
  /// bridge's ``fir.save_result`` support is needed for that path and
  /// remains a separate item.
  bool rewriteOne(hlfir::EvaluateInMemoryOp op, unsigned& counter) {
    auto& body = op.getBody();
    if (body.empty()) return false;
    auto* entry = &body.front();
    if (entry->getNumArguments() != 1) return false;

    // The block-arg is the result buffer: ``fir.ref<array<...>>``.
    auto bufArg = entry->getArgument(0);
    auto bufRefTy = mlir::dyn_cast<fir::ReferenceType>(bufArg.getType());
    if (!bufRefTy) return false;
    auto arrTy = mlir::dyn_cast<fir::SequenceType>(bufRefTy.getEleTy());
    if (!arrTy) return false;

    // Look for the canonical ``fir.save_result %loaded to %bufArg`` at the
    // end of the body, where ``%loaded`` came from a ``fir.load`` of a
    // declare.  If found, we can bypass creating a new buffer.
    fir::SaveResultOp finalSave;
    for (auto& innerOp : *entry) {
      if (auto sr = mlir::dyn_cast<fir::SaveResultOp>(innerOp)) {
        finalSave = sr;
        break;
      }
    }
    mlir::Value srcDeclareBox{};
    fir::LoadOp srcLoad{};
    if (finalSave) {
      if (auto* def = finalSave.getValue().getDefiningOp()) {
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
          srcLoad = ld;
          srcDeclareBox = traceToDeclareResult(ld.getMemref());
        }
      }
    }

    mlir::OpBuilder builder(op);
    mlir::IRMapping mapper;
    auto* parentBlock = op->getBlock();
    auto insertPoint = mlir::Block::iterator(op);

    if (srcDeclareBox) {
      // Bypass path: splice every op EXCEPT the load + save_result.  Map
      // the buffer block-arg to the source declare's box result so any
      // op that happened to reference the block-arg gets routed through
      // the source declare (defensive -- the canonical body shape doesn't
      // reference the block-arg outside the save_result).
      mapper.map(bufArg, srcDeclareBox);
      for (auto& innerOp : llvm::make_early_inc_range(*entry)) {
        if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
        if (&innerOp == finalSave.getOperation()) continue;
        if (srcLoad && &innerOp == srcLoad.getOperation()) continue;
        auto* clone = innerOp.clone(mapper);
        parentBlock->getOperations().insert(insertPoint, clone);
        for (auto pair :
             llvm::zip(innerOp.getResults(), clone->getResults()))
          mapper.map(std::get<0>(pair), std::get<1>(pair));
      }
      // Find the cloned source declare's result (whatever the original
      // ``srcDeclareBox`` mapped to inside the spliced body).
      auto cloned = mapper.lookupOrDefault(srcDeclareBox);
      // Drop any ``hlfir.destroy`` of the eval_in_mem's result -- those
      // expect an ``!hlfir.expr``, which we no longer produce.
      llvm::SmallVector<hlfir::DestroyOp, 2> destroys;
      for (auto* user : op.getResult().getUsers())
        if (auto d = mlir::dyn_cast<hlfir::DestroyOp>(user))
          destroys.push_back(d);
      for (auto d : destroys) d.erase();

      // Rewrite ``hlfir.apply %expr (%i)`` users to
      // ``fir.load (hlfir.designate %src_decl (%i))``.  The bypass
      // replaces the ``!hlfir.expr<NxT>`` result with the source's
      // ``!fir.ref<!fir.array<NxT>>``, which the apply op's verifier
      // would reject (its operand must be expr-typed).  Designate +
      // load on the source memref reads the same element value.
      // Production case: ``c(:, i) = a + make3(src(i))`` where the
      // fn return is the operand of an ``arith.addf`` inside an
      // ``hlfir.elemental`` block -- the elemental's body uses
      // ``hlfir.apply`` to extract each element.
      llvm::SmallVector<hlfir::ApplyOp, 4> applies;
      for (auto* user : op.getResult().getUsers())
        if (auto a = mlir::dyn_cast<hlfir::ApplyOp>(user))
          applies.push_back(a);
      for (auto a : applies) {
        mlir::OpBuilder ab(a);
        auto elemTy = a.getResult().getType();
        auto eltRefTy = fir::ReferenceType::get(elemTy);
        auto designate = ab.create<hlfir::DesignateOp>(
            a.getLoc(), eltRefTy, /*memref=*/cloned,
            /*indices=*/a.getIndices(),
            /*typeparams=*/mlir::ValueRange{},
            /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
        auto loaded = ab.create<fir::LoadOp>(a.getLoc(), designate.getResult());
        a.getResult().replaceAllUsesWith(loaded.getResult());
        a.erase();
      }

      op.getResult().replaceAllUsesWith(cloned);
      op.erase();
      ++counter;  // keep counter consistent across paths even though no
                  // fresh name was minted on this path
      return true;
    }

    // Fallback path: introduce a fresh ``fir.alloca`` + ``hlfir.declare``
    // and splice the body verbatim with the buffer block-arg mapped to
    // the declare's first result.  Needs bridge ``fir.save_result``
    // support to lower correctly -- pinned as a follow-up.
    auto loc = op.getLoc();
    auto alloca = builder.create<fir::AllocaOp>(loc, arrTy);
    std::string name = "_eval_in_mem_" + std::to_string(counter++);
    auto refTy = fir::ReferenceType::get(arrTy);
    auto declare = builder.create<hlfir::DeclareOp>(
        loc,
        /*resultType0=*/refTy,
        /*resultType1=*/refTy,
        /*memref=*/alloca.getResult(),
        /*shape=*/op.getShape(),
        /*typeparams=*/op.getTypeparams(),
        /*dummy_scope=*/mlir::Value{},
        /*uniq_name=*/builder.getStringAttr(name),
        /*fortran_attrs=*/fir::FortranVariableFlagsAttr{},
        /*data_attr=*/cuf::DataAttributeAttr{});

    mapper.map(bufArg, declare.getResult(0));
    for (auto& innerOp : llvm::make_early_inc_range(*entry)) {
      if (innerOp.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
      auto* clone = innerOp.clone(mapper);
      parentBlock->getOperations().insert(insertPoint, clone);
      for (auto pair : llvm::zip(innerOp.getResults(), clone->getResults()))
        mapper.map(std::get<0>(pair), std::get<1>(pair));
    }
    llvm::SmallVector<hlfir::DestroyOp, 2> destroys;
    for (auto* user : op.getResult().getUsers())
      if (auto d = mlir::dyn_cast<hlfir::DestroyOp>(user))
        destroys.push_back(d);
    for (auto d : destroys) d.erase();
    op.getResult().replaceAllUsesWith(declare.getResult(0));
    op.erase();
    return true;
  }

  /// Bypass the ``.result``-buffer pattern flang emits for a
  /// derived-type (or any by-value scalar/aggregate) function return:
  ///
  ///   %result_buf = fir.alloca !fir.type<vec3> {bindc_name = ".result"}
  ///   ... callee inlined: writes to its own ``%r`` declare ...
  ///   %loaded = fir.load %r#0 : !fir.ref<!fir.type<vec3>>
  ///   fir.save_result %loaded to %result_buf
  ///   %result_decl = hlfir.declare %result_buf
  ///   %expr        = hlfir.as_expr %result_decl#0 move %false
  ///   hlfir.assign %expr to %p_decl
  ///
  /// flang does not wrap this in ``hlfir.eval_in_mem`` (that wrapper is
  /// reserved for array-by-value returns) -- it goes directly through a
  /// ``.result`` buffer + ``hlfir.as_expr``.  After
  /// ``hlfir-inline-all`` splices the callee, the ``.result`` buffer
  /// becomes redundant: the inlined ``r`` declare already holds the
  /// value.  We replace uses of the ``.result``-backed declare with
  /// the inlined source declare, then erase the intermediate
  /// (save_result + load + as_expr + the .result alloca's declare).
  ///
  /// The end IR has ``hlfir.assign %r_decl#0 to %p_decl#0`` (whole-DT
  /// copy from a stack-local), which ``hlfir-flatten-structs``
  /// rewrites into per-member companions just like any user-declared
  /// struct assignment.
  void rewriteResultBuffers(mlir::ModuleOp module) {
    // Collect all ``.result``-named allocas in one walk so we can
    // mutate without invalidating the walker.
    llvm::SmallVector<fir::AllocaOp, 8> bufs;
    module.walk([&](fir::AllocaOp op) {
      auto bn = op.getBindcName();
      if (bn && bn.value() == ".result") bufs.push_back(op);
    });
    for (auto buf : bufs) {
      // Find a ``hlfir.declare`` wrapping the buf and a
      // ``fir.save_result`` writing to it.  The save_result may sit
      // before or after the declare in source order; both shapes
      // appear in practice.
      hlfir::DeclareOp bufDecl;
      fir::SaveResultOp save;
      for (auto* u : buf.getResult().getUsers()) {
        if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) bufDecl = d;
        if (auto sr = mlir::dyn_cast<fir::SaveResultOp>(u)) save = sr;
      }
      if (!save && bufDecl) {
        // save_result may write to the declare's result rather than the
        // bare alloca; check both result#0 and result#1.
        for (auto* u : bufDecl.getResult(0).getUsers())
          if (auto sr = mlir::dyn_cast<fir::SaveResultOp>(u)) save = sr;
        if (!save)
          for (auto* u : bufDecl.getResult(1).getUsers())
            if (auto sr = mlir::dyn_cast<fir::SaveResultOp>(u)) save = sr;
      }
      if (!save) continue;

      // The value being saved must be a ``fir.load`` of a declare.
      auto load = mlir::dyn_cast_or_null<fir::LoadOp>(
          save.getValue().getDefiningOp());
      if (!load) continue;
      auto srcDecl = traceToDeclareResult(load.getMemref());
      if (!srcDecl) continue;

      // Replace uses of the ``.result``-backed declare (both results)
      // with the source declare.  Then walk the chain forward and
      // erase the intermediate ``hlfir.as_expr`` that wraps the
      // ``.result`` declare for an expr-typed ``hlfir.assign`` source
      // -- the bridge handles ref-typed assign sources directly, no
      // expr wrapper needed.
      if (bufDecl) {
        // Snapshot ``hlfir.as_expr`` users.  We splice them OUT entirely
        // (not re-issue) so the downstream ``hlfir.assign %expr to %p``
        // sees a plain ref-typed source -- the bridge handles
        // ``hlfir.assign`` between two memrefs via the existing whole-
        // DT copy path that ``hlfir-flatten-structs`` then expands into
        // per-member copies.  Keeping the as_expr wrapper would force
        // the bridge through its expr-typed assign path, which does
        // not yet flatten through to per-member access.
        llvm::SmallVector<hlfir::AsExprOp, 2> asExprs;
        for (auto* u : bufDecl.getResult(0).getUsers())
          if (auto ae = mlir::dyn_cast<hlfir::AsExprOp>(u))
            asExprs.push_back(ae);
        for (auto ae : asExprs) {
          // Drop any ``hlfir.destroy`` users of the expr (no expr to
          // destroy once we splice the wrapper out).
          llvm::SmallVector<hlfir::DestroyOp, 2> destroys;
          for (auto* eu : ae.getResult().getUsers())
            if (auto d = mlir::dyn_cast<hlfir::DestroyOp>(eu))
              destroys.push_back(d);
          for (auto d : destroys) d.erase();
          ae.getResult().replaceAllUsesWith(srcDecl);
          ae.erase();
        }
        // Any remaining ref-typed users of the declare get
        // re-pointed at the source declare directly.
        bufDecl.getResult(0).replaceAllUsesWith(srcDecl);
        if (!bufDecl.getResult(1).use_empty())
          bufDecl.getResult(1).replaceAllUsesWith(srcDecl);
        bufDecl.erase();
      }

      // Erase the save_result + the load (if it has no other uses).
      save.erase();
      if (load.getResult().use_empty()) load.erase();

      // Erase the ``.result`` alloca if it's now dead.  DCE would
      // catch this later, but eagerly erasing keeps the IR cleaner
      // for downstream passes that walk allocas.
      if (buf.getResult().use_empty()) buf.erase();
    }
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Walk every function and unwrap its eval_in_mem ops.  Done
    // per-function so the counter scope is clear and we never collide
    // names across functions.
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      unsigned counter = 0;
      llvm::SmallVector<hlfir::EvaluateInMemoryOp, 4> ops;
      func.walk([&](hlfir::EvaluateInMemoryOp op) { ops.push_back(op); });
      for (auto op : ops) (void)rewriteOne(op, counter);
    }

    // Second walk: bypass the ``.result``-buffer pattern flang uses
    // for derived-type-by-value function returns (no eval_in_mem
    // involved).  This runs AFTER the eval_in_mem rewrites so any
    // save_result that the eval_in_mem unwrap exposed gets a
    // consistent treatment.
    rewriteResultBuffers(module);
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createUnwrapEvalInMemPass() {
  return std::make_unique<UnwrapEvalInMemPass>();
}

}  // namespace hlfir_bridge
