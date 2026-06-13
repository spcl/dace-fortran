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
#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace hlfir_bridge {

//
// Per-op dispatcher.  Owns:
//   * buildScfIfAsConditional  --  scf.if -> ASTNode kind=conditional.
//   * walkSCFBeforeRegion  --  the faithful scf.while walker.
//   * buildWhileNode  --  scf.while -> ASTNode kind=while.
//   * traceLoopIter  --  find a fir.do_loop's induction var.
//   * buildAST(Block&)  --  the per-op switch that walks an MLIR block,
//     picks the right shape builder for each hlfir.assign /
//     fir.do_loop / fir.if / etc., and wires alloc-alias
//     binds for fir.allocmem-bound stores.
//   * extractAST(ModuleOp)  --  public entry point; calls buildAST
//     on the first func.func body and returns the AST.
//
// This file is included verbatim from extract_ast.cpp via
// #include "bridge/ast/dispatch.cpp" and shares that translation
// unit's namespace, includes, and file-static state.  It MUST NOT be
// added to the build's compile list  --  CMakeLists.txt deliberately omits
// it.  The split is purely for readability: the AST builder used to
// be a single 2800-line file.

// Lower an ``scf.index_switch`` (the per-exit side-effect dispatch
// lift-cf-to-scf emits for a loop with a conditional EXIT/RETURN/GOTO that
// carries a side effect) into a chain of conditional ASTNodes keyed on the
// loop's exit-reason synth scalar.  Defined after ``buildAST`` (it walks the
// case regions with it); forward-declared here for ``walkSCFBeforeRegion``.
static std::vector<ASTNode> buildIndexSwitchNodes(mlir::scf::IndexSwitchOp sw);

static ASTNode buildScfIfAsConditional(mlir::scf::IfOp ifOp) {
  ASTNode c;
  c.kind = "conditional";
  c.condition = buildBoolExpr(ifOp.getCondition(), 0);

  auto walkArm = [&](mlir::Region& region) -> std::vector<ASTNode> {
    if (region.empty()) return {};
    auto arm = walkSCFBeforeRegion(region.front());
    // If the scf.if yields values, append one scalar_assign per result
    // reading the matching operand of the arm's scf.yield.
    if (ifOp.getNumResults() > 0) {
      mlir::scf::YieldOp yieldOp;
      for (auto& op : region.front())
        if (auto y = mlir::dyn_cast<mlir::scf::YieldOp>(op)) {
          yieldOp = y;
          break;
        }
      if (yieldOp) {
        for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
          auto target = scfSynthName(ifOp.getResult(i));
          auto expr = yieldedExpr(yieldOp.getOperand(i));
          ASTNode a;
          a.kind = "assign";
          a.target = target;
          a.expr = expr;
          a.target_is_array = false;
          arm.push_back(std::move(a));
        }
      }
    }
    return arm;
  };

  c.children = walkArm(ifOp.getThenRegion());
  if (!ifOp.getElseRegion().empty())
    c.else_children = walkArm(ifOp.getElseRegion());
  return c;
}

// Forward declaration used by ``walkSCFBeforeRegion``'s ``fir.do_loop``
// dispatch (definition lives further down the file).  ``traceLB`` and
// ``traceConstInt`` / ``buildIndexExpr`` come in via ``ast_helpers.h``.
static std::string traceLoopIter(fir::DoLoopOp loop);

std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block& block) {
  std::vector<ASTNode> out;
  // Snapshot the scf.condition's break value at the START of the body
  // -- BEFORE any nested scf.if mutates the SSA operands of the
  // condition.  Fortran's ``do; body; if (cond) exit; counter+=1;
  // end do`` shape (lift-cf-to-scf form) puts the increment INSIDE
  // the BEFORE region (in the else arm of a scf.if guarded by the
  // exit condition), then the scf.condition reads a SEPARATE cmpi
  // that the bridge's expression renderer re-evaluates at the
  // break-check state -- by then counter has been incremented and
  // the break fires one iteration too early.
  //
  // Detect the pattern up front: any ``scf.condition`` whose value
  // depends on a scalar that an in-body ``scf.if`` mutates needs a
  // pre-body snapshot.  Mint ``__brk_<N>`` (an interstate-edge
  // symbol via ``auto_declare_synth`` prefix), emit the snapshot
  // as the first assign in ``out``, and rewrite the
  // ``scf.condition`` handler to read the snapshot instead of
  // re-rendering the condition.
  //
  // NPB LU's ssor istep loop (``do istep = 1, niter; sweep;
  // if (rsdnm < tolrsd) return; end do``) has the same shape: each
  // istep iteration runs RHS + the sweep, then checks convergence
  // for the early return, then increments istep.  Without this
  // snapshot the SSOR sweep is effectively a no-op and residuals
  // stay at ~1e5 instead of converging to ~1e-2.
  static thread_local int kBrkCounter = 0;
  std::string brkSynthName;
  mlir::scf::ConditionOp pendingCondOp;
  for (auto& op : block) {
    if (auto condOp = mlir::dyn_cast<mlir::scf::ConditionOp>(op)) {
      pendingCondOp = condOp;
      break;
    }
  }
  if (pendingCondOp) {
    auto condVal = pendingCondOp.getCondition();
    auto b = buildBoolExpr(condVal, 0);
    if (!b.empty() && b != "?") {
      brkSynthName = "__brk_" + std::to_string(kBrkCounter++);
      ASTNode snap;
      snap.kind = "assign";
      snap.target = brkSynthName;
      snap.expr = b;
      snap.target_is_array = false;
      out.push_back(std::move(snap));
    }
  }
  for (auto& op : block) {
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      out.push_back(buildScfIfAsConditional(ifOp));
      continue;
    }
    // ``fir.if`` parked inside an ``scf.while`` BEFORE region.  Flang
    // emits ``fir.if`` for explicit Fortran ``IF (cond) THEN ... END
    // IF`` blocks; ``lift-cf-to-scf`` lowers cf.cond_br into ``scf.if``
    // but leaves the original ``fir.if`` ops as-is.  Without this
    // dispatch the op falls through as a "pure-value op" and the IF
    // body (any assigns / nested loops) is SILENTLY DROPPED from the
    // AST -- producing an SDFG with no writes from the gated block.
    // NPB LU's ``ssor`` istep loop hit this via
    // ``if (mod(istep, inorm) == 0 .or. istep == itmax) call l2norm(
    // ..., rsdnm)`` inside the istep do-loop with a following
    // ``if (rsdnm < tolrsd) return``: rsdnm's per-iter recompute was
    // dropped, rsdnm froze at the pre-loop value, residuals reported
    // pre-sweep state regardless of itmax.  See
    // ``tests/if_then_return_in_loop_elision_test.py`` for the
    // distilled repro.  The dispatch mirrors the toplevel ``fir.if``
    // handler at the bottom of ``buildAST`` (line 2246) -- same
    // shape, just reached via the scf.while walker.
    if (auto firIfOp = mlir::dyn_cast<fir::IfOp>(op)) {
      ASTNode n;
      n.kind = "conditional";
      n.condition = buildBoolExpr(firIfOp.getCondition(), 0);
      // Walk the condition's IR for array-element reads so the
      // Python emitter can lift the condition into a tasklet when
      // it references arrays.  Without this, ``emit_cond`` hoists
      // the rendered string to an interstate-edge assignment, but
      // DaCe treats array names there as bare Symbols (no
      // connector + no memlet) and the C++ codegen emits the data
      // pointer where it expected a scalar (graupel's
      // ``max(q_x_1, q_x_1, q_x_1)`` if_cond was the surfacer:
      // ``double* > 1e-15`` type-error).
      collectReadAccesses(firIfOp.getCondition(), n.accesses, 0);
      if (!firIfOp.getThenRegion().empty())
        n.children = buildAST(firIfOp.getThenRegion().front());
      if (!firIfOp.getElseRegion().empty())
        n.else_children = buildAST(firIfOp.getElseRegion().front());
      out.push_back(std::move(n));
      continue;
    }
    if (auto condOp = mlir::dyn_cast<mlir::scf::ConditionOp>(op)) {
      // Capture the enclosing scf.while's carried results into their synth
      // scalars (each iteration, so they hold the value at exit).  A post-loop
      // ``scf.index_switch`` reads these to run the side-effect of whichever
      // EXIT fired -- without this the exit reason is lost and the dispatched
      // side-effect (e.g. ``no_fall = .false.``) never happens.
      if (auto whileOp = condOp->getParentOfType<mlir::scf::WhileOp>()) {
        auto args = condOp.getArgs();
        for (unsigned i = 0; i < args.size() && i < whileOp.getNumResults();
             ++i) {
          ASTNode a;
          a.kind = "assign";
          a.target = scfSynthName(whileOp.getResult(i));
          a.expr = yieldedExpr(args[i]);
          a.target_is_array = false;
          out.push_back(std::move(a));
        }
      }
      // ``scf.condition(%c)``: break when %c is false.  Use the
      // pre-body snapshot ``__brk_<N>`` if we created one (see the
      // top-of-function block) so the break check sees the SSA-
      // operand values from the start of the iteration, not the
      // post-mutation values produced by an in-body scf.if.
      ASTNode guard;
      guard.kind = "conditional";
      if (!brkSynthName.empty()) {
        guard.condition = "not (" + brkSynthName + ")";
      } else {
        auto b = buildBoolExpr(condOp.getCondition(), 0);
        guard.condition = "not (" + b + ")";
      }
      ASTNode brk;
      brk.kind = "break";
      guard.children.push_back(std::move(brk));
      out.push_back(std::move(guard));
      continue;
    }
    if (auto sw = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op)) {
      auto chain = buildIndexSwitchNodes(sw);
      for (auto& c : chain) out.push_back(std::move(c));
      continue;
    }
    if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
      // Route through the normal assign dispatcher so copy/memset /
      // reduction / elemental shapes stay recognised inside the loop.
      auto src = assign.getOperand(0);
      auto dst = assign.getOperand(1);
      bool dst_is_array = isArrayRef(dst.getType());
      bool src_is_array = isArrayRef(src.getType());
      if (dst_is_array && src_is_array) {
        out.push_back(buildCopyNode(assign));
      } else if (dst_is_array && !src_is_array && isConstantZero(src)) {
        out.push_back(buildMemsetNode(assign));
      } else {
        out.push_back(buildAssignNode(assign));
      }
      continue;
    }
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
      // IV / counter bump stores Flang emits inside the lifted
      // scf.while body (``i = i + 1``, ``counter = counter - 1``).
      // Handled uniformly for declared vars and bare-alloca scratch
      // counters.
      auto memref = st.getMemref();
      auto target = traceToDecl(memref);
      if (target.empty())
        if (auto* md = memref.getDefiningOp())
          if (mlir::isa<fir::AllocaOp>(md)) target = allocaSynthName(memref);
      if (target.empty()) continue;
      auto expr = buildExpr(st.getValue(), 0);
      // Drop stores whose RHS we couldn't resolve.  These are almost
      // always Flang's implicit IV writeback at the end of a
      // ``fir.do_loop`` body: the stored value is a block arg of the
      // surrounding do-loop that buildExpr can't express on its own
      //  --  and the regular do-loop emitter already handles the IV
      // through ``initialize_expr`` / ``update_expr``.
      if (expr == "?") continue;
      ASTNode a;
      a.kind = "assign";
      a.target = target;
      a.expr = expr;
      a.target_is_array = false;
      out.push_back(std::move(a));
      continue;
    }
    // ``fir.do_loop`` parked inside an ``scf.while`` BEFORE region.
    // This is the shape ``lift-cf-to-scf`` produces from a Fortran
    // ``do`` whose containing istep loop has an early ``return``
    // (NPB LU's ``ssor``: the istep loop with ``if (rsdnm < tolrsd)
    // return``).  Without this dispatch the op falls through as a
    // "pure-value op" and EVERY assign in the loop body is silently
    // dropped from the AST.  Emit a minimal "loop" node and recurse
    // into the body block; the existing emit_loop fallbacks recover
    // bounds and iter from the SSA chain when loop_iter / loop_bound
    // are absent.
    if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      ASTNode n;
      n.kind = "loop";
      n.loop_iter = traceLoopIter(doLoop);
      if (auto c = traceConstInt(doLoop.getUpperBound())) {
        n.loop_bound = std::to_string(*c);
      } else {
        auto sym = traceToDecl(doLoop.getUpperBound());
        if (!sym.empty()) n.loop_bound = sym;
        else n.loop_bound = buildIndexExpr(doLoop.getUpperBound(), 0);
      }
      n.loop_lower = traceLB(doLoop.getLowerBound());
      if (n.loop_lower < 0) {
        auto sym = traceToDecl(doLoop.getLowerBound());
        if (!sym.empty()) n.loop_lower_expr = sym;
        else n.loop_lower_expr = buildIndexExpr(doLoop.getLowerBound(), 0);
      }
      if (auto stepC = traceConstInt(doLoop.getStep()))
        n.loop_step = *stepC;
      static thread_local int kSCFDoLoopIterCounter = 0;
      bool pushedBlockArg = false;
      auto& loopBlock = doLoop.getRegion().front();
      if (n.loop_iter.empty() && loopBlock.getNumArguments() > 0) {
        n.loop_iter = "_scfdoit_" + std::to_string(kSCFDoLoopIterCounter++);
        indexStack().push_back({loopBlock.getArgument(0), n.loop_iter});
        pushedBlockArg = true;
      }
      n.children = buildAST(loopBlock);
      if (pushedBlockArg) indexStack().pop_back();
      out.push_back(std::move(n));
      continue;
    }
    // Pure-value ops -- no AST node, their values flow inline through
    // SSA into the consuming side-effect op (which IS handled above).
    //
    // Defensive guard: if we reach here with an op that DOES carry
    // observable side effects (writes / nested side-effecting
    // regions), the per-op-type switch above is missing a handler
    // and the op's effects would be silently dropped from the SDFG.
    // The fir.if elision bug (NPB LU residuals stuck at pre-loop
    // state) was exactly this -- ``fir.if`` was missing from the
    // dispatch above and its IF body fell through here.
    //
    // MLIR's ``MemoryEffectOpInterface`` distinguishes pure-value
    // ops (arith.*, fir.load, hlfir.designate, ...) from
    // side-effecting ones; only the latter need a handler.  Throw
    // immediately rather than warn-and-continue: a warning could be
    // missed in CI noise and the SDFG would then silently compute
    // the wrong result.  An unhandled side-effecting op is a real
    // bridge gap that must be fixed by adding the corresponding
    // handler -- the error names the op so the gap surfaces at
    // parse time, not later in residual diffs.
    // Known-benign ops that carry no observable SDFG effect even
    // though they have nested regions or lack a MemoryEffect
    // interface.  Anything else is treated as a side-effect gap.
    auto opName = op.getName().getStringRef();
    static const llvm::StringSet<> kKnownBenign = {
        // Scope marker for dummy arguments -- pure metadata, no
        // runtime effect.
        "fir.dummy_scope",
    };
    if (kKnownBenign.contains(opName)) continue;

    auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op);
    bool hasWriteEffect = false;
    if (effects) {
      llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 4> instances;
      effects.getEffects(instances);
      for (auto& e : instances)
        if (mlir::isa<mlir::MemoryEffects::Write>(e.getEffect())) {
          hasWriteEffect = true;
          break;
        }
    } else if (op.getNumRegions() > 0) {
      // No MemoryEffectOpInterface but has nested regions -- the
      // regions themselves may carry effects (custom dialect ops
      // wrapping a body).  Treat as side-effecting for the guard;
      // allowlist via ``kKnownBenign`` above for legitimate markers.
      hasWriteEffect = true;
    }
    if (hasWriteEffect) {
      throw std::runtime_error(
          "walkSCFBeforeRegion: unhandled side-effecting op '" + opName.str() +
          "' inside an scf.while body.  Its effects would be silently "
          "dropped from the SDFG.  Add a handler in "
          "bridge/ast/dispatch.cpp::walkSCFBeforeRegion (see the "
          "fir::IfOp pattern at the top of the for-loop), or "
          "allowlist via ``kKnownBenign`` if the op is a pure marker.");
    }
  }
  return out;
}

static ASTNode buildWhileNode(mlir::scf::WhileOp whileOp) {
  ASTNode n;
  n.kind = "while";
  n.condition = "True";  // all break decisions live inside the body.

  if (whileOp.getBefore().empty()) return n;
  n.children = walkSCFBeforeRegion(whileOp.getBefore().front());
  return n;
}

static std::string traceLoopIter(fir::DoLoopOp loop) {
  for (auto& op : loop.getRegion().front())
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
      auto n = traceToDecl(st.getMemref());
      if (!n.empty()) return n;
    }
  return "";
}

// ---------------------------------------------------------------------------
// MPI point-to-point recognition
//
// Flang lowers ``call MPI_Send(...)`` to ``fir.call @_QPmpi_send(...)``
// (Fortran is case-insensitive -> flang lowercases the external name).
// There is no MLIR ``mpi`` dialect from flang, so we pattern-match the
// callee symbol and map it to a DaCe ``dace.libraries.mpi`` node.  The
// positional ABI (probed from real HLFIR) is:
//   mpi_send(buf, count, datatype, dest, tag, comm, ierr)
//   mpi_recv(buf, count, datatype, src,  tag, comm, status, ierr)
// ---------------------------------------------------------------------------

/// :returns: the normalised MPI op tag (``"mpi_send"`` / ``"mpi_recv"``
/// / ``"mpi_isend"`` / ``"mpi_irecv"`` / ``"mpi_wait"``) for a
/// recognised callee, else empty.
static std::string mpiCalleeTag(const std::string& callee) {
  std::string s = callee;
  if (!s.empty() && s[0] == '@') s.erase(0, 1);
  if (s.rfind("_QP", 0) == 0) s.erase(0, 3);  // external/global mangling
  std::string low = llvm::StringRef(s).lower();
  if (low == "mpi_send" || low == "mpi_recv" || low == "mpi_isend" ||
      low == "mpi_irecv" || low == "mpi_wait" || low == "mpi_alltoall")
    return low;
  return std::string{};
}

// ---------------------------------------------------------------------------
// BLAS / LAPACK recognition
//
// Pattern: Fortran source calls a vendor BLAS / LAPACK routine directly
// (e.g. ``call dgemm('N','N', m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)``).
// Flang lowers each as ``fir.call @dgemm_(...)`` (or ``@_QPdgemm`` depending
// on the binding flavour).  We pattern-match the callee by canonical name
// and the Python ``emit_blas`` / ``emit_lapack`` handlers stamp a
// :class:`dace.libraries.blas.*` / :class:`dace.libraries.lapack.*`
// library node with operand memlets.
//
// The recognised subset (first wave -- the highest-frequency routines in
// the cloudsc / ICON / QE working set):
//   BLAS L1: DAXPY / DSCAL / DDOT
//   BLAS L2: DGEMV
//   BLAS L3: DGEMM
//   LAPACK : DGETRF / DPOTRF
// Single-precision twins (S-prefix) are accepted alongside the D-prefix
// names so the same recognition path covers real32 callers; complex
// (C / Z) twins are out of scope for the first wave.

/// Strip Fortran mangling decorations and lowercase.
static std::string normaliseBlasName(const std::string& callee) {
  std::string s = callee;
  if (!s.empty() && s[0] == '@') s.erase(0, 1);
  if (s.rfind("_QP", 0) == 0) s.erase(0, 3);
  while (!s.empty() && s.back() == '_') s.pop_back();
  return llvm::StringRef(s).lower();
}

/// Return the canonical BLAS routine name (e.g. ``"dgemm"``) for a recognised
/// callee, else empty.  Accepts the D-prefix (real64) and S-prefix (real32)
/// names that map onto the same DaCe lib node (dtype-dispatched at expansion).
static std::string blasCalleeTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  static const std::set<std::string> recognised = {
      // L1
      "daxpy", "saxpy", "dscal", "sscal", "ddot", "sdot",
      "dnrm2", "snrm2", "dasum", "sasum", "idamax", "isamax",
      "dcopy", "scopy", "dswap", "sswap",
      // L2
      "dgemv", "sgemv", "dger", "sger",
      "dtrsv", "strsv", "dtrmv", "strmv", "dsymv", "ssymv",
      // L3
      "dgemm", "sgemm",
      "dtrsm", "strsm", "dtrmm", "strmm",
      "dsymm", "ssymm", "dsyrk", "ssyrk",
  };
  if (recognised.count(low)) return low;
  return std::string{};
}

/// Return the canonical LAPACK routine name (e.g. ``"dgetrf"``) for a
/// recognised callee, else empty.
static std::string lapackCalleeTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  static const std::set<std::string> recognised = {
      "dgetrf", "sgetrf", "dpotrf", "spotrf",
      "dpotrs", "spotrs",
      "dgeqrf", "sgeqrf", "dorgqr", "sorgqr",
  };
  if (recognised.count(low)) return low;
  return std::string{};
}

// ---------------------------------------------------------------------------
// Library-prefix "near-miss" detection
//
// When a Fortran call site matches the prefix of a recognised library
// (MPI / FFTW3 / BLAS / LAPACK) but the exact routine isn't in the
// supported subset, fall back to a clear ``unsupported_libcall`` ASTNode
// so the Python builder raises a precise ``NotImplementedError`` instead
// of silently emitting a broken generic call (``_out = ?`` -- the failure
// mode this layer prevents).

static const std::set<std::string>& knownBlasNames() {
  static const std::set<std::string> names = {
      // Real BLAS routines we *would* recognise once their handlers ship.
      "drotg", "srotg", "drot", "srot", "drotmg", "srotmg", "drotm", "srotm",
      "dnrm2", "snrm2", "scnrm2", "dznrm2",
      "dasum", "sasum", "scasum", "dzasum",
      "idamax", "isamax", "icamax", "izamax",
      "dswap", "sswap", "dcopy", "scopy",
      "dsdot",
      // Already recognised in ``blasCalleeTag`` -- listed so the detector
      // recognises THEM as BLAS and routes through the normal path.
      "daxpy", "saxpy", "dscal", "sscal", "ddot", "sdot",
      "dgemv", "sgemv", "dgemm", "sgemm",
      // BLAS L2 / L3 routines whose handlers are pending:
      "dger", "sger", "dsymv", "ssymv", "dsbmv", "ssbmv",
      "dtrmv", "strmv", "dtrsv", "strsv", "dgbmv", "sgbmv",
      "dsymm", "ssymm", "dsyrk", "ssyrk", "dsyr2k", "ssyr2k",
      "dtrmm", "strmm", "dtrsm", "strsm",
  };
  return names;
}

static const std::set<std::string>& knownLapackNames() {
  static const std::set<std::string> names = {
      "dgetrf", "sgetrf", "dgetri", "sgetri", "dgetrs", "sgetrs",
      "dgesv", "sgesv",
      "dpotrf", "spotrf", "dpotrs", "spotrs", "dpotri", "spotri",
      "dposv", "sposv",
      "dsyev", "ssyev", "dsyevd", "ssyevd",
      "dgeev", "sgeev", "dgesvd", "sgesvd",
      "dgeqrf", "sgeqrf", "dorgqr", "sorgqr", "dormqr", "sormqr",
  };
  return names;
}

// ---------------------------------------------------------------------------
// QE FFT-interfaces recognition
//
// Quantum ESPRESSO exposes a high-level generic FFT interface in
// ``FFTXlib/src/fft_interfaces.f90``::
//
//     CALL fwfft(fft_kind, f, dfft [, howmany])   ! G -> R
//     CALL invfft(fft_kind, f, dfft [, howmany])  ! R -> G
//
// The generic resolves to specific subroutines (``fwfft_y`` / ``invfft_y``
// for the standard grid, ``invfft_b`` for the box grid).  ``fft_kind`` is
// a literal character ('Rho' / 'Wave' / 'tgWave'), ``f`` is the (typically
// 1-D) complex buffer that gets transformed in place, and ``dfft`` is the
// :type:`fft_type_descriptor` that carries the 3-D grid sizes.
//
// For the bridge first cut we ignore ``dfft`` (the 3-D dims) and the
// optional ``howmany`` argument, and emit a single ``fftcall`` ASTNode
// referencing the buffer with the direction derived from the routine
// name.  The Python ``emit_fft`` handler stamps an :class:`FFT` /
// :class:`IFFT` library node with the buffer as the ``_inp`` / ``_out``
// operand.  This matches the recognition layer; correct multi-D FFT
// semantics (descriptor-driven dim extraction) is a follow-up gap.

/// Return ``"forward"`` for ``fwfft``-family callees, ``"backward"`` for
/// ``invfft``-family, else empty.  Accepts both the generic names
/// (``fwfft`` / ``invfft``) and the specific subroutines (``fwfft_y``,
/// ``invfft_y``, ``fwfft_b``, ``invfft_b``).
static std::string qeFftCalleeTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  // Strip QE module prefix (e.g. ``_QMfft_interfacesPinvfft_y`` after
  // ``normaliseBlasName`` drops the leading ``_QP``).  Be permissive about
  // the in-between ``_QM<mod>P``-style decorations.
  auto p = low.find('p');
  if (p != std::string::npos && low.rfind("_qm", 0) == 0)
    low = low.substr(p + 1);
  if (low == "fwfft" || low == "fwfft_y" || low == "fwfft_b") return "forward";
  if (low == "invfft" || low == "invfft_y" || low == "invfft_b") return "backward";
  return std::string{};
}

/// Returns ``"real"`` / ``"complex"`` for QE's ``fft_interpolate_real`` /
/// ``fft_interpolate_complex`` specific subroutines (the generic
/// ``fft_interpolate`` resolves to one of these), else empty.
static std::string qeFftInterpolateCalleeTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  auto p = low.find('p');
  if (p != std::string::npos && low.rfind("_qm", 0) == 0)
    low = low.substr(p + 1);
  if (low == "fft_interpolate_real") return "real";
  if (low == "fft_interpolate_complex") return "complex";
  return std::string{};
}

// ---------------------------------------------------------------------------
// QE parallel pencil-pipeline recognition
//
// QE's parallel 3-D FFT (``FFTXlib/src/fft_parallel.f90``) is a five-step
// pipeline of batched 1-D FFTs and MPI alltoalls:
//
//     cft_1z(f)              ! 1-D FFT along the z axis
//     fft_scatter_yz(f)      ! MPI alltoall across desc%comm3
//     cft_1y(f)              ! 1-D FFT along the y axis
//     fft_scatter_xy(f)      ! MPI alltoall across desc%comm2
//     cft_1x(f)              ! 1-D FFT along the x axis
//
// The bridge recognises the per-axis FFTs (``cft_1x`` / ``cft_1y`` /
// ``cft_1z``) and emits an :class:`FFT` lib node tagged with the axis,
// and recognises the scatter routines (``fft_scatter_xy`` / ``yz``) and
// emits an :class:`Alltoall` lib node.  Both are first-cut recognisers;
// the buffer-to-3-D-grid reinterpretation that fully captures QE's
// runtime semantics is a follow-up gap.

/// Returns ``"axis=X,dir=Y"`` (where X is 0/1/2 and Y is forward/backward)
/// for a recognised ``cft_1z`` / ``cft_1y`` / ``cft_1x`` callee, else
/// empty.  Axis assignment: ``cft_1x`` -> 0, ``cft_1y`` -> 1, ``cft_1z`` -> 2
/// (matches the C / row-major dimension order downstream code expects).
static std::string qePencilCalleeTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  auto p = low.find('p');
  if (p != std::string::npos && low.rfind("_qm", 0) == 0)
    low = low.substr(p + 1);
  if (low == "cft_1x") return "axis=0";
  if (low == "cft_1y") return "axis=1";
  if (low == "cft_1z") return "axis=2";
  return std::string{};
}

/// Build the ``fftcall`` ASTNode for a recognised QE pencil routine.
/// Signature: ``cft_1z(c, nsl, nz, ldz, isign, cout)``.  We treat
/// ``c`` (arg 0) as the input buffer and ``cout`` (arg 5) as the
/// output; the ``isign`` runtime sign is read literally when
/// available (and otherwise defaults to ``forward``).
static ASTNode buildQePencilCallNode(fir::CallOp call, const std::string& axisTag) {
  ASTNode n;
  auto args = call.getArgOperands();
  if (args.size() < 6) return n;
  std::string cin = traceToDecl(args[0]);
  std::string cout = traceToDecl(args[5]);
  if (cin.empty() || cout.empty()) return n;
  // isign: positive = backward, negative = forward.  When the literal
  // is unavailable we conservatively pick forward.
  std::string direction = "forward";
  if (auto c = traceConstInt(args[4])) {
    direction = (*c > 0) ? "backward" : "forward";
  }
  n.kind = "fftcall";
  n.callee = "fft_execute";
  n.expr = direction;
  n.target = cout;
  // ``call_args[0]`` / ``[1]`` are the in / out buffer names; subsequent
  // entries carry the axis tag so ``emit_fft`` can set ``node.axis``
  // when we later wire up axis-aware FFTW3 / cuFFT expansions.
  n.call_args = {cin, cout, axisTag};
  return n;
}

/// Returns ``"xy"`` or ``"yz"`` for QE's ``fft_scatter_xy`` / ``fft_scatter_yz``
/// alltoall transposes, else empty.
static std::string qeScatterCalleeTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  auto p = low.find('p');
  if (p != std::string::npos && low.rfind("_qm", 0) == 0)
    low = low.substr(p + 1);
  if (low == "fft_scatter_xy") return "xy";
  if (low == "fft_scatter_yz") return "yz";
  return std::string{};
}

/// Build the ``mpicall`` ASTNode for a recognised QE scatter routine.
/// Signature: ``fft_scatter_xy(desc, f_in, f_aux, nxx_, isgn[, comm])``.
/// We map this onto the existing ``mpi_alltoall`` ASTNode the Python
/// builder already lowers to :class:`Alltoall`; ``desc`` is the
/// descriptor (ignored at recognition), ``f_in`` (arg 1) is the send
/// buffer, ``f_aux`` (arg 2) is the receive buffer.
static ASTNode buildQeScatterCallNode(fir::CallOp call, const std::string& plane) {
  ASTNode n;
  auto args = call.getArgOperands();
  if (args.size() < 5) return n;
  std::string sendbuf = traceToDecl(args[1]);
  std::string recvbuf = traceToDecl(args[2]);
  if (sendbuf.empty() || recvbuf.empty()) return n;
  n.kind = "mpicall";
  n.callee = "mpi_alltoall";
  n.call_args = {sendbuf, recvbuf};
  // Carry the plane tag through ``expr`` so a downstream emitter could
  // wire the matching descriptor sub-communicator (desc%comm2 vs comm3);
  // the current ``emit_mpi`` Alltoall path ignores it and defaults to
  // ``MPI_COMM_WORLD``.
  n.expr = plane;
  return n;
}

/// Build an ``fftcall`` ASTNode for a recognised QE FFT call.  The buffer
/// is the 2nd argument (after the ``fft_kind`` literal).  We do not look
/// at the descriptor or ``howmany`` for the recognition first cut.
static ASTNode buildQeFftCallNode(fir::CallOp call, const std::string& direction) {
  ASTNode n;
  auto args = call.getArgOperands();
  if (args.size() < 2) return n;
  std::string buf = traceToDecl(args[1]);
  if (buf.empty()) return n;
  n.kind = "fftcall";
  n.callee = "fft_execute";
  n.expr = direction;
  n.target = buf;
  n.call_args = {buf, buf};  // in-place
  return n;
}

/// Returns "mpi" / "blas" / "lapack" / "fftw3" if the callee matches one of
/// those library's call conventions; else empty.  Used by the dispatch
/// loop's near-miss detector.
static std::string libraryFamilyTag(const std::string& callee) {
  std::string low = normaliseBlasName(callee);
  if (low.rfind("mpi_", 0) == 0) return "mpi";
  if (low.rfind("fftw_", 0) == 0 || low.rfind("fftwf_", 0) == 0) return "fftw3";
  if (knownBlasNames().count(low)) return "blas";
  if (knownLapackNames().count(low)) return "lapack";
  return std::string{};
}

/// Resolve a fir.call operand to either a decl name or a literal
/// (constant int) for the BLAS / LAPACK / MPI recognisers.  An empty
/// return means the operand is neither traceable to a declared name
/// nor a known constant -- the dispatch loop drops the recognition
/// gracefully.
static std::string resolveCallArg(mlir::Value v) {
  std::string n = traceToDecl(v);
  if (!n.empty()) return n;
  if (auto c = traceConstInt(v)) return std::to_string(*c);
  return std::string{};
}

/// Build the ``ASTNode`` for a recognised BLAS call.  ``call_args`` carry
/// the resolved decl / constant names in the routine's positional order
/// (drops the ``N`` / leading-dim args -- the lib node derives them from
/// memlets at expansion time).  Char-arg routines (DGEMM, DGEMV) capture
/// the ``TRANS`` literal in ``ASTNode.expr`` so the Python builder can
/// set the matching node property.
static ASTNode buildBlasCallNode(fir::CallOp call, const std::string& routine) {
  ASTNode n;
  n.kind = "blascall";
  n.callee = routine;
  auto args = call.getArgOperands();

  auto push = [&](mlir::Value v) {
    n.call_args.push_back(resolveCallArg(v));
  };

  if (routine == "daxpy" || routine == "saxpy") {
    // axpy(n, alpha, x, incx, y, incy) -- in-place y := alpha*x + y
    if (args.size() < 6) { n.kind.clear(); return n; }
    push(args[1]);  // alpha
    push(args[2]);  // x
    push(args[4]);  // y (inout)
    return n;
  }
  if (routine == "dscal" || routine == "sscal") {
    // scal(n, alpha, x, incx) -- in-place x := alpha*x
    if (args.size() < 4) { n.kind.clear(); return n; }
    push(args[1]);  // alpha
    push(args[2]);  // x
    return n;
  }
  if (routine == "ddot" || routine == "sdot") {
    // ddot(n, x, incx, y, incy) -- returns scalar; assigned by the user as
    // ``r = ddot(...)``.  The dot result is handled at the hlfir.assign
    // for the result variable; here we only emit nothing for the call
    // itself.  The result-name path is handled in the assign recogniser
    // (parallel to the FFTW3 plan-create case).
    n.kind.clear();
    return n;
  }
  if (routine == "dgemv" || routine == "sgemv") {
    // gemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    if (args.size() < 11) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]);  // trans char (literal)
    push(args[3]);  // alpha
    push(args[4]);  // A
    push(args[6]);  // x
    push(args[8]);  // beta
    push(args[9]);  // y (inout)
    return n;
  }
  if (routine == "dgemm" || routine == "sgemm") {
    // gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    if (args.size() < 13) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]) + "," + resolveCallArg(args[1]);  // transA,transB
    push(args[5]);   // alpha
    push(args[6]);   // A
    push(args[8]);   // B
    push(args[10]);  // beta
    push(args[11]);  // C (inout)
    return n;
  }
  if (routine == "dnrm2" || routine == "snrm2" ||
      routine == "dasum" || routine == "sasum" ||
      routine == "idamax" || routine == "isamax") {
    // ?nrm2(n, x, incx) / ?asum(n, x, incx) / i?amax(n, x, incx) -- scalar result;
    // the assign-side handler picks up the result variable.
    n.kind.clear();
    return n;
  }
  if (routine == "dcopy" || routine == "scopy") {
    // copy(n, x, incx, y, incy) -- y := x
    if (args.size() < 5) { n.kind.clear(); return n; }
    push(args[1]);  // x
    push(args[3]);  // y (out)
    return n;
  }
  if (routine == "dswap" || routine == "sswap") {
    // swap(n, x, incx, y, incy) -- x, y := y, x
    if (args.size() < 5) { n.kind.clear(); return n; }
    push(args[1]);  // x (inout)
    push(args[3]);  // y (inout)
    return n;
  }
  if (routine == "dger" || routine == "sger") {
    // ger(m, n, alpha, x, incx, y, incy, A, lda) -- A := alpha*x*y' + A
    if (args.size() < 9) { n.kind.clear(); return n; }
    push(args[2]);  // alpha
    push(args[3]);  // x
    push(args[5]);  // y
    push(args[7]);  // A (inout)
    return n;
  }
  if (routine == "dtrsv" || routine == "strsv" ||
      routine == "dtrmv" || routine == "strmv") {
    // trsv/trmv(uplo, trans, diag, n, A, lda, x, incx)
    if (args.size() < 8) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]) + "," + resolveCallArg(args[1]) + "," +
             resolveCallArg(args[2]);  // uplo,trans,diag
    push(args[4]);  // A
    push(args[6]);  // x (inout)
    return n;
  }
  if (routine == "dsymv" || routine == "ssymv") {
    // symv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    if (args.size() < 10) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]);  // uplo
    push(args[2]);  // alpha
    push(args[3]);  // A
    push(args[5]);  // x
    push(args[7]);  // beta
    push(args[8]);  // y (inout)
    return n;
  }
  if (routine == "dtrsm" || routine == "strsm" ||
      routine == "dtrmm" || routine == "strmm") {
    // trsm/trmm(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
    if (args.size() < 11) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]) + "," + resolveCallArg(args[1]) + "," +
             resolveCallArg(args[2]) + "," + resolveCallArg(args[3]);
    push(args[6]);  // alpha
    push(args[7]);  // A
    push(args[9]);  // B (inout)
    return n;
  }
  if (routine == "dsymm" || routine == "ssymm") {
    // symm(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
    if (args.size() < 12) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]) + "," + resolveCallArg(args[1]);
    push(args[4]);  // alpha
    push(args[5]);  // A
    push(args[7]);  // B
    push(args[9]);  // beta
    push(args[10]); // C (inout)
    return n;
  }
  if (routine == "dsyrk" || routine == "ssyrk") {
    // syrk(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    if (args.size() < 10) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]) + "," + resolveCallArg(args[1]);
    push(args[4]);  // alpha
    push(args[5]);  // A
    push(args[7]);  // beta
    push(args[8]);  // C (inout)
    return n;
  }
  n.kind.clear();
  return n;
}

/// Build the ``ASTNode`` for a recognised LAPACK call.
static ASTNode buildLapackCallNode(fir::CallOp call, const std::string& routine) {
  ASTNode n;
  n.kind = "lapackcall";
  n.callee = routine;
  auto args = call.getArgOperands();

  if (routine == "dgetrf" || routine == "sgetrf") {
    // getrf(m, n, A, lda, ipiv, info) -- LU factorisation
    if (args.size() < 6) { n.kind.clear(); return n; }
    n.call_args.push_back(resolveCallArg(args[2]));  // A (inout: factor in place)
    n.call_args.push_back(resolveCallArg(args[4]));  // ipiv (out)
    n.call_args.push_back(resolveCallArg(args[5]));  // info (out)
    return n;
  }
  if (routine == "dpotrf" || routine == "spotrf") {
    // potrf(uplo, n, A, lda, info) -- Cholesky factorisation
    if (args.size() < 5) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]);  // 'U' or 'L'
    n.call_args.push_back(resolveCallArg(args[2]));  // A (inout)
    n.call_args.push_back(resolveCallArg(args[4]));  // info (out)
    return n;
  }
  if (routine == "dpotrs" || routine == "spotrs") {
    // potrs(uplo, n, nrhs, A, lda, B, ldb, info)
    if (args.size() < 8) { n.kind.clear(); return n; }
    n.expr = resolveCallArg(args[0]);                // 'U' or 'L'
    n.call_args.push_back(resolveCallArg(args[3]));  // A
    n.call_args.push_back(resolveCallArg(args[5]));  // B (inout)
    n.call_args.push_back(resolveCallArg(args[7]));  // info (out)
    return n;
  }
  if (routine == "dgeqrf" || routine == "sgeqrf") {
    // geqrf(m, n, A, lda, tau, work, lwork, info)
    if (args.size() < 8) { n.kind.clear(); return n; }
    n.call_args.push_back(resolveCallArg(args[2]));  // A (inout)
    n.call_args.push_back(resolveCallArg(args[4]));  // tau (out)
    n.call_args.push_back(resolveCallArg(args[7]));  // info (out)
    return n;
  }
  if (routine == "dorgqr" || routine == "sorgqr") {
    // orgqr(m, n, k, A, lda, tau, work, lwork, info)
    if (args.size() < 9) { n.kind.clear(); return n; }
    n.call_args.push_back(resolveCallArg(args[3]));  // A (inout)
    n.call_args.push_back(resolveCallArg(args[5]));  // tau (in)
    n.call_args.push_back(resolveCallArg(args[8]));  // info (out)
    return n;
  }
  n.kind.clear();
  return n;
}

/// Build an ``mpicall`` ASTNode for a recognised MPI point-to-point
/// call.  ``call_args`` layout (the names the Python builder wires to
/// the DaCe library node's connectors):
///   * send / recv:    ``[buffer, partner(dest|src), tag]``
///   * isend / irecv:  ``[buffer, partner(dest|src), tag, request]``
///   * wait:           ``[request]``
/// A non-default (runtime / user) communicator is appended as one extra
/// trailing entry (``[..., comm]``); the default ``MPI_COMM_WORLD`` adds
/// nothing, so the optional comm is unambiguous per callee from the
/// ``call_args`` length (send/recv 3 vs 4, isend/irecv 4 vs 5).  The
/// Python builder lowers the trailing comm to an ``opaque(MPI_Comm)``
/// ``_comm`` connector; the c-binding wrapper does ``MPI_Comm_f2c`` on
/// the Fortran integer handle.
/// count is taken from the buffer memlet downstream; datatype / status
/// / ierr are not modelled (DaCe derives the MPI datatype from the
/// buffer and uses ``MPI_STATUS_IGNORE``).  Positional ABI:
///   send (buf,count,dt,dest,tag,comm,ierr)
///   recv (buf,count,dt,src,tag,comm,status,ierr)
///   isend(buf,count,dt,dest,tag,comm,request,ierr)
///   irecv(buf,count,dt,src,tag,comm,request,ierr)
///   wait (request,status,ierr)
static ASTNode buildMpiCallNode(fir::CallOp call, const std::string& mpiOp) {
  ASTNode n;
  n.kind = "mpicall";
  n.callee = mpiOp;
  auto args = call.getArgOperands();
  auto resolve = [&](mlir::Value v, const char* what) -> std::string {
    auto nm = traceToDecl(v);
    if (nm.empty())
      throw std::runtime_error("MPI " + mpiOp + ": cannot resolve the " +
                               std::string(what) + " argument to a name");
    return nm;
  };

  if (mpiOp == "mpi_wait") {
    if (args.empty())
      throw std::runtime_error("MPI mpi_wait: no request argument");
    n.call_args = {resolve(args[0], "request")};
    return n;
  }

  if (mpiOp == "mpi_alltoall") {
    // MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount,
    //              recvtype, comm, ierr)
    if (args.size() < 7)
      throw std::runtime_error("MPI mpi_alltoall: unexpected argument count " +
                               std::to_string(args.size()));
    std::string sendbuf = resolve(args[0], "sendbuf");
    std::string recvbuf = resolve(args[3], "recvbuf");
    n.call_args = {sendbuf, recvbuf};
    // comm decoding -- same rule as send/recv.  Append on a runtime/user comm.
    std::string commName = traceToDecl(args[6]);
    std::string low = llvm::StringRef(commName).lower();
    bool isDefault = commName.empty() || low.rfind("__", 0) == 0 ||
                     low.find("mpi_comm_world") != std::string::npos;
    if (!isDefault) n.call_args.push_back(commName);
    return n;
  }

  if (args.size() < 6)
    throw std::runtime_error("MPI " + mpiOp + ": unexpected argument count " +
                             std::to_string(args.size()));
  bool isSendLike = (mpiOp == "mpi_send" || mpiOp == "mpi_isend");
  std::string buf = resolve(args[0], "buffer");
  std::string partner = resolve(args[3], isSendLike ? "dest" : "src");
  std::string tag = resolve(args[4], "tag");

  // comm (arg 5).  Flang materialises a ``parameter`` / literal
  // ``MPI_COMM_WORLD`` as a compiler-synthetic entity (``__assoc_scalar_*``
  // -- a Fortran user identifier can never start with ``_``); ``use mpi``
  // exposes it as an entity literally named ``mpi_comm_world``; an
  // un-nameable operand is a bare folded constant.  Those three are the
  // default WORLD (nothing appended -- DaCe emits ``MPI_COMM_WORLD``).
  // A real named variable (a dummy ``comm`` / ``MPI_Comm_split`` result)
  // is a runtime/user communicator: append it so the builder threads an
  // ``opaque(MPI_Comm)`` ``_comm`` connector into the libnode.
  std::string commName = traceToDecl(args[5]);
  std::string low = llvm::StringRef(commName).lower();
  bool isDefault = commName.empty() || low.rfind("__", 0) == 0 ||
                   low.find("mpi_comm_world") != std::string::npos;

  n.call_args = {buf, partner, tag};
  if (mpiOp == "mpi_isend" || mpiOp == "mpi_irecv") {
    if (args.size() < 7)
      throw std::runtime_error("MPI " + mpiOp +
                               ": expected a request argument");
    n.call_args.push_back(resolve(args[6], "request"));
  }
  if (!isDefault) n.call_args.push_back(commName);
  return n;
}

// ---------------------------------------------------------------------------
// Fortran I/O recognition
// ---------------------------------------------------------------------------
//
// Flang lowers Fortran I/O to a sequence of ``fir.call @_FortranAio*`` runtime
// calls.  We fold an ``open`` / ``read`` / ``write`` / ``close`` region into a
// single ``kind="iocall"`` ASTNode the Python builder lowers to a
// ``dace.libraries.fortran_io`` node.  The filename must be a compile-time
// literal (a ``@_QQclX<hex>`` constant) -- it is baked into the node as
// ``target`` because DaCe cannot pass a string at runtime; a non-literal
// filename is unsupported (the statement is dropped, as is any I/O with no
// associated file such as a ``write`` to stdout).

/// Decode a flang character-literal global symbol ``_QQclX<hex>`` to its text
/// (the hex digits are the literal's bytes).  Empty if not that form.
static std::string decodeCharLiteralSymbol(llvm::StringRef sym) {
  if (!sym.consume_front("_QQclX")) return {};
  auto hexVal = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  std::string out;
  for (size_t i = 0; i + 1 < sym.size() + 1 && i + 1 <= sym.size(); i += 2) {
    int hi = hexVal(sym[i]), lo = hexVal(sym[i + 1]);
    if (hi < 0 || lo < 0) return {};
    out.push_back(static_cast<char>(hi * 16 + lo));
  }
  // Namelist group / member name literals carry a trailing NUL the filename
  // / status literals do not; drop it so the decoded name is a clean token.
  while (!out.empty() && out.back() == '\0') out.pop_back();
  return out;
}

/// Resolve a ``SetFile`` operand to the string literal it addresses, tracing
/// through the ``fir.convert`` / ``fir.coordinate_of`` shims and the
/// ``hlfir.declare`` flang wraps the ``fir.address_of`` in.  Empty if the
/// operand is not a literal (a runtime filename, which we cannot bake in).
static std::string ioLiteralString(mlir::Value v) {
  for (int i = 0; i < limits::kSsaBackWalkDepth && v; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
    if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d)) { v = co.getRef(); continue; }
    if (auto hd = mlir::dyn_cast<hlfir::DeclareOp>(d)) { v = hd.getMemref(); continue; }
    if (auto fd = mlir::dyn_cast<fir::DeclareOp>(d)) { v = fd.getMemref(); continue; }
    if (auto ad = mlir::dyn_cast<fir::AddrOfOp>(d))
      return decodeCharLiteralSymbol(ad.getSymbol().getRootReference().str());
    break;
  }
  return {};
}

/// The value stored into the alloca ``ref`` (the ``fir.store`` whose memref is
/// ``ref``), or null.  Flang materialises the namelist descriptor / its member
/// array on the stack and stores the built-up aggregate; this recovers it.
static mlir::Value ioStoredValue(mlir::Value ref) {
  for (auto* user : ref.getUsers())
    if (auto st = mlir::dyn_cast<fir::StoreOp>(user))
      if (st.getMemref() == ref) return st.getValue();
  return {};
}

/// Walk the ``fir.insert_value`` chain defining ``agg`` and return the value
/// inserted at the single-element index ``idx`` (null if absent).
static mlir::Value ioInsertedAt(mlir::Value agg, int64_t idx) {
  for (int i = 0; i < limits::kSsaBackWalkDepth && agg; ++i) {
    auto iv = mlir::dyn_cast_or_null<fir::InsertValueOp>(agg.getDefiningOp());
    if (!iv) break;
    auto coor = iv.getCoor();
    if (coor.size() == 1)
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(coor[0]))
        if (ia.getInt() == idx) return iv.getVal();
    agg = iv.getAdt();
  }
  return {};
}

/// Extract a namelist group's name and member names from the descriptor the
/// ``InputNamelist`` call receives.  The descriptor is a stack tuple
/// ``{groupName, count, members, defIoTable}``; ``members`` is an array of
/// ``{name, box}`` pairs built by a second insert-value chain.  A member's
/// name is its Fortran variable name, which is also its SDFG array name.
static void extractNamelist(mlir::Value descRef, std::string& group, std::vector<std::string>& members) {
  for (int i = 0; i < limits::kSsaBackWalkDepth && descRef; ++i) {
    auto* d = descRef.getDefiningOp();
    if (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(d)) { descRef = cv.getValue(); continue; }
    break;
  }
  mlir::Value top = ioStoredValue(descRef);
  if (!top) return;
  group = ioLiteralString(ioInsertedAt(top, 0));
  mlir::Value memberArr = ioStoredValue(ioInsertedAt(top, 2));
  if (!memberArr) return;
  // Member-array insert chain: ``[i, 0]`` is member i's name literal.  The
  // chain is built bottom-up, so collect (index, name) and sort ascending.
  std::vector<std::pair<int64_t, std::string>> byIndex;
  for (int i = 0; i < limits::kSsaBackWalkDepth && memberArr; ++i) {
    auto iv = mlir::dyn_cast_or_null<fir::InsertValueOp>(memberArr.getDefiningOp());
    if (!iv) break;
    auto coor = iv.getCoor();
    if (coor.size() == 2)
      if (auto slot = mlir::dyn_cast<mlir::IntegerAttr>(coor[1]))
        if (slot.getInt() == 0)
          if (auto mi = mlir::dyn_cast<mlir::IntegerAttr>(coor[0])) {
            std::string nm = ioLiteralString(iv.getVal());
            if (!nm.empty()) byIndex.emplace_back(mi.getInt(), nm);
          }
    memberArr = iv.getAdt();
  }
  std::sort(byIndex.begin(), byIndex.end());
  for (auto& [idx, nm] : byIndex) members.push_back(nm);
}

/// State threaded across the consecutive ``_FortranAio*`` calls of one
/// open/read/write/close region (all in one block, in program order).
/// ``open_file`` holds the filename from the most recent ``open`` until its
/// ``close``; ``op`` / ``stmt_file`` / ``items`` accumulate the in-flight
/// transfer statement between its ``Begin*`` and ``EndIoStatement``.
struct IoState {
  std::string open_file;
  std::string op;
  std::string stmt_file;
  std::string group;  // namelist group name (op == "namelist_read")
  std::vector<std::string> items;
};

// ---------------------------------------------------------------------------
// FFTW3 plan recognition
//
// Pattern this consumes -- the standard FFTW3 C ABI driven from Fortran via
// ``iso_c_binding`` (and the FFTW-compat ABI of cuFFT / MKL):
//
//     plan = fftw_plan_dft_2d(N, M, in, out, FFTW_FORWARD, FFTW_ESTIMATE)
//     call fftw_execute_dft(plan, in, out)
//     call fftw_destroy_plan(plan)
//
// The opaque ``TYPE(C_PTR) :: plan`` SSA value cannot be modeled in DaCe.
// We therefore (1) recognise the three calls by callee name, (2) capture
// the (rank, dims, direction) at the plan-create site and stash it under
// the destination variable name, and (3) on the ``execute_dft`` call
// emit a single ``fftcall`` ``ASTNode`` carrying the input/output array
// names + the looked-up direction.  The plan-create and destroy calls
// are dropped (no ASTNode emitted) -- the FFT lib node's expansion
// owns the plan lifecycle.
struct FftPlanInfo {
  int rank;                       // 2 or 3
  std::vector<std::string> dims;  // dimension expressions / literals
  std::string direction;          // "forward" or "backward"
};

/// Return the normalised tag for a recognised FFTW3 callee, else empty.
/// Accepts both ``fftw_*`` (double precision) and ``fftwf_*`` (single).
static std::string fftw3CalleeTag(const std::string& callee) {
  std::string s = callee;
  if (!s.empty() && s[0] == '@') s.erase(0, 1);
  if (s.rfind("_QP", 0) == 0) s.erase(0, 3);  // external/global mangling
  // Strip optional trailing underscore (older Fortran external mangling).
  while (!s.empty() && s.back() == '_') s.pop_back();
  std::string low = llvm::StringRef(s).lower();
  if (low == "fftw_plan_dft_2d" || low == "fftwf_plan_dft_2d") return "fft_plan_2d";
  if (low == "fftw_plan_dft_3d" || low == "fftwf_plan_dft_3d") return "fft_plan_3d";
  if (low == "fftw_execute_dft" || low == "fftwf_execute_dft") return "fft_execute";
  if (low == "fftw_destroy_plan" || low == "fftwf_destroy_plan") return "fft_destroy";
  return std::string{};
}

/// Build the ``ASTNode`` for a recognised FFTW3 call.  Returns an empty
/// ``ASTNode`` (kind="") to mean "consumed, emit nothing"; the dispatch
/// loop then ``continue``-s without pushing.
///
/// ``plans`` is threaded across the block so the execute call can look
/// up the (rank, dims, direction) recorded at the plan-create site by
/// destination variable name.
static ASTNode buildFftw3CallNode(fir::CallOp call, const std::string& fftOp,
                                  std::map<std::string, FftPlanInfo>& plans) {
  ASTNode n;
  auto args = call.getArgOperands();

  if (fftOp == "fft_plan_2d" || fftOp == "fft_plan_3d") {
    // Signature: fftw_plan_dft_{2,3}d(n0, n1[, n2], in, out, sign, flags)
    int rank = (fftOp == "fft_plan_2d") ? 2 : 3;
    if ((int)args.size() < rank + 4) return n;  // safety -- malformed call
    FftPlanInfo info;
    info.rank = rank;
    for (int i = 0; i < rank; ++i) {
      std::string dim;
      if (auto c = traceConstInt(args[i])) dim = std::to_string(*c);
      else dim = traceToDecl(args[i]);
      info.dims.push_back(dim);
    }
    // Sign: FFTW_FORWARD = -1, FFTW_BACKWARD = +1.
    int sign = 0;
    if (auto c = traceConstInt(args[rank + 2])) sign = (int)*c;
    info.direction = (sign == -1) ? "forward" : "backward";
    // The plan-create call returns the plan; track which user variable
    // it is stored into so the matching execute can look it up.
    std::string planVar;
    for (auto u : call.getResult(0).getUsers()) {
      if (auto store = mlir::dyn_cast<fir::StoreOp>(u)) {
        planVar = traceToDecl(store.getMemref());
        if (!planVar.empty()) break;
      }
    }
    if (!planVar.empty()) plans[planVar] = info;
    return n;  // consumed -- emit nothing
  }

  if (fftOp == "fft_execute") {
    // Signature: fftw_execute_dft(plan, in, out)
    if (args.size() < 3) return n;
    std::string planVar = traceToDecl(args[0]);
    std::string inArr = traceToDecl(args[1]);
    std::string outArr = traceToDecl(args[2]);
    if (planVar.empty() || inArr.empty()) return n;
    auto it = plans.find(planVar);
    if (it == plans.end()) return n;  // unknown plan -- give up cleanly
    n.kind = "fftcall";
    n.callee = "fft_execute";
    n.target = outArr.empty() ? inArr : outArr;
    n.expr = it->second.direction;  // "forward" or "backward"
    n.call_args.push_back(inArr);
    n.call_args.push_back(outArr.empty() ? inArr : outArr);
    for (auto& d : it->second.dims) n.call_args.push_back(d);
    return n;
  }

  // fft_destroy: drop -- plan lifecycle is owned by the lib node expansion.
  return n;
}

/// Advance the I/O state machine by one ``_FortranAio*`` ``call`` (callee
/// ``c``), and on a completed read/write statement push an ``iocall`` ASTNode.
/// Every such call is consumed by the caller, so none leaks out as a generic
/// ``call`` node (its char-literal operands would otherwise mint invalid
/// arrays).  Only statements with a literal file and >=1 data item are
/// emitted -- a stdout write or a runtime-named file has no bindable filename
/// and is dropped.
static void recognizeIoCall(fir::CallOp call, llvm::StringRef c, IoState& s, std::vector<ASTNode>& nodes) {
  auto args = call.getArgOperands();
  if (c.contains("SetFile")) {
    if (args.size() >= 2) s.open_file = ioLiteralString(args[1]);
  } else if (c.contains("BeginExternalListInput") || c.contains("BeginExternalFormattedInput")) {
    s.op = "read";
    s.stmt_file = s.open_file;
    s.items.clear();
  } else if (c.contains("BeginExternalListOutput") || c.contains("BeginExternalFormattedOutput")) {
    s.op = "write";
    s.stmt_file = s.open_file;
    s.items.clear();
  } else if (c.contains("InputNamelist") || c.contains("OutputNamelist")) {
    // ``read(u, nml=grp)`` / ``write(u, nml=grp)``: the descriptor (operand 1)
    // yields the group name + member (= variable = array) names.
    s.op = c.contains("Input") ? "namelist_read" : "namelist_write";
    s.group.clear();
    s.items.clear();
    if (args.size() >= 2) extractNamelist(args[1], s.group, s.items);
  } else if ((c.contains("Input") || c.contains("Output")) && !c.contains("Begin") && !c.contains("Ascii")) {
    // A transfer item: Input/OutputDescriptor(box) or the scalar
    // Input/OutputReal*/Integer*(ref); operand 1 is the data.
    if (!s.op.empty() && args.size() >= 2) {
      std::string nm = traceToDecl(args[1]);
      if (!nm.empty()) s.items.push_back(nm);
    }
  } else if (c.contains("BeginClose")) {
    s.open_file.clear();
  } else if (c.contains("EndIoStatement") && !s.op.empty()) {
    if (!s.stmt_file.empty() && !s.items.empty()) {
      // Fuse with an immediately-preceding transfer of the SAME op to the
      // SAME file: consecutive ``read`` / ``write`` statements between one
      // ``open`` and ``close`` then share a single open, so sequential
      // reads advance the file position instead of each re-opening from the
      // start.  Each item still lowers to its own ``read``/``write`` call (a
      // fresh record), matching list-directed statement semantics.  Only
      // fused when adjacent (``nodes.back()`` is that transfer) so any
      // intervening computation forces a separate open -- preserving order.
      // Namelist transfers carry a group name in ``expr`` and never fuse.
      bool fused = false;
      if ((s.op == "read" || s.op == "write") && s.group.empty() &&
          !nodes.empty() && nodes.back().kind == "iocall" &&
          nodes.back().callee == s.op && nodes.back().target == s.stmt_file &&
          nodes.back().expr.empty()) {
        for (auto& it : s.items) nodes.back().call_args.push_back(it);
        fused = true;
      }
      if (!fused) {
        ASTNode io;
        io.kind = "iocall";
        io.callee = s.op;
        io.target = s.stmt_file;
        io.expr = s.group;  // namelist group name (empty for list-directed)
        io.call_args = s.items;
        nodes.push_back(std::move(io));
      }
    }
    s.op.clear();
    s.stmt_file.clear();
    s.group.clear();
    s.items.clear();
  }
}

// ---------------------------------------------------------------------------
// Block walker
// ---------------------------------------------------------------------------

std::vector<ASTNode> buildAST(mlir::Block& block) {
  std::vector<ASTNode> nodes;

  // Bind / advance the alloc-alias for an ``ALLOCATE``-bound store into an
  // allocatable's box descriptor (``fir.store (fir.embox-of-fir.allocmem) to
  // <decl_box_ref>``).  The target buffer is the site's CLASS buffer (see
  // ``groupAllocSites``), not a per-store counter, so conditional branches
  // and sequential re-allocation route correctly.
  // Returns the allocatable's raw Fortran name on a successful match
  // (so the caller can emit a state-change ``<name>_allocated = 1``
  // ASTNode), empty string otherwise.
  auto bindAllocSite = [&](mlir::Operation* op) -> std::string {
    auto store = mlir::dyn_cast<fir::StoreOp>(op);
    if (!store) return {};
    auto valDef = store.getValue().getDefiningOp();
    if (!valDef) return {};
    auto embox = mlir::dyn_cast<fir::EmboxOp>(valDef);
    if (!embox) return {};
    auto allocmem = mlir::dyn_cast_or_null<fir::AllocMemOp>(
        embox.getMemref().getDefiningOp());
    if (!allocmem) return {};
    // Only the user-visible allocs we model  --  skip embox-of-zero_bits
    // (the empty-init store the bridge already filters out elsewhere).
    auto un = allocmem.getUniqName();
    if (!un || !un->ends_with(".alloc")) return {};
    auto memDef = store.getMemref().getDefiningOp();
    if (!memDef) return {};
    auto decl = mlir::dyn_cast<hlfir::DeclareOp>(memDef);
    if (!decl) return {};
    std::string raw = extractName(decl.getUniqName().str());
    if (raw.empty()) return {};
    // Route this ALLOCATE to its buffer CLASS (one DaCe transient per
    // class).  Sequential re-allocation -> a singleton class per epoch
    // (``a``, ``a_alloc1``, ...).  A conditional ALLOCATE -> a multi-site
    // class: every branch site shares ONE buffer and assigns its
    // branch-dependent extent symbol (``<buf>_d<i> = <this branch's
    // extent>``) here, in the branch; the writes merge at the IF join and
    // bind the shape.  See ALLOC_BUFFER_SSA_DESIGN.md.
    auto mod = decl->getParentOfType<mlir::ModuleOp>();
    auto classes = mod ? groupAllocSites(decl.getUniqName().str(), mod)
                       : std::vector<std::vector<fir::AllocMemOp>>{};
    unsigned cls = 0;
    for (unsigned ci = 0; ci < classes.size(); ++ci)
      for (auto site : classes[ci])
        if (site.getOperation() == allocmem.getOperation()) cls = ci;
    std::string bufName = allocAliasName(raw, cls);
    setAllocAlias(raw, bufName);
    // Bind the buffer's per-dim extent symbol ``<buf>_d<i>`` here, at the
    // site, from the ALLOCATE's own shape operand.  Every ALLOCATE passes
    // its dimensions (``allocate(a(n))`` -> extent ``n``), so we always know
    // them: assigning the symbol here keeps it from leaking onto the program
    // signature as a free symbol and lets ``size(a)`` / ``LBOUND`` / ``UBOUND``
    // (which lower to ``fir.box_dims`` rendered as ``<buf>_d<i>``) resolve --
    // for the base buffer, conditional branches, and versioned re-allocations
    // alike.
    {
      unsigned d = 0;
      for (auto sz : allocmem.getShape()) {
        std::string ext = traceExtentExpr(sz);
        if (!ext.empty()) {
          ASTNode an;
          an.kind = "assign";
          an.target = bufName + "_d" + std::to_string(d);
          an.target_is_array = false;
          an.expr = ext;
          nodes.push_back(std::move(an));
        }
        ++d;
      }
    }
    // Mint a position symbol for every constant-indexed element in the
    // allocation's shape (``allocate(buf(max(dims(1), dims(2))))``) so each
    // one gets a ``symbol_init`` -- even an element that appears only in
    // the extent and nowhere else (no loop bound / index would otherwise
    // mint it, leaving the shape symbol an unbound program argument).
    for (auto sz : allocmem.getShape())
      forEachConstIndexedElement(
          sz, [](const std::string& arr, const std::vector<int64_t>& idxs) {
            internPosSymbol(arr, idxs);
          });
    return raw;
  };
  auto emitAllocStateChange = [&](const std::string& name, int value) {
    ASTNode n;
    n.kind = "assign";
    n.target = name + "_allocated";
    n.target_is_array = false;
    n.expr = std::to_string(value);
    nodes.push_back(std::move(n));
  };
  IoState io_state;  // threaded across this block's ``_FortranAio*`` calls
  // FFT plan info keyed by plan variable name (see ``fftw3CalleeTag``).
  std::map<std::string, FftPlanInfo> fft_plans;
  for (auto& op : block) {
    // Bind / advance the alloc-alias for this allocatable, then
    // emit a state-change ``<name>_allocated = 1`` so downstream
    // ``ALLOCATED(arr)`` reads see the right value.  The ALLOCATE
    // store itself produces no other observable side effect in the
    // SDFG model  --  we treat allocatables as live for the whole
    // scope.
    if (auto allocName = bindAllocSite(&op); !allocName.empty()) {
      emitAllocStateChange(allocName, 1);
      continue;
    }

    // Standalone ``fir.freemem``  --  Flang's DEALLOCATE expansion at
    // top level (the trailing ``fir.if (alloc_status != 0) { ... }``
    // is the implicit end-of-scope cleanup, handled separately as
    // ``isAllocCleanup``).  Trace through ``fir.box_addr`` and
    // ``fir.load`` to find the underlying ``hlfir.declare`` and
    // emit ``<rawname>_allocated = 0`` against the declare's RAW
    // Fortran name (NOT the current alloc-alias) so multi-site
    // allocatables ``x -> x_alloc1 -> x_alloc2`` still funnel state
    // updates through the original ``x_allocated`` symbol.
    if (auto fm = mlir::dyn_cast<fir::FreeMemOp>(&op)) {
      mlir::Value cur = fm.getHeapref();
      for (int i = 0; i < limits::kConvertChainDepth && cur; ++i) {
        auto* cd = cur.getDefiningOp();
        if (!cd) break;
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(cd)) {
          cur = cv.getValue();
          continue;
        }
        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(cd)) {
          cur = ba.getVal();
          continue;
        }
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(cd)) {
          cur = ld.getMemref();
          continue;
        }
        break;
      }
      std::string name;
      if (cur)
        if (auto* cd = cur.getDefiningOp())
          if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(cd))
            name = extractName(decl.getUniqName().str());
      if (!name.empty()) emitAllocStateChange(name, 0);
      continue;
    }

    if (auto doLoop = mlir::dyn_cast<fir::DoLoopOp>(op)) {
      ASTNode n;
      n.kind = "loop";
      n.loop_iter = traceLoopIter(doLoop);
      // Bound resolution.  ``traceToDecl`` is a useful shortcut for
      // the scalar-variable case (``DO i = 1, n`` where ``n`` is a
      // dummy scalar) but it is wrong for array-element loads:
      // ``DO j = row_ptr(i), row_ptr(i+1)-1`` would otherwise resolve
      // to the bare name ``row_ptr`` because the load chain bottoms
      // out at the array's declare.  Detect that case and route
      // through ``buildIndexExpr`` so the bound is rendered as the
      // proper subscripted expression ``row_ptr[(i) - offset_row_ptr_d0]``.
      auto isArrayElementLoad = [](mlir::Value v) -> bool {
        for (int i = 0; i < 32 && v; ++i) {
          auto* d = v.getDefiningOp();
          if (!d) break;
          if (mlir::isa<hlfir::DesignateOp>(d)) return true;
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
            v = cv.getValue();
            continue;
          }
          if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
            v = ld.getMemref();
            continue;
          }
          break;
        }
        return false;
      };
      if (auto c = traceConstInt(doLoop.getUpperBound())) {
        n.loop_bound = std::to_string(*c);
      } else if (!isArrayElementLoad(doLoop.getUpperBound())) {
        n.loop_bound = traceToDecl(doLoop.getUpperBound());
      }
      if (n.loop_bound.empty())
        n.loop_bound = buildIndexExpr(doLoop.getUpperBound(), 0);
      // ``buildIndexExpr`` suppresses ``fir.box_dims`` on a local allocatable
      // (it keeps the whole-array-section copy fallback for ``arr(:)`` reads).
      // A loop bound is a scalar expression, not a subset, so ``do i = 1,
      // size(a)`` should resolve like the scalar assignment ``sz = size(a)``
      // does -- fall back to ``buildExpr``, which renders the extent as the
      // bound ``<buf>_d<i>`` symbol.
      if (n.loop_bound == "?")
        n.loop_bound = buildExpr(doLoop.getUpperBound(), 0);
      n.loop_lower = traceLB(doLoop.getLowerBound());
      if (n.loop_lower < 0) {
        // Non-constant lower bound (``DO jk = nflatlev, nlev`` /
        // ``DO j = row_ptr(i), ...``).  Capture the symbolic form
        // so emit_loop can thread it through instead of silently
        // defaulting to 1.
        if (!isArrayElementLoad(doLoop.getLowerBound())) {
          auto sym = traceToDecl(doLoop.getLowerBound());
          if (!sym.empty()) n.loop_lower_expr = sym;
        }
        if (n.loop_lower_expr.empty())
          n.loop_lower_expr = buildIndexExpr(doLoop.getLowerBound(), 0);
      }
      // Step.  Reverse-direction ``DO i = N, 1, -1`` (LU
      // back-substitution) carries step -1; the bridge needs
      // this to flip init/cond/update in emit_loop.  Constant
      // steps only  --  symbolic-step loops would silently default
      // to step=1 and produce a wrong-direction iteration if
      // the symbol is actually negative, so throw loudly when
      // the step is non-constant AND non-trivial (i.e. not the
      // default ``%c1``).
      if (auto stepC = traceConstInt(doLoop.getStep())) {
        n.loop_step = *stepC;
      } else {
        // Symbolic step (``DO jbnd = jstart, jend, many_fft`` where
        // ``many_fft`` is a runtime config integer).  Capture the
        // symbolic form so emit_loop threads it through as the
        // iteration update.  Defaults to forward iteration; a
        // runtime-negative symbol falls out as zero-or-one
        // iterations under the ``uid <= bound`` condition, matching
        // Fortran's trip-count semantics for mismatched-direction
        // loops.
        //
        // ``traceToDecl`` lifts the underlying scalar's name; if
        // that's empty (the step comes from an inline arithmetic
        // expression like ``2*chunk``) fall back to ``buildIndexExpr``
        // which renders the SSA tree as a Fortran-style expression
        // string.  Either way ``loop_step`` stays at the default 1
        // and the emitter consults ``loop_step_expr`` first.
        if (!isArrayElementLoad(doLoop.getStep())) {
          auto sym = traceToDecl(doLoop.getStep());
          if (!sym.empty()) n.loop_step_expr = sym;
        }
        if (n.loop_step_expr.empty())
          n.loop_step_expr = buildIndexExpr(doLoop.getStep(), 0);
        if (n.loop_step_expr.empty() || n.loop_step_expr == "?") {
          // Fallback failed -- emit a location-rich diagnostic so
          // the user can find the offending DO in their source.
          std::string locStr;
          llvm::raw_string_ostream locOS(locStr);
          doLoop.getLoc().print(locOS);
          throw std::runtime_error(
              "fir.do_loop with unrenderable symbolic step at " + locStr +
              "  --  step expression couldn't be lifted to a "
              "Fortran-style scalar.  Open an issue if you need this "
              "shape (typically a step computed from a function call "
              "or struct member).");
        }
      }
      // Elemental-inlined bodies use the fir.do_loop block arg
      // directly as the hlfir.designate index  --  no fir.store ->
      // alloca -> fir.load indirection.  traceLoopIter returns ""
      // for that shape; push the block arg onto indexStack() with
      // a synthetic name so resolveIndex() can recover it when the
      // inner designate's index is the raw block arg.
      static thread_local int kDoLoopIterCounter = 0;
      bool pushedBlockArg = false;
      auto& loopBlock = doLoop.getRegion().front();
      if (n.loop_iter.empty() && loopBlock.getNumArguments() > 0) {
        n.loop_iter = "_doit_" + std::to_string(kDoLoopIterCounter++);
        indexStack().push_back({loopBlock.getArgument(0), n.loop_iter});
        pushedBlockArg = true;
      }
      n.children = buildAST(loopBlock);
      if (pushedBlockArg) indexStack().pop_back();
      nodes.push_back(std::move(n));
      continue;
    }
    if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op)) {
      auto src = assign.getOperand(0);
      auto dst = assign.getOperand(1);

      // Fortran-runtime transformational intrinsics return their
      // result either as a scalar (``_FortranANorm2_*``) or as a
      // newly-heap-allocated array whose descriptor lives in an
      // alloca passed as the first operand
      // (``_FortranASpread`` / ``_FortranAEoshiftVector`` / etc.).
      //
      // Scalar form ``hlfir.assign %fXX_call_result to %dst``:
      // detect the call directly.
      // Array form ``hlfir.assign %as_expr to %dst`` where
      // ``%as_expr = hlfir.as_expr %decl move %true`` and ``%decl``
      // declares the runtime ``.tmp.intrinsic_result``: walk back
      // through ``declare -> box_addr -> load %alloca`` and look at
      // users of ``%alloca`` for the matching runtime call.
      auto canonicalCallee = [](fir::CallOp call) -> std::string {
        auto cref = call.getCallee();
        if (!cref) return "";
        std::string cs;
        llvm::raw_string_ostream os(cs);
        cref->print(os);
        std::string callee = cs;
        if (!callee.empty() && callee.front() == '@') callee.erase(0, 1);
        if (callee.size() >= 2 && callee.front() == '"' && callee.back() == '"')
          callee = callee.substr(1, callee.size() - 2);
        return callee;
      };

      if (auto* sop = src.getDefiningOp()) {
        if (auto call = mlir::dyn_cast<fir::CallOp>(sop)) {
          std::string callee = canonicalCallee(call);
          // ``_FortranANorm2_<bits>`` -> ``Norm2`` lib node.
          if (callee.rfind("_FortranANorm2_", 0) == 0) {
            ASTNode n;
            n.kind = "libcall";
            n.callee = "norm2";
            if (auto* dd = dst.getDefiningOp())
              if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
                n.target =
                    allocAliasFor(extractName(decl.getUniqName().str()));
            if (n.target.empty()) n.target = traceToDecl(dst);
            n.target_is_array = false;
            auto args = call.getArgOperands();
            if (args.size() >= 1) {
              auto srcName = traceToDecl(args[0]);
              if (srcName.empty())
                throw std::runtime_error(
                    "_FortranANorm2: cannot resolve source array name");
              n.call_args.push_back(srcName);
              n.call_arg_subsets.push_back("");
            }
            if (args.size() >= 4) {
              if (auto c = traceConstInt(args[3])) {
                if (*c > 0) n.reduce_axes.push_back(*c - 1);
              }
            }
            nodes.push_back(std::move(n));
            continue;
          }
        }
        // Heap-result form: trace through as_expr / declare to the
        // alloca whose store-from-runtime-call seeded it.
        if (auto as_expr = mlir::dyn_cast<hlfir::AsExprOp>(sop)) {
          mlir::Value v = as_expr.getVar();
          // declare -> box_addr -> load -> alloca chain.
          mlir::Operation* allocaOp = nullptr;
          for (int hop = 0; hop < 8 && v; ++hop) {
            auto* d = v.getDefiningOp();
            if (!d) break;
            if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
              v = decl.getMemref();
              continue;
            }
            if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
              v = ba.getVal();
              continue;
            }
            if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
              v = ld.getMemref();
              continue;
            }
            if (auto al = mlir::dyn_cast<fir::AllocaOp>(d)) {
              allocaOp = al.getOperation();
              break;
            }
            break;
          }
          if (allocaOp) {
            // Look for ``_FortranASpread`` / ``_FortranAEoshiftVector``
            // among users of the alloca, peeling through
            // ``fir.convert`` (the runtime call takes a
            // ``!fir.ref<!fir.box<none>>`` so the alloca's typed
            // result is first reboxed via convert).
            std::vector<mlir::Operation*> queue;
            for (auto* u : allocaOp->getResult(0).getUsers())
              queue.push_back(u);
            for (size_t qi = 0; qi < queue.size(); ++qi) {
              auto* user = queue[qi];
              if (auto cv = mlir::dyn_cast<fir::ConvertOp>(user)) {
                for (auto* uu : cv.getResult().getUsers()) queue.push_back(uu);
                continue;
              }
              auto rtcall = mlir::dyn_cast<fir::CallOp>(user);
              if (!rtcall) continue;
              std::string callee = canonicalCallee(rtcall);
              auto rtargs = rtcall.getArgOperands();
              // SPREAD / EOSHIFT: lower directly to the lib node
              // writing INTO the user's destination -- src -> lib node
              // -> dst, no intermediate transient.  Fortran's
              // assignment semantics guarantees the heap-result shape
              // matches dst's shape, so the runtime allocation is just
              // ceremony around what is semantically a copy from a
              // shape-transformed source.
              std::string dstName;
              if (auto* dd = dst.getDefiningOp())
                if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
                  dstName =
                      allocAliasFor(extractName(decl.getUniqName().str()));
              if (dstName.empty()) dstName = traceToDecl(dst);
              if (callee == "_FortranASpread" && rtargs.size() >= 4) {
                auto srcName = traceToDecl(rtargs[1]);
                if (srcName.empty())
                  throw std::runtime_error(
                      "_FortranASpread: cannot resolve source array name");
                ASTNode n;
                n.kind = "libcall";
                n.callee = "broadcast";
                n.target = dstName;
                n.target_is_array = true;
                n.call_args.push_back(srcName);
                n.call_arg_subsets.push_back("");
                if (auto c = traceConstInt(rtargs[2]))
                  n.reduce_axes.push_back(*c - 1);
                nodes.push_back(std::move(n));
                goto runtime_call_handled;
              }
              if ((callee == "_FortranAEoshiftVector" ||
                   callee.rfind("_FortranAEoshift", 0) == 0) &&
                  rtargs.size() >= 3) {
                auto srcName = traceToDecl(rtargs[1]);
                if (srcName.empty())
                  throw std::runtime_error(
                      "_FortranAEoshift: cannot resolve source array name");
                ASTNode n;
                n.kind = "libcall";
                n.callee = "eoshift";
                n.target = dstName;
                n.target_is_array = true;
                n.call_args.push_back(srcName);
                n.call_arg_subsets.push_back("");
                if (auto c = traceConstInt(rtargs[2])) {
                  n.options["shift"] = std::to_string(*c);
                } else {
                  auto sExpr = buildIndexExpr(rtargs[2], 0);
                  if (!sExpr.empty() && sExpr != "?")
                    n.options["shift"] = sExpr;
                }
                if (rtargs.size() >= 4) {
                  if (auto c = traceConstInt(rtargs[3]))
                    n.options["boundary"] = std::to_string(*c);
                }
                nodes.push_back(std::move(n));
                goto runtime_call_handled;
              }
            }
          }
        }
      }
      goto runtime_call_not_handled;
    runtime_call_handled:
      continue;
    runtime_call_not_handled:;

      // Recognise + suppress the ``plan = fftw_plan_dft_*(...)`` user
      // statement.  Flang lowers it through a ``.result`` temp:
      //   ``%158 = fir.call @fftw_plan_dft_2d(...)``
      //   ``fir.save_result %158 to %0  (the .result alloca)``
      //   ``%159 = hlfir.declare %0``
      //   ``%160 = hlfir.as_expr %159``
      //   ``hlfir.assign %160 to %141#0``  (the user's ``plan`` variable)
      // We walk back through the as_expr -> declare -> alloca chain and
      // ask whether the alloca is the destination of a fir.save_result
      // of a recognised FFTW3 plan-create call. If so, record the plan
      // variable's (rank, dims, direction) under ``fft_plans`` for the
      // matching ``fftw_execute_dft`` to look up, and skip the assign --
      // the opaque ``TYPE(C_PTR)`` SSA value has no SDFG representation.
      {
        bool is_fft_plan_assign = false;
        if (auto* sop = src.getDefiningOp()) {
          if (auto as_expr = mlir::dyn_cast<hlfir::AsExprOp>(sop)) {
            if (auto* dop = as_expr.getVar().getDefiningOp()) {
              if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dop)) {
                auto memref = decl.getMemref();
                for (auto u : memref.getUsers()) {
                  auto sr = mlir::dyn_cast<fir::SaveResultOp>(u);
                  if (!sr) continue;
                  auto* srdef = sr.getValue().getDefiningOp();
                  auto call = srdef ? mlir::dyn_cast<fir::CallOp>(srdef) : fir::CallOp{};
                  if (!call) continue;
                  auto ref = call.getCallee();
                  if (!ref) continue;
                  std::string cs;
                  llvm::raw_string_ostream os(cs);
                  ref->print(os);
                  std::string tag = fftw3CalleeTag(cs);
                  if (tag != "fft_plan_2d" && tag != "fft_plan_3d") continue;
                  // Confirmed: this assign is the user's ``plan = fftw_plan_dft_*``.
                  is_fft_plan_assign = true;
                  std::string planVar = traceToDecl(dst);
                  if (!planVar.empty()) {
                    FftPlanInfo info;
                    info.rank = (tag == "fft_plan_2d") ? 2 : 3;
                    auto args = call.getArgOperands();
                    for (int i = 0; i < info.rank; ++i) {
                      std::string dim;
                      if (auto c = traceConstInt(args[i])) dim = std::to_string(*c);
                      else dim = traceToDecl(args[i]);
                      info.dims.push_back(dim);
                    }
                    int sign = 0;
                    if (auto c = traceConstInt(args[info.rank + 2])) sign = (int)*c;
                    info.direction = (sign == -1) ? "forward" : "backward";
                    fft_plans[planVar] = info;
                  }
                  break;
                }
              }
            }
          }
        }
        if (is_fft_plan_assign) continue;
      }

      // Suppress per-element stores into a Flang-synthesised
      // ``.tmp.arrayctor`` heap buffer.  The final
      // ``hlfir.assign %as_expr_of_arrayctor to %dst`` site below
      // walks the parent block and emits per-element assigns
      // retargeted to ``%dst``; if we let the per-element stores
      // through here they'd surface as orphan assigns into
      // ``.tmp.arrayctor`` and break downstream memlet parsing.
      if (auto* dd = dst.getDefiningOp()) {
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd)) {
          if (auto* md = dg.getMemref().getDefiningOp()) {
            if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(md)) {
              if (decl.getUniqName().str().find(".tmp.") != std::string::npos) {
                continue;
              }
            }
          }
        }
      }
      bool dst_is_array = isArrayRef(dst.getType());
      bool src_is_array = isArrayRef(src.getType());

      // ``hlfir.expr``-valued sources (any HLFIR op whose result
      // type peels to ``!hlfir.expr<...>``: ``hlfir.elemental``,
      // ``hlfir.matmul``, ``hlfir.transpose``, ``hlfir.dot_product``,
      // ``hlfir.count``, ``hlfir.sum``, ...) are array-typed but
      // NOT array refs  --  they have no memory backing.
      // ``buildCopyNode`` would call ``traceToDecl`` on them and
      // get an empty name; route them to the elemental / libcall
      // handlers below instead.  This is what makes
      // ``res(:) = a(:) - b(:)`` (Flang-generated elemental) and
      // ``res = COUNT(mask, dim=1)`` (libcall returning expr)
      // work without falling through to a degenerate copy.
      bool src_is_hlfir_expr = false;
      if (auto srcOp = src.getDefiningOp()) {
        if (mlir::isa<hlfir::ExprType>(
                peelWrappers(srcOp->getResult(0).getType())))
          src_is_hlfir_expr = true;
      }

      // Section-to-section assign  --  both sides are non-trivial
      // ``hlfir.designate``s with at least one triplet dim.  Walk
      // the structure explicitly because ``buildCopyNode`` would
      // otherwise treat it as a whole-array copy and silently
      // ignore scalar dims and slice offsets (e.g.
      // ``res(nval, pos(1):pos(2)) = a(nval, pos(3):pos(4))``).
      if (dst_is_array && src_is_array && !src_is_hlfir_expr) {
        bool dstIsSection = (bool)asSectionDesignate(dst);
        bool srcIsSection = (bool)asSectionDesignate(src);
        // Either side carrying section info is enough  --  the
        // helper handles bare-decl on whichever side is plain
        // whole-array.  Without this the dst-bare-decl form
        // (``t0_w = p_prog_pprog_w(1, 1:5:1, 1:5:1)`` produced
        // by Phase 2 nested-DT flattening of an AoS-element
        // whole-struct copy) would fall through to
        // ``buildCopyNode`` and copy the entire 3D companion.
        if (dstIsSection || srcIsSection) {
          auto built = buildSectionToSectionAssign(assign, dst);
          if (!built.empty()) {
            for (auto& n : built) nodes.push_back(std::move(n));
            continue;
          }
        }
      }
      // Whole-array copy: both sides are array boxes / refs (and
      // not an ``hlfir.expr``-producer that we want to walk into).
      if (dst_is_array && src_is_array && !src_is_hlfir_expr) {
        nodes.push_back(buildCopyNode(assign));
        continue;
      }
      // Scalar-zero -> array fill: MemsetLibraryNode.
      if (dst_is_array && !src_is_array && isConstantZero(src)) {
        nodes.push_back(buildMemsetNode(assign));
        continue;
      }

      // Array-section ``res(a:b) = <scalar>``  --  detect the LHS
      // hlfir.designate with triplet operands and synthesise a
      // nested loop over the section bounds.  Handled before the
      // elemental dispatch below because Flang emits a plain
      // scalar RHS here (no hlfir.elemental wrapping).
      if (!src_is_array) {
        if (auto sec = asSectionDesignate(dst)) {
          auto built = buildSectionScalarAssign(assign, sec);
          if (!built.empty()) {
            for (auto& built_n : built) nodes.push_back(std::move(built_n));
            continue;
          }
        }
        // ``res = <non-zero scalar>``  --  broadcast across the whole
        // array.  Memset already handled zero above; synthesise a
        // nested loop here for any other constant / scalar RHS.
        if (dst_is_array) {
          auto built = buildWholeArrayScalarBroadcast(assign);
          if (!built.empty()) {
            for (auto& built_n : built) nodes.push_back(std::move(built_n));
            continue;
          }
        }
      }

      // b = <elementwise-expression>   --  Flang wraps the RHS in one or
      // more composed hlfir.elemental ops, the outermost of which is
      // the assign's source.  Synthesise a nested loop over the shape
      // instead of treating it as a scalar assign.
      //
      // Special-case: ``b = MERGE(t, f, mask)`` on arrays lowers to
      // ``hlfir.elemental { hlfir.designate; arith.select;
      // yield_element }``.  Detect that exact shape and route to
      // ``MergeLibraryNode`` directly so the per-element select
      // stays inside the library node's expansion (modular  --
      // bridge doesn't inline).  Anything more elaborate falls
      // through to ``buildElementalAssign``'s per-element tasklet
      // path (which uses the generic select/cmp fallback).
      //
      // Implicit Fortran kind/type conversion: ``logical :: res``
      // assigned from an integer-valued ``COUNT`` (or any libcall
      // returning a different kind from the destination) puts a
      // ``fir.convert`` between the libcall and the assign.  Peel
      // it here so the dispatch below pattern-matches the real
      // producer, not the convert.
      mlir::Value srcPeeled = src;
      while (auto* pdef = srcPeeled.getDefiningOp()) {
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(pdef)) {
          srcPeeled = cv.getValue();
          continue;
        }
        break;
      }
      // Fortran ``out = SHAPE(arr)`` (and similar array-constructor
      // shapes Flang lowers via a heap-allocated ``.tmp.arrayctor``):
      //
      //     %tmp   = fir.allocmem !fir.array<NxiK> {bindc_name =
      //     ".tmp.arrayctor"} %tmpD  = hlfir.declare %tmp(...) hlfir.assign
      //     <extent_0> to %tmpD[c1] hlfir.assign <extent_1> to %tmpD[c2]
      //     ...
      //     %expr = hlfir.as_expr %tmpD move %true
      //     hlfir.assign %expr to %dst
      //
      // The bridge can't model the ``.tmp.arrayctor`` buffer
      // (heap alloc + per-element store + as_expr), but it
      // doesn't need to: each per-element value is whatever the
      // intrinsic resolved to (e.g. SHAPE returns the source
      // array's per-dim extents, which the bridge already tracks
      // as VarInfo shape symbols).  Walk the parent block, find
      // each ``hlfir.assign <val> to %tmpD[<const>]``, and emit
      // one scalar assign per element directly into ``%dst``.
      if (auto asExpr = mlir::dyn_cast_or_null<hlfir::AsExprOp>(
              srcPeeled.getDefiningOp())) {
        hlfir::DeclareOp tmpDecl;
        if (auto* vd = asExpr.getVar().getDefiningOp())
          tmpDecl = mlir::dyn_cast<hlfir::DeclareOp>(vd);
        bool is_arrayctor = false;
        if (tmpDecl) {
          auto un = tmpDecl.getUniqName().str();
          is_arrayctor = (un.find(".tmp.arrayctor") != std::string::npos);
        }
        if (is_arrayctor) {
          // Resolve the destination's Fortran name once.
          std::string dst_name;
          if (auto* dd = dst.getDefiningOp())
            if (auto declOp = mlir::dyn_cast<hlfir::DeclareOp>(dd))
              dst_name = extractName(declOp.getUniqName().str());
          if (dst_name.empty()) dst_name = traceToDecl(dst);
          if (!dst_name.empty()) {
            std::vector<ASTNode> elem_assigns;
            bool every_idx_const = true;
            for (auto& op2 : *assign->getBlock()) {
              auto inner = mlir::dyn_cast<hlfir::AssignOp>(&op2);
              if (!inner) continue;
              if (inner == assign) break;  // stop at the final assign
              auto inner_dst = inner.getOperand(1);
              auto* iddef = inner_dst.getDefiningOp();
              if (!iddef) continue;
              auto inner_dg = mlir::dyn_cast<hlfir::DesignateOp>(iddef);
              if (!inner_dg) continue;
              // Designate's memref must trace back to the temp arrayctor.
              if (traceToDecl(inner_dg.getMemref()) !=
                  extractName(tmpDecl.getUniqName().str()))
                continue;
              auto idxOps = inner_dg.getIndices();
              if (idxOps.size() != 1) {
                every_idx_const = false;
                continue;
              }
              auto cidx = traceConstInt(idxOps[0]);
              if (!cidx) {
                every_idx_const = false;
                continue;
              }
              // Build the per-element assign: dst(<cidx>) = buildExpr(<val>).
              std::string val_expr = buildExpr(inner.getOperand(0), 0);
              if (val_expr.empty() || val_expr == "?") {
                every_idx_const = false;
                continue;
              }
              ASTNode a;
              a.kind = "assign";
              a.target = dst_name;
              a.target_is_array = true;
              a.expr = val_expr;
              AccessInfo wa;
              wa.array_name = dst_name;
              wa.is_write = true;
              wa.index_exprs.push_back(std::to_string(*cidx));
              wa.index_vars.push_back("?");
              a.accesses.push_back(std::move(wa));
              elem_assigns.push_back(std::move(a));
            }
            if (every_idx_const && !elem_assigns.empty()) {
              for (auto& n : elem_assigns) nodes.push_back(std::move(n));
              continue;
            }
          }
        }
      }

      if (auto* sd = srcPeeled.getDefiningOp()) {
        // ``hlfir.reshape %src %shape`` -> flat copy from the
        // reshape's source array into the assignment's destination.
        // Fortran ``RESHAPE`` requires both sides to carry the same
        // total element count, so a ``CopyLibraryNode`` whose
        // expansion collapses each side to a 1-D walker handles the
        // rank/extent mismatch.  ``hlfir.reshape``'s optional
        // ``pad`` / ``order`` operands are NOT supported in this
        // first cut  --  fall through to the existing
        // libcall/elemental path with a clear NotImplemented at the
        // bridge layer when present, so the surfaced gap is the
        // unsupported variant rather than a silent miscopy.
        if (auto reshape = mlir::dyn_cast<hlfir::ReshapeOp>(sd)) {
          if (sd->getNumOperands() > 2) {
            throw std::runtime_error(
                "hlfir.reshape: pad= / order= variants are not yet supported");
          }
          ASTNode n;
          n.kind = "copy";
          if (auto* dd = dst.getDefiningOp())
            if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(dd))
              n.target = allocAliasFor(extractName(decl.getUniqName().str()));
          if (n.target.empty()) n.target = traceToDecl(dst);
          n.target_is_array = true;
          n.reduce_src = traceToDecl(reshape.getOperand(0));
          if (n.reduce_src.empty())
            throw std::runtime_error(
                "hlfir.reshape: cannot resolve source array name");
          nodes.push_back(std::move(n));
          continue;
        }
        if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(sd)) {
          auto merge_built = buildMergeLibcall(assign, elem);
          if (!merge_built.empty()) {
            for (auto& n : merge_built) nodes.push_back(std::move(n));
            continue;
          }
          for (auto& n : buildElementalAssign(assign, elem))
            nodes.push_back(std::move(n));
          continue;
        }
        // Linear-algebra ops are first-class in HLFIR; each lowers
        // to a dedicated DaCe library node.  MatMul's SpecializeMatMul
        // handles matrix-matrix / matrix-vector / vector-matrix via
        // operand rank, so we don't disambiguate here.
        auto srcOpName = sd->getName().getStringRef();
        struct LibEntry {
          llvm::StringRef op;
          llvm::StringRef callee;
        };
        static const LibEntry kLibTable[] = {
            {"hlfir.matmul", "matmul"},
            // ``MATMUL(TRANSPOSE(A), B)`` lowers under the optimised
            // ``hlfir-optimized-bufferization`` pass as a single
            // ``hlfir.matmul_transpose``.  The Python emitter expands
            // it to a ``Transpose`` + ``MatMul`` libcall pair so the
            // operand-order semantics are correct without a
            // dedicated lib node; future cuBLAS/MKL acceleration
            // can swap in a fused expansion.
            {"hlfir.matmul_transpose", "matmul_transpose"},
            {"hlfir.transpose", "transpose"},
            {"hlfir.dot_product", "dot_product"},
            // Fortran ``COUNT(mask [, dim])``  --  routed through
            // ``CountLibraryNode`` so its ``cast -> Reduce``
            // expansion handles the integer-cast and the
            // per-target reduction lowering.  ``buildLibCallNode``
            // picks up the optional ``dim`` operand and threads
            // it through the ASTNode for ``emit_libcall``.
            {"hlfir.count", "count"},
            // Fortran ``MINLOC(array [, dim [, mask [, back]]])``
            // and the symmetric ``MAXLOC``  --  routed through the
            // ``ArgMin`` / ``ArgMax`` library nodes (pure WCR
            // expansion mirroring the ``numpy.argmin`` / ``numpy.argmax``
            // replacement pattern).  ``buildLibCallNode`` threads any
            // ``dim`` (Fortran 1-based) and ``back`` flag through the
            // ASTNode for ``emit_libcall``.
            {"hlfir.minloc", "argmin"},
            {"hlfir.maxloc", "argmax"},
            // Fortran ``CSHIFT(array, shift [, dim])`` -- circular
            // shift along ``dim`` (default 1).  Routed through
            // ``CShift`` (pure expansion = Map of mod-indexed reads
            // along the chosen axis).
            {"hlfir.cshift", "cshift"},
        };
        bool libMatched = false;
        for (auto& e : kLibTable) {
          if (srcOpName == e.op) {
            // Mode C: ``hlfir.count`` whose first operand is
            // an ``hlfir.elemental`` (comparison-as-mask /
            // compound boolean expression).  Synthesise a
            // transient int32 mask via a per-element loop,
            // then route through ``CountLibraryNode``.
            if (e.op == "hlfir.count" && sd->getNumOperands() > 0) {
              auto mask_src = sd->getOperand(0);
              if (auto* ms = mask_src.getDefiningOp()) {
                if (auto elem_src = mlir::dyn_cast<hlfir::ElementalOp>(ms)) {
                  auto built = buildElementalCountLibcall(assign, elem_src);
                  if (!built.empty()) {
                    for (auto& n : built) nodes.push_back(std::move(n));
                    libMatched = true;
                    break;
                  }
                }
              }
            }
            // Libcall-over-elemental fix-up.  When a
            // libcall operand is an inline ``hlfir.elemental``
            // (e.g. ``transpose(1.0 - d)``, or
            // ``transpose(d(firstcols(secondcols), :))``
            // where the gather chain produces an
            // ``hlfir.expr`` rather than a named array),
            // the bridge's source-name resolver returns
            // empty and the Python emitter fails at
            // ``ctx.sdfg.arrays['']``.  Pre-materialise
            // each elemental operand into a synthetic
            // ``_libsrc_<n>`` transient via a fill loop,
            // then patch the libcall's ``call_args`` to
            // point at the transient.  Excluded: the
            // ``hlfir.count`` path, handled above with
            // its own COUNT-specific transient.
            std::vector<std::string> elemSubst(sd->getNumOperands());
            bool needSubst = false;
            for (unsigned i = 0; i < sd->getNumOperands(); ++i) {
              auto opnd = sd->getOperand(i);
              auto* od = opnd.getDefiningOp();
              if (!od) continue;
              if (auto elem = mlir::dyn_cast<hlfir::ElementalOp>(od)) {
                auto [trName, prelude] = materialiseElementalForLibcall(elem);
                if (trName.empty()) continue;
                for (auto& n : prelude) nodes.push_back(std::move(n));
                elemSubst[i] = std::move(trName);
                needSubst = true;
                continue;
              }
              // Inline ``hlfir.transpose %A`` operand -- materialise a
              // ``transpose`` libcall into a fresh ``_libsrc_<n>``
              // transient, then point the outer libcall at the
              // transient.  Covers ``MATMUL(A, TRANSPOSE(B))`` and
              // ``MATMUL(TRANSPOSE(A), TRANSPOSE(B))`` in the default
              // (unfused) HLFIR pipeline.
              //
              // SKIP this materialisation when the outer libcall is
              // ``matmul`` / ``matmul_transpose`` -- the BLAS call's
              // ``transA`` / ``transB`` flag handles the transpose
              // in-place (``CblasTrans`` / ``CUBLAS_OP_T``), no
              // transient + no extra copy.  ``buildLibCallNode``
              // (assigns.cpp) detects the same shape and sets
              // ``options[transA/transB]=true`` so the two paths
              // line up.  For ``matmul``: both arg 0 and arg 1
              // qualify.  For ``matmul_transpose``: only arg 1
              // qualifies (LHS transpose is already in the op
              // itself).
              bool isMatmulFamily = (e.callee == "matmul" ||
                                     e.callee == "matmul_transpose");
              bool foldsViaBlas =
                  (e.callee == "matmul" && i < 2) ||
                  (e.callee == "matmul_transpose" && i == 1);
              if (auto tp = mlir::dyn_cast<hlfir::TransposeOp>(od);
                  tp && isMatmulFamily && foldsViaBlas) {
                // Defer to buildLibCallNode -- it will see the
                // hlfir.transpose operand, set the BLAS flag, and
                // re-bind to the un-transposed source.
                continue;
              }
              if (auto tp = mlir::dyn_cast<hlfir::TransposeOp>(od)) {
                auto srcVal = tp.getOperand();
                auto srcName = traceToDecl(srcVal);
                if (srcName.empty()) continue;
                std::string trName =
                    "_libsrc_t_" + std::to_string(kSynthTransientCounter++);
                ASTNode decl;
                decl.kind = "declare_transient";
                decl.target = trName;
                decl.expr = exprDtypeString(tp.getType());
                AccessInfo shape_info;
                shape_info.array_name = trName;
                // ``hlfir.transpose`` result is rank-2 with the
                // source's dims reversed.  Derive each result-dim
                // extent from the source array's ``box_dims`` so the
                // transient gets a symbolic shape rather than the
                // ``?`` placeholder Flang puts in the expression-type
                // shape vector for assumed-shape sources.
                shape_info.index_exprs.push_back(srcName + "_d1");
                shape_info.index_exprs.push_back(srcName + "_d0");
                decl.accesses.push_back(std::move(shape_info));
                nodes.push_back(std::move(decl));
                ASTNode tcall;
                tcall.kind = "libcall";
                tcall.callee = "transpose";
                tcall.target = trName;
                tcall.target_is_array = true;
                tcall.call_args.push_back(srcName);
                tcall.call_arg_subsets.push_back("");
                nodes.push_back(std::move(tcall));
                elemSubst[i] = std::move(trName);
                needSubst = true;
                continue;
              }
            }
            auto lib = buildLibCallNode(assign, sd, e.callee.str());
            if (needSubst) {
              for (unsigned i = 0;
                   i < elemSubst.size() && i < lib.call_args.size(); ++i)
                if (!elemSubst[i].empty()) lib.call_args[i] = elemSubst[i];
            }
            nodes.push_back(std::move(lib));
            libMatched = true;
            break;
          }
        }
        if (libMatched) continue;

        // Scalar reductions land as their own dedicated op; pattern-
        // match each one and hand the shared reduce-lowering helper
        // the right wcr + identity.
        auto opName = sd->getName().getStringRef();
        struct RedEntry {
          llvm::StringRef op;
          llvm::StringRef wcr;       // DaCe wcr lambda string
          llvm::StringRef identity;  // initial accumulator value (float-typed)
          llvm::StringRef py_op;     // Python binary op for
                                     // section-reduce loop body;
                                     // empty -> fall back to
                                     // buildReduceNode (whole-array)
        };
        // Identity strings here are the FLOAT-TYPED defaults.  The
        // ``identityForType`` helper below specialises them per
        // element type because ``inf`` / ``-inf`` cast to an integer
        // is undefined behaviour: at -O3 the compiler folds it to
        // INT_MAX/INT_MIN; at -O0 the un-folded ``INFINITY`` to
        // ``int`` conversion gives INT_MIN regardless of intent,
        // breaking integer MINVAL (e.g. NPB LU's class-S tests
        // surface this as ``MINVAL(arr) == -2147483648`` at -O0).
        // Identity strings use the bare ``inf`` token (not
        // ``math.inf``) so DaCe's cppunparse -- which maps
        // ``inf`` -> ``INFINITY`` via _py2c_reserved -- emits
        // a valid C++ literal in the section-reduce init
        // tasklet.  The whole-array Reduce path's eval()
        // namespace is patched with ``inf=math.inf`` for
        // the same string.
        static const RedEntry kRedTable[] = {
            {"hlfir.sum", "lambda a, b: a + b", "0", "+"},
            {"hlfir.product", "lambda a, b: a * b", "1", "*"},
            {"hlfir.minval", "lambda a, b: min(a, b)", "inf", "min"},
            {"hlfir.maxval", "lambda a, b: max(a, b)", "-inf", "max"},
            // Logical reductions  --  ANY / ALL on ``fir.logical``
            // arrays (ICON's levelmask / maskflag patterns).
            {"hlfir.any", "lambda a, b: a or b", "False", "or"},
            {"hlfir.all", "lambda a, b: a and b", "True", "and"},
            // ``hlfir.count`` is intentionally absent  --  handled
            // in ``kLibTable`` above as a ``CountLibraryNode``
            // libcall (covers Fortran COUNT's int-cast semantics
            // and the optional ``dim`` argument).
        };
        // Pick the type-correct identity literal for the reduction
        // op's element type.  MINVAL / MAXVAL on integer arrays must
        // initialise with INT_MAX / INT_MIN; ``inf`` / ``-inf`` to
        // int is undefined behaviour and breaks at -O0 (see
        // ``tests/integer_reduction_identity_test.py``).
        auto identityForType = [&](const RedEntry& entry,
                                   mlir::Type elemTy) -> std::string {
          // SUM / PRODUCT identities (``0`` / ``1``) are type-
          // compatible for both int and float -- pass through.
          if (entry.identity == "0" || entry.identity == "1" ||
              entry.identity == "True" || entry.identity == "False") {
            return entry.identity.str();
          }
          // MINVAL / MAXVAL: pick the type-correct sentinel.
          bool isInf = (entry.identity == "inf");
          bool isNegInf = (entry.identity == "-inf");
          if (!isInf && !isNegInf) return entry.identity.str();
          if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
            // Emit the literal min/max value for the integer width.
            // We use literal integers (not ``std::numeric_limits<>``)
            // because the builder's ``_parse_reduce_identity``
            // (emit_library.py:138) round-trips the identity through
            // Python ``int(s)``: a string of the right literal goes
            // straight through, while ``std::numeric_limits<>::max()``
            // forces a separate Python-side parser.  Fortran's INTEGER
            // KINDs map to MLIR signed integer widths 8/16/32/64;
            // signed semantics, two's complement min/max.
            unsigned w = intTy.getWidth();
            // Two's-complement max = 2^(w-1) - 1, min = -2^(w-1).
            // ``__int128`` would overflow llvm::APInt construction
            // from int64; we cap at w == 64 and fall back to the
            // float identity for wider types (none arise from
            // Fortran).
            if (w > 64) return entry.identity.str();
            int64_t maxVal = (w == 64) ? std::numeric_limits<int64_t>::max()
                                       : ((int64_t(1) << (w - 1)) - 1);
            int64_t minVal = (w == 64) ? std::numeric_limits<int64_t>::min()
                                       : (-(int64_t(1) << (w - 1)));
            return std::to_string(isInf ? maxVal : minVal);
          }
          // Default: float identity (``inf`` / ``-inf``) -- DaCe's
          // cppunparse maps to ``INFINITY`` / ``-INFINITY``.
          return entry.identity.str();
        };
        // Compute the reduction's element type once; ``sd`` is the
        // hlfir.minval / maxval / sum / ... op.  Its result type is
        // the per-element scalar type for these "reduce to scalar"
        // ops.
        mlir::Type redElemTy;
        if (sd->getNumResults() > 0) redElemTy = sd->getResult(0).getType();
        bool matched = false;
        for (auto& e : kRedTable) {
          if (opName == e.op) {
            // If the reduction source is a section designate
            // (``mask(lo:hi, jk)``) we can't use DaCe's Reduce
            // node directly  --  it reduces whole arrays.  Fall
            // back to a loop-accumulator lowering when a
            // Python op is available.
            bool emitted = false;
            if (!e.py_op.empty() && sd->getNumOperands() > 0) {
              auto srcVal = sd->getOperand(0);
              // Peel ``fir.convert`` chains so a section
              // designate hidden behind a box rebox (the
              // shape canonicalisation that shows up after
              // ``hlfir-rewrite-sequence-association``  --
              // ``box<array<NxT>>`` -> ``box<array<?xT>>``)
              // still matches the section-reduce path.
              // Safe because at this point ``srcVal`` is
              // a box/ref of an array element type  --  the
              // converts here are shape-bookkeeping only,
              // never value-altering casts (which only
              // appear at scalar value sites).
              while (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(
                         srcVal.getDefiningOp())) {
                srcVal = cv.getValue();
              }
              if (auto* srcOp = srcVal.getDefiningOp()) {
                if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(srcOp)) {
                  bool hasTrip = false;
                  for (bool t : dg.getIsTriplet())
                    if (t) {
                      hasTrip = true;
                      break;
                    }
                  if (hasTrip) {
                    auto built = buildSectionReduceAssign(
                        assign, dg, e.py_op.str(),
                        identityForType(e, redElemTy));
                    if (!built.empty()) {
                      for (auto& bn : built) nodes.push_back(std::move(bn));
                      emitted = true;
                    }
                  }
                }
              }
            }
            // Mode-C for ANY reduction op whose source is an
            // ``hlfir.elemental`` (compound boolean expression for
            // ANY/ALL, element-wise arithmetic like ``SUM(q ** 2)``
            // for SUM/PRODUCT/MINVAL/MAXVAL).  ``traceToDecl``
            // returns "" for an elemental result -- the plain
            // Reduce path then explodes with ``reduction source ''
            // not registered``.  Materialise the elemental into a
            // transient via a per-element loop and route the
            // Reduce over the transient.  ``buildElementalAnyAllReduce``
            // is op-agnostic (its name is historical -- the wcr +
            // identity arguments make it work for any reduction)
            // so we route SUM/PRODUCT/MINVAL/MAXVAL the same way.
            // QE's ``vcut_get`` (3 occurrences of ``SUM(... ** 2)``)
            // was the surfacing case.
            if (!emitted && sd->getNumOperands() > 0) {
              auto srcVal = sd->getOperand(0);
              if (auto* srcOp = srcVal.getDefiningOp())
                if (auto elem_src = mlir::dyn_cast<hlfir::ElementalOp>(srcOp)) {
                  auto built = buildElementalAnyAllReduce(
                      assign, elem_src, e.wcr.str(),
                      identityForType(e, redElemTy));
                  if (!built.empty()) {
                    for (auto& bn : built) nodes.push_back(std::move(bn));
                    emitted = true;
                  }
                }
            }
            if (!emitted) {
              nodes.push_back(buildReduceNode(assign, sd, e.wcr.str(),
                                              identityForType(e, redElemTy)));
            }
            matched = true;
            break;
          }
        }
        if (matched) continue;
      }
      nodes.push_back(buildAssignNode(assign));
      continue;
    }
    if (auto ifOp = mlir::dyn_cast<fir::IfOp>(op)) {
      // Allocatable deallocate-guard: ``fir.if (alloc_status != 0) {
      // fir.freemem, reset box to zero }``.  Carries no observable
      // side effect in the SDFG model (we treat allocatables as
      // single-allocation transients)  --  skip the whole construct.
      auto isAllocCleanup = [](mlir::Region& region) {
        if (region.empty()) return false;
        bool hasFreemem = false;
        for (auto& op : region.front()) {
          auto nm = op.getName().getStringRef();
          if (nm == "fir.freemem") {
            hasFreemem = true;
            continue;
          }
          if (nm == "fir.box_addr" || nm == "fir.zero_bits" ||
              nm == "fir.embox" || nm == "fir.shape" || nm == "fir.store" ||
              nm == "fir.load" || nm == "fir.if" || nm == "fir.result" ||
              nm == "arith.constant")
            continue;
          return false;
        }
        return hasFreemem;
      };
      if (isAllocCleanup(ifOp.getThenRegion()) &&
          (ifOp.getElseRegion().empty() ||
           isAllocCleanup(ifOp.getElseRegion()))) {
        continue;
      }
      ASTNode n;
      n.kind = "conditional";
      n.condition = buildBoolExpr(ifOp.getCondition(), 0);
      // Walk the condition's IR for array-element reads -- same as
      // the ``fir.if`` handler at the top of ``walkBlock`` and the
      // ``scf.if`` handler below.  Without this the Python emitter
      // has no per-occurrence access info to lift array-read
      // conditions through the tasklet path.
      collectReadAccesses(ifOp.getCondition(), n.accesses, 0);
      if (!ifOp.getThenRegion().empty())
        n.children = buildAST(ifOp.getThenRegion().front());
      if (!ifOp.getElseRegion().empty())
        n.else_children = buildAST(ifOp.getElseRegion().front());
      nodes.push_back(std::move(n));
      continue;
    }
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      ASTNode n;
      n.kind = "conditional";
      n.condition = buildBoolExpr(ifOp.getCondition(), 0);
      // Same array-read collection as the ``fir.if`` branch above.
      collectReadAccesses(ifOp.getCondition(), n.accesses, 0);
      if (!ifOp.getThenRegion().empty())
        n.children = buildAST(ifOp.getThenRegion().front());
      if (!ifOp.getElseRegion().empty())
        n.else_children = buildAST(ifOp.getElseRegion().front());
      nodes.push_back(std::move(n));
      continue;
    }
    if (auto call = mlir::dyn_cast<fir::CallOp>(op)) {
      ASTNode n;
      n.kind = "call";
      if (auto ref = call.getCallee()) {
        std::string s;
        llvm::raw_string_ostream os(s);
        ref->print(os);
        n.callee = s;
      }
      std::string mpiOp = mpiCalleeTag(n.callee);
      if (!mpiOp.empty()) {
        nodes.push_back(buildMpiCallNode(call, mpiOp));
        continue;
      }
      // FFTW3 plan-create / execute / destroy triple: consume all three;
      // only the execute emits an ``fftcall`` ASTNode (the plan-create
      // and destroy are absorbed -- the FFT lib node's expansion owns
      // the plan lifecycle).
      std::string fftOp = fftw3CalleeTag(n.callee);
      if (!fftOp.empty()) {
        auto fn = buildFftw3CallNode(call, fftOp, fft_plans);
        if (!fn.kind.empty()) nodes.push_back(std::move(fn));
        continue;
      }
      // QE generic FFT (fwfft / invfft) and its specific subroutines
      // (fwfft_y / invfft_y / fwfft_b / invfft_b).  Map to the same
      // ``fftcall`` ASTNode the FFTW3 execute emits so ``emit_fft``
      // handles both uniformly.
      std::string qeDir = qeFftCalleeTag(n.callee);
      if (!qeDir.empty()) {
        auto qn = buildQeFftCallNode(call, qeDir);
        if (!qn.kind.empty()) nodes.push_back(std::move(qn));
        continue;
      }
      // QE Fourier interpolation (fft_interpolate_real / _complex).
      // ABI: fft_interpolate(dfft_in, v_in, dfft_out, v_out) -- we use
      // operand 1 as the input array and operand 3 as the output array.
      std::string qeIntr = qeFftInterpolateCalleeTag(n.callee);
      if (!qeIntr.empty()) {
        auto args = call.getArgOperands();
        if (args.size() >= 4) {
          std::string vin = traceToDecl(args[1]);
          std::string vout = traceToDecl(args[3]);
          if (!vin.empty() && !vout.empty()) {
            ASTNode in;
            in.kind = "fft_interpolate";
            in.callee = qeIntr;  // "real" or "complex"
            in.target = vout;
            in.call_args = {vin, vout};
            nodes.push_back(std::move(in));
            continue;
          }
        }
      }
      // QE per-axis pencil FFT (cft_1z / cft_1y / cft_1x).
      std::string qePencil = qePencilCalleeTag(n.callee);
      if (!qePencil.empty()) {
        auto pn = buildQePencilCallNode(call, qePencil);
        if (!pn.kind.empty()) nodes.push_back(std::move(pn));
        continue;
      }
      // QE pencil-pipeline scatter routines (fft_scatter_xy / fft_scatter_yz).
      std::string qeScatter = qeScatterCalleeTag(n.callee);
      if (!qeScatter.empty()) {
        auto sn = buildQeScatterCallNode(call, qeScatter);
        if (!sn.kind.empty()) nodes.push_back(std::move(sn));
        continue;
      }
      // BLAS routine call site (DAXPY / DSCAL / DGEMM / ...): emit a
      // ``blascall`` ASTNode the Python builder lowers to the matching
      // ``dace.libraries.blas`` library node.  ``ddot`` is special-cased
      // -- the result-carrying assign handler picks it up at the
      // matching ``hlfir.assign`` site instead.
      std::string blasOp = blasCalleeTag(n.callee);
      if (!blasOp.empty()) {
        auto bn = buildBlasCallNode(call, blasOp);
        if (!bn.kind.empty()) nodes.push_back(std::move(bn));
        continue;
      }
      // LAPACK routine call site (DGETRF / DPOTRF / ...).
      std::string lapOp = lapackCalleeTag(n.callee);
      if (!lapOp.empty()) {
        auto ln = buildLapackCallNode(call, lapOp);
        if (!ln.kind.empty()) nodes.push_back(std::move(ln));
        continue;
      }
      // Library-prefix near-miss: the callee matches a recognised library's
      // call convention (MPI / FFTW3 / BLAS / LAPACK) but the specific
      // routine isn't in our supported subset.  Emit an explicit
      // ``unsupported_libcall`` ASTNode so the Python builder can raise a
      // clear ``NotImplementedError`` (better than silently degrading to a
      // generic ``call`` node that mints ``_out = ?`` placeholders).
      std::string libFam = libraryFamilyTag(n.callee);
      if (!libFam.empty()) {
        // Recognised + supported routines are caught above; reaching here
        // means the callee is in the library-prefix universe but not in
        // the supported set.
        bool isSupported =
            !mpiCalleeTag(n.callee).empty() ||
            !fftw3CalleeTag(n.callee).empty() ||
            !blasCalleeTag(n.callee).empty() ||
            !lapackCalleeTag(n.callee).empty();
        if (!isSupported) {
          ASTNode un;
          un.kind = "unsupported_libcall";
          un.callee = normaliseBlasName(n.callee);
          un.expr = libFam;  // "mpi" / "fftw3" / "blas" / "lapack"
          nodes.push_back(std::move(un));
          continue;
        }
      }
      // Fortran I/O runtime call: advance the open/read/write/close state
      // machine (see ``recognizeIoCall``).  Consumed here either way.
      if (n.callee.find("_FortranAio") != std::string::npos) {
        recognizeIoCall(call, n.callee, io_state, nodes);
        continue;
      }
      // Resolve each operand to a decl name so the Python builder can
      // lower a registered external (bind(c)) call to a tasklet.
      // Harmless for unregistered callees (the builder ignores them).
      // A by-value integer-constant operand (e.g. ``CALL ext(a, 16)``) has no
      // decl  --  emit its literal value so it reaches the C call by name
      // rather than as an empty term.
      for (auto v : call.getArgOperands()) {
        std::string nm = traceToDecl(v);
        if (nm.empty())
          if (auto c = traceConstInt(v)) nm = std::to_string(*c);
        n.call_args.push_back(nm);
      }
      // Carry the AoS-marshalling grouping the marshal pass tagged on the
      // callee so emit_call can re-pack the (now-SoA-flat) member args into a
      // local AoS buffer for the external.
      if (auto ref = call.getCallee())
        if (auto mod = call->getParentOfType<mlir::ModuleOp>())
          if (auto fn = mod.lookupSymbol<mlir::func::FuncOp>(
                  ref->getLeafReference()))
            if (auto g = fn->getAttrOfType<mlir::DenseI64ArrayAttr>(
                    "hlfir.aos_marshal_groups"))
              n.aos_marshal_groups.assign(g.asArrayRef().begin(),
                                          g.asArrayRef().end());
      nodes.push_back(std::move(n));
      continue;
    }
    if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op)) {
      nodes.push_back(buildWhileNode(whileOp));
      continue;
    }
    if (auto sw = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op)) {
      auto chain = buildIndexSwitchNodes(sw);
      for (auto& c : chain) nodes.push_back(std::move(c));
      continue;
    }
    if (auto sel = mlir::dyn_cast<fir::SelectCaseOp>(op)) {
      nodes.push_back(buildSelectCaseChain(sel));
      continue;
    }
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
      // Top-level ``fir.store`` is Flang's lowering for lifted
      // DO / DO-WHILE init (``fir.store %c1 to %i``) and internal
      // scratch counters.  Emit as a plain scalar assign.  Regular
      // ``fir.do_loop``s' internal IV stores never reach here  --
      // they live inside the loop's body region, which we walk
      // with the existing do-loop handler that takes care of the
      // IV through ``init_expr`` / ``update_expr``.
      auto memref = st.getMemref();
      auto target = traceToDecl(memref);
      if (target.empty())
        if (auto* md = memref.getDefiningOp())
          if (mlir::isa<fir::AllocaOp>(md)) target = allocaSynthName(memref);
      if (target.empty()) continue;
      // Drop stores whose RHS is the return value of a recognised FFTW3
      // ``fftw_plan_dft_*`` call -- the plan SSA value is opaque
      // (``TYPE(C_PTR)``) and is consumed by the matching ``execute_dft``
      // recognition, so the user's ``plan = fftw_plan_dft_2d(...)``
      // statement has no observable side effect in the SDFG.
      if (auto* def = st.getValue().getDefiningOp())
        if (auto src_call = mlir::dyn_cast<fir::CallOp>(def))
          if (auto ref = src_call.getCallee()) {
            std::string s;
            llvm::raw_string_ostream os(s);
            ref->print(os);
            if (!fftw3CalleeTag(s).empty()) continue;
          }
      auto expr = buildExpr(st.getValue(), 0);
      // Drop stores with unresolvable RHS  --  see note in
      // ``walkSCFBeforeRegion``'s fir.store handler.
      if (expr == "?") continue;
      ASTNode a;
      a.kind = "assign";
      a.target = target;
      a.expr = expr;
      a.target_is_array = false;
      nodes.push_back(std::move(a));
      continue;
    }
  }
  return nodes;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

std::vector<ASTNode> extractAST(mlir::ModuleOp module,
                                const std::string &entry_symbol) {
  // Fresh synthetic-name counters / maps per module so two consecutive
  // extractAST calls don't interleave __sc_5 / __al_2 across unrelated
  // SDFGs.
  kScfValueCounter = 0;
  kScfValueMap.clear();
  kAllocaCounter = 0;
  kAllocaMap.clear();
  kHlfirExprToTransient.clear();
  kLibTmpCounter = 0;
  kPosSymbolRegistry.clear();
  kSynthTransientCounter = 0;
  // Defensive: ``kBoolExprNoSubscripts`` is a context flag that should
  // be ``false`` at module-walk start.  Mode-C helpers toggle it via
  // an RAII guard, but if a previous extractAST was aborted mid-walk
  // (an exception reaches here), the flag could still be set.
  kBoolExprNoSubscripts = false;
  clearAllocAliases();

  // D1: reseed per-thread extraction state.  Without this, calling
  // ``extractAST`` standalone (without a preceding ``extractVariables``)
  // or on a DIFFERENT module than the prior ``extractVariables`` would
  // leak stale ``kEntryScope`` / ``kShortNameCollisions`` and produce
  // names that diverge from any subsequent extract on this thread.
  // Same helper ``extractVariables`` uses -- shared so the two paths
  // can't drift.
  prepareExtractionState(module, entry_symbol);

  std::vector<ASTNode> result;
  module.walk([&](mlir::func::FuncOp func) {
    if (!result.empty()) return;  // first PUBLIC func only
    // Skip private siblings.  Set-entry mangles every other
    // function private; after ``fir-polymorphic-op`` resolves a
    // dispatch, the dispatched callee survives as private (kept
    // alive by the type_info dispatch_table).  Walking its body
    // would shadow the real entry's AST whenever its definition
    // appears before the entry's in module order.
    if (func.isPrivate()) return;
    if (!func.getBody().empty()) result = buildAST(func.getBody().front());
  });

  // Prepend one ``kind="symbol_init"`` node per registered position
  // symbol.  Each such node tells the Python emitter to add the
  // symbol to the SDFG and stage an interstate-edge load
  // ``<symbol> = <array>[<one_based_idx> - 1]`` ahead of the body.
  // Stable order (sorted by symbol name) keeps the emitted SDFG
  // deterministic across runs.
  // Prepend one ``<arr>_allocated = 0`` init per allocatable so that
  // a ``res = ALLOCATED(arr)`` read BEFORE the first ALLOCATE returns
  // the correct ``0`` instead of whatever DaCe leaves in the
  // uninitialised transient scalar.  Walks the module's declares and
  // collects every one with the ``allocatable`` Fortran attribute;
  // sorted so the order is deterministic across runs.
  {
    std::vector<std::string> allocNames;
    module.walk([&](hlfir::DeclareOp op) {
      auto attrs = op.getFortranAttrs();
      if (!attrs) return;
      if (!bitEnumContainsAny(*attrs,
                              fir::FortranVariableFlagsEnum::allocatable))
        return;
      std::string raw = extractName(op.getUniqName().str());
      if (raw.empty()) return;
      // Skip allocatables with neither ALLOCATE writes nor
      // ALLOCATED(...) reads  --  the tracker would be dead weight
      // (Phase H).  ``needsAllocatedTracker`` keys on the
      // declare's full uniq_name.
      if (!needsAllocatedTracker(op.getUniqName().str(), module)) return;
      allocNames.push_back(std::move(raw));
    });
    std::sort(allocNames.begin(), allocNames.end());
    allocNames.erase(std::unique(allocNames.begin(), allocNames.end()),
                     allocNames.end());
    if (!allocNames.empty()) {
      std::vector<ASTNode> initNodes;
      initNodes.reserve(allocNames.size());
      for (const auto& n : allocNames) {
        ASTNode init;
        init.kind = "assign";
        init.target = n + "_allocated";
        init.target_is_array = false;
        init.expr = "0";
        initNodes.push_back(std::move(init));
      }
      initNodes.insert(initNodes.end(), result.begin(), result.end());
      result = std::move(initNodes);
    }
  }

  if (!kPosSymbolRegistry.empty()) {
    // (symbol name, source array, per-dim 1-based indices).
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>>>
        entries;
    for (auto& kv : kPosSymbolRegistry)
      entries.emplace_back(kv.second, kv.first.first, kv.first.second);
    std::sort(entries.begin(), entries.end());
    std::vector<ASTNode> initNodes;
    initNodes.reserve(entries.size());
    for (auto& e : entries) {
      ASTNode init;
      init.kind = "symbol_init";
      init.target = std::get<0>(e);       // symbol name
      init.expr = std::get<1>(e);         // source array name
      init.pos_indices = std::get<2>(e);  // per-dim 1-based indices
      // Back-compat scalar mirror for any reader still keying on it.
      init.loop_lower = init.pos_indices.empty() ? 0 : init.pos_indices.front();
      initNodes.push_back(std::move(init));
    }
    initNodes.insert(initNodes.end(), result.begin(), result.end());
    result = std::move(initNodes);
  }
  return result;
}

// Name of the synth scalar carrying an scf.index_switch's selector value.
// The selector traces (through index_cast / convert) back to the scf.while
// result that ``scf.condition`` carried out -- the loop's exit reason -- whose
// synth scalar the scf.condition handler assigns each iteration.
static std::string scfSwitchValueName(mlir::Value v) {
  for (int i = 0; i < limits::kTraceToDeclMax && v; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto cc = mlir::dyn_cast<mlir::arith::IndexCastUIOp>(d)) {
      v = cc.getIn();
      continue;
    }
    if (auto cc = mlir::dyn_cast<mlir::arith::IndexCastOp>(d)) {
      v = cc.getIn();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    break;
  }
  return scfSynthName(v);
}

static std::vector<ASTNode> buildIndexSwitchNodes(mlir::scf::IndexSwitchOp sw) {
  std::string val = scfSwitchValueName(sw.getArg());
  auto cases = sw.getCases();  // ArrayRef<int64_t>: one selector value per case
  // Innermost else == the default region's body.
  std::vector<ASTNode> chain;
  if (!sw.getDefaultRegion().empty())
    chain = buildAST(sw.getDefaultRegion().front());
  // Wrap each case (last first) as ``if (selector == case_i) {body} else
  // {chain-so-far}`` so the per-exit side-effects run.
  auto caseRegions = sw.getCaseRegions();
  for (int i = static_cast<int>(cases.size()) - 1; i >= 0; --i) {
    ASTNode c;
    c.kind = "conditional";
    c.condition = val + " == " + std::to_string(cases[i]);
    if (!caseRegions[i].empty()) c.children = buildAST(caseRegions[i].front());
    c.else_children = std::move(chain);
    chain.clear();
    chain.push_back(std::move(c));
  }
  return chain;
}

}  // namespace hlfir_bridge
