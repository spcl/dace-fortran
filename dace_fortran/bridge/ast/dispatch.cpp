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
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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

std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block& block) {
  std::vector<ASTNode> out;
  for (auto& op : block) {
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      out.push_back(buildScfIfAsConditional(ifOp));
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
      // ``scf.condition(%c)``: break when %c is false.
      ASTNode guard;
      guard.kind = "conditional";
      auto b = buildBoolExpr(condOp.getCondition(), 0);
      guard.condition = "not (" + b + ")";
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
    // Pure-value ops  --  no AST node, their values flow inline.
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
      low == "mpi_irecv" || low == "mpi_wait")
    return low;
  return std::string{};
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
        throw std::runtime_error(
            "fir.do_loop with non-constant step  --  bridge "
            "currently lowers only constant-step loops. The "
            "step's sign decides forward-vs-reverse codegen; "
            "with a symbolic step we'd silently default to +1 "
            "and produce wrong-direction iteration when the "
            "symbol is negative.");
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
      {
        auto* sop = src.getDefiningOp();
        FILE* fd = fopen("/tmp/probes/dbg.log", "a");
        if (fd) {
          fprintf(fd, "DEBUG ASSIGN src=%s\n",
                  sop ? sop->getName().getStringRef().str().c_str() : "<null>");
          fclose(fd);
        }
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
            {"hlfir.transpose", "transpose"},
            {"hlfir.dot_product", "dot_product"},
            // Fortran ``COUNT(mask [, dim])``  --  routed through
            // ``CountLibraryNode`` so its ``cast -> Reduce``
            // expansion handles the integer-cast and the
            // per-target reduction lowering.  ``buildLibCallNode``
            // picks up the optional ``dim`` operand and threads
            // it through the ASTNode for ``emit_libcall``.
            {"hlfir.count", "count"},
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
              auto elem = mlir::dyn_cast<hlfir::ElementalOp>(od);
              if (!elem) continue;
              auto [trName, prelude] = materialiseElementalForLibcall(elem);
              if (trName.empty()) continue;
              for (auto& n : prelude) nodes.push_back(std::move(n));
              elemSubst[i] = std::move(trName);
              needSubst = true;
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
          llvm::StringRef identity;  // initial accumulator value
          llvm::StringRef py_op;     // Python binary op for
                                     // section-reduce loop body;
                                     // empty -> fall back to
                                     // buildReduceNode (whole-array)
        };
        static const RedEntry kRedTable[] = {
            {"hlfir.sum", "lambda a, b: a + b", "0", "+"},
            {"hlfir.product", "lambda a, b: a * b", "1", "*"},
            // Identity strings use the bare ``inf`` token (not
            // ``math.inf``) so DaCe's cppunparse  --  which maps
            // ``inf`` -> ``INFINITY`` via _py2c_reserved  --  emits
            // a valid C++ literal in the section-reduce init
            // tasklet.  The whole-array Reduce path's eval()
            // namespace is patched with ``inf=math.inf`` for
            // the same string.
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
                        assign, dg, e.py_op.str(), e.identity.str());
                    if (!built.empty()) {
                      for (auto& bn : built) nodes.push_back(std::move(bn));
                      emitted = true;
                    }
                  }
                }
              }
            }
            // Mode-C for ``hlfir.any`` / ``hlfir.all``: the
            // reduction's source is an ``hlfir.elemental``
            // (compound boolean expression), so ``traceToDecl``
            // returns "" and the plain Reduce path explodes
            // with ``reduction source '' not registered``.
            // Materialise the elemental into a transient mask
            // via a per-element loop (same pattern as Mode-C
            // COUNT) and route the Reduce over the transient.
            if (!emitted && (e.op == "hlfir.any" || e.op == "hlfir.all") &&
                sd->getNumOperands() > 0) {
              auto srcVal = sd->getOperand(0);
              if (auto* srcOp = srcVal.getDefiningOp())
                if (auto elem_src = mlir::dyn_cast<hlfir::ElementalOp>(srcOp)) {
                  auto built = buildElementalAnyAllReduce(
                      assign, elem_src, e.wcr.str(), e.identity.str());
                  if (!built.empty()) {
                    for (auto& bn : built) nodes.push_back(std::move(bn));
                    emitted = true;
                  }
                }
            }
            if (!emitted) {
              nodes.push_back(
                  buildReduceNode(assign, sd, e.wcr.str(), e.identity.str()));
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

std::vector<ASTNode> extractAST(mlir::ModuleOp module) {
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
