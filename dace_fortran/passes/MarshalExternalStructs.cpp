// ============================================================================
// MarshalExternalStructs.cpp  --  expand struct args of external calls to
// per-member arguments, for deep-copy SoA<->AoS marshalling.
// ============================================================================
//
// A ``keep_external`` procedure receives a Fortran derived type by reference,
// expecting the contiguous array-of-structs (AoS) layout.  But
// ``hlfir-flatten-structs`` later splits a struct into per-member SoA arrays
// for DaCe, so a struct passed whole at the call site would have no coherent
// home after flattening.
//
// This pass (run BEFORE ``hlfir-flatten-structs``) rewrites each external call
// ``ext(S, ...)`` and the external's declaration so the struct travels as its
// individual members: ``ext(S%m1, S%m2, ..., ...)``.  ``hlfir-flatten-structs``
// then turns each ``S%mi`` designate into the SoA flat ``S_mi``, so the call
// ends up referencing the SoA arrays directly -- the SDFG dataflow stays SoA.
// The callee is tagged (``hlfir.aos_marshal_groups`` = flat ``[start, count,
// ...]``) so the binding emitter knows that args ``[start, start+count)`` are
// one struct's members and re-packs them into a local AoS buffer inside the
// generated C tasklet (the AoS cast never appears in the SDFG).
//
// v1 handles a struct whose members are all simple scalars (the per-member
// designates are scalar refs needing no shape).  Array / nested members are a
// follow-up (their designates need shape operands and an outer copy loop).
// ============================================================================

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// A simple scalar type the marshalling handles (matches the dtypes the
/// flatten pass and ``extract_vars`` agree on).
static bool isScalarMember(mlir::Type t) {
  if (t.isF32() || t.isF64()) return true;
  if (t.isInteger(8) || t.isInteger(16) || t.isInteger(32) || t.isInteger(64))
    return true;
  if (mlir::isa<fir::LogicalType>(t)) return true;
  return false;
}

/// A member the *inline-flat AoS pack/unpack* path handles directly: a
/// simple scalar, or a static-shape array of one -- both inline
/// contiguous storage the binding emitter can deep-copy through a C
/// struct buffer.  Distinct from :func:`isRecursiveInlineFlatMember`,
/// which extends the predicate to nested derived types whose own
/// members are inline-flat.
static bool isInlineFlatMember(mlir::Type t) {
  if (isScalarMember(t)) return true;
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) {
    if (!isScalarMember(seq.getEleTy())) return false;
    for (auto d : seq.getShape())
      if (d == fir::SequenceType::getUnknownExtent()) return false;
    return true;
  }
  return false;
}

/// Recursive variant of :func:`isInlineFlatMember`: also accepts a
/// member that is itself a derived type whose every member satisfies
/// this predicate.  The expansion step (``marshal``) walks such a
/// member recursively and unrolls each leaf into its own call arg, so
/// at the binding emitter all that survives is a flat list of scalar /
/// static-shape-array members in declaration order -- exactly what the
/// inline-flat AoS pack/unpack path expects.
///
/// Box / pointer / allocatable / dynamic-shape members still fail this
/// check; those need a separate marshalling shape (the data lives at a
/// runtime descriptor, not inline in the struct's storage) and the
/// :func:`emit_call` AoS body has no path for them.  Refusing here
/// keeps that boundary clean and surfaces through the structured
/// diagnostic in ``emit_library.emit_call``.
static bool isRecursiveInlineFlatMember(mlir::Type t) {
  if (isInlineFlatMember(t)) return true;
  if (auto rec = mlir::dyn_cast<fir::RecordType>(t)) {
    if (rec.getTypeList().empty()) return false;
    for (auto &p : rec.getTypeList())
      if (!isRecursiveInlineFlatMember(p.second)) return false;
    return true;
  }
  return false;
}

/// True iff every member of ``rec`` is recursively inline-flat (see
/// :func:`isRecursiveInlineFlatMember`).
static bool allRecursiveInlineFlatMembers(fir::RecordType rec) {
  if (rec.getTypeList().empty()) return false;
  for (auto &p : rec.getTypeList())
    if (!isRecursiveInlineFlatMember(p.second)) return false;
  return true;
}

/// The marshalable struct a reference type points at, or null.
///
/// Accepts both v1 (every member directly inline-flat) and the v2.1
/// recursive variant (nested derived types whose members are
/// recursively inline-flat).  Box / pointer / allocatable members
/// remain unsupported -- those are the full v2 boundary still to land.
static fir::RecordType scalarStructPointee(mlir::Type argTy) {
  auto ref = mlir::dyn_cast<fir::ReferenceType>(argTy);
  if (!ref) return {};
  auto rec = mlir::dyn_cast<fir::RecordType>(ref.getEleTy());
  if (rec && allRecursiveInlineFlatMembers(rec)) return rec;
  return {};
}

struct MarshalExternalStructsPass
    : public mlir::PassWrapper<MarshalExternalStructsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarshalExternalStructsPass)

  llvm::StringRef getArgument() const final {
    return "hlfir-marshal-external-structs";
  }
  llvm::StringRef getDescription() const final {
    return "Expand struct arguments of registered external calls to per-member "
           "arguments (deep-copy SoA<->AoS marshalling); tags the callee with "
           "the member grouping for the binding emitter.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::StringSet<> externals;
    if (auto a =
            module->getAttrOfType<mlir::ArrayAttr>("hlfir.external_symbols"))
      for (auto e : a)
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(e))
          externals.insert(s.getValue());
    if (externals.empty()) return;  // nothing registered -> no-op

    // Match the same fuzzy convention :func:`externalize_symbols`
    // uses: the registered ``name`` matches a func.func if its symbol
    // is exactly ``name`` (a ``bind(c, name="...")`` external) or ends
    // with ``P<name>`` (a module-procedure mangling like
    // ``_QMm_v2Pext_v2``) or ``_QP<name>`` (a free-procedure mangling
    // like ``_QPext_v2``).  Without this the strict equality refused
    // every non-``bind(c)`` external whose symbol carried a flang
    // mangle prefix; the v1 v2 boundary diagnostic in
    // ``emit_call`` then fired even though the struct shape was
    // perfectly marshalable.
    auto matchesRegistered = [&](llvm::StringRef sym) {
      for (auto &kv : externals) {
        llvm::StringRef n = kv.getKey();
        if (sym == n) return true;
        std::string p1 = ("P" + n).str();
        std::string p2 = ("_QP" + n).str();
        if (sym.ends_with(p1) || sym.ends_with(p2)) return true;
      }
      return false;
    };

    llvm::SmallVector<mlir::func::FuncOp, 4> targets;
    module.walk([&](mlir::func::FuncOp f) {
      if (!f.isDeclaration() || !matchesRegistered(f.getSymName())) return;
      for (mlir::Type t : f.getArgumentTypes())
        if (scalarStructPointee(t)) {
          targets.push_back(f);
          break;
        }
    });
    for (auto f : targets) marshal(f, module);
  }

  /// One leaf of an expanded marshalable struct argument.  The
  /// ``path`` is the chain of component names to walk from the
  /// caller's struct base down to this leaf; for a top-level member
  /// ``foo``, ``path`` is ``["foo"]``; for a nested ``ip%u``,
  /// ``path`` is ``["ip", "u"]``.  ``type`` is the leaf type (the
  /// final scalar / static-shape array of scalar -- nested record
  /// members never appear as leaves, they are walked through).
  struct ExpandedLeaf {
    llvm::SmallVector<mlir::StringAttr, 2> path;
    mlir::Type type;
  };

  /// Recursively walk ``rec`` and append one :struct:`ExpandedLeaf`
  /// per terminal (scalar / static-shape-array) member, prefixing
  /// each leaf's ``path`` with ``prefix`` so the caller can chain
  /// component designates from the outer struct base.  Members that
  /// are themselves derived types are walked through (not emitted as
  /// leaves); the existing ``allRecursiveInlineFlatMembers`` check
  /// guarantees the recursion bottoms out at inline-flat leaves only.
  static void enumerateLeaves(
      fir::RecordType rec, mlir::MLIRContext *ctx,
      llvm::ArrayRef<mlir::StringAttr> prefix,
      llvm::SmallVectorImpl<ExpandedLeaf>& leaves) {
    for (auto &p : rec.getTypeList()) {
      auto name = mlir::StringAttr::get(ctx, p.first);
      if (auto nested = mlir::dyn_cast<fir::RecordType>(p.second)) {
        llvm::SmallVector<mlir::StringAttr, 4> next(prefix.begin(),
                                                    prefix.end());
        next.push_back(name);
        enumerateLeaves(nested, ctx, next, leaves);
        continue;
      }
      ExpandedLeaf leaf;
      leaf.path.append(prefix.begin(), prefix.end());
      leaf.path.push_back(name);
      leaf.type = p.second;
      leaves.push_back(std::move(leaf));
    }
  }

  /// Rewrite ``fn``'s declaration to take each scalar-struct arg's members
  /// individually, tag the grouping, and expand every call site.
  void marshal(mlir::func::FuncOp fn, mlir::ModuleOp module) {
    auto *ctx = fn.getContext();

    // Plan: per original arg, whether it is a scalar struct and the
    // ordered list of *leaf* members (recursive flatten of nested
    // records).  Each leaf becomes one C-ABI arg in the expanded
    // function type; the call sites build the corresponding designate
    // chain.
    llvm::SmallVector<mlir::Type, 8> newArgTys;
    llvm::SmallVector<int64_t, 8> groups;  // flat [start, count, ...]
    llvm::SmallVector<bool, 8> isStruct;
    llvm::SmallVector<llvm::SmallVector<ExpandedLeaf, 4>, 8> members;
    for (mlir::Type t : fn.getArgumentTypes()) {
      auto rec = scalarStructPointee(t);
      if (rec) {
        int64_t start = static_cast<int64_t>(newArgTys.size());
        llvm::SmallVector<ExpandedLeaf, 4> leaves;
        enumerateLeaves(rec, ctx, /*prefix=*/{}, leaves);
        for (auto &leaf : leaves)
          newArgTys.push_back(fir::ReferenceType::get(leaf.type));
        groups.push_back(start);
        groups.push_back(static_cast<int64_t>(leaves.size()));
        isStruct.push_back(true);
        members.push_back(std::move(leaves));
      } else {
        newArgTys.push_back(t);
        isStruct.push_back(false);
        members.push_back({});
      }
    }
    if (groups.empty()) return;

    fn.setType(mlir::FunctionType::get(ctx, newArgTys, fn.getResultTypes()));
    fn->setAttr("hlfir.aos_marshal_groups",
                mlir::DenseI64ArrayAttr::get(ctx, groups));

    llvm::SmallVector<fir::CallOp, 4> calls;
    module.walk([&](fir::CallOp c) {
      if (auto callee = c.getCallee())
        if (callee->getLeafReference().getValue() == fn.getSymName())
          calls.push_back(c);
    });
    for (auto call : calls) rewriteCall(call, isStruct, members);
  }

  /// Replace each scalar-struct operand with per-leaf component designate
  /// chains, rebuilding the call with the expanded operand list.  A top-
  /// level leaf produces one ``hlfir.designate``; a nested-record leaf
  /// produces one designate per path element, chained through the
  /// intermediate record references.
  void rewriteCall(
      fir::CallOp call, llvm::ArrayRef<bool> isStruct,
      llvm::ArrayRef<llvm::SmallVector<ExpandedLeaf, 4>> members) {
    mlir::OpBuilder b(call);
    auto loc = call.getLoc();
    auto args = call.getArgs();
    llvm::SmallVector<mlir::Value, 8> newOperands;
    for (unsigned i = 0; i < args.size(); ++i) {
      if (i < isStruct.size() && isStruct[i]) {
        mlir::Value base = args[i];
        // Resolve the base's pointee record so we can type each
        // intermediate designate -- the path's first element selects a
        // member of *this* record, the next selects from that member's
        // record, and so on.
        auto baseRec = mlir::cast<fir::RecordType>(
            mlir::cast<fir::ReferenceType>(base.getType()).getEleTy());
        for (auto &leaf : members[i]) {
          mlir::Value cursor = base;
          fir::RecordType cursorRec = baseRec;
          for (size_t pi = 0; pi < leaf.path.size(); ++pi) {
            mlir::StringAttr comp = leaf.path[pi];
            // Lookup the next type in the cursorRec's member list.
            mlir::Type nextTy;
            for (auto &p : cursorRec.getTypeList())
              if (p.first == comp.getValue()) {
                nextTy = p.second;
                break;
              }
            // The last path element is the leaf; everything before is
            // a record member we walk through.
            bool isLast = (pi == leaf.path.size() - 1);
            auto refTy = fir::ReferenceType::get(nextTy);
            mlir::Value memShape;
            // Only a static-shape array leaf needs a fir.shape (the
            // designate verifier requires it for a non-box array result).
            if (isLast) {
              if (auto seq = mlir::dyn_cast<fir::SequenceType>(nextTy)) {
                llvm::SmallVector<mlir::Value, 4> dims;
                for (auto e : seq.getShape())
                  dims.push_back(b.create<mlir::arith::ConstantOp>(
                      loc, b.getIndexType(), b.getIndexAttr(e)));
                memShape = b.create<fir::ShapeOp>(loc, dims);
              }
            }
            auto dg = b.create<hlfir::DesignateOp>(
                loc, refTy, cursor, /*component=*/comp,
                /*component_shape=*/memShape, /*indices=*/mlir::ValueRange{},
                /*is_triplet=*/b.getDenseBoolArrayAttr({}),
                /*substring=*/mlir::ValueRange{},
                /*complex_part=*/mlir::BoolAttr{}, /*shape=*/memShape,
                /*typeparams=*/mlir::ValueRange{},
                /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
            cursor = dg.getResult();
            if (!isLast)
              cursorRec = mlir::cast<fir::RecordType>(nextTy);
          }
          newOperands.push_back(cursor);
        }
      } else {
        newOperands.push_back(args[i]);
      }
    }
    // A direct (symbol) ``fir.call``'s operands are exactly its variadic
    // arguments (the callee is an attribute), so replacing the operand list in
    // place expands the argument count without rebuilding the op  --  results,
    // callee, and attributes are preserved.
    call->setOperands(newOperands);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createMarshalExternalStructsPass() {
  return std::make_unique<MarshalExternalStructsPass>();
}

}  // namespace hlfir_bridge
