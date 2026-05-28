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

/// A member the *inline-flat AoS pack/unpack* path handles: a simple scalar,
/// or a static-shape array of one -- both inline contiguous storage the
/// binding emitter can deep-copy through a C struct buffer.
///
/// Distinct from the marshal expansion step (which is permissive for ANY
/// member -- v2): box / nested-record / char members are admissible for
/// expansion so flatten-structs + emit_call can wire the call's per-member
/// SoA leaves, but they cannot participate in the inline-flat AoS pack/unpack
/// path the binding emitter runs for ``Arg(kind="aos")``.  The inline-only
/// callee path (``inline_external``) sidesteps the pack/unpack entirely.
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

/// True iff every member of ``rec`` is inline-flat (scalar or static array of
/// scalar); the marshalling can then deep-copy each member to / from the AoS
/// buffer.  Distinct from ``allExpandableMembers`` (v2) -- inline-flat is the
/// stricter shape the binding emitter's AoS pack/unpack path requires.
static bool allInlineFlatMembers(fir::RecordType rec) {
  if (rec.getTypeList().empty()) return false;
  for (auto &p : rec.getTypeList())
    if (!isInlineFlatMember(p.second)) return false;
  return true;
}

/// The marshalable struct a reference type points at, or null.
///
/// v1 strict shape (in current use): every member is inline-flat (a scalar
/// or a static-shape array of scalar).  The v2 permissive shape (every
/// member admissible) is a separate experiment -- see
/// ``project_external_call_inline_marshal_v2`` in memory for the design.
static fir::RecordType scalarStructPointee(mlir::Type argTy) {
  auto ref = mlir::dyn_cast<fir::ReferenceType>(argTy);
  if (!ref) return {};
  auto rec = mlir::dyn_cast<fir::RecordType>(ref.getEleTy());
  if (rec && allInlineFlatMembers(rec)) return rec;
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

    llvm::SmallVector<mlir::func::FuncOp, 4> targets;
    module.walk([&](mlir::func::FuncOp f) {
      if (!f.isDeclaration() || !externals.contains(f.getSymName())) return;
      for (mlir::Type t : f.getArgumentTypes())
        if (scalarStructPointee(t)) {
          targets.push_back(f);
          break;
        }
    });
    for (auto f : targets) marshal(f, module);
  }

  /// Rewrite ``fn``'s declaration to take each scalar-struct arg's members
  /// individually, tag the grouping, and expand every call site.
  void marshal(mlir::func::FuncOp fn, mlir::ModuleOp module) {
    auto *ctx = fn.getContext();

    // Plan: per original arg, whether it is a scalar struct and its members.
    llvm::SmallVector<mlir::Type, 8> newArgTys;
    llvm::SmallVector<int64_t, 8> groups;  // flat [start, count, ...]
    llvm::SmallVector<bool, 8> isStruct;
    llvm::SmallVector<llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>, 4>, 8>
        members;
    for (mlir::Type t : fn.getArgumentTypes()) {
      auto rec = scalarStructPointee(t);
      if (rec) {
        int64_t start = static_cast<int64_t>(newArgTys.size());
        llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>, 4> mems;
        for (auto &p : rec.getTypeList()) {
          newArgTys.push_back(fir::ReferenceType::get(p.second));
          mems.push_back({mlir::StringAttr::get(ctx, p.first), p.second});
        }
        groups.push_back(start);
        groups.push_back(static_cast<int64_t>(mems.size()));
        isStruct.push_back(true);
        members.push_back(std::move(mems));
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

  /// Replace each scalar-struct operand with per-member component designates,
  /// rebuilding the call with the expanded operand list.
  void rewriteCall(
      fir::CallOp call, llvm::ArrayRef<bool> isStruct,
      llvm::ArrayRef<llvm::SmallVector<std::pair<mlir::StringAttr, mlir::Type>, 4>>
          members) {
    mlir::OpBuilder b(call);
    auto loc = call.getLoc();
    auto args = call.getArgs();
    llvm::SmallVector<mlir::Value, 8> newOperands;
    for (unsigned i = 0; i < args.size(); ++i) {
      if (i < isStruct.size() && isStruct[i]) {
        mlir::Value base = args[i];
        for (auto &m : members[i]) {
          auto refTy = fir::ReferenceType::get(m.second);
          // An array-valued member needs a fir.shape operand (the designate
          // verifier requires it for a non-box array result); scalar members
          // pass a null shape.
          mlir::Value memShape;
          if (auto seq = mlir::dyn_cast<fir::SequenceType>(m.second)) {
            llvm::SmallVector<mlir::Value, 4> dims;
            for (auto e : seq.getShape())
              dims.push_back(b.create<mlir::arith::ConstantOp>(
                  loc, b.getIndexType(), b.getIndexAttr(e)));
            memShape = b.create<fir::ShapeOp>(loc, dims);
          }
          auto dg = b.create<hlfir::DesignateOp>(
              loc, refTy, base, /*component=*/m.first,
              /*component_shape=*/memShape, /*indices=*/mlir::ValueRange{},
              /*is_triplet=*/b.getDenseBoolArrayAttr({}),
              /*substring=*/mlir::ValueRange{},
              /*complex_part=*/mlir::BoolAttr{}, /*shape=*/memShape,
              /*typeparams=*/mlir::ValueRange{},
              /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
          newOperands.push_back(dg.getResult());
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
