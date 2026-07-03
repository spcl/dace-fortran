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

#include <functional>
#include <string>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

namespace hlfir_bridge {

namespace {

/// A simple scalar type the marshalling handles (matches the dtypes the
/// flatten pass and ``extract_vars`` agree on).
static bool isScalarMember(mlir::Type t) {
  if (t.isF32() || t.isF64()) return true;
  if (t.isInteger(8) || t.isInteger(16) || t.isInteger(32) || t.isInteger(64)) return true;
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

/// A box-typed member whose box pointee is an array of scalar -- the
/// shape the v2 marshal extension supports.  Covers both
/// ``fir.box<fir.heap<seq<scalar>>>`` (Fortran ``allocatable``) and
/// ``fir.box<fir.ptr<seq<scalar>>>`` (Fortran ``pointer``).  Any
/// rank (including dynamic extents) is accepted -- the call-site
/// expansion emits ``fir.load`` + ``fir.box_addr`` to extract the
/// data pointer at runtime, so a static shape is not required.
static bool isBoxOfScalarArray(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return false;
  mlir::Type inner = box.getEleTy();
  if (auto heap = mlir::dyn_cast<fir::HeapType>(inner))
    inner = heap.getEleTy();
  else if (auto ptr = mlir::dyn_cast<fir::PointerType>(inner))
    inner = ptr.getEleTy();
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(inner)) return isScalarMember(seq.getEleTy());
  return false;
}

/// A record type all of whose members are simple scalars -- the
/// element type of a *value-record array* member (ICON's
/// ``t_tangent_vectors {v1:f64, v2:f64}``).  Distinct from a general
/// nested record: this is the leaf record the value-record-array path
/// scatters field-by-field, so every member must be a plain scalar
/// (no nested records, no arrays, no boxes).
static bool isScalarRecord(fir::RecordType rec) {
  if (rec.getTypeList().empty()) return false;
  for (auto& p : rec.getTypeList())
    if (!isScalarMember(p.second)) return false;
  return true;
}

/// A box-typed member whose box pointee is an array of a *value record
/// of scalars* -- ICON's ``TYPE(t_tangent_vectors), ALLOCATABLE ::
/// primal_normal_cell(:,:,:)`` (``box<heap<array<record{v1,v2}>>>``).
/// Covers both allocatable (``heap``) and pointer (``ptr``) at any
/// rank.  The flatten pass splits such a member into one per-record-
/// FIELD SoA companion (``..._v1`` / ``..._v2``, each the AoS
/// element rank), and ``bind_c_shim._emit_value_record_array`` emits
/// one C-ABI slot per field; ``enumerateLeaves`` mirrors that by
/// emitting one leaf per record field so the two ABIs coincide.
static bool isBoxOfScalarRecordArray(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return false;
  mlir::Type inner = box.getEleTy();
  if (auto heap = mlir::dyn_cast<fir::HeapType>(inner))
    inner = heap.getEleTy();
  else if (auto ptr = mlir::dyn_cast<fir::PointerType>(inner))
    inner = ptr.getEleTy();
  auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
  if (!seq) return false;
  auto rec = mlir::dyn_cast<fir::RecordType>(seq.getEleTy());
  return rec && isScalarRecord(rec);
}

/// Recursive variant of :func:`isInlineFlatMember`: also accepts a
/// member that is itself a derived type whose every member satisfies
/// this predicate, a box-typed member whose pointee is a
/// scalar-element array, *and* (v2.2) a box-typed member whose pointee
/// is an array of a value record of scalars.  The expansion step
/// (``marshal``) walks recursive-record members down to leaves and
/// treats a box-of-scalar-array member as its own leaf and a
/// box-of-value-record-array member as one leaf per record field;
/// ``rewriteCall`` emits the designate chain for each.  The expanded
/// function's per-leaf arg type is the box pointee (``fir.heap<seq<
/// ...>>`` / ``fir.ptr<seq<...>>``) for a scalar-array leaf, or the
/// per-field scalar ref for a value-record-array field leaf -- so the
/// C ABI of each leaf collapses to a single ``<scalar>*`` pointer
/// matching the per-member SoA slots the sibling-SDFG ``bind_c_shim``
/// produces.
static bool isRecursiveInlineFlatMember(mlir::Type t) {
  if (isInlineFlatMember(t)) return true;
  if (isBoxOfScalarArray(t)) return true;
  if (isBoxOfScalarRecordArray(t)) return true;
  if (auto rec = mlir::dyn_cast<fir::RecordType>(t)) {
    if (rec.getTypeList().empty()) return false;
    for (auto& p : rec.getTypeList())
      if (!isRecursiveInlineFlatMember(p.second)) return false;
    return true;
  }
  return false;
}

/// A SCALAR pointer / allocatable to a record -- ``box<ptr|heap<
/// record>>`` (NOT an array; ``isBoxOfScalarRecordArray`` covers the
/// array case).  This is a *linked-structure handle*: a reference to
/// another aggregate (ICON's ``t_comm_pattern_orig, POINTER ::
/// comm_pat_c`` -- a halo-exchange descriptor).  Such a member has no
/// SoA image -- its data is reached only by chasing the pointer -- so
/// the flatten pass never mints a companion for it (it is skipped,
/// mirroring ``collectFlatLeaves``'s ``partial`` skip in
/// ``FlattenStructs.cpp``).  The marshaller likewise emits NO leaf for
/// it PROVIDED the callee does not read its data
/// (``handleMemberDataUsed`` below); a handle whose data the callee
/// actually needs is a genuine gap and fails loudly rather than
/// silently dropping a live member.
static bool isPointerToRecordHandle(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return false;
  mlir::Type inner = box.getEleTy();
  if (auto heap = mlir::dyn_cast<fir::HeapType>(inner))
    inner = heap.getEleTy();
  else if (auto ptr = mlir::dyn_cast<fir::PointerType>(inner))
    inner = ptr.getEleTy();
  else
    return false;  // a bare ``box<record>`` is not a pointer/allocatable handle
  return mlir::isa<fir::RecordType>(inner);
}

/// True iff every member of ``rec`` is either recursively inline-flat
/// (see :func:`isRecursiveInlineFlatMember`) or a skippable
/// pointer-to-record handle (see :func:`isPointerToRecordHandle`).  At
/// least one member must be genuinely marshalable (an all-handles
/// record has nothing to expand and is not a marshal target).
static bool allRecursiveInlineFlatMembers(fir::RecordType rec) {
  if (rec.getTypeList().empty()) return false;
  bool anyMarshalable = false;
  for (auto& p : rec.getTypeList()) {
    if (isRecursiveInlineFlatMember(p.second))
      anyMarshalable = true;
    else if (!isPointerToRecordHandle(p.second))
      return false;
  }
  return anyMarshalable;
}

/// The marshalable struct a reference type points at, or null.
///
/// Accepts v1 (every member directly inline-flat), the v2.1 recursive
/// variant (nested derived types whose members are recursively
/// inline-flat), v2.2 (value-record-array members), and structs that
/// additionally carry skippable pointer-to-record *handle* members
/// (:func:`isPointerToRecordHandle`) as long as at least one member is
/// genuinely marshalable.
static fir::RecordType scalarStructPointee(mlir::Type argTy) {
  auto ref = mlir::dyn_cast<fir::ReferenceType>(argTy);
  if (!ref) return {};
  auto rec = mlir::dyn_cast<fir::RecordType>(ref.getEleTy());
  if (rec && allRecursiveInlineFlatMembers(rec)) return rec;
  return {};
}

struct MarshalExternalStructsPass
    : public mlir::PassWrapper<MarshalExternalStructsPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarshalExternalStructsPass)

  llvm::StringRef getArgument() const final { return "hlfir-marshal-external-structs"; }
  llvm::StringRef getDescription() const final {
    return "Expand struct arguments of registered external calls to per-member "
           "arguments (deep-copy SoA<->AoS marshalling); tags the callee with "
           "the member grouping for the binding emitter.";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    llvm::StringSet<> externals;
    if (auto a = module->getAttrOfType<mlir::ArrayAttr>("hlfir.external_symbols"))
      for (auto e : a)
        if (auto s = mlir::dyn_cast<mlir::StringAttr>(e)) externals.insert(s.getValue());
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
      for (auto& kv : externals) {
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
  /// final scalar / static-shape array of scalar / box -- nested
  /// record members never appear as leaves, they are walked through).
  ///
  /// A *value-record-array field* leaf (``recordField`` set) is the
  /// v2.2 shape: the last path element is a record FIELD reached
  /// through a box-of-value-record-array member.  ``path`` ends with
  /// ``[..., member, field]``; ``type`` is the box member type (used
  /// to build the ``fir.load`` chain) and ``recordFieldScalar`` is the
  /// field's scalar type (the per-leaf callee arg type).
  /// ``rewriteCall`` builds ``<member-box>`` -> load -> element(1..)
  /// -> ``{field}`` for these, matching the flatten pass's per-field
  /// companion (``..._<field>``) and ``bind_c_shim``'s per-field slot.
  struct ExpandedLeaf {
    llvm::SmallVector<mlir::StringAttr, 2> path;
    mlir::Type type;
    bool recordField = false;
    mlir::Type recordFieldScalar;  // valid iff ``recordField``
    unsigned recordArrayRank = 0;  // AoS element rank of the box array (iff ``recordField``)
  };

  /// Recursively walk ``rec`` and append one :struct:`ExpandedLeaf`
  /// per terminal member, prefixing each leaf's ``path`` with
  /// ``prefix``.  Members that are themselves derived types are
  /// walked through; *box-typed* members are emitted as leaves with
  /// their original (box) type recorded -- ``rewriteCall`` will emit
  /// the ``fir.load`` + ``fir.box_addr`` chain to extract the data
  /// pointer at the call site, and ``marshal`` rewrites the function
  /// arg type to the box pointee.  The existing
  /// ``allRecursiveInlineFlatMembers`` check guarantees every leaf is
  /// either inline-flat or box-of-scalar-array.
  static void enumerateLeaves(fir::RecordType rec, mlir::MLIRContext* ctx, llvm::ArrayRef<mlir::StringAttr> prefix,
                              llvm::SmallVectorImpl<ExpandedLeaf>& leaves) {
    for (auto& p : rec.getTypeList()) {
      auto name = mlir::StringAttr::get(ctx, p.first);
      // Pointer-to-record handle (``comm_pat_c`` -- a halo descriptor):
      // no SoA image, emit no leaf.  ``allRecursiveInlineFlatMembers``
      // already admitted the enclosing struct with such a member; the
      // ``handleMemberUsed`` safety walk (in ``marshal``) fails loudly
      // if the callee actually reads its data, so a silently-dropped
      // LIVE member can't slip through.
      if (isPointerToRecordHandle(p.second)) continue;
      // Box of a value-record array (``primal_normal_cell(:,:,:)`` of
      // ``t_tangent_vectors{v1,v2}``): expand ONE leaf per record
      // FIELD.  Each field leaf's path is ``[...prefix, member,
      // field]`` and it carries the field scalar + the AoS element
      // rank so ``rewriteCall`` builds the element+field designate the
      // flatten pass's ``..._<field>`` companion is derived from -- and
      // ``bind_c_shim._emit_value_record_array`` emits one C slot per
      // field, so the two ABIs coincide.
      if (isBoxOfScalarRecordArray(p.second)) {
        auto box = mlir::cast<fir::BoxType>(p.second);
        mlir::Type inner = box.getEleTy();
        if (auto heap = mlir::dyn_cast<fir::HeapType>(inner))
          inner = heap.getEleTy();
        else if (auto ptr = mlir::dyn_cast<fir::PointerType>(inner))
          inner = ptr.getEleTy();
        auto seq = mlir::cast<fir::SequenceType>(inner);
        auto elemRec = mlir::cast<fir::RecordType>(seq.getEleTy());
        for (auto& f : elemRec.getTypeList()) {
          ExpandedLeaf leaf;
          leaf.path.append(prefix.begin(), prefix.end());
          leaf.path.push_back(name);
          leaf.path.push_back(mlir::StringAttr::get(ctx, f.first));
          leaf.type = p.second;  // the box member type (drives the load chain)
          leaf.recordField = true;
          leaf.recordFieldScalar = f.second;
          leaf.recordArrayRank = seq.getShape().size();
          leaves.push_back(std::move(leaf));
        }
        continue;
      }
      if (mlir::isa<fir::RecordType>(p.second) && !isBoxOfScalarArray(p.second)) {
        llvm::SmallVector<mlir::StringAttr, 4> next(prefix.begin(), prefix.end());
        next.push_back(name);
        enumerateLeaves(mlir::cast<fir::RecordType>(p.second), ctx, next, leaves);
        continue;
      }
      ExpandedLeaf leaf;
      leaf.path.append(prefix.begin(), prefix.end());
      leaf.path.push_back(name);
      leaf.type = p.second;
      leaves.push_back(std::move(leaf));
    }
  }

  /// For a box-typed leaf, the *callee* expects the box pointee --
  /// the data buffer ``fir.box_addr`` returns -- not the box itself.
  /// Map ``fir.box<fir.heap<seq<T>>>`` (allocatable) and
  /// ``fir.box<fir.ptr<seq<T>>>`` (pointer) to the corresponding
  /// inner type so the function type rewrite emits the right per-leaf
  /// arg.  Non-box leaves are returned unchanged.
  static mlir::Type boxLeafCalleeType(mlir::Type t) {
    auto box = mlir::dyn_cast<fir::BoxType>(t);
    if (!box) return t;
    return box.getEleTy();
  }

  /// Safety gate for the skipped pointer-to-record handle members
  /// (:func:`isPointerToRecordHandle`).  Returns the name of the first
  /// handle member of ``structRec`` whose POINTED-TO data is actually
  /// read/written in ``module`` -- i.e. some ``hlfir.designate``
  /// selects a component of the handle's record after chasing the
  /// pointer -- or empty if every handle is a pure pass-through
  /// (only its whole-pointer value is used, e.g. forwarded to a
  /// dropped ``sync`` call).  A non-empty result means the callee
  /// genuinely needs that member's aggregate data, which the
  /// per-member-SoA marshalling cannot supply (recursive record
  /// flattening through a runtime pointer is unimplemented) -- the
  /// caller then fails loudly rather than silently dropping a live
  /// member (the compiles-clean-then-corrupts hazard).
  ///
  /// Detection: a handle's data is used iff some component designate
  /// ``%d{"field"}`` has a memref that, peeling ``fir.load`` /
  /// ``fir.box_addr`` / intermediate element designates, resolves to a
  /// ``%h{"<handleMember>"}`` component designate (the handle selector
  /// itself).  A bare ``%h{"<handleMember>"}`` with no further
  /// component/element user is a pure pass-through and does NOT count.
  static std::string handleMemberDataUsed(fir::RecordType structRec, mlir::ModuleOp module) {
    llvm::StringSet<> handleNames;
    for (auto& p : structRec.getTypeList())
      if (isPointerToRecordHandle(p.second)) handleNames.insert(p.first);
    if (handleNames.empty()) return {};

    // Peel storage-transparent wrappers (load / box_addr) and
    // intermediate element designates (no component) to reach the
    // producing component designate, mirroring the bridge's
    // ``traceToDecl`` peel.
    std::function<mlir::Value(mlir::Value)> peel = [&](mlir::Value v) -> mlir::Value {
      for (int i = 0; i < 16 && v; ++i) {
        auto* d = v.getDefiningOp();
        if (!d) break;
        if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
          v = ld.getMemref();
          continue;
        }
        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
          v = ba.getVal();
          continue;
        }
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
          if (dg.getComponentAttr()) break;  // a component selector -- stop here
          v = dg.getMemref();                // element/section -- keep peeling
          continue;
        }
        break;
      }
      return v;
    };

    std::string used;
    module.walk([&](hlfir::DesignateOp dg) {
      if (!used.empty()) return;
      if (!dg.getComponentAttr()) return;  // only a component access reads record data
      mlir::Value base = peel(dg.getMemref());
      auto* bd = base ? base.getDefiningOp() : nullptr;
      auto parent = mlir::dyn_cast_or_null<hlfir::DesignateOp>(bd);
      if (!parent) return;
      auto pc = parent.getComponentAttr();
      if (pc && handleNames.contains(pc.getValue())) used = pc.getValue().str();
    });
    return used;
  }

  /// Rewrite ``fn``'s declaration to take each scalar-struct arg's members
  /// individually, tag the grouping, and expand every call site.
  void marshal(mlir::func::FuncOp fn, mlir::ModuleOp module) {
    auto* ctx = fn.getContext();

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
        // Loud-fail gate: a pointer-to-record handle member the
        // marshaller SKIPS (``comm_pat_c`` &c.) must be a pure
        // pass-through -- if the callee actually reads its pointed-to
        // data, dropping it would silently corrupt the C ABI.
        if (std::string used = handleMemberDataUsed(rec, module); !used.empty()) {
          fn.emitError() << "hlfir-marshal-external-structs: external '" << fn.getSymName()
                         << "' takes a struct whose pointer-to-record member '" << used
                         << "' has its data read/written -- marshalling a linked-structure "
                            "handle through a runtime pointer is unsupported (recursive record "
                            "flattening not implemented).  Restructure the callee to take the "
                            "member's data as flat arrays, or drop the call.";
          signalPassFailure();
          return;
        }
        int64_t start = static_cast<int64_t>(newArgTys.size());
        llvm::SmallVector<ExpandedLeaf, 4> leaves;
        enumerateLeaves(rec, ctx, /*prefix=*/{}, leaves);
        for (auto& leaf : leaves) {
          // Per-leaf arg type.  A value-record-array FIELD leaf's
          // operand is an element+field designate ``<member>(1..)
          // {field}`` (see ``rewriteCall``), whose result is the
          // field's ``ref<scalar>`` -- so the callee arg is
          // ``ref<fieldScalar>`` (the fir.call then verifies operand
          // == arg).  The C ABI is a ``<scalar>*`` regardless (the
          // binding emitter derives it from the SoA companion array,
          // not this HLFIR arg type), matching ``bind_c_shim``'s
          // per-field pointer slot.  For a box-typed leaf the callee
          // expects the box pointee (the data buffer pointer
          // ``fir.box_addr`` extracts at the call site); for an
          // inline-flat leaf the existing ``ref<scalar | static-array>``
          // shape is preserved.
          if (leaf.recordField) {
            newArgTys.push_back(fir::ReferenceType::get(leaf.recordFieldScalar));
            continue;
          }
          mlir::Type callTy = boxLeafCalleeType(leaf.type);
          if (mlir::isa<fir::BoxType>(leaf.type))
            newArgTys.push_back(callTy);
          else
            newArgTys.push_back(fir::ReferenceType::get(callTy));
        }
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
    fn->setAttr("hlfir.aos_marshal_groups", mlir::DenseI64ArrayAttr::get(ctx, groups));

    llvm::SmallVector<fir::CallOp, 4> calls;
    module.walk([&](fir::CallOp c) {
      if (auto callee = c.getCallee())
        if (callee->getLeafReference().getValue() == fn.getSymName()) calls.push_back(c);
    });
    for (auto call : calls) rewriteCall(call, isStruct, members);
  }

  /// Replace each scalar-struct operand with per-leaf component designate
  /// chains, rebuilding the call with the expanded operand list.  A top-
  /// level leaf produces one ``hlfir.designate``; a nested-record leaf
  /// produces one designate per path element, chained through the
  /// intermediate record references.
  void rewriteCall(fir::CallOp call, llvm::ArrayRef<bool> isStruct,
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
        auto baseRec = mlir::cast<fir::RecordType>(mlir::cast<fir::ReferenceType>(base.getType()).getEleTy());
        for (auto& leaf : members[i]) {
          // A value-record-array field leaf ends its path with the
          // record FIELD (``[...record-members..., box-member,
          // field]``).  Walk the record members to the box member
          // ``ref<box<...>>``, then build the element+field access the
          // flatten pass's ``..._<field>`` companion is derived from:
          // ``fir.load`` the box, designate element ``(1,1,...)`` (the
          // AoS element rank, all-ones), then ``{field}``.  The result
          // is ``ref<fieldScalar>`` -- ``traceToDecl`` composes the
          // flat name ``<...>_<member>_<field>`` from the ``{field}``
          // component regardless of the element indices, so the call
          // arg resolves to the same SoA companion the callee's
          // ``bind_c_shim`` reconstructs, and the fir.call verifies
          // against the ``ref<fieldScalar>`` arg the type rewrite set.
          if (leaf.recordField) {
            mlir::Value cursor = base;
            fir::RecordType cursorRec = baseRec;
            // Walk every path element up to (but not including) the
            // field -- the last-but-one is the box member.
            for (size_t pi = 0; pi + 1 < leaf.path.size(); ++pi) {
              mlir::StringAttr comp = leaf.path[pi];
              mlir::Type nextTy;
              for (auto& p : cursorRec.getTypeList())
                if (p.first == comp.getValue()) {
                  nextTy = p.second;
                  break;
                }
              bool isBoxMember = (pi + 2 == leaf.path.size());
              auto refTy = fir::ReferenceType::get(nextTy);
              auto dg = b.create<hlfir::DesignateOp>(
                  loc, refTy, cursor, /*component=*/comp, /*component_shape=*/mlir::Value{},
                  /*indices=*/mlir::ValueRange{}, /*is_triplet=*/b.getDenseBoolArrayAttr({}),
                  /*substring=*/mlir::ValueRange{}, /*complex_part=*/mlir::BoolAttr{}, /*shape=*/mlir::Value{},
                  /*typeparams=*/mlir::ValueRange{}, /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
              cursor = dg.getResult();
              if (!isBoxMember) cursorRec = mlir::cast<fir::RecordType>(nextTy);
            }
            // ``cursor`` is now ``ref<box<heap|ptr<array<record>>>>``.
            // Load the box, take an element at all-ones indices, then
            // the field.
            auto box = mlir::cast<fir::BoxType>(leaf.type);
            mlir::Type boxInner = box.getEleTy();
            if (auto heap = mlir::dyn_cast<fir::HeapType>(boxInner))
              boxInner = heap.getEleTy();
            else if (auto ptr = mlir::dyn_cast<fir::PointerType>(boxInner))
              boxInner = ptr.getEleTy();
            auto elemRec = mlir::cast<fir::RecordType>(mlir::cast<fir::SequenceType>(boxInner).getEleTy());
            auto loaded = b.create<fir::LoadOp>(loc, cursor);
            llvm::SmallVector<mlir::Value, 4> idxs;
            auto c1 = b.create<mlir::arith::ConstantOp>(loc, b.getIndexType(), b.getIndexAttr(1));
            for (unsigned d = 0; d < leaf.recordArrayRank; ++d) idxs.push_back(c1.getResult());
            auto elemDg = b.create<hlfir::DesignateOp>(loc, fir::ReferenceType::get(elemRec), loaded.getResult(),
                                                       /*indices=*/idxs,
                                                       /*typeparams=*/mlir::ValueRange{},
                                                       /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
            mlir::StringAttr fieldComp = leaf.path.back();
            auto fieldDg = b.create<hlfir::DesignateOp>(
                loc, fir::ReferenceType::get(leaf.recordFieldScalar), elemDg.getResult(),
                /*component=*/fieldComp, /*component_shape=*/mlir::Value{},
                /*indices=*/mlir::ValueRange{}, /*is_triplet=*/b.getDenseBoolArrayAttr({}),
                /*substring=*/mlir::ValueRange{}, /*complex_part=*/mlir::BoolAttr{}, /*shape=*/mlir::Value{},
                /*typeparams=*/mlir::ValueRange{}, /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
            newOperands.push_back(fieldDg.getResult());
            continue;
          }
          mlir::Value cursor = base;
          fir::RecordType cursorRec = baseRec;
          for (size_t pi = 0; pi < leaf.path.size(); ++pi) {
            mlir::StringAttr comp = leaf.path[pi];
            // Lookup the next type in the cursorRec's member list.
            mlir::Type nextTy;
            for (auto& p : cursorRec.getTypeList())
              if (p.first == comp.getValue()) {
                nextTy = p.second;
                break;
              }
            // The last path element is the leaf; everything before is
            // a record member we walk through.
            bool isLast = (pi == leaf.path.size() - 1);
            // Result type of *this* designate.  For an inline-flat
            // leaf the existing ``ref<...>`` wrapping is used; for a
            // box-typed leaf we keep the ``ref<box<...>>`` wrapping
            // (the box address as stored in the parent record), then
            // the post-walk ``fir.load`` + ``fir.box_addr`` extracts
            // the data pointer.
            auto refTy = fir::ReferenceType::get(nextTy);
            mlir::Value memShape;
            // Only a static-shape array leaf needs a fir.shape (the
            // designate verifier requires it for a non-box array result).
            // Box leaves do *not* get a shape -- the box already
            // carries its dynamic extents and the designate's result
            // is ``ref<box<...>>``.
            if (isLast && !mlir::isa<fir::BoxType>(nextTy)) {
              if (auto seq = mlir::dyn_cast<fir::SequenceType>(nextTy)) {
                llvm::SmallVector<mlir::Value, 4> dims;
                for (auto e : seq.getShape())
                  dims.push_back(b.create<mlir::arith::ConstantOp>(loc, b.getIndexType(), b.getIndexAttr(e)));
                memShape = b.create<fir::ShapeOp>(loc, dims);
              }
            }
            auto dg = b.create<hlfir::DesignateOp>(loc, refTy, cursor, /*component=*/comp,
                                                   /*component_shape=*/memShape, /*indices=*/mlir::ValueRange{},
                                                   /*is_triplet=*/b.getDenseBoolArrayAttr({}),
                                                   /*substring=*/mlir::ValueRange{},
                                                   /*complex_part=*/mlir::BoolAttr{}, /*shape=*/memShape,
                                                   /*typeparams=*/mlir::ValueRange{},
                                                   /*fortran_attrs=*/fir::FortranVariableFlagsAttr{});
            cursor = dg.getResult();
            if (!isLast) cursorRec = mlir::cast<fir::RecordType>(nextTy);
          }
          // Box-typed leaf: at this point ``cursor`` is a ``ref<box<...>>``
          // (the address of the box descriptor stored inside the
          // struct).  Extract the data pointer the external expects
          // via the canonical ``fir.load`` + ``fir.box_addr`` chain:
          // load the box value, then ``box_addr`` to drop the
          // descriptor and surface ``fir.heap<seq<...>>`` (allocatable)
          // or ``fir.ptr<seq<...>>`` (pointer) -- the per-leaf arg
          // type ``marshal`` rewrote the function declaration to.
          if (mlir::isa<fir::BoxType>(leaf.type)) {
            auto box = mlir::cast<fir::BoxType>(leaf.type);
            auto loaded = b.create<fir::LoadOp>(loc, cursor);
            auto addr = b.create<fir::BoxAddrOp>(loc, box.getEleTy(), loaded.getResult());
            cursor = addr.getResult();
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
