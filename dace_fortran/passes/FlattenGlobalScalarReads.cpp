// ============================================================================
// FlattenGlobalScalarReads.cpp  --  Rewrite read-only scalar-member reads of a
//                                    module-global struct into reads of a
//                                    synthetic per-member global, and record
//                                    the provenance in the flatten plan.
// ============================================================================

#include <string>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "passes/Passes.h"

#define DEBUG_TYPE "flatten-global-scalar-reads"

namespace hlfir_bridge {

namespace {

constexpr unsigned kTraceDepth = 16;

llvm::StringRef traceToGlobalSym(mlir::Value v) {
  for (unsigned d = 0; d < kTraceDepth && v; ++d) {
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto addrOf = mlir::dyn_cast<fir::AddrOfOp>(def)) return addrOf.getSymbol().getRootReference().getValue();
    if (auto decl = mlir::dyn_cast<hlfir::DeclareOp>(def)) {
      v = decl.getMemref();
      continue;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(def)) {
      v = conv.getValue();
      continue;
    }
    if (auto boxAddr = mlir::dyn_cast<fir::BoxAddrOp>(def)) {
      v = boxAddr.getVal();
      continue;
    }
    return {};
  }
  return {};
}

llvm::StringRef traceWriteToGlobalSym(mlir::Value v) {
  for (unsigned d = 0; d < kTraceDepth && v; ++d) {
    mlir::Operation* def = v.getDefiningOp();
    if (!def) return {};
    if (auto desig = mlir::dyn_cast<hlfir::DesignateOp>(def)) {
      v = desig.getMemref();
      continue;
    }
    return traceToGlobalSym(v);
  }
  return {};
}

bool splitModuleScopeSymbol(llvm::StringRef sym, llvm::StringRef& mod, llvm::StringRef& entity) {
  llvm::StringRef s = sym;
  if (!s.consume_front("_QM")) return false;
  size_t e = llvm::StringRef::npos;
  for (size_t i = 0; i < s.size(); ++i)
    if (std::isupper(static_cast<unsigned char>(s[i]))) {
      if (s[i] != 'E') return false;
      e = i;
      break;
    }
  if (e == llvm::StringRef::npos || e == 0 || e + 1 >= s.size()) return false;
  mod = s.take_front(e);
  entity = s.drop_front(e + 1);
  for (char const c : entity)
    if (!std::islower(static_cast<unsigned char>(c)) && !std::isdigit(static_cast<unsigned char>(c)) && c != '_')
      return false;
  return true;
}

std::string scalarDtypeName(mlir::Type t) {
  if (t.isF32()) return "float32";
  if (t.isF64()) return "float64";
  if (t.isInteger(8)) return "int8";
  if (t.isInteger(16)) return "int16";
  if (t.isInteger(32)) return "int32";
  if (t.isInteger(64)) return "int64";
  if (t.isInteger(1) || mlir::isa<fir::LogicalType>(t)) return "bool";
  if (auto ct = mlir::dyn_cast<mlir::ComplexType>(t)) {
    mlir::Type const et = ct.getElementType();
    if (et.isF32()) return "complex64";
    if (et.isF64()) return "complex128";
  }
  return "";
}

bool isPlainComponentDesignate(hlfir::DesignateOp d) {
  return d.getComponent().has_value() && d.getIndices().empty() && d.getSubstring().empty() && !d.getComponentShape() &&
         !d.getShape() && d.getTypeparams().empty() && !d.getComplexPart().has_value() && d->getNumResults() == 1;
}

struct Candidate {
  hlfir::DesignateOp designate;
  std::string symbol;
  std::string module;
  std::string entity;
  std::string member;
  std::string dtype;
  mlir::Type scalarTy;
};

struct FlattenGlobalScalarReadsPass
    : public mlir::PassWrapper<FlattenGlobalScalarReadsPass, mlir::OperationPass<mlir::ModuleOp>> {
  // NOLINTNEXTLINE(misc-const-correctness): 'id' is defined by the LLVM MLIR_DEFINE_*_TYPE_ID macro.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenGlobalScalarReadsPass)

  llvm::StringRef getArgument() const final { return "hlfir-flatten-global-scalar-reads"; }
  llvm::StringRef getDescription() const final {
    return "Rewrite a never-written scalar component read of a module-scope "
           "fir.global record (``mod%entity%member``) into a read of a "
           "synthetic bodiless ``fir.global`` carrying that member alone, and "
           "record {symbol, module, entity, member, dtype} in the "
           "hlfir.flatten_plan side table so the bindings emitter can assign "
           "``symbol = entity%member`` at the call boundary.";
  }

  void runOnOperation() override {
    auto module = getOperation();

    llvm::StringSet<> writtenGlobals;
    module.walk([&](mlir::Operation* op) {
      mlir::Value target;
      if (auto store = mlir::dyn_cast<fir::StoreOp>(op))
        target = store.getMemref();
      else if (auto assign = mlir::dyn_cast<hlfir::AssignOp>(op))
        target = assign.getLhs();
      else
        return;
      llvm::StringRef const sym = traceWriteToGlobalSym(target);
      if (!sym.empty()) writtenGlobals.insert(sym);
    });

    llvm::StringSet<> takenNames;
    module.walk([&](fir::GlobalOp g) { takenNames.insert(g.getSymName()); });
    module.walk([&](hlfir::DeclareOp d) { takenNames.insert(d.getUniqName()); });

    llvm::SmallVector<Candidate, 4> candidates;
    llvm::StringSet<> rejected;
    module.walk([&](hlfir::DesignateOp d) {
      if (!isPlainComponentDesignate(d)) return;
      llvm::StringRef const sym = traceToGlobalSym(d.getMemref());
      if (sym.empty()) return;
      llvm::StringRef mod;
      llvm::StringRef entity;
      if (!splitModuleScopeSymbol(sym, mod, entity)) return;
      auto g = module.lookupSymbol<fir::GlobalOp>(sym);
      if (!g || (g.getConstant().has_value() && *g.getConstant())) return;
      if (!mlir::isa<fir::RecordType>(g.getType())) return;

      llvm::StringRef const member = d.getComponent()->getValue();
      std::string const newSym = (llvm::Twine(sym) + "_" + member).str();
      auto refTy = mlir::dyn_cast<fir::ReferenceType>(d.getResult().getType());
      if (!refTy) return;
      std::string const dtype = scalarDtypeName(refTy.getEleTy());

      bool ok = !dtype.empty() && !writtenGlobals.contains(sym) && !takenNames.contains(newSym);
      for (mlir::Operation* user : d.getResult().getUsers())
        if (!mlir::isa<fir::LoadOp>(user)) ok = false;
      if (!ok) {
        rejected.insert(newSym);
        return;
      }
      candidates.push_back(Candidate{d, sym.str(), mod.str(), entity.str(), member.str(), dtype, refTy.getEleTy()});
    });

    mlir::Builder b(&getContext());
    llvm::SmallVector<mlir::Attribute, 4> table;
    llvm::StringSet<> emitted;
    unsigned rewritten = 0;
    for (Candidate& c : candidates) {
      std::string const newSym = c.symbol + "_" + c.member;
      if (rejected.contains(newSym)) continue;
      if (emitted.insert(newSym).second) {
        mlir::OpBuilder gb(&getContext());
        gb.setInsertionPointToEnd(module.getBody());
        gb.create<fir::GlobalOp>(c.designate.getLoc(), llvm::StringRef(newSym), c.scalarTy,
                                 llvm::ArrayRef<mlir::NamedAttribute>{});
        table.push_back(b.getDictionaryAttr({
            b.getNamedAttr("symbol", b.getStringAttr(newSym)),
            b.getNamedAttr("module", b.getStringAttr(c.module)),
            b.getNamedAttr("entity", b.getStringAttr(c.entity)),
            b.getNamedAttr("member", b.getStringAttr(c.member)),
            b.getNamedAttr("dtype", b.getStringAttr(c.dtype)),
        }));
      }
      mlir::OpBuilder rb(c.designate);
      mlir::Location const loc = c.designate.getLoc();
      auto refTy = fir::ReferenceType::get(c.scalarTy);
      auto addr = rb.create<fir::AddrOfOp>(loc, refTy, mlir::SymbolRefAttr::get(&getContext(), newSym));
      auto decl = rb.create<hlfir::DeclareOp>(loc, addr.getResult(), newSym, /*shape=*/mlir::Value{});
      c.designate.getResult().replaceAllUsesWith(decl.getResult(0));
      c.designate.erase();
      ++rewritten;
    }

    if (table.empty()) return;

    llvm::SmallVector<mlir::NamedAttribute, 2> planAttrs;
    if (auto existing = module->getAttrOfType<mlir::DictionaryAttr>("hlfir.flatten_plan"))
      for (mlir::NamedAttribute na : existing)
        if (na.getName() != "synthetic_globals") planAttrs.push_back(na);
    planAttrs.push_back(b.getNamedAttr("synthetic_globals", b.getArrayAttr(table)));
    module->setAttr("hlfir.flatten_plan", b.getDictionaryAttr(planAttrs));

    LLVM_DEBUG(llvm::dbgs() << "FlattenGlobalScalarReads: " << table.size() << " synthetic global(s), " << rewritten
                            << " designate(s) rewritten\n");
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::Pass> createFlattenGlobalScalarReadsPass() {
  return std::make_unique<FlattenGlobalScalarReadsPass>();
}

}  // namespace hlfir_bridge
