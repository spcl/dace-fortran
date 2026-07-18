// trace_utils.cpp -- shared SSA tracing utilities.

#include "bridge/trace_utils.h"

#include <algorithm>
#include <unordered_map>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

// Disambiguation overrides for Fortran short-name collisions across inlined scopes (hlfir-inline-all can leave a caller
// declare and callee dummy declare with the same short name); extract_vars populates mangled -> unique_short_name for
// colliding entries.
static thread_local std::unordered_map<std::string, std::string> kManglingOverride;

void setManglingOverride(const std::string& mangled, const std::string& shortName) {
  kManglingOverride[mangled] = shortName;
}

// Per-thread entry F-scope, set once by setEntryScope; extractName consults it to scope-qualify non-entry-scope
// declares. Empty means skip qualification (back-compat).
static thread_local std::string kEntryScope;
static thread_local std::set<std::string> kShortNameCollisions;

// Cache of every hlfir.declare uniq_name in the module, built lazily on first flatCompanionName query, invalidated in
// clearManglingOverrides; avoids an O(declares) walk per access.
static thread_local llvm::StringSet<> kModuleDeclUniqs;
static thread_local bool kModuleDeclUniqsBuilt = false;

void clearManglingOverrides() {
  kManglingOverride.clear();
  kEntryScope.clear();
  kShortNameCollisions.clear();
  kModuleDeclUniqs.clear();
  kModuleDeclUniqsBuilt = false;
}

void setEntryScope(const std::string& scope) { kEntryScope = scope; }

void setShortNameCollisions(const std::set<std::string>& collisions) { kShortNameCollisions = collisions; }

std::string getFScope(const std::string& uniq) {
  // Fortran mangled-name shape: _QM<mod>F<func>E<name> (nested F segments possible); take the F immediately before the
  // last E.
  auto eP = uniq.rfind('E');
  if (eP == std::string::npos) return {};
  auto fP = uniq.rfind('F', eP);
  if (fP == std::string::npos || fP + 1 >= eP) return {};
  return uniq.substr(fP + 1, eP - fP - 1);
}

std::string extractName(const std::string& m) {
  auto it = kManglingOverride.find(m);
  if (it != kManglingOverride.end()) return it->second;
  auto p = m.rfind('E');
  std::string name = p != std::string::npos ? m.substr(p + 1) : m;

  // Scope-qualify non-entry-scope declares on demand: <scope>_<short> for those, bare short name for entry-scope
  // declares and module globals (no F segment) so the SDFG signature matches the Fortran interface. Skips re-prefixing
  // names that already start with <scope>_.
  if (!kEntryScope.empty()) {
    std::string const scope = getFScope(m);
    if (!scope.empty() && scope != kEntryScope) {
      // Qualify ONLY when the short name actually collides across scopes (kShortNameCollisions, from extractVariables's
      // pre-walk) -- else unused inlined-callee dummies gain extra SDFG signature variables.
      bool const collides = kShortNameCollisions.count(name) > 0;
      if (collides) {
        std::string const prefix = scope + "_";
        if (name.compare(0, prefix.size(), prefix) != 0) name = prefix + name;
      }
    }
  }

  // Sanitize dots: flang emits compiler-generated globals like _QQro.4xi4.0 whose names contain '.', which DaCe's
  // NestedDict rejects as a nested-key separator. Collision-free since Fortran identifiers can't contain '.'.
  std::replace(name.begin(), name.end(), '.', '_');

  // Double-underscore names (e.g. __assoc_scalar_N from ExpandVectorSubscriptGather) are invalid Fortran dummies and
  // make the generated binding uncompilable; prefix a letter. Scoped to "__" not bare "_" -- single-underscore names
  // are internal markers that must stay unrenamed (renaming desyncs shape-symbol construction).
  if (name.rfind("__", 0) == 0) name = "f" + name;
  return name;
}

// Allocatable re-allocation alias map, keyed by the raw Fortran name; updated as the IR walker passes
// fir.allocmem-bound fir.store ops (extract_ast.cpp), read by traceToDecl so downstream accesses land on the live SDFG
// transient.
static thread_local std::unordered_map<std::string, std::string> kAllocAlias;

std::string allocAliasFor(const std::string& raw) {
  auto it = kAllocAlias.find(raw);
  return it == kAllocAlias.end() ? raw : it->second;
}

void setAllocAlias(const std::string& raw, const std::string& alias) {
  if (alias == raw)
    kAllocAlias.erase(raw);
  else
    kAllocAlias[raw] = alias;
}

void clearAllocAliases() { kAllocAlias.clear(); }

// True iff mr (a declare's memref) peels to an hlfir.designate selecting a struct COMPONENT -- identifies an
// inlined-call dummy bound to a caller struct member (dummy scope but memref is a component designate, not a
// block-arg/alloca). Single source of truth for the storage-transparent reinterpret peel
// (fir.convert/load/embox/rebox/box_addr, hlfir.copy_in/coordinate_of/as_expr) shared by every access-path walker;
// centralising it here is what keeps per-walker peel subsets from drifting apart.
mlir::Value peelBoxReinterpret(mlir::Value v, int maxDepth) {
  for (int i = 0; i < maxDepth && v; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto c = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = c.getValue();
      continue;
    }
    if (auto l = mlir::dyn_cast<fir::LoadOp>(d)) {
      v = l.getMemref();
      continue;
    }
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      v = eb.getMemref();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      v = rb.getBox();
      continue;
    }
    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
      v = ba.getVal();
      continue;
    }
    if (auto cp = mlir::dyn_cast<hlfir::CopyInOp>(d)) {
      v = cp.getVar();
      continue;
    }
    if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d)) {
      v = co.getRef();
      continue;
    }
    if (auto ae = mlir::dyn_cast<hlfir::AsExprOp>(d)) {
      v = ae.getVar();
      continue;
    }
    break;  // designate / declare / alloca / block-arg / arith -- caller handles
  }
  return v;
}

bool leadsToComponentDesignate(mlir::Value mr) {
  // Peel via the shared reinterpret set, then require the terminal to be a struct-COMPONENT designate; using the shared
  // peel (not an inline cascade) keeps this gate in lock-step with the other access-path walkers.
  mr = peelBoxReinterpret(mr);
  auto* d = mr ? mr.getDefiningOp() : nullptr;
  auto dg = d ? mlir::dyn_cast<hlfir::DesignateOp>(d) : nullptr;
  return dg && static_cast<bool>(dg.getComponentAttr());
}

// Peel a section's base through box reinterprets, inlined whole-array-dummy aliases, and nested non-component
// designates; true iff it reaches a struct-COMPONENT designate. Stricter-peeling variant of leadsToComponentDesignate,
// which stops at the intermediate inlined dummy declare.
static bool sectionBaseReachesComponent(mlir::Value mr) {
  for (int i = 0; i < limits::kAliasMemrefWalkDepth && mr; ++i) {
    if (mlir::Value const peeled = peelBoxReinterpret(mr); peeled != mr) {
      mr = peeled;
      continue;
    }
    auto* d = mr.getDefiningOp();
    if (!d) return false;
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      if (dg.getComponentAttr()) return true;  // reached the struct member
      mr = dg.getMemref();                     // nested section / element -- keep walking
      continue;
    }
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      if (auto outer = asAssumedShapeAlias(dc)) {
        mr = outer.getResult(0);
        continue;
      }
      mr = dc.getMemref();  // inlined whole-array dummy -> its caller-side source
      continue;
    }
    return false;
  }
  return false;
}

hlfir::DesignateOp asSectionOverComponent(mlir::Value v) {
  v = peelBoxReinterpret(v);
  auto* d = v ? v.getDefiningOp() : nullptr;
  auto sec = d ? mlir::dyn_cast<hlfir::DesignateOp>(d) : nullptr;
  if (!sec) return {};
  // A PURE section: triplet/scalar subscripts, no component selector.
  if (sec.getIsTriplet().empty() || sec.getComponentAttr()) return {};
  // AoR only: the section's element must be a derived (record) type; a plain-real member section is a
  // flattened-companion + rewriteSectionedAliasLeaf case and must stay on that path.
  mlir::Type et = sec.getResult().getType();
  if (auto bt = mlir::dyn_cast<fir::BoxType>(et)) et = bt.getEleTy();
  if (auto rt = mlir::dyn_cast<fir::ReferenceType>(et)) et = rt.getEleTy();
  if (auto seq = mlir::dyn_cast<fir::SequenceType>(et)) et = seq.getEleTy();
  if (!mlir::isa<fir::RecordType>(et)) return {};
  if (!sectionBaseReachesComponent(sec.getMemref())) return {};
  return sec;
}

hlfir::DesignateOp asInlinedSectionOverComponent(hlfir::DeclareOp decl) {
  return asSectionOverComponent(decl.getMemref());
}

unsigned countScalarSectionDims(hlfir::DesignateOp sec) {
  unsigned n = 0;
  for (bool const t : sec.getIsTriplet())
    if (!t) ++n;
  return n;
}

mlir::Value traceLocalPointerRebindSource(hlfir::DeclareOp decl) {
  // Gate 1: a LOCAL Fortran POINTER (pointer attribute set, no dummy scope -- an entry/inlined-callee pointer dummy is
  // bound by its caller, not rebound here).
  auto attrs = decl.getFortranAttrs();
  if (!attrs || !bitEnumContainsAny(*attrs, fir::FortranVariableFlagsEnum::pointer)) return {};
  if (decl.getDummyScope()) return {};
  // Gate 2: the pointee is a whole DERIVED-TYPE OBJECT (box<ptr|heap<record>>), not scalar/array -- a scalar/array
  // pointer VIEW is lowered by view_alias and must keep its own name.
  mlir::Type pointee = decl.getResult(0).getType();
  if (auto ref = mlir::dyn_cast<fir::ReferenceType>(pointee)) pointee = ref.getEleTy();
  if (!pointerToRecordMember(pointee)) return {};
  // Gate 3: exactly ONE non-nullify rebind store fir.store <box> to decl#0; the initial embox(zero_bits) nullify is
  // skipped, and more than one live rebind is ambiguous -> refuse.
  mlir::Value stored;
  for (auto* u : decl.getResult(0).getUsers()) {
    auto st = mlir::dyn_cast<fir::StoreOp>(u);
    if (!st || st.getMemref() != decl.getResult(0)) continue;
    if (auto eb = mlir::dyn_cast_or_null<fir::EmboxOp>(st.getValue().getDefiningOp()))
      if (mlir::isa_and_nonnull<fir::ZeroOp>(eb.getMemref().getDefiningOp())) continue;
    if (stored) return {};
    stored = st.getValue();
  }
  if (!stored) return {};
  // Peel the storage-transparent box reinterprets to the source declare or component designate -- the same op set
  // RewritePointerAssigns::traceRebindChain peels.
  mlir::Value const src = peelBoxReinterpret(stored);
  if (!src) return {};
  // Gate 4: the source must root at a struct DUMMY (component designate or struct-dummy declare, incl. multi-hop dummy
  // chains). A rebind onto a whole named object (p => g) is left to the object_aliases mechanism instead -- following
  // it here would rename the rebind's own store target and drop the alias edge.
  auto* sd = src.getDefiningOp();
  if (auto dg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(sd)) return dg.getComponentAttr() ? src : mlir::Value{};
  if (auto dc = mlir::dyn_cast_or_null<hlfir::DeclareOp>(sd)) return dc.getDummyScope() ? src : mlir::Value{};
  return {};
}

mlir::Value matchAssociatedStatusBoxRef(mlir::arith::CmpIOp cmp) {
  // The intrinsic lowers to a NE-against-null comparison.
  if (cmp.getPredicate() != mlir::arith::CmpIPredicate::ne) return {};
  bool rhsZero = false;
  if (auto c = traceConstInt(cmp.getRhs())) rhsZero = (*c == 0);
  if (!rhsZero) return {};
  // Peel the heap-addr -> i64 fir.convert chain on the LHS back to the fir.box_addr.
  mlir::Value cur = cmp.getLhs();
  for (int i = 0; i < limits::kConvertChainDepth && cur; ++i) {
    auto* cd = cur.getDefiningOp();
    if (!cd) break;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(cd)) {
      cur = cv.getValue();
      continue;
    }
    break;
  }
  auto* cd = cur ? cur.getDefiningOp() : nullptr;
  auto ba = cd ? mlir::dyn_cast<fir::BoxAddrOp>(cd) : nullptr;
  if (!ba) return {};
  // box_addr's operand is the box, usually loaded from a box reference; trace through that fir.load so the returned
  // value is what traceToDecl names.
  mlir::Value src = ba.getVal();
  if (auto* sd = src.getDefiningOp())
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(sd)) src = ld.getMemref();
  return src;
}

// True iff some hlfir.declare in anyOp's module has uniq_name uniq. Backed by a per-build cache (kModuleDeclUniqs).
static bool moduleHasDeclare(mlir::Operation* anyOp, llvm::StringRef uniq) {
  if (!kModuleDeclUniqsBuilt) {
    if (auto mod = anyOp->getParentOfType<mlir::ModuleOp>()) {
      mod.walk([&](hlfir::DeclareOp dc) { kModuleDeclUniqs.insert(dc.getUniqName()); });
      kModuleDeclUniqsBuilt = true;
    }
  }
  return kModuleDeclUniqs.contains(uniq);
}

// A component designate over a FLATTENED struct must resolve to the companion's OWN registered name, not a re-composed
// extractName(base) + "_" + member -- base and companion can have different cross-scope collision status. Resolves by
// construction when such a declare exists; "" otherwise (caller falls back to plain parent_member composition).
static std::string flatCompanionName(hlfir::DesignateOp dg, llvm::StringRef member) {
  // Peel the memref to the base declare's uniq_name, mirroring traceToDecl's walk so we land on the same storage
  // declare (incl. assumed-shape aliases -> outer storage).
  mlir::Value v = dg.getMemref();
  std::string baseUniq;
  for (int i = 0; i < 64 && v; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      if (auto outer = asAssumedShapeAlias(dc)) {
        v = outer.getResult(0);
        continue;
      }
      baseUniq = dc.getUniqName().str();
      break;
    }
    if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d)) {
      baseUniq = dc.getUniqName().str();
      break;
    }
    // Shared storage-transparent peel, runs before the designate case below (a designate value returns unchanged, so a
    // nested-component parent still falls through to the multi-level guard).
    if (mlir::Value const peeled = peelBoxReinterpret(v); peeled != v) {
      v = peeled;
      continue;
    }
    if (auto pd = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      // A nested component parent (a % b % member) is a multi-level path the single-underscore companion convention
      // doesn't cover -- leave it to the caller's recursive composition.
      if (pd.getComponentAttr()) return {};
      v = pd.getMemref();
      continue;
    }
    break;
  }
  if (baseUniq.empty()) return {};
  std::string const compUniq = baseUniq + "_" + member.str();
  if (!moduleHasDeclare(dg.getOperation(), compUniq)) return {};
  return extractName(compUniq);
}

mlir::Value presentBranchOfRuntimeOptional(fir::IfOp ifOp, mlir::Value result) {
  if (ifOp.getElseRegion().empty()) return {};
  unsigned idx = 0;
  bool found = false;
  for (auto r : ifOp.getResults()) {
    if (r == result) {
      found = true;
      break;
    }
    ++idx;
  }
  if (!found) return {};
  auto thenTerm = mlir::dyn_cast<fir::ResultOp>(ifOp.getThenRegion().front().getTerminator());
  auto elseTerm = mlir::dyn_cast<fir::ResultOp>(ifOp.getElseRegion().front().getTerminator());
  if (!thenTerm || !elseTerm) return {};
  if (idx >= thenTerm.getNumOperands() || idx >= elseTerm.getNumOperands()) return {};
  mlir::Value const thenV = thenTerm.getOperand(idx);
  mlir::Value const elseV = elseTerm.getOperand(idx);
  auto isAbsent = [](mlir::Value v) { return mlir::isa_and_nonnull<fir::AbsentOp>(v.getDefiningOp()); };
  if (isAbsent(elseV) && !isAbsent(thenV)) return thenV;
  if (isAbsent(thenV) && !isAbsent(elseV)) return elseV;
  return {};
}

std::string traceToDecl(mlir::Value val, int max) {
  for (int i = 0; i < max && val; ++i) {
    auto* d = val.getDefiningOp();
    if (!d) break;
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      // Walk through inlined assumed-shape aliases to the outer caller declare so downstream SDFG emission references
      // the real storage by its caller-side name.
      if (auto outer = asAssumedShapeAlias(dc)) {
        val = outer.getResult(0);
        continue;
      }
      // Gate #11: an inlined-call dummy bound to a struct-MEMBER actual (memref leads to a component designate,
      // asAssumedShapeAlias can't fold it) resolves through to that designate's caller-side flat name instead of the
      // dummy's unsourced name. Gate #11a (twin): an inlined AoR-section dummy's box_addr/copy_in memref peels to a
      // component-less section, so hop through the section to its base for the same reason.
      if (dc.getDummyScope())
        if (auto sec = asInlinedSectionOverComponent(dc)) {
          val = sec.getMemref();
          continue;
        }
      if (dc.getDummyScope() && leadsToComponentDesignate(dc.getMemref())) {
        val = dc.getMemref();
        continue;
      }
      // Runtime-present OPTIONAL forwarded a POINTER/ALLOCATABLE actual: the dummy's memref is a
      // fir.if(pointer-associated){present}else{absent} select.  Follow the present branch so accesses AND
      // extent queries (box_dims -> <name>_d<i>) resolve to the source's flat name -- else a section_alias'd
      // dummy's box_dims name onto its own (descriptor-less) name and the extent symbol is undefined.
      if (dc.getDummyScope())
        if (auto ifOp = mlir::dyn_cast_or_null<fir::IfOp>(dc.getMemref().getDefiningOp()))
          if (mlir::Value const pres = presentBranchOfRuntimeOptional(ifOp, dc.getMemref())) {
            val = pres;
            continue;
          }
      // Local whole-object POINTER rebound to a struct-dummy component: follow the rebind to the source chain so the
      // access renders as the caller-side flat name instead of the local pointer's own name.
      if (mlir::Value const src = traceLocalPointerRebindSource(dc)) {
        val = src;
        continue;
      }
      return allocAliasFor(extractName(dc.getUniqName().str()));
    }
    if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d)) return allocAliasFor(extractName(dc.getUniqName().str()));
    // Storage-transparent box reinterprets go through the shared peel (peelBoxReinterpret), which runs before the
    // designate/select-clamp cases below and returns other values unchanged.
    if (mlir::Value const peeled = peelBoxReinterpret(val); peeled != val) {
      val = peeled;
      continue;
    }
    // Section/element designates (a(lo:hi), a(i)) walk through to the underlying memref (e.g. hlfir.any over %levmask
    // resolves to levmask). Struct field designates (vcut % a) carry a componentAttr instead -- walking through would
    // land on the bare struct base, not the flattened name (vcut_a) the SDFG arglist uses, so build
    // <parent>_<component> from the component attribute and the recursively-traced parent name.
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      if (auto comp = dg.getComponentAttr()) {
        // Prefer the flattened companion declare's own registered name when it exists (see flatCompanionName); falls
        // through to composition below for non-flattened members.
        if (std::string cn = flatCompanionName(dg, comp.getValue()); !cn.empty()) return cn;
        auto parent = traceToDecl(dg.getMemref(), max - i);
        if (!parent.empty()) {
          // Flat-name is parent_member by default. Flang's POINTER/ALLOCATABLE companion alloca uses a
          // double-underscore form (parent__member, e.g. dfftt__nl) that never collides since Fortran member names
          // can't start with '_'; prefer that name when a matching declare exists, else fall back to the
          // single-underscore form.
          std::string singleU = parent + "_" + comp.getValue().str();
          bool wantPtr = false;
          if (auto attrs = dg.getFortranAttrs()) {
            auto fa = *attrs;
            wantPtr = bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::pointer) ||
                      bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::allocatable);
          }
          if (wantPtr) {
            // Search the enclosing func.func for a declare whose uniq_name's E-scope short tail equals parent__member;
            // found -> use its name.
            std::string doubleU = parent + "__" + comp.getValue().str();
            auto* func = dg->getParentOfType<mlir::func::FuncOp>().getOperation();
            bool found = false;
            if (func) {
              mlir::dyn_cast<mlir::func::FuncOp>(func).walk([&](hlfir::DeclareOp candidate) {
                if (found) return;
                auto un = candidate.getUniqName().str();
                auto eP = un.rfind('E');
                if (eP == std::string::npos) return;
                std::string const tail = un.substr(eP + 1);
                if (tail == doubleU) found = true;
              });
            }
            if (found) return doubleU;
          }
          return singleU;
        }
      }
      val = dg.getMemref();
      continue;
    }
    if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      // Follow the select ONLY for the extent CLAMP idiom (select(cmp(x,0), x, 0), one branch a constant 0); a genuine
      // MIN/MAX select is NOT an alias to one branch (following it would drop the other operand and any subscript) and
      // is left for the min/max idiom in buildIndexExpr/buildExpr.
      auto trueC = traceConstInt(s.getTrueValue());
      auto falseC = traceConstInt(s.getFalseValue());
      bool const trueZero = trueC && *trueC == 0;
      bool const falseZero = falseC && *falseC == 0;
      if (trueZero != falseZero) {
        val = trueZero ? s.getFalseValue() : s.getTrueValue();
        continue;
      }
      break;
    }
    break;
  }
  return "";
}

std::optional<int64_t> traceConstInt(mlir::Value v) {
  for (int i = 0; i < limits::kTraceConstIntMax; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(d))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) return ia.getInt();
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    // Flang wraps each static extent in a select(extent>0, extent, 0) clamp; follow the true branch, but only when the
    // false value is the constant 0 -- else it might be a genuine Fortran MAX/MIN, not a clamp.
    if (auto s = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      auto* fdef = s.getFalseValue().getDefiningOp();
      bool false_is_zero = false;
      if (fdef) {
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(fdef))
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) false_is_zero = (ia.getInt() == 0);
      }
      if (false_is_zero) {
        v = s.getTrueValue();
        continue;
      }
    }
    break;
  }
  return std::nullopt;
}

std::string posSymbolName(const std::string& array, const std::vector<int64_t>& one_based_idxs) {
  // Must stay in lockstep with internPosSymbol (ast/expressions.cpp) -- this mints the name, that mints the matching
  // symbol_init. Each 1-based index appends _<i>: shp(1,2,1) -> __sym_shp_1_2_1.
  std::string s = "__sym_" + array;
  for (auto i : one_based_idxs) s += "_" + std::to_string(i);
  return s;
}

std::optional<std::pair<std::string, std::vector<int64_t>>> constIndexedElementLoad(mlir::Value v) {
  if (!v) return std::nullopt;
  for (int i = 0; i < limits::kConvertChainDepth && v; ++i) {
    auto* d = v.getDefiningOp();
    if (auto cv = mlir::dyn_cast_or_null<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    break;
  }
  auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(v.getDefiningOp());
  if (!ld) return std::nullopt;
  auto dg = mlir::dyn_cast_or_null<hlfir::DesignateOp>(ld.getMemref().getDefiningOp());
  if (!dg) return std::nullopt;
  auto idxs = dg.getIndices();
  if (idxs.empty()) return std::nullopt;
  auto triplets = dg.getIsTriplet();
  // Every dimension must be a single constant scalar index (no section/triplet) for the element to fold to one position
  // symbol.
  std::vector<int64_t> consts;
  for (unsigned d = 0; d < idxs.size(); ++d) {
    if (d < triplets.size() && triplets[d]) return std::nullopt;
    auto c = traceConstInt(idxs[d]);
    if (!c) return std::nullopt;
    consts.push_back(*c);
  }
  auto arr = traceToDecl(dg.getMemref());
  if (arr.empty()) return std::nullopt;
  return std::make_pair(arr, std::move(consts));
}

static void forEachConstIndexedElementImpl(
    mlir::Value v, const std::function<void(const std::string&, const std::vector<int64_t>&)>& fn, int depth,
    llvm::SmallPtrSet<mlir::Operation*, 32>* visited = nullptr) {
  if (depth > limits::kTraceToDeclMax || !v) return;
  if (auto e = constIndexedElementLoad(v)) {
    fn(e->first, e->second);
    return;
  }
  auto* def = v.getDefiningOp();
  if (!def) return;
  // Branches into every operand: on a shared-subexpression DAG the depth cap alone isn't enough (a diamond re-explores
  // exponentially), so mark each op visited once; the callback is idempotent so this is behaviour-preserving.
  llvm::SmallPtrSet<mlir::Operation*, 32> seen;
  if (!visited) visited = &seen;
  if (!visited->insert(def).second) return;
  // Recurse through the same wrapper/arithmetic/max-min/select op set traceExtentExpr renders, so every element leaf is
  // reached.
  if (mlir::isa<fir::ConvertOp, mlir::arith::SelectOp, mlir::arith::CmpIOp, mlir::arith::AddIOp, mlir::arith::SubIOp,
                mlir::arith::MulIOp, mlir::arith::DivSIOp, mlir::arith::DivUIOp, mlir::arith::MaxSIOp,
                mlir::arith::MaxUIOp, mlir::arith::MinSIOp, mlir::arith::MinUIOp>(def)) {
    for (auto op : def->getOperands()) forEachConstIndexedElementImpl(op, fn, depth + 1, visited);
  }
}

void forEachConstIndexedElement(mlir::Value v,
                                const std::function<void(const std::string&, const std::vector<int64_t>&)>& fn) {
  forEachConstIndexedElementImpl(v, fn, 0);
}

static std::string traceExtentExprMemo(mlir::Value v, llvm::DenseMap<mlir::Operation*, std::string>& memo) {
  if (!v) return "";
  auto* def = v.getDefiningOp();
  if (!def) return "";

  // Extent expressions are DAGs; without memoizing per defining op, the recursive render re-walks shared subtrees --
  // exponential work on a real kernel.
  if (auto it = memo.find(def); it != memo.end()) return it->second;
  std::string result = [&]() -> std::string {
    // Transparent peels.
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) return traceExtentExprMemo(cv.getValue(), memo);

    // fir.box_dims extent result renders as <name>_d<dim>; a transient sized from another array's runtime extent (e.g.
    // LiftAosPointerRecords' gather temp) must reuse the source's extent symbol, or it falls through to "?", defaults
    // to 1, and under-allocates (heap overflow). fir.box_dims yields (lowerBound, extent, byteStride); only result #1
    // is an extent.
    if (auto bd = mlir::dyn_cast<fir::BoxDimsOp>(def)) {
      if (v == bd.getResult(1)) {
        if (auto dim = traceConstInt(bd.getDim())) {
          auto base = traceToDecl(bd.getVal());
          if (!base.empty()) return base + "_d" + std::to_string(*dim);
        }
      }
      return "";
    }

    // Constant integer literal.
    if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue())) return std::to_string(ia.getInt());

    // Load of a Fortran scalar renders as its short name; a load of a constant-indexed array element (dims(1)) becomes
    // its position symbol instead, since promoting the whole array would collide it with its own data descriptor.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
      if (auto e = constIndexedElementLoad(v)) return posSymbolName(e->first, e->second);
      auto mem = ld.getMemref();
      auto* md = mem.getDefiningOp();
      if (!md) return "";
      if (mlir::isa<hlfir::DeclareOp>(md) || mlir::isa<fir::DeclareOp>(md)) {
        return traceToDecl(mem);
      }
      return "";
    }

    // arith.select over arith.cmpi is both Flang's non-negativity clamp (false arm = constant 0 -> drop wrap, return
    // the extent) and genuine Fortran MAX/MIN (both arms are operands -> render max(a,b)/min(a,b)); cmp operands must
    // match the select arms or it's an unmodeled conditional.
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
      auto* cdef = sel.getCondition().getDefiningOp();
      auto cmp = cdef ? mlir::dyn_cast<mlir::arith::CmpIOp>(cdef) : nullptr;
      if (!cmp || cmp.getLhs() != sel.getTrueValue() || cmp.getRhs() != sel.getFalseValue()) return "";
      using P = mlir::arith::CmpIPredicate;
      auto pred = cmp.getPredicate();
      bool falseIsZero = false;
      if (auto* fdef = sel.getFalseValue().getDefiningOp())
        if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(fdef))
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) falseIsZero = (ia.getInt() == 0);
      if (falseIsZero && (pred == P::sgt || pred == P::sge))
        return traceExtentExprMemo(sel.getTrueValue(), memo);  // non-neg clamp
      auto a = traceExtentExprMemo(sel.getTrueValue(), memo);
      auto b = traceExtentExprMemo(sel.getFalseValue(), memo);
      if (a.empty() || b.empty()) return "";
      if (pred == P::sgt || pred == P::sge || pred == P::ugt || pred == P::uge) return "max(" + a + ", " + b + ")";
      if (pred == P::slt || pred == P::sle || pred == P::ult || pred == P::ule) return "min(" + a + ", " + b + ")";
      return "";
    }

    // Binary integer arithmetic, rendered parenthesised so it composes when nested; arith.max*i/min*i are the direct
    // MAX/MIN lowering (vs the select-over-cmp form above).
    auto nm = def->getName().getStringRef();
    static const std::map<llvm::StringRef, std::string> bin = {
        {"arith.addi", " + "},   {"arith.subi", " - "},   {"arith.muli", " * "},
        {"arith.divsi", " // "}, {"arith.divui", " // "},
    };
    if (auto it = bin.find(nm); it != bin.end() && def->getNumOperands() == 2) {
      auto l = traceExtentExprMemo(def->getOperand(0), memo);
      auto r = traceExtentExprMemo(def->getOperand(1), memo);
      if (l.empty() || r.empty()) return "";
      return "(" + l + it->second + r + ")";
    }
    if ((nm == "arith.maxsi" || nm == "arith.maxui" || nm == "arith.minsi" || nm == "arith.minui") &&
        def->getNumOperands() == 2) {
      auto l = traceExtentExprMemo(def->getOperand(0), memo);
      auto r = traceExtentExprMemo(def->getOperand(1), memo);
      if (l.empty() || r.empty()) return "";
      const char* fn = (nm == "arith.maxsi" || nm == "arith.maxui") ? "max" : "min";
      return std::string(fn) + "(" + l + ", " + r + ")";
    }

    return "";
  }();
  memo[def] = result;
  return result;
}

std::string traceExtentExpr(mlir::Value v) {
  llvm::DenseMap<mlir::Operation*, std::string> memo;
  return traceExtentExprMemo(v, memo);
}

// Extent expressions form a DAG; without a visited set the operand recursion re-explores shared subtrees (exponential).
// visited marks each defining op once so the walk is linear; the public entry seeds a fresh set.
static void collectExtentExprScalarsRec(mlir::Value v, std::set<std::string>& out,
                                        llvm::SmallPtrSet<mlir::Operation*, 32>& visited) {
  if (!v) return;
  auto* def = v.getDefiningOp();
  if (!def || !visited.insert(def).second) return;
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
    collectExtentExprScalarsRec(cv.getValue(), out, visited);
    return;
  }
  if (mlir::isa<mlir::arith::ConstantOp>(def)) return;
  if (auto ld = mlir::dyn_cast<fir::LoadOp>(def)) {
    auto mem = ld.getMemref();
    if (auto* md = mem.getDefiningOp())
      if (mlir::isa<hlfir::DeclareOp>(md) || mlir::isa<fir::DeclareOp>(md)) {
        auto n = traceToDecl(mem);
        if (!n.empty()) out.insert(n);
      }
    return;
  }
  if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(def)) {
    // Walk both branches; cmp condition leaves are already covered by the operands themselves.
    collectExtentExprScalarsRec(sel.getTrueValue(), out, visited);
    collectExtentExprScalarsRec(sel.getFalseValue(), out, visited);
    return;
  }
  auto nm = def->getName().getStringRef();
  if ((nm == "arith.addi" || nm == "arith.subi" || nm == "arith.muli" || nm == "arith.divsi" || nm == "arith.divui" ||
       nm == "arith.cmpi") &&
      def->getNumOperands() == 2) {
    collectExtentExprScalarsRec(def->getOperand(0), out, visited);
    collectExtentExprScalarsRec(def->getOperand(1), out, visited);
    return;
  }
}

void collectExtentExprScalars(mlir::Value v, std::set<std::string>& out) {
  llvm::SmallPtrSet<mlir::Operation*, 32> visited;
  collectExtentExprScalarsRec(v, out, visited);
}

hlfir::DeclareOp asAssumedShapeAlias(hlfir::DeclareOp decl) {
  // Detects Flang's post-hlfir-inline-all callee dummy_scope declare aliasing the caller's outer declare (memref traces
  // to another hlfir.declare, possibly through fir.convert/rebox); either assumed- or fixed-shape, storage is shared.
  // Exception: rank-promotion/reduction (Fortran sequence association can pass a 1D array to a multi-D dummy) -- refuse
  // the alias collapse when ranks differ, or a 3D designate would resolve to a 1D source name and emit an unresolved
  // memlet subset. rankOfDeclResult below strips fir.ref/box/heap/ptr layers to read the declare's array rank (0 for
  // scalars/non-arrays).
  auto rankOfDeclResult = [](mlir::Value v) -> int {
    mlir::Type t = v.getType();
    for (int i = 0; i < 8; ++i) {
      if (auto refTy = mlir::dyn_cast<fir::ReferenceType>(t)) {
        t = refTy.getEleTy();
        continue;
      }
      if (auto bx = mlir::dyn_cast<fir::BoxType>(t)) {
        t = bx.getEleTy();
        continue;
      }
      if (auto hp = mlir::dyn_cast<fir::HeapType>(t)) {
        t = hp.getEleTy();
        continue;
      }
      if (auto pt = mlir::dyn_cast<fir::PointerType>(t)) {
        t = pt.getEleTy();
        continue;
      }
      break;
    }
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(t)) return seq.getDimension();
    return 0;
  };
  int const innerRank = rankOfDeclResult(decl.getResult(0));
  auto mr = decl.getMemref();
  for (int i = 0; i < limits::kAliasMemrefWalkDepth && mr; ++i) {
    auto* d = mr.getDefiningOp();
    if (!d) break;
    if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      int const outerRank = rankOfDeclResult(outer.getResult(0));
      if (innerRank > 0 && outerRank > 0 && innerRank != outerRank) return {};
      return outer;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      mr = conv.getValue();
      continue;
    }
    // Internal-subprogram inlining wraps the outer fixed-shape array in a fir.embox so the inlined assumed-shape callee
    // sees a fir.box; peel through to the underlying declare.
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      mr = eb.getMemref();
      continue;
    }
    // Inlined-callee aliases on CLASS-allocatable/box-typed dummies: memref comes from a fir.load of the caller's
    // box-slot declare, possibly preceded by fir.rebox (CLASS<heap<T>> -> CLASS<T>); walking through both catches
    // monomorphic CLASS dummies as aliases of their caller-side allocatable.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      mr = ld.getMemref();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      mr = rb.getBox();
      continue;
    }
    // Explicit-shape array dummy bound to a POINTER/VIEW actual: FoldCopyInOut rewrites the inlined declare to
    // hlfir.declare(convert(box_addr(load(p_box)))), dropping the no-op copy_in/copy_out; peel box_addr so the walk
    // reaches the source declare as a same-rank alias.
    if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
      mr = ba.getVal();
      continue;
    }
    break;
  }
  return {};
}

ShapeOperandInfo classifyShapeOperand(mlir::Value shape) {
  ShapeOperandInfo si;
  if (!shape) return si;
  auto* def = shape.getDefiningOp();
  if (auto sh = mlir::dyn_cast_or_null<fir::ShapeOp>(def)) {
    si.kind = ShapeOperandInfo::Shape;
    for (auto e : sh.getExtents()) si.extents.push_back(e);
    si.rank = si.extents.size();
  } else if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(def)) {
    si.kind = ShapeOperandInfo::ShapeShift;
    auto ops = ss->getOperands();
    for (unsigned i = 0; i + 1 < ops.size(); i += 2) {
      si.lbs.push_back(ops[i]);
      si.extents.push_back(ops[i + 1]);
    }
    si.rank = si.lbs.size();
  } else if (auto sf = mlir::dyn_cast_or_null<fir::ShiftOp>(def)) {
    si.kind = ShapeOperandInfo::Shift;
    for (auto lb : sf->getOperands()) si.lbs.push_back(lb);
    si.rank = si.lbs.size();
  }
  return si;
}

std::vector<std::optional<int64_t>> declareLowerBounds(hlfir::DeclareOp decl) {
  std::vector<std::optional<int64_t>> lbs;
  auto si = classifyShapeOperand(decl.getShape());
  if (si.kind == ShapeOperandInfo::Shape) {
    // Plain fir.shape: every dim defaults to lbound=1.
    lbs.assign(si.rank, std::optional<int64_t>(1));
  } else if (si.kind == ShapeOperandInfo::ShapeShift || si.kind == ShapeOperandInfo::Shift) {
    // fir.shape_shift (lb+extent) and fir.shift (assumed-shape dummy with explicit lb, a(0:)) both carry per-dim lower
    // bounds in si.lbs; trace each to its constant -- dropping the fir.shift bound defeats the assumed-shape alias
    // offset rebase and causes an off-by-(lb-1) read.
    for (auto lb : si.lbs) lbs.push_back(traceConstInt(lb));
  }
  // no shape operand (pure box, runtime bounds): leave empty (caller default).
  return lbs;
}

llvm::SmallVector<mlir::Value, 4> extractExtents(mlir::Value shape) {
  auto ext = classifyShapeOperand(shape).extents;
  return {ext.begin(), ext.end()};
}

fir::RecordType pointerToRecordMember(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return {};
  mlir::Type inner;
  if (auto h = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
    inner = h.getEleTy();
  else if (auto p = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
    inner = p.getEleTy();
  else
    return {};
  return mlir::dyn_cast<fir::RecordType>(inner);
}

fir::RecordType allocOrPtrArrayOfRecordsMember(mlir::Type t) {
  auto box = mlir::dyn_cast<fir::BoxType>(t);
  if (!box) return {};
  mlir::Type inner;
  if (auto h = mlir::dyn_cast<fir::HeapType>(box.getEleTy()))
    inner = h.getEleTy();
  else if (auto p = mlir::dyn_cast<fir::PointerType>(box.getEleTy()))
    inner = p.getEleTy();
  else
    return {};
  auto seq = mlir::dyn_cast<fir::SequenceType>(inner);
  if (!seq) return {};
  return mlir::dyn_cast<fir::RecordType>(seq.getEleTy());
}

}  // namespace hlfir_bridge
