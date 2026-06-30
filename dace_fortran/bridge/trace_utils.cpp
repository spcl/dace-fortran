// ============================================================================
// trace_utils.cpp  --  Shared SSA tracing utilities
// ============================================================================

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

// Disambiguation overrides for Fortran short-name collisions across
// inlined scopes.  When ``hlfir-inline-all`` splices a callee's body
// into the caller, both the caller's argument declare
// (``_QFmainEinp``) and the callee's dummy declare
// (``_QFinner_loopsEinp``) end up in one function with the same
// trailing short name (``inp``).  Without disambiguation,
// ``builder.arrays`` keys collide and view-alias linking edges
// self-loop.  ``extract_vars`` populates this map with
// ``mangled -> unique_short_name`` for the colliding entries; every
// subsequent ``extractName`` call resolves to the unique form.
static thread_local std::unordered_map<std::string, std::string> kManglingOverride;

void setManglingOverride(const std::string& mangled, const std::string& shortName) {
  kManglingOverride[mangled] = shortName;
}

// Per-thread entry F-scope.  Set once per build by ``setEntryScope``,
// consulted by ``extractName`` to scope-qualify every NON-entry-scope
// declare's short name on demand.  Empty until set; in that case
// ``extractName`` skips the scope-qualification (back-compat with
// callers that haven't migrated to the new flow).
static thread_local std::string kEntryScope;
static thread_local std::set<std::string> kShortNameCollisions;

// Cache of every ``hlfir.declare`` uniq_name in the module, built lazily on
// first ``flatCompanionName`` query and invalidated per build (in
// ``clearManglingOverrides``).  Lets a component-designate name resolve to the
// flattened companion's OWN declare name only when that companion actually
// exists, without an O(declares) walk per access.
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
  // Fortran mangled-name shape: ``_QM<mod>F<func>E<name>`` (with
  // optional nested ``F`` segments for procedure-internal
  // procedures).  Take the F immediately before the last E.
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

  // Scope-qualify NON-entry-scope declares' short names on demand.
  // Replaces the upfront inlined-callee disambiguator pass at
  // ``extract_vars.cpp:1187`` (which only renamed ``fir.alloca``-
  // backed declares -- missing the inlined PURE FUNCTION scalar
  // dummies that arrive backed by the caller's loaded value, which
  // surfaced as graupel's ``_in_qc_0`` unresolved-free-symbol bug).
  //
  // Rule: every declare in a non-entry F-scope gets ``<scope>_<short>``;
  // entry-scope declares (the kernel's own dummies + locals) keep
  // their bare short name so the SDFG signature matches the
  // user-facing Fortran procedure interface.  Module globals
  // (no F segment) keep their bare name -- they live in the
  // entry's symbol-table sense.  Empty ``kEntryScope`` (set yet?
  // legacy caller?) also keeps the bare name.
  //
  // Already-prefixed names (the old disambiguator's ``setAttr`` rename
  // left some IRs with ``scope_short`` as the trailing E segment;
  // the new logic would otherwise produce ``scope_scope_short``)
  // are detected by checking whether ``name`` already starts with
  // ``<scope>_``.
  if (!kEntryScope.empty()) {
    std::string scope = getFScope(m);
    if (!scope.empty() && scope != kEntryScope) {
      // Qualify ONLY when the short name actually collides across
      // scopes.  ``kShortNameCollisions`` is populated by
      // ``extractVariables`` from a pre-walk of every declare.
      // Without this guard, qualifying every non-entry-scope
      // declare creates EXTRA signature variables for unused
      // inlined-callee dummies -- e.g. ``test_fortran_frontend_present``
      // has ``tf2``'s OPTIONAL ``a`` folded by ``is_present`` so it
      // never reaches a tasklet, but ``tf2_a`` still landed on the
      // SDFG signature and broke the caller's ``a=5`` binding.
      bool collides = kShortNameCollisions.count(name) > 0;
      if (collides) {
        std::string prefix = scope + "_";
        if (name.compare(0, prefix.size(), prefix) != 0) name = prefix + name;
      }
    }
  }

  // Sanitize dots  --  flang emits compiler-generated globals like
  // ``_QQro.4xi4.0`` (read-only constant pool for array literals)
  // whose names contain ``.``.  DaCe's ``NestedDict`` reserves
  // ``.`` as a nested-key separator and rejects dotted keys
  // outright.  Fortran identifiers can't contain ``.``, so
  // replacing every ``.`` with ``_`` is collision-free w.r.t.
  // user names.  Done at the boundary (extractName is the
  // canonical "MLIR mangled -> Python-side name" helper) so the
  // raw mangled names in the IR stay intact.
  std::replace(name.begin(), name.end(), '.', '_');
  return name;
}

// Allocatable re-allocation alias map.  Keyed by the raw Fortran name
// (what the declare chain alone would resolve to).  Updated as the
// bridge's IR walker passes ``fir.allocmem``-bound ``fir.store`` ops
// (see extract_ast.cpp); read by ``traceToDecl`` so every downstream
// access lands on the currently-live SDFG transient.
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

// True iff ``mr`` (a declare's memref) leads -- through the usual
// reinterpret peels -- to an ``hlfir.designate`` that selects a struct
// COMPONENT.  This distinguishes an INLINED-call dummy bound to a caller
// struct member (``call get_index_range(patch % edges % in_domain, ...)``
// after ``hlfir-inline-all`` splices the callee body in) from an entry
// dummy (memref is a block argument) or a local (memref is an alloca) --
// neither of those leads to a designate.  ``traceToDecl`` uses it to
// resolve such an alias to the caller-side member path rather than the
// inlined dummy's own ``<dummy>_call<idx>`` name, which nothing sources.
// Single source of truth for the storage-transparent reinterpret peel shared by
// every access-path walker.  ``fir.convert`` (type erase), ``fir.load`` (deref a
// ref/box), ``fir.embox`` (wrap a ref/ptr into a box -- e.g. a polymorphic
// ``CLASS(t)`` dummy bound to a pointer member), ``fir.rebox`` (retype a box),
// ``fir.box_addr`` (data ptr out of a box), ``hlfir.copy_in`` (contiguous temp of
// a non-contiguous actual), ``fir.coordinate_of`` (sub-element address), and
// ``hlfir.as_expr`` (materialised-variable-as-expr) ALL preserve the underlying
// storage identity.  Callers peel to the first non-transparent op (designate /
// declare / alloca / block-arg / arith) and handle it themselves.  Centralising
// the set here is what stops the per-loop subsets from drifting (the ICON halo
// gather's ``fir.embox`` gap -- a class dummy bound to ``p_patch % comm_pat_c``
// whose member reads must resolve to ``p_patch_comm_pat_c_*`` rather than the
// inlined dummy's local name -- was exactly such a drift).
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
  // Peel the shared storage-transparent reinterprets, then the terminal must be
  // an ``hlfir.designate`` selecting a struct COMPONENT.  Using the shared peel
  // (vs an inline op cascade) keeps this gate in lock-step with the other
  // access-path walkers -- the ``fir.embox`` polymorphic-class-dummy case
  // resolves here because ``peelBoxReinterpret`` covers embox.
  mr = peelBoxReinterpret(mr);
  auto* d = mr ? mr.getDefiningOp() : nullptr;
  auto dg = d ? mlir::dyn_cast<hlfir::DesignateOp>(d) : nullptr;
  return dg && static_cast<bool>(dg.getComponentAttr());
}

// True iff some ``hlfir.declare`` in ``anyOp``'s module has uniq_name
// ``uniq``.  Backed by a per-build cache (see ``kModuleDeclUniqs``).
static bool moduleHasDeclare(mlir::Operation* anyOp, llvm::StringRef uniq) {
  if (!kModuleDeclUniqsBuilt) {
    if (auto mod = anyOp->getParentOfType<mlir::ModuleOp>()) {
      mod.walk([&](hlfir::DeclareOp dc) { kModuleDeclUniqs.insert(dc.getUniqName()); });
      kModuleDeclUniqsBuilt = true;
    }
  }
  return kModuleDeclUniqs.contains(uniq);
}

// A component designate ``base % member`` over a FLATTENED struct must resolve
// to the SAME name the companion was registered under -- ``extractName(<base
// uniq>_<member>)`` -- not a re-composition ``extractName(base) + "_" +
// member``.  The two diverge when the base and its companion have different
// cross-scope collision status: e.g. a runtime-local AoS whose base declare was
// erased in one inlined scope (so its short no longer collides -> bare) while
// the companion survives in two scopes (collides -> scope-qualified).  Resolve
// the companion declare's own name by construction whenever such a declare
// exists; return "" otherwise (member of a non-flattened struct -> the caller
// keeps the plain ``parent_member`` composition).
static std::string flatCompanionName(hlfir::DesignateOp dg, llvm::StringRef member) {
  // Peel the memref to the base declare's uniq_name, mirroring the transparent
  // walk ``traceToDecl`` does so we land on the same storage declare its
  // ``parent`` would (incl. assumed-shape aliases -> outer storage).
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
    // Shared storage-transparent peel -- runs before the designate case below
    // (a designate value returns unchanged, so a nested-component parent still
    // falls through to the multi-level guard).
    if (mlir::Value peeled = peelBoxReinterpret(v); peeled != v) {
      v = peeled;
      continue;
    }
    if (auto pd = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      // A nested component parent (``a % b % member``) is a multi-level path
      // the single-underscore companion convention doesn't cover -- leave it
      // to the caller's recursive composition.
      if (pd.getComponentAttr()) return {};
      v = pd.getMemref();
      continue;
    }
    break;
  }
  if (baseUniq.empty()) return {};
  std::string compUniq = baseUniq + "_" + member.str();
  if (!moduleHasDeclare(dg.getOperation(), compUniq)) return {};
  return extractName(compUniq);
}

std::string traceToDecl(mlir::Value val, int max) {
  for (int i = 0; i < max && val; ++i) {
    auto* d = val.getDefiningOp();
    if (!d) break;
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      // Walk through inlined assumed-shape aliases to the outer
      // caller declare so downstream SDFG emission references
      // the real storage by its caller-side name.
      if (auto outer = asAssumedShapeAlias(dc)) {
        val = outer.getResult(0);
        continue;
      }
      // Inlined-call dummy bound to a struct-MEMBER actual argument
      // (``get_index_range(patch % edges % in_domain, ...)``): the dummy
      // declare's memref leads to a component designate of the caller's
      // struct.  ``asAssumedShapeAlias`` can't fold it (it looks for a
      // whole outer declare, not a member designate), so walk through to
      // that designate -- the symbol then resolves to the caller-side flat
      // path (``patch_3d_p_patch_2d_edges_in_domain_start_index``), which
      // the binding's ``_struct_member_symbol_sources`` sources, instead of
      // the inlined dummy's unsourced ``subset_range_call<idx>`` name (gate
      // #11).  Gated on a dummy scope so only inlined aliases match.
      if (dc.getDummyScope() && leadsToComponentDesignate(dc.getMemref())) {
        val = dc.getMemref();
        continue;
      }
      return allocAliasFor(extractName(dc.getUniqName().str()));
    }
    if (auto dc = mlir::dyn_cast<fir::DeclareOp>(d)) return allocAliasFor(extractName(dc.getUniqName().str()));
    // Storage-transparent box reinterprets (convert / load / coord / rebox /
    // embox / box_addr / copy_in / as_expr) go through the shared peel -- the
    // single source of truth (see ``peelBoxReinterpret``).  Runs before the
    // designate / select-clamp cases below; the peel returns those values
    // unchanged so each still falls through to its dedicated handler.
    if (mlir::Value peeled = peelBoxReinterpret(val); peeled != val) {
      val = peeled;
      continue;
    }
    // Section / element designates (``a(lo:hi)``, ``a(i)``)  --  walk
    // through to the underlying memref so a reduce over an
    // ``hlfir.any %levmask(i_startblk:i_endblk, jk)`` resolves its
    // source array to ``levmask``.
    //
    // Struct field designates (``vcut % a``) are different: they
    // carry a ``componentAttr`` naming the member.  Walking through
    // would land on the struct base (``vcut``) -- which is NOT the
    // flattened name the SDFG arglist uses (the bridge's
    // hlfir-flatten-structs pass produces ``vcut_a`` for DUMMY
    // args, and the bindings layer maps a MODULE-LEVEL struct
    // global the same way).  Build the flattened name
    // ``<parent>_<component>`` from this designate's component
    // attribute and the recursively-traced parent name.  Mirrors
    // the ``_QMmFvcut_getEvcut_a`` form Flang produces for the
    // dummy-arg case AND the flat-name convention the SDFG
    // arglist uses for module-level struct globals.  QE's
    // ``vcut_get`` (called from ``g2_convolution`` over a module-
    // level ``vcut`` global) was the surfacing case -- the libcall
    // dispatcher's ``traceToDecl`` on the matmul's first operand
    // had been returning the bare struct name ``vcut`` instead of
    // ``vcut_a``, causing ``KeyError: 'vcut'`` at SDFG arglist
    // lookup.
    if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      if (auto comp = dg.getComponentAttr()) {
        // Prefer the flattened companion declare's OWN registered name when it
        // exists -- correct-by-construction vs. re-composing from the base
        // (see ``flatCompanionName``).  Falls through to the composition below
        // for non-flattened struct members (no companion declare).
        if (std::string cn = flatCompanionName(dg, comp.getValue()); !cn.empty()) return cn;
        auto parent = traceToDecl(dg.getMemref(), max - i);
        if (!parent.empty()) {
          // Flat-name construction is just string join ``parent_member``.
          // Flang's pointer-companion alloca uses a DOUBLE-underscore
          // form (``dfftt__nl``) that doesn't collide with our
          // single-underscore convention as long as Fortran member
          // names don't start with ``_`` (which they can't -- Fortran
          // identifiers must start with a letter).  In practice we
          // walk the module's hlfir.declares for a companion whose
          // uniq_name ends in ``E<parent>__<member>`` (the
          // POINTER / ALLOCATABLE struct member snapshot path Flang
          // synthesises) and prefer THAT name when found, so the
          // SDFG arglist key matches what the rest of the pipeline
          // registered.  Falls back to the single-underscore form
          // when no companion exists (the common case).
          std::string singleU = parent + "_" + comp.getValue().str();
          bool wantPtr = false;
          if (auto attrs = dg.getFortranAttrs()) {
            auto fa = *attrs;
            wantPtr = bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::pointer) ||
                      bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::allocatable);
          }
          if (wantPtr) {
            // Search the enclosing func.func / module for a declare
            // whose uniq_name's E-scope short tail equals
            // ``<parent>__<member>``.  Found -> use its name.
            std::string doubleU = parent + "__" + comp.getValue().str();
            auto* func = dg->getParentOfType<mlir::func::FuncOp>().getOperation();
            bool found = false;
            if (func) {
              mlir::dyn_cast<mlir::func::FuncOp>(func).walk([&](hlfir::DeclareOp candidate) {
                if (found) return;
                auto un = candidate.getUniqName().str();
                auto eP = un.rfind('E');
                if (eP == std::string::npos) return;
                std::string tail = un.substr(eP + 1);
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
      // Follow the select ONLY for the extent CLAMP idiom -- ``max(x, 0)`` /
      // ``select(cmp(x, 0), x, 0)`` -- where ONE branch is the constant ZERO
      // and the OTHER is the real value (Flang clamps a computed extent to be
      // non-negative this way).  A genuine ``MIN(a, b)`` / ``MAX(a, b)`` --
      // even against a non-zero constant (``MIN(x, 100)``) -- is NOT an alias
      // to one branch: following it would silently drop the other operand AND
      // any subscript (a struct-member array element ``dolic_e(je, jb)`` ->
      // bare
      // ``..._dolic_e``).  Leave those for the min/max idiom in
      // ``buildIndexExpr`` / ``buildExpr`` to render.
      auto trueC = traceConstInt(s.getTrueValue());
      auto falseC = traceConstInt(s.getFalseValue());
      bool trueZero = trueC && *trueC == 0;
      bool falseZero = falseC && *falseC == 0;
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
    // Flang wraps each static extent in a `select extent>0, extent, 0`
    // clamp; follow the true branch to reach the original value.
    // Restrict to that exact shape -- the false value must be the
    // constant 0 -- so we don't accidentally follow Fortran ``MAX``
    // / ``MIN`` (also lowered as ``arith.select`` over a cmp) and
    // collapse a non-constant bound to its first operand.
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
  // Keep in lockstep with ``internPosSymbol`` (ast/expressions.cpp): the
  // descriptor-shape side mints the name here, the AST builder mints the
  // matching ``symbol_init`` there, and they must agree.  Each 1-based
  // index appends ``_<i>``, so ``shp(1,2,1)`` -> ``__sym_shp_1_2_1`` and
  // the 1-D ``dims(1)`` -> ``__sym_dims_1`` (unchanged).
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
  // Every dimension must be a single constant scalar index (no section /
  // triplet) for the element to fold to one position symbol.
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
  // Branches into every operand: on a shared-subexpression DAG the depth cap
  // alone is not enough (a diamond re-explores exponentially), so mark each op
  // once.  The callback (``internPosSymbol``) is idempotent, so visiting a
  // shared element leaf once instead of twice is behaviour-preserving.
  llvm::SmallPtrSet<mlir::Operation*, 32> seen;
  if (!visited) visited = &seen;
  if (!visited->insert(def).second) return;
  // Recurse through the same wrapper / arithmetic / max-min / select op
  // set ``traceExtentExpr`` renders, so every element leaf is reached.
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

  // Extent expressions are DAGs (a shared grid-parameter sub-expression feeds
  // many array bounds, and recurs within one extent).  Without a cache this
  // recursive render re-walks shared subtrees and re-builds their strings --
  // exponential work on a real kernel.  Memoize the rendered string per
  // defining op so each subexpression is built exactly once.
  if (auto it = memo.find(def); it != memo.end()) return it->second;
  std::string result = [&]() -> std::string {
    // Transparent peels.
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) return traceExtentExprMemo(cv.getValue(), memo);

    // ``fir.box_dims`` extent result of an array descriptor: render as that
    // array's synthetic extent symbol ``<name>_d<dim>``.  A transient sized
    // from another array's *runtime* extent -- e.g. the AoS-of-pointer-records
    // gather temp (``LiftAosPointerRecords``), whose inner shape is recovered
    // via ``fir.box_dims`` on a rebind target's assumed-shape box -- must reuse
    // the source array's extent symbol.  Otherwise the extent falls through to
    // ``"?"`` and the caller mints a fresh ``<temp>_d<i>`` that no passed array
    // backs, so the call-time auto-fill defaults it to ``1`` and the transient
    // is under-allocated (heap overflow).  ``fir.box_dims`` yields
    // ``(lowerBound, extent, byteStride)``; only result #1 is an extent.
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

    // Load of a Fortran scalar -- render as its short name.  A load of a
    // constant-indexed array element (``dims(1)``) becomes its position
    // symbol so the shape stays symbolic; promoting the whole array would
    // collide it with its own data descriptor.
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

    // ``arith.select`` over an ``arith.cmpi`` is BOTH Flang's
    // non-negativity clamp on an extent AND a genuine Fortran
    // ``MAX``/``MIN`` -- they must be told apart:
    //
    //   * Clamp ``max(ext, 0)``: ``select(ext sgt 0, ext, 0)`` -- the
    //     false arm is the constant ``0``.  Array extents are
    //     non-negative by construction, so this is dead defensive code;
    //     drop the wrap and return the underlying extent (keeps shapes
    //     readable -- ``klon`` not ``max(klon, 0)`` -- and lets sympy
    //     fold).
    //   * Genuine ``MAX(a, b)`` / ``MIN(a, b)``:
    //     ``select(a sgt b, a, b)`` / ``select(a slt b, a, b)`` -- the
    //     two arms are the operands.  Render ``max(a, b)`` / ``min(a, b)``
    //     so a real two-operand bound (``allocate(x(max(n, 1)))``)
    //     survives instead of being dropped.
    //
    // The cmp operands must match the select arms; otherwise it is some
    // other conditional we don't model.
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

    // Binary integer arithmetic.  Render parenthesised so the result
    // composes cleanly when nested.  ``arith.max*i`` / ``arith.min*i`` are
    // the direct MAX/MIN lowering (vs the select-over-cmp form above).
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

// Extent expressions form a DAG: a shared sub-expression (a grid-parameter
// product reused across array bounds) feeds many operands and recurs within
// one extent.  Without a visited set the operand recursion re-explores shared
// subtrees, which is exponential on a real kernel.  ``visited`` marks each
// defining op once so the walk is linear in the DAG size; the public entry
// seeds a fresh set and threads it through.
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
    // Walk both branches; cmp condition leaves are already
    // covered by the operands themselves.
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
  // Signature: memref produced by another ``hlfir.declare`` (possibly
  // behind ``fir.convert`` rebox ops).  This is precisely what Flang
  // emits for the callee's dummy_scope declare after
  // ``hlfir-inline-all`` splices the callee's body into the caller  --
  // the callee declare aliases the caller's outer declare for
  // both assumed-shape (no shape operand on the inner declare) and
  // fixed-shape (the inner declare carries its own copy of the
  // callee-side shape) callees, the only difference being whether
  // the inner declare reissues a shape.  Either way the storage is
  // shared and downstream tracing should walk to the outer declare.
  //
  // Exception: rank-promotion / rank-reduction.  Fortran sequence
  // association lets a caller pass a 1D array to a multi-D dummy
  // (or vice versa) -- ssor's ``tv(N)`` flowing into buts's ``tv(5,
  // M, K)`` is the canonical case.  The storage is shared but the
  // dummy's accesses are at a DIFFERENT RANK than the source's
  // descriptor; if we treat the dummy as a transparent alias then
  // ``traceToDecl`` resolves a 3D ``hlfir.designate`` to the
  // source's 1D name and the bridge emits a 3D memlet subset
  // against a 1D array (unresolved per-dim offset symbols).  Refuse
  // the alias collapse in that case so extract_vars mints a real
  // VarInfo for the dummy and the view-alias path can wire it as
  // a rank-promoted view over the source's flat storage.
  // Strip ``fir.ref`` / ``fir.box`` / ``fir.heap`` / ``fir.ptr`` layers
  // off a declare result type until we hit the underlying ``fir.array``
  // (``fir.SequenceType``); read its rank.  Scalars and non-array
  // types report rank 0.
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
  int innerRank = rankOfDeclResult(decl.getResult(0));
  auto mr = decl.getMemref();
  for (int i = 0; i < limits::kAliasMemrefWalkDepth && mr; ++i) {
    auto* d = mr.getDefiningOp();
    if (!d) break;
    if (auto outer = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      int outerRank = rankOfDeclResult(outer.getResult(0));
      if (innerRank > 0 && outerRank > 0 && innerRank != outerRank) return {};
      return outer;
    }
    if (auto conv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      mr = conv.getValue();
      continue;
    }
    // Internal-subprogram inlining wraps the outer fixed-shape array
    // in a ``fir.embox`` so the inlined assumed-shape callee sees a
    // ``fir.box``.  Peel through to the underlying declare.
    if (auto eb = mlir::dyn_cast<fir::EmboxOp>(d)) {
      mr = eb.getMemref();
      continue;
    }
    // Inlined-callee aliases on CLASS-allocatable / box-typed
    // dummies: the alias's memref comes from a ``fir.load`` of the
    // caller's box-slot declare, possibly preceded by ``fir.rebox``
    // peels (CLASS<heap<T>> -> CLASS<T>).  Walking through both is
    // what catches monomorphic CLASS subroutine dummies as
    // aliases of their caller-side allocatable.
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      mr = ld.getMemref();
      continue;
    }
    if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) {
      mr = rb.getBox();
      continue;
    }
    // Explicit-shape array dummy bound to a POINTER/VIEW actual
    // (``call scale(p, n)``, ``p`` a flatten view, ``real :: v(n)``):
    // ``FoldCopyInOut`` rewrites the inlined ``v`` to
    // ``hlfir.declare(convert(box_addr(load(p_box))))`` (it drops the
    // copy_in/copy_out, which are element-wise no-ops under no-aliasing).
    // Peel ``box_addr`` so the walk reaches ``p``'s declare; ``v`` is then
    // a same-rank alias of ``p`` and ``traceToDecl`` resolves ``v(i)`` to
    // ``p(i)`` (folded to the parent by p's bounds-remap view).
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
  } else if (si.kind == ShapeOperandInfo::ShapeShift) {
    for (auto lb : si.lbs) lbs.push_back(traceConstInt(lb));
  }
  // fir.shift / no shape: leave empty (caller default), as before.
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
