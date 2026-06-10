// ============================================================================
// extract_vars.cpp  --  Collect and classify every hlfir.declare.
// ============================================================================

#include "bridge/extract_vars.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <utility>

#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace hlfir_bridge {

// ---------------------------------------------------------------------------
// Shape and lower-bound resolution for one declare.
// ---------------------------------------------------------------------------

// ``ShapeOperandInfo`` / ``classifyShapeOperand`` live in trace_utils
// now (shared with declareLowerBounds / extractExtents).

/// Strip the array-descriptor wrappers Flang stacks over an element
/// type -- ``fir.box`` / ``fir.ref`` / ``fir.heap`` / ``fir.ptr`` --
/// repeating until none remain (or ``maxDepth`` is hit).  Used wherever
/// the bridge needs the ``fir.array`` / element type underneath an
/// allocatable / pointer / boxed declare.
static mlir::Type peelTypeLayers(mlir::Type t,
                                 int maxDepth = limits::kTypeWrapperPeelDepth) {
  for (int i = 0; i < maxDepth; ++i) {
    if (auto b = mlir::dyn_cast<fir::BoxType>(t)) {
      t = b.getEleTy();
      continue;
    }
    if (auto r = mlir::dyn_cast<fir::ReferenceType>(t)) {
      t = r.getEleTy();
      continue;
    }
    if (auto h = mlir::dyn_cast<fir::HeapType>(t)) {
      t = h.getEleTy();
      continue;
    }
    if (auto p = mlir::dyn_cast<fir::PointerType>(t)) {
      t = p.getEleTy();
      continue;
    }
    break;
  }
  return t;
}

/// Mangled symbol name for an array-element value used as a symbol:
/// ``__sym_<array>_<index>`` (index sanitised to a valid identifier).  Shares
/// the ``__sym_`` prefix with the constant-indexed ``posSymbolName`` so both
/// kinds of array-value symbol read consistently.
static std::string valueSymbolName(const std::string& array,
                                   const std::string& index) {
  std::string s = "__sym_" + array + "_";
  for (char c : index)
    s += (std::isalnum((unsigned char)c) || c == '_') ? c : '_';
  return s;
}

/// Recognise an extent that is a *runtime*-indexed array element
/// ``arr(idx)`` (the ICON ``z_raylfac(nrdmax(jg))`` shape pattern): a load of a
/// single-index ``hlfir.designate`` whose index is NOT a compile-time constant
/// (the constant case is handled by ``traceExtentExpr`` -> ``__sym_arr_N``).
/// Returns ``(array, index_expr)`` -- the array read from and the 1-based
/// Fortran index expression -- or ``nullopt`` when the extent is not such an
/// element (v1 handles a single index only).
static std::optional<std::pair<std::string, std::string>>
arrayElementExtent(mlir::Value ext) {
  // Peel the kind-coercion converts and Flang's non-negativity clamp
  // (``max(ext, 0)`` = ``select(cmpi sgt/sge X, 0, X, 0)``) that wrap an
  // automatic-array extent, the same way ``traceExtentExpr`` does, to reach
  // the underlying element load.
  mlir::Value v = ext;
  for (int i = 0; i < limits::kConvertChainDepth && v; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) break;
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    if (auto sel = mlir::dyn_cast<mlir::arith::SelectOp>(d)) {
      auto* cdef = sel.getCondition().getDefiningOp();
      auto cmp = cdef ? mlir::dyn_cast<mlir::arith::CmpIOp>(cdef) : nullptr;
      using P = mlir::arith::CmpIPredicate;
      if (cmp && cmp.getLhs() == sel.getTrueValue() &&
          cmp.getRhs() == sel.getFalseValue() &&
          (cmp.getPredicate() == P::sgt || cmp.getPredicate() == P::sge)) {
        if (auto c = traceConstInt(sel.getFalseValue()); c && *c == 0) {
          v = sel.getTrueValue();
          continue;
        }
      }
      return std::nullopt;  // some other conditional we don't model
    }
    break;
  }
  if (!v) return std::nullopt;
  auto ld = mlir::dyn_cast_or_null<fir::LoadOp>(v.getDefiningOp());
  if (!ld) return std::nullopt;
  if (constIndexedElementLoad(v)) return std::nullopt;  // const case handled elsewhere
  auto dg =
      mlir::dyn_cast_or_null<hlfir::DesignateOp>(ld.getMemref().getDefiningOp());
  if (!dg) return std::nullopt;
  auto indices = dg.getIndices();
  if (indices.size() != 1) return std::nullopt;  // v1: single-index only
  std::string array = traceToDecl(dg.getMemref());
  if (array.empty()) return std::nullopt;
  std::string idx = traceExtentExpr(indices[0]);
  if (idx.empty()) idx = traceToDecl(indices[0]);
  if (idx.empty()) return std::nullopt;
  return std::make_pair(array, idx);
}

static std::vector<std::string> resolveShapeSyms(
    hlfir::DeclareOp decl, std::vector<ValueSymbol>* valueSyms = nullptr) {
  std::vector<std::string> syms;

  // (1) Check the attribute set by the shape-propagation pass.
  if (auto hint = decl->getAttrOfType<mlir::ArrayAttr>(kShapeHintAttr)) {
    for (auto a : hint) {
      auto s = mlir::cast<mlir::StringAttr>(a).str();
      syms.push_back(s.empty() ? "?" : s);
    }
    return syms;
  }

  // (2) Trace the shape operand.
  auto shape = decl.getShape();
  if (!shape) return syms;

  // Two unknown-extent sentinels reach the shape op as plain
  // ``arith.constant`` operands and must NOT be stringified into
  // the DaCe descriptor (which rejects negative extents with
  // "Found negative shape in Data"):
  //
  //   * ``fir::SequenceType::getUnknownExtent()`` (= INT64_MIN)  --
  //     the canonical "this dim is dynamic" marker on
  //     ``fir.array<?xT>`` types.
  //   * ``-1``  --  the convention flang uses on the shape op of an
  //     assumed-size dummy (``arr(*)``).  See the IR
  //     ``%shape = fir.shape %c-1 : (index) -> !fir.shape<1>``
  //     emitted for ``real, intent(in) :: src(*)``  --  flang picks
  //     ``-1`` rather than ``INT64_MIN`` here because the operand
  //     is an ``index`` value the runtime would otherwise treat as
  //     a real extent.
  //
  // Either case -> push ``"?"`` so the per-dim synthetic-name
  // fallback at the caller site mints ``<name>_d<i>``.  Any other
  // negative integer is genuinely invalid and we let it surface
  // (flang shouldn't emit such a thing for legal programs).
  auto pushExtent = [&](mlir::Value ext) {
    // Constant extents first -- this catches the dynamic / assumed-size
    // sentinels (INT64_MIN, -1) before they reach traceExtentExpr, which
    // would otherwise stringify them into an invalid negative shape.
    if (auto c = traceConstInt(ext)) {
      if (*c == fir::SequenceType::getUnknownExtent() || *c == -1) {
        syms.push_back("?");
        return;
      }
      syms.push_back(std::to_string(*c));
      return;
    }
    // Runtime-indexed array element as an extent (``z_raylfac(nrdmax(jg))``):
    // ``traceExtentExpr`` / ``traceToDecl`` would collapse it to the bare
    // array name, colliding the array with its own data descriptor.  Mint a
    // distinct value-symbol (``__sym_nrdmax_jg``) and record it so the builder
    // seeds it from the element read (and asserts the element stays constant).
    if (auto ae = arrayElementExtent(ext)) {
      std::string sym = valueSymbolName(ae->first, ae->second);
      syms.push_back(sym);
      if (valueSyms) valueSyms->push_back({sym, ae->first, ae->second});
      return;
    }
    // Symbolic extent.  ``traceExtentExpr`` resolves a scalar, arithmetic,
    // or constant-indexed element extent -- the last as its position
    // symbol (``dims(1)`` -> ``__sym_dims_1``, peeling Flang's
    // ``max(ext, 0)`` clamp) rather than collapsing the element read to
    // its whole-array name the way ``traceToDecl`` would (which then
    // collides the array with its own data descriptor).  It also renders
    // the dynamic gather-temp extent
    // (``arith.select(cmpi_sgt, addi(subi(load_ub, load_lb), 1), 0)``)
    // as a closed-form expression over already-promoted scalar symbols.
    auto expr = traceExtentExpr(ext);
    if (!expr.empty()) {
      syms.push_back(expr);
      return;
    }
    auto n = traceToDecl(ext);
    if (!n.empty()) {
      syms.push_back(n);
      return;
    }
    syms.push_back("?");
  };
  // ``fir.shift`` carries no extents (they live on the box); leaving
  // ``syms`` empty lets the caller's SequenceType / synthetic-name
  // fallback supply them, which is correct for assumed-shape.
  for (auto ext : classifyShapeOperand(shape).extents) pushExtent(ext);

  return syms;
}

/// Build the :type:`AllocSitesIndex` (declared in extract_vars.h) with one
/// module walk, so the per-variable helpers below look a name up instead of
/// re-walking the module each time.
static AllocSitesIndex buildAllocSitesIndex(mlir::ModuleOp module) {
  AllocSitesIndex idx;
  module.walk([&](fir::AllocMemOp a) {
    if (auto un = a.getUniqName()) idx[un->str()].push_back(a);
  });
  return idx;
}

/// Collect every ``fir.allocmem`` whose ``uniq_name`` matches
/// ``<declUniqName>.alloc``, in IR walk order.  Multiple matches indicate
/// that the user wrote more than one ``ALLOCATE`` for the variable
/// (e.g. across an explicit ``DEALLOCATE`` + re-``ALLOCATE``).
///
/// :param idx: when given, the prebuilt name->sites index is consulted in O(1)
///     instead of walking the module; pass it from a loop over many variables.
std::vector<fir::AllocMemOp> collectAllocSites(const std::string& declName,
                                               mlir::ModuleOp module,
                                               const AllocSitesIndex* idx) {
  if (declName.empty()) return {};
  std::string allocName = declName + ".alloc";
  if (idx) {
    auto it = idx->find(allocName);
    return it == idx->end() ? std::vector<fir::AllocMemOp>{} : it->second;
  }
  std::vector<fir::AllocMemOp> sites;
  module.walk([&](fir::AllocMemOp a) {
    auto un = a.getUniqName();
    if (un && un->str() == allocName) sites.push_back(a);
  });
  return sites;
}

/// Are these ALLOCATE sites mutually exclusive  --  each in a different
/// branch of one common ``scf.if`` / ``fir.if`` (a conditional ALLOCATE,
/// ``IF (c) ALLOCATE(a(n)) ELSE ALLOCATE(a(m))``) rather than sequential
/// re-allocation (``ALLOCATE; DEALLOCATE; ALLOCATE``)?
///
/// The two differ fundamentally: a conditional ALLOCATE stores to the same
/// box and the array is used jointly after the IF, so it must stay ONE
/// transient with a branch-dependent extent symbol (each branch assigns
/// the extent; they merge at the join).  Versioning it into ``a_alloc1``
/// would split it into two transients and bind post-IF reads statically to
/// whichever branch ran last  --  wrong.
///
/// Recognised shape: every site sits inside one common enclosing if, and
/// no two sites share that if's branch region (so exactly one alloc fires).
bool allocSitesInExclusiveBranches(const std::vector<fir::AllocMemOp>& sites) {
  if (sites.size() < 2) return false;
  // Map each enclosing if of ``op`` to the region of it that holds ``op``.
  auto ifRegions = [](mlir::Operation* op) {
    std::map<mlir::Operation*, mlir::Region*> m;
    for (mlir::Region* r = op->getParentRegion(); r;) {
      mlir::Operation* p = r->getParentOp();
      if (!p) break;
      if (mlir::isa<mlir::scf::IfOp, fir::IfOp>(p)) m[p] = r;
      r = p->getParentRegion();
    }
    return m;
  };
  // Two ops are mutually exclusive iff a common-ancestor if holds them in
  // different regions (then vs else).
  auto exclusive = [&](mlir::Operation* a, mlir::Operation* b) {
    auto am = ifRegions(a);
    for (auto& kv : ifRegions(b)) {
      auto it = am.find(kv.first);
      if (it != am.end() && it->second != kv.second) return true;
    }
    return false;
  };
  for (size_t i = 0; i < sites.size(); ++i)
    for (size_t j = i + 1; j < sites.size(); ++j) {
      fir::AllocMemOp si = sites[i], sj = sites[j];
      if (!exclusive(si.getOperation(), sj.getOperation())) return false;
    }
  return true;
}

std::vector<std::vector<fir::AllocMemOp>> groupAllocSites(
    const std::string& declName, mlir::ModuleOp module,
    const AllocSitesIndex* idx) {
  auto sites = collectAllocSites(declName, module, idx);
  std::vector<std::vector<fir::AllocMemOp>> classes;
  unsigned n = sites.size();
  if (n == 0) return classes;
  // The reaching-set walk below only resolves which of *several* sites merge
  // into one buffer; the overwhelmingly common single-ALLOCATE case needs none
  // of it -- and that walk is over the whole enclosing function, which is the
  // expensive part on a fully-inlined entry.  Short-circuit it.
  if (n == 1) {
    classes.push_back(std::move(sites));
    return classes;
  }

  // site index by allocmem op; union-find over indices.
  std::map<mlir::Operation*, unsigned> idxOf;
  for (unsigned i = 0; i < n; ++i) idxOf[sites[i].getOperation()] = i;
  std::vector<unsigned> parent(n);
  for (unsigned i = 0; i < n; ++i) parent[i] = i;
  std::function<unsigned(unsigned)> find = [&](unsigned x) {
    while (parent[x] != x) x = parent[x] = parent[parent[x]];
    return x;
  };
  auto unite = [&](unsigned a, unsigned b) { parent[find(a)] = find(b); };

  // Structured reaching-set walk.  A new ALLOCATE replaces the current
  // buffer (Fortran allows only one live buffer per name); an scf.if /
  // fir.if whose two branches BOTH stay live at the join merges their
  // sites (they are alternatives for the post-IF buffer).  No explicit
  // DEALLOCATE handling is needed: re-ALLOCATE already replaces, and a
  // branch that allocates-then-frees still ends with no extra live site.
  using Reaching = std::set<unsigned>;
  std::function<Reaching(mlir::Block&, Reaching)> walk =
      [&](mlir::Block& blk, Reaching reaching) -> Reaching {
    auto mergeBranches = [&](const Reaching& t, const Reaching& e) {
      if (t.empty() || e.empty()) return;
      std::vector<unsigned> all(t.begin(), t.end());
      all.insert(all.end(), e.begin(), e.end());
      for (size_t k = 1; k < all.size(); ++k) unite(all[0], all[k]);
    };
    for (auto& op : blk) {
      if (auto am = mlir::dyn_cast<fir::AllocMemOp>(&op)) {
        auto it = idxOf.find(&op);
        if (it != idxOf.end()) {
          reaching = {it->second};
          continue;
        }
      }
      if (auto sif = mlir::dyn_cast<mlir::scf::IfOp>(&op)) {
        Reaching rt = walk(sif.getThenRegion().front(), reaching);
        Reaching re = sif.getElseRegion().empty()
                          ? reaching
                          : walk(sif.getElseRegion().front(), reaching);
        mergeBranches(rt, re);
        reaching.clear();
        reaching.insert(rt.begin(), rt.end());
        reaching.insert(re.begin(), re.end());
        continue;
      }
      if (auto fif = mlir::dyn_cast<fir::IfOp>(&op)) {
        Reaching rt = walk(fif.getThenRegion().front(), reaching);
        Reaching re = fif.getElseRegion().empty()
                          ? reaching
                          : walk(fif.getElseRegion().front(), reaching);
        mergeBranches(rt, re);
        reaching.clear();
        reaching.insert(rt.begin(), rt.end());
        reaching.insert(re.begin(), re.end());
        continue;
      }
      // Any other region-bearing op (loops, ...): thread the reaching set
      // through each contained block.  An ALLOCATE inside a loop body
      // re-allocates per iteration -- ``reaching`` simply tracks the last.
      for (auto& reg : op.getRegions())
        for (auto& b : reg.getBlocks()) reaching = walk(b, reaching);
    }
    return reaching;
  };

  if (auto fop = sites[0].getOperation()->getParentOfType<mlir::func::FuncOp>())
    walk(fop.getBody().front(), {});

  // Gather classes by root.  Members are pushed in site (first-def) order,
  // so members[0] is the class's minimum index; order classes by it.
  std::map<unsigned, std::vector<fir::AllocMemOp>> byRoot;
  for (unsigned i = 0; i < n; ++i) byRoot[find(i)].push_back(sites[i]);
  for (auto& kv : byRoot) classes.push_back(std::move(kv.second));
  std::sort(classes.begin(), classes.end(),
            [&](const std::vector<fir::AllocMemOp>& a,
                const std::vector<fir::AllocMemOp>& b) {
              fir::AllocMemOp fa = a.front(), fb = b.front();
              return idxOf[fa.getOperation()] < idxOf[fb.getOperation()];
            });
  return classes;
}

/// Resolve the runtime shape of one ``fir.allocmem`` site to a symbol
/// name list, the same way ``resolveShapeSyms`` resolves a static
/// declare's shape  --  trace each size operand to its host declare
/// (preferred), fall back to a constant literal, then to ``?``.
static std::vector<std::string> shapeFromAllocSite(fir::AllocMemOp alloc) {
  std::vector<std::string> syms;
  for (auto sz : alloc.getShape()) {
    // ``traceExtentExpr`` peels Flang's ``max(ext, 0)`` clamp and
    // resolves the underlying extent uniformly: a constant-indexed array
    // element (``dims(1)``) becomes its position symbol ``__sym_dims_1``
    // rather than the whole array name -- promoting the array would
    // collide it with its own data descriptor.  Falls back to the plain
    // scalar / constant resolvers for shapes it doesn't recognise.
    auto e = traceExtentExpr(sz);
    if (!e.empty()) {
      syms.push_back(e);
      continue;
    }
    auto n = traceToDecl(sz);
    if (!n.empty()) {
      syms.push_back(n);
      continue;
    }
    if (auto c = traceConstInt(sz)) {
      syms.push_back(std::to_string(*c));
      continue;
    }
    syms.push_back("?");
  }
  return syms;
}

/// Recover per-dim lower bounds from an ``ALLOCATE(arr(lb:ub))``
/// site.  Flang lowers this to a chain
///
///     %alloc = fir.allocmem !fir.array<?xT>, %extent
///     %ss    = fir.shape_shift %lb, %extent : !fir.shapeshift<1>
///     %box   = fir.embox %alloc(%ss) : ...
///     fir.store %box, %decl_box_slot
///
/// where the first operand of every (lb, extent) pair on the
/// ``shape_shift`` is the Fortran-declared lower bound.  Find the
/// ``embox`` consuming this allocmem, peel through any
/// ``fir.convert`` wrappers, then read the ``shape_shift``'s
/// lower-bound operands.  Per-dim values are stringified literal
/// integers, ``traceToDecl``-mapped symbol names, or ``"?"`` when
/// neither resolves.
static std::vector<std::string> lowerBoundsFromAllocSite(
    fir::AllocMemOp alloc) {
  std::vector<std::string> lbs;

  // Walk users for an embox.  ``fir.convert`` may sit between the
  // allocmem result and the embox memref operand; peel it via a
  // tight worklist (depth bounded).
  auto peelToEmbox = [](mlir::Value v) -> fir::EmboxOp {
    for (int i = 0; i < limits::kSsaBackWalkDepth && v; ++i) {
      for (auto* u : v.getUsers()) {
        if (auto eb = mlir::dyn_cast<fir::EmboxOp>(u)) return eb;
      }
      // ``fir.convert`` produces a fresh value -- check its users.
      mlir::Value next;
      for (auto* u : v.getUsers()) {
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(u)) {
          next = cv.getResult();
          break;
        }
      }
      if (!next) break;
      v = next;
    }
    return nullptr;
  };
  auto embox = peelToEmbox(alloc.getResult());
  if (!embox) return lbs;

  auto shape = embox.getShape();
  if (!shape) return lbs;
  auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(shape.getDefiningOp());
  if (!ss) return lbs;

  // ShapeShift operand layout: (lb_d0, ext_d0, lb_d1, ext_d1, ...).
  auto ops = ss->getOperands();
  for (unsigned i = 0; i < ops.size(); i += 2) {
    if (auto c = traceConstInt(ops[i]))
      lbs.push_back(std::to_string(*c));
    else {
      auto n = traceToDecl(ops[i]);
      lbs.push_back(n.empty() ? "?" : n);
    }
  }
  return lbs;
}

/// True iff some ``fir.box_addr`` op in the module reads the
/// allocatable / pointer descriptor of the declare whose
/// (short, post-``extractName``) name is ``shortName``.
/// ``ALLOCATED(arr)`` and ``ASSOCIATED(ptr)`` both lower to
/// ``box_addr(load arr_box) != 0``; if no such reader exists the
/// per-allocatable ``<arr>_allocated`` tracker scalar and its init
/// state are dead weight in the SDFG.
/// Short (post-``extractName``) names that have an ``ALLOCATED`` / ``ASSOCIATED``
/// reader (a ``fir.box_addr``), built with one module walk.  Lets
/// ``needsAllocatedTracker`` look a name up instead of re-walking per variable.
static std::set<std::string> buildAllocatedReaderNames(mlir::ModuleOp module) {
  std::set<std::string> names;
  module.walk([&](fir::BoxAddrOp ba) {
    auto src = ba.getVal();
    if (auto* sd = src.getDefiningOp())
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(sd)) src = ld.getMemref();
    auto n = traceToDecl(src);
    if (!n.empty()) names.insert(n);
  });
  return names;
}

static bool hasAllocatedReader(const std::string& shortName,
                               mlir::ModuleOp module,
                               const std::set<std::string>* readerNames = nullptr) {
  if (shortName.empty()) return false;
  if (readerNames) return readerNames->count(shortName) > 0;
  bool found = false;
  module.walk([&](fir::BoxAddrOp ba) {
    if (found) return;
    // ``box_addr``'s operand is normally a ``fir.load`` of a
    // box-ref; trace through that load to the declare.
    auto src = ba.getVal();
    if (auto* sd = src.getDefiningOp())
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(sd)) src = ld.getMemref();
    // ``traceToDecl`` returns the short (extracted) name  --  match
    // against ``shortName``, not the full mangled uniq_name.
    if (traceToDecl(src) == shortName) found = true;
  });
  return found;
}

/// True iff the allocatable / pointer ``declUniqName`` needs the
/// ``<short>_allocated`` tracker scalar  --  either because some
/// kernel-body code writes it (an ALLOCATE / DEALLOCATE site exists,
/// keyed on the full mangled uniq_name) OR because some kernel-body
/// code reads it (an ``ALLOCATED(arr)`` / ``ASSOCIATED(ptr)`` reader
/// exists, keyed on the short post-``extractName`` name).  Dummy /
/// module-level allocatables passed in already-allocated and never
/// queried by ``ALLOCATED(...)`` skip the tracker entirely.
bool needsAllocatedTracker(const std::string& declUniqName,
                           mlir::ModuleOp module,
                           const AllocSitesIndex* idx,
                           const std::set<std::string>* readerNames) {
  if (declUniqName.empty()) return false;
  if (!collectAllocSites(declUniqName, module, idx).empty()) return true;
  return hasAllocatedReader(extractName(declUniqName), module, readerNames);
}

/// First ALLOCATE keeps the allocatable's original Fortran name (so
/// every existing single-allocation test stays green); subsequent
/// allocations mint fresh transient names ``<x>_alloc1``,
/// ``<x>_alloc2``, ... one per re-allocation site.
std::string allocAliasName(const std::string& fortran, unsigned site) {
  if (site == 0) return fortran;
  return fortran + "_alloc" + std::to_string(site);
}

static std::vector<std::string> resolveLowerBounds(hlfir::DeclareOp decl) {
  std::vector<std::string> lbs;

  // ``hlfir-flatten-structs`` lb_hint: authoritative per-dim lower
  // bounds for a synthesised flat companion whose declare carries
  // only a plain ``fir.shape`` (the nested member's real lb lived on
  // the rewritten-away designate's ``fir.shape_shift``).  Consulted
  // before the shape operand so the SequenceType fallback can't
  // default the dims to 1 (E8).
  if (auto hint = decl->getAttrOfType<mlir::ArrayAttr>(kLbHintAttr)) {
    for (auto a : hint) lbs.push_back(mlir::cast<mlir::StringAttr>(a).str());
    return lbs;
  }

  auto si = classifyShapeOperand(decl.getShape());
  if (si.kind == ShapeOperandInfo::None) return lbs;

  // ``fir.shape``: HLFIR guarantees lbs are omitted iff every dim is
  // the Fortran default 1 -- authoritative, no tracing needed.
  if (si.kind == ShapeOperandInfo::Shape) {
    lbs.assign(si.rank, "1");
    return lbs;
  }

  // ``fir.shape_shift`` (lb,ext pairs) and ``fir.shift`` (lbs only)
  // both carry the authoritative explicit per-dim lower bounds for
  // an assumed-shape / pointer dummy declared with explicit local
  // bounds (``a(10:,20:)``) -- ``si.lbs`` holds them for both forms.
  for (auto lb : si.lbs) {
    if (auto c = traceConstInt(lb))
      lbs.push_back(std::to_string(*c));
    else {
      auto n = traceToDecl(lb);
      lbs.push_back(n.empty() ? "?" : n);
    }
  }
  return lbs;
}

/// True iff ``dg``'s memref chain bottoms out at ``decl``'s result.
/// Walks through the same op set ``traceToDecl`` peels (fir.load,
/// fir.rebox, fir.convert, fir.box_addr, hlfir.designate as a
/// chain link).  Bounded depth to keep the walk cheap.
static bool designateRootedAt(hlfir::DesignateOp dg, hlfir::DeclareOp decl) {
  mlir::Value v = dg.getMemref();
  for (int i = 0; i < limits::kSsaBackWalkDepth && v; ++i) {
    auto* d = v.getDefiningOp();
    if (!d) return false;
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
      if (dc == decl) return true;
      // Inlined-callee aliasing declare: its memref derives from
      // ``decl#0`` (or a peelable chain over it).  Trace through
      // to keep matching designates that live inside inlined
      // subroutines on the same root storage.
      v = dc.getMemref();
      continue;
    }
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      v = cv.getValue();
      continue;
    }
    if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
      v = ld.getMemref();
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
    if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d)) {
      v = co.getRef();
      continue;
    }
    if (auto inner = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
      v = inner.getMemref();
      continue;
    }
    break;
  }
  return false;
}

/// Static lower-bound inference for deferred-shape allocatable /
/// pointer arrays whose declare carries no explicit bounds.
///
/// Background: ``INTEGER, ALLOCATABLE :: arr(:)`` (or the equivalent
/// dummy-arg form) leaves the lower bound unknown at extract time --
/// it's set at runtime by the upstream ``ALLOCATE(arr(lb:ub))``,
/// which the bridge generally can't see.  ``resolveLowerBounds``
/// returns empty; the caller (line 919/946) fills with ``"1"``s,
/// which is correct iff every access in the body uses an index >= 1.
///
/// ICON breaks that assumption: ``p_patch%edges%start_block(:)`` is
/// declared deferred-shape, allocated upstream with bounds
/// ``min_rlcell_int:max_rlcell_int`` (= ~``-10:7``), and read in the
/// kernel body via literal negative indices like ``end_block(-10)``.
/// Without an offset correction the access lowers to ``end_block[-11]``
/// -- invalid pointer dereference at runtime.
///
/// Inference: a literal index ``N`` appearing on ``arr`` in any
/// ``hlfir.designate`` is a *lower bound on the array's actual lower
/// bound*: the array must extend to at least ``N`` for the access to
/// be valid.  Take the min over all literal indices per dim; if it
/// drops below the current default, replace.  Symbolic indices
/// (loop iterators, indirect table reads) don't contribute -- those
/// are out of scope for this pass.
///
/// Backward-compatible: arrays whose body accesses use only literals
/// ``>= 1`` keep ``lb = "1"`` unchanged.
///
/// Populates ``seenLit`` (out): per-dim flag for whether ANY literal
/// index was observed.  Used by the dummy-arg-allocatable free-offset
/// fallback to distinguish "purely symbolic access (need caller-bound
/// offset)" from "literal-positive access (1-based default is fine)".
/// Try to recover a constant integer that ``v`` evaluates to,
/// peeling one or more ``fir.load %decl`` indirections by scanning
/// the function for ``fir.store <const>, %decl`` writes.
///
/// Used by ``inferLowerBoundsFromLiteralAccesses`` to handle the
/// inlined-callee pattern ICON's ``get_indices_c`` uses:
///
///     irl_end = opt_rl_end             ! ``fir.store -5, %irl_end_decl``
///     i_endidx_in = arr(irl_end)       ! designate index = ``fir.load
///     %irl_end_decl``
///
/// Plain ``traceConstInt`` returns nullopt for the loaded value;
/// this helper recursively peels the inlined-callee chain:
///   * ``hlfir.associate %c-5``        (callee arg materialised by value)
///   * ``hlfir.declare`` aliases       (inlined dummy re-declares)
///   * ``fir.convert``                 (i32 -> i64 index coercions)
///   * ``fir.load %X`` -> ``fir.store <v>, %X``  (local stash)
/// recursing on the stored value at each store.  Returns the
/// most-negative literal that reaches the index (matching the
/// per-dim ``min`` semantics in the caller).  Bounded recursion.
///
/// :param v: SSA value used as a designate index.
/// :param func: enclosing function (scopes the store walk).
/// :param depth: recursion guard.
/// :returns: const value if a literal reaches the index, else nullopt.
/// Writes (``fir.store`` / ``hlfir.assign``) to a function's memrefs, indexed
/// once so ``traceConstIntThroughLoad`` need not re-walk every store/assign per
/// load.  ``byValue`` keys the exact target SSA value; ``byName`` keys
/// ``traceToDecl`` of the target -- the inlined-alias case where a load reads
/// ``decl#1`` while the store wrote ``decl#0``.
struct FuncWrites {
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> byValue;
  std::map<std::string, std::vector<mlir::Value>> byName;
};

/// Build (once, cached per function) the store/assign target index.  This
/// replaces two per-load ``func.walk``s that, on a large kernel, re-scanned the
/// whole function for every loaded index -- O(declares x designates x loads x
/// IR), the dominant ``extractVariables`` cost on ICON's ``solve_nh``.
static const FuncWrites& funcWritesFor(
    mlir::func::FuncOp func, std::map<mlir::Operation*, FuncWrites>& cache) {
  auto it = cache.find(func.getOperation());
  if (it != cache.end()) return it->second;
  FuncWrites& w = cache[func.getOperation()];
  func.walk([&](fir::StoreOp st) {
    w.byValue[st.getMemref()].push_back(st.getValue());
    auto n = traceToDecl(st.getMemref());
    if (!n.empty()) w.byName[n].push_back(st.getValue());
  });
  func.walk([&](hlfir::AssignOp as) {
    w.byValue[as.getLhs()].push_back(as.getRhs());
    auto n = traceToDecl(as.getLhs());
    if (!n.empty()) w.byName[n].push_back(as.getRhs());
  });
  return w;
}

static std::optional<int64_t> traceConstIntThroughLoad(
    mlir::Value v, mlir::func::FuncOp func,
    std::map<mlir::Operation*, FuncWrites>& writeCache, int depth = 0,
    llvm::SmallPtrSet<mlir::Operation*, 16>* visited = nullptr) {
  if (depth > 64 || !v) return std::nullopt;
  if (auto c = traceConstInt(v)) return c;
  auto* def = v.getDefiningOp();
  if (!def) return std::nullopt;
  // SSA def chains are DAGs (a stored value reused by several loads); mark each
  // op once so a shared sub-chain is explored once, not re-walked per path.
  // The most-negative literal is still reached -- each literal's op is visited
  // exactly once and folded into the caller's running min.
  llvm::SmallPtrSet<mlir::Operation*, 16> seen;
  if (!visited) visited = &seen;
  if (!visited->insert(def).second) return std::nullopt;

  // hlfir.associate %c {adapt.valuebyref} -- the callee received
  // the literal by value; its source is the constant.
  if (auto as = mlir::dyn_cast<hlfir::AssociateOp>(def))
    return traceConstIntThroughLoad(as.getSource(), func, writeCache, depth + 1,
                                    visited);
  // fir.convert (kind coercion).
  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
    return traceConstIntThroughLoad(cv.getValue(), func, writeCache, depth + 1,
                                    visited);
  // hlfir.declare alias -- trace its backing memref.
  if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(def))
    return traceConstIntThroughLoad(dc.getMemref(), func, writeCache, depth + 1,
                                    visited);

  auto ld = mlir::dyn_cast<fir::LoadOp>(def);
  if (!ld) return std::nullopt;
  auto target = ld.getMemref();

  // Candidate writes to this load's target.  The precomputed index yields the
  // same set the old per-load store/assign walks matched via ``sameTarget``:
  // when the target has a declare name, every write to that name (covers the
  // inlined-alias case AND exact-value writes, since a write to ``target`` has
  // the same ``traceToDecl`` name); otherwise exact-value writes only.
  auto targetName = traceToDecl(target);
  const FuncWrites& w = funcWritesFor(func, writeCache);
  std::optional<int64_t> result;
  auto consider = [&](mlir::Value writeVal) {
    if (auto c = traceConstIntThroughLoad(writeVal, func, writeCache, depth + 1,
                                          visited))
      if (!result || *c < *result) result = c;
  };
  if (!targetName.empty()) {
    if (auto it = w.byName.find(targetName); it != w.byName.end())
      for (auto wv : it->second) consider(wv);
  } else if (auto it = w.byValue.find(target); it != w.byValue.end()) {
    for (auto wv : it->second) consider(wv);
  }
  if (result) return result;

  // No store reached -- the load may read a declare that aliases an
  // ``hlfir.associate`` (inlined by-value dummy with no explicit
  // store).  Peel the load target through the declare chain.
  if (auto* tdef = target.getDefiningOp()) {
    if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(tdef))
      return traceConstIntThroughLoad(dc.getMemref(), func, writeCache,
                                      depth + 1, visited);
  }
  return std::nullopt;
}

static void inferLowerBoundsFromLiteralAccesses(
    hlfir::DeclareOp decl, std::vector<std::string>& lbs, int rank,
    std::map<mlir::Operation*, FuncWrites>& writeCache,
    const llvm::DenseMap<mlir::Operation*,
                         llvm::SmallVector<hlfir::DesignateOp, 4>>&
        designatesByDecl,
    std::vector<bool>* seenLitOut = nullptr) {
  if (rank <= 0) return;
  auto func = decl->getParentOfType<mlir::func::FuncOp>();
  if (!func) return;

  std::vector<int64_t> minLit(rank, std::numeric_limits<int64_t>::max());
  std::vector<bool> seenLit(rank, false);

  // Designates rooted at ``decl`` are looked up from the once-built index
  // (see the builder in ``extractVariables``) instead of re-walking every
  // designate in the function per declare.
  auto dit = designatesByDecl.find(decl.getOperation());
  if (dit != designatesByDecl.end())
    for (auto dg : dit->second) {
      auto indices = dg.getIndices();
      unsigned nIdx = std::min<unsigned>(indices.size(), (unsigned)rank);
      for (unsigned d = 0; d < nIdx; ++d) {
        // Peel a single ``fir.load %decl`` indirection if needed
        // (inlined-callee pattern: caller passes -5, callee stores
        // it to a local, then loads it for the designate index).
        if (auto c = traceConstIntThroughLoad(indices[d], func, writeCache)) {
          if (*c < minLit[d]) minLit[d] = *c;
          seenLit[d] = true;
        }
      }
    }

  if ((int)lbs.size() < rank) lbs.resize(rank, "1");
  for (int d = 0; d < rank; ++d) {
    if (!seenLit[d]) continue;
    // Only adjust the current default ``"1"`` (the bridge's
    // unknown-bound fallback).  An explicit non-default value
    // (e.g. extracted from a fir.ShapeShiftOp on the declare)
    // wins -- it's authoritative source-of-truth.
    if (lbs[d] != "1") continue;
    int curr = 1;
    if (minLit[d] < curr) lbs[d] = std::to_string(minLit[d]);
  }

  if (seenLitOut) *seenLitOut = std::move(seenLit);
}

/// Find the fir.do_loop induction variable's Fortran name by looking for
/// `fir.store %block_arg, %alloca` in the loop body.
static std::string traceLoopIter(fir::DoLoopOp loop) {
  for (auto& op : loop.getRegion().front())
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
      auto n = traceToDecl(st.getMemref());
      if (!n.empty()) return n;
    }
  return "";
}

/// Walk the defining-op graph backwards from an integer-valued
/// expression, collecting the Fortran names of every ``hlfir.declare``'d
/// integer scalar that feeds into it.  Used by pass 2c to promote scalars
/// that appear in an array-index expression (``a(base + jh)`` -> ``base``
/// is a symbol) and by pass 2d for loop / branch-condition scalars.
///
/// The principle is the same for both: a scalar whose value reaches an
/// index subset or a control-flow condition must be a symbol, because
/// DaCe memlet subsets and interstate conditions are symbolic.  Writing
/// to a symbol then routes the assignment through the interstate-edge
/// path in ``_emit_assign``, keeping the value live across states.
///
/// Recognised shape: ``arith.cmp*`` (condition leaf), the integer
/// arithmetic Flang emits inside an index (``arith.addi/subi/muli/
/// divsi/divui/remsi/remui``), the logical combinators / int casts the
/// lift-cf-to-scf chain wraps a condition in (``arith.xori/andi/ori/
/// trunci/extui/extsi``) and ``fir.convert``.  Stops at a ``fir.load``
/// (hands off to ``traceToDecl``) or an op it doesn't recognise.
static void collectIntegerScalarReads(
    mlir::Value v, std::set<std::string>& out,
    llvm::SmallPtrSet<mlir::Operation*, 32>* visited = nullptr) {
  if (!v) return;
  auto* def = v.getDefiningOp();
  if (!def) return;
  // Index expressions form a DAG: one arith sub-expression (a shared ``base``,
  // a reused induction term) feeds many ``hlfir.designate`` indices and recurs
  // within a single index.  Without a visited set the operand recursion
  // re-explores shared subtrees, which is exponential on a real kernel (ICON's
  // ``solve_nh`` spins here for tens of minutes at 100% CPU).  Visit each
  // defining op once instead: a fresh set is seeded on the top-level call and
  // threaded through the recursion, making the walk linear in the DAG size.
  llvm::SmallPtrSet<mlir::Operation*, 32> seen;
  if (!visited) visited = &seen;
  if (!visited->insert(def).second) return;

  // Comparison leaves: recurse into both operands to catch both sides
  // of ``i < n`` (i and n both become symbols if they're declared).
  if (mlir::isa<mlir::arith::CmpFOp, mlir::arith::CmpIOp>(def)) {
    for (auto operand : def->getOperands())
      collectIntegerScalarReads(operand, out, visited);
    return;
  }

  // Integer arithmetic inside an index expression (``base + jh``,
  // ``c*i + d``), the logical combinators and int casts the
  // lift-cf-to-scf chain emits: recurse into every operand so each
  // declared scalar leaf is promoted.
  if (mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp, mlir::arith::MulIOp,
                mlir::arith::DivSIOp, mlir::arith::DivUIOp,
                mlir::arith::RemSIOp, mlir::arith::RemUIOp, mlir::arith::XOrIOp,
                mlir::arith::AndIOp, mlir::arith::OrIOp, mlir::arith::TruncIOp,
                mlir::arith::ExtUIOp, mlir::arith::ExtSIOp, fir::ConvertOp>(
          def)) {
    for (auto operand : def->getOperands())
      collectIntegerScalarReads(operand, out, visited);
    return;
  }

  // Scalar read: trace to its declare; every op on the trace chain
  // (fir.load + hlfir.declare) resolves to the Fortran name.  Only
  // collect INTEGER-typed scalars -- float and LOGICAL scalars used
  // in branch conditions (e.g. ``IF (zsupsat > zepsec)``,
  // ``IF (llo1)`` where ``llo1 = (a>b) .AND. (c>d) .AND. ...``) must
  // stay as plain scalars so their assignments route through the
  // tasklet path (which preserves complex RHS like
  // ``MAX((a-b*c)/d, 0)`` or a multi-AND boolean expression); the
  // interstate-edge path used for symbol writes only handles trivial
  // single-array-read RHSs, so promoting a non-integer scalar here
  // drops everything past the first array read in the expression.
  if (mlir::isa<fir::LoadOp>(def)) {
    if (v.getType().isIntOrIndex()) {
      auto n = traceToDecl(v);
      if (!n.empty()) out.insert(n);
    }
    return;
  }

  // Anything else (constants, an unrecognised producer, ...)  --  trace
  // through traceToDecl as a last resort; it already handles several
  // pass-through ops.  Same integer-only filter so non-integer scalars
  // don't get promoted to symbols here either.
  if (v.getType().isIntOrIndex()) {
    auto n = traceToDecl(v);
    if (!n.empty()) out.insert(n);
  }
}

// Extract the dense initial values of a ``fir.global ... constant``
// op into a flat ``std::vector<double>`` (row-major).  Returns an
// empty vector when the global isn't a recognisable constant pool
// entry (no initialiser, non-dense init, non-numeric element type).
//
// Background: Flang lowers Fortran array literals like
// ``(/ 2.0d0, 3.0d0, 4.0d0 /)`` to a read-only ``fir.global`` with
// a ``dense<[...]>`` attribute, addressed via ``fir.address_of``.
// We surface the data on the corresponding VarInfo so the SDFG
// builder can synthesise an init state writing those values into
// the transient  --  the kernel's reads then see the right data
// instead of zeros.
//
// All values widen to ``double`` for transport; the Python side
// narrows to the actual SDFG dtype (``int32`` / ``float32`` / ...)
// at descriptor-write time.
static std::vector<double> extractGlobalInitData(fir::GlobalOp gop) {
  std::vector<double> out;
  if (!gop) return out;
  // Path 1: a ``DenseElementsAttr`` initialiser living directly on the
  // ``fir.global`` op.  Two Fortran shapes lower this way -- both are an
  // "array of constants" that the SDFG bakes into a constexpr array
  // (``sdfg.add_constant``) downstream:
  //   * ``parameter, dimension(...) :: x = (/ ... /)`` / ``parameter ::
  //     x = <literal>``  --  the global is marked ``constant``;
  //   * a DATA-statement-initialised array (``real :: c(3); data c
  //     /.../``  --  an array of extrapolation / coefficient constants,
  //     common in scientific codes).  The global is NOT marked
  //     ``constant`` (DATA variables are mutable), but the dense
  //     attribute is still its canonical static initial data.
  // Extract in BOTH cases; the classification downstream bakes a
  // read-only one and seeds a kernel-written one (``is_written``).
  if (auto initOpt = gop.getInitVal()) {
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(*initOpt)) {
      auto eleTy = dense.getElementType();
      if (eleTy.isF64()) {
        for (auto v : dense.getValues<double>()) out.push_back(v);
      } else if (eleTy.isF32()) {
        for (auto v : dense.getValues<float>()) out.push_back((double)v);
      } else if (eleTy.isInteger(8)) {
        for (auto v : dense.getValues<int8_t>()) out.push_back((double)v);
      } else if (eleTy.isInteger(16)) {
        for (auto v : dense.getValues<int16_t>()) out.push_back((double)v);
      } else if (eleTy.isInteger(32)) {
        for (auto v : dense.getValues<int32_t>()) out.push_back((double)v);
      } else if (eleTy.isInteger(64)) {
        for (auto v : dense.getValues<int64_t>()) out.push_back((double)v);
      } else if (eleTy.isInteger(1)) {
        for (auto v : dense.getValues<bool>()) out.push_back(v ? 1.0 : 0.0);
      }
      if (!out.empty()) return out;
    }
  }
  // Path 2: scalar ``fir.global`` (e.g. ``real :: bob = 1`` declared
  // at module scope without ``parameter``).  The initialiser lives
  // in the body as an ``arith.constant`` feeding a ``fir.has_value``
  // terminator  --  extract the constant attribute, narrowing to a
  // single-element ``out`` vector.
  if (gop.getRegion().empty()) return out;
  for (auto& op : gop.getRegion().front()) {
    auto hv = mlir::dyn_cast<fir::HasValueOp>(op);
    if (!hv) continue;
    auto* def = hv.getResval().getDefiningOp();
    // Peel a kind/representation ``fir.convert`` (a LOGICAL init is
    // ``arith.constant false : i1`` -> ``fir.convert i1 to logical<4>``).
    while (def)
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def))
        def = cv.getValue().getDefiningOp();
      else
        break;
    if (!def) return out;
    auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def);
    if (!cst) return out;
    auto attr = cst.getValue();
    if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(attr)) {
      out.push_back(fa.getValueAsDouble());
    } else if (auto ba = mlir::dyn_cast<mlir::BoolAttr>(attr)) {
      out.push_back(ba.getValue() ? 1.0 : 0.0);
    } else if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
      out.push_back((double)ia.getInt());
    }
    return out;
  }
  return out;
}

// Trace a declare's memref back to the global it references via
// ``fir.address_of``.  Returns the symbol name (without leading
// ``@``) or empty string if the chain doesn't end at an address_of.
// Walks through ``fir.convert`` shims that flang occasionally
// inserts between the address_of and the declare's memref.
static std::string traceToGlobalSymbol(mlir::Value memref) {
  for (int i = 0; i < limits::kSsaBackWalkDepth && memref; ++i) {
    auto* d = memref.getDefiningOp();
    if (!d) return {};
    if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
      memref = cv.getValue();
      continue;
    }
    if (auto ad = mlir::dyn_cast<fir::AddrOfOp>(d)) {
      return ad.getSymbol().getRootReference().str();
    }
    return {};
  }
  return {};
}

std::pair<std::string, std::string> decodeModuleGlobalSymbol(
    const std::string& sym) {
  llvm::StringRef s(sym);
  // Module data only.  ``_QF`` = function-scope SAVE local (private,
  // not caller-bindable), ``_QP`` = program/procedure, ``_QQ`` =
  // compiler-synthesised (read-only literal constant pool); none of
  // those is a ``USE``-importable module global.
  if (!s.consume_front("_QM")) return {};
  // The entity is the segment after the FINAL scope separator.  For a
  // plain module variable the symbol is ``_QM<mod>E<entity>``; for a
  // submodule / nested-module member Flang inserts further ``S`` / ``N``
  // scope letters before the terminal ``E`` (``_QM<mod>S<sub>E<ent>``).
  // We split on the last ``E`` so the module segment carries whatever
  // inner scoping Flang produced  --  the emitter only needs a name it
  // can ``USE``; the top-level module name is its leading token, and
  // ``USE`` of a submodule member resolves through the parent module.
  auto eP = s.rfind('E');
  if (eP == llvm::StringRef::npos || eP == 0 || eP + 1 >= s.size()) return {};
  std::string mod = s.substr(0, eP).str();
  std::string name = s.substr(eP + 1).str();
  // Reject names that still contain scope letters or dots  --  those
  // are compiler-internal (type-info tables, constructor thunks), not
  // user module data.  A real Fortran entity name is lower-case
  // identifier characters only (Flang lowercases source identifiers).
  for (char c : name)
    if (!(std::islower(static_cast<unsigned char>(c)) ||
          std::isdigit(static_cast<unsigned char>(c)) || c == '_'))
      return {};
  if (mod.empty()) return {};
  return {mod, name};
}

// True iff the module-scope global ``sym`` is written anywhere in the module
// (a ``fir.store`` to it, or an ``hlfir.assign`` to it or to an
// ``hlfir.designate`` rooted at it -- e.g. an element write ``tablew(i) =
// ...``).  Distinguishes a written module-scope scratch global (a lookup
// table an init routine such as ``qsmith_init_w`` fills, then a reader
// consumes) from a read-only caller-supplied config global: the former is
// the kernel's own transient, not an input the caller must provide.
/// Set of global symbols written somewhere in the module, built with one walk.
/// Mirrors ``globalIsWritten``'s per-symbol logic (a write is a store/assign
/// whose target is a global-backed declare's result, or a designate rooted at
/// one) so a variable-loop caller can look a symbol up instead of re-walking
/// the module -- twice -- per global-backed variable.
static std::set<std::string> buildWrittenGlobals(mlir::ModuleOp module) {
  llvm::DenseMap<mlir::Value, std::string> symByResult;
  llvm::SmallVector<std::pair<hlfir::DeclareOp, std::string>, 8> globalDecls;
  module.walk([&](hlfir::DeclareOp d) {
    std::string sym = traceToGlobalSymbol(d.getMemref());
    if (sym.empty()) return;
    symByResult[d.getResult(0)] = sym;
    symByResult[d.getResult(1)] = sym;
    globalDecls.push_back({d, sym});
  });
  std::set<std::string> written;
  auto markWrite = [&](mlir::Value dest) {
    auto it = symByResult.find(dest);
    if (it != symByResult.end()) {
      written.insert(it->second);
      return;
    }
    if (auto* dd = dest.getDefiningOp())
      if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd))
        for (auto& [d, sym] : globalDecls)
          if (designateRootedAt(dg, d)) {
            written.insert(sym);
            break;
          }
  };
  module.walk([&](mlir::Operation* op) {
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op))
      markWrite(st.getMemref());
    else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(op))
      markWrite(as.getLhs());
  });
  return written;
}

static bool globalIsWritten(const std::string& sym, mlir::ModuleOp module,
                            const std::set<std::string>* writtenGlobals = nullptr) {
  if (writtenGlobals) return writtenGlobals->count(sym) > 0;
  llvm::SmallVector<hlfir::DeclareOp, 4> decls;
  module.walk([&](hlfir::DeclareOp d) {
    if (traceToGlobalSymbol(d.getMemref()) == sym) decls.push_back(d);
  });
  if (decls.empty()) return false;
  auto writes = [&](mlir::Value dest) -> bool {
    for (auto d : decls) {
      if (dest == d.getResult(0) || dest == d.getResult(1)) return true;
      if (auto* dd = dest.getDefiningOp())
        if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(dd))
          if (designateRootedAt(dg, d)) return true;
    }
    return false;
  };
  bool written = false;
  module.walk([&](mlir::Operation* op) {
    if (written) return;
    if (auto st = mlir::dyn_cast<fir::StoreOp>(op)) {
      if (writes(st.getMemref())) written = true;
    } else if (auto as = mlir::dyn_cast<hlfir::AssignOp>(op)) {
      if (writes(as.getLhs())) written = true;
    }
  });
  return written;
}

// ---------------------------------------------------------------------------
// Main extraction
// ---------------------------------------------------------------------------

std::vector<VarInfo> extractVariables(mlir::ModuleOp module,
                                      std::vector<ValueSymbol>* value_symbols) {
  std::vector<VarInfo> vars;

  // Build, with one module walk each, the indices the per-variable passes below
  // would otherwise rebuild by re-walking the whole module (or function) once
  // per variable -- the O(variables x module) cost that dominates extraction of
  // a fully-inlined whole-program entry.  Extraction does not mutate the IR, so
  // these stay valid for every lookup below.
  const AllocSitesIndex allocIdx = buildAllocSitesIndex(module);
  const std::set<std::string> writtenGlobals = buildWrittenGlobals(module);
  const std::set<std::string> allocatedReaderNames =
      buildAllocatedReaderNames(module);

  // Reset thread-local extractName-override map.  Pass 3's view-alias
  // detection below populates it for inlined-callee declares whose
  // short name collides with their view source's; without a clean
  // reset the previous module's overrides would leak into this one.
  clearManglingOverrides();

  // Pass 0: disambiguate inlined-callee locals.  Two procedures with the
  // same variable name (Fortran's auto-generated ``a`` for ``result(a)``,
  // or simply two routines that both name a local ``dz``) get inlined into
  // a common parent and surface as two ``hlfir.declare`` ops with different
  // full uniq_names but the same ``extractName`` short name.  Downstream
  // code keys SDFG arrays / scalars by the short name; without
  // disambiguation the two declares race on one access node.  Rewrite each
  // colliding inlined-callee declare's uniq_name to encode its source scope.
  {
    auto getFScope = [](llvm::StringRef un) -> std::string {
      auto eP = un.rfind('E');
      if (eP == llvm::StringRef::npos) return {};
      auto fP = un.rfind('F', eP);
      if (fP == llvm::StringRef::npos || fP + 1 >= eP) return {};
      return un.substr(fP + 1, eP - fP - 1).str();
    };
    // True iff ``op``'s memref is the variable's OWN storage rather than an
    // alias of storage named elsewhere: a fresh ``fir.alloca`` /
    // ``fir.allocmem`` local, or a ``func.func`` entry block argument (the
    // entry kernel's own dummy).  Aliases (declare-of-declare, embox /
    // convert / box_addr chains, ``fir.absent``-backed optional dummies)
    // resolve through ``traceToDecl`` to their source and never mint an
    // array under their own short name, so they don't participate in
    // collisions.
    auto isOwnStorage = [](hlfir::DeclareOp op) -> bool {
      mlir::Value memref = op.getMemref();
      if (auto* def = memref.getDefiningOp())
        return mlir::isa<fir::AllocaOp, fir::AllocMemOp>(def);
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(memref))
        return mlir::isa_and_nonnull<mlir::func::FuncOp>(
            ba.getOwner()->getParentOp());
      return false;
    };
    // Distinct F-scopes that hold an OWN-STORAGE declare of each short
    // name.  A short name owned by more than one scope is the
    // inlined-callee collision shape  --  including a callee local that
    // shadows an entry dummy of the same name (``implicit_fall``'s local
    // ``dz`` vs the kernel's ``intent(in) dz`` block argument).  A
    // pass-through alias (``ze`` received by an inlined callee) is NOT
    // own storage, so a single genuine local keeps its bare name.
    llvm::StringMap<llvm::StringSet<>> ownStorageScopes;
    module.walk([&](hlfir::DeclareOp op) {
      auto* fn = op->getParentOfType<mlir::func::FuncOp>().getOperation();
      if (auto f = mlir::dyn_cast_or_null<mlir::func::FuncOp>(fn))
        if (f.isPrivate()) return;
      if (!isOwnStorage(op)) return;
      auto un = op.getUniqName().str();
      ownStorageScopes[extractName(un)].insert(getFScope(un));
    });
    // Rename candidates: ``fir.alloca``-backed locals only.  Block-arg
    // dummies are own storage too (they drive collision detection above)
    // but renaming one mints a phantom flat scalar that extract_vars would
    // surface as a top-level program kwarg, so they keep their name; they
    // are always entry-scope here and skipped by the ``entryScope`` guard
    // below regardless.
    llvm::StringMap<llvm::SmallVector<hlfir::DeclareOp, 2>> byShort;
    module.walk([&](hlfir::DeclareOp op) {
      auto* fn = op->getParentOfType<mlir::func::FuncOp>().getOperation();
      if (auto f = mlir::dyn_cast_or_null<mlir::func::FuncOp>(fn))
        if (f.isPrivate()) return;
      auto* def = op.getMemref().getDefiningOp();
      if (!def || !mlir::isa<fir::AllocaOp>(def)) return;
      byShort[extractName(op.getUniqName().str())].push_back(op);
    });
    // Entry's F-scope: the single public ``func.func`` left in the module
    // (set_entry_symbol made every other function private).  Symbol like
    // ``_QPmain`` / ``_QMmodPname``  --  the name segment is everything
    // after the last ``P``, matching ``getFScope``'s F-segment.  Entry
    // declares keep their original short name; inlined-callee siblings get
    // ``<callee_scope>_<short>``.
    std::string entryScope;
    for (auto fn : module.getOps<mlir::func::FuncOp>()) {
      if (fn.isPrivate()) continue;
      auto sn = fn.getSymName().str();
      auto pPos = sn.rfind('P');
      if (pPos == std::string::npos) continue;
      entryScope = sn.substr(pPos + 1);
      break;
    }
    for (auto& kv : byShort) {
      // Only rename a short name owned by more than one procedure.  A
      // single-owner name (the common case, including same-scope
      // shape-hint duplicate declares) is unambiguous; leave it alone so
      // extract_vars dedup downstream handles it.
      if (ownStorageScopes[kv.getKey()].size() < 2) continue;
      for (auto op : kv.second) {
        auto un = op.getUniqName().str();
        std::string scope = getFScope(un);
        if (scope == entryScope) continue;  // keep entry's name
        auto eP = un.rfind('E');
        std::string shortNm = un.substr(eP + 1);
        std::string newShort = scope + "_" + shortNm;
        std::string newUniq = un.substr(0, eP + 1) + newShort;
        op->setAttr("uniq_name",
                    mlir::StringAttr::get(op.getContext(), newUniq));
      }
    }
  }

  // Pass 0b: disambiguate multi-callsite duplicates of the same
  // inlined callee, but ONLY when the inlined dummy is backed by a
  // section-slice memref chain (``fir.convert`` of ``fir.box_addr``
  // of ``hlfir.designate``).  Whole-array pass-through aliases trace
  // through the convert chain back to the caller's own declare /
  // block-arg, and the bridge's downstream alias chain handles
  // per-callsite disambiguation correctly without renaming those.
  // Section-slice aliases instead get a fresh box-of-the-slice per
  // call site, so the bridge's view_subset / view_source machinery
  // needs distinct VarInfo entries to keep per-site slice
  // information from collapsing.
  {
    auto leadsToDesignate = [](mlir::Value v) -> bool {
      for (int i = 0; i < limits::kSsaBackWalkDepth && v; ++i) {
        auto* d = v.getDefiningOp();
        if (!d) return false;
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
          v = cv.getValue();
          continue;
        }
        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) {
          v = ba.getVal();
          continue;
        }
        if (mlir::isa<hlfir::DesignateOp>(d)) return true;
        return false;
      }
      return false;
    };
    llvm::StringMap<llvm::SmallVector<hlfir::DeclareOp, 4>> byUniq;
    module.walk([&](hlfir::DeclareOp op) {
      auto* fn = op->getParentOfType<mlir::func::FuncOp>().getOperation();
      if (auto f = mlir::dyn_cast_or_null<mlir::func::FuncOp>(fn))
        if (f.isPrivate()) return;
      if (!op.getDummyScope()) return;
      if (!leadsToDesignate(op.getMemref())) return;
      byUniq[op.getUniqName()].push_back(op);
    });
    for (auto& kv : byUniq) {
      auto& group = kv.second;
      if (group.size() < 2) continue;
      llvm::SmallPtrSet<mlir::Operation*, 4> scopes;
      for (auto op : group)
        if (auto ds = op.getDummyScope().getDefiningOp()) scopes.insert(ds);
      if (scopes.size() < 2) continue;
      unsigned idx = 0;
      for (auto op : group) {
        auto un = op.getUniqName().str();
        std::string newUniq = un + "_call" + std::to_string(idx++);
        op->setAttr("uniq_name",
                    mlir::StringAttr::get(op.getContext(), newUniq));
      }
    }
  }

  // Pass 1: collect every hlfir.declare.  Skip assumed-shape alias
  // declares inserted by ``hlfir-inline-all``  --  they share storage
  // with the caller's outer declare, and downstream SDFG emission
  // routes accesses to the outer name via traceToDecl.  Registering
  // both would give DaCe two non-transient arrays over one buffer.
  std::vector<hlfir::DeclareOp> decls;
  module.walk([&](hlfir::DeclareOp op) {
    // Skip declares inside private functions.  The bridge only
    // builds an SDFG for the single public entry; callees that
    // were already inlined into it leave behind their original
    // bodies as private siblings (kept alive only by a
    // dispatch_table after ``fir-polymorphic-op`` resolved the
    // callsites).  Their dummy declares  --  typed e.g.
    // ``fir.class<T>``  --  would otherwise surface as phantom
    // top-level program args at SDFG-build time.
    auto* parentOp = op->getParentOfType<mlir::func::FuncOp>().getOperation();
    if (auto fn = mlir::dyn_cast_or_null<mlir::func::FuncOp>(parentOp))
      if (fn.isPrivate()) return;
    if (asAssumedShapeAlias(op)) return;
    // Skip Flang-synthesised array-constructor temporaries
    // (``.tmp.arrayctor`` etc.) -- those are heap-allocated buffers
    // that ``dispatch.cpp`` recognises and lowers via per-element
    // assigns to the user's destination.  Registering them here
    // would surface ``.tmp.arrayctor`` on the SDFG and downstream
    // memlet parsing rejects the dotted name.
    if (op.getUniqName().str().find(".tmp.") != std::string::npos) return;
    // Skip Flang-internal type-info metadata declares  --  these are
    // string descriptors emitted for every derived type and its
    // components (``.n.<typename>``, ``.n.<field>``, ``.b.<type>``,
    // ``.di.<type>``).  They never represent user variables and
    // their dotted names break DaCe's ``NestedDict`` (which
    // interprets dots as nested-key separators).  Filter once
    // here so the rest of the pipeline never sees them.
    {
      auto un = op.getUniqName().str();
      auto p = un.rfind('E');
      llvm::StringRef tail = (p != std::string::npos)
                                 ? llvm::StringRef(un).drop_front(p + 1)
                                 : llvm::StringRef(un);
      if (tail.starts_with(".n.") || tail.starts_with(".b.") ||
          tail.starts_with(".di.") || tail.starts_with(".dt."))
        return;
    }
    // Drop unused SCALAR dummy arguments.  A subroutine like
    // ``subroutine main(arg1, arg2, res1) ; res1 = exp(arg1)``
    // (verbatim-port test pattern) leaves ``arg2`` declared but
    // never read or written; ``hlfir-default-intent`` adds
    // ``intent_inout`` to every dummy, so a "drop only if no
    // explicit intent" guard would keep ``arg2`` and the SDFG
    // signature would break Python callers that (correctly)
    // didn't pass it.
    //
    // Restrict the filter to *scalar* dummies (and to dummies
    // whose declare result has rank 0).  Arrays are kept
    // unconditionally even when ``size(a)``-style references
    // get folded by ``hlfir-propagate-shapes``: the array dummy
    // may be the sole carrier of shape symbols for other dummies
    // (``a(n, m)`` where ``m`` is used as an SDFG symbol via
    // ``a``'s extent), and dropping ``a`` breaks the symbol
    // classification cascade.
    auto resTy = peelTypeLayers(op.getResult(0).getType());
    bool isArrayLike = mlir::isa<fir::SequenceType>(resTy);
    if (op.getDummyScope() && !isArrayLike && op.getResult(0).use_empty() &&
        op.getResult(1).use_empty()) {
      return;
    }
    decls.push_back(op);
  });

  // Pass 2a: loop iterators.  A Fortran DO induction variable is
  // always a symbol downstream  --  the LoopRegion uses it as
  // ``loop_var`` in its init / update / condition expressions, and
  // any ``a(i)`` body uses it as an index (which only symbols may
  // be).  Add to symbolNames directly; there's no reason to keep a
  // separate ``loop_iter`` role when every consumer wants ``symbol``
  // semantics.
  std::set<std::string> symbolNames;
  module.walk([&](fir::DoLoopOp lp) {
    auto n = traceLoopIter(lp);
    if (!n.empty()) symbolNames.insert(n);
  });

  // Pass 2b: shape symbols + do-loop bounds (both lower and upper).
  // Lower bounds are promoted symmetrically with upper bounds so
  // ``DO jk = nflatlev, nlev`` recognises ``nflatlev`` as a symbol  --
  // otherwise codegen generates an int*-vs-int64_t mismatch in the
  // loop initialiser.
  for (auto& op : decls) {
    for (auto& s : resolveShapeSyms(op)) {
      if (s == "?") continue;
      // Bare-name results (single declare name / integer literal)
      // get inserted directly.  Expression-string results (from
      // ``traceExtentExpr`` -- a dynamic gather-temp extent like
      // ``"max((endcol - startcol) + 1, 0)"``) contain operators;
      // insert the leaf scalar declares instead via the shape
      // SSA walker below.
      if (s.find_first_of("+-*/()") == std::string::npos) symbolNames.insert(s);
    }
    // Walk the shape SSA chain directly to promote every scalar
    // leaf referenced in a closed-form extent expression
    // (``traceExtentExpr`` resolves these for the descriptor; the
    // leaves must be SDFG symbols for the expression to compile).
    for (auto ext : classifyShapeOperand(op.getShape()).extents)
      collectExtentExprScalars(ext, symbolNames);
  }
  module.walk([&](fir::DoLoopOp lp) {
    auto ub = traceToDecl(lp.getUpperBound());
    if (!ub.empty()) symbolNames.insert(ub);
    auto lb = traceToDecl(lp.getLowerBound());
    if (!lb.empty()) symbolNames.insert(lb);
  });
  // Allocatable shape sources: every ``fir.allocmem`` site's shape
  // operands are runtime extents of the resulting array  --  promote
  // their traced declares to symbols so ``allocate(x(n))``
  // (without any surrounding do-loop) still flips ``n`` from scalar
  // to symbol.  Bug fix for Phase 5a (allocatable struct members):
  // ``s%w`` allocates only at the explicit ``allocate(s%w(n))``
  // statement and may have no other use of ``n``, so neither
  // ``resolveShapeSyms`` (declare has no shape) nor the do-loop
  // pass picks up ``n``.  Without this walk, ``n`` lands as a
  // ``scalar`` data-descriptor and collides with the symbol the
  // SDFG construction step then tries to emit for the array
  // extent.
  module.walk([&](fir::AllocMemOp am) {
    for (auto sz : am.getShape()) {
      auto n = traceToDecl(sz);
      if (!n.empty()) symbolNames.insert(n);
    }
  });

  // Pass 2c: scalars used as array indices (``a(i)``) are also symbols.
  // Catches the DO-with-EXIT / DO-WHILE shape where lift-cf-to-scf
  // removed the fir.do_loop that pass 2a would otherwise trace, plus
  // any index-only scalar the user declares by hand.  Writing to a
  // symbol then routes through the interstate-edge path in
  // _emit_assign, which is the state-change DaCe needs to keep the
  // index value live across loop iterations.
  module.walk([&](hlfir::DesignateOp dg) {
    auto operands = dg.getIndices();
    auto triplets = dg.getIsTriplet();
    if (triplets.empty()) {
      // Plain scalar-indices: recurse through each operand so every
      // declared scalar leaf is promoted, including ones hidden behind
      // index arithmetic (``a(base + jh)`` -> ``base``).  A bare
      // ``traceToDecl`` would stop at the ``arith.addi`` and miss
      // ``base``, leaving it a transient scalar that the memlet subset
      // then references by name -- which DaCe can't allocate across the
      // states that read it.
      for (auto idx : operands) collectIntegerScalarReads(idx, symbolNames);
      return;
    }
    // Triplet-aware walk: each true entry in ``triplets`` consumes
    // three operands (lb, ub, step), each false entry consumes one
    // (scalar index).  Promote the lb and ub of every triplet so
    // Flang's ``ub - lb + 1`` extent expression on a gather temp's
    // shape can resolve to a closed-form symbol expression in
    // ``resolveShapeSyms`` / ``traceExtentExpr``.  The step is
    // almost always literal-``1`` and harmless to skip.
    unsigned cursor = 0;
    for (unsigned d = 0; d < triplets.size(); ++d) {
      if (triplets[d]) {
        for (unsigned k = 0; k < 2 && cursor + k < operands.size(); ++k)
          collectIntegerScalarReads(operands[cursor + k], symbolNames);
        cursor += 3;
      } else {
        if (cursor < operands.size())
          collectIntegerScalarReads(operands[cursor], symbolNames);
        cursor += 1;
      }
    }
  });

  // Pass 2d: scalars read by any control-flow condition are also
  // symbols.  Principle: loop variables, while-loop counters and
  // if-branch guards all go through the symbol / interstate-edge
  // write path so DaCe's condition evaluators see every update.
  // Without this, ``DO WHILE (i < n)`` reads the scalar's initial
  // zero-init and the loop body never runs.
  module.walk([&](mlir::scf::IfOp ifOp) {
    collectIntegerScalarReads(ifOp.getCondition(), symbolNames);
  });
  module.walk([&](fir::IfOp ifOp) {
    collectIntegerScalarReads(ifOp.getCondition(), symbolNames);
  });
  module.walk([&](mlir::scf::ConditionOp condOp) {
    collectIntegerScalarReads(condOp.getCondition(), symbolNames);
  });

  // Per-function store/assign index, built lazily once per function and shared
  // across all declares (see ``funcWritesFor``); without it the lower-bound
  // inference below re-walked the whole function per loaded designate index.
  std::map<mlir::Operation*, FuncWrites> writeCache;

  // Group every designate under each declare in its root chain, once.  The
  // chain matches ``designateRootedAt`` exactly, so a declare's bucket is the
  // set of designates ``designateRootedAt`` would have matched -- but the walk
  // happens once for the whole module instead of once per declare
  // (O(declares x designates) -> O(designates) + a lookup per declare).
  llvm::DenseMap<mlir::Operation*, llvm::SmallVector<hlfir::DesignateOp, 4>>
      designatesByDecl;
  module.walk([&](hlfir::DesignateOp dg) {
    mlir::Value v = dg.getMemref();
    for (int i = 0; i < limits::kSsaBackWalkDepth && v; ++i) {
      auto* d = v.getDefiningOp();
      if (!d) break;
      if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(d)) {
        designatesByDecl[dc.getOperation()].push_back(dg);
        v = dc.getMemref();
        continue;
      }
      if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) { v = cv.getValue(); continue; }
      if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) { v = ld.getMemref(); continue; }
      if (auto rb = mlir::dyn_cast<fir::ReboxOp>(d)) { v = rb.getBox(); continue; }
      if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(d)) { v = ba.getVal(); continue; }
      if (auto co = mlir::dyn_cast<fir::CoordinateOp>(d)) { v = co.getRef(); continue; }
      if (auto inner = mlir::dyn_cast<hlfir::DesignateOp>(d)) {
        v = inner.getMemref();
        continue;
      }
      break;
    }
  });

  // Pass 3: build one VarInfo per declare.
  for (auto& op : decls) {
    VarInfo v;
    v.mangled_name = op.getUniqName().str();
    v.fortran_name = extractName(v.mangled_name);

    // Intent
    if (auto a = op.getFortranAttrs()) {
      auto fa = *a;
      if (bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::intent_inout))
        v.intent = "inout";
      else if (bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::intent_in))
        v.intent = "in";
      else if (bitEnumContainsAny(fa,
                                  fir::FortranVariableFlagsEnum::intent_out))
        v.intent = "out";
      // An OPTIONAL dummy without an explicit intent is still a
      // dummy -- treat it as ``intent(in)`` by default so
      // descriptors.py doesn't misclassify it as a transient
      // local.  The Fortran spec allows any intent for an
      // unspecified OPTIONAL; ``in`` is the common case (and
      // widens safely to ``inout`` via the caller's own buffer).
      if (v.intent.empty() &&
          bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::optional))
        v.intent = "in";
      // ``REAL(8), VALUE :: x`` is a C-interop scalar passed by
      // value -- equivalent to intent(in) since the callee gets
      // its own copy.  Mark intent so the rank-0 path doesn't
      // misclassify it as a transient.  Below (after the
      // role-classification block) we further promote VALUE
      // scalars to SDFG SYMBOLS so callers can bind them with
      // plain Python int / float instead of a 1-element numpy
      // array.
      if (v.intent.empty() &&
          bitEnumContainsAny(fa, fir::FortranVariableFlagsEnum::value))
        v.intent = "in";
    }

    // Unwrap FIR type wrappers to find element type + rank.
    //
    // Plain dummy / local arrays surface a single layer (Box, Ref,
    // Heap, or Ptr) over the SequenceType, so a single sequential
    // unwrap suffices.  Allocatable declares add two extra layers
    // (``ref<box<heap<array<...>>>>``); loop through the wrappers
    // only when the declare is allocatable so POINTER and other
    // box-typed dummies stay rank-0 (scalar passthrough).
    auto ty = op.getResult(0).getType();
    bool isAllocatableAttr = false;
    bool isPointerAttr = false;
    if (auto a = op.getFortranAttrs()) {
      if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::allocatable))
        isAllocatableAttr = true;
      if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::pointer))
        isPointerAttr = true;
    }
    // Pointer declares peel the same way as allocatables: their
    // declared type is ``ref<box<ptr<array<?xT>>>>`` and we want
    // the inner array's element type + rank for the SDFG
    // descriptor.  Without peeling, a top-level ``real, pointer
    // :: w(:)`` (or a Phase 5b flat companion ``s_w`` for a
    // pointer struct member) ends up classified as a scalar of
    // dtype ``!fir.box<!fir.ptr<...>>``  --  useless to the SDFG.
    //
    // Guard: only peel pointer declares whose results are
    // actually USED downstream.  Pointer declares with all-empty
    // results survive ``hlfir-rewrite-pointer-assigns`` only as
    // dangling artifacts (rebind successfully collapsed -> all
    // reads forwarded -> declare is dead but not yet erased) or
    // as cross-procedure / unsupported-target leftovers.  Peel
    // always exposes a phantom rank>0 array on the SDFG
    // signature; without the guard, even a successfully-collapsed
    // pointer demanded its own ``_d0`` symbols.
    bool peelPointer = false;
    if (isPointerAttr) {
      if (!op.getResult(0).use_empty() || !op.getResult(1).use_empty())
        peelPointer = true;
    }
    if (isAllocatableAttr || peelPointer) {
      ty = peelTypeLayers(ty);
    } else {
      // Single B->R->H->P sweep (one level each)  --  deliberately NOT
      // peel-to-fixpoint: a plain dummy keeps any inner wrapper.
      if (auto b = mlir::dyn_cast<fir::BoxType>(ty)) ty = b.getEleTy();
      if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) ty = r.getEleTy();
      if (auto h = mlir::dyn_cast<fir::HeapType>(ty)) ty = h.getEleTy();
      if (auto p = mlir::dyn_cast<fir::PointerType>(ty)) ty = p.getEleTy();
    }
    // Capture the SequenceType's per-dim extents as a fallback for
    // ``shape_symbols``: a declare synthesised by ``hlfir-flatten-structs``
    // for a per-field array carries the shape only in the type
    // (``!fir.array<5x5x5xf32>``), not as an explicit ``fir.shape``
    // operand.  Without this, ``resolveShapeSyms`` returns empty
    // and the fallback assumed-shape ``<name>_d<i>`` synth fires  --
    // but those synth symbols would be unwired because the extent
    // is statically known.
    std::vector<std::string> seqExtents;
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty)) {
      for (auto d : seq.getShape()) {
        if (d == fir::SequenceType::getUnknownExtent()) {
          v.is_dynamic = true;
          seqExtents.push_back("?");
        } else {
          seqExtents.push_back(std::to_string(d));
        }
      }
      v.rank = seq.getShape().size();
      ty = seq.getEleTy();
    }

    // Element type string.
    if (ty.isF64())
      v.dtype = "float64";
    else if (ty.isF32())
      v.dtype = "float32";
    else if (ty.isInteger(8))
      v.dtype = "int8";  // Fortran INTEGER(1)
    else if (ty.isInteger(16))
      v.dtype = "int16";  // Fortran INTEGER(2)
    else if (ty.isInteger(32))
      v.dtype = "int32";
    else if (ty.isInteger(64))
      v.dtype = "int64";
    // Fortran ``COMPLEX(kind)`` lowers to ``mlir::ComplexType`` over
    // an ``f32`` / ``f64`` element.  DaCe has native ``complex64`` /
    // ``complex128`` dtypes that match numpy's ABI.
    else if (auto ct = mlir::dyn_cast<mlir::ComplexType>(ty)) {
      auto et = ct.getElementType();
      if (et.isF32())
        v.dtype = "complex64";
      else if (et.isF64())
        v.dtype = "complex128";
      else {
        std::string s;
        llvm::raw_string_ostream os(s);
        ty.print(os);
        v.dtype = s;
      }
    }
    // MLIR ``i1`` and Fortran ``LOGICAL(KIND=N)`` (any kind) both
    // surface as ``bool`` on the SDFG signature (= ``np.bool_`` =
    // C++ ``bool``, 1 byte).  Element-wise boolean ops in tasklets
    // render as ``bool`` operations directly  --  no ``(x != 0)``
    // truthiness coercion needed.  The caller-side bindings
    // wrapper translates between the original ``LOGICAL(KIND=N)``
    // image and the SDFG's bool layout at the Fortran boundary.
    else if (ty.isInteger(1))
      v.dtype = "bool";
    else if (mlir::isa<fir::LogicalType>(ty)) {
      v.dtype = "bool";
    } else if (mlir::isa<fir::RecordType>(ty)) {
      // ``fir.RecordType`` declares fall into three categories:
      //
      //   1. Flang-internal type-info metadata
      //      (``_QM__fortran_type_info...`` tables, component
      //      descriptors named ``.b.<type>.<field>``)  --  never
      //      user-visible.  Drop them.
      //   2. DUMMY-arg struct that ``hlfir-flatten-structs``
      //      already lowered to per-field declares -- this
      //      original struct declare's designates have been
      //      replaced.  Drop the leftover.
      //   3. MODULE-LEVEL struct global (``type(t) :: g`` at
      //      module scope) -- the flatten pass does NOT process
      //      these (it only walks dummy args).  Field accesses
      //      (``g % a``) reach the bridge as
      //      ``hlfir.designate %g_decl{"a"}``; ``traceToDecl``
      //      returns the flat ``g_a`` name (the bridge's
      //      ``<parent>_<member>`` convention).  Without a
      //      matching SDFG array the libcall dispatcher raises
      //      ``KeyError: 'g_a'``.
      //
      // Fix for category 3: synthesise one per-field VarInfo
      // per UNIQUE (struct, field) pair actually accessed,
      // marked as a TRANSIENT (no signature exposure -- module
      // globals are internal state).  The per-field VarInfo's
      // type and shape come from the struct's member type; the
      // bindings layer can either bake the initial-value
      // ``fir.global`` contents OR copy from the host on entry.
      //
      // Discriminator: a module-scope global declare's memref
      // traces back to ``fir.address_of @_QM<m>E<n>`` (or the
      // ``_QM<m>F<f>E<n>`` SAVE-local form, which we deliberately
      // include too -- SAVE-locals are persistent state with the
      // same access pattern).  Type-info tables (category 1) and
      // dummy structs (category 2) DON'T have the address_of
      // trace, so this branch fires only on real category-3
      // shapes.
      auto rec = mlir::cast<fir::RecordType>(ty);
      std::string globalSym = traceToGlobalSymbol(op.getMemref());
      auto designates_it = designatesByDecl.find(op.getOperation());
      bool hasFieldUses = (designates_it != designatesByDecl.end() &&
                           !designates_it->second.empty());
      if (!globalSym.empty() && hasFieldUses) {
        // Recursive emit: walk the (struct, field) chain, generating
        // one VarInfo per LEAF field whose type is a supported
        // scalar / array.  For nested records (``g % inner % a``)
        // the inner field designate's USERS include the further
        // ``a`` designate, so we recurse through ``designatesByDecl``
        // / direct users on the SSA result to discover deeper
        // levels.  Tracks ``visitedKey`` to keep recursion
        // bounded for cyclic or self-referential type structures.
        //
        // Type-to-dtype mapping is shared with the top-level path
        // via the lambda below; non-supported leaf types
        // (CharacterType, PointerType, allocatable boxes, ...)
        // are skipped silently and the downstream traceToDecl
        // lookup will fail loudly if a kernel actually reads the
        // unsupported leaf.
        auto dtypeFor = [](mlir::Type elemTy,
                           std::string& outDtype) -> bool {
          if (auto fty = mlir::dyn_cast<mlir::FloatType>(elemTy)) {
            unsigned w = fty.getWidth();
            outDtype = (w == 32) ? "fp32"
                                 : (w == 64) ? "fp64" : "fp" + std::to_string(w);
            return true;
          }
          if (auto ity = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
            unsigned w = ity.getWidth();
            outDtype = (w == 8) ? "i8"
                                : (w == 16) ? "i16"
                                            : (w == 32) ? "i32"
                                                        : (w == 64) ? "i64" : "i" + std::to_string(w);
            return true;
          }
          if (mlir::isa<fir::LogicalType>(elemTy) || elemTy.isInteger(1)) {
            outDtype = "bool";
            return true;
          }
          return false;
        };
        std::set<std::string> emittedFlatNames;
        std::function<void(mlir::Value, fir::RecordType,
                            const std::string&, const std::string&, int)>
            walkLevel = [&](mlir::Value designateResult,
                            fir::RecordType levelRec,
                            const std::string& flatNameBase,
                            const std::string& mangledBase, int depth) {
              if (depth > 8) return;  // bounded recursion
              // Collect user designates of designateResult that
              // carry a component attribute -- each one is one
              // more level of nesting.
              std::set<std::string> componentsSeen;
              for (auto* u : designateResult.getUsers()) {
                auto childDg =
                    mlir::dyn_cast_or_null<hlfir::DesignateOp>(u);
                if (!childDg) continue;
                auto childComp = childDg.getComponentAttr();
                if (!childComp) continue;
                std::string childName = childComp.getValue().str();
                if (!componentsSeen.insert(childName).second) continue;
                mlir::Type childMemberTy;
                for (auto& p : levelRec.getTypeList())
                  if (p.first == childName) {
                    childMemberTy = p.second;
                    break;
                  }
                if (!childMemberTy) continue;
                std::string newFlat =
                    flatNameBase + "_" + childName;
                std::string newMangled =
                    mangledBase + "_" + childName;
                // Nested record: recurse into ITS designates.
                if (auto childRec = mlir::dyn_cast<fir::RecordType>(
                        childMemberTy)) {
                  walkLevel(childDg.getResult(), childRec, newFlat,
                            newMangled, depth + 1);
                  continue;
                }
                // Leaf -- emit a VarInfo if the dtype is
                // supported.  Memoise on the flat name so two
                // independent leaf-access sites collapse to one.
                if (!emittedFlatNames.insert(newFlat).second)
                  continue;
                VarInfo mv;
                mv.fortran_name = newFlat;
                mv.mangled_name = newMangled;
                mv.intent = "";
                mlir::Type elemTy = childMemberTy;
                if (auto seq = mlir::dyn_cast<fir::SequenceType>(
                        childMemberTy)) {
                  elemTy = seq.getEleTy();
                  for (auto dimd : seq.getShape())
                    mv.shape_symbols.push_back(std::to_string(dimd));
                }
                mv.rank = mv.shape_symbols.size();
                mv.role = (mv.rank > 0) ? "array" : "scalar";
                for (size_t d = 0; d < mv.shape_symbols.size(); ++d)
                  mv.lower_bounds.push_back("1");
                std::string dtype;
                if (!dtypeFor(elemTy, dtype)) continue;
                mv.dtype = dtype;
                vars.push_back(std::move(mv));
              }
            };
        // Kick off the recursive walk from the struct declare's
        // own result (the FIRST level of designates).
        walkLevel(op.getResult(0), rec, v.fortran_name,
                  v.mangled_name, 0);
      }
      continue;
    } else {
      std::string s;
      llvm::raw_string_ostream os(s);
      ty.print(os);
      v.dtype = s;
    }

    v.shape_symbols = resolveShapeSyms(op, value_symbols);
    v.lower_bounds = resolveLowerBounds(op);

    // SequenceType-extent fallback: a declare with no ``fir.shape``
    // operand (e.g. one synthesised by ``hlfir-flatten-structs`` for
    // a per-field array) still carries concrete extents in its type.
    // Use them when ``resolveShapeSyms`` came back empty so the
    // SDFG signature gets literal shape (``[5,5,5]``) instead of a
    // free symbol per dim that the caller must bind manually.
    if (v.shape_symbols.empty() && !seqExtents.empty()) {
      v.shape_symbols = seqExtents;
      if (v.lower_bounds.size() != v.shape_symbols.size())
        v.lower_bounds.assign(v.shape_symbols.size(), "1");
    }

    // Allocatable: hlfir.declare has no shape; pull it from the
    // matching ``fir.allocmem`` site(s).  One ALLOCATE -> use the
    // first site for ``x``'s shape.  Multiple ALLOCATEs (re-
    // allocation across an explicit DEALLOCATE) -> register one
    // extra synthetic VarInfo per additional site, named
    // ``x_alloc1``, ``x_alloc2``, ... (allocAliasName); the bridge's
    // alias map (see extract_ast.cpp) will route per-site reads /
    // writes to the right transient at AST-build time.
    // Group allocatable + pointer here: both are descriptor-bearing and both
    // get an ``<arr>_allocated`` tracker (the emission at ast/expressions.cpp
    // and ast/control_flow.cpp returns ``<arr>_allocated`` for any
    // ``box_addr`` -- the lowering of ``ALLOCATED(arr)`` AND ``ASSOCIATED(ptr)``
    // -- so a pointer member without the tracker is referenced by emission but
    // never registered, surfacing as a sdfg.arglist KeyError later).
    bool isAllocatable = false;
    if (auto a = op.getFortranAttrs())
      if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::allocatable) ||
          bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::pointer))
        isAllocatable = true;
    std::vector<fir::AllocMemOp> allocSites;
    if (isAllocatable && v.rank > 0)
      allocSites = collectAllocSites(v.mangled_name, module, &allocIdx);
    // Partition the ALLOCATE sites into buffer classes (one DaCe transient
    // each): a class with >1 site is a conditional (mutually-exclusive
    // branches sharing one buffer with a branch-dependent extent symbol);
    // a singleton class is a plain / sequentially-versioned buffer.  The
    // base name ``a`` is class 0 (first definition); classes 1.. become
    // ``a_alloc1``, ``a_alloc2``, ...  See ALLOC_BUFFER_SSA_DESIGN.md.
    auto allocClasses = groupAllocSites(v.mangled_name, module, &allocIdx);
    // ``baseCondAlloc``: is the base buffer (class 0) a conditional?  If so
    // skip the front-site shape (it would pin ``a`` to one branch's extent)
    // -- the synthesize-``a_d<i>`` fallback gives the branch-symbol shape,
    // and the AST builder assigns it per branch.
    bool baseCondAlloc =
        !allocClasses.empty() && allocClasses.front().size() > 1;
    if (!baseCondAlloc && !allocSites.empty() &&
        (v.shape_symbols.empty() ||
         std::all_of(v.shape_symbols.begin(), v.shape_symbols.end(),
                     [](const std::string& s) { return s == "?"; }))) {
      auto from_alloc = shapeFromAllocSite(allocSites.front());
      if (!from_alloc.empty()) {
        v.shape_symbols = std::move(from_alloc);
        if (v.lower_bounds.size() != v.shape_symbols.size())
          v.lower_bounds.assign(v.shape_symbols.size(), "1");
      }
      // Authoritative lower-bound recovery: read the fir.shape_shift
      // paired with this allocmem in its consuming fir.embox.  This
      // captures the runtime ``ALLOCATE(arr(lb:ub))`` bounds even
      // when no literal index appears in the body.
      auto lb_from_alloc = lowerBoundsFromAllocSite(allocSites.front());
      for (size_t d = 0; d < lb_from_alloc.size() && d < v.lower_bounds.size();
           ++d) {
        if (lb_from_alloc[d] != "?") v.lower_bounds[d] = lb_from_alloc[d];
      }
    }

    // Assumed-shape fallback: synthesise per-dim symbol names.
    // Two entry shapes:
    //   * ``shape_symbols`` is empty entirely (no shape op on the
    //     declare)  --  synthesize all dims.
    //   * ``shape_symbols`` has per-dim ``"?"`` slots (an
    //     unknown-extent sentinel reached us, e.g. assumed-size
    //     ``arr(*)``)  --  replace just the unresolved slots, keep
    //     the resolved ones.
    if (v.shape_symbols.empty() && v.rank > 0)
      for (int dim = 0; dim < v.rank; ++dim)
        v.shape_symbols.push_back(v.fortran_name + "_d" + std::to_string(dim));
    else
      for (size_t dim = 0; dim < v.shape_symbols.size(); ++dim)
        if (v.shape_symbols[dim] == "?")
          v.shape_symbols[dim] = v.fortran_name + "_d" + std::to_string(dim);

    // Lower-bound inference from literal designate accesses.
    // Catches ICON's refined-cell-tag pattern (``end_block(-10)``
    // on a deferred-shape ALLOCATABLE) by walking every
    // ``hlfir.designate`` rooted at ``op``'s result and taking
    // the per-dim min of literal-integer indices.  No-op when
    // every observed literal index is >= 1.  See the inference
    // function's docstring for the full rationale.
    // The literal-access heuristic only recovers EXPLICIT negative
    // lower bounds on deferred-shape ALLOCATABLE/POINTER arrays
    // (ICON's ``end_block(min_rl:)`` pattern, where
    // ``resolveLowerBounds`` saw no shape op).  A plain
    // ``fir.ShapeOp`` declare -- an automatic / explicit-shape
    // local like ``ZQX(KLON,KLEV,NCLV)`` -- has Fortran-default
    // lower bound 1 in every dim, which ``resolveLowerBounds``
    // already returned authoritatively.  Running the heuristic
    // there lets a mis-traced non-literal subscript (a loop
    // induction var or a folded PARAMETER pulled through
    // ``traceConstIntThroughLoad``) poison a known-good bound --
    // observed as ``offset_zqx_d2 = -999`` from the mixed
    // ``ZQX(JL,JK,NCLDQV)`` / ``ZQX(JL,JK,JM)`` subscripts, which
    // turned the write subset into a wild out-of-bounds store.
    // Skip it for plain-ShapeOp declares; ShapeShiftOp (explicit
    // bounds) stays authoritative via the ``lbs[d] != "1"`` guard
    // inside the heuristic.
    std::vector<bool> seenLit;
    bool plainShapeOp = op.getShape() && mlir::isa_and_nonnull<fir::ShapeOp>(
                                             op.getShape().getDefiningOp());
    if (plainShapeOp)
      seenLit.assign(v.rank, false);
    else
      inferLowerBoundsFromLiteralAccesses(op, v.lower_bounds, v.rank, writeCache,
                                          designatesByDecl, &seenLit);

    // Dummy-arg deferred-shape ALLOCATABLE/POINTER fallback: the
    // declare is a function block-arg, its declared type has no
    // static shape, and the body has no literal-index designate
    // for some dim (purely symbolic access).  We can't see the
    // upstream ``ALLOCATE`` that set the bound -- it lives in
    // the caller.  Leave the per-dim offset as ``"?"`` so the
    // SDFG signature carries ``offset_<arr>_d<i>`` as a free
    // symbol; the caller (or the bindings emitter via
    // ``lbound(arr, dim=...)``) binds it at call time.
    //
    // Predicate gate (all must hold):
    //   * variable is rank > 0 (array, not scalar)
    //   * Fortran attr carries ALLOCATABLE or POINTER
    //   * declare's memref is a function block argument
    //   * no fir.ShapeOp / fir.ShapeShiftOp on the declare
    //     (resolveLowerBounds returned nothing)
    //   * literal-index inference saw no literals for that dim
    bool isDummyArg = false;
    if (auto blk = mlir::dyn_cast<mlir::BlockArgument>(op.getMemref())) {
      auto* parent = blk.getOwner()->getParentOp();
      if (mlir::isa_and_nonnull<mlir::func::FuncOp>(parent)) isDummyArg = true;
    }
    bool isAllocOrPointerAttr = false;
    if (auto a = op.getFortranAttrs()) {
      if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::allocatable) ||
          bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::pointer))
        isAllocOrPointerAttr = true;
    }
    bool declHasNoShape = (op.getShape() == nullptr);
    if (v.rank > 0 && isDummyArg && isAllocOrPointerAttr && declHasNoShape) {
      if ((int)v.lower_bounds.size() < v.rank)
        v.lower_bounds.resize(v.rank, "1");
      for (int d = 0; d < v.rank; ++d) {
        bool lit = (d < (int)seenLit.size()) && seenLit[d];
        if (!lit && v.lower_bounds[d] == "1") v.lower_bounds[d] = "?";
      }
    }

    // Classify.
    if (v.rank > 0)
      v.role = "array";
    else if (symbolNames.count(v.fortran_name))
      v.role = "symbol";
    else
      v.role = "scalar";

    // View-alias detection.  Fortran storage-association reshape  --
    // ``call cb(d(:, :, 1))`` where ``cb`` declares ``dd(16)``  --  has
    // Flang emit:
    //   %sec = hlfir.designate %d (1:4, 1:4, 1) shape <4,4>
    //   %flat = fir.convert %sec : ref<4x4xf64> -> ref<16xf64>
    //   %dd = hlfir.declare %flat ...
    // After ``hlfir-inline-all`` splices the callee's body in,
    // accesses to ``dd`` reach the bridge's AST walker with no
    // memlet linking ``dd`` to ``d``, so writes are dropped.
    // Detect the pattern here and surface the source + per-dim
    // subset; ``descriptors.py`` then stages copy-in / copy-out
    // states so writes round-trip through the alias.
    if (v.role == "array") {
      mlir::Value m = op.getMemref();
      // Peel through:
      //   * ``fir.convert``   --  same-type rebox or shape-changing
      //     reinterpret (Fortran storage-association reshape).
      //   * ``fir.box_addr``  --  extract a raw ref from a box.
      //   * ``hlfir.copy_in``  --  Flang's contiguous-buffer
      //     materialisation when a non-contiguous section is
      //     passed to a callee whose dummy is declared
      //     contiguous.  Treating the buffer as a view of the
      //     underlying section skips the copy and reverses
      //     ``hlfir.copy_out`` automatically (writes propagate
      //     through the view).
      for (int i = 0; i < limits::kSsaBackWalkDepth && m; ++i) {
        auto* def = m.getDefiningOp();
        if (!def) break;
        if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
          m = cv.getValue();
          continue;
        }
        if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(def)) {
          m = ba.getVal();
          continue;
        }
        if (auto cp = mlir::dyn_cast<hlfir::CopyInOp>(def)) {
          m = cp.getVar();
          continue;
        }
        break;
      }
      if (auto* defOp = m.getDefiningOp()) {
        if (auto sec = mlir::dyn_cast<hlfir::DesignateOp>(defOp)) {
          auto srcName = traceToDecl(sec.getMemref());
          auto triplets = sec.getIsTriplet();
          auto secIdx = sec.getIndices();
          if (!srcName.empty() && !triplets.empty()) {
            // Walk the section's per-dim spec.  For each
            // parent dim: triplet -> 3 operands (lo, hi,
            // stride) collapsed to ``"lo-1:hi"`` DaCe form
            // (or ``"lo-1:hi:stride"`` if stride != 1  --  the
            // non-contiguous slice variant Flang lowers
            // ``a(1:7:2)`` to);  scalar -> 1 operand
            // collapsed to ``"k-1"``.  When a bound is a
            // runtime value (loop iter, dummy scalar, ...)
            // fall back to a small symbol renderer so the
            // subset stays expressible.
            auto renderSym = [](mlir::Value v) -> std::string {
              for (int i = 0; i < limits::kSsaBackWalkDepth && v; ++i) {
                auto* d = v.getDefiningOp();
                if (!d) return "";
                if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
                  v = cv.getValue();
                  continue;
                }
                if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
                  auto n = traceToDecl(ld.getMemref());
                  return n;
                }
                if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d)) {
                  if (auto ia =
                          mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
                    return std::to_string(ia.getInt());
                  return "";
                }
                return "";
              }
              return "";
            };
            auto renderBound = [&](mlir::Value v) -> std::string {
              if (auto c = traceConstInt(v)) return std::to_string(*c);
              auto s = renderSym(v);
              if (!s.empty()) return s;
              // Runtime triplet upper bound that goes through
              // Flang's ``max(ext, 0)`` clamp (assumed-shape /
              // explicit-shape with a dummy extent).
              // ``traceExtentExpr`` already recognises this
              // shape and renders to the underlying dummy
              // scalar's short name.
              return traceExtentExpr(v);
            };
            std::vector<std::string> subset;
            // Parallel walk: build ``view_dim_map`` alongside
            // ``subset``.  dim_map[d] is either ``"_d<N>"``
            // (surviving triplet dim with N = 0-based dummy-
            // dim index) or a 0-based scalar index expression
            // (dropped scalar dim).  ``is_trivial_section``
            // tracks whether the section is just a name +
            // index-suffix alias (every triplet has lo=1,
            // stride=1)  --  for those we route accesses through
            // the source array instead of registering a view.
            std::vector<std::string> dim_map;
            bool is_trivial_section = !triplets.empty();
            unsigned surviving = 0;
            unsigned cursor = 0;
            for (unsigned d = 0; d < triplets.size(); ++d) {
              if (triplets[d] && cursor + 2 < secIdx.size()) {
                std::string lo = renderBound(secIdx[cursor]);
                std::string hi = renderBound(secIdx[cursor + 1]);
                std::string st = renderBound(secIdx[cursor + 2]);
                auto loC = traceConstInt(secIdx[cursor]);
                auto stC = traceConstInt(secIdx[cursor + 2]);
                bool is_full = (loC && *loC == 1) && (stC && *stC == 1);
                if (!is_full) is_trivial_section = false;
                if (!lo.empty() && !hi.empty()) {
                  // DaCe subset uses ``lo-1:hi`` (0-based,
                  // inclusive upper).  Literal ``-1`` for
                  // constant lo; ``(lo)-1`` for symbolic.
                  std::string s;
                  if (loC)
                    s = std::to_string(*loC - 1);
                  else
                    s = "(" + lo + ")-1";
                  s += ":" + hi;
                  if (!st.empty() && st != "1") s += ":" + st;
                  subset.push_back(std::move(s));
                } else {
                  subset.push_back("0:?");
                  is_trivial_section = false;
                }
                dim_map.push_back("_d" + std::to_string(surviving++));
                cursor += 3;
              } else if (!triplets[d] && cursor < secIdx.size()) {
                std::string k = renderBound(secIdx[cursor]);
                if (!k.empty()) {
                  // ``subset`` stays in 0-based DaCe
                  // form for the view_alias path
                  // ("(k)-1" / literal-minus-one).
                  std::string zero_based;
                  if (auto c = traceConstInt(secIdx[cursor]))
                    zero_based = std::to_string(*c - 1);
                  else
                    zero_based = "(" + k + ")-1";
                  subset.push_back(zero_based);
                  // ``dim_map`` stays 1-based  --  it's
                  // spliced into index_exprs which
                  // build_memlet_index offsets uniformly.
                  dim_map.push_back(k);
                } else {
                  subset.push_back("?");
                  dim_map.push_back("");
                  is_trivial_section = false;
                }
                cursor += 1;
              }
            }
            // Only mark as view_alias when every dim's subset
            // resolved to a closed-form expression; bail on
            // ``?`` entries so we don't emit broken memlets.
            bool allOk = !subset.empty();
            for (auto& s : subset)
              if (s.find('?') != std::string::npos) {
                allOk = false;
                break;
              }
            if (allOk) {
              // If the resolved source name collides with
              // the alias's own ``fortran_name``, rename
              // the alias so SDFG keying doesn't self-loop.
              // This is the inlined-callee shape:
              // ``_QFmainEinp`` (caller arg) and
              // ``_QFinner_loopsEinp`` (inlined callee
              // dummy) both ``extractName`` to ``"inp"``;
              // the alias gets ``inner_loops_inp``
              // (callee-scope prefix) and the linking
              // edge wires correctly.  Register the
              // override on the thread-local map so the
              // subsequent AST extraction sees the same
              // renamed short name for every reference
              // to the inlined dummy.  Confined to the
              // view-alias path (allOk + traced srcName)
              // so unrelated inlined declares (optional
              // args, exact aliases) keep their names.
              if (srcName == v.fortran_name) {
                auto eP = v.mangled_name.rfind('E');
                auto fP = v.mangled_name.rfind('F', eP);
                if (eP != std::string::npos && fP != std::string::npos &&
                    fP + 1 < eP) {
                  std::string scope =
                      v.mangled_name.substr(fP + 1, eP - fP - 1);
                  std::string newName = scope + "_" + v.fortran_name;
                  setManglingOverride(v.mangled_name, newName);
                  v.fortran_name = newName;
                }
              }
              v.view_source = srcName;
              // A trivial section is also a "rank-
              // preserving" alias: the dummy's rank must
              // match the count of surviving triplet
              // dims.  Storage-association reshape
              // (``call sub(d(:, :, 1))`` with callee
              // ``dd(16)``) has dummy rank 1 but two
              // surviving triplets  --  that's an actual
              // shape change Flang inserts a
              // ``fir.convert`` to re-shape, and needs
              // the view_alias path's stride remapping.
              bool rank_matches = ((int)surviving == v.rank);
              if (is_trivial_section && rank_matches) {
                // Trivial section: name + index suffix
                // alias.  No SDFG view registration  --
                // every dummy access rewrites to a
                // source-array memlet via dim_map.
                v.role = "section_alias";
                v.view_dim_map = std::move(dim_map);
              } else {
                v.role = "view_alias";
                v.view_subset = std::move(subset);
              }
            }
          }
        } else if (auto srcDecl = mlir::dyn_cast<hlfir::DeclareOp>(defOp)) {
          // Whole-array RANK reinterpretation -- ssor's ``tv(N)`` 1D
          // passed unmodified to buts's ``tv(5, M, K)`` 3D dummy.
          // Same Fortran feature as the section-reshape branch above
          // (storage-association reshape) but the IR has NO designate
          // (no slicing); just a ``fir.convert`` from the source's
          // typed ref to the dummy's reinterpreted ref.  After the
          // peel loop above lands ``m`` on the source's
          // ``hlfir.declare``.
          //
          // ``asAssumedShapeAlias`` already refuses the alias collapse
          // when ranks differ (see ``trace_utils.cpp``), so this dummy
          // is now a separate VarInfo.  Mark it as a view_alias over
          // the source's flat storage; ``descriptors.py`` registers
          // the view with column-major strides over its OWN shape so
          // the source AccessNode -> view ViewAccessNode linking
          // memlet wires 1D source range -> ND view range correctly.
          // Single sentinel marker in ``view_subset`` -- one empty
          // string -- signals "whole-array rank reinterpretation"
          // to ``descriptors.py`` and ``access.py``.
          auto rankOfDeclResult = [](mlir::Value val) -> int {
            auto t = peelTypeLayers(val.getType());
            if (auto seq = mlir::dyn_cast<fir::SequenceType>(t))
              return seq.getDimension();
            return 0;
          };
          int srcRank = rankOfDeclResult(srcDecl.getResult(0));
          if (srcRank > 0 && srcRank != v.rank) {
            auto srcName = extractName(srcDecl.getUniqName().str());
            // ssor's tv was renamed to ``ssor_tv`` by Pass 0a's F-
            // scope-prefix path (own-storage + multi-procedure
            // short-name collision); buts's tv kept the bare name
            // ``tv``.  When they collide give the dummy its own
            // scope-prefix.
            if (srcName == v.fortran_name) {
              auto eP = v.mangled_name.rfind('E');
              auto fP = v.mangled_name.rfind('F', eP);
              if (eP != std::string::npos && fP != std::string::npos &&
                  fP + 1 < eP) {
                std::string scope =
                    v.mangled_name.substr(fP + 1, eP - fP - 1);
                std::string newName = scope + "_" + v.fortran_name;
                setManglingOverride(v.mangled_name, newName);
                v.fortran_name = newName;
              }
            }
            if (!srcName.empty() && srcName != v.fortran_name) {
              v.view_source = srcName;
              v.role = "view_alias";
              // Single sentinel "" entry: distinct from "rank-
              // preserving section_alias" (no entries) and from
              // "section reshape" (per-source-dim entries).  The
              // descriptor / access-node wiring sees one entry and
              // routes through the rank-reinterpret stride path.
              v.view_subset.clear();
              v.view_subset.push_back("");
            }
          }
        }
      }
    }

    // OPTIONAL dummy -> companion presence flag.  Fortran's
    // ``present(x)`` lowers to ``fir.is_present %x -> i1``, and the
    // bridge renders that as the name ``<x>_present``.  Register a
    // symbol VarInfo for that name here so callers see it on the
    // SDFG signature (non-zero = present, 0 = absent).  We register
    // it BEFORE pushing v, since the caller position should follow
    // the Fortran dummy order  --  the flag sits alongside its host.
    bool isOptional = false;
    if (auto a = op.getFortranAttrs()) {
      if (bitEnumContainsAny(*a, fir::FortranVariableFlagsEnum::optional))
        isOptional = true;
    }
    if (isOptional) {
      VarInfo pv;
      pv.fortran_name = v.fortran_name + "_present";
      pv.mangled_name = v.mangled_name + "_present";
      pv.dtype = "int32";  // plain Fortran integer
      pv.rank = 0;
      pv.intent = "in";
      pv.role = "symbol";
      vars.push_back(std::move(pv));
    }

    // Companion ``<arr>_allocated`` int32 transient for every
    // allocatable.  The AST builder writes ``1`` at each ALLOCATE
    // site and ``0`` at each DEALLOCATE site so the Fortran
    // ``ALLOCATED(arr)`` intrinsic  --  which Flang lowers to
    // ``box_addr(load arr_box) != 0``  --  can read this scalar
    // instead of inspecting the descriptor's heap pointer (which
    // DaCe's data model doesn't surface).  Initial value is 0
    // (DaCe default for transient scalars).
    if (isAllocatable && needsAllocatedTracker(v.mangled_name, module, &allocIdx,
                                               &allocatedReaderNames)) {
      // Role ``symbol`` (not ``scalar``) so writes land on
      // interstate edges and reads see the latest value across
      // state boundaries.  A plain transient scalar would let
      // DaCe's intra-state DAG scheduler interleave the
      // ALLOCATE-time write with surrounding ``ALLOCATED(arr)``
      // reads, producing the wrong intermediate value.  Symbols
      // also auto-register on the SDFG signature, so no extra
      // ``add_symbol`` plumbing is needed.
      VarInfo av;
      av.fortran_name = v.fortran_name + "_allocated";
      av.mangled_name = v.mangled_name + "_allocated";
      av.dtype = "int32";
      av.rank = 0;
      av.intent = "";
      av.role = "symbol";
      symbolNames.insert(av.fortran_name);
      vars.push_back(std::move(av));
    }

    // Register the BASE buffer's branch-extent symbols when class 0 is a
    // conditional (``a_d<i>``).  The AST builder assigns each at its
    // branch's ALLOCATE site, so it routes through the interstate-edge
    // symbol-write path and -- defined on every branch before the join --
    // stays off the program signature (like ``a_allocated``).
    auto registerExtentSyms = [&](const std::vector<std::string>& syms) {
      for (const auto& s : syms) {
        if (s.empty() || symbolNames.count(s)) continue;
        VarInfo dv;
        dv.fortran_name = s;
        dv.mangled_name = s;
        dv.dtype = "int64";
        dv.rank = 0;
        dv.intent = "";
        dv.role = "symbol";
        symbolNames.insert(s);
        vars.push_back(std::move(dv));
      }
    };
    if (baseCondAlloc) {
      registerExtentSyms(v.shape_symbols);
    } else if (!allocSites.empty() && v.rank > 0) {
      // Concrete-shape base buffer (``allocate(a(n))``).  The shape stays
      // the concrete extent (``n``), but ``size(a)`` / ``LBOUND`` / ``UBOUND``
      // lower to ``fir.box_dims`` which the bridge renders as ``<name>_d<i>``.
      // The ALLOCATE site has the dimensions, so the AST builder binds
      // ``<name>_d<i> = <extent>`` there -- register the symbol so it
      // resolves off-signature instead of leaking as a free program argument.
      std::vector<std::string> baseExtentSyms;
      for (int d = 0; d < v.rank; ++d)
        baseExtentSyms.push_back(v.fortran_name + "_d" + std::to_string(d));
      registerExtentSyms(baseExtentSyms);
    }

    // Register one synthetic transient per NON-base buffer class
    // (``a_alloc1``, ``a_alloc2``, ...).  A singleton class is a
    // sequentially re-allocated buffer -> concrete per-site shape; a
    // multi-site class is a conditional buffer -> a branch-extent symbol
    // (``a_allocK_d<i>``), assigned per branch by the AST builder.
    for (unsigned g = 1; g < allocClasses.size(); ++g) {
      const auto& cls = allocClasses[g];
      fir::AllocMemOp site0 = cls.front();
      VarInfo av;
      av.fortran_name = allocAliasName(v.fortran_name, g);
      av.mangled_name = v.mangled_name + "_alloc" + std::to_string(g);
      av.intent = "";  // local transient, no caller-side ABI
      av.dtype = v.dtype;
      av.rank = v.rank;
      av.is_dynamic = v.is_dynamic;
      av.role = "array";
      // A non-base buffer always uses a per-dim extent symbol
      // (``a_allocK_d<i>``), bound at its ALLOCATE site(s) by the AST
      // builder.  Unlike the base ``a`` (whose bare ``a_d<i>`` is bound by
      // the caller for a deferred-shape dummy), a versioned buffer is a
      // transient with no caller binding, so the symbol must be assigned
      // here -- this also lets ``size(a_allocK)`` resolve.  Works for both
      // a singleton (one site) and a conditional (branch) class.
      for (int d = 0; d < v.rank; ++d)
        av.shape_symbols.push_back(av.fortran_name + "_d" + std::to_string(d));
      av.lower_bounds.assign(av.shape_symbols.size(), "1");
      // Per-buffer lower-bound recovery from the embox shape_shift, so a
      // re-ALLOCATE with a non-default bound (``ALLOCATE(arr(0:10))``)
      // offsets correctly instead of defaulting to 1.
      auto lb_from_alloc = lowerBoundsFromAllocSite(site0);
      for (size_t d = 0; d < lb_from_alloc.size() && d < av.lower_bounds.size();
           ++d)
        if (lb_from_alloc[d] != "?") av.lower_bounds[d] = lb_from_alloc[d];
      registerExtentSyms(av.shape_symbols);
      vars.push_back(std::move(av));
    }

    // Init-value detection.  Two shapes feed the same path:
    //   * ``parameter`` declares pointing at ``fir.global ... constant``
    //      --  the read-only constant pool Flang synthesises for array
    //     / scalar literals.
    //   * Plain module-data declares pointing at ``fir.global`` with
    //     a ``fir.has_value`` body init (Fortran's ``real :: bob = 1``
    //     at module scope, no ``parameter`` attribute).
    // ``extractGlobalInitData`` covers both.  The SDFG side treats
    // the data as the transient's initial-value vector; writes to
    // the variable still flow through normally.
    std::string sym = traceToGlobalSymbol(op.getMemref());
    if (!sym.empty()) {
      auto gop = module.lookupSymbol<fir::GlobalOp>(sym);
      v.const_data = extractGlobalInitData(gop);
      const bool written = globalIsWritten(sym, module, &writtenGlobals);
      const bool isParameter =
          (gop && gop.getConstant().has_value() && *gop.getConstant());
      // A genuine module-scope VARIABLE is ``_QM<module>E<entity>`` -- the
      // first scope separator after the lowercase module name is ``E``.  A
      // module-PROCEDURE SAVE-local is ``_QM<module>F<proc>E<entity>`` (an
      // ``F``/``P`` scope before the ``E``); it is function-private, NOT
      // host-shared module data, so it must NOT take the inout / write-back /
      // provenance paths -- it stays a writable transient (or a baked
      // constexpr when read-only) seeded from its initialiser, like an
      // external-procedure ``_QF`` SAVE-local.
      const bool isModuleScope = [&] {
        llvm::StringRef s = sym;
        if (!s.consume_front("_QM")) return false;
        for (char ch : s)
          if (std::isupper(static_cast<unsigned char>(ch))) return ch == 'E';
        return false;
      }();
      auto recordModuleOrigin = [&] {
        // ``decodeModuleGlobalSymbol`` filters non-``_QM..E..`` shapes
        // (function-scope SAVE-locals, the literal constant pool), so it
        // is a no-op for those.
        auto origin = decodeModuleGlobalSymbol(sym);
        if (!origin.first.empty()) {
          v.module_origin_mod = origin.first;
          v.module_origin_name = origin.second;
        }
      };
      if (written) {
        v.is_written = true;
        if (isModuleScope && !isParameter && v.intent.empty()) {
          // A WRITTEN module-scope global is host-shared INOUT state: the
          // kernel's update must be visible to the caller after the call.
          // Surface it as an inout arg with module-origin provenance -- the
          // binding ``USE``-imports the host value (copy-in) and writes the
          // final value back (copy-out) on exit.  Its initialiser is the
          // host's default, not a baked constant, so drop ``const_data`` (no
          // constant pool, no entry seed; the arg carries the value).
          v.intent = "inout";
          v.const_data.clear();
          recordModuleOrigin();
        }
        // A WRITTEN function-scope SAVE-local (``_QF`` -- e.g. ``logical ::
        // bla = .false.`` that Fortran implicitly SAVEs) is private to its
        // function: it falls through keeping ``is_written`` + ``const_data``
        // so the builder seeds it as an internal writable transient (a
        // read-only ``constexpr`` would make the kernel's store fail to
        // compile).  No host linkage.
      } else {
        // Read-only module global.  Uninitialised (``INTEGER :: nrdmax(10)``
        // with no ``fir.has_value`` body) -> a USE-imported caller input;
        // mark ``inout`` so it surfaces as a non-transient kwarg the caller
        // fills, not a transient read from uninitialised memory.  Function-
        // scope SAVE-locals (``_QF``) are excluded -- the caller can't bind
        // them.  ``v.intent.empty()`` keeps dummy-arg shadows untouched.
        if (v.const_data.empty() && v.intent.empty() && gop && !isParameter &&
            isModuleScope)
          v.intent = "inout";
        // Record provenance for every read-only module global (initialised
        // or not), so the binding ``USE``-imports it.  An initialised
        // read-only global that took the ``const_data`` path bakes its
        // default, but still records provenance for an optional host
        // override.  PARAMETERs / the literal constant pool and
        // function-scope SAVE-locals are excluded.
        if (gop && !isParameter && isModuleScope) recordModuleOrigin();
      }
    }

    // Detect Fortran 2003 bounds-remapping pointer views tagged by
    // ``hlfir-mark-bounds-remap-views``.  When the tag is present:
    //   1. Trace the LAST rebind store (skipping any nullify) back
    //      through ``fir.rebox`` / ``fir.convert`` / ``hlfir.designate``
    //      to the underlying parent ``hlfir.declare``.
    //   2. Record the parent's Fortran name as ``bounds_remap_source``.
    //   3. Record the rebox's shape-shift extent operand (the flat
    //      1-D size) as ``bounds_remap_total_extent``, rendering it
    //      as a string the Python builder can parse / emit as a
    //      symbol or symbolic expression.
    //
    // The actual SDFG ``add_view`` + offset-symbol emission lives in
    // ``descriptors.py``; this block only surfaces the three fields.
    // A tagged declare whose rebind chain doesn't trace cleanly
    // falls through with ``bounds_remap_view = false`` so the
    // pipeline doesn't crash on an unsupported shape (the existing
    // rewriter would have rejected such a shape too).
    if (op->hasAttr("hlfir_bridge.bounds_remap_view")) {
      // Find the rebind store: the LAST non-nullify ``fir.store``
      // whose memref is ``op.getResult(0)``.
      fir::StoreOp rebindStore;
      for (auto* u : op.getResult(0).getUsers()) {
        auto st = mlir::dyn_cast<fir::StoreOp>(u);
        if (!st) continue;
        auto* valDef = st.getValue().getDefiningOp();
        if (auto eb = mlir::dyn_cast_or_null<fir::EmboxOp>(valDef))
          if (mlir::isa_and_nonnull<fir::ZeroOp>(
                  eb.getMemref().getDefiningOp()))
            continue;  // skip nullify
        if (!rebindStore || rebindStore->isBeforeInBlock(st))
          rebindStore = st;
      }
      if (rebindStore) {
        // The outermost ``fir.rebox`` (rebox form) or ``fir.embox``
        // (rank-changing remap form -- ``p(1:M,1:K) => arr1d``) carries
        // the shape_shift whose extent operands give the view's
        // multi-dim extents.  Trace through ``fir.convert`` to find it.
        mlir::Value cur = rebindStore.getValue();
        fir::ReboxOp topRebox;
        fir::EmboxOp topEmbox;
        for (int hops = 0; cur && hops < 8 && !topRebox && !topEmbox; ++hops) {
          auto* def = cur.getDefiningOp();
          if (!def) break;
          if (auto rb = mlir::dyn_cast<fir::ReboxOp>(def)) {
            topRebox = rb;
            break;
          }
          if (auto eb = mlir::dyn_cast<fir::EmboxOp>(def)) {
            topEmbox = eb;
            break;
          }
          if (auto cv = mlir::dyn_cast<fir::ConvertOp>(def)) {
            cur = cv.getValue();
            continue;
          }
          break;
        }
        if (topEmbox) {
          // Embox path: shape carries the multi-dim shape_shift; the
          // total extent is the product of its extents (rank-change
          // means the source is 1D / different rank).  For the
          // ``p(1:M, 1:K) => arr1d`` form the renderer needs to emit
          // ``M * K`` so descriptors.py mints the right total-extent
          // size.  Reuse the same renderExtent recursion below by
          // multiplying the per-dim extent strings.
          mlir::Value shape = topEmbox.getShape();
          if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(
                  shape ? shape.getDefiningOp() : nullptr)) {
            auto pairs = ss.getPairs();
            std::function<std::string(mlir::Value)> renderExt =
                [&](mlir::Value vv) -> std::string {
              for (int hops = 0; vv && hops < 8; ++hops) {
                auto* d = vv.getDefiningOp();
                if (!d) return "";
                if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
                  vv = cv.getValue();
                  continue;
                }
                if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d)) {
                  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
                    return std::to_string(ia.getInt());
                  return "";
                }
                if (auto ld = mlir::dyn_cast<fir::LoadOp>(d))
                  return traceToDecl(ld.getMemref());
                if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(d)) {
                  auto l = renderExt(mul.getLhs());
                  auto r = renderExt(mul.getRhs());
                  if (l.empty() || r.empty()) return "";
                  return l + "*" + r;
                }
                return "";
              }
              return "";
            };
            // Multiply per-dim extents for the total; ALSO fill
            // ``v.shape_symbols`` per dim so the SDFG View carries
            // the multi-D shape directly (no synthetic ``<ptr>_d<i>``
            // fallback that would leave the View's stride symbols
            // unbound at runtime).  ``pairs`` layout: lb0, ext0,
            // lb1, ext1, ...
            std::string total;
            std::vector<std::string> perDim;
            for (size_t i = 1; i < pairs.size(); i += 2) {
              auto e = renderExt(pairs[i]);
              if (e.empty()) { total.clear(); perDim.clear(); break; }
              perDim.push_back(e);
              total = total.empty() ? e : total + "*" + e;
            }
            v.bounds_remap_total_extent = total;
            if (!perDim.empty() && (int)perDim.size() == v.rank)
              v.shape_symbols = std::move(perDim);
          }
          // Walk the embox's memref back to the parent declare.
          mlir::Value parent = topEmbox.getMemref();
          for (int hops = 0; parent && hops < 16; ++hops) {
            auto* pd = parent.getDefiningOp();
            if (!pd) break;
            if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(pd)) {
              v.bounds_remap_source = extractName(dc.getUniqName().str());
              v.bounds_remap_view = !v.bounds_remap_source.empty();
              break;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(pd)) {
              parent = cv.getValue();
              continue;
            }
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(pd)) {
              parent = dg.getMemref();
              continue;
            }
            break;
          }
        }
        if (topRebox) {
          // Render the shape-shift's first extent operand
          // (lb0, ext0, lb1, ext1, ...) as the total extent.  For a
          // rank-1 shape_shift -- the only shape the mark pass
          // recognises -- this is the single ``ext0`` entry.
          if (auto ss = mlir::dyn_cast_or_null<fir::ShapeShiftOp>(
                  topRebox.getShape().getDefiningOp())) {
            auto pairs = ss.getPairs();
            if (pairs.size() >= 2) {
              // Render the extent SSA value as a parseable string.
              // Supports: (a) constant integer, (b) load of a named
              // declare (Fortran scalar name), (c) ``arith.muli`` of
              // two loads / constants (the common ``n*k`` shape).
              // Falls back to empty string for anything more complex;
              // descriptors.py then mints a synthetic
              // ``<ptr>_total_extent_d0`` symbol the caller binds.
              std::function<std::string(mlir::Value)> renderExtent =
                  [&](mlir::Value vv) -> std::string {
                for (int hops = 0; vv && hops < 8; ++hops) {
                  auto* d = vv.getDefiningOp();
                  if (!d) return "";
                  if (auto cv = mlir::dyn_cast<fir::ConvertOp>(d)) {
                    vv = cv.getValue();
                    continue;
                  }
                  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(d)) {
                    if (auto ia =
                            mlir::dyn_cast<mlir::IntegerAttr>(cst.getValue()))
                      return std::to_string(ia.getInt());
                    return "";
                  }
                  if (auto ld = mlir::dyn_cast<fir::LoadOp>(d)) {
                    return traceToDecl(ld.getMemref());
                  }
                  if (auto mul = mlir::dyn_cast<mlir::arith::MulIOp>(d)) {
                    std::string lhs = renderExtent(mul.getLhs());
                    std::string rhs = renderExtent(mul.getRhs());
                    if (lhs.empty() || rhs.empty()) return "";
                    return lhs + "*" + rhs;
                  }
                  return "";
                }
                return "";
              };
              v.bounds_remap_total_extent = renderExtent(pairs[1]);
            }
          }
          // Trace back to the parent declare through any
          // ``hlfir.designate`` / ``fir.convert`` chain on the
          // rebox's input box.
          mlir::Value parent = topRebox.getBox();
          for (int hops = 0; parent && hops < 16; ++hops) {
            auto* pd = parent.getDefiningOp();
            if (!pd) break;
            if (auto dc = mlir::dyn_cast<hlfir::DeclareOp>(pd)) {
              v.bounds_remap_source = extractName(dc.getUniqName().str());
              v.bounds_remap_view = !v.bounds_remap_source.empty();
              break;
            }
            if (auto dg = mlir::dyn_cast<hlfir::DesignateOp>(pd)) {
              parent = dg.getMemref();
              continue;
            }
            if (auto cv = mlir::dyn_cast<fir::ConvertOp>(pd)) {
              parent = cv.getValue();
              continue;
            }
            if (auto rb = mlir::dyn_cast<fir::ReboxOp>(pd)) {
              parent = rb.getBox();
              continue;
            }
            if (auto eb = mlir::dyn_cast<fir::EmboxOp>(pd)) {
              parent = eb.getMemref();
              continue;
            }
            break;
          }
        }
      }
    }

    vars.push_back(std::move(v));
  }
  // De-duplicate the collected value-symbols: the same array-element extent
  // (``__sym_nrdmax_jg``) can size several arrays, but each symbol is seeded
  // and constancy-checked once.
  if (value_symbols && !value_symbols->empty()) {
    std::set<std::string> seen;
    std::vector<ValueSymbol> uniq;
    for (auto& vs : *value_symbols)
      if (seen.insert(vs.symbol).second) uniq.push_back(vs);
    *value_symbols = std::move(uniq);
  }
  return vars;
}

FortranInterfaceInfo extractFortranInterface(mlir::ModuleOp module,
                                             const std::string& entry) {
  FortranInterfaceInfo out;

  // Locate the entry function: by mangled name, else the single public one.
  mlir::func::FuncOp fn;
  module.walk([&](mlir::func::FuncOp f) {
    if (fn) return;
    if (!entry.empty()) {
      if (f.getSymName() == entry) fn = f;
    } else if (!f.isDeclaration() &&
               mlir::SymbolTable::getSymbolVisibility(f) ==
                   mlir::SymbolTable::Visibility::Public) {
      fn = f;
    }
  });
  if (!fn || fn.getBody().empty()) return out;

  // Demangled -> mangled map so a module-parameter array bound (a shape
  // symbol like ``N``) can be decoded back to its defining module for the
  // wrapper's ``use`` list.
  llvm::StringMap<std::string> nameToMangled;
  module.walk([&](hlfir::DeclareOp d) {
    nameToMangled.try_emplace(extractName(d.getUniqName().str()),
                              d.getUniqName().str());
  });
  auto addUseFromMangled = [&](llvm::StringRef mangled) {
    auto md = decodeModuleGlobalSymbol(mangled.str());
    if (!md.first.empty() && !md.second.empty())
      out.used_modules[md.first].insert(md.second);
  };

  // ``_QM<mod>T<type>`` -> (mod, type); ``_QF..T<type>`` / ``_QT<type>``
  // -> ("", type) for a non-module (host-associated / program) type.
  auto parseRecordName = [](llvm::StringRef rn, std::string& mod,
                            std::string& tname) {
    if (rn.consume_front("_QM")) {
      auto t = rn.find('T');
      if (t != llvm::StringRef::npos) {
        mod = rn.substr(0, t).str();
        tname = rn.substr(t + 1).str();
        return;
      }
    }
    rn.consume_front("_Q");
    auto t = rn.rfind('T');
    tname = (t == llvm::StringRef::npos) ? rn.str() : rn.substr(t + 1).str();
  };

  // Recursively register every ``fir.RecordType`` reachable from
  // ``rec`` (top-level dummy struct + every nested derived-type
  // member) in ``out.struct_types``.  Each member entry carries
  // either a scalar element dtype + rank + per-dim static-shape
  // literals (box-of-array members unwrap to the underlying
  // sequence), or a populated ``struct_name`` + empty dtype for a
  // nested record member.  Unsupported leaf shapes (complex /
  // character / function pointer) keep both dtype and struct_name
  // empty so the Python side can flag them clearly.
  std::function<void(fir::RecordType, const std::string&,
                     const std::string&)>
      recordStructLayoutRecursive = [&](fir::RecordType rec,
                                        const std::string& mod,
                                        const std::string& tname) {
        if (out.struct_types.find(tname) != out.struct_types.end()) return;
        FortranStructLayout layout;
        layout.name = tname;
        layout.module = mod;
        // Insert before recursing so a self-referential / mutually-recursive
        // type graph terminates -- the second visit hits the early-out
        // above.  The members of this entry are filled in place below.
        auto& slot =
            (out.struct_types[tname] = std::move(layout));
        std::vector<std::pair<fir::RecordType,
                              std::pair<std::string, std::string>>> nested;
        for (auto& p : rec.getTypeList()) {
          FortranMemberInfo m;
          m.name = p.first;
          mlir::Type mt = p.second;
          // v2 box-of-array member: unwrap ``fir.box<fir.heap|fir.ptr<
          // seq<...>>>`` to expose the underlying sequence -- the
          // data buffer the post-marshal-expansion external sees via
          // ``fir.box_addr``.  Rank + shape come from the sequence;
          // dtype from its element type.
          if (auto box = mlir::dyn_cast<fir::BoxType>(mt)) {
            mlir::Type inner = box.getEleTy();
            if (auto heap = mlir::dyn_cast<fir::HeapType>(inner))
              inner = heap.getEleTy();
            else if (auto ptr = mlir::dyn_cast<fir::PointerType>(inner))
              inner = ptr.getEleTy();
            mt = inner;
          }
          if (auto seq = mlir::dyn_cast<fir::SequenceType>(mt)) {
            m.rank = (int)seq.getShape().size();
            for (auto e : seq.getShape()) {
              if (e == fir::SequenceType::getUnknownExtent())
                m.shape_symbols.emplace_back("?");
              else
                m.shape_symbols.emplace_back(std::to_string(e));
            }
            mt = seq.getEleTy();
          }
          if (mt.isF64()) m.dtype = "float64";
          else if (mt.isF32()) m.dtype = "float32";
          else if (mt.isInteger(8)) m.dtype = "int8";
          else if (mt.isInteger(16)) m.dtype = "int16";
          else if (mt.isInteger(32)) m.dtype = "int32";
          else if (mt.isInteger(64)) m.dtype = "int64";
          else if (mt.isInteger(1) || mlir::isa<fir::LogicalType>(mt))
            m.dtype = "bool";  // see ``dtypeName`` in FlattenStructs.cpp
          else if (auto nested_rec = mlir::dyn_cast<fir::RecordType>(mt)) {
            // Nested derived-type member: register its name + module
            // on this member, queue the type for its own layout entry.
            parseRecordName(nested_rec.getName(), m.struct_module,
                            m.struct_name);
            if (!m.struct_module.empty() && !m.struct_name.empty())
              out.used_modules[m.struct_module].insert(m.struct_name);
            if (!m.struct_name.empty())
              nested.emplace_back(nested_rec,
                                  std::make_pair(m.struct_module,
                                                 m.struct_name));
          }
          // Complex / character / function pointer: leave both
          // ``dtype`` and ``struct_name`` empty.
          slot.members.push_back(std::move(m));
        }
        for (auto& q : nested)
          recordStructLayoutRecursive(q.first, q.second.first,
                                      q.second.second);
      };

  auto& block = fn.getBody().front();
  for (auto barg : block.getArguments()) {
    hlfir::DeclareOp decl;
    for (auto* u : barg.getUsers())
      if (auto d = mlir::dyn_cast<hlfir::DeclareOp>(u)) {
        decl = d;
        break;
      }
    if (!decl) continue;

    FortranArgInfo a;
    a.name = extractName(decl.getUniqName().str());
    if (auto fa = decl.getFortranAttrs()) {
      auto f = *fa;
      if (bitEnumContainsAny(f, fir::FortranVariableFlagsEnum::intent_inout))
        a.intent = "inout";
      else if (bitEnumContainsAny(f, fir::FortranVariableFlagsEnum::intent_in))
        a.intent = "in";
      else if (bitEnumContainsAny(f, fir::FortranVariableFlagsEnum::intent_out))
        a.intent = "out";
    }

    // Peel box / ref / heap / pointer wrappers, then a SequenceType for
    // the rank, leaving the element (scalar or RecordType).
    mlir::Type ty = decl.getResult(0).getType();
    for (bool changed = true; changed;) {
      changed = true;
      if (auto b = mlir::dyn_cast<fir::BoxType>(ty)) ty = b.getEleTy();
      else if (auto r = mlir::dyn_cast<fir::ReferenceType>(ty)) ty = r.getEleTy();
      else if (auto h = mlir::dyn_cast<fir::HeapType>(ty)) ty = h.getEleTy();
      else if (auto p = mlir::dyn_cast<fir::PointerType>(ty)) ty = p.getEleTy();
      else changed = false;
    }
    if (auto seq = mlir::dyn_cast<fir::SequenceType>(ty)) {
      a.rank = (int)seq.getShape().size();
      ty = seq.getEleTy();
    }

    if (auto rec = mlir::dyn_cast<fir::RecordType>(ty)) {
      a.is_struct = true;
      parseRecordName(rec.getName(), a.struct_module, a.struct_name);
      if (!a.struct_module.empty() && !a.struct_name.empty())
        out.used_modules[a.struct_module].insert(a.struct_name);
      // Record the struct's member layout once per distinct type so
      // the Python ``build_auto_interface`` can populate
      // ``OriginalInterface.struct_types`` without the caller
      // hand-authoring it.  Walk recursively: each nested record
      // member's type is enqueued so it lands in ``struct_types``
      // too, keyed by its own name -- the Python side then looks
      // each nested struct up by ``FortranMemberInfo.struct_name``.
      // Box-of-array members unwrap to the underlying sequence's
      // dtype + rank + shape (the data buffer the post-marshal
      // ``fir.box_addr`` exposes); nested record members carry a
      // populated ``struct_name`` + empty dtype.  Other
      // unsupported leaf shapes (complex / character / function
      // pointer) keep empty dtype + empty struct_name so the
      // Python side can flag them clearly.
      if (!a.struct_name.empty())
        recordStructLayoutRecursive(rec, a.struct_module, a.struct_name);
    } else if (ty.isF64()) a.dtype = "float64";
    else if (ty.isF32()) a.dtype = "float32";
    else if (ty.isInteger(8)) a.dtype = "int8";
    else if (ty.isInteger(16)) a.dtype = "int16";
    else if (ty.isInteger(32)) a.dtype = "int32";
    else if (ty.isInteger(64)) a.dtype = "int64";
    // Top-level ``LOGICAL`` dummy args stay ``bool`` regardless of
    // KIND; the existing ``_build_logical_bridges`` Python pass
    // wraps the wrapper's outer LOGICAL(KIND=N) dummy in a
    // ``logical(c_bool)`` scratch + per-element bridge so the
    // SDFG-facing storage is always 1 byte.  The struct-member
    // walker below uses KIND-driven width because there is no such
    // bridge layer for struct-internal flatten companions (the
    // ``c_loc`` / ``c_f_pointer`` reinterpret needs the storage
    // size to match the source LOGICAL slot byte-for-byte).
    else if (ty.isInteger(1) || mlir::isa<fir::LogicalType>(ty)) a.dtype = "bool";
    else if (auto ct = mlir::dyn_cast<mlir::ComplexType>(ty)) {
      a.dtype = ct.getElementType().isF32() ? "complex64" : "complex128";
    } else {
      // Unknown element (e.g. character) -- leave dtype empty so the
      // Python side can fall back / raise rather than mis-bind.
    }

    a.shape_symbols = resolveShapeSyms(decl);
    for (auto& s : a.shape_symbols) {
      auto it = nameToMangled.find(s);
      if (it != nameToMangled.end()) addUseFromMangled(it->second);
    }
    out.args.push_back(std::move(a));
  }
  return out;
}

}  // namespace hlfir_bridge
