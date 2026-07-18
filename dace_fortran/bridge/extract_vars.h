// extract_vars.h -- Collect and classify every hlfir.declare in a module.

#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

/// One VarInfo per hlfir.declare, describing a Fortran variable (fortran_name, mangled_name, intent, dtype, rank,
/// is_dynamic, lower_bounds, role are self-explanatory); shape_symbols resolves per-dim as shape_hint attribute >
/// fir.shape/fir.shape_shift operand > synthetic "<var>_d<i>".
struct VarInfo {
  std::string fortran_name, mangled_name, intent, dtype;
  int rank = 0;
  bool is_dynamic = false;
  /// True for a module-scope global the kernel WRITES; if it also carries const_data, it's a writable transient seeded
  /// at SDFG entry, not a read-only constexpr.
  bool is_written = false;
  std::vector<std::string> shape_symbols;
  std::vector<std::string> lower_bounds;
  std::string role;
  /// Compile-time constant data for the read-only pool (Flang's ``_QQro.<shape>x<dtype>.<counter>`` globals); non-empty
  /// triggers an SDFG init state. Row-major doubles, one per element; Python narrows to actual dtype, booleans as
  /// 0.0/1.0.
  std::vector<double> const_data;
  /// For role == "view_alias" only: view_source is the underlying array's Fortran name, view_subset is one 0-based
  /// DaCe-form entry per source dim ("0:4" full range, "2" fixed). Set when Flang emits hlfir.declare on a
  /// fir.convert-reshaped section (storage-association reshape); descriptors stages copy-in/copy-out through the alias.
  std::string view_source;
  std::vector<std::string> view_subset;
  /// For role == "section_alias" only: one entry per source dim, surviving dims are "_d<N>" placeholders, dropped dims
  /// hold a 0-based index expr. Python splices dummy index_exprs into the placeholders (no separate SDFG view). Only
  /// for structurally trivial sections (lo=1,stride=1); strided/sub-range sections use view_alias instead.
  std::vector<std::string> view_dim_map;
  /// Fortran module-global provenance: non-empty when storage traces via fir.address_of to a ``_QM<module>E<entity>``
  /// global (USE-associated, not a dummy). module_origin_mod/module_origin_name are the lowercased module/entity name;
  /// the binding generator auto-emits the use-import from these without a hand-authored map. Empty for ordinary vars
  /// and the constant pool.
  std::string module_origin_mod;
  std::string module_origin_name;
  /// Module-origin global's storage class (fir.global box kind): allocatable/pointer flags, both false for static.
  /// Copy-in binding guards a deferred-storage host with allocated()/associated() so an absent conditional global isn't
  /// read; a static global is always present.
  bool module_origin_allocatable = false;
  bool module_origin_pointer = false;
  /// Fortran 2003 bounds-remap pointer view metadata (set by hlfir-mark-bounds-remap-views); bounds_remap_view gates
  /// the other two: bounds_remap_source is the rebox chain's root array name, bounds_remap_total_extent is the flat 1-D
  /// extent expr (e.g. "n*k"). descriptors.py emits an aliased add_view plus a fresh offset_<name>_d0 symbol bound to
  /// the rebox column-offset.
  bool bounds_remap_view = false;
  std::string bounds_remap_source;
  std::string bounds_remap_total_extent;
  /// True for an inlined-callee array dummy bound to a section the bridge can't view (e.g. QE's becxx(ikq)%k(:,jbnd),
  /// an unallocated AoS component); descriptors.py registers it as a read-only transient instead of a program arg since
  /// it's dead on every non-allocating path. See extract_vars.cpp's component-base guard.
  bool unbindable_section = false;
  /// Marshalling metadata for a flattened component of a module-level array-of-structs global (QE ``us_exx``'s
  /// ``becxx(ikq)%k``): aos_origin_mod/aos_origin_struct/aos_member_path identify the owning module/global/%-joined
  /// member path, aos_outer_rank is the AoS dim count, global_alloc_inside means the kernel itself ALLOCATEs it
  /// (binding must allocate the host and skip copy-in). All empty/0 for ordinary vars.
  std::string aos_origin_mod;
  std::string aos_origin_struct;
  std::string aos_member_path;
  int aos_outer_rank = 0;
  bool global_alloc_inside = false;
  /// aos_struct_pointer/aos_member_pointer flag a Fortran POINTER (vs ALLOCATABLE) struct/component; binding must guard
  /// with associated() not allocated() (wrong intrinsic is a hard type error), falling back to a degenerate buffer
  /// instead of an undefined size() when unassociated.
  bool aos_struct_pointer = false;
  bool aos_member_pointer = false;
  /// Per-source-dim subset the bounds-remap view covers, as 0-based DaCe subset strings (e.g. {"0:nrows",
  /// "(c0)-1:(c0)-1+ncols"} for p(1:n*k) => a(:, c0:c1)); empty for a whole-array reinterpretation. access.py uses it
  /// to keep the column offset symbolic in the original->view memlet.
  std::vector<std::string> bounds_remap_source_subset;
};

/// Decodes a Flang module-global mangled symbol (``_QM<module>E<entity>``) into (module, entity); empty pair if not a
/// module-global form (function-scope ``_QF..``, program ``_QP..``, ``_QQro`` constant pool, etc.).
std::pair<std::string, std::string> decodeModuleGlobalSymbol(const std::string& sym);

/// An array element value used where the SDFG needs a symbol (e.g. ICON's z_raylfac(nrdmax(jg))); DaCe rejects the bare
/// array name as both descriptor and symbol, so the bridge mints ``__sym_<array>_<index>``. Builder seeds it from the
/// element read and asserts it stays constant in scope.
struct ValueSymbol {
  std::string symbol;      // mangled symbol, e.g. ``__sym_nrdmax_jg``
  std::string array;       // the data descriptor read from, e.g. ``nrdmax``
  std::string index_expr;  // 1-based Fortran index expression, e.g. ``jg``
};

/// Walks the module, builds one VarInfo per hlfir.declare; if value_symbols is non-null, also collects
/// array-element-as-symbol promotions (see ValueSymbol). entry_symbol (as passed to set_entry_symbol) anchors
/// extractName's scope qualification; empty disables it.
std::vector<VarInfo> extractVariables(mlir::ModuleOp module, std::vector<ValueSymbol>* value_symbols = nullptr,
                                      const std::string& entry_symbol = "");

/// Prepares per-thread extraction state shared by extractVariables/extractAST (mangling overrides, entry F-scope,
/// ``_call<idx>`` disambiguation, short-name collision set). Idempotent; without it, extractAST called standalone leaks
/// stale kEntryScope/kShortNameCollisions from a prior extraction. Throws if a user variable's short name collides with
/// an inlined intrinsic (min/max/sum/sqrt, etc.).
void prepareExtractionState(mlir::ModuleOp module, const std::string& entry_symbol);

/// One entry dummy arg, pre-flatten view; produced by extractFortranInterface so the binding emitter can auto-derive an
/// OriginalInterface. Must be read BEFORE hlfir-flatten-structs runs -- flattening destroys the struct dummy's AoS
/// view.
struct FortranArgInfo {
  std::string name;       // Fortran dummy name (``pts``)
  std::string dtype;      // element dtype (``complex128`` / ``float64`` / ...)
                          // empty for a derived-type arg (see ``is_struct``)
  std::string intent;     // ``in`` / ``out`` / ``inout`` / ``""``
  bool optional = false;  // dummy declared OPTIONAL (Fortran ``present(x)``
                          // companion ``<name>_present`` is a real symbol)
  int rank = 0;
  std::vector<std::string> shape_symbols;  // per-dim extent symbol / literal
  bool is_struct = false;                  // derived-type dummy
  std::string struct_name;                 // ``point`` when ``is_struct``
  std::string struct_module;               // defining module (``mo_pt``) or ``""``
};

/// One field of a Fortran derived-type entry dummy, populated alongside FortranArgInfo; extractFortranInterface walks
/// the fir::RecordType member list. Read by the binding emitter (build_auto_interface) to auto-derive a struct-arg
/// interface.
struct FortranMemberInfo {
  std::string name;   // member name (``a``)
  std::string dtype;  // scalar element dtype, empty for unsupported (nested
                      // struct, complex, character)
  int rank = 0;
  std::vector<std::string> shape_symbols;  // static-shape literal ints / "?"
  std::string struct_name;                 // for a nested-derived-type member: the
                                           // member type's name (``t_grid_cells``).  Lets
                                           // the Python side look the layout up in
                                           // ``OriginalInterface.struct_types``;
                                           // populated only when the member is itself a
                                           // ``fir.RecordType``.  Empty otherwise.
  std::string struct_module;               // defining module of the nested type
  std::string alloc;                       // deferred-storage class of the member:
                                           // "allocatable" (``box<heap<..>>``),
                                           // "pointer" (``box<ptr<..>>``), else "".
                                           // Drives the binding emitter's presence
                                           // guards (``allocated``/``associated``).
                                           // (``mo_model_domain``), or ``""``.
};

/// One derived-type layout the entry's dummies reference.
struct FortranStructLayout {
  std::string name;    // ``t_fld``
  std::string module;  // defining module (``mo_fld``), or ``""`` for a
                       // host-associated / program-local type
  std::vector<FortranMemberInfo> members;
};

/// The whole caller-facing surface of one entry: its dummies in order, plus the ``use <mod>, only: <syms>`` set the
/// wrapper needs to resolve derived-type names and module-parameter bounds.
struct FortranInterfaceInfo {
  std::vector<FortranArgInfo> args;
  /// module name -> referenced symbols (derived-type names + shape params).
  std::map<std::string, std::set<std::string>> used_modules;
  /// struct name -> layout, one entry per distinct derived type in ``args`` whose ``fir::RecordType`` was reachable in
  /// the entry's signature.
  std::map<std::string, FortranStructLayout> struct_types;
};

/// Walks the entry function's block args IN ORDER, describing each as the caller sees it (pre-flatten). ``entry`` is
/// the mangled symbol (empty selects the single public function); returns empty args if none resolvable.
FortranInterfaceInfo extractFortranInterface(mlir::ModuleOp module, const std::string& entry);

/// Index of every fir.allocmem keyed by uniq_name, built with one module walk; passing it to the helpers below turns
/// their per-variable module.walk into an O(1) lookup (O(module+variables) vs O(variables x module)).
using AllocSitesIndex = std::map<std::string, std::vector<fir::AllocMemOp>>;

/// True iff allocatable/pointer declName needs the per-variable ``<declName>_allocated`` int32 tracker (body
/// ALLOCATEs/DEALLOCATEs it, or an ALLOCATED()/ASSOCIATED() reader exists); already-allocated dummies never queried by
/// ALLOCATED() skip the tracker.
bool needsAllocatedTracker(const std::string& declUniqName, mlir::ModuleOp module, const AllocSitesIndex* idx = nullptr,
                           const std::set<std::string>* readerNames = nullptr);

/// Per-site name for an allocatable ALLOCATE: site 0 keeps the original name, site 1+ mints ``x_alloc1``, ``x_alloc2``,
/// ...; shared by extractVariables (registers synthetic VarInfos) and extractAST (keeps the trace-utils alias map in
/// sync).
std::string allocAliasName(const std::string& fortran, unsigned site);

/// Every fir.allocmem whose uniq_name is ``<declName>.alloc``, in IR walk order; uses the prebuilt ``idx`` when given
/// instead of a module walk.
std::vector<fir::AllocMemOp> collectAllocSites(const std::string& declName, mlir::ModuleOp module,
                                               const AllocSitesIndex* idx = nullptr);

/// True iff the ALLOCATE sites are mutually exclusive (different branches of one scf.if/fir.if) rather than sequential;
/// such an array stays one transient with a branch-dependent extent symbol instead of versioning into x_allocK.
bool allocSitesInExclusiveBranches(const std::vector<fir::AllocMemOp>& sites);

/// Partitions an allocatable's ALLOCATE sites into buffer equivalence classes (one DaCe transient each): sites
/// co-reaching an scf.if/fir.if join as alternatives share a class, sequential re-allocations land in separate classes.
/// Class >1 site = conditional (branch-dependent extent symbol); singleton = concrete shape. See
/// ALLOC_BUFFER_SSA_DESIGN.md.
std::vector<std::vector<fir::AllocMemOp>> groupAllocSites(const std::string& declName, mlir::ModuleOp module,
                                                          const AllocSitesIndex* idx = nullptr);

}  // namespace hlfir_bridge
