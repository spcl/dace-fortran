// ============================================================================
// extract_vars.h  --  Collect and classify every hlfir.declare in a module.
// ============================================================================

#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

/// One per hlfir.declare.  Describes a Fortran variable.
///
///   fortran_name    --  short Fortran name, e.g. "nproma"
///   mangled_name    --  Flang unique name, e.g. "_QFcompute_z_v_grad_wEnproma"
///   intent          --  "in" | "out" | "inout" | "" (local)
///   dtype           --  "float64" | "float32" | "int32" | "int64" | raw type
///   rank            --  number of array dimensions (0 for scalars)
///   is_dynamic      --  true if any dim is ? (unknown extent)
///   shape_symbols   --  per-dim extent name.  Resolution order:
///                      1. hlfir_bridge.shape_hint attribute (from passes)
///                      2. fir.shape / fir.shape_shift operand
///                      3. synthetic "<var>_d<i>" for assumed-shape (:,:)
///   lower_bounds    --  per-dim Fortran lower bound as string
///   role            --  "array" | "symbol" | "scalar"
struct VarInfo {
  std::string fortran_name, mangled_name, intent, dtype;
  int rank = 0;
  bool is_dynamic = false;
  /// True when this is a module-scope global the kernel WRITES.  Such a
  /// variable is "not really constant": if it also carries an initial value
  /// (``const_data``) it becomes a writable transient seeded with that value
  /// at SDFG entry, not a read-only constant-pool ``constexpr``.
  bool is_written = false;
  std::vector<std::string> shape_symbols;
  std::vector<std::string> lower_bounds;
  std::string role;
  /// Compile-time constant data for the read-only constant pool
  /// (Flang's ``_QQro.<shape>x<dtype>.<counter>`` globals).  When
  /// non-empty the SDFG builder synthesises an init state writing
  /// these values into the transient before the kernel body runs.
  /// Empty for ordinary variables.  Value layout: row-major doubles
  /// (one per element)  --  the Python side narrows to the actual
  /// dtype on use.  Booleans surface as 0.0 / 1.0.
  std::vector<double> const_data;
  /// For ``role == "view_alias"`` only.  ``view_source`` is the
  /// underlying array's Fortran name; ``view_subset`` is one entry
  /// per source-array dim in 0-based DaCe form  --  ``"0:4"`` for a
  /// full range, ``"2"`` for a fixed scalar.  The alias surface is
  /// a (possibly rank-changed) re-interpretation of ``view_source``
  /// over the section indicated by ``view_subset``.  ``descriptors``
  /// uses this to stage a copy-in at SDFG entry and a copy-out at
  /// SDFG exit so writes propagate back through the alias.  Set
  /// when Flang emits ``hlfir.declare %converted`` where the
  /// memref ultimately threads through a ``fir.convert`` that
  /// re-shapes a section designate's element type to a different
  /// array shape (Fortran storage-association reshape).
  std::string view_source;
  std::vector<std::string> view_subset;
  /// For ``role == "section_alias"`` only.  One entry per source-array
  /// dim; surviving dims are placeholders ``"_d<N>"`` (N = 0-based
  /// dummy-dim index), dropped scalar dims hold a 0-based DaCe-form
  /// index expression (``"(k)-1"`` for symbolic, ``"<int>"`` for
  /// constant).  The Python builder splices the inlined-body's
  /// dummy index_exprs into the placeholders to produce a full
  /// source-array memlet  --  no separate SDFG view is registered.
  /// Set only when the section is structurally trivial (every triplet
  /// has lo=1, stride=1), so the alias is just a name + index suffix.
  /// Non-trivial sections (strided / sub-range) stay on the
  /// ``view_alias`` path.
  std::vector<std::string> view_dim_map;
  /// Fortran *module*-global provenance.  Non-empty only when this
  /// declare's storage traces (through ``fir.address_of``) to a
  /// module-scope ``fir.global`` whose mangled symbol has the
  /// ``_QM<module>E<entity>`` form  --  i.e. the value is read from
  /// module data (``USE <module>, ONLY: <entity>``), not received as
  /// a dummy argument.  ``module_origin_mod`` is the (lowercased)
  /// Fortran module name; ``module_origin_name`` is the entity name.
  /// The binding generator consumes these to auto-emit the
  /// ``use``-import + assignment WITHOUT a hand-authored
  /// ``module_symbol_sources`` map.  Both empty for ordinary
  /// dummies / locals and for the read-only literal constant pool
  /// (those carry ``const_data`` instead).
  std::string module_origin_mod;
  std::string module_origin_name;
};

/// Decode a Flang module-global mangled symbol of the form
/// ``_QM<module>E<entity>`` into its ``(module, entity)`` pair.
///
/// :param sym: mangled symbol (no leading ``@``), e.g.
///     ``_QMmo_parallel_configEnproma``.
/// :returns: ``(module, entity)`` on a successful module-scope
///     decode (``_QMmo_parallel_config``, ``nproma``); an empty
///     pair when ``sym`` is not a ``_QM..E..`` module global
///     (function-scope ``_QF..``, program ``_QP..``, the
///     ``_QQro`` constant pool, or any non-conforming name).
std::pair<std::string, std::string> decodeModuleGlobalSymbol(
    const std::string& sym);

/// An array *element value* used where the SDFG needs a symbol -- a
/// data-access dimension or bound, e.g. ICON's ``z_raylfac(nrdmax(jg))`` whose
/// extent is the runtime-indexed element ``nrdmax(jg)``.  Collapsing that to the
/// bare array name would make ``nrdmax`` both a data descriptor and a symbol
/// (DaCe rejects it), so the bridge mints a distinct mangled symbol
/// ``__sym_<array>_<index>`` and records it here.  The SDFG builder then (a)
/// seeds the symbol from the element read and (b) asserts the element is
/// constant in the symbol's scope -- a write after the symbol is frozen would
/// be a stale-value bug.
struct ValueSymbol {
  std::string symbol;      // mangled symbol, e.g. ``__sym_nrdmax_jg``
  std::string array;       // the data descriptor read from, e.g. ``nrdmax``
  std::string index_expr;  // 1-based Fortran index expression, e.g. ``jg``
};

/// Walk the module and build one VarInfo per hlfir.declare.  When
/// ``value_symbols`` is non-null, also collect the array-element-as-symbol
/// promotions encountered while resolving array extents (see ``ValueSymbol``).
std::vector<VarInfo> extractVariables(
    mlir::ModuleOp module, std::vector<ValueSymbol>* value_symbols = nullptr);

/// One entry-subroutine dummy argument, in the caller's pre-flatten view.
/// Produced by ``extractFortranInterface`` so the binding emitter can
/// auto-derive an ``OriginalInterface`` (name / type / rank / shape /
/// intent + derived-type origin) instead of the caller hand-writing it.
/// Must be read BEFORE ``hlfir-flatten-structs`` runs -- flattening
/// destroys the struct dummy's AoS view and reorders nothing the caller
/// would recognise.
struct FortranArgInfo {
  std::string name;     // Fortran dummy name (``pts``)
  std::string dtype;    // element dtype (``complex128`` / ``float64`` / ...)
                        // empty for a derived-type arg (see ``is_struct``)
  std::string intent;   // ``in`` / ``out`` / ``inout`` / ``""``
  int rank = 0;
  std::vector<std::string> shape_symbols;  // per-dim extent symbol / literal
  bool is_struct = false;       // derived-type dummy
  std::string struct_name;      // ``point`` when ``is_struct``
  std::string struct_module;    // defining module (``mo_pt``) or ``""``
};

/// One field of a Fortran derived type used as an entry dummy.  Populated
/// alongside :class:`FortranArgInfo` when the dummy is a derived-type:
/// ``extractFortranInterface`` walks the ``fir::RecordType`` member list and
/// records each member's caller-facing shape.  Read by the binding emitter
/// (``build_auto_interface``) so a struct-arg interface can be auto-derived
/// rather than hand-authored.
struct FortranMemberInfo {
  std::string name;     // member name (``a``)
  std::string dtype;    // scalar element dtype, empty for unsupported (nested
                        // struct, box / heap / pointer member)
  int rank = 0;
  std::vector<std::string> shape_symbols;  // static-shape literal ints / "?"
};

/// One derived-type layout the entry's dummies reference.
struct FortranStructLayout {
  std::string name;     // ``t_fld``
  std::string module;   // defining module (``mo_fld``), or ``""`` for a
                        // host-associated / program-local type
  std::vector<FortranMemberInfo> members;
};

/// The whole caller-facing surface of one entry: its dummies in order,
/// plus the ``use <mod>, only: <syms>`` set the wrapper needs to resolve
/// derived-type names and module-parameter array bounds.
struct FortranInterfaceInfo {
  std::vector<FortranArgInfo> args;
  /// module name -> referenced symbols (derived-type names + shape params).
  std::map<std::string, std::set<std::string>> used_modules;
  /// struct name -> layout, one entry per distinct derived type that
  /// appears in ``args`` (and whose ``fir::RecordType`` was reachable in
  /// the entry's signature).
  std::map<std::string, FortranStructLayout> struct_types;
};

/// Walk the entry function's block arguments IN ORDER and describe each
/// as the caller sees it (pre-flatten).  ``entry`` is the mangled symbol
/// (``_QPkernel``); empty selects the single public function.  Returns an
/// empty ``args`` vector if the entry has no resolvable declares.
FortranInterfaceInfo extractFortranInterface(mlir::ModuleOp module,
                                             const std::string& entry);

/// Index of every ``fir.allocmem`` keyed by its ``uniq_name``, built once with
/// a single module walk.  Passing it to the helpers below replaces their
/// per-variable ``module.walk`` with an O(1) lookup -- the difference between
/// O(variables x module) and O(module + variables) when extracting a
/// fully-inlined whole-program entry.
using AllocSitesIndex = std::map<std::string, std::vector<fir::AllocMemOp>>;

/// True iff the allocatable / pointer ``declName`` needs the
/// per-variable ``<declName>_allocated`` int32 tracker scalar  --  i.e.
/// either the kernel body writes it (an ALLOCATE / DEALLOCATE site
/// exists) or reads it (an ``ALLOCATED(arr)`` / ``ASSOCIATED(ptr)``
/// reader exists, lowered to ``fir.box_addr``).  Dummies passed in
/// already-allocated and never queried by ``ALLOCATED(...)`` skip the
/// tracker entirely.
///
/// :param allocIdx: optional prebuilt alloc-site index (see above).
/// :param readerNames: optional prebuilt set of short names with an
///     ``ALLOCATED`` / ``ASSOCIATED`` reader.
bool needsAllocatedTracker(const std::string& declName, mlir::ModuleOp module,
                           const AllocSitesIndex* allocIdx = nullptr,
                           const std::set<std::string>* readerNames = nullptr);

/// Per-site name for an allocatable ``ALLOCATE``.  Site 0 keeps the
/// original Fortran name (``x``); site 1+ mints synthetic transient
/// names (``x_alloc1``, ``x_alloc2``, ...).  Shared between
/// ``extractVariables`` (which registers the synthetic VarInfos) and
/// ``extractAST`` (which keeps the trace-utils alias map in sync as
/// it walks the IR).
std::string allocAliasName(const std::string& fortran, unsigned site);

/// Every ``fir.allocmem`` whose ``uniq_name`` is ``<declName>.alloc`` (the
/// ALLOCATE sites of one allocatable), in IR walk order.  When ``idx`` is
/// given the result comes from that prebuilt index instead of a module walk.
std::vector<fir::AllocMemOp> collectAllocSites(const std::string& declName,
                                               mlir::ModuleOp module,
                                               const AllocSitesIndex* idx = nullptr);

/// True iff the ALLOCATE sites are mutually exclusive  --  each in a
/// different branch of one common ``scf.if`` / ``fir.if`` (a conditional
/// ALLOCATE) rather than sequential re-allocation.  Such an array stays
/// one transient with a branch-dependent extent symbol (the AST builder
/// assigns ``<name>_d<i>`` per branch), not versioned into ``x_allocK``.
bool allocSitesInExclusiveBranches(const std::vector<fir::AllocMemOp>& sites);

/// Partition an allocatable's ALLOCATE sites into buffer equivalence
/// classes (one DaCe transient each), ordered by first definition.  Two
/// sites share a class iff their buffers co-reach an ``scf.if`` / ``fir.if``
/// join as alternatives (the conditional / branch case); sites never
/// simultaneously live (sequential re-allocation) land in separate
/// classes.  A class with >1 site is conditional (use a branch-dependent
/// extent symbol); a singleton class uses the site's concrete shape.  See
/// ALLOC_BUFFER_SSA_DESIGN.md.
std::vector<std::vector<fir::AllocMemOp>> groupAllocSites(
    const std::string& declName, mlir::ModuleOp module,
    const AllocSitesIndex* idx = nullptr);

}  // namespace hlfir_bridge
