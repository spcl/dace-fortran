// bridge.cpp -- nanobind Python boundary for the HLFIR bridge. Thin layer: delegates extraction to
// extract_vars.cpp/extract_ast.cpp, passes to the MLIR pass pipeline parser; walks no IR itself.

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cctype>

// MLIR core
#include "llvm/Support/thread.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

// Flang FIR + HLFIR
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"

// DialectInlinerInterface registration required -- without it mlir::inlineCall segfaults in
// InlinerInterface::handleArgument (no per-dialect interface attached).
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"

// Standard MLIR dialects needed by Flang's HLFIR output
#include <stdexcept>
#include <string>

#include "bridge/extract_ast.h"
#include "bridge/extract_vars.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "passes/Passes.h"
#include "passes/shallow_alias.h"

namespace nb = nanobind;
using namespace hlfir_bridge;

/// Stack size for the pass-pipeline worker thread: 2 GB reserved (lazily-committed) covers whole-program-inlined
/// nesting depths that overflow the default 8 MB stack in MLIR's recursive region cloning/verifying.
static constexpr unsigned kPassPipelineStackBytes = 2U * 1024U * 1024U * 1024U;

// HLFIRModule -- Python-facing container for one parsed HLFIR module.

class HLFIRModule {
 public:
  HLFIRModule() {
    registry_.insert<fir::FIROpsDialect, hlfir::hlfirDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect,
                     mlir::scf::SCFDialect, mlir::math::MathDialect, mlir::complex::ComplexDialect, mlir::DLTIDialect,
                     mlir::cf::ControlFlowDialect, mlir::vector::VectorDialect, mlir::affine::AffineDialect,
                     mlir::omp::OpenMPDialect, mlir::acc::OpenACCDialect, mlir::LLVM::LLVMDialect>();
    // Attach DialectInlinerInterface per dialect (mirrors Flang's InitFIR.h); without it mlir::inlineCall dereferences
    // a null dialect interface.
    mlir::func::registerInlinerExtension(registry_);
    mlir::LLVM::registerInlinerInterface(registry_);
    fir::addFIRInlinerExtension(registry_);
    ctx_.appendDialectRegistry(registry_);
    ctx_.loadAllAvailableDialects();
  }

  bool parse(const std::string& t) {
    // Reset cached entry symbol -- must not leak across parses (D1).
    entry_symbol_.clear();
    module_ = mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(t), &ctx_);
    return static_cast<bool>(module_);
  }

  bool parse_file(const std::string& p) {
    entry_symbol_.clear();
    module_ = mlir::parseSourceFile<mlir::ModuleOp>(llvm::StringRef(p), &ctx_);
    return static_cast<bool>(module_);
  }

  /// Parses & merges multiple HLFIR files into one module (first file = base) so hlfir-inline-all can flatten
  /// cross-file call trees; dedups moved-across symbols by name, definition wins over external declaration.
  bool parse_files(const std::vector<std::string>& paths) {
    entry_symbol_.clear();
    if (paths.empty()) return false;
    module_ = mlir::parseSourceFile<mlir::ModuleOp>(llvm::StringRef(paths[0]), &ctx_);
    if (!module_) return false;
    if (paths.size() == 1) return true;

    auto& baseBody = module_->getBodyRegion().front();
    mlir::SymbolTable baseTab(*module_);
    auto symName = [](mlir::Operation* op) -> llvm::StringRef {
      if (auto a = op->getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName())) return a.getValue();
      return {};
    };

    for (size_t i = 1; i < paths.size(); ++i) {
      auto extra = mlir::parseSourceFile<mlir::ModuleOp>(llvm::StringRef(paths[i]), &ctx_);
      if (!extra) return false;
      auto& extraBody = extra->getBodyRegion().front();

      // Move each op individually so inc_range is safe under mutation.
      for (auto& op : llvm::make_early_inc_range(extraBody)) {
        auto nm = symName(&op);
        if (!nm.empty()) {
          if (auto* existing = baseTab.lookup(nm)) {
            auto existingFn = mlir::dyn_cast<mlir::func::FuncOp>(existing);
            auto newFn = mlir::dyn_cast<mlir::func::FuncOp>(&op);
            if (existingFn && newFn && existingFn.isDeclaration() && !newFn.isDeclaration()) {
              baseTab.erase(existingFn);  // replace decl with def
            } else {
              op.erase();  // keep base's version
              continue;
            }
          }
        }
        op.remove();
        baseBody.push_back(&op);
        if (!nm.empty()) baseTab.insert(&op);
      }
    }
    return true;
  }

  /// Runs an mlir-opt-syntax pipeline; pass names from passes/*.cpp are usable here via registerAllBridgePasses() at
  /// NB_MODULE init.
  void run_passes(const std::string& pipeline) {
    if (!module_) throw std::runtime_error("run_passes: no module parsed");
    mlir::PassManager pm(&ctx_);
    if (mlir::failed(mlir::parsePassPipeline(pipeline, pm)))
      throw std::runtime_error("run_passes: bad pipeline: " + pipeline);
    // DACE_HLFIR_TRACE_PASSES: prints each pass name+op to stderr before it runs, so the last line before a crash IDs
    // the culprit pass. Off unless the env var is set.
    if (std::getenv("DACE_HLFIR_TRACE_PASSES")) {
      struct PassTracer : public mlir::PassInstrumentation {
        void runBeforePass(mlir::Pass* pass, mlir::Operation* op) override {
          llvm::errs() << "HLFIR_PASS_TRACE before: " << pass->getName() << " on " << op->getName() << "\n";
        }
      };
      pm.addInstrumentation(std::make_unique<PassTracer>());
    }
    // Multithreading disabled so nested passes run serially on the big-stack worker below -- MLIR's own thread pool
    // uses default ~8 MB stacks that deep inline recursion (e.g. hlfir-inline-all) overflows, crashing with a
    // corrupt-stack SIGSEGV.
    bool const prev_mt = ctx_.isMultithreadingEnabled();
    ctx_.disableMultithreading();
    // Runs on a large-stack worker thread: deep inlined IR overflows the default stack during MLIR's recursive region
    // cloning/verification. Context touched by one thread at a time (joined before return).
    bool ok = false;
    llvm::thread worker(std::optional<unsigned>(kPassPipelineStackBytes),
                        [&] { ok = mlir::succeeded(pm.run(*module_)); });
    worker.join();
    if (prev_mt) ctx_.enableMultithreading();
    if (!ok) throw std::runtime_error("run_passes: pipeline failed");
  }

  /// Print the current IR as text (useful for debugging from Python).
  std::string dump() {
    if (!module_) return "";
    std::string s;
    llvm::raw_string_ostream os(s);
    module_->print(os);
    return s;
  }

  std::vector<VarInfo> get_variables() {
    if (!module_) return {};
    value_symbols_.clear();
    return extractVariables(*module_, &value_symbols_, entry_symbol_);
  }

  /// Array-element values promoted to SDFG symbols (``__sym_<arr>_<idx>``) during the last get_variables call; builder
  /// seeds each from its element read and asserts constancy.
  std::vector<ValueSymbol> get_value_symbols() { return value_symbols_; }

  std::vector<ASTNode> get_ast() {
    if (!module_) return {};
    // Cached entry symbol re-seeds kEntryScope/kShortNameCollisions identically to extractVariables (both share
    // prepareExtractionState).
    return extractAST(*module_, entry_symbol_);
  }

  /// Lists every top-level func.func symbol name; used by the multi-file driver to confirm the entry survived inlining
  /// + symbol-dce.
  std::vector<std::string> list_functions() {
    std::vector<std::string> names;
    if (!module_) return names;
    module_->walk([&](mlir::func::FuncOp f) { names.push_back(f.getSymName().str()); });
    return names;
  }

  /// Reads back the ``hlfir.flatten_plan`` module attribute (stamped by hlfir-flatten-structs) into a dict mirroring
  /// FlattenPlan.to_dict(); empty-entries dict when no plan is present. Runs the shallow-alias analysis over every
  /// derived type in the module; read-only diagnostic for whether structs can pointer-alias to array-of-structs
  /// externals with no deep copy.
  nb::object shallow_alias_report() {
    nb::list out;
    if (!module_) return out;
    for (const auto& info : computeShallowAliasReport(*module_)) {
      nb::dict d;
      d["name"] = info.name;
      d["shallow_aliasable"] = info.shallow_aliasable;
      d["count"] = info.count;
      d["elem_dtype"] = info.elem_dtype;
      out.append(d);
    }
    return out;
  }

  nb::object get_flatten_plan() {
    if (!module_) return nb::dict();
    auto attr = module_->getOperation()->getAttr("hlfir.flatten_plan");
    nb::dict out;
    nb::list entries;
    out["entries"] = entries;
    if (!attr) return out;

    auto planDict = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
    if (!planDict) return out;
    auto entriesAttr = planDict.get("entries");
    if (!entriesAttr) return out;
    auto entriesArr = mlir::dyn_cast<mlir::ArrayAttr>(entriesAttr);
    if (!entriesArr) return out;

    auto asStr = [](mlir::Attribute a) -> std::string {
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(a)) return s.str();
      return "";
    };
    auto asInt = [](mlir::Attribute a) -> int64_t {
      if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(a)) return i.getInt();
      return 0;
    };
    auto asBool = [](mlir::Attribute a) -> bool {
      if (auto b = mlir::dyn_cast<mlir::BoolAttr>(a)) return b.getValue();
      return false;
    };
    auto asStrList = [&](mlir::Attribute a) -> nb::list {
      nb::list out;
      if (auto arr = mlir::dyn_cast<mlir::ArrayAttr>(a))
        for (auto e : arr) out.append(asStr(e));
      return out;
    };

    for (auto entryAttr : entriesArr) {
      auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(entryAttr);
      if (!entry) continue;
      nb::dict entryDict;
      entryDict["outer_expr"] = asStr(entry.get("outer_expr"));
      entryDict["outer_type"] = asStr(entry.get("outer_type"));
      entryDict["writeback_intent"] = asStr(entry.get("writeback_intent"));

      nb::dict recipeDict;
      if (auto recipe = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(entry.get("recipe"))) {
        recipeDict["flat_names"] = asStrList(recipe.get("flat_names"));
        recipeDict["read_exprs"] = asStrList(recipe.get("read_exprs"));
        recipeDict["write_expr"] = asStr(recipe.get("write_expr"));
        recipeDict["rank"] = asInt(recipe.get("rank"));
        recipeDict["shape_exprs"] = asStrList(recipe.get("shape_exprs"));
        recipeDict["aliasable"] = asBool(recipe.get("aliasable"));
        recipeDict["scratch_dtype"] = asStr(recipe.get("scratch_dtype"));
        recipeDict["aos_alloc"] = asBool(recipe.get("aos_alloc"));
        recipeDict["cap_symbol"] = asStr(recipe.get("cap_symbol"));
        recipeDict["source_logical_kind"] = asInt(recipe.get("source_logical_kind"));
      }
      entryDict["recipe"] = recipeDict;
      entries.append(entryDict);
    }
    return out;
  }

  /// Describes the entry's dummy args pre-flatten so Python can auto-derive an OriginalInterface; MUST run before
  /// hlfir-flatten-structs. A member with non-empty struct_name is a nested record, looked up in struct_types by that
  /// name.
  nb::object get_fortran_interface(const std::string& entry) {
    nb::dict out;
    nb::list args;
    nb::dict used;
    nb::dict struct_types;
    out["args"] = args;
    out["used_modules"] = used;
    out["struct_types"] = struct_types;
    if (!module_) return out;
    auto info = extractFortranInterface(*module_, entry);
    for (auto& a : info.args) {
      nb::dict d;
      d["name"] = a.name;
      d["dtype"] = a.dtype;
      d["intent"] = a.intent;
      d["optional"] = a.optional;
      d["rank"] = a.rank;
      nb::list sh;
      for (auto& s : a.shape_symbols) sh.append(s);
      d["shape"] = sh;
      d["is_struct"] = a.is_struct;
      d["struct_name"] = a.struct_name;
      d["struct_module"] = a.struct_module;
      args.append(d);
    }
    for (auto& kv : info.used_modules) {
      nb::list syms;
      for (auto& s : kv.second) syms.append(s);
      used[kv.first.c_str()] = syms;
    }
    for (auto& kv : info.struct_types) {
      const auto& layout = kv.second;
      nb::dict sd;
      sd["name"] = layout.name;
      sd["module"] = layout.module;
      nb::list members;
      for (auto& m : layout.members) {
        nb::dict md;
        md["name"] = m.name;
        md["dtype"] = m.dtype;
        md["rank"] = m.rank;
        nb::list sh;
        for (auto& s : m.shape_symbols) sh.append(s);
        md["shape"] = sh;
        md["struct_name"] = m.struct_name;
        md["struct_module"] = m.struct_module;
        md["alloc"] = m.alloc;
        members.append(md);
      }
      sd["members"] = members;
      struct_types[kv.first.c_str()] = sd;
    }
    return out;
  }

  /// Marks ``name`` public, everything else private, so symbol-dce drops post-inline siblings. Raises if the entry
  /// isn't in the module.
  void set_entry_symbol(const std::string& name) {
    if (!module_) throw std::runtime_error("set_entry_symbol: no module parsed");
    // Accepts a plain Fortran name or a flang-mangled symbol; a non-``_Q``-prefixed name is resolved by demangling
    // module symbols to their procedure name (text after the last ``P`` -- flang lower-cases identifiers so M/P/F
    // markers are the only upper-case letters).
    std::string target = name;
    if (name.rfind("_Q", 0) != 0) {
      std::string want = name;
      std::transform(want.begin(), want.end(), want.begin(), [](unsigned char c) { return std::tolower(c); });
      std::vector<std::string> matches;
      module_->walk([&](mlir::func::FuncOp f) {
        std::string const sym = f.getSymName().str();
        if (sym.rfind("_Q", 0) != 0) return;
        auto p = sym.rfind('P');
        std::string const proc = (p != std::string::npos && p > 1) ? sym.substr(p + 1) : sym;
        if (proc == want) matches.push_back(sym);
      });
      if (matches.size() == 1) {
        target = matches[0];
      } else if (matches.empty()) {
        throw std::runtime_error("set_entry_symbol: no Fortran procedure named '" + name + "'");
      } else {
        std::string msg = "set_entry_symbol: entry '" + name + "' is ambiguous (";
        for (size_t i = 0; i < matches.size(); ++i) {
          if (i) msg += ", ";
          msg += matches[i];
        }
        msg += "); pass the mangled symbol to disambiguate";
        throw std::runtime_error(msg);
      }
    }
    bool found = false;
    module_->walk([&](mlir::func::FuncOp f) {
      if (f.getSymName() == target) {
        mlir::SymbolTable::setSymbolVisibility(f, mlir::SymbolTable::Visibility::Public);
        found = true;
      } else {
        mlir::SymbolTable::setSymbolVisibility(f, mlir::SymbolTable::Visibility::Private);
      }
    });
    if (!found) throw std::runtime_error("set_entry_symbol: '" + target + "' not found");
    // Cached so extract_variables can install the original F-scope into kEntryScope -- later passes may rename the
    // public symbol (kernel -> kernel_soa), so we MUST anchor on the pre-pass scope for declares whose uniq_name still
    // references it.
    entry_symbol_ = target;
  }

  /// User-provided entry symbol (from set_entry_symbol); anchors kEntryScope for extractName's on-demand scope
  /// qualification.
  std::string entry_symbol_;

  /// Strips named procedures' bodies to external declarations before hlfir-inline-all runs (so a keep_external callee
  /// isn't inlined; the following symbol-dce then drops it). Matches a function by symbol == name or its
  /// ``...P<name>``/``_QP<name>`` mangling; returns the symbols actually stripped.
  std::vector<std::string> externalize_symbols(const std::vector<std::string>& names) {
    if (!module_) throw std::runtime_error("externalize_symbols: no module parsed");
    std::vector<std::string> stripped;
    module_->walk([&](mlir::func::FuncOp f) {
      if (f.isDeclaration()) return;
      llvm::StringRef const sym = f.getSymName();
      for (const std::string& n : names) {
        if (sym == n || sym.ends_with("P" + n) || sym.ends_with("_QP" + n)) {
          // Must drop all references (operands + terminator block successors) BEFORE erasing blocks -- else a
          // multi-block body (e.g. fir.select_type) can free one block while another's terminator still references it
          // -> heap corruption.
          for (mlir::Block& b : f.getBody().getBlocks()) b.dropAllReferences();
          f.getBody().getBlocks().clear();  // empty region == declaration
          stripped.push_back(sym.str());
          break;
        }
      }
    });
    return stripped;
  }

  /// Records keep_external symbol names as a module attribute so hlfir-marshal-external-structs knows which calls need
  /// struct args expanded to per-member arguments (deep-copy marshalling).
  void set_external_symbols(const std::vector<std::string>& names) {
    if (!module_) throw std::runtime_error("set_external_symbols: no module parsed");
    llvm::SmallVector<mlir::Attribute, 4> attrs;
    for (const std::string& n : names) attrs.push_back(mlir::StringAttr::get(&ctx_, n));
    module_->getOperation()->setAttr("hlfir.external_symbols", mlir::ArrayAttr::get(&ctx_, attrs));
  }

 private:
  mlir::DialectRegistry registry_;
  mlir::MLIRContext ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::vector<ValueSymbol> value_symbols_;
};

// nanobind bindings

// NOLINTNEXTLINE(performance-unnecessary-value-param): 'm' signature fixed by the NB_MODULE macro.
NB_MODULE(hlfir_bridge, m) {
  m.doc() =
      "HLFIR -> Python bridge: parses Flang HLFIR, runs passes, "
      "exposes VarInfo list and recursive ASTNode tree.";

  // Registers the bridge pass library once per process; registerAllBridgePasses() is idempotent so safe to call
  // repeatedly.
  registerAllBridgePasses();

  nb::class_<VarInfo>(m, "VarInfo")
      .def_ro("fortran_name", &VarInfo::fortran_name)
      .def_ro("mangled_name", &VarInfo::mangled_name)
      .def_ro("intent", &VarInfo::intent)
      .def_ro("rank", &VarInfo::rank)
      .def_ro("dtype", &VarInfo::dtype)
      .def_ro("is_dynamic", &VarInfo::is_dynamic)
      .def_ro("is_written", &VarInfo::is_written)
      .def_ro("shape_symbols", &VarInfo::shape_symbols)
      .def_ro("lower_bounds", &VarInfo::lower_bounds)
      .def_ro("role", &VarInfo::role)
      .def_ro("const_data", &VarInfo::const_data)
      .def_ro("view_source", &VarInfo::view_source)
      .def_ro("view_subset", &VarInfo::view_subset)
      .def_ro("view_dim_map", &VarInfo::view_dim_map)
      .def_ro("module_origin_mod", &VarInfo::module_origin_mod)
      .def_ro("module_origin_name", &VarInfo::module_origin_name)
      .def_ro("module_origin_allocatable", &VarInfo::module_origin_allocatable)
      .def_ro("module_origin_pointer", &VarInfo::module_origin_pointer)
      .def_ro("bounds_remap_view", &VarInfo::bounds_remap_view)
      .def_ro("bounds_remap_source", &VarInfo::bounds_remap_source)
      .def_ro("bounds_remap_total_extent", &VarInfo::bounds_remap_total_extent)
      .def_ro("bounds_remap_source_subset", &VarInfo::bounds_remap_source_subset)
      .def_ro("unbindable_section", &VarInfo::unbindable_section)
      .def_ro("aos_origin_mod", &VarInfo::aos_origin_mod)
      .def_ro("aos_origin_struct", &VarInfo::aos_origin_struct)
      .def_ro("aos_member_path", &VarInfo::aos_member_path)
      .def_ro("aos_outer_rank", &VarInfo::aos_outer_rank)
      .def_ro("global_alloc_inside", &VarInfo::global_alloc_inside)
      .def_ro("aos_struct_pointer", &VarInfo::aos_struct_pointer)
      .def_ro("aos_member_pointer", &VarInfo::aos_member_pointer)
      .def("__repr__", [](const VarInfo& v) {
        std::string s = "<" + v.role + " '" + v.fortran_name + "'";
        if (v.rank > 0) {
          s += "(";
          for (size_t i = 0; i < v.shape_symbols.size(); ++i) {
            if (i) s += ",";
            if (i < v.lower_bounds.size() && v.lower_bounds[i] != "1") s += v.lower_bounds[i] + ":";
            s += v.shape_symbols[i];
          }
          s += ")";
        }
        s += " " + v.dtype;
        if (!v.intent.empty()) s += " intent(" + v.intent + ")";
        s += ">";
        return s;
      });

  nb::class_<ValueSymbol>(m, "ValueSymbol")
      .def_ro("symbol", &ValueSymbol::symbol)
      .def_ro("array", &ValueSymbol::array)
      .def_ro("index_expr", &ValueSymbol::index_expr);

  nb::class_<AccessInfo>(m, "AccessInfo")
      .def_ro("array_name", &AccessInfo::array_name)
      .def_ro("index_vars", &AccessInfo::index_vars)
      .def_ro("index_exprs", &AccessInfo::index_exprs)
      .def_ro("is_read", &AccessInfo::is_read)
      .def_ro("is_write", &AccessInfo::is_write);

  nb::class_<ASTNode>(m, "ASTNode")
      .def_ro("kind", &ASTNode::kind)
      .def_ro("loop_iter", &ASTNode::loop_iter)
      .def_ro("loop_bound", &ASTNode::loop_bound)
      .def_ro("loop_lower", &ASTNode::loop_lower)
      .def_ro("loop_lower_expr", &ASTNode::loop_lower_expr)
      .def_ro("loop_step", &ASTNode::loop_step)
      .def_ro("loop_step_expr", &ASTNode::loop_step_expr)
      .def_ro("target", &ASTNode::target)
      .def_ro("expr", &ASTNode::expr)
      .def_ro("accesses", &ASTNode::accesses)
      .def_ro("pos_indices", &ASTNode::pos_indices)
      .def_ro("target_is_array", &ASTNode::target_is_array)
      .def_ro("condition", &ASTNode::condition)
      .def_ro("callee", &ASTNode::callee)
      .def_ro("call_args", &ASTNode::call_args)
      .def_ro("call_arg_subsets", &ASTNode::call_arg_subsets)
      .def_ro("aos_marshal_groups", &ASTNode::aos_marshal_groups)
      .def_ro("reduce_src", &ASTNode::reduce_src)
      .def_ro("reduce_wcr", &ASTNode::reduce_wcr)
      .def_ro("reduce_identity", &ASTNode::reduce_identity)
      .def_ro("reduce_axes", &ASTNode::reduce_axes)
      .def_ro("options", &ASTNode::options)
      .def_ro("children", &ASTNode::children)
      .def_ro("else_children", &ASTNode::else_children)
      .def("__repr__", [](const ASTNode& n) {
        if (n.kind == "loop")
          return std::string("Loop(") + n.loop_iter + "=" + std::to_string(n.loop_lower) + ":" + n.loop_bound + ", " +
                 std::to_string(n.children.size()) + " children)";
        if (n.kind == "assign")
          return std::string("Assign(") + n.target + " = " + n.expr.substr(0, 40) + (n.expr.size() > 40 ? "..." : "") +
                 ")";
        if (n.kind == "conditional") return std::string("If(") + n.condition + ")";
        if (n.kind == "call") return std::string("Call(") + n.callee + ")";
        if (n.kind == "reduce")
          return std::string("Reduce(") + n.target + " = reduce(" + n.reduce_src + ", wcr=" + n.reduce_wcr + ")";
        if (n.kind == "copy") return std::string("Copy(") + n.target + " <- " + n.reduce_src + ")";
        if (n.kind == "memset") return std::string("Memset(") + n.target + ")";
        if (n.kind == "libcall") {
          std::string s = "LibCall(" + n.target + " = " + n.callee + "(";
          for (size_t i = 0; i < n.call_args.size(); ++i) {
            if (i) s += ", ";
            s += n.call_args[i];
          }
          s += "))";
          return s;
        }
        if (n.kind == "break") return std::string("Break()");
        if (n.kind == "return") return std::string("Return()");
        return std::string("<") + n.kind + ">";
      });

  nb::class_<HLFIRModule>(m, "HLFIRModule")
      .def(nb::init<>())
      .def("parse", &HLFIRModule::parse, "Parse HLFIR from a string")
      .def("parse_file", &HLFIRModule::parse_file, "Parse HLFIR from a file path")
      .def("parse_files", &HLFIRModule::parse_files,
           "Parse multiple HLFIR files and merge them into one module "
           "(dedup by symbol name; definition wins over declaration)")
      .def("run_passes", &HLFIRModule::run_passes,
           "Run an mlir-opt-syntax pipeline "
           "(e.g. 'builtin.module(hlfir-propagate-shapes)')")
      .def("dump", &HLFIRModule::dump, "Return the current IR as a string")
      .def("get_variables", &HLFIRModule::get_variables, "Classify all hlfir.declare ops -> list[VarInfo]")
      .def("get_value_symbols", &HLFIRModule::get_value_symbols,
           "Array-element values promoted to symbols -> list[ValueSymbol] "
           "(call after get_variables)")
      .def("get_ast", &HLFIRModule::get_ast, "Recursive AST of the subroutine body -> list[ASTNode]")
      .def("list_functions", &HLFIRModule::list_functions, "Names of every top-level func.func still in the module")
      .def("set_entry_symbol", &HLFIRModule::set_entry_symbol,
           "Mark the named function public and everything else private so "
           "symbol-dce can drop post-inlining dead siblings")
      .def("externalize_symbols", &HLFIRModule::externalize_symbols,
           "Strip the bodies of the named procedures (keep_external callees) "
           "so they stay external declarations through hlfir-inline-all; "
           "returns the symbols actually stripped")
      .def("set_external_symbols", &HLFIRModule::set_external_symbols,
           "Record registered external (keep_external) symbol names so "
           "hlfir-marshal-external-structs expands their struct args to "
           "per-member arguments for deep-copy marshalling")
      .def("get_fortran_interface", &HLFIRModule::get_fortran_interface,
           "Describe the entry's dummies (pre-flatten) for auto-iface")
      .def("get_flatten_plan", &HLFIRModule::get_flatten_plan,
           "Read back the ``hlfir.flatten_plan`` module attribute set by "
           "``hlfir-flatten-structs`` as a plain dict that mirrors "
           "``FlattenPlan.to_dict()``")
      .def("shallow_alias_report", &HLFIRModule::shallow_alias_report,
           "Per derived type: whether it is shallow-aliasable to an "
           "array-of-structs external (no deep copy) -> list of dicts");
}
