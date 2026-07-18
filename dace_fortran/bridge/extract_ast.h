// extract_ast.h -- builds a recursive statement tree for a Fortran subroutine.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

/// One array access inside an hlfir.assign tree. `index_vars[d]` is the best-effort source name per dim;
/// `index_exprs[d]` is richer (e.g. indirect access renders as ``edge_idx[jc,1]`` so the SDFG generator detects
/// indirection via ``[``).
struct AccessInfo {
  std::string array_name;
  std::vector<std::string> index_vars;
  std::vector<std::string> index_exprs;
  bool is_read = false;
  bool is_write = false;
};

/// `kind` selects which fields below are populated; see the per-field-group comments in the struct body for the
/// mapping.
struct ASTNode {
  std::string kind;

  // loop
  std::string loop_iter, loop_bound;
  // loop_lower_expr (if non-empty) wins over loop_lower; covers symbolic lower bounds (e.g. array-section assigns).
  int64_t loop_lower = -1;
  std::string loop_lower_expr;
  // loop_step_expr (if non-empty) wins over loop_step; symbolic step assumed positive, negative -> zero/one iterations.
  int64_t loop_step = 1;
  std::string loop_step_expr;

  // assign
  std::string target, expr;
  bool target_is_array = false;
  std::vector<AccessInfo> accesses;

  // symbol_init: per-dim 1-based indices of the source element read (e.g. ``__sym_shp_1_2_1 = shp(1,2,1)``); ``expr``
  // holds the source array name.
  std::vector<int64_t> pos_indices;

  // conditional / while
  std::string condition;

  // call
  std::string callee;
  std::vector<std::string> call_args;
  // Per-call-arg slice subset, parallel to call_args. Empty = whole array; non-empty = a Fortran 1-based slice
  // expression (e.g. "1:3") for a sliced memlet.
  std::vector<std::string> call_arg_subsets;
  // AoS-marshalling groups (hlfir-marshal-external-structs output): flat [start, count, ...] pairs marking
  // call_args[start..start+count) as one struct's members to re-pack in the generated C tasklet. Empty for an ordinary
  // call.
  std::vector<int64_t> aos_marshal_groups;

  // reduce
  std::string reduce_src;            // input array name
  std::string reduce_wcr;            // lambda string, e.g. "lambda a, b: a + b"
  std::string reduce_identity;       // initial-accumulator string, e.g. "0"
  std::vector<int64_t> reduce_axes;  // empty = reduce all dimensions

  // libcall options: free-form key=value carrier for per-callee booleans/enums (e.g. MINLOC back) without a dedicated
  // field per option.
  std::map<std::string, std::string> options;

  // recursive
  std::vector<ASTNode> children, else_children;
};

/// Builds the AST for the first func.func in the module. `entry_symbol` (same as passed to extractVariables) anchors
/// on-demand scope qualification in trace_utils.cpp; empty disables it. Keeps the two extraction paths in lockstep for
/// inlined-callee dummy naming.
std::vector<ASTNode> extractAST(mlir::ModuleOp module, const std::string& entry_symbol = "");

}  // namespace hlfir_bridge
