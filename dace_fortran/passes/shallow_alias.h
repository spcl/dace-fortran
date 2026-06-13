// ============================================================================
// shallow_alias.h  --  diagnostic report of the shallow-alias layout analysis.
// ============================================================================
// ``hlfir-flatten-structs`` decides per derived type whether it is
// *shallow-aliasable* (one contiguous run of a single scalar type, so an
// array-of-structs external pointer can alias it with no deep copy).  This
// header exposes that analysis as a whole-module report so callers can test
// up front whether every struct a program uses is shallow-copyable.
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace hlfir_bridge {

/// One record type's shallow-alias verdict.
struct ShallowAliasInfo {
  std::string name;        ///< the derived type's record name
  bool shallow_aliasable;  ///< true iff it can be pointer-aliased to AoS
  int64_t count;           ///< contiguous element count when aliasable, else 0
  std::string elem_dtype;  ///< uniform element dtype when aliasable, else ""
};

/// Run the shallow-alias analysis over every derived type the module declares
/// (deduplicated by record name) and return one verdict each.  Pure read-only
/// diagnostic; mutates nothing.
std::vector<ShallowAliasInfo> computeShallowAliasReport(mlir::ModuleOp module);

}  // namespace hlfir_bridge
