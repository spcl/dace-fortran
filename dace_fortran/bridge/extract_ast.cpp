// extractAST() entry point; impl split across ast/*.cpp, sharing state via ast_helpers.h. This file only holds shared
// includes.

#include "bridge/extract_ast.h"

#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "bridge/extract_vars.h"
#include "bridge/trace_utils.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// ast_helpers.h: cross-file decls + thread-locals shared by all ast/*.cpp.
#include "bridge/ast/ast_helpers.h"

namespace hlfir_bridge {}  // namespace hlfir_bridge
