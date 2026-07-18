// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
// Internal cross-TU decls for HLFIR AST extraction; bodies in ast/*.cpp, public API in ast_helpers.h.
#pragma once

#include "bridge/ast/ast_helpers.h"

namespace hlfir_bridge {

ASTNode buildAssignNode(hlfir::AssignOp assign);

ASTNode buildCopyNode(hlfir::AssignOp assign);

ASTNode buildLibCallNode(hlfir::AssignOp assign, mlir::Operation* srcOp, std::string_view callee);

ASTNode buildMemsetNode(hlfir::AssignOp assign);

ASTNode buildReduceNode(hlfir::AssignOp assign, mlir::Operation* redOp, std::string_view wcr,
                        std::string_view identity);

/// hlfir.assign scalar-reduce (sum/product/minval/maxval/any/all, any source shape) -> AST; shared by buildAST and
/// walkSCFBeforeRegion so a do-while body (LiftReductionOperands-hoisted ``_QQred_lift_N`` temp) matches top-level
/// lowering; matched=true iff sd is a recognised reduction op.
std::vector<ASTNode> buildReductionAssignNodes(hlfir::AssignOp assign, mlir::Operation* sd, bool& matched);

std::vector<ASTNode> buildSectionReduceAssign(hlfir::AssignOp assign, hlfir::DesignateOp src, std::string_view pyOp,
                                              std::string_view identity);

std::vector<ASTNode> buildSectionScalarAssign(hlfir::AssignOp assign, hlfir::DesignateOp dst);

std::vector<ASTNode> buildSectionToSectionAssign(hlfir::AssignOp assign, mlir::Value dst);

ASTNode buildSelectCaseChain(fir::SelectCaseOp sel);

std::vector<ASTNode> buildWholeArrayScalarBroadcast(hlfir::AssignOp assign);

// collectReadAccesses/exprDtypeString/exprResultShape/lowerIsPresent/resolveExtent/resolveIndex declared in
// ast_helpers.h (included above).

std::string scfSynthName(mlir::Value v);

std::vector<ASTNode> walkSCFBeforeRegion(mlir::Block& block);

std::string yieldedExpr(mlir::Value v);

std::vector<ASTNode> buildMergeLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem);

std::vector<ASTNode> buildElementalAssign(hlfir::AssignOp assign, hlfir::ElementalOp elem);

std::vector<ASTNode> buildElementalCountLibcall(hlfir::AssignOp assign, hlfir::ElementalOp elem);

std::vector<ASTNode> buildElementalAnyAllReduce(hlfir::AssignOp assign, hlfir::ElementalOp elem, std::string_view wcr,
                                                std::string_view identity);

/// Materialises hlfir.elemental into a synthetic transient of the elemental's dtype (libcall-over-elemental path);
/// returns {transient_name, AST_nodes}, empty on failure.
std::pair<std::string, std::vector<ASTNode>> materialiseElementalForLibcall(hlfir::ElementalOp elem);

}  // namespace hlfir_bridge
