// ============================================================================
// Passes.h  --  Public API for the hlfir_bridge pass library.
// ============================================================================
// Each pass gets a constructor function here and a registration line in
// Passes.cpp's registerAllBridgePasses().  That single registration call
// wires every pass into MLIR's global registry so they become available
// both through the Python run_passes() entry point and through the
// standalone hlfir-bridge-opt tool.
// ============================================================================

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace hlfir_bridge {

// --- Individual pass constructors ---
std::unique_ptr<mlir::Pass> createPropagateShapesPass();
std::unique_ptr<mlir::Pass> createInlineAllPass();
std::unique_ptr<mlir::Pass> createLowerFirSelectCasePass();
std::unique_ptr<mlir::Pass> createFlattenStructsPass();
std::unique_ptr<mlir::Pass> createSplitAoRDummiesPass();
std::unique_ptr<mlir::Pass> createDefaultIntentPass();
std::unique_ptr<mlir::Pass> createVerifyNoUnresolvedCallsPass();
std::unique_ptr<mlir::Pass> createFoldElementAliasesPass();
std::unique_ptr<mlir::Pass> createFoldCopyInOutPass();
std::unique_ptr<mlir::Pass> createExpandVectorSubscriptGatherPass();
std::unique_ptr<mlir::Pass> createExpandVectorSubscriptScatterPass();
std::unique_ptr<mlir::Pass> createRejectPolymorphismPass();
std::unique_ptr<mlir::Pass> createRewritePointerAssignsPass();
std::unique_ptr<mlir::Pass> createRewriteSequenceAssociationPass();
std::unique_ptr<mlir::Pass> createLiftReductionOperandsPass();
std::unique_ptr<mlir::Pass> createLiftAllocArrayOfRecordsPass();
std::unique_ptr<mlir::Pass> createLiftAosPointerRecordsPass();
std::unique_ptr<mlir::Pass> createPruneUnreachablePass();
std::unique_ptr<mlir::Pass> createMarshalExternalStructsPass();
std::unique_ptr<mlir::Pass> createUnwrapEvalInMemPass();
std::unique_ptr<mlir::Pass> createStripErrorHelpersPass();
std::unique_ptr<mlir::Pass> createStripRuntimeIoPass();
std::unique_ptr<mlir::Pass> createStripCharacterRuntimePass();
std::unique_ptr<mlir::Pass> createPreserveMutableGlobalsPass();
std::unique_ptr<mlir::Pass> createMarkBoundsRemapViewsPass();
std::unique_ptr<mlir::Pass> createFoldAssumedRankQueriesPass();

// --- Registry ---

/// Register every bridge pass with MLIR's global pass registry.
void registerAllBridgePasses();

}  // namespace hlfir_bridge
