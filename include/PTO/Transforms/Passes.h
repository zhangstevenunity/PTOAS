// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// Pass factory declarations for PTO transform pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"
#include "PTO/IR/PTODialect.h"
#include "PTO/Transforms/TileLibService.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL
#include "PTO/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createLoweringSyncToPipePass();
std::unique_ptr<Pass> createPTOAssignDefaultFrontendPipeIdPass();
std::unique_ptr<Pass> createPTOLowerFrontendPipeOpsPass();
std::unique_ptr<Pass> createPTOInferValidatePipeInitPass();
std::unique_ptr<Pass> createPTOResolveReservedBuffersPass();
std::unique_ptr<Pass> createPTOWrapFunctionsInSectionsPass();
std::unique_ptr<Pass> createPTONormalizeUncoveredTileSectionsPass();
std::unique_ptr<Pass> createPTOValidatePhysicalSectionBoundariesPass();
std::unique_ptr<Pass> createPTOMaterializeTileOpSectionsPass();
std::unique_ptr<Pass> createVPTOSplitCVModulePass();
std::unique_ptr<Pass> createVPTONormalizeContainerPass();
std::unique_ptr<Pass> createPTOVerifyTFreePass();

// Creates a pass for ...
std::unique_ptr<Pass> createPTOInsertSyncPass();
std::unique_ptr<Pass> createPTOInjectBarrierAllSyncPass();
std::unique_ptr<Pass>
createPTOBufidSyncPass(const PTOBufidSyncOptions &options = {});

// Default arch is A3 unless overridden by callers.
std::unique_ptr<Pass> createEmitPTOManualPass();
// Explicitly select target arch for codegen.
std::unique_ptr<Pass> createEmitPTOManualPass(PTOArch arch);


/// Create a pass to convert ops from other dialects to PTO Ops.
std::unique_ptr<Pass> createConvertToPTOOpPass();

/// Create a pass to infer, propagate, and add memory scope information to
/// PTO Ops.
std::unique_ptr<Pass> createInferPTOMemScopePass();

std::unique_ptr<Pass>
createPlanMemoryModernPass(const PlanMemoryOptions &options = {});
std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass();
std::unique_ptr<Pass> createPTOValidateIntToPtrUsesPass();
std::unique_ptr<Pass> createPTORematerializeFixpipeVectorQuantPass();
std::unique_ptr<Pass>
createPTOMaterializeImplicitTmpPass(bool requireExplicitTmp = false);
std::unique_ptr<Pass> createPTOResolveBufferSelectPass();
std::unique_ptr<Pass> createInferPTOLayoutPass();
std::unique_ptr<Pass> createPTOA5NormalizeTMovPass();
std::unique_ptr<Pass> createPTORemoveIdentityTMovPass();
std::unique_ptr<Pass> createPreFusionAnalysisPass();
std::unique_ptr<Pass> createPrintPreFusionAnalysisPass();
std::unique_ptr<Pass> createFusionPlanPass();
std::unique_ptr<Pass>
createFusionPlanPass(const FusionPlanOptions &options);
std::unique_ptr<Pass> createOpSchedulingPass();
std::unique_ptr<Pass> createPTOMarkLastUsePass();
std::unique_ptr<Pass> createPTOFusionRegionGenPass();

LogicalResult validateIntToPtrUses(func::FuncOp func);

std::unique_ptr<Pass> createPTOUnrollLoopsPass();
/// Backward-compatible alias of createPTOUnrollLoopsPass().
std::unique_ptr<Pass> createPTOUnrollSIMTForPass();
std::unique_ptr<Pass> createPTOConvertSCFToCFWithLoopHintsPass();
std::unique_ptr<Pass> createPTOPromotePersistentFragmentLoopsPass();
std::unique_ptr<Pass> createPTONarrowVPTOLoopCountersPass();
std::unique_ptr<Pass> createPTOAnalyzeSIMTPersistentFragmentPass();
std::unique_ptr<Pass> createPTOMaterializeSIMTPersistentFragmentPass();
std::unique_ptr<Pass> createPTOOutlineSIMTSectionsPass();
std::unique_ptr<Pass> createPTOInferVPTOVecScopePass();
std::unique_ptr<Pass> createVPTOExpandWrapperOpsPass();
std::unique_ptr<Pass> createVPTOSoftPostUpdatePass();
std::unique_ptr<Pass> createVPTOGuardedLICMPass();
std::unique_ptr<Pass> createPTOPrintAddressAnalysisPass();
std::unique_ptr<Pass> createPTOVPTOPtrBoundaryPass();
std::unique_ptr<Pass>
createPTOLowLevelLoopFusionPass(const PTOLowLevelLoopFusionOptions &options = {});
std::unique_ptr<Pass> createPTOFusionPredicateElisionPass();
std::unique_ptr<Pass> createPTOFusionLoadStoreElisionPass();
std::unique_ptr<Pass> createPTOVexpdifFusionPass();
std::unique_ptr<Pass> createPTOUnrollAfterLoopFusionPass();
std::unique_ptr<Pass> createPTOFlattenFusionRegionPass();
std::unique_ptr<Pass> createVPTOPtrNormalizePass();
std::unique_ptr<Pass> createVPTOPtrCastCleanupPass();
std::unique_ptr<Pass> createVPTOCombineReductionsPass();
std::unique_ptr<Pass> createVPTOOptimizeVcvtPass();
std::unique_ptr<Pass> createVPTOMaskSimplifyPass();
std::unique_ptr<Pass>
createVPTOSchedulerPass(const VPTOSchedulerOptions &options = {});
LogicalResult validateVPTOAuthoringIR(ModuleOp module,
                                      llvm::raw_ostream *diagOS = nullptr);
LogicalResult validateVPTOEmissionIR(ModuleOp module,
                                     llvm::raw_ostream *diagOS = nullptr);
std::unique_ptr<Pass> createPTOValidateVPTOIRPass();
std::unique_ptr<Pass> createPTOValidateVPTOEmissionIRPass();
LogicalResult validateVMIProducerBoundaryIR(ModuleOp module,
                                            llvm::raw_ostream *diagOS = nullptr);
LogicalResult validateVMILayoutAssignedIR(ModuleOp module,
                                          llvm::raw_ostream *diagOS = nullptr,
                                          bool verifyHelperSupport = true);
std::unique_ptr<Pass> createPTOValidateVMIIRPass();
std::unique_ptr<Pass> createPTOValidateVMILayoutIRPass();
std::unique_ptr<Pass> createVMIPreAssignmentCombinePass();
std::unique_ptr<Pass> createVMIMaskGranularityAssignmentPass();
std::unique_ptr<Pass> createVMILayoutRematerializeWeakProducersPass();
std::unique_ptr<Pass> createVMILayoutAssignmentPass();
std::unique_ptr<Pass> createVMILayoutFoldPass();
std::unique_ptr<Pass> createVMILayoutRematerializePass();
std::unique_ptr<Pass> createVMILayoutSinkMaterializationPass();
std::unique_ptr<Pass> createVMILegalizeArithSelectPass();
std::unique_ptr<Pass> createVMILowerUnifiedToLegacyPass();
std::unique_ptr<Pass> createVMINormalizeSignlessIntToUnsignedPass();
std::unique_ptr<Pass> createVMIToVPTOPass();
std::unique_ptr<Pass> createVPTOStatefulStreamFusionPass();
std::unique_ptr<Pass> createPTOExpandSoftLibPass();
std::unique_ptr<Pass> createInsertTemplateAttributesPass();
std::unique_ptr<Pass> createExpandTileOpPass();
std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass();
std::unique_ptr<Pass> createFoldTileBufIntrinsicsPass(llvm::StringRef foldMode);
std::unique_ptr<Pass> createPTOCanonicalizeIRPass();
std::unique_ptr<Pass> createLowerPTOToUBufOpsPass();
std::unique_ptr<Pass>
createPTOInlineLibCallPass(const PTOInlineLibCallOptions &options = {});
std::unique_ptr<Pass> createPTOInlineBackendHelpersPass(
    const PTOInlineBackendHelpersOptions &options = {});

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#undef GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "PTO/Transforms/Passes.h.inc"

} // namespace pto
} // namespace mlir


#endif // MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
