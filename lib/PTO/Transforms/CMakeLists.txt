# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(_ptoas_transforms_private_link_libs)
if(TARGET vfsim::native_core)
  list(APPEND _ptoas_transforms_private_link_libs
    $<BUILD_INTERFACE:vfsim::native_core>)
endif()
if(TARGET vfsim::ir_planner)
  list(APPEND _ptoas_transforms_private_link_libs
    $<BUILD_INTERFACE:vfsim::ir_planner>)
endif()

add_mlir_dialect_library(PTOTransforms
  ENABLE_AGGREGATION

  TileFusion/FusionAnalysis.cpp
  TileFusion/FusionOpSemantics.cpp
  TileFusion/PTOPreFusionAnalysis.cpp
  TileFusion/PTOPrintPreFusionAnalysis.cpp
  TileFusion/PTOFusionPlan.cpp
  TileFusion/PTOOpScheduling.cpp
  TileFusion/PTOMarkLastUse.cpp
  TileFusion/PTOFusionRegionGen.cpp
  TileFusion/PTOLowLevelLoopFusion.cpp
  TileFusion/PTOFusionPredicateElision.cpp
  TileFusion/PTOFusionLoadStoreElision.cpp
  TileFusion/PTOVexpdifFusion.cpp
  TileFusion/PTOUnrollAfterLoopFusion.cpp
  TileFusion/PTOFlattenFusionRegion.cpp
  VPTOLLVMEmitter.cpp
  VPTOCANN900LLVMEmitter.cpp
  VPTOCANN900LLVMEmitterArithmeticPatterns.cpp
  VPTOCANN900LLVMEmitterCalleeCore.cpp
  VPTOCANN900LLVMEmitterCalleeMemory.cpp
  VPTOCANN900LLVMEmitterMemoryPatterns.cpp
  VPTOCANN900LLVMEmitterPacking.cpp
  VPTOCANN900LLVMEmitterPipeline.cpp
  VPTOCANN900LLVMEmitterScalarPatterns.cpp
  VPTOCANN900LLVMEmitterTypeHelpers.cpp
  VPTOCANN900LLVMEmitterTypePatterns.cpp
  VPTOLLVMEmitterDispatcher.cpp
  VPTOLLVMEmitterHelper.cpp
  VPTOPtrNormalize.cpp
  VPTOPtrCastCleanup.cpp
  VPTOCombineReductions.cpp
  VPTOOptimizeVcvt.cpp
  VPTOMaskSimplify.cpp
  VPTOExpandWrapperOps.cpp
  VPTOSoftPostUpdate.cpp
  VPTOGuardedLICM.cpp
  PTOPrintAddressAnalysis.cpp
  VPTOScheduler/VPTORegPressureTracker.cpp
  VPTOScheduler/VPTOSchedBoundary.cpp
  VPTOScheduler/VPTOSchedDAG.cpp
  VPTOScheduler/VPTOSchedDAGBuilder.cpp
  VPTOScheduler/VPTOSchedModel.cpp
  VPTOScheduler/VPTOSchedRegion.cpp
  VPTOScheduler/VPTOSchedResourceTracker.cpp
  VPTOScheduler/VPTOSchedStrategy.cpp
  VPTOScheduler/VPTOScheduler.cpp
  VPTOScheduler/VPTOSchedulerPass.cpp
  PTOVPTOPtrBoundary.cpp
  VPTOBufferMaterialization.cpp
  PTOValidateVPTOIR.cpp
  PTONarrowVPTOLoopCounters.cpp
  PTOConvertSCFToCFWithLoopHintsPass.cpp
  PTOPromotePersistentFragmentLoopsPass.cpp
  PTOUnrollLoopsPass.cpp
  PTOValidateVMIIR.cpp
  VMIPreAssignmentCombine.cpp
  VMIControlFlowSupport.cpp
  VMILegalizeArithSelect.cpp
  VMIMaskGranularityAssignment.cpp
  VMILowerUnifiedToLegacy.cpp
  VMINormalizeSignlessIntToUnsigned.cpp
  VMILayoutRematerializeWeakProducers.cpp
  VMILayoutAssignment.cpp
  VMILayoutFold.cpp
  VMILayoutPropagation.cpp
  VMILayoutSupport.cpp
  VMILayoutRematerialize.cpp
  VMILayoutSinkMaterialization.cpp
  VMIToVPTO.cpp
  VPTOStatefulStreamFusion.cpp
  PTOExpandSoftLib.cpp
  SIMTPersistentFragmentAnalysis.cpp
  PTOAnalyzeSIMTPersistentFragment.cpp
  PTOMaterializeSIMTPersistentFragment.cpp
  PTOOutlineSIMTSections.cpp
  PTOInferVPTOVecScope.cpp

  InsertSync/PTOInsertSync.cpp
  PTOInjectBarrierAllSync.cpp
  InsertSync/InsertSyncDebug.cpp
  PTOMaterializeImplicitTmp.cpp
  PTORematerializeFixpipeVectorQuant.cpp
  PTOValidateIntToPtrUses.cpp
  TileLibService.cpp
  SoftLibService.cpp
  InsertTemplateAttributes.cpp
  ExpandTileOp.cpp
  FoldTileBufIntrinsics.cpp
  LowerPTOToUBufOps.cpp
  PTOLowerToOpLibCalls.cpp
  PTOInstantiateAndInlineOpLib.cpp
  PTOToEmitC.cpp
  CppPostprocess.cpp
  Utils.cpp
  OptMemPlanForPipeline.cpp
  InferPTOMemScope.cpp
  PTOPlanMemoryModern.cpp
  PTORemoveRedundantBarrier.cpp
  InferPTOLayout.cpp
  PTOA5NormalizeTMovPass.cpp
  PTORemoveIdentityTMov.cpp
  PTOCanonicalizeIR.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvertToPTOOp.cpp
  PTOAssignDefaultFrontendPipeIdPass.cpp
  PTOLowerFrontendPipeOpsPass.cpp
  PTOInferValidatePipeInitPass.cpp
  PTOResolveBufferSelect.cpp
  PTOResolveReservedBuffersPass.cpp
  SlotAffineAnalysis.cpp
  PTOWrapFunctionsInSectionsPass.cpp
  PTONormalizeUncoveredTileSections.cpp
  PTOValidatePhysicalSectionBoundaries.cpp
  PTOMaterializeTileOpSections.cpp
  VPTONormalizeContainer.cpp
  VPTOSplitCVModule.cpp
  InsertSync/PTOIRTranslator.cpp
  InsertSync/SyncMacroModel.cpp
  InsertSync/SyncCommon.cpp
  InsertSync/InsertSyncAnalysis.cpp
  InsertSync/MemoryDependentAnalyzer.cpp
  InsertSync/MoveSyncState.cpp
  InsertSync/RemoveRedundantSync.cpp
  InsertSync/SyncEventIdAllocation.cpp
  InsertSync/SyncCodegen.cpp
  BufidSync/BufidSyncPass.cpp
  BufidSync/BufidSyncAnalysis.cpp
  BufidSync/BufidSyncIdAlloc.cpp
  BufidSync/BufidSyncCodegen.cpp
  LoweringSyncToPipe.cpp
  PTOVerifyTFreePass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOPassesIncGen
  PTOOpsIncGen

  LINK_COMPONENTS
  Analysis

  LINK_LIBS
  PUBLIC
  PTOAnalysis
  PTOIR
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRAffineDialect
  MLIREmitCDialect
  MLIRTargetCpp
  MLIRTransformUtils
  MLIRTransforms
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRSCFUtils
  MLIRVectorDialect
  MLIRParser
  MLIRArithTransforms
  MLIRSCFToEmitC
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRConvertToLLVMPass
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
  PRIVATE
  ${_ptoas_transforms_private_link_libs}
)

install(TARGETS PTOTransforms
  EXPORT PTOASTargets      # 将此 Target 加入导出集
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# 设置头文件包含路径，确保自动获得头文件路径
target_include_directories(PTOTransforms INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(TARGET vfsim::native_core AND TARGET obj.PTOTransforms)
  # Preserve the simulator target's compile usage requirements on the object
  # library; its link dependency is already captured in aggregation metadata.
  target_link_libraries(obj.PTOTransforms PRIVATE
    $<BUILD_INTERFACE:vfsim::native_core>)
endif()

if(TARGET vfsim::ir_planner)
  target_compile_definitions(PTOTransforms PRIVATE PTO_ENABLE_VFSIM_IR_PLANNER=1)
  if(TARGET obj.PTOTransforms)
    target_link_libraries(obj.PTOTransforms PRIVATE
      $<BUILD_INTERFACE:vfsim::ir_planner>)
    target_compile_definitions(obj.PTOTransforms PRIVATE PTO_ENABLE_VFSIM_IR_PLANNER=1)
  endif()
endif()
