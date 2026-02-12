add_mlir_dialect_library(PTOTransforms
  InsertLoadStoreForMixCV.cpp
  PTOInsertCVMov.cpp
  PTOConvertToDPS.cpp
  PTOInsertSync.cpp
  PTOViewToMemref.cpp
  PTOToEmitC.cpp
  Utils.cpp
  OptMemPlanForPipeline.cpp
  AllocToPointerCast.cpp
  InferPTOMemScope.cpp
  PTOPlanMemory.cpp
  PTORemoveRedundantBarrier.cpp
  InferPTOLayout.cpp
  BufferizableOpInterfaceImpl.cpp
  ConvertToPTOOp.cpp
  PTOHighDimLowering.cpp
  PTOVFloopGather.cpp
  PTOIRTranslator.cpp
  SyncCommon.cpp
  BlockSyncAnalysis.cpp
  MemoryDependentAnalyzer.cpp
  MoveSyncState.cpp
  RemoveRedundantSync.cpp
  SyncEventIdAllocation.cpp
  SyncCodegen.cpp
  LoweringSyncToPipe.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOPassesIncGen
  PTOOpsIncGen

  LINK_LIBS PUBLIC
  PTOIR
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
  MLIRSCFToEmitC
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
