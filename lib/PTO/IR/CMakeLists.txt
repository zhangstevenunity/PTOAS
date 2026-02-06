# [关键] 库名重命名为 PTOIR，避免与 LLVM 里的 PTODialect/MLIRPTODialect 冲突
add_mlir_dialect_library(PTOIR
  PTO.cpp
  PTOAttrs.cpp
  PTOTypeDefs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOOpsIncGen  # 对应 include/PTO/IR/CMakeLists.txt 里定义的那个目标

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

install(TARGETS PTOIR
  EXPORT PTOASTargets      # 将此 Target 加入导出集
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

# 设置头文件包含路径，确保自动获得头文件路径
target_include_directories(PTOIR INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)