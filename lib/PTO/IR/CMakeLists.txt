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

# [关键] 库名重命名为 PTOIR，避免与 LLVM 里的 PTODialect/MLIRPTODialect 冲突
add_mlir_dialect_library(PTOIR
  ENABLE_AGGREGATION

  PTO.cpp
  VPTO.cpp
  VPTOAddressSemantics.cpp
  VPTOMemoryDist.cpp
  VPTOScheduling.cpp
  VMI.cpp
  VPTOUbOps.cpp
  PTOAttrs.cpp
  PTOLayoutUtils.cpp
  PTOSyncUtils.cpp
  PTOTypeDefs.cpp
  PTOTypeUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/PTO

  DEPENDS
  PTOOpsIncGen  # 对应 include/PTO/IR/CMakeLists.txt 里定义的那个目标

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
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
