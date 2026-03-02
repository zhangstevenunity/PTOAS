set(LLVM_LINK_COMPONENTS
  Support
  Core
)

# [修改 1] 重命名目标: ptoas -> pto-opt
# 原因：LLVM 构建目录里已经有一个 ptoas 了，重名会导致 CMake 报错。
add_llvm_executable(pto-opt
  ptoas.cpp
)
# =========================================================
# [新增] 魔法操作：修改最终输出的文件名为 "ptoas"
# =========================================================
set_target_properties(pto-opt PROPERTIES OUTPUT_NAME "ptoas")
# [修改 2] 更新链接库名称
# 原因：In-tree 时你的库叫 MLIRPTODialect，但现在 Out-of-tree 它们是你自己定义的
# (通常在 lib/PTO/IR/CMakeLists.txt 里被定义为 PTOIR 或类似名字)。
target_link_libraries(pto-opt PRIVATE
  # === 你的本地库 (请确认 lib/PTO/*/CMakeLists.txt 里的定义名) ===
  PTOIR            # 替代 MLIRPTODialect
  PTOTransforms    # 替代 MLIRPTOTransforms
  ptobc_lib

  # === MLIR 基础设施 ===
  MLIRMlirOptMain  # [重要] 包含了 main 函数入口逻辑，通常必须链接
  
  # === MLIR 标准库 ===
  MLIRBufferizationTransforms
  MLIRArithTransforms
  MLIRTensorTransforms
  MLIRFuncTransforms
  MLIRIR
  MLIRParser
  MLIRPass
  MLIREmitCDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRTransforms
  MLIRTargetCpp
  MLIRAffineDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRControlFlowDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils
)

# [修改 3] 更新依赖
# 确保在编译 pto-opt 之前，TableGen 已经生成了头文件
# "PTOOpsIncGen" 这个名字必须和你 include/PTO/IR/CMakeLists.txt 里定义的 add_public_tablegen_target 一致
add_dependencies(pto-opt
  PTOOpsIncGen 
)
