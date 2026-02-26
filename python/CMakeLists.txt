# =========================================================
# Python Package Configuration
# =========================================================

add_custom_target(PTOPythonModules)

# [修改] 依赖必须指向 lib/Bindings/Python 中定义的 Target (_pto)
add_dependencies(PTOPythonModules _pto) 

# 生成 TableGen 绑定代码
set(LLVM_TARGET_DEFINITIONS pto/dialects/PTOOps.td)
mlir_tablegen(pto/dialects/_pto_ops_gen.py
  -gen-python-op-bindings
  -bind-dialect=pto
  -I${PROJECT_SOURCE_DIR}/include
)
add_public_tablegen_target(PTOOpsPyGen)

add_dependencies(PTOPythonModules PTOOpsPyGen)
