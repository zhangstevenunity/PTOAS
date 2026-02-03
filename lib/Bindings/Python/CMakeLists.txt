# lib/Bindings/Python/CMakeLists.txt

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(TableGen)
include(AddMLIR)

# ---- 1) Python native extension: mlir._mlir_libs._pto ----
pybind11_add_module(_pto MODULE
  PTOModule.cpp
)

target_include_directories(_pto PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_link_options(_pto PRIVATE)

target_link_libraries(_pto PRIVATE

  PTOCAPI
  PTOIR
  MLIRIR
  MLIRCAPIIR
  MLIRSupport
  MLIRArithDialect
  MLIRDestinationStyleOpInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRLoopLikeInterface
  MLIRViewLikeInterface
  MLIRFunctionInterfaces
)

# 关键：放到 mlir/_mlir_libs 下（匹配 MLIR dialect python 的 import 习惯）
set_target_properties(_pto PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  BUILD_RPATH "${LLVM_BUILD_LIBRARY_DIR}"
  INSTALL_RPATH "${LLVM_BUILD_LIBRARY_DIR}"
)
# macOS: 避免 ld 警告 "-undefined suppress is deprecated"，仅保留 -undefined dynamic_lookup
if(APPLE)
  target_link_options(_pto PRIVATE "LINKER:-undefined,dynamic_lookup")
endif()

install(TARGETS _pto
  LIBRARY DESTINATION "${MLIR_PYTHON_PACKAGE_DIR}/mlir/_mlir_libs"
)

# ---- 2) Generate ODS Python op bindings: _pto_ops_gen.py ----
set(PTO_TD_DIR "${CMAKE_SOURCE_DIR}/include/PTO/IR")
set(LLVM_TARGET_DEFINITIONS ${PTO_TD_DIR}/PTOOps.td)

mlir_tablegen(_pto_ops_gen.py
  -gen-python-op-bindings
  --bind-dialect=pto
)

add_public_tablegen_target(PTOPythonGen)
add_dependencies(_pto PTOPythonGen)

# ---- 3) Copy generated python + handwritten pto.py into build/python ----
set(PTO_PY_SRC "${CMAKE_SOURCE_DIR}/python/pto/dialects/pto.py")

add_custom_command(TARGET _pto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/python/mlir/dialects"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${PTO_PY_SRC}"
          "${CMAKE_BINARY_DIR}/python/mlir/dialects/pto.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_BINARY_DIR}/_pto_ops_gen.py"
          "${CMAKE_BINARY_DIR}/python/mlir/dialects/_pto_ops_gen.py"
  VERBATIM
)

install(FILES
  "${PTO_PY_SRC}"
  "${CMAKE_CURRENT_BINARY_DIR}/_pto_ops_gen.py"
  DESTINATION mlir/dialects
)

add_custom_command(TARGET _pto POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  VERBATIM
)


