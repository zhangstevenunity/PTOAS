# ptobc lives in-tree inside PTOAS.
#
# NOTE: Do NOT call cmake_minimum_required()/project() or find_package() here.
# PTOAS top-level already configures LLVM/MLIR and exports PTOAS:: targets.

add_library(ptobc_lib STATIC
  src/leb128.cpp
  src/ptobc_format.cpp
  src/mlir_helpers.cpp
  src/mlir_encode.cpp
  src/canonical_printer.cpp
  src/ptobc_decode_print.cpp
)

target_include_directories(ptobc_lib PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
)

# MLIR/LLVM deps
llvm_map_components_to_libnames(PTObc_LLVM_LIBS Support)

target_link_libraries(ptobc_lib PUBLIC
  ${PTObc_LLVM_LIBS}
  PTOIR
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
)

add_executable(ptobc
  src/main.cpp
)

target_link_libraries(ptobc PRIVATE ptobc_lib)

# (Optional) install rules can be added once the layout stabilizes.
